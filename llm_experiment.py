#!/usr/bin/env python3
"""Correctness-Aware Evolutionary Design of 8-Puzzle Heuristics with Language Models.

Real-LLM runner (local Ollama + Qwen3 by default).  Scientific status:
  * experiment.py is the OFFLINE SURROGATE (0 LLM calls, 0 tokens).  Its numbers
    are never produced or altered here.
  * This file makes REAL model calls only.  Every LLM call, token count, latency
    and failure is logged.  Unknown token counts are null, never 0.
  * "estimated_monetary_cost_usd" is 0.0 for local inference ONLY in the sense of
    "no provider/API charge"; local compute (time, energy) is not priced.
  * --pilot is a plumbing check (Ollama -> LLM -> extraction -> AST -> sandbox ->
    smoke test).  It is NOT an experimental result.

Windows compatible: `resource` is only used inside the sandbox child process and
only if it exists.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import heapq
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from experiment import (GOAL, all_distances, astar, linear_conflict, manhattan,  # noqa: E402,F401
                        misplaced, neighbors, scramble, wilcoxon_signed_rank)

PROMPT_VERSION = "8puzzle-llm-v2"  # v2: repair prompt + counterexample feedback; stop-after-code

# Builtins visible to candidate code inside the sandbox child process.
ALLOWED_NAMES = {"abs", "min", "max", "sum", "len", "range", "int", "float", "round",
                 "enumerate", "zip", "tuple", "list", "dict", "set", "bool",
                 "divmod", "sorted", "all", "any", "pow", "reversed",
                 "True", "False", "None"}
FORBIDDEN_NAMES = {"eval", "exec", "open", "compile", "__import__", "input", "globals", "locals",
                   "vars", "getattr", "setattr", "delattr", "breakpoint", "help", "memoryview",
                   "dir", "type", "object", "super", "classmethod", "staticmethod", "property"}

LOCAL_PROVIDERS = {"ollama-local", "local-openai-compatible"}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class MissingLLM(RuntimeError):
    pass


class EmptyLLMResponse(RuntimeError):
    pass


class LLMCallError(RuntimeError):
    """Structured LLM failure; `record` is the (already logged) call record."""

    def __init__(self, error_type: str, message: str, record: Optional[dict] = None):
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message
        self.record = record or {}


def classify_llm_error(e: BaseException) -> Tuple[str, bool, str]:
    """Return (error_type, retryable, message)."""
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            body = ""
        if e.code == 404:
            return "ModelNotFound", False, f"HTTP 404 (model or endpoint not found; is the model pulled?) {body}"
        return "HTTPError", (e.code >= 500 or e.code == 429), f"HTTP {e.code}: {body}"
    if isinstance(e, urllib.error.URLError):
        reason = e.reason
        if isinstance(reason, TimeoutError):
            return "TimeoutError", True, f"timed out while connecting/reading: {reason}"
        if isinstance(reason, ConnectionRefusedError):
            return "ConnectionError", False, "connection refused - is Ollama running (ollama serve)?"
        return "ConnectionError", True, f"endpoint not reachable: {reason}"
    if isinstance(e, TimeoutError):
        return "TimeoutError", True, str(e) or "timed out"
    if isinstance(e, ConnectionRefusedError):
        return "ConnectionError", False, "connection refused - is Ollama running (ollama serve)?"
    if isinstance(e, ConnectionError):
        return "ConnectionError", True, str(e)
    if isinstance(e, json.JSONDecodeError):
        return "MalformedJSON", False, str(e)
    if isinstance(e, EmptyLLMResponse):
        return "EmptyResponse", True, str(e)
    return type(e).__name__, False, str(e)


def derive_call_seed(seed: int, candidate_id: str) -> int:
    """Deterministic per-call sampling seed so a population is reproducible yet not 8 identical copies."""
    h = hashlib.sha256(f"{seed}|{candidate_id}".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


# --------------------------------------------------------------------------- #
# LLM client (native Ollama /api/chat, or OpenAI-compatible)
# --------------------------------------------------------------------------- #
class OpenAICompatibleClient:
    def __init__(self, cfg: Dict[str, Any], call_log: Path):
        self.base_url = os.environ.get(cfg.get("base_url_env", "LLM_BASE_URL"), cfg.get("base_url", ""))
        self.model = os.environ.get(cfg.get("model_env", "LLM_MODEL"), cfg.get("model", ""))
        self.model_version = cfg.get("model_version", "unspecified")
        self.api_key = os.environ.get(cfg.get("api_key_env", "LLM_API_KEY"), cfg.get("api_key", ""))
        self.provider = cfg.get("provider", "openai-compatible")
        self.temperature = float(cfg.get("temperature", 0.2))
        self.max_output_tokens = int(cfg.get("max_output_tokens", 400))
        self.timeout_s = float(cfg.get("request_timeout_s", 120))
        self.disable_thinking = bool(cfg.get("disable_thinking", True))
        self.pricing = cfg.get("pricing", {})
        self.max_retries = int(cfg.get("max_retries", 2))
        self.retry_backoff_s = float(cfg.get("retry_backoff_s", 5))
        self.pilot_timeout_s = float(cfg.get("pilot_timeout_s", 120))
        self.pilot_max_output_tokens = int(cfg.get("pilot_max_output_tokens", 400))
        # Qwen3 on Ollama can still reason out loud in plain text even with think=false, burning the token budget.
        default_sys = ("You are a code generator. Do not think out loud, do not restate the problem, and do not "
                       "analyse. Your reply must begin immediately with a ```python code block. /no_think")
        self.system_prompt = cfg.get("system_prompt", default_sys if self.provider == "ollama-local" else "")
        # Assistant prefill (Ollama native chat): the model continues INSIDE a code fence instead of reasoning in prose.
        # Set "assistant_prefill": "" in the config to disable.
        self.prefill = cfg.get("assistant_prefill", "```python\n" if self.provider == "ollama-local" else "")
        # Stop generation at the closing code fence (saves ~70% of latency; the explanation is not generated/used).
        # Only applied with the Ollama prefill, where the reply starts inside a fence.  Disable: "stop_after_code": false
        self.stop_after_code = bool(cfg.get("stop_after_code", True))
        self.stop_sequences = ["\n```"] if (self.stop_after_code and self.prefill and self.provider == "ollama-local") else []
        self.call_log = call_log
        self.call_log.parent.mkdir(parents=True, exist_ok=True)
        self.stats = {"calls": 0, "successes": 0, "failures": 0, "retries": 0,
                      "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                      "successful_calls_with_unknown_tokens": 0, "latency_s": 0.0}
        if not self.base_url or not self.model:
            raise MissingLLM("LLM endpoint and model are required; set LLM_BASE_URL and LLM_MODEL or config values.")

    # -- endpoints ---------------------------------------------------------- #
    def _root(self) -> str:
        u = self.base_url.rstrip("/")
        for suffix in ("/chat/completions", "/v1", "/api/chat", "/api"):
            if u.endswith(suffix):
                u = u[: -len(suffix)]
        return u

    def endpoint(self) -> str:
        u = self.base_url.rstrip("/")
        if self.provider == "ollama-local":
            # Native Ollama API so Qwen3 thinking mode can be disabled explicitly.
            return self._root() + "/api/chat"
        if u.endswith("/chat/completions"):
            return u
        if u.endswith("/v1"):
            return u + "/chat/completions"
        return u + "/v1/chat/completions"

    def check_endpoint(self, timeout_s: float = 8.0) -> Dict[str, Any]:
        """Cheap reachability/model check (Ollama only). Makes NO generation call."""
        if self.provider != "ollama-local":
            return {"checked": False, "reachable": None, "model_available": None}
        url = self._root() + "/api/tags"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout_s) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            et, _, msg = classify_llm_error(e)
            return {"checked": True, "reachable": False, "model_available": None,
                    "error_type": et, "error": msg, "url": url}
        names = [m.get("name", "") for m in (data.get("models") or [])]
        want = self.model
        ok = any(n == want or n == want + ":latest" or (":" not in want and n.startswith(want + ":")) for n in names)
        return {"checked": True, "reachable": True, "model_available": ok, "installed_models": names, "url": url}

    # -- cost / payload ------------------------------------------------------ #
    def _cost(self, inp, out):
        if self.provider in LOCAL_PROVIDERS:
            return 0.0, "local_open_model_no_provider_charge_(local_compute_not_priced)"
        if inp is None or out is None:
            return None, "token_counts_or_pricing_unavailable"
        pi = self.pricing.get("input_usd_per_million")
        po = self.pricing.get("output_usd_per_million")
        if pi is None or po is None:
            return None, "pricing_not_configured"
        return (inp * pi + out * po) / 1_000_000.0, "configured_pricing"

    def _messages(self, prompt: str) -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        msgs = msgs + [{"role": "user", "content": prompt}]
        if self.prefill and self.provider == "ollama-local":
            msgs.append({"role": "assistant", "content": self.prefill})
        return msgs

    def _payload(self, prompt: str, max_tokens: int, call_seed: int) -> Dict[str, Any]:
        if self.provider == "ollama-local":
            payload = {"model": self.model,
                       "messages": self._messages(prompt),
                       "stream": False,
                       "options": {"temperature": self.temperature,
                                   "num_predict": max_tokens,
                                   "seed": call_seed}}
            if self.stop_sequences:
                payload["options"]["stop"] = list(self.stop_sequences)
            if self.disable_thinking:
                payload["think"] = False
            return payload
        return {"model": self.model, "messages": self._messages(prompt),
                "temperature": self.temperature, "max_tokens": max_tokens}

    # -- main call ------------------------------------------------------------ #
    def complete(self, *, prompt, seed, generation, candidate_id, parent_id,
                 max_output_tokens: Optional[int] = None, timeout_s: Optional[float] = None,
                 max_retries: Optional[int] = None):
        started = time.time()
        ts = datetime.now(timezone.utc).isoformat()
        max_tokens = int(max_output_tokens or self.max_output_tokens)
        base_timeout = float(timeout_s or self.timeout_s)
        retries_allowed = self.max_retries if max_retries is None else max(0, int(max_retries))
        call_seed = derive_call_seed(int(seed), str(candidate_id))
        payload = self._payload(prompt, max_tokens, call_seed)
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        endpoint = self.endpoint()

        success = False
        response_raw = ""
        response_text = ""
        error = None
        error_type = None
        inp = out = total = None
        done_reason = None
        cost = None
        cost_basis = None
        attempts: List[Dict[str, Any]] = []

        for attempt in range(retries_allowed + 1):
            attempt_timeout = base_timeout * (1.0 + 0.5 * attempt)  # escalate timeout on retry
            t_att = time.time()
            try:
                req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=attempt_timeout) as r:
                    raw = json.loads(r.read().decode("utf-8"))
                if self.provider == "ollama-local":
                    if raw.get("error"):
                        raise RuntimeError(f"Ollama error: {raw.get('error')}")
                    response_raw = (raw.get("message") or {}).get("content") or ""
                    inp = raw.get("prompt_eval_count")
                    out = raw.get("eval_count")
                    total = (inp + out) if inp is not None and out is not None else None
                    done_reason = raw.get("done_reason")
                else:
                    message = raw["choices"][0].get("message") or {}
                    response_raw = message.get("content") or ""
                    usage = raw.get("usage") or {}
                    inp = usage.get("prompt_tokens")
                    out = usage.get("completion_tokens")
                    total = usage.get("total_tokens")
                    done_reason = raw["choices"][0].get("finish_reason")
                response_text = clean_llm_response(response_raw)
                if (self.prefill and self.provider == "ollama-local" and response_text.strip()
                        and not response_text.lstrip().startswith("```")):
                    # Ollama returns only the continuation; restore the prefilled fence so extraction sees one code block.
                    response_text = self.prefill + response_text
                if not response_text.strip():
                    raise EmptyLLMResponse("LLM returned empty content after removing <think> sections "
                                           "(for Ollama/Qwen3 use native /api/chat with think=false).")
                cost, cost_basis = self._cost(inp, out)
                success = True
                attempts.append({"attempt": attempt, "ok": True, "timeout_s": attempt_timeout,
                                 "elapsed_s": time.time() - t_att})
                break
            except Exception as e:  # noqa: BLE001 - classified below
                error_type, retryable, msg = classify_llm_error(e)
                error = f"{error_type}: {msg}"
                attempts.append({"attempt": attempt, "ok": False, "timeout_s": attempt_timeout,
                                 "elapsed_s": time.time() - t_att, "error_type": error_type, "error": msg})
                if (not retryable) or attempt >= retries_allowed:
                    break
                time.sleep(self.retry_backoff_s * (attempt + 1))

        latency = time.time() - started
        retry_count = max(0, len(attempts) - 1)
        record = {"timestamp": ts, "seed": seed, "generation": generation, "candidate_id": candidate_id,
                  "parent_id": parent_id, "model": self.model, "model_version": self.model_version,
                  "temperature": self.temperature, "max_output_tokens": max_tokens,
                  "sampling_seed": call_seed if self.provider == "ollama-local" else None,
                  "prompt_version": PROMPT_VERSION, "system_prompt": self.system_prompt, "assistant_prefill": self.prefill,
                  "stop_sequences": self.stop_sequences, "prompt": prompt,
                  "response": response_text, "response_raw": response_raw,
                  "done_reason": done_reason,
                  "input_tokens": inp, "output_tokens": out, "total_tokens": total,
                  "latency_s": latency, "retry_count": retry_count, "attempts": attempts,
                  "estimated_monetary_cost_usd": cost, "cost_basis": cost_basis,
                  "api_provider": self.provider, "endpoint": endpoint,
                  "success": success, "error_type": error_type, "error": error}
        with self.call_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        st = self.stats
        st["calls"] += 1
        st["retries"] += retry_count
        st["latency_s"] += latency
        if success:
            st["successes"] += 1
            if total is None:
                st["successful_calls_with_unknown_tokens"] += 1
            else:
                st["input_tokens"] += inp
                st["output_tokens"] += out
                st["total_tokens"] += total
        else:
            st["failures"] += 1
            raise LLMCallError(error_type or "LLMError", (error or "").split(": ", 1)[-1], record)
        return response_text, record


# --------------------------------------------------------------------------- #
# Prompts, response cleaning, code extraction, static validation
# --------------------------------------------------------------------------- #
def build_prompt(*, task="generate", parent_code="", parent_explanation="", feedback=""):
    intro = ("You are generating executable Python for an 8-Puzzle A* heuristic. "
             "Return a Python code block containing exactly one callable function "
             "heuristic(state) -> float and then a short explanation. Do not return "
             "pseudocode. The state is a length-9 tuple/list in row-major order; tiles "
             "1..8 are numbered and 0 is the blank. The goal is (1,2,3,4,5,6,7,8,0). "
             "Each move has cost 1. The heuristic must be finite, nonnegative, cheap, "
             "admissible, and consistent. Do not import modules. Do not access files, "
             "the network, subprocesses, reflection, eval, exec, or global state. "
             "Use only the state argument and basic Python expressions. The evaluator "
             "will reject syntax errors, forbidden names, timeouts, nonfinite output, "
             "inadmissibility, and inconsistency. Return only code plus explanation.")
    if task == "generate":
        return intro + "\nDesign a useful original heuristic under these constraints."
    if task == "repair":
        return (intro + "\nThe candidate below was REJECTED by the evaluator. Fix it so it passes, keeping the "
                "interface heuristic(state) -> float.\nRejected code:\n```python\n" + parent_code +
                "\n```\nEvaluator result:\n" + feedback)
    return (intro + "\nImprove the parent candidate below while preserving the interface and correctness requirements.\n"
            "Parent code:\n```python\n" + parent_code + "\n```\nParent explanation:\n" + parent_explanation +
            "\nFeedback:\n" + feedback)


PILOT_PROMPT = ("Reply with ONLY this code block, completing the body, and nothing else:\n"
                "```python\n"
                "def heuristic(state):\n"
                "    # state: length-9 tuple, row-major, 0 is the blank; goal is (1,2,3,4,5,6,7,8,0)\n"
                "    # return the sum over tiles 1..8 of |row-goal_row| + |col-goal_col|, as a float\n"
                "    ...\n"
                "```\n"
                "No imports. No explanation. No reasoning.")


def clean_llm_response(text: str) -> str:
    """Remove Qwen/Ollama reasoning wrappers before candidate extraction."""
    text = text or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    # Unclosed <think>: if a code fence appears after it, keep from the fence; otherwise drop the rest.
    m = re.search(r"<think>", text, flags=re.I)
    if m:
        tail = text[m.end():]
        fence = re.search(r"```", tail)
        text = text[:m.start()] + (tail[fence.start():] if fence else "")
    text = re.sub(r"</think>", "", text, flags=re.I)  # stray closing tag
    return text.strip()


def _def_block(text: str) -> str:
    """Safely cut a bare `def heuristic` block (no fence): def line + following indented/blank lines."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if re.match(r"\s*def\s+heuristic\s*\(", ln):
            out = [ln]
            for nxt in lines[i + 1:]:
                if nxt.strip() == "" or nxt[:1] in (" ", "\t"):
                    out.append(nxt)
                else:
                    break
            return "\n".join(out).strip()
    return ""


def extract_code(text: str) -> str:
    text = clean_llm_response(text)
    blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", text, flags=re.I | re.S)
    if not blocks:  # unclosed fence (e.g. output truncated by num_predict)
        m = re.search(r"```(?:python|py)?[ \t]*\r?\n(.*)$", text, flags=re.I | re.S)
        if m:
            blocks = [m.group(1)]
    for b in blocks:
        if re.search(r"def\s+heuristic\s*\(", b):
            return b.strip()
    if blocks:
        return blocks[0].strip()
    return _def_block(text)


def validate_code(code: str):
    """Static AST gate. Returns (status, message, complexity)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return "syntax_error", str(e), None
    except Exception as e:  # e.g. ValueError for null bytes
        return "syntax_error", f"{type(e).__name__}: {e}", None
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            return "forbidden_import", "imports are forbidden", None
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            return "invalid_interface", "global/nonlocal state is forbidden", None
        if isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
            return "invalid_interface", f"{type(n).__name__} is forbidden", None
        if isinstance(n, ast.Name) and (n.id in FORBIDDEN_NAMES or n.id.startswith("__")):
            return "invalid_interface", f"forbidden name {n.id}", None
        if isinstance(n, ast.Attribute) and n.attr.startswith("__"):
            return "invalid_interface", "dunder attribute forbidden", None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "heuristic"]
    if len(funcs) != 1 or len(funcs[0].args.args) != 1:
        return "invalid_interface", "heuristic(state) is required (exactly one, one argument)", None
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):  # docstring
            continue
        if isinstance(n, ast.Assign):
            try:
                ast.literal_eval(n.value)
                continue
            except Exception:
                pass
        return "invalid_interface", "only heuristic(state) and literal constants are allowed at module level", None
    complexity = sum(1 for _ in ast.walk(tree))
    return "valid", "", complexity


# --------------------------------------------------------------------------- #
# Sandbox (separate Python process; Windows-safe)
# --------------------------------------------------------------------------- #
SANDBOX_RUNNER = r'''
import sys, json, base64, math, builtins
try:
 import resource
 resource.setrlimit(resource.RLIMIT_CPU,(120,120))
 resource.setrlimit(resource.RLIMIT_AS,(1024*1024*1024,1024*1024*1024))
except Exception:
 pass
code=base64.b64decode(sys.argv[1]).decode('utf-8')
names=json.loads(sys.argv[2])
safe={k:getattr(builtins,k) for k in names if hasattr(builtins,k)}
env={'__builtins__':safe}
try:
 exec(compile(code,'candidate.py','exec'),env,env)
 f=env.get('heuristic')
 if not callable(f): raise RuntimeError('missing heuristic')
except Exception as e:
 print(json.dumps({'ok':False,'category':'runtime_error','error':f'{type(e).__name__}: {e}'}),flush=True); sys.exit(0)
try:
 states=json.loads(sys.stdin.read())
 vals=[]
 for raw in states:
  v=f(tuple(raw))
  if isinstance(v,bool): v=float(v)
  v=float(v)
  if not math.isfinite(v): raise ValueError('nonfinite output')
  vals.append(v)
 print(json.dumps({'ok':True,'values':vals}),flush=True)
except Exception as e:
 cat='nonfinite_output' if 'nonfinite' in str(e).lower() else 'runtime_error'
 print(json.dumps({'ok':False,'category':cat,'error':f'{type(e).__name__}: {e}'}),flush=True)
'''


def execute_candidate(code: str, states: List[Tuple[int, ...]], timeout_s: float):
    encoded = base64.b64encode(code.encode("utf-8")).decode()
    names = json.dumps(sorted(n for n in ALLOWED_NAMES if n not in {"True", "False", "None"}))
    payload = json.dumps([list(s) for s in states], separators=(",", ":"))
    try:
        with tempfile.TemporaryDirectory(prefix="cand_") as tmp:  # working-directory restriction
            p = subprocess.run([sys.executable, "-I", "-S", "-c", SANDBOX_RUNNER, encoded, names],
                               input=payload, text=True, encoding="utf-8", errors="replace",
                               capture_output=True, timeout=timeout_s, cwd=tmp)
    except subprocess.TimeoutExpired:
        return "timeout", "candidate exceeded wall-clock timeout", None
    except Exception as e:
        return "runtime_error", f"sandbox launch failed: {type(e).__name__}: {e}", None
    if not p.stdout.strip():
        return "runtime_error", (p.stderr or "")[-1000:] or "no sandbox response", None
    try:
        result = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return "runtime_error", p.stdout[-1000:], None
    if not result.get("ok"):
        return result.get("category", "runtime_error"), result.get("error", "sandbox failure"), None
    return "valid", "", result["values"]


# --------------------------------------------------------------------------- #
# Search / correctness
# --------------------------------------------------------------------------- #
def custom_astar(start, hmap, timeout_expansions=200000):
    if start == GOAL:
        return {'solved': 1, 'cost': 0, 'expansions': 0, 'runtime_ms': 0.0}
    t0 = time.perf_counter()
    g = {start: 0}
    heap = [(hmap[start], 0, 0, start)]
    counter = 0
    closed = set()
    expansions = 0
    while heap and expansions < timeout_expansions:
        f, ng, _, s = heapq.heappop(heap)
        if ng != g.get(s) or s in closed:
            continue
        closed.add(s)
        expansions += 1
        if s == GOAL:
            return {'solved': 1, 'cost': ng, 'expansions': expansions, 'runtime_ms': (time.perf_counter() - t0) * 1000}
        for t in neighbors(s):
            ng2 = ng + 1
            if ng2 < g.get(t, 10 ** 9):
                g[t] = ng2
                counter += 1
                heapq.heappush(heap, (ng2 + hmap[t], ng2, counter, t))
    return {'solved': 0, 'cost': None, 'expansions': expansions, 'runtime_ms': (time.perf_counter() - t0) * 1000}


def check_values(states, distances, vals):
    """Exhaustive admissibility/consistency check. Returns (status, h, max_over, max_violation, counterexample)."""
    h = dict(zip(states, vals))
    admiss = True
    consistent = True
    max_over = 0.0
    max_violation = 0.0
    ce_over = None
    ce_viol = None
    for s, d in distances.items():
        v = h[s] - d
        if v > 1e-9:
            admiss = False
            if v > max_over:
                max_over = v
                ce_over = {'state': list(s), 'h': h[s], 'true_distance': d}
        for t in neighbors(s):
            q = h[s] - h[t] - 1
            if q > 1e-9:
                consistent = False
                if q > max_violation:
                    max_violation = q
                    ce_viol = {'state': list(s), 'h': h[s], 'successor': list(t), 'successor_h': h[t]}
    if not admiss:
        return "inadmissible", h, max_over, max_violation, ce_over
    if not consistent:
        return "inconsistent", h, max_over, max_violation, ce_viol
    return "valid", h, max_over, max_violation, None


def code_hash(code):
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def safe_mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None


def random_code(rng):
    # Same executable candidate interface; random evolution has no semantic model calls.
    a, b = rng.random(), rng.random()
    c = rng.random()
    total = a + b + c
    a, b, c = [x / total for x in (a, b, c)]
    return f'''def heuristic(state):\n    goal=(1,2,3,4,5,6,7,8,0)\n    m=0\n    for i,x in enumerate(state):\n        if x:\n            j=goal.index(x); m += abs(i//3-j//3)+abs(i%3-j%3)\n    misplaced=sum(1 for i,x in enumerate(state) if x and x != goal[i])\n    return {a:.12f}*m + {b:.12f}*misplaced + {c:.12f}*0.0\n'''


def baseline_code(name):
    if name == "zero":
        return "def heuristic(state):\n    return 0.0\n"
    if name == "misplaced":
        return "def heuristic(state):\n    goal=(1,2,3,4,5,6,7,8,0)\n    return float(sum(1 for i,x in enumerate(state) if x and x != goal[i]))\n"
    if name == "manhattan":
        return "def heuristic(state):\n    goal=(1,2,3,4,5,6,7,8,0)\n    total=0\n    for i,x in enumerate(state):\n        if x:\n            j=goal.index(x); total += abs(i//3-j//3)+abs(i%3-j%3)\n    return float(total)\n"
    if name == "linear_conflict":
        # NOTE: counts every inverted pair (not the minimal tile removals), so it is NOT guaranteed
        # admissible/consistent.  Whatever the exhaustive checks report is recorded as-is.
        # It also differs from the isolated linear-conflict *feature* used in experiment.py.
        return "def heuristic(state):\n    goal=(1,2,3,4,5,6,7,8,0)\n    total=0\n    for i,x in enumerate(state):\n        if x:\n            j=goal.index(x); total += abs(i//3-j//3)+abs(i%3-j%3)\n    conflicts=0\n    for row in range(3):\n        vals=[]\n        for col in range(3):\n            x=state[3*row+col]\n            if x and goal.index(x)//3==row: vals.append(goal.index(x)%3)\n        conflicts += sum(vals[i]>vals[j] for i in range(len(vals)) for j in range(i+1,len(vals)))\n    for col in range(3):\n        vals=[]\n        for row in range(3):\n            x=state[3*row+col]\n            if x and goal.index(x)%3==col: vals.append(goal.index(x)//3)\n        conflicts += sum(vals[i]>vals[j] for i in range(len(vals)) for j in range(i+1,len(vals)))\n    return float(total + 2*conflicts)\n"
    raise ValueError(name)


def evaluate(code, candidate_id, parent_id, seed, generation, model, states, distances, train, test, timeout_s,
             execute=True, model_version=None):
    code = code or ""
    if not code.strip():
        status, msg, complexity = "no_code_extracted", "no Python code could be extracted from the response", None
    else:
        status, msg, complexity = validate_code(code)
    base = {'candidate_id': candidate_id, 'parent_id': parent_id, 'seed': seed, 'generation': generation,
            'model': model, 'model_version': model_version or model,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'candidate_code_hash': code_hash(code), 'candidate_code': code, 'code_complexity': complexity,
            'status': status, 'failure_category': status if status != "valid" else "", 'failure_message': msg}
    if status != "valid" or not execute:
        return base
    run_status, run_msg, vals = execute_candidate(code, states, timeout_s)
    if run_status != "valid":
        base.update(status=run_status, failure_category=run_status, failure_message=run_msg)
        return base
    if any(v < -1e-9 for v in vals):
        base.update(status="negative_output", failure_category="negative_output",
                    failure_message="heuristic returned a negative value")
        return base
    check, hmap, max_over, max_v, counterexample = check_values(states, distances, vals)
    if check != "valid":
        base.update(status=check, failure_category=check, failure_message=check, counterexample=counterexample,
                    admissible=int(check != "inadmissible"), consistent=0,
                    max_overestimate=max_over, max_consistency_violation=max_v)
        return base
    train_rows = [custom_astar(s, hmap) for s in train]
    test_rows = [custom_astar(s, hmap) for s in test]
    base.update(status="valid", failure_category="", failure_message="", admissible=1, consistent=1,
                max_overestimate=max_over, max_consistency_violation=max_v,
                train_solved=sum(r['solved'] for r in train_rows),
                train_mean_expansions=safe_mean(r['expansions'] for r in train_rows),
                train_mean_solution_cost=safe_mean(r['cost'] for r in train_rows if r['cost'] is not None),
                train_mean_runtime_ms=safe_mean(r['runtime_ms'] for r in train_rows),
                test_solved=sum(r['solved'] for r in test_rows),
                test_mean_expansions=safe_mean(r['expansions'] for r in test_rows),
                test_mean_solution_cost=safe_mean(r['cost'] for r in test_rows if r['cost'] is not None),
                test_mean_runtime_ms=safe_mean(r['runtime_ms'] for r in test_rows),
                behavioral_probe=vals[::max(1, len(vals) // 32)][:32])
    return base


def fitness_key(r):
    return (1 if r.get('status') == 'valid' else 0, r.get('train_solved', -1),
            -(r.get('train_mean_expansions') if r.get('train_mean_expansions') is not None else 10 ** 18),
            -(r.get('train_mean_runtime_ms') if r.get('train_mean_runtime_ms') is not None else 10 ** 18),
            -(r.get('code_complexity') if r.get('code_complexity') is not None else 10 ** 9))


def select_population(rows, n, diverse=True):
    """Correctness is lexicographically prior: only status=='valid' candidates can be selected."""
    valid = sorted([r for r in rows if r.get('status') == 'valid'], key=fitness_key, reverse=True)
    if not diverse or len(valid) <= n:
        return valid[:n]
    selected = []
    while valid and len(selected) < n:
        if not selected:
            chosen = valid.pop(0)
        else:
            def dist(x):
                a = x.get('behavioral_probe', [])
                return min(sum(abs(u - v) for u, v in zip(a, y.get('behavioral_probe', []))) for y in selected)
            chosen = max(valid, key=lambda x: (dist(x), fitness_key(x)))
            valid.remove(chosen)
        selected.append(chosen)
    return selected


REPAIRABLE = {'inadmissible', 'inconsistent', 'syntax_error', 'forbidden_import', 'invalid_interface',
              'runtime_error', 'timeout', 'nonfinite_output', 'negative_output'}


def rejection_feedback(r: Dict[str, Any]) -> str:
    """Concrete, factual evaluator feedback for a rejected candidate (no hints about the solution)."""
    st = r.get('status')
    ce = r.get('counterexample') or {}
    if st == 'inadmissible':
        return (f"Gate failed: INADMISSIBLE. h overestimates the true shortest distance by up to {r.get('max_overestimate'):g}. "
                f"Counterexample: state={ce.get('state')} has h={ce.get('h'):g} but the true distance to the goal is "
                f"{ce.get('true_distance')}. h must never exceed the true distance.")
    if st == 'inconsistent':
        return (f"Gate failed: INCONSISTENT. Required: h(s) <= 1 + h(s') for every legal move. Counterexample: "
                f"s={ce.get('state')} has h={ce.get('h'):g}; neighbour s'={ce.get('successor')} has "
                f"h={ce.get('successor_h'):g}, so h(s) - h(s') = {ce.get('h') - ce.get('successor_h'):g} > 1.")
    return f"Gate failed: {st}. Detail: {r.get('failure_message')}"


def improve_feedback(r: Dict[str, Any], n_train: int) -> str:
    return (f"Passed all correctness checks (admissible and consistent). Training result: solved "
            f"{r.get('train_solved')}/{n_train} instances, mean node expansions={r.get('train_mean_expansions')} "
            f"(lower is better). Reduce expansions while staying admissible and consistent.")


def pick_repair_sources(rows):
    def rank(r):
        st = r.get('status')
        if st == 'inadmissible':
            return (0, r.get('max_overestimate') or 0.0)
        if st == 'inconsistent':
            return (0, r.get('max_consistency_violation') or 0.0)
        return (1, 0.0)
    cand = [r for r in rows if r.get('status') in REPAIRABLE and (r.get('candidate_code') or '').strip()]
    return sorted(cand, key=rank)


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summary_stats(rows):
    out = []
    for setting in sorted(set(r['setting'] for r in rows)):
        rr = [r for r in rows if r['setting'] == setting]

        def agg(k):
            x = [float(r[k]) for r in rr if r.get(k) is not None]
            return (statistics.mean(x), statistics.stdev(x) if len(x) > 1 else 0.0) if x else (None, None)
        em, es = agg('test_mean_expansions')
        tm, ts = agg('test_mean_runtime_ms')
        cm, cs = agg('total_tokens')
        out.append({'setting': setting, 'n_seeds': len(rr), 'n_valid': sum(r.get('status') == 'valid' for r in rr),
                    'test_expansions_mean': em, 'test_expansions_sd': es,
                    'test_runtime_ms_mean': tm, 'test_runtime_ms_sd': ts, 'total_tokens_mean': cm, 'total_tokens_sd': cs,
                    'solved_rate': statistics.mean(r.get('test_solved', 0) / max(1, r.get('test_n', 1)) for r in rr),
                    'admissibility_rate': statistics.mean(r.get('admissible', 0) for r in rr),
                    'consistency_rate': statistics.mean(r.get('consistent', 0) for r in rr),
                    'llm_calls': sum(r.get('llm_calls', 0) for r in rr),
                    'estimated_cost_usd': sum(r.get('estimated_cost_usd', 0) or 0 for r in rr)})
    return out


# --------------------------------------------------------------------------- #
# Pilot (isolated: never touches baselines / evolution / statistics)
# --------------------------------------------------------------------------- #
def local_distances(max_depth: int = 14) -> Dict[Tuple[int, ...], int]:
    """Exact goal distances for states within BFS depth <= max_depth (~5k states; cheap)."""
    dist = {GOAL: 0}
    q = deque([GOAL])
    while q:
        s = q.popleft()
        if dist[s] >= max_depth:
            continue
        for t in neighbors(s):
            if t not in dist:
                dist[t] = dist[s] + 1
                q.append(t)
    return dist


def smoke_test_candidate(code: str, timeout_s: float = 20.0, max_depth: int = 14) -> Dict[str, Any]:
    dist = local_distances(max_depth)
    states = list(dist.keys())
    scope = f"PARTIAL smoke test: all {len(states)} states within BFS depth <= {max_depth} of goal (not exhaustive; passing is NOT a correctness proof)"
    status, msg, vals = execute_candidate(code, states, timeout_s)
    res = {'status': status, 'message': msg, 'states_tested': len(states), 'scope': scope,
           'goal_value': None, 'max_overestimate': None, 'max_consistency_violation': None}
    if status != "valid":
        return res
    h = dict(zip(states, vals))
    res['goal_value'] = h[GOAL]
    if min(vals) < -1e-9:
        res.update(status="negative_output", message="heuristic returned a negative value")
        return res
    max_over = max(max(0.0, h[s] - d) for s, d in dist.items())
    max_viol = 0.0
    for s in states:
        for t in neighbors(s):
            if t in h:
                max_viol = max(max_viol, h[s] - h[t] - 1)
    res['max_overestimate'] = max_over
    res['max_consistency_violation'] = max(0.0, max_viol)
    if max_over > 1e-9:
        res.update(status="inadmissible", message=f"h exceeds true distance by up to {max_over}")
    elif max_viol > 1e-9:
        res.update(status="inconsistent", message=f"h(s) > 1 + h(s') by up to {max_viol}")
    return res


def run_pilot(cfg: Dict[str, Any], out_root: Path) -> int:
    t_start = time.time()
    llm_cfg = cfg.get('llm', {})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = out_root / f"pilot-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        'status': 'pilot_failed', 'llm_status': False, 'model': llm_cfg.get('model'),
        'provider': llm_cfg.get('provider'), 'candidate_status': None, 'candidate_correctness': None,
        'elapsed_seconds': None, 'tokens': None, 'input_tokens': None, 'output_tokens': None,
        'retries': 0, 'error_type': None, 'error': None, 'pipeline_ok': False, 'candidate_accepted': False,
        'pilot_note': ("Plumbing check only (the pilot prompt instructs Manhattan distance). "
                       "This is NOT an experimental result.")}

    def finish(code: int) -> int:
        result['elapsed_seconds'] = round(time.time() - t_start, 3)
        text = json.dumps(result, indent=2, ensure_ascii=True)
        for p in (out / 'pilot_result.json', HERE / 'pilot_result.json'):
            try:
                p.write_text(text, encoding='utf-8')
            except Exception:
                pass
        print(text)
        return code

    try:
        client = OpenAICompatibleClient(llm_cfg, out / 'llm_calls.jsonl')
    except MissingLLM as e:
        result.update(error_type='MissingLLM', error=str(e))
        return finish(1)
    result['model'] = client.model
    result['endpoint'] = client.endpoint()

    health = client.check_endpoint()
    result['endpoint_check'] = {k: v for k, v in health.items() if k != 'installed_models'}
    if health.get('checked') and not health.get('reachable'):
        result.update(error_type=health.get('error_type', 'ConnectionError'),
                      error=health.get('error', 'endpoint not reachable'))
        return finish(1)
    if health.get('checked') and health.get('model_available') is False:
        result.update(error_type='ModelNotFound',
                      error=f"model '{client.model}' not installed; installed: {health.get('installed_models')}. "
                            f"Run: ollama pull {client.model}")
        return finish(1)

    seed = (cfg.get('seeds') or [1])[0]
    result['seed'] = seed
    try:
        text, meta = client.complete(prompt=PILOT_PROMPT, seed=seed, generation=0,
                                     candidate_id=f'pilot-s{seed}-g0-0', parent_id=None,
                                     max_output_tokens=client.pilot_max_output_tokens,
                                     timeout_s=client.pilot_timeout_s,
                                     max_retries=min(client.max_retries, 1))
    except LLMCallError as e:
        rec = e.record or {}
        result.update(error_type=e.error_type, error=e.message, retries=rec.get('retry_count', 0))
        return finish(1)
    except Exception as e:  # noqa: BLE001
        result.update(error_type=type(e).__name__, error=str(e))
        return finish(1)

    result.update(llm_status=True, llm_latency_s=round(meta.get('latency_s') or 0.0, 3),
                  tokens=meta.get('total_tokens'), input_tokens=meta.get('input_tokens'),
                  output_tokens=meta.get('output_tokens'), retries=meta.get('retry_count', 0),
                  done_reason=meta.get('done_reason'), response_preview=text[:1000])

    code = extract_code(text)
    result['extracted_code'] = code
    if not code.strip():
        trunc = (meta.get('done_reason') == 'length')
        result.update(candidate_status='no_code_extracted', error_type='no_code_extracted',
                      error=('no Python code could be extracted; the response was TRUNCATED by the output-token '
                             f"limit ({client.pilot_max_output_tokens}) - the model produced prose instead of code. "
                             'Raise pilot_max_output_tokens or inspect response_preview.') if trunc
                      else 'no Python code could be extracted from the response')
        return finish(1)
    status, msg, _ = validate_code(code)
    result['validation'] = {'status': status, 'message': msg}
    if status != 'valid':
        result.update(candidate_status=status, error_type=status, error=msg)
        return finish(1)
    smoke = smoke_test_candidate(code, timeout_s=float(cfg.get('candidate_timeout_s', 10)) + 10)
    result['candidate_correctness'] = smoke
    result['candidate_status'] = smoke['status']
    # Rejection by a correctness gate (inadmissible / inconsistent / negative output) means the whole chain
    # Ollama -> LLM -> extraction -> AST -> sandbox -> checks ran end-to-end: the PIPELINE works and the gate
    # did its job.  Sandbox crashes / timeouts / non-finite output are ambiguous (could be a sandbox bug), so
    # those stay pilot_failed.
    gate_outcomes = {'valid', 'inadmissible', 'inconsistent', 'negative_output'}
    result['pipeline_ok'] = smoke['status'] in gate_outcomes
    result['candidate_accepted'] = (smoke['status'] == 'valid')
    if smoke['status'] not in gate_outcomes:
        result.update(error_type=smoke['status'], error=smoke['message'])
        return finish(1)
    if smoke['status'] == 'valid':
        result.update(status='pilot_complete', error=None, error_type=None)
    else:
        result.update(status='pilot_complete', error_type=None, error=None,
                      candidate_rejection={'gate': smoke['status'], 'detail': smoke['message']},
                      pilot_note=result['pilot_note'] + " The candidate was REJECTED by the correctness gate "
                                 "(this is the gate working, not a pipeline failure).")
    return finish(0)


# --------------------------------------------------------------------------- #
# Full experiment helpers
# --------------------------------------------------------------------------- #
class Budget:
    def __init__(self, max_calls: Optional[int]):
        self.max = max_calls
        self.used = 0

    def exhausted(self) -> bool:
        return self.max is not None and self.used >= self.max


def llm_candidate(client, budget, ctx, *, prompt, seed, generation, cid, parent_id, setting, task='generate'):
    """One real LLM call -> extraction -> evaluation. LLM failures are recorded, never silently turned into candidates."""
    llm_fields = {'llm_calls': 0, 'llm_success': None, 'llm_error_type': None, 'llm_retry_count': 0,
                  'total_tokens': None, 'llm_input_tokens': None, 'llm_output_tokens': None,
                  'llm_latency_s': None, 'estimated_cost_usd': None, 'temperature': client.temperature,
                  'llm_done_reason': None, 'llm_prompt': prompt, 'llm_response': None}
    if budget.exhausted():
        row = evaluate("", cid, parent_id, seed, generation, client.model, *ctx['eval_args'],
                       model_version=client.model_version, execute=False)
        row.update(status='llm_budget_exhausted', failure_category='llm_budget_exhausted',
                   failure_message=f'--max-llm-calls={budget.max} reached; call not made')
    else:
        budget.used += 1
        try:
            text, meta = client.complete(prompt=prompt, seed=seed, generation=generation,
                                         candidate_id=cid, parent_id=parent_id)
            row = evaluate(extract_code(text), cid, parent_id, seed, generation, client.model,
                           *ctx['eval_args'], model_version=client.model_version)
            if row.get('status') == 'no_code_extracted' and meta.get('done_reason') == 'length':
                row['failure_message'] += f" (response truncated by max_output_tokens={meta.get('max_output_tokens')})"
            llm_fields.update(llm_calls=1, llm_success=True, llm_response=text)
        except LLMCallError as e:
            meta = e.record or {}
            row = evaluate("", cid, parent_id, seed, generation, client.model, *ctx['eval_args'],
                           model_version=client.model_version, execute=False)
            row.update(status='llm_error', failure_category=e.error_type, failure_message=e.message)
            llm_fields.update(llm_calls=1, llm_success=False, llm_error_type=e.error_type,
                              llm_response=meta.get('response'))
        llm_fields.update(llm_retry_count=meta.get('retry_count', 0), total_tokens=meta.get('total_tokens'),
                          llm_input_tokens=meta.get('input_tokens'), llm_output_tokens=meta.get('output_tokens'),
                          llm_latency_s=meta.get('latency_s'), llm_done_reason=meta.get('done_reason'),
                          estimated_cost_usd=meta.get('estimated_monetary_cost_usd'))
    row.update(llm_fields)
    row.update(setting=setting, test_n=ctx['test_n'], task=task)
    return row


def parse_seeds(s: str) -> List[int]:
    try:
        seeds = [int(x) for x in s.split(',') if x.strip() != '']
    except ValueError:
        raise SystemExit(f"--seeds must be comma-separated integers, got: {s!r}")
    if not seeds:
        raise SystemExit("--seeds is empty")
    return seeds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument('--config', default=str(HERE / 'llm_config.json'))
    ap.add_argument('--out-root', default=str(HERE / 'results' / 'llm_runs'))
    ap.add_argument('--dry-run', action='store_true', help='print the plan (incl. number of LLM calls); no network, no calls')
    ap.add_argument('--pilot', action='store_true',
                    help='ONE small real LLM call + extraction + AST + sandbox smoke test, then exit')
    ap.add_argument('--seeds', default=None, help='override seeds, e.g. 1 or 1,2')
    ap.add_argument('--population-size', type=int, default=None)
    ap.add_argument('--generations', type=int, default=None)
    ap.add_argument('--max-llm-calls', type=int, default=None, help='hard cap on LLM calls (attempts, not retries)')
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    outroot = Path(args.out_root)
    llm_cfg = cfg['llm']

    seeds = parse_seeds(args.seeds) if args.seeds else [int(s) for s in cfg['seeds']]
    pop_n = args.population_size if args.population_size is not None else int(cfg['population_size'])
    generations = args.generations if args.generations is not None else int(cfg['generations'])
    if pop_n < 1 or generations < 0:
        raise SystemExit("--population-size must be >= 1 and --generations >= 0")
    planned_llm_calls = len(seeds) * (1 + pop_n + generations * pop_n)
    reduced_run = (seeds != [int(s) for s in cfg['seeds']] or pop_n != int(cfg['population_size'])
                   or generations != int(cfg['generations']) or args.max_llm_calls is not None)

    if args.dry_run:
        print(json.dumps({'status': 'dry_run', 'requires_endpoint': True,
                          'base_url_env': llm_cfg.get('base_url_env'), 'model_env': llm_cfg.get('model_env'),
                          'model': llm_cfg.get('model'), 'seeds': seeds, 'population_size': pop_n,
                          'generations': generations, 'planned_llm_calls': planned_llm_calls,
                          'max_llm_calls': args.max_llm_calls, 'reduced_run': reduced_run,
                          'llm_calls_made': 0}, indent=2))
        return 0

    # ---- PILOT: isolated; returns before baselines / one-shot / evolution / statistics ----
    if args.pilot:
        return run_pilot(cfg, outroot)

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + os.urandom(3).hex()
    out = outroot / run_id
    out.mkdir(parents=True)
    effective = {'seeds': seeds, 'population_size': pop_n, 'generations': generations,
                 'max_llm_calls': args.max_llm_calls, 'planned_llm_calls': planned_llm_calls,
                 'reduced_run': reduced_run}
    (out / 'configuration.json').write_text(json.dumps({**cfg, 'effective_run': effective}, indent=2), encoding='utf-8')
    call_log = out / 'llm_calls.jsonl'
    try:
        client = OpenAICompatibleClient(llm_cfg, call_log)
    except MissingLLM as e:
        (out / 'execution_error.json').write_text(json.dumps({'status': 'not_executed', 'reason': str(e)}, indent=2))
        print(json.dumps({'status': 'not_executed', 'error_type': 'MissingLLM', 'error': str(e)}, indent=2))
        return 1
    effective.update(prompt_version=PROMPT_VERSION, system_prompt=client.system_prompt,
                     assistant_prefill=client.prefill, stop_sequences=client.stop_sequences,
                     max_output_tokens=client.max_output_tokens, temperature=client.temperature)
    (out / 'configuration.json').write_text(json.dumps({**cfg, 'effective_run': effective}, indent=2), encoding='utf-8')
    health = client.check_endpoint()
    if health.get('checked') and (not health.get('reachable') or health.get('model_available') is False):
        info = {'status': 'not_executed', 'reason': 'LLM endpoint/model unavailable', 'endpoint_check': health}
        (out / 'execution_error.json').write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2))
        return 1

    budget = Budget(args.max_llm_calls)
    timeout_s = max(float(cfg['candidate_timeout_s']), float(cfg.get('all_state_check_timeout_s', 60)))
    distances = all_distances()
    all_states = list(distances.keys())
    raw: List[Dict[str, Any]] = []
    histories: List[Dict[str, Any]] = []
    run_status = 'executed'

    try:
        for seed in seeds:
            rng = random.Random(seed)
            train = [scramble(rng, cfg['train_depth']) for _ in range(cfg['train_n'])]
            test = [scramble(rng, cfg['test_depth']) for _ in range(cfg['test_n'])]
            ctx = {'eval_args': (all_states, distances, train, test, timeout_s), 'test_n': len(test)}
            div = cfg.get('diversity_preservation', True)

            for b in ['zero', 'misplaced', 'manhattan', 'linear_conflict']:
                r = evaluate(baseline_code(b), f'{b}-s{seed}', None, seed, 0, 'hand-coded', all_states, distances,
                             train, test, timeout_s)
                r.update(setting=b, test_n=len(test), llm_calls=0, total_tokens=0, estimated_cost_usd=0.0)
                raw.append(r)

            # One-shot LLM.
            r = llm_candidate(client, budget, ctx, prompt=build_prompt(task='generate'), seed=seed, generation=0,
                              cid=f'oneshot-s{seed}-g0-0', parent_id=None, setting='one_shot_llm')
            raw.append(r)

            # Random evolution: same population/generation budget, no model calls.
            pop = []
            for i in range(pop_n):
                r = evaluate(random_code(rng), f'random-s{seed}-g0-{i}', None, seed, 0, 'random-generator',
                             all_states, distances, train, test, timeout_s)
                r.update(setting='random_evolution', test_n=len(test), llm_calls=0, total_tokens=0, estimated_cost_usd=0.0)
                pop.append(r)
                raw.append(r)
            for gen in range(1, generations + 1):
                selected = select_population(pop, pop_n, diverse=False)
                for i in range(pop_n):
                    parent = selected[i % len(selected)]['candidate_id'] if selected else None
                    r = evaluate(random_code(rng), f'random-s{seed}-g{gen}-{i}', parent, seed, gen, 'random-generator',
                                 all_states, distances, train, test, timeout_s)
                    r.update(setting='random_evolution', test_n=len(test), llm_calls=0, total_tokens=0, estimated_cost_usd=0.0)
                    pop.append(r)
                    raw.append(r)
            sel = select_population(pop, pop_n, False)
            if sel:
                raw.append({**sel[0], 'setting': 'random_evolution_final', 'generation': generations})
            else:
                raw.append({'setting': 'random_evolution_final', 'seed': seed, 'status': 'no_valid_candidate',
                            'test_n': len(test), 'generation': generations, 'llm_calls': 0,
                            'total_tokens': 0, 'estimated_cost_usd': 0.0})

            # LLM evolution.
            pop = []
            for i in range(pop_n):
                r = llm_candidate(client, budget, ctx, prompt=build_prompt(task='generate'), seed=seed, generation=0,
                                  cid=f'llm-s{seed}-g0-{i}', parent_id=None, setting='llm_evolution')
                pop.append(r)
                raw.append(r)
            for gen in range(1, generations + 1):
                selected = select_population(pop, pop_n, diverse=div)
                sources = [] if selected else pick_repair_sources(pop)
                mode = 'improve' if selected else ('repair' if sources else 'generate')
                histories.append({'seed': seed, 'generation': gen, 'setting': 'llm_evolution', 'mode': mode,
                                  'valid_population': len(selected),
                                  'best_train_expansions': selected[0].get('train_mean_expansions') if selected else None,
                                  'behavioral_diversity': len({tuple(r.get('behavioral_probe', [])) for r in selected}),
                                  'fallback_regenerate': mode == 'generate'})
                new = []
                for i in range(pop_n):
                    cid = f'llm-s{seed}-g{gen}-{i}'
                    if mode == 'improve':
                        parent = selected[i % len(selected)]
                        prompt = build_prompt(task='improve', parent_code=parent.get('candidate_code', ''),
                                              parent_explanation='', feedback=improve_feedback(parent, len(train)))
                        parent_id = parent['candidate_id']
                    elif mode == 'repair':  # no valid parent: repair the least-bad rejected candidate (new candidate id)
                        parent = sources[i % len(sources)]
                        prompt = build_prompt(task='repair', parent_code=parent.get('candidate_code', ''),
                                              feedback=rejection_feedback(parent))
                        parent_id = parent['candidate_id']
                    else:  # nothing repairable either: regenerate from scratch (recorded in history)
                        prompt, parent_id = build_prompt(task='generate'), None
                    r = llm_candidate(client, budget, ctx, prompt=prompt, seed=seed, generation=gen, cid=cid,
                                      parent_id=parent_id, setting='llm_evolution', task=mode)
                    new.append(r)
                    raw.append(r)
                pop = selected + new  # elitism: valid parents are carried over (random evolution also keeps all)
            sel = select_population(pop, pop_n, div)
            if sel:
                raw.append({**sel[0], 'setting': 'llm_evolution_final', 'generation': generations})
            else:
                raw.append({'setting': 'llm_evolution_final', 'seed': seed, 'status': 'no_valid_candidate',
                            'test_n': len(test), 'generation': generations, 'llm_calls': 0,
                            'total_tokens': None, 'estimated_cost_usd': None})
    except KeyboardInterrupt:
        run_status = 'interrupted'

    # ---- outputs (also written for interrupted runs) ----
    final_names = {'zero', 'misplaced', 'manhattan', 'linear_conflict', 'one_shot_llm',
                   'random_evolution_final', 'llm_evolution_final'}
    finals = [r for r in raw if r.get('setting') in final_names]
    big_text = {'llm_prompt', 'llm_response'}
    if finals:
        with (out / 'raw_runs.csv').open('w', newline='', encoding='utf-8') as f:
            fields = sorted({k for r in finals for k, v in r.items() if not isinstance(v, list) and k not in big_text})
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(finals)
    write_jsonl(out / 'candidate_log.jsonl', [{k: v for k, v in r.items() if k != 'behavioral_probe'} for r in raw])
    with (out / 'evolution_history.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['seed', 'generation', 'setting', 'valid_population', 'best_train_expansions',
                                          'behavioral_diversity', 'fallback_regenerate', 'mode'])
        w.writeheader()
        w.writerows(histories)
    sums = summary_stats(finals) if finals else []
    if sums:
        with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(sums[0].keys()))
            w.writeheader()
            w.writerows(sums)

    evo = {r['seed']: r for r in finals if r['setting'] == 'llm_evolution_final' and r.get('test_mean_expansions') is not None}
    man = {r['seed']: r for r in finals if r['setting'] == 'manhattan' and r.get('test_mean_expansions') is not None}
    pair_seeds = sorted(set(evo) & set(man))
    paired = {'n': 0, 'statistic': None, 'p_value': None, 'effect_size_rank_biserial': None,
              'pairs_used_seeds': pair_seeds, 'seeds_requested': seeds}
    if pair_seeds:
        x = [evo[s]['test_mean_expansions'] for s in pair_seeds]
        y = [man[s]['test_mean_expansions'] for s in pair_seeds]
        paired.update(wilcoxon_signed_rank(x, y))
        d = [a - b for a, b in zip(x, y) if a != b]
        paired['effect_size_rank_biserial'] = (sum(v < 0 for v in d) - sum(v > 0 for v in d)) / len(d) if d else 0.0
    else:
        paired['note'] = 'no seed has both a valid LLM-evolution final candidate and a valid Manhattan baseline'

    llm_usage = dict(client.stats)
    llm_usage.update(provider=client.provider, model=client.model, model_version=client.model_version,
                     max_llm_calls=args.max_llm_calls,
                     cost_note="estimated_monetary_cost_usd is 0.0 for local inference only as 'no provider charge'; local compute is not priced.",
                     tokens_note="token sums include only successful calls with reported counts; unknown counts are null, not 0.")
    stats = {'llm_evolution_vs_manhattan_test_expansions': paired, 'status': run_status,
             'reduced_run': reduced_run, 'effective_run': effective, 'llm_usage': llm_usage,
             'note': 'Results here come from real LLM calls only; offline surrogate (experiment.py) results are separate.'}
    (out / 'statistical_tests.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps({'status': run_status, 'run_directory': str(out), 'llm_calls_file': str(call_log),
                      'reduced_run': reduced_run, 'llm_calls': llm_usage['calls'],
                      'llm_successes': llm_usage['successes'], 'llm_failures': llm_usage['failures'],
                      'llm_retries': llm_usage['retries']}, indent=2))
    return 0 if run_status == 'executed' else 130


if __name__ == '__main__':
    sys.exit(main())