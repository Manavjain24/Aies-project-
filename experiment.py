#!/usr/bin/env python3
"""Reproducible offline 8-Puzzle experiment for the research package.

The script executes a small, deterministic evolutionary search over a safe,
interpretable heuristic representation. It does not call an LLM unless an
external adapter is added; therefore the executed experiment is an offline
surrogate, not an LLM experiment. All numeric outputs are written to CSV/JSON.
"""
from __future__ import annotations
import argparse, csv, heapq, json, math, os, random, statistics, time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

GOAL = (1,2,3,4,5,6,7,8,0)
MOVES = ((-1,0),(1,0),(0,-1),(0,1))


def neighbors(state: Tuple[int,...]):
    z = state.index(0); r,c = divmod(z,3)
    for dr,dc in MOVES:
        nr,nc=r+dr,c+dc
        if 0<=nr<3 and 0<=nc<3:
            j=nr*3+nc; s=list(state); s[z],s[j]=s[j],s[z]
            yield tuple(s)


def all_distances():
    dist={GOAL:0}; q=deque([GOAL])
    while q:
        s=q.popleft()
        for t in neighbors(s):
            if t not in dist:
                dist[t]=dist[s]+1; q.append(t)
    return dist


def scramble(rng: random.Random, depth: int):
    s=GOAL; prev=None
    for _ in range(depth):
        opts=[t for t in neighbors(s) if t!=prev]
        t=rng.choice(opts); prev,s=s,t
    return s


def misplaced(s):
    return sum(x != y and x != 0 for x,y in zip(s,GOAL))


def manhattan(s):
    total=0
    for i,x in enumerate(s):
        if x:
            gi=GOAL.index(x); total += abs(i//3-gi//3)+abs(i%3-gi%3)
    return total


def linear_conflict(s):
    conflicts=0
    # A pair in the same goal row/column but reversed contributes 2.
    for row in range(3):
        vals=[]
        for col in range(3):
            x=s[3*row+col]
            if x and (GOAL.index(x)//3)==row: vals.append(GOAL.index(x)%3)
        conflicts += 2*sum(vals[i]>vals[j] for i in range(len(vals)) for j in range(i+1,len(vals)))
    for col in range(3):
        vals=[]
        for row in range(3):
            x=s[3*row+col]
            if x and (GOAL.index(x)%3)==col: vals.append(GOAL.index(x)//3)
        conflicts += 2*sum(vals[i]>vals[j] for i in range(len(vals)) for j in range(i+1,len(vals)))
    return conflicts


def heuristic(s, w):
    return w[0]*manhattan(s) + w[1]*misplaced(s) + w[2]*linear_conflict(s)


def astar(start, w, timeout_expansions=200000):
    if start==GOAL: return {'solved':1,'cost':0,'expansions':0,'runtime_ms':0.0}
    t0=time.perf_counter(); g={start:0}; parent={}; counter=0
    heap=[(heuristic(start,w),0,counter,start)]
    closed=set(); expansions=0
    while heap and expansions < timeout_expansions:
        f,ng,_,s=heapq.heappop(heap)
        if ng != g.get(s): continue
        if s in closed: continue
        closed.add(s); expansions += 1
        if s==GOAL:
            return {'solved':1,'cost':ng,'expansions':expansions,'runtime_ms':(time.perf_counter()-t0)*1000}
        for t in neighbors(s):
            ng2=ng+1
            if ng2 < g.get(t,10**9):
                g[t]=ng2; counter+=1
                heapq.heappush(heap,(ng2+heuristic(t,w),ng2,counter,t))
    return {'solved':0,'cost':None,'expansions':expansions,'runtime_ms':(time.perf_counter()-t0)*1000}


def candidate_key(w): return tuple(round(float(x),4) for x in w)


def random_candidate(rng):
    # Safe representation: a convex combination of admissible basis heuristics.
    a=rng.random(); b=rng.random()*(1-a); c=1-a-b
    return candidate_key((a,b,c))


def fitness(w, instances):
    rows=[astar(s,w) for s in instances]
    solved=sum(x['solved'] for x in rows)
    exp=[x['expansions'] for x in rows]
    return (solved, -statistics.mean(exp), -sum(exp)), rows


_CHECK_CACHE={}

def exact_checks(w, dist):
    w=candidate_key(w)
    if w in _CHECK_CACHE: return _CHECK_CACHE[w].copy()
    admissible=True; consistent=True; max_over=0; max_consistency_violation=0
    for s,d in dist.items():
        hv=heuristic(s,w); over=hv-d
        if over>1e-9: admissible=False; max_over=max(max_over,over)
        for t in neighbors(s):
            v=heuristic(t,w)-hv-1
            if v>1e-9: consistent=False; max_consistency_violation=max(max_consistency_violation,v)
    result={'admissible':int(admissible),'consistent':int(consistent),'max_overestimate':max_over,'max_consistency_violation':max_consistency_violation}
    _CHECK_CACHE[w]=result
    return result.copy()


def evolve(seed, train, generations=4, population_size=8, offspring_size=8):
    rng=random.Random(seed); pop=[(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0)] + [random_candidate(rng) for _ in range(population_size-3)]
    history=[]
    for gen in range(generations+1):
        scored=[]
        for w in pop:
            fit,_=fitness(w,train); scored.append((fit,w))
        scored.sort(reverse=True)
        bestfit,best=scored[0]
        history.append({'seed':seed,'generation':gen,'best_w':best,'best_train_solved':bestfit[0],'best_train_mean_neg_expansions':bestfit[1],'population_unique':len(set(pop))})
        if gen==generations: break
        elites=[w for _,w in scored[:max(2,population_size//2)]]
        children=[]
        for _ in range(offspring_size):
            base=list(rng.choice(elites)); j=rng.randrange(3); base[j]=max(0.0,base[j]+rng.uniform(-0.25,0.25))
            total=sum(base)
            if total==0: base=[1.0,0.0,0.0]
            else: base=[x/total for x in base]
            children.append(candidate_key(base))
        pop=list(dict.fromkeys(elites+children))[:population_size]
        while len(pop)<population_size: pop.append(random_candidate(rng))
    return best, history


def evaluate_setting(name, seed, w, train, test, dist):
    fit_train,train_rows=fitness(w,train); fit_test,test_rows=fitness(w,test)
    checks=exact_checks(w,dist)
    return {'setting':name,'seed':seed,'w_manhattan':w[0],'w_misplaced':w[1],'w_linear_conflict':w[2],
      'train_solved':sum(x['solved'] for x in train_rows),'test_solved':sum(x['solved'] for x in test_rows),
      'train_mean_expansions':statistics.mean(x['expansions'] for x in train_rows),'test_mean_expansions':statistics.mean(x['expansions'] for x in test_rows),
      'test_mean_solution_cost':statistics.mean([x['cost'] for x in test_rows if x['cost'] is not None]),
      'test_mean_runtime_ms':statistics.mean(x['runtime_ms'] for x in test_rows),
      'admissible':checks['admissible'],'consistent':checks['consistent'],'max_overestimate':checks['max_overestimate'],
      'max_consistency_violation':checks['max_consistency_violation'],'llm_calls':0,'input_tokens':0,'output_tokens':0,'estimated_cost_usd':0.0,
      'status':'executed_offline_surrogate'}


def wilcoxon_signed_rank(x,y):
    # Exact two-sided sign-rank for nonzero paired differences, no scipy dependency.
    diffs=[a-b for a,b in zip(x,y) if abs(a-b)>1e-12]
    n=len(diffs)
    if n==0: return {'n':0,'statistic':0.0,'p_value':1.0}
    absvals=sorted((abs(d),i) for i,d in enumerate(diffs)); ranks=[0.0]*n; k=0
    while k<n:
        j=k
        while j<n and absvals[j][0]==absvals[k][0]: j+=1
        r=(k+1+j)/2
        for q in range(k,j): ranks[absvals[q][1]]=r
        k=j
    Wplus=sum(r for r,d in zip(ranks,diffs) if d>0); Wminus=sum(r for r,d in zip(ranks,diffs) if d<0); W=min(Wplus,Wminus)
    # Enumerate signs for n<=20; experiment has 5 pairs.
    counts={}
    for mask in range(1<<n):
        s=sum(ranks[i] for i in range(n) if mask>>i&1); counts[s]=counts.get(s,0)+1
    total=2**n; p=sum(c for s,c in counts.items() if s<=W+1e-12 or s>=sum(ranks)-W-1e-12)/total
    return {'n':n,'statistic':W,'p_value':min(1.0,p)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='outputs/llm_heuristic_project/results'); ap.add_argument('--seeds',default='1,2,3,4,5'); ap.add_argument('--train-n',type=int,default=8); ap.add_argument('--test-n',type=int,default=8); ap.add_argument('--depth',type=int,default=8)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    seeds=[int(x) for x in args.seeds.split(',')]; dist=all_distances(); rows=[]; histories=[]
    for seed in seeds:
        rng=random.Random(seed)
        train=[scramble(rng,args.depth) for _ in range(args.train_n)]
        test=[scramble(rng,args.depth+2) for _ in range(args.test_n)]
        settings={'zero':(0.0,0.0,0.0),'misplaced':(0.0,1.0,0.0),'manhattan':(1.0,0.0,0.0),'linear_conflict':(0.0,0.0,1.0)}
        best,hist=evolve(seed,train); histories.extend(hist); settings['offline_evolution']=best
        for name,w in settings.items(): rows.append(evaluate_setting(name,seed,w,train,test,dist))
    fields=list(rows[0].keys())
    with (out/'results.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    with (out/'evolution_history.csv').open('w',newline='') as f:
        fields2=list(histories[0].keys()); wr=csv.DictWriter(f,fieldnames=fields2); wr.writeheader(); wr.writerows(histories)
    man=[r for r in rows if r['setting']=='manhattan']; evo=[r for r in rows if r['setting']=='offline_evolution']
    stats={'experiment':'offline_8puzzle','seeds':seeds,'instances_per_seed':{'train':args.train_n,'test':args.test_n,'train_depth':args.depth,'test_depth':args.depth+2},'distance_states':len(dist),'wilcoxon_evolution_vs_manhattan_test_mean_expansions':wilcoxon_signed_rank([r['test_mean_expansions'] for r in evo],[r['test_mean_expansions'] for r in man]),'llm_execution':{'status':'not_executed','reason':'No LLM endpoint or model weights were available; the executed loop uses an offline candidate generator.','calls':0,'tokens':0,'cost_usd':0.0},'code_execution':'executed'}
    (out/'statistical_tests.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps({'status':'executed','results':str(out/'results.csv'),'rows':len(rows),'distance_states':len(dist),'stats':str(out/'statistical_tests.json')},indent=2))

if __name__=='__main__': main()
