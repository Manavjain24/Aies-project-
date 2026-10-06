"""Minimal FastAPI product backend. [NOT EXECUTED]"""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess, sys, json

app=FastAPI(title='Evolving Heuristics Demo')
ROOT=Path(__file__).parent
RESULTS=ROOT/'results'

class RunRequest(BaseModel):
    seeds: str='1,2,3,4,5'
    generations: int=4

@app.get('/health')
def health(): return {'status':'ok','execution':'not verified'}

@app.post('/evolve')
def evolve(req: RunRequest):
    if not (1 <= req.generations <= 100): raise HTTPException(400,'generations must be 1..100')
    cmd=[sys.executable,str(ROOT/'experiment.py'),'--out',str(RESULTS),'--seeds',req.seeds]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=300)
    if p.returncode: raise HTTPException(500,p.stderr[-1000:])
    return json.loads(p.stdout)

@app.get('/results')
def results():
    f=RESULTS/'results.csv'
    if not f.exists(): return {'status':'not_run','rows':[]}
    import csv
    return {'status':'available','rows':list(csv.DictReader(f.open()))}
