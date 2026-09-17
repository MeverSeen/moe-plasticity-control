"""Validate v8 locks, held-out trajectories, reward accounting, and training evidence."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run
    lock=json.loads((run/'locked_protocol.json').read_text()); source=ROOT/'experiments/v8_allocation_ppo'
    for name,h in lock['source_hashes'].items():assert digest(source/name)==h,name
    for name,h in lock['policy_hashes'].items():assert digest(run/name)==h,name
    splits=lock['splits']; seen=set()
    for seeds in splits.values():assert not seen.intersection(seeds);seen.update(seeds)
    historical=set()
    def walk(x):
        if isinstance(x,dict):
            for key,v in x.items():
                if key in ['train','dev','test'] and isinstance(v,list) and all(isinstance(a,int) for a in v):historical.update(v)
                else:walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    for p in (ROOT/'experiments').glob('v[1-7]*/results/**/locked_protocol.json'):walk(json.loads(p.read_text()))
    assert not seen.intersection(historical)
    rows=json.loads((run/'test_online.json').read_text()); count=0
    for name,group in rows.items():
        assert [r['root'] for r in group]==splits['test']
        for r in group:
            assert len(r['segments'])==3 and len(r['segment_hits'])==3
            assert r['success']==all(r['segment_hits'])
            assert r['steps']==sum(s['steps'] for s in r['segments'])<=300
            assert all(s['steps']<=100 for s in r['segments'])
            assert r['work']<=r['budget']+1e-6
            assert len(r['decisions'])==len(r['events'])
            assert all(0<=d['action']<15 and len(d['observation'])==18 for d in r['decisions'])
            if '_a5_' in name:assert all(d['action']<5 for d in r['decisions'])
            assert r['final_all_B_monitor_hit']==all(v<=.08 for v in r['final_B_monitor'])
            count+=1
    episodes=0;updates=0
    for p in (run/'models').glob('ppo_*/trajectories.jsonl'):
        for line in p.read_text().splitlines():
            record=json.loads(line);t=record['trajectory'];s=record['summary']
            assert abs(sum(r['reward'] for r in t)+s['forgetting']/.01)<1e-8
            assert s['failure']==float(not all(r['success'] for r in s['segments']))
            ret=0
            for r in reversed(t):
                ret+=r['reward'];assert abs(r['return']-ret)<1e-8
                assert r['cost_return']==s['failure']
            if t:assert t[-1]['terminal_cost']==s['failure']
            episodes+=1
        curve=json.loads((p.parent/'training.json').read_text())
        assert any(r['actor_parameter_delta']>0 for r in curve)
        updates+=len(curve)*4
    report=dict(test_rollouts=count,ppo_episodes=episodes,ppo_optimizer_steps=updates,source_and_models_verified=True,history_splits_disjoint=True)
    (run/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
