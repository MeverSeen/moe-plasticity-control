"""Reconstruct behavior trajectories to audit saved training states and collection cost."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/v8_allocation_ppo'))
import sequence_env as e
import control as l

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    plan=json.loads((a.run/'plan.json').read_text());config=plan['config']
    records=[json.loads(line) for line in (a.run/'data/within_branches.jsonl').read_text().splitlines()]
    by_id={r['state_id']:r for r in records};verified=[];rows=[]
    for seed in plan['splits']['train']:
        item=e.make_item(seed,config)
        for behavior in ['middle','random']:
            env=e.SequenceEnv(item,config);rng=np.random.default_rng(seed+91)
            while not env.done:
                if env.local_steps in [0,10,30]:
                    key=f'{seed}:{behavior}:{env.stage}:{env.local_steps}'
                    np.testing.assert_array_equal(env.observe(),np.array(by_id[key]['x'],dtype=np.float32))
                    verified.append(key)
                env.act(2 if behavior=='middle' else int(rng.integers(5)),'within')
            rows.append(dict(root=seed,behavior=behavior,steps=env.steps,work=env.work))
    assert len(verified)==len(by_id)==len(records)
    l.save(a.run/'data/behavior_audit.json',dict(verified_states=len(verified),rollouts=len(rows),
        updates=sum(r['steps'] for r in rows),work=sum(r['work'] for r in rows),rows=rows,
        note='Exact deterministic reconstruction of original collection behaviors, without counterfactual branches. Audit re-execution adds this same behavior work plus repeated train-root pretraining; reported separately from main runtime.'))
    print(f'Reconstructed {len(verified)} collection states; {len(rows)} behavior trajectories')
if __name__=='__main__':main()
