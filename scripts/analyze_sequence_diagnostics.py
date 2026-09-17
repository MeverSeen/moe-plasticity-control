"""Reproduce v7 descriptive checks added after the locked primary analysis.

These checks do not train/select policies, alter the primary intervals, or draw
new test histories. Their post-run status is recorded in the output.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    root = args.run
    rows = json.loads((root/'test_online.json').read_text())
    lock = json.loads((root/'locked_protocol.json').read_text())
    training, comparisons = {}, {}
    for frequency in ['block', 'within']:
        states = [json.loads(line) for line in (root/'data'/f'{frequency}_branches.jsonl').read_text().splitlines()]
        hit = np.array([[[r['success'] for r in action] for action in s['branches']] for s in states])
        forgetting = np.array([[[r['forgetting'] for r in action] for action in s['branches']] for s in states])
        hm = hit.mean(2)
        fm = (forgetting*hit).sum(2)/np.maximum(hit.sum(2), 1)
        actions = [min(range(5), key=lambda a: (-hm[i, a], fm[i, a])) for i in range(len(states))]
        stages = {}
        for stage in range(3):
            subset = [s for s in states if s['stage'] == stage]
            stages[str(stage)] = dict(states=len(subset),
                nonzero_A_feedback=sum(abs(s['x'][-2])+abs(s['x'][-1]) > 1e-8 for s in subset))
        training[frequency] = dict(states=len(states), active_roots=len(set(s['root'] for s in states)),
            all_actions_feasible=int(hit.all((1, 2)).sum()), no_action_ever_feasible=int((~hit.any((1, 2))).sum()),
            teacher_best_action_counts=np.bincount(actions, minlength=5).tolist(), stages=stages,
            note='Descriptive training-data diagnostics; not an additional model selection criterion. Conditional mean forgetting breaks ties after branch reach.')
        field = lambda mode, key: np.array([[r[key] for r in rows[f'{frequency}_{mode}_{seed}']] for seed in lock['seeds']])
        p, f = field('progress', 'forgetting'), field('feedback', 'forgetting')
        ph, fh = field('progress', 'success'), field('feedback', 'success')
        common = (ph & fh).all(0)
        delta = (f-p).mean(0)[common]
        if not len(delta):
            comparisons[frequency] = dict(common_n=0)
            continue
        rng = np.random.default_rng(7821)
        boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(1)
        per_seed = []
        for i, seed in enumerate(lock['seeds']):
            mask = ph[i] & fh[i]
            per_seed.append(dict(seed=seed, common_n=int(mask.sum()),
                difference=float((f[i, mask]-p[i, mask]).mean()) if mask.any() else None,
                reach_difference=float(fh[i].mean()-ph[i].mean())))
        comparisons[frequency] = dict(common_n=int(common.sum()), progress_F=float(p[:, common].mean()),
            feedback_F=float(f[:, common].mean()),
            relative_reduction=float(1-f[:, common].mean()/p[:, common].mean()) if p[:, common].mean() > 0 else None,
            nominal_97_5_interval=np.quantile(boot, [.0125, .9875]).tolist(), per_seed=per_seed)
    for path, digest in lock['policy_hashes'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest() == digest
    for group in rows.values():
        for r in group:
            assert len(r['segments']) == 3 and r['success'] == all(r['segment_hits'])
            assert r['steps'] <= 300 and all(s['steps'] <= 100 for s in r['segments'])
            assert r['work'] <= r['budget']+1e-6
            assert [s['global_steps'] for s in r['segments']] == np.cumsum([s['steps'] for s in r['segments']]).tolist()
            assert all(np.isfinite(r[key]) for key in ['forgetting', 'steps', 'work'])
    save(root/'data/train_diagnostics.json', training)
    save(root/'analysis/additional_diagnostics.json', dict(comparisons=comparisons,
        note='Post-run descriptive checks, no model retuning. 97.5% per-comparison intervals are a Bonferroni sensitivity for the two primary comparisons; only root sampling, not population training-seed uncertainty.'))
    print(f'Checked {sum(map(len, rows.values()))} sequence rollouts and saved descriptive diagnostics.')


if __name__ == '__main__':
    main()
