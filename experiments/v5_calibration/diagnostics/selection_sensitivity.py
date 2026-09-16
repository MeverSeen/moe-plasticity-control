"""Post-confirmation sensitivity audit; does not select or retrain any policy.

Added after viewing v5 confirmation: test whether its constant-oracle gain over
the locked m=.25 baseline also exceeds always freezing the protected group.
This diagnostic was NOT part of the locked confirmatory protocol.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibrate as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output file')
    lock = json.loads((args.run / 'locked_protocol.json').read_text())
    candidates = lock['candidates']
    rows = c.b.load_states(args.run / f"confirmation_{lock['selected_setting']['name']}.jsonl")
    roots, values = c.cube(rows, candidates)
    indices = c.history_choices(values, list(range(5)))
    plans = c.history_choices(values, list(range(len(candidates))))
    oracle = c.selected(values, indices, future=1)
    plan_oracle = c.selected(values, plans, future=1)
    frozen = c.selected(values, 0, future=1)
    fixed = c.selected(values, lock['selected_baselines']['best_constant'], future=1)
    all_feasible = (values['success'][:, :5, 0] > 0).all(1)
    start = {r['root']: r for r in rows if r['future'] == 0 and r['plan'] == 'fixed_0'}
    active = np.array([not start[root]['initially_done'] and start[root]['protected_mass'] > 1e-8
                       for root in roots])
    result = dict(status='Post-confirmation sensitivity, not a prespecified primary result',
        privileged_information='Oracle uses future-0 endpoint OLD TEST forgetting and B monitor feasibility. Future 1 changes minibatches, not test data or task identity. Never available to the deployed policy.',
        cohorts={})
    for name, mask in [('all', np.ones(len(roots), dtype=bool)),
                       ('all_constants_feasible', all_feasible),
                       ('all_feasible_active_protected', all_feasible & active)]:
        result['cohorts'][name] = dict(histories=int(mask.sum()),
            constant_oracle_actions=np.bincount(indices[mask], minlength=5).tolist(),
            fixed_0=c.stats({k: v[mask] for k, v in frozen.items()}),
            locked_constant=c.stats({k: v[mask] for k, v in fixed.items()}),
            constant_oracle=c.stats({k: v[mask] for k, v in oracle.items()}),
            plan_oracle=c.stats({k: v[mask] for k, v in plan_oracle.items()}),
            constant_oracle_vs_fixed_0=c.compare(oracle, frozen, mask),
            plan_oracle_vs_constant_oracle=c.compare(plan_oracle, oracle, mask))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
