"""Development-only learning pressure and module intervention experiment.

No controller is fitted here and no official test images are loaded. All cells,
thresholds and roots are written to the protocol before any rollout starts.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from pressure_env import PressureEnv, PREFIX_TARGET, run_prefix, run_suffix
from data import CIFAR100_MD5, class_partition, load_dataset
from environment import DEFAULT, evaluate_endpoints
from run_retention import history_for

ROOT = Path(__file__).resolve().parents[2]
METHODS = ['global_0', 'global_0.5', 'global_1', 'group_rule', 'local_rule', 'coordinated_rule']
TARGETS = [1.55, 1.45, 1.35, 1.25]


def cells():
    result = [dict(mode='early_stop', scope='full', method=m, target_loss=t)
              for t in TARGETS for m in METHODS]
    mechanisms = [('full','global_0'), ('full','global_1'),
                  ('frozen_stem','global_1'), ('heads_only','global_0')]
    result += [dict(mode='early_stop', scope=s, method=m, target_loss=1.35)
               for s, m in mechanisms[2:]]
    result += [dict(mode='fixed_budget', scope=s, method=m, target_loss=1.35)
               for s, m in mechanisms]
    return result


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def source_hashes():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT/'experiments/v8_multilayer_retention').glob('*.py'))
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def fingerprint(env):
    """Fingerprint model, Adam, retained samples/references and the future RNG."""
    digest = hashlib.sha256()
    def update(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((str(value.dtype), list(value.shape))).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            for part in value:
                update(part)
        else:
            digest.update(repr(value).encode())
    update(env.model.state_dict())
    update(env.optimizer.state_dict())
    update(env.rng.get_state())
    update(env.memory.retained_replay_ids)
    update(env.memory.retained_monitor_ids)
    update(env.memory.monitor()[1])
    update([env.stage, env.local_steps, env.current_loss, env.last, env.completed])
    return digest.hexdigest()


def sum_costs(costs):
    keys = set().union(*(c.keys() for c in costs)) if costs else set()
    return {key:sum(c.get(key, 0) for c in costs) for key in sorted(keys)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache/datasets')
    parser.add_argument('--pilot', action='store_true', help='Use 2 rather than 6 new development roots.')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('choose a fresh output directory')
    archive = ROOT/'experiments'
    if archive in args.out.resolve().parents:
        parser.error('run outside the archived experiments directories')
    torch.set_num_threads(1)
    args.out.mkdir(parents=True)
    out = args.out
    config = dict(DEFAULT, target_loss=PREFIX_TARGET)
    # These were never used by v8; their classes are explicitly development
    # classes, not new independent category draws or an unseen test claim.
    roots = list(range(88410, 88412 if args.pilot else 88416))
    expected_cells = cells()
    protocol = dict(status='development_diagnostic', date='2026-10-02',
        config=config, roots=roots, expected_cells=expected_cells,
        targets=TARGETS, prefix_target=PREFIX_TARGET,
        class_pool=class_partition()['dev'], dataset=dict(name='CIFAR-100', loaded_splits=['train'],
            archive_md5=CIFAR100_MD5, official_test_loaded=False),
        source_hashes=source_hashes(),
        questions=['Do stricter targets make expert displacement valuable within this budget?',
                   'Does adaptation reside in the CNN stem, experts, or task heads?',
                   'Do fixed-budget train/monitor/heldout curves indicate generalization limits?'],
        prefix='One common task-0 run per root: global_1, CE 1.55 or 80 updates, full model/Adam/memory/RNG cloned for every cell.',
        stopping='Every target cell is an independent true suffix rollout. Early stop checks monitor every 5 updates. Fixed budget uses exactly 80 updates per suffix task and only runs at CE 1.35; it is not truncated to reconstruct early stopping.',
        scopes={'full':'stem and heads update normally; expert Adam displacement follows action',
                'frozen_stem':'after task0 stem weights and Adam moments frozen; experts follow action and heads train',
                'heads_only':'after task0 stem and expert weights/moments frozen; only heads train'},
        diagnostics='After each window measure current train and disjoint heldout dev CE; never feed heldout metrics to controller, reward or stopping. Full trajectories are for diagnosis, not heldout-min checkpoint selection.',
        data_limits='6 roots reuse v8 dev 25-class pool. Official training file only; v8 audit-class/test outcomes never used here. All findings are development evidence, not confirmatory inference.',
        statistics='Failure, prefix feasibility, monitor-ever/terminal and heldout-diagonal reach separate; mean positive old loss changes use each task own fixed diagonal. Pair roots descriptively; no non-inferiority claim.',
        costs='Same learner-update cap, not equal FLOPs. Raw rows include logical prefix for fair per-policy cost; physical total counts common prefix once and every suffix/evaluation/clone separately. Diagnostic example passes/time are separate fields inside measured wall time.')
    save(out/'locked_protocol.json', protocol)
    save(out/'protocol.json', protocol)
    save(out/'environment.json', dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                     platform=platform.platform(), threads=1))
    start = time.perf_counter()
    images, labels = load_dataset(args.cache, 'train', download=args.download)
    rows, prefixes, branch_costs = [], [], []
    clone_seconds = 0.
    for root in roots:
        history = history_for(images, labels, protocol['class_pool'], root, config)
        prefix = run_prefix(PressureEnv(history, root, config, capture=True))
        prefix_hash = fingerprint(prefix)
        prefixes.append(dict(root=root, fingerprint=prefix_hash,
            task=prefix.completed[0], cost=copy.deepcopy(prefix.cost),
            classes=[t['classes'] for t in history]))
        save(out/'prefixes.json', prefixes)
        for i, cell in enumerate(expected_cells):
            copy_start = time.perf_counter()
            env = prefix.clone()
            clone_seconds += time.perf_counter()-copy_start
            run_suffix(env, cell['method'], cell['target_loss'], cell['mode'], cell['scope'])
            row = evaluate_endpoints(env, [t['eval'] for t in history], 'development', cell['method'])
            row.update(cell, prefix_target=PREFIX_TARGET, prefix_fingerprint=prefix_hash)
            # Existing endpoint evaluations are offline and new for each cell.
            # Subtract only the reused prefix from other counters.
            branch_cost = {k:v-prefix.cost.get(k, 0) for k,v in row['cost'].items()}
            assert all(v >= -1e-8 for v in branch_cost.values()), branch_cost
            branch_costs.append(branch_cost)
            rows.append(row)
            save(out/'results.json', rows)
            suffix = row['tasks'][1:]
            print(f'root={root} cell={i+1}/{len(expected_cells)} {cell["mode"]} {cell["scope"]} {cell["method"]} target={cell["target_loss"]} '
                  f'ever={sum(t["ever_hit"] for t in suffix)}/4 terminal={sum(t["success"] for t in suffix)}/4 '
                  f'updates={sum(t["steps"] for t in suffix)}', flush=True)
        if fingerprint(prefix) != prefix_hash:
            raise RuntimeError('a suffix mutated its shared prefix')
    if source_hashes() != protocol['source_hashes']:
        raise RuntimeError('source changed during locked development run')
    prefix_cost = sum_costs([p['cost'] for p in prefixes])
    suffix_cost = sum_costs(branch_costs)
    save(out/'total_cost.json', dict(common_prefix=prefix_cost,
        suffixes_and_endpoint_evaluation=suffix_cost,
        total_without_reused_prefix=sum_costs([prefix_cost, suffix_cost]),
        clone_seconds=clone_seconds, wall_seconds=time.perf_counter()-start,
        rows=len(rows), note='diagnostic_seconds is already inside rollout seconds, do not add it again; endpoint evaluation seconds are separate'))
    print(f'Completed {len(rows)} development rollouts: {out}', flush=True)


if __name__ == '__main__':
    main()
