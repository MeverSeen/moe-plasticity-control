"""Paired one-update reproduction under default and deterministic CUDA settings."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy
import json
import time
from pathlib import Path
import torch
from common import save_json
from environment import Config, hash_tree
from logged_environment import LoggedEnvironment, LoggedLearner

ROOT = Path('/root/autodl-tmp/moe-v15-offline-20260920')
ALLOC = Path('/root/autodl-tmp/moe-plasticity-v13-20260920')
OUT = ROOT / 'restore_diagnosis'
OUT.mkdir(exist_ok=True)
start = time.time()
state = torch.load(ROOT / 'data/prefixes/train_1411001.pt', weights_only=False)
source_hash = hash_tree(state)
data = torch.load(ALLOC / 'dataset.pt', weights_only=False)
learner = LoggedLearner(str(ALLOC / 'model'))
cfg = Config(**state['config'])
capture = {}
original_loss = learner.loss


def logged_loss(rows, role='monitor', routes=False, pad_to=None):
    result = original_loss(rows, role, routes, pad_to)
    capture.setdefault('losses', []).append({'role': role, 'batch_hash': hash_tree(rows),
                                            'losses': result[0].detach().cpu().tolist()})
    return result


learner.loss = logged_loss
original_step = learner.optim.step


def logged_step(action):
    capture['gradient_sha256'] = hash_tree([p.grad for p in learner.parameters])
    capture['gradient_sq'] = sum(float(p.grad.detach().double().square().sum()) for p in learner.parameters if p.grad is not None)
    return original_step(action)


learner.optim.step = logged_step
records = []
for mode in ['default', 'deterministic']:
    torch.use_deterministic_algorithms(mode == 'deterministic')
    for repeat in range(2):
        capture = {}
        e = LoggedEnvironment.from_state(learner, data, cfg, state['environment'], state['learner'])
        before = learner.snapshot()
        before_hash = hash_tree(before)
        expected_hash = hash_tree(state['learner'])
        try:
            ll, _ = e._measure(e.stop_rows)
            e._update(torch.ones(16, 64))
            after = learner.snapshot()
            row = {'mode': mode, 'repeat': repeat, 'before_sha256': before_hash,
                   'expected_sha256': expected_hash, 'restore_exact': before_hash == expected_hash,
                   'fresh_initial_ce': ll, 'after_weights_sha256': hash_tree(after['weights']),
                   'after_optimizer_sha256': hash_tree(after['optimizer']),
                   'monitor_ce_after': e.current_loss, 'capture': capture,
                   'input_immutable': hash_tree(state) == source_hash}
        except RuntimeError as error:
            row = {'mode': mode, 'repeat': repeat, 'error': repr(error), 'capture': capture}
        records.append(row)
        save_json(OUT / 'result.json', {'records': records, 'native_cost': learner.cost,
                  'wall_seconds': time.time() - start, 'torch': torch.__version__})
        print(json.dumps({k:v for k,v in row.items() if k not in ('capture','fresh_initial_ce')}), flush=True)
        if 'error' in row: break
