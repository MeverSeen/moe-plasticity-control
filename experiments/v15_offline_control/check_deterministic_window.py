"""Repeat the exact failing five-update window with deterministic CUDA enabled."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import time
from pathlib import Path
import torch
from common import save_json
from environment import Config, hash_tree
from logged_environment import LoggedLearner, LoggedEnvironment

torch.use_deterministic_algorithms(True)
root = Path('/root/autodl-tmp/moe-v15-offline-20260920')
alloc = Path('/root/autodl-tmp/moe-plasticity-v13-20260920')
state = torch.load(root / 'data/prefixes/train_1411001.pt', weights_only=False)
data = torch.load(alloc / 'dataset.pt', weights_only=False)
learner = LoggedLearner(str(alloc / 'model'))
records = []
started = time.time()
for i in range(2):
    e = LoggedEnvironment.from_state(learner, data, Config(**state['config']), state['environment'], state['learner'])
    before = learner.snapshot()
    assert hash_tree(before) == hash_tree(state['learner'])
    losses = []
    for _ in range(5):
        e._update(torch.ones(16, 64)); losses.append(e.current_loss)
    after = learner.snapshot()
    records.append({'losses': losses, 'weights': hash_tree(after['weights']), 'optimizer': hash_tree(after['optimizer'])})
    print(json.dumps(records[-1]), flush=True)
equal = records[0] == records[1]
save_json(root / 'restore_diagnosis/deterministic_window.json', {'equal': equal, 'records': records,
          'native_cost': learner.cost, 'wall_seconds': time.time() - started,
          'scope': 'one previously failing saved root; full workflow also requires its own checks'})
assert equal, 'Deterministic complete window still diverges'
