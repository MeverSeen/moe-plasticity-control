"""Continue from collection to offline fitting and evaluation within one budget."""
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from common import run_root, HERE, V14, save_json, file_sha

# Keep the supervisor's direct child alive while the evaluator checkpoints.
# The child evaluator also receives SIGUSR1 through the same process group.
signal.signal(signal.SIGUSR1, lambda *_: None)


def main():
    outer = run_root()
    root = outer / 'data'
    allocation = Path('/root/autodl-tmp/moe-plasticity-v13-20260920')
    protocol = HERE / 'protocol.json'
    plan = json.loads(protocol.read_text())
    started = time.time()
    save_json(outer / 'offline_source_lock.json', {
        'locked_epoch': started, 'protocol_sha256': file_sha(protocol),
        'files': {str(p): file_sha(p) for d in (HERE, V14) for p in d.glob('*.py')
                  if not p.name.startswith('._')}})
    save_json(outer / 'pipeline_progress.json', {'stage': 'waiting_for_collection', 'epoch': started})
    while True:
        receipt = outer / 'collection_receipt.json'
        if receipt.exists():
            state = json.loads(receipt.read_text())
            if 'exit_code' in state:
                if state['exit_code'] != 0:
                    raise RuntimeError(f"Collection failed with exit code {state['exit_code']}")
                break
        if time.time() >= plan['hard_deadline_epoch'] - 1800:
            raise RuntimeError('Insufficient remaining authorized time after collection')
        time.sleep(10)
    commands = [
        ('offline_fit', [sys.executable, '-u', str(HERE / 'fit_dataset.py'),
                         '--root', str(root), '--protocol', str(protocol)]),
        ('closed_loop_evaluation', [sys.executable, '-u', str(HERE / 'evaluate.py'),
                         '--root', str(root), '--allocation', str(allocation), '--protocol', str(protocol)]),
        ('analysis', [sys.executable, '-u', str(HERE / 'analyze.py'),
                     '--root', str(root), '--output', str(root / 'analysis')])]
    stages = []
    for name, command in commands:
        if time.time() >= plan['hard_deadline_epoch'] - 900:
            raise RuntimeError('Predeclared archival reserve reached')
        begin = time.time()
        save_json(outer / 'pipeline_progress.json', {'stage': name, 'epoch': begin, 'stages': stages})
        with (outer / (name + '.log')).open('wb', buffering=0) as log:
            result = subprocess.run(command, stdout=log, stderr=log)
        stages.append({'stage': name, 'exit_code': result.returncode, 'wall_seconds': time.time() - begin})
        save_json(outer / 'pipeline_progress.json', {'stage': name + '_finished', 'epoch': time.time(), 'stages': stages})
        if result.returncode:
            raise RuntimeError(f'{name} failed; preserve its log and do not silently continue')
    save_json(outer / 'pipeline_summary.json', {'status': 'completed', 'stages': stages,
               'wall_seconds_including_wait': time.time() - started, 'ended_epoch': time.time()})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save_json(run_root() / 'pipeline_error.json',
                  {'error': repr(error), 'epoch': time.time()})
        raise
