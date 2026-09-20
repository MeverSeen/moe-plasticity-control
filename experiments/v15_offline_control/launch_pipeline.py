import json
import subprocess
import sys
from pathlib import Path
from common import HERE, save_json

root = Path('/root/autodl-tmp/moe-v15-offline-20260920')
if (root / 'pipeline_launch.json').exists():
    raise SystemExit('Pipeline already launched; inspect before retrying')
plan = json.loads((HERE / 'protocol.json').read_text())
command = [sys.executable, str(HERE / 'supervise.py'), '--deadline', str(plan['hard_deadline_epoch']),
           '--receipt', str(root / 'pipeline_receipt.json'), '--',
           sys.executable, '-u', str(HERE / 'pipeline.py')]
with (root / 'pipeline.log').open('ab', buffering=0) as log:
    child = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                             start_new_session=True, close_fds=True)
save_json(root / 'pipeline_launch.json', {'supervisor_pid': child.pid, 'command': command,
                                        'hard_deadline_utc': plan['hard_deadline_utc']})
print(json.dumps({'pipeline_supervisor_pid': child.pid}))
