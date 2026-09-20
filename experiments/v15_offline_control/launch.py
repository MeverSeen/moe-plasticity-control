"""Launch a source-locked collection job under an independent deadline guard."""
import json
import os
import subprocess
import sys
from pathlib import Path
from common import HERE, V14, file_sha, save_json

root = Path('/root/autodl-tmp/moe-v15-offline-20260920')
allocation = Path('/root/autodl-tmp/moe-plasticity-v13-20260920')
protocol = HERE / 'protocol.json'
plan = json.loads(protocol.read_text())
lock = {'protocol_sha256': file_sha(protocol), 'files': {str(p): file_sha(p)
        for directory in (HERE, V14) for p in directory.glob('*.py') if not p.name.startswith('._')}}
if (root / 'source_lock.json').exists():
    raise SystemExit('Existing launch lock: inspect rather than double-start')
save_json(root / 'source_lock.json', lock)
command = [sys.executable, str(HERE / 'supervise.py'), '--deadline', str(plan['hard_deadline_epoch']),
           '--receipt', str(root / 'collection_receipt.json'), '--',
           sys.executable, '-u', str(HERE / 'collect.py'), '--root', str(root / 'data'),
           '--allocation', str(allocation), '--protocol', str(protocol)]
with (root / 'collection.log').open('ab', buffering=0) as log:
    child = subprocess.Popen(command, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                             start_new_session=True, close_fds=True)
save_json(root / 'launch.json', {'supervisor_pid': child.pid, 'command': command,
                                'hard_deadline_utc': plan['hard_deadline_utc'],
                                'protocol_sha256': lock['protocol_sha256']})
print(json.dumps({'supervisor_pid': child.pid, 'deadline': plan['hard_deadline_utc']}))
