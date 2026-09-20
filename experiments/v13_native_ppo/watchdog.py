"""Independent last-resort deadline guard for registered allocation process groups."""
import json,os,signal,time
from pathlib import Path
root=Path('/root/autodl-tmp/moe-plasticity-v13-20260920');allocation=json.loads((root/'allocation.json').read_text());tracked={}
def identity(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
  return int(fields[1]),int(fields[2]),int(fields[19]) # ppid, pgrp, start ticks
 except (FileNotFoundError,ProcessLookupError):return None
while time.time()<allocation['deadline_epoch']-30:
 try:
  status=json.loads((root/'supervisor_status.json').read_text());pid=status.get('active_pid')
  if pid:
   state=identity(pid)
   if state and state[:2]==(allocation['supervisor_pid'],pid):tracked[pid]=state[2]
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(2)
(root/'STOP').touch();killed=[]
for pid,ticks in tracked.items():
 state=identity(pid)
 if state and state[1:]==(pid,ticks):
  try:os.killpg(pid,signal.SIGKILL);killed.append(pid)
  except ProcessLookupError:pass
(root/'watchdog_finished.json').write_text(json.dumps({'epoch':time.time(),'groups_killed':killed,'tracked_groups':len(tracked)},indent=2)+'\n')
