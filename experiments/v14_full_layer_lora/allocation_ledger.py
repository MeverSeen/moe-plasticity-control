"""Account for the original allocation window; job wall time is not GPU busy time."""
import datetime,json
from pathlib import Path

def ledger(allocation,receipts,observed_epoch):
 start=datetime.datetime.fromisoformat('2026-09-20T05:25:00+00:00').timestamp();deadline=datetime.datetime.fromisoformat('2026-09-20T13:25:00+00:00').timestamp();jobs=[];intervals=[]
 for r in receipts:
  if 'started' not in r or 'ended' not in r:continue
  jobs.append({'job':r['job'],'started_utc':datetime.datetime.fromtimestamp(r['started'],datetime.timezone.utc).isoformat(),'ended_utc':datetime.datetime.fromtimestamp(r['ended'],datetime.timezone.utc).isoformat(),'wall_seconds':r['ended']-r['started'],'exit_code':r['exit_code'],'terminated':r.get('terminated',False),'killed':r.get('killed',False)})
  a=max(start,r['started']);b=min(observed_epoch,r['ended'])
  if b>a:intervals.append((a,b))
 intervals.sort();merged=[]
 for a,b in intervals:
  if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
  else:merged.append([a,b])
 busy=sum(b-a for a,b in merged);elapsed=observed_epoch-start
 return {'allocation_start_utc':'2026-09-20T05:25:00Z','hard_deadline_utc':'2026-09-20T13:25:00Z','observed_utc':datetime.datetime.fromtimestamp(observed_epoch,datetime.timezone.utc).isoformat(),'allocation_elapsed_seconds':elapsed,'remaining_authorized_seconds':max(deadline-observed_epoch,0),'supervised_job_wall_union_seconds':busy,'other_elapsed_allocation_seconds':elapsed-busy,'job_wall_is_gpu_busy_time':False,'notes':'The full elapsed allocation includes deployment, downloads, CPU preprocessing, local work and idle intervals. Job intervals include CPU work and are not GPU utilization integrals. No budget reset at v14 start.','jobs':jobs,'allocation_record':allocation}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--v13-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();receipts=[json.loads(f.read_text()) for folder in [a.v13_archive/'receipts',a.archive/'receipts'] for f in folder.glob('*.json')];status=json.loads((a.archive/'final_remote_status.json').read_text());result=ledger(json.loads((a.v13_archive/'allocation.json').read_text()),receipts,status['observed_epoch']);a.output.write_text(json.dumps(result,indent=2)+'\n')
