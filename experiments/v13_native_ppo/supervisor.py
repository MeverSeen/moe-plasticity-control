"""Single allocation owner: durable queue, scoped process groups, absolute hard stop."""
import argparse,datetime,json,os,signal,subprocess,time
from pathlib import Path

def save(path,value):
 temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--deadline',required=True);a=p.parse_args()
 deadline=datetime.datetime.fromisoformat(a.deadline.replace('Z','+00:00')).timestamp();root=a.root
 root.mkdir(exist_ok=True,parents=True)
 for name in ['queue','logs','receipts']:(root/name).mkdir(exist_ok=True)
 lock=os.open(root/'supervisor.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(lock,str(os.getpid()).encode());os.close(lock)
 allocation={'start_utc':'2026-09-20T05:25:00Z','deadline_utc':a.deadline,'deadline_epoch':deadline,'supervisor_pid':os.getpid(),'scope':str(root),'soft_stop_seconds_before_deadline':180,'terminate_seconds_before_deadline':60,'kill_seconds_before_deadline':30}
 save(root/'allocation.json',allocation);active=None;log=None;started=0;soft=term=killed=False;task=None
 def send(sig):
  if active is not None:
   try:os.killpg(active.pid,sig)
   except ProcessLookupError:pass
 while True:
  now=time.time()
  if active is not None and active.poll() is not None:
   save(root/'receipts'/(task.stem+'.json'),{'job':task.name,'exit_code':active.returncode,'started':started,'ended':now,'soft_stop':soft,'terminated':term,'killed':killed});log.close();active=None
  if now>=deadline-180 and not soft:send(signal.SIGUSR1);soft=True
  if now>=deadline-60 and not term:send(signal.SIGTERM);term=True
  if now>=deadline-30 and not killed:send(signal.SIGKILL);killed=True
  stop=(root/'STOP').exists() or now>=deadline-180
  if active is None and not stop:
   pending=[q for q in sorted((root/'queue').glob('*.json')) if not (root/'receipts'/(q.stem+'.json')).exists()]
   if pending:
    task=pending[0];job=json.loads(task.read_text());env=os.environ.copy();env.update(job.get('env',{}));env['V13_DEADLINE_EPOCH']=str(deadline);env['V13_ALLOCATION_ROOT']=str(root)
    log=(root/'logs'/(task.stem+'.log')).open('ab',buffering=0);started=time.time();active=subprocess.Popen(job['argv'],cwd=job.get('cwd',str(root)),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(root/'supervisor_status.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remaining_seconds':deadline-now,'active_job':task.name if active else None,'active_pid':active.pid if active else None,'stopping':stop,'supervisor_pid':os.getpid()})
  if stop and active is None:break
  if now>=deadline:send(signal.SIGKILL);break
  time.sleep(1)
 save(root/'supervisor_finished.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline':a.deadline,'active_pid':active.pid if active else None})
if __name__=='__main__':main()
