import pathlib,json,hashlib,tarfile,time,subprocess
alloc=pathlib.Path('/root/autodl-tmp/moe-plasticity-v13-20260920');root=alloc/'v14_methods_20260920_0814';receipt=alloc/'receipts/030-v14-methods-ppo.json';assert receipt.exists(),'Wait for completed supervisor receipt'
lock=json.loads((root/'source_lock.json').read_text())
for name,expected in lock['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
status={'observed_epoch':time.time(),'supervisor':json.loads((alloc/'supervisor_status.json').read_text()),'pending_jobs':[p.name for p in (alloc/'queue').glob('*.json') if not (alloc/'receipts'/p.name).exists()],'gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),'instance_shutdown':False,'original_hard_deadline_utc':'2026-09-20T13:25:00Z'}
(root/'final_remote_status.json').write_text(json.dumps(status,indent=2)+'\n')
output=root/'completed-methods-archive.tar.gz'
with tarfile.open(output,'w:gz') as tar:
 for name in lock['sha256']:tar.add(root/name,arcname='provenance/'+name)
 for name in ['method_protocol.json','protocol.json','source_lock.json','data_manifest.json','progress.json','stage_costs.json','learning.json','training_status.json','training_status_final.json','fixed_lock.json','confirmation_lock.json','summary.json','final_remote_status.json']:
  if (root/name).exists():tar.add(root/name,arcname=name)
 for name in ['results','controllers','prefixes','targets']:
  if (root/name).exists():tar.add(root/name,arcname=name)
 for name in ['initial_general.json','initial_adapter.json','targets.json','target_reference.json','prefix.json','uniform1.json','static050.json']:
  tar.add(alloc/'v14/results/short_comparison'/name,arcname='reused_pilot/'+name)
 tar.add(alloc/'v14/general_manifest.json',arcname='general_manifest.json');tar.add(alloc/'allocation.json',arcname='allocation.json');tar.add(receipt,arcname='receipt.json');tar.add(alloc/'logs/030-v14-methods-ppo.log',arcname='workflow.log')
print(json.dumps({'path':str(output),'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'final_status':status}))
