"""Account for newly executed work without recharging reused v11 trajectories."""
import hashlib,json
from pathlib import Path

def read(p):return json.loads(p.read_text())
def summed(costs):
 costs=list(costs)
 return {k:sum(c[k] for c in costs) for k in costs[0]} if costs else {}
def main():
 root=Path('runs');learning=root/'v12-learning';plan=read(Path(__file__).with_name('plan.json'))
 cal=root/'v12-calibration';sources=read(cal/'sources.json');rows=read(cal/'results.json');smoke=root/'v12-smoke'
 new=[r for k,rr in rows.items() if sources[k]=='new' for r in rr];reused=[r for k,rr in rows.items() if sources[k]!='new' for r in rr]
 smoke_batches=[read(smoke/(a+'.json'))['batch'] for a in plan['arms']]
 jobs=[learning/f'{a}_{s}' for a in plan['arms'] for s in plan['controller_seeds']]
 costs=[read(p/'cost.json') for p in jobs];prefixes=[c for p in jobs for c in read(p/'prefix_cost.json')]
 evaluation={stage:[r for p in jobs for modes in read(p/(stage+'.json')).values() for rr in modes.values() for r in rr] for stage in ['development','audit']}
 assert summed(r['cost'] for rr in evaluation.values() for r in rr)==summed(c['evaluation_cost'] for c in costs)
 stages={
  'calibration_new':{'histories':len(new),'cost':summed(r['cost'] for r in new)},
  'reuse_parity':{'histories':1,'cost':read(cal/'reuse_parity.json')['cost']},
  'smoke_prefix':{'prefixes':1,'cost':read(smoke/'prefix_cost.json')},
  'smoke_branches':{'histories':sum(len(b['results']) for b in smoke_batches),'cost':summed(r['branch_cost'] for b in smoke_batches for r in b['results'])},
  'training_prefixes':{'prefixes':len(prefixes),'cost':summed(prefixes)},
  'training_branches':{'histories':sum(c['training_branches'] for c in costs),'cost':summed(c['training_cost'] for c in costs)},
  **{stage:{'histories':len(rr),'cost':summed(r['cost'] for r in rr)} for stage,rr in evaluation.items()},
  'baselines':{'histories':read(root/'v12-baselines/cost.json')['episodes'],'cost':read(root/'v12-baselines/cost.json')['cost']}}
 # One isolated clone observation is used for coverage per model / smoke arm.
 # Its counters live on a discarded clone, so account from the frozen call sites.
 probe_count=len(jobs)+len(smoke_batches)
 probe={'copies':probe_count,'observation_examples':probe_count*(250+plan['config']['memory']//2),'updates':0,'scope':'Analytical count from one post-task-0 clone observation per job and smoke arm; copy duration not retained.'}
 weights={str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for job in jobs for p in sorted(job.glob('*.pt'))}
 assert len(weights)==36
 (learning/'model_manifest.json').write_text(json.dumps(weights,indent=2)+'\n')
 total=summed(v['cost'] for v in stages.values());total['copies']+=probe['copies'];total['observation_examples']+=probe['observation_examples']
 status=read(learning/'status.json');assert len(status)==12 and all(r['exit_code']==0 for r in status)
 result={'stages':stages,'total_new_cost':total,'total_new_full_histories':sum(v.get('histories',0) for v in stages.values()),'total_new_prefixes':sum(v.get('prefixes',0) for v in stages.values()),'reused_v11':{'histories':len(reused),'updates_not_charged':sum(r['cost']['updates'] for r in reused)},'coverage_probe_extra':probe,'coverage_policy_draws':len(jobs)*2000+len(smoke_batches)*1000,'policy_optimizer_steps_main':sum(c['policy_optimizer_steps'] for c in costs),'policy_optimizer_steps_smoke':sum(b['optimizer_steps'] for b in smoke_batches),'training_job_wall_seconds_sum':sum(c['wall_seconds'] for c in costs),'launcher_span_seconds':(learning/'status.json').stat().st_mtime-learning.stat().st_birthtime,'max_single_process_rss_bytes':max(c['max_rss_bytes'] for c in costs),'model_files':len(weights),'model_bytes':sum(v['bytes'] for v in weights.values()),'limitations':['Work counters are examples/updates, not FLOPs. Training examples include replay examples.','Copy seconds cover recorded branch clones; 15 coverage-probe clone durations were not retained.','Training job wall times overlap across three workers; summed times are not elapsed runtime or process CPU time. Baselines initially overlapped as a fourth process.','Peak RSS is maximum single-process high-water mark, not measured aggregate peak.','Unit tests and posthoc probability replay are not included in the real-data learner-update ledger; replay has zero learner updates.','Reused v11 rows retain their original counter semantics and are not charged as newly executed work.']}
 (learning/'work_ledger.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
