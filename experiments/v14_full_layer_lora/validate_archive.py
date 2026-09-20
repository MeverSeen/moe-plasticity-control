"""Replay actual rule factors and audit costs, evaluation matrices and source locks."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'archive/provenance/comparison-executed'))
from structured_policy import Rule,shaped_rewards

def read(path):return json.loads(path.read_text())
def close(a,b):assert abs(a-b)<=1e-6*max(1.,abs(a),abs(b)),(a,b)
def main():
 archive=ROOT/'archive';folder=archive/'results/short_comparison';summary=read(folder/'summary.json');prefix=read(folder/'prefix.json');reference=read(folder/'target_reference.json');initial=read(folder/'initial_general.json');general_manifest=read(archive/'general_manifest.json');smoke=read(ROOT/'smoke_result.json');source=read(archive/'comparison_source_lock.json');decisions=0;endpoints=[];general=[]
 assert summary['status']=='completed' and smoke['status']=='completed';assert smoke['trainable_parameters']==37748736 and smoke['groups']==1024 and smoke['projections']==3072;assert smoke['frozen_base_bitwise_unchanged'] and smoke['initial_logits_bitwise_equal'];assert smoke['checkpoint']['weights_exact'] and smoke['checkpoint']['optimizer_moments_exact'] and smoke['checkpoint']['cpu_cuda_rng_exact'];assert summary['frozen_base_hash']==smoke['base_hash'];assert general_manifest['status']=='passed' and general_manifest['ni_examples_audited']==35840 and sum(general_manifest['ability_counts'].values())==612
 assert initial['adapter_enabled'] and len(initial['items'])==612
 for ability,row in initial['abilities'].items():
  items=[x for x in initial['items'] if x['ability']==ability];assert row['n']==len(items);close(np.mean([x['correct'] for x in items]),row['accuracy']);close(np.mean([x['answer_nll'] for x in items]),row['answer_nll'])
  for item in items:assert item['predicted_index']==int(np.argmax(item['choice_log_probabilities']));assert item['correct']==(item['predicted_index']==item['answer_index'])
 for name,sha in source['sha256'].items():assert hashlib.sha256((archive/'provenance/comparison-executed'/name).read_bytes()).hexdigest()==sha,name
 for name in ['uniform1','static050','rule']:
  r=read(folder/(name+'.json'));endpoints.append(r);assert r['history_completed'] and r['phase']=='development_short';assert r['checkpoint']['prefix_sha256']==prefix['sha256'];assert r['comparison_plan_sha256']==hashlib.sha256((ROOT/'comparison_plan.json').read_bytes()).hexdigest();assert r['n_fail']==sum(t['success'] is False for t in r['tasks']);assert r['history_success']==all(t['success'] for t in r['tasks']);assert r['cost_path']['updates']==sum(t['steps'] for t in r['tasks']);assert r['cost_actual']['updates']+prefix['cost']['updates']==r['cost_path']['updates'];assert sum(t['duration'] for t in r['trace'])==sum(t['steps'] for t in r['tasks'][1:])
  for key,v in prefix['cost'].items():close(r['cost_path'][key],r['cost_actual'][key]+v)
  m=r['evaluation']['matrix'];s=r['evaluation']['score_matrix'];assert all(m[k][j] is not None and s[k][j] is not None for k in range(3) for j in range(k+1));close(np.mean([max(m[-1][j]-m[j][j],0) for j in range(2)]),r['evaluation']['forgetting']);close(np.mean([m[-1][j]-m[j][j] for j in range(2)]),r['evaluation']['signed_forgetting']);close(np.mean(m[-1]),r['evaluation']['final_loss'])
  for cell,score in r['evaluation']['score_details'].items():
   k,j=map(int,cell.split('/'));assert score['n']==len(score['items'])==128;close(np.mean([x['exact_match'] for x in score['items']]),s[k][j]);close(np.mean([x['token_f1'] for x in score['items']]),score['token_f1']);assert all(x['generated_tokens']<=32 for x in score['items'])
  rule=Rule(r['candidate']['base_indices'],mode=r['candidate']['mode'])
  for t in r['trace']:
   obs=tuple(torch.tensor(v) for v in t['observation']);latent=[torch.tensor(v) for v in t['latent']];a,z,lp,entropy,value=rule.act(obs,latent);assert torch.equal(a,torch.tensor(t['experts']));close(float(lp),t['policy_logp']);assert a.shape==(16,64);assert 0<t['duration']<=5;assert t['route_observation']['current_cache_age_updates']==0 and t['route_observation']['old_cache_age_updates']==0;assert len(t['route_observation']['current_tokens_each_layer'])==16;decisions+=1
  rewards=shaped_rewards([t['potential'] for t in r['trace']],r['monitor']['forgetting'],r['n_fail'],r['reward']['scale'],r['reward']['kappa']);assert np.allclose(rewards,[t['reward'] for t in r['trace']],atol=1e-9);close(sum(rewards),r['reward']['shaped_return']);assert abs(r['reward']['telescope_error'])<1e-8
  g=read(folder/(name+'__general.json'));general.append(g);assert g['adapter_enabled'] and g['manifest_sha256']==initial['manifest_sha256'];assert len(g['items'])==612
  for ability,row in g['abilities'].items():
   items=[x for x in g['items'] if x['ability']==ability];assert row['ids']==initial['abilities'][ability]['ids'];assert row['n']==len(items);close(np.mean([x['correct'] for x in items]),row['accuracy']);close(np.mean([x['answer_nll'] for x in items]),row['answer_nll']);close(row['accuracy']-initial['abilities'][ability]['accuracy'],r['general_ability'][ability]['accuracy_delta'])
   for item in items:assert item['predicted_index']==int(np.argmax(item['choice_log_probabilities']));assert item['correct']==(item['predicted_index']==item['answer_index'])
 cost=summary['native_cost']
 for key,value in cost.items():
  expected=reference['cost_actual'][key]+prefix['cost'][key]+prefix['snapshot_cost'][key]+sum(r['cost_actual'][key] for r in endpoints)
  mapping={'general_eval_input_tokens':'input_tokens','general_eval_padded_tokens':'padded_tokens','general_eval_target_tokens':'target_tokens_scored','general_eval_seconds':'wall_seconds'}
  if key in mapping:expected+=sum(g['cost'][mapping[key]] for g in [initial]+general)
  close(value,expected)
 files=0
 if (ROOT/'archive_manifest.json').exists():
  for name,sha in read(ROOT/'archive_manifest.json')['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name;files+=1
 print(json.dumps({'status':'passed','completed_candidates':3,'rule_decisions_replayed':decisions,'general_scores_recomputed':4*612,'matrix_cells':3*6,'comparison_actual_updates':cost['updates'],'smoke_actual_updates':smoke['native_cost']['updates'],'manifest_files':files}))
if __name__=='__main__':main()
