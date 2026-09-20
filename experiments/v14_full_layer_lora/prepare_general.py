"""CPU-only source/text isolation audit and locked unscored evaluation sample."""
import collections,hashlib,json,os,re,time
from pathlib import Path
import torch

def normalized(text):return re.sub(r'\W+',' ',text.lower()).strip()
def shingles(text):
 words=normalized(text).split();return {hashlib.blake2b(' '.join(words[i:i+5]).encode(),digest_size=8).digest() for i in range(max(1,len(words)-4))}
def overlaps(text,index,lengths,exact,threshold=.90):
 norm=normalized(text)
 if len(norm.split())<5:return []
 if norm in exact:return [{'record':i,'jaccard':1.} for i in exact[norm]]
 sh=shingles(text);counts=collections.Counter(i for part in sh for i in index.get(part,[]));return [{'record':i,'jaccard':count/(len(sh)+lengths[i]-count)} for i,count in counts.items() if count/(len(sh)+lengths[i]-count)>=threshold]

def main():
 from transformers import AutoTokenizer
 root=Path(os.environ['V14_WORK_ROOT']);allocation=Path(os.environ['V13_ALLOCATION_ROOT']);started=time.time();manifest=json.loads((allocation/'data_manifest.json').read_text());data=torch.load(allocation/'dataset.pt',weights_only=False);tokenizer=AutoTokenizer.from_pretrained(allocation/'model',local_files_only=True);candidates=json.loads((root/'general_candidates.json').read_text());families={s for item in manifest['tasks'].values() for s in item['sources']};candidate_families={r['source'] for r in candidates};assert not families&candidate_families
 index=collections.defaultdict(list);exact=collections.defaultdict(list);lengths=[];metadata=[];lookup={}
 for task,roles in data.items():
  raw=json.loads((allocation/'raw_data/tasks'/(task+'.json')).read_text());byid={r['id']:r for r in raw['Instances']};lookup[task]={}
  for role,rows in roles.items():
   for encoded in rows:
    rawrow=byid[encoded['id']];lookup[task][encoded['id']]=rawrow['output'];rid=len(lengths);norm=normalized(rawrow['input']);sh=shingles(norm);lengths.append(len(sh));metadata.append({'task':task,'role':role,'id':encoded['id']});exact[norm].append(rid)
    for item in sh:index[item].append(rid)
 print('INDEX',len(metadata),'NI examples',len(index),'shingles',flush=True)
 groups=collections.Counter();selected=[];selected_meta=[];excluded=[];audit_queries=0
 for r in candidates:
  quota=4 if r['source']=='mmlu' else 128
  if groups[r['group']]>=quota:continue
  ids=tokenizer.encode(r['prompt'],add_special_tokens=False)
  if len(ids)>512:excluded.append({'id':r['id'],'reason':'prompt_exceeds512','tokens':len(ids)});continue
  choice_ids=[tokenizer.encode(' '+letter,add_special_tokens=False) for letter in r['choice_letters']];assert all(len(x)==1 for x in choice_ids)
  hits=[]
  for text in r['source_texts']:
   audit_queries+=1;hits.extend(overlaps(text,index,lengths,exact))
  if hits:
   excluded.append({'id':r['id'],'reason':'NI_text_overlap','matches':[metadata[h['record']]|{'jaccard':h['jaccard']} for h in hits]});continue
  selected.append({k:r[k] for k in ['id','source','ability','group','answer_index']}|{'prompt_ids':ids,'choice_token_ids':[x[0] for x in choice_ids]});selected_meta.append({k:r[k] for k in ['id','source','ability','group','answer_index']}|{'prompt_sha256':hashlib.sha256(r['prompt'].encode()).hexdigest(),'input_tokens':len(ids)});groups[r['group']]+=1
 expected={r['group']:4 if r['source']=='mmlu' else 128 for r in candidates};assert dict(groups)==expected,(dict(groups),expected)
 encoded_sha=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest();out={'status':'passed','usage':'independent_evaluation_only','created_epoch':time.time(),'source_family_disjoint':True,'source_family_scope':'Declared benchmark dataset identities versus all40 selected NI source identities; shared web origins and pretraining contamination are not ruled out.','ni_source_families':sorted(families),'general_source_families':sorted(candidate_families),'ni_examples_audited':len(metadata),'ni_role_counts':dict(collections.Counter(x['role'] for x in metadata)),'ni_manifest_dataset_sha256':manifest['dataset_sha256'],'selection':'Before any model score: SHA256(1414/id) order within group, first4 eligible per MMLU subject and first128 per other benchmark; prompt<=512 and no declared text overlap. No answer/performance screening.','text_check':'Normalized exact text plus exhaustive inverted-index5-word-shingle Jaccard>=.90 over declared question/full-input/passage fields vs selected NI raw inputs. Fields<5 words excluded from short-fragment check. Does not establish semantic independence or unknown-source absence.','audit_queries':audit_queries,'excluded':excluded,'selected_counts':dict(groups),'ability_counts':dict(collections.Counter(x['ability'] for x in selected)),'encoded_examples_sha256':encoded_sha,'items':selected_meta,'source_download_manifest':json.loads((root/'general_download_manifest.json').read_text()),'wall_seconds':time.time()-started,'pretraining_contamination':'unknown; no claim of unseen pretraining data','scoring':'Fixed zero-shot option-letter conditional likelihood; no CoT; limited subsets, not full benchmark or TRACE replication','model_scores_queried_during_audit':False}
 (root/'general_examples.json').write_text(json.dumps(selected)+'\n');(root/'general_manifest.json').write_text(json.dumps(out,indent=2)+'\n');torch.save(lookup,root/'answer_lookup.pt');print('GENERAL_READY',out['ability_counts'],len(excluded),out['wall_seconds'],flush=True)
if __name__=='__main__':main()
