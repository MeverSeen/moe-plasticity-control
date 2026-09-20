"""Freeze source families, near-duplicate groups, and disjoint example roles."""
import hashlib,json,os,random,re,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
root=Path(os.environ['V13_ALLOCATION_ROOT']);raw=root/'raw_data';acq=json.loads((raw/'acquisition.json').read_text());tokenizer=AutoTokenizer.from_pretrained(root/'model',local_files_only=True);seed=130015
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def family(s):
 for key in ['sst','snli','amazon','yelp','glue','squad','cosmosqa','mctaco']:
  if key in s:return key
 return s
class Union:
 def __init__(self):self.parent=[]
 def add(self):i=len(self.parent);self.parent.append(i);return i
 def find(self,x):
  while self.parent[x]!=x:self.parent[x]=self.parent[self.parent[x]];x=self.parent[x]
  return x
 def merge(self,a,b):a=self.find(a);b=self.find(b);self.parent[max(a,b)]=min(a,b)
uf=Union();tasks=[];excluded=[];records=[];buckets={};exact={};rng=np.random.default_rng(seed);a=rng.integers(1,2**63,size=32,dtype=np.uint64)|1;b=rng.integers(0,2**63,size=32,dtype=np.uint64);near_pairs=0;task_uf=Union();seen_sources={}
for meta in acq['selected']:
 obj=json.loads((raw/'tasks'/meta['name']).read_text());candidates=list(obj['Instances']);random.Random(seed+int(digest(meta['name'])[:8],16)).shuffle(candidates);kept=[]
 for item in candidates:
  if not (6<=len(item['input'].split())<=220 and item.get('output') and isinstance(item['output'][0],str) and 1<=len(item['output'][0].split())<=12):continue
  p=tokenizer.encode('Instruction: '+obj['Definition'][0].strip()+'\nInput: '+item['input'].strip()+'\nOutput:',add_special_tokens=False);answer=tokenizer.encode(' '+item['output'][0].strip(),add_special_tokens=False)+[tokenizer.eos_token_id]
  if len(p)+len(answer)>512 or len(answer)>33:continue
  norm=re.sub(r'\W+',' ',item['input'].lower()).strip();words=norm.split();shingles={int.from_bytes(hashlib.blake2b(' '.join(words[i:i+5]).encode(),digest_size=8).digest(),'little') for i in range(max(1,len(words)-4))};kept.append(({'ids':p+answer,'labels':[-100]*len(p)+answer,'input_tokens':len(p+answer),'target_tokens':len(answer),'id':item['id'],'input_hash':digest(norm),'output_hash':digest(item['output'][0].strip())},shingles))
  if len(kept)>=2000:break
 if len(kept)<1200:excluded.append({'task':meta['name'],'reason':'fewer than 1200 token-length-eligible examples','eligible':len(kept)});continue
 tid=task_uf.add();tasks.append({'meta':meta,'record_ids':[]})
 for source in map(family,meta['sources']):
  if source in seen_sources:task_uf.merge(tid,seen_sources[source])
  seen_sources[source]=tid
 for item,shingles in kept:
  rid=uf.add();records.append((item,shingles,tid));tasks[-1]['record_ids'].append(rid);input_hash=item['input_hash']
  if input_hash in exact:other=exact[input_hash];uf.merge(rid,other);task_uf.merge(tid,records[other][2])
  else:exact[input_hash]=rid
  with np.errstate(over='ignore'):signature=(np.array(list(shingles),dtype=np.uint64)[:,None]*a+b).min(0)
  keys=[(band,*map(int,signature[band*4:band*4+4])) for band in range(8)];candidates=set(q for key in keys for q in buckets.get(key,[]))
  for other in candidates:
   other_set=records[other][1]
   if min(len(shingles),len(other_set))/max(len(shingles),len(other_set))<.9:continue
   if len(shingles&other_set)/len(shingles|other_set)>=.9:uf.merge(rid,other);task_uf.merge(tid,records[other][2]);near_pairs+=1
  for key in keys:buckets.setdefault(key,[]).append(rid)
 print('TOKENIZED',len(tasks),meta['name'],len(kept),flush=True)
eligible=[];seen_components=set();role_distributions={}
for tid,task in enumerate(tasks):
 component=task_uf.find(tid)
 if component in seen_components:excluded.append({'task':task['meta']['name'],'reason':'source or near-duplicate connected component already represented'});continue
 roles={'train':[],'monitor':[],'eval':[]}
 for rid in task['record_ids']:
  group=uf.find(rid);u=int(digest(str(seed)+'/'+records[group][0]['input_hash'])[:12],16)%100;role='train' if u<40 else 'monitor' if u<80 else 'eval';roles[role].append(rid)
 if any(len(roles[k])<n for k,n in [('train',384),('monitor',384),('eval',128)]):excluded.append({'task':task['meta']['name'],'reason':'insufficient disjoint role counts','counts':{k:len(v) for k,v in roles.items()}});continue
 role_distributions[tid]=roles;eligible.append(tid);seen_components.add(component)
random.Random(seed).shuffle(eligible);reserved=[t for t in eligible if any('sst' in family(s) for s in tasks[t]['meta']['sources'])];eligible=reserved+[t for t in eligible if t not in reserved]
assert len(eligible)>=32,len(eligible);ntrain=min(24,len(eligible)-16);selected=eligible[:ntrain+16];splits={'train':selected[:ntrain],'development':selected[ntrain:ntrain+8],'confirmation':selected[ntrain+8:ntrain+16]};dataset={};manifest={'revision':acq['revision'],'seed':seed,'roles':{'train':384,'monitor':384,'eval':128},'max_sequence_length':512,'max_answer_tokens_including_eos':33,'near_duplicate_rule':'normalized exact inputs plus 32-hash, 8-band/4-row MinHash candidate search; verified 5-word-shingle Jaccard>=0.90; group roles together and choose one task per connected source/near-duplicate component. Approximate candidate search is not an exhaustive proof of no near duplicates.','near_pairs':near_pairs,'eligible_components':len(eligible),'excluded':excluded,'splits':{},'tasks':{}}
for split,tids in splits.items():
 manifest['splits'][split]=[]
 for tid in tids:
  task=tasks[tid];name=task['meta']['name'].removesuffix('.json');manifest['splits'][split].append(name);roles={}
  for role,n in manifest['roles'].items():
   ids=role_distributions[tid][role];random.Random(seed+tid+len(role)*31).shuffle(ids);roles[role]=[records[r][0]|{'duplicate_group':uf.find(r)} for r in ids[:n]]
  groups=[{e['duplicate_group'] for e in rr} for rr in roles.values()];assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
  dataset[name]=roles;manifest['tasks'][name]=task['meta']|{'split':split,'role_counts':{k:len(v) for k,v in roles.items()},'role_token_counts':{k:sum(e['input_tokens'] for e in v) for k,v in roles.items()},'ids_sha256':{k:digest('\n'.join(e['id'] for e in v)) for k,v in roles.items()},'component':task_uf.find(tid)}
torch.save(dataset,root/'dataset.pt');manifest['dataset_sha256']=hashlib.sha256((root/'dataset.pt').read_bytes()).hexdigest();(root/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('DATA_READY',{k:len(v) for k,v in manifest['splits'].items()},'near pairs',near_pairs,flush=True)
