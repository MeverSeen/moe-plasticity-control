"""Bounded pinned task acquisition, independent of any policy performance."""
import concurrent.futures,hashlib,json,random,re,time,urllib.request
from pathlib import Path
tree=json.loads(Path('/tmp/v13-ni-tree.json').read_text());out=Path('runs/v13-data');out.mkdir(exist_ok=True);(out/'tasks').mkdir(exist_ok=True)
allowed={'Sentiment Analysis','Text Categorization','Textual Entailment','Question Answering','Information Extraction','Named Entity Recognition','Coreference Resolution','Paraphrase Identification','Dialogue Act Recognition','Entity Relation Classification','Cause Effect Classification'}
files=[p for p in tree['tree'] if p['path'].startswith('tasks/') and p['path'].endswith('.json') and 150000<p.get('size',0)<12000000 and any(s in p['path'] for s in ['classification','extraction','answer_generation','question_answering']) and not any(s in p['path'] for s in ['wrong','incorrect','translation','language_identification','check_prime'])];random.Random(130013).shuffle(files)
selected=[];excluded=[];used=set()
def fetch(item):
 name=Path(item['path']).name
 try:
  with urllib.request.urlopen('https://raw.githubusercontent.com/allenai/natural-instructions/'+tree['revision']+'/'+item['path'],timeout=25) as r:data=r.read()
  assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['sha'];v=json.loads(data)
  if any(v.get(k)!=['English'] for k in ['Input_language','Output_language','Instruction_language']):return name,'language',None
  if not (set(v.get('Categories',[]))&allowed):return name,'category',None
  sources={re.sub('[^a-z0-9]','',s.lower()) for s in v.get('Source',[])}
  if not sources:return name,'missing source',None
  valid=[x for x in v['Instances'] if 6<=len(x['input'].split())<=220 and x.get('output') and isinstance(x['output'][0],str) and 1<=len(x['output'][0].split())<=12]
  if len(valid)<1400:return name,'insufficient short examples',None
  meta={'name':name,'sources':sorted(sources),'original_sources':v['Source'],'categories':v['Categories'],'definition':v['Definition'][0],'raw_instances':len(v['Instances']),'short_candidates':len(valid),'blob_sha1':item['sha'],'sha256':hashlib.sha256(data).hexdigest(),'license':v.get('Instance License',[])}
  return name,meta,data
 except Exception as e:return name,type(e).__name__,None
for offset in range(0,min(len(files),280),12):
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(fetch,files[offset:offset+12]))
 for name,meta,data in results:
  if data is None:excluded.append({'name':name,'reason':meta});continue
  if set(meta['sources'])&used:excluded.append({'name':name,'reason':'source already represented'});continue
  (out/'tasks'/name).write_bytes(data);selected.append(meta);used.update(meta['sources']);print('ACCEPT',len(selected),name,meta['short_candidates'],flush=True)
 (out/'acquisition.json').write_text(json.dumps({'revision':tree['revision'],'seed':130013,'selection_rule':'Filename/type/English/source disjointness and >=1400 raw short examples only; no policy results consulted. Final tokenizer and near-duplicate grouping still required.','selected':selected,'excluded':excluded},indent=2)+'\n')
 if len(selected)>=60:break
print('COMPLETE',len(selected),flush=True)
