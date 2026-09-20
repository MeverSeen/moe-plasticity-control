"""Create deterministic unscored candidate lists from downloaded official sources."""
import csv,hashlib,io,json,re,tarfile
from pathlib import Path
import pyarrow.parquet as pq
root=Path('runs/v14-general-raw');rows=[]
def add(source,ability,group,id_,text,options,answer,source_texts):
 if options is None:prompt='Choose the correct option. Answer with its letter.\n'+text+'\nAnswer:';n=5
 else:prompt='Choose the correct option. Answer with its letter.\n'+text+'\n'+'\n'.join(chr(65+i)+'. '+v for i,v in enumerate(options))+'\nAnswer:';n=len(options)
 rows.append({'source':source,'ability':ability,'group':group,'id':id_,'prompt':prompt,'choice_letters':[chr(65+i) for i in range(n)],'answer_index':answer,'source_texts':source_texts})
with tarfile.open(root/'mmlu.tar') as tar:
 for member in sorted(tar.getmembers(),key=lambda m:m.name):
  if '/test/' not in member.name or not member.name.endswith('.csv'):continue
  subject=Path(member.name).stem.removesuffix('_test')
  for i,r in enumerate(csv.reader(io.TextIOWrapper(tar.extractfile(member),encoding='utf-8'))):add('mmlu','knowledge_mmlu_subset',subject,f'mmlu/{subject}/{i}',r[0],r[1:5],'ABCD'.index(r[5]),[r[0],r[0]+' '+' '.join(r[1:5])])
labels=[int(s) for s in (root/'piqa-labels.lst').read_text().splitlines()]
for i,line in enumerate((root/'piqa-valid.jsonl').read_text().splitlines()):
 r=json.loads(line);add('piqa','commonsense_piqa_subset','piqa',f'piqa/validation/{i}',r['goal'],[r['sol1'],r['sol2']],labels[i],[r['goal'],r['goal']+' '+r['sol1']+' '+r['sol2']])
for i,r in enumerate(pq.read_table(root/'boolq-validation.parquet').to_pylist()):add('boolq','reading_boolq_subset','boolq',f'boolq/validation/{i}','Passage: '+r['passage']+'\nQuestion: '+r['question'],['yes','no'],0 if r['answer'] else 1,[r['passage'],r['question']+' '+r['passage']])
for i,r in enumerate(json.loads((root/'bbh-logical-five.json').read_text())['examples']):add('bbh','reasoning_bbh_logical5_subset','bbh_logical5',f'bbh/logical_deduction_five_objects/{i}',r['input'],None,'ABCDE'.index(r['target'].strip('()')),[r['input']])
rows.sort(key=lambda r:(r['group'],hashlib.sha256(('1414/'+r['id']).encode()).hexdigest()))
(root/'candidates.json').write_text(json.dumps(rows)+'\n');print('Unscored candidates',len(rows),'bytes',(root/'candidates.json').stat().st_size)
