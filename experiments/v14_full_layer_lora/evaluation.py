"""Read-only deployment evaluation. This module never emits training rewards."""
import hashlib,json,re,string,time
import numpy as np
import torch

def normalize_answer(text):
 text=text.lower();text=''.join(c for c in text if c not in string.punctuation);text=re.sub(r'\b(a|an|the)\b',' ',text);return ' '.join(text.split())
def token_f1(prediction,reference):
 from collections import Counter
 p=normalize_answer(prediction).split();r=normalize_answer(reference).split()
 if not p or not r:return float(p==r)
 common=sum((Counter(p)&Counter(r)).values())
 return 0. if not common else 2*common/(len(p)+len(r))
def compare_general(initial,final):
 assert initial['manifest_sha256']==final['manifest_sha256'];result={}
 for ability,before in initial['abilities'].items():
  after=final['abilities'][ability];assert before['ids']==after['ids'];result[ability]={'n':before['n'],'initial_accuracy':before['accuracy'],'final_accuracy':after['accuracy'],'accuracy_delta':after['accuracy']-before['accuracy'],'initial_answer_nll':before['answer_nll'],'final_answer_nll':after['answer_nll'],'answer_nll_delta':after['answer_nll']-before['answer_nll'],'initial_correct':before['correct'],'final_correct':after['correct'],'changed_correctness':sum(a!=b for a,b in zip(before['correct'],after['correct']))}
 return result

class GeneralEvaluator:
 def __init__(self,artifact,manifest):
  self.examples=artifact;self.manifest=manifest;assert manifest['status']=='passed' and manifest['source_family_disjoint'];assert manifest['usage']=='independent_evaluation_only';self.sha=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
  expected=manifest['encoded_examples_sha256'];actual=hashlib.sha256(json.dumps(artifact,sort_keys=True).encode()).hexdigest();assert actual==expected
 @torch.no_grad()
 def run(self,learner,batch_size=8):
  assert all(a.enabled for a in learner.adapters),'Evaluate the actual adapter-enabled deployment model';learner.model.eval();start=time.perf_counter();by={};input_tokens=0;padded=0;records=[]
  for offset in range(0,len(self.examples),batch_size):
   rows=self.examples[offset:offset+batch_size];packed=[{'ids':r['prompt_ids'],'labels':[-100]*len(r['prompt_ids'])} for r in rows];ids,labels,mask=learner.pack(packed);input_tokens+=int(mask.sum());padded+=ids.numel()
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=learner.model(input_ids=ids,attention_mask=mask,use_cache=False).logits
   indices=mask.sum(1)-1;last=logits[torch.arange(len(rows),device=ids.device),indices].float().log_softmax(-1)
   for i,r in enumerate(rows):
    scores=last[i,r['choice_token_ids']];pred=int(scores.argmax());answer=r['answer_index'];item={'id':r['id'],'ability':r['ability'],'source':r['source'],'correct':pred==answer,'predicted_index':pred,'answer_index':answer,'answer_nll':float(-last[i,r['choice_token_ids'][answer]]),'choice_normalized_nll':float(-scores.log_softmax(-1)[answer]),'choice_log_probabilities':scores.cpu().tolist()};records.append(item);by.setdefault(r['ability'],[]).append(item)
  torch.cuda.synchronize();wall=time.perf_counter()-start;learner.cost['general_eval_input_tokens']+=input_tokens;learner.cost['general_eval_padded_tokens']+=padded;learner.cost['general_eval_target_tokens']+=len(records);learner.cost['general_eval_seconds']+=wall;abilities={}
  for ability,items in by.items():abilities[ability]={'n':len(items),'accuracy':float(np.mean([x['correct'] for x in items])),'answer_nll':float(np.mean([x['answer_nll'] for x in items])),'ids':[x['id'] for x in items],'correct':[x['correct'] for x in items]}
  return {'manifest_sha256':self.sha,'adapter_enabled':True,'usage':'independent_evaluation_only','abilities':abilities,'items':records,'cost':{'input_tokens':input_tokens,'padded_tokens':padded,'target_tokens_scored':len(records),'wall_seconds':wall},'scoring':'Zero-shot fixed prompt, conditional log probability of a single option-letter token; no answer generation or CoT, not an official full benchmark/TRACE reproduction.'}

@torch.no_grad()
def task_generation_score(learner,examples,answer_lookup,batch_size=8,max_new_tokens=32):
 assert all(a.enabled for a in learner.adapters);learner.model.eval();start=time.perf_counter();items=[];input_tokens=0;padded_tokens=0;generated_tokens=0;decode_step_tokens=0
 for offset in range(0,len(examples),batch_size):
  rows=examples[offset:offset+batch_size];prompts=[[token for token,label in zip(r['ids'],r['labels']) if label==-100] for r in rows];width=max(map(len,prompts));ids=torch.full((len(rows),width),learner.tokenizer.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
  for i,prompt in enumerate(prompts):ids[i,width-len(prompt):]=torch.tensor(prompt,device='cuda');mask[i,width-len(prompt):]=1
  with torch.autocast('cuda',dtype=torch.bfloat16):sequences=learner.model.generate(input_ids=ids,attention_mask=mask,do_sample=False,max_new_tokens=max_new_tokens,use_cache=True,pad_token_id=learner.tokenizer.pad_token_id,eos_token_id=learner.tokenizer.eos_token_id)
  input_tokens+=int(mask.sum());padded_tokens+=ids.numel();decode_step_tokens+=len(rows)*max(sequences.shape[1]-width-1,0)
  for row,seq in zip(rows,sequences[:,width:]):
   tokens=seq.cpu().tolist();end=tokens.index(learner.tokenizer.eos_token_id)+1 if learner.tokenizer.eos_token_id in tokens else len(tokens);generated_tokens+=end;text=learner.tokenizer.decode(tokens[:end],skip_special_tokens=True).strip();references=answer_lookup[row['id']];assert references;items.append({'id':row['id'],'exact_match':any(normalize_answer(text)==normalize_answer(r) for r in references),'token_f1':max(token_f1(text,r) for r in references),'prediction_sha256':hashlib.sha256(text.encode()).hexdigest(),'generated_tokens':end,'ended_with_eos':learner.tokenizer.eos_token_id in tokens})
 torch.cuda.synchronize();elapsed=time.perf_counter()-start;learner.cost['generation_input_tokens']+=input_tokens;learner.cost['generation_padded_prompt_tokens']+=padded_tokens;learner.cost['generation_output_tokens']+=generated_tokens;learner.cost['generation_decode_step_tokens']+=decode_step_tokens;learner.cost['generation_seconds']+=elapsed;return {'n':len(items),'normalized_exact_match':float(np.mean([x['exact_match'] for x in items])),'token_f1':float(np.mean([x['token_f1'] for x in items])),'items':items,'cost':{'input_tokens':input_tokens,'padded_prompt_tokens':padded_tokens,'generated_tokens':generated_tokens,'decode_step_tokens':decode_step_tokens,'wall_seconds':elapsed},'scoring':'Greedy generation <=32 new tokens; normalized EM/token-F1 against all provided NI references, not an official aggregate SuperNI score.'}
