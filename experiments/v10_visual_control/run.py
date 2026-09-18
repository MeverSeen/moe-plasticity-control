"""Reproducible development, training and sealed held-out evaluation entrypoints."""
import argparse
import copy
import hashlib
import json
import platform
import resource
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from core import Config,Data,Environment
from control import Policy,rollout,train_batch

HERE=Path(__file__).resolve().parent

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,ensure_ascii=False))

def score(rows):
    # Development ranking first feasibility, then all-root final old forgetting; absolute loss tie-break.
    return (-np.mean([r['success'] for r in rows]),np.mean([r['forgetting'] for r in rows]),np.mean([r['final_loss'] for r in rows]))

def evaluate(data,cfg,seeds,policy,split='dev',evaluation='eval'):
    rows=[]
    for seed in seeds:
        torch.manual_seed(seed+999)
        r,_=rollout(Environment(data,cfg,seed,split,evaluation),policy); rows.append(r)
    return rows

def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))}

def calibration(args,data):
    records=[];start=time.perf_counter()
    for mode,lr in [('fixed',.001),('dynamic',.001),('dynamic',.003)]:
        cfg=Config(steps=args.steps,lr=lr,routing=mode,target=args.target)
        for seed in [1000100,1000101]:
            r,_=rollout(Environment(data,cfg,seed,'train'),'fixed:1')
            records.append({'config':asdict(cfg),'result':r});save(args.out/'calibration.json',records)
            print('calibration',mode,lr,seed,'success',r['success'],'loss',[round(t['loss'],3) for t in r['tasks']],'seconds',round(r['wall_seconds'],2),flush=True)
    save(args.out/'cost.json',{'seconds':time.perf_counter()-start,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source_hashes':source_hashes(),'platform':platform.platform(),'torch':torch.__version__})

def train(args,data):
    cfg=Config(**json.loads(Path(args.config).read_text())) if args.config else Config(steps=args.steps,target=args.target)
    spec={
       'grpo_feedback':('grpo',False,False,'structured'),
       'grpo_progress':('grpo',False,True,'structured'),
       'ppo_feedback':('ppo',False,False,'structured'),
       'grpo_local':('grpo',True,False,'factorized'),
       'grpo_factorized':('grpo',False,False,'factorized'),
       'grpo_groups':('grpo',False,False,'groups'),
       'grpo_global':('grpo',False,False,'global')}
    if args.variant:spec={args.variant:spec[args.variant]}
    protocol={'config':asdict(cfg),'source_hashes':source_hashes(),'data_md5':data.md5,'variants':spec,'rounds':args.rounds,'group':args.group,'root_pool':args.roots,'controller_seeds':args.seeds,
        'train_roots':list(range(1010000,1010000+args.roots)),'dev_roots':[1020000,1020001],
        'heldout_roots':list(range(1030000,1030000+args.test_roots)), 'classes':{'train':[0,59],'dev':[60,79],'test':[80,99]},
        'noninferiority_margin':.05,'minimum_sequence_success':.9,'formal_power_claim':False,
        'selection':'lexicographic sequence success, all-root forgetting, final absolute loss; epochs 0, half, final; no final test selection',
        'reward':'-retained old-monitor positive mean loss increase / .1 - 3 per failed task',
        'comparisons':['Q1: expert/group/global matched alphabet','Q2: feedback/progress','Q3: GRPO/PPO','coordination: local/factorized/structured/rules'],
        'interaction_matching':'same root/trajectory/update caps, actual early-stop updates separately reported; not exact consumed-update equality',
        'stopping':'initial and every 10 updates; one check <= target; preserve all failures and initial successes',
        'memory':'total 200: equal per-task replay and monitor quota; old prefixes only; no reacquisition; stored per-example completion losses',
        'scope':'resource-sized visual pilot, four five-class tasks; class-disjoint heldout tasks; no 20x5 claim'}
    save(args.out/'protocol.json',protocol)
    roots=[]
    for seed in protocol['train_roots']:
        e=Environment(data,cfg,seed,'train')
        while e.k==0:
            ob=e.observation()
            if float(ob[1][0])<=cfg.target or e.t>=cfg.steps:e.finish(float(ob[1][0]))
            else:e.update(torch.ones(cfg.layers,cfg.experts))
        roots.append(e)
    save(args.out/'prefix_cost.json',[r.cost for r in roots])
    baseline={}
    for name in ['fixed:0','fixed:.25','fixed:.5','fixed:.75','fixed:1','fixed:1.5','schedule','local','coordinated']:
        baseline[name]=evaluate(data,cfg,protocol['dev_roots'],name);save(args.out/'baseline_dev.json',baseline)
        print('baseline',name,score(baseline[name]),flush=True)
    selected_baseline=min([n for n in baseline if n.startswith('fixed:')],key=lambda n:score(baseline[n]))
    selected={'best_fixed':selected_baseline,'schedule':'schedule','local':'local','coordinated':'coordinated'}
    training={};dev={}
    for name,(algo,local,progress,interface) in spec.items():
      for seed in args.seeds:
        key=f'{name}_{seed}';torch.manual_seed(seed);p=Policy(local,progress,interface);opt=torch.optim.Adam(p.parameters(),lr=.0003)
        candidates={};history=[]
        epochs=sorted(set([0,args.rounds//2,args.rounds]))
        for u in range(args.rounds+1):
            if u in epochs:
                rows=evaluate(data,cfg,protocol['dev_roots'],p); candidates[u]=rows
                torch.save(p.state_dict(),args.out/f'{key}_epoch{u}.pt')
                print('dev',key,u,score(rows),flush=True)
            if u==args.rounds:break
            # Root choice cycles independently of stochastic action RNG; same for all variants.
            root=roots[u%len(roots)]; torch.manual_seed(seed*10000+u)
            batch=train_batch(p,opt,[root],algo,args.group);batch['round']=u;batch['root']=root.seed;history.append(batch)
            save(args.out/f'train_{key}.json',history)
            print('train',key,u,'seconds',round(batch['wall_seconds'],1),'std',round(batch.get('reward_std',0),3),'delta',batch.get('parameter_change',0),flush=True)
        epoch=min(candidates,key=lambda u:score(candidates[u]));selected[key]={'variant':name,'seed':seed,'epoch':epoch,'path':f'{key}_epoch{epoch}.pt'}
        dev[key]=candidates;save(args.out/'dev.json',dev);save(args.out/'selection.json',selected)
    hashes={v['path']:hashlib.sha256((args.out/v['path']).read_bytes()).hexdigest() for v in selected.values() if isinstance(v,dict)}
    save(args.out/'locked.json',{'protocol_sha256':hashlib.sha256((args.out/'protocol.json').read_bytes()).hexdigest(),'source_hashes':source_hashes(),'selected_models':hashes,'selection':selected,'time_unix':time.time()})
    print('locked; evaluation is a separate command',flush=True)

def test(args):
    out=args.out;protocol=json.loads((out/'protocol.json').read_text());lock=json.loads((out/'locked.json').read_text())
    assert source_hashes()==lock['source_hashes'],'Source changed after lock; use new run.'
    assert hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()==lock['protocol_sha256']
    if (out/'test.json').exists():raise RuntimeError('Final evaluation already exists; do not overwrite')
    data=Data(args.data,final=True);cfg=Config(**protocol['config']);results={}
    for key,item in lock['selection'].items():
        if isinstance(item,str):policy=item
        else:
            algo,local,progress,interface=protocol['variants'][item['variant']];policy=Policy(local,progress,interface)
            path=out/item['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==lock['selected_models'][item['path']]
            policy.load_state_dict(torch.load(path,weights_only=True))
        results[key]=evaluate(data,cfg,protocol['heldout_roots'],policy,'test','test');save(out/'test.partial.json',results)
        print('heldout',key,score(results[key]),flush=True)
    save(out/'test.json',results)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['calibrate','train','test']);ap.add_argument('--data',default='.cache/datasets/cifar-complete.tar.gz');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--config');ap.add_argument('--steps',type=int,default=200);ap.add_argument('--target',type=float,default=1.25);ap.add_argument('--rounds',type=int,default=8);ap.add_argument('--group',type=int,default=2);ap.add_argument('--roots',type=int,default=8);ap.add_argument('--seeds',type=int,nargs='+',default=[1001,1002]);ap.add_argument('--test-roots',type=int,default=8);ap.add_argument('--variant');args=ap.parse_args();torch.set_num_threads(1)
    started=time.perf_counter()
    if args.mode!='test':args.out.mkdir(parents=True,exist_ok=False)
    if args.mode=='test':test(args)
    elif args.mode=='calibrate':calibration(args,Data(args.data))
    else:train(args,Data(args.data))
    save(args.out/f'run_cost_{args.mode}.json',{'wall_seconds':time.perf_counter()-started,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'platform':platform.platform(),'torch':torch.__version__,'cpu_threads':torch.get_num_threads()})
if __name__=='__main__':main()
