"""Post-hoc train/dev coverage check; does not train, select or change protocol."""
import argparse
import json
from pathlib import Path
import time
from run import old, rollout, save


def main(directory):
    out=Path(directory);protocol=json.loads((out/'protocol.json').read_text())
    scale=json.loads((out/'scale_lock.json').read_text())['scale']
    result={};started=time.time()
    for split in ('train','dev'):
        bank=old.make_bank(protocol[split+'_seeds'],protocol['environment'])
        result[split]={}
        for action in (0,1):
            rows=[rollout(x,protocol['environment'],1,'overlap',scale,action=action)[0] for x in bank]
            result[split][str(action)]={'successes':sum(r['success'] for r in rows),'n':len(rows),'rows':rows}
            print(split,old.ACTIONS[action],result[split][str(action)]['successes'],len(rows))
    result['diagnostic_only']=True;result['seconds']=time.time()-started
    save(out/'freeze_coverage.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    main(args.directory)
