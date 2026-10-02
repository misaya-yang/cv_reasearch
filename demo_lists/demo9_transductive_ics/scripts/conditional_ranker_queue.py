"""Own bounded preparation -> training continuation, no CUDA while supervising."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import defaultdict

ROOT=Path('/root/autodl-tmp/demo9_transductive_ics')
parser=argparse.ArgumentParser()
parser.add_argument('--name',default='conditional_ranker_pilot_v1')
parser.add_argument('--samples',type=int,default=40)
parser.add_argument('--max-bytes',type=int,default=2_000_000_000)
parser.add_argument('--classes',default='0,4,8,12,16,20')
parser.add_argument('--per-class',type=int,default=10)
parser.add_argument('--held-count',type=int,default=2)
parser.add_argument('--dev-count',type=int,default=1)
parser.add_argument('--controls',action='store_true')
parser.add_argument('--development-gate',action='store_true')
args=parser.parse_args()
if '/' in args.name or args.name.startswith('.'):raise ValueError('Own result directory name required')
OUT=ROOT/'results'/args.name
STATUS=OUT/'queue_status.json'
state=dict(state='RUNNING',started=time.time(),stages=[])
def save():
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(state));tmp.replace(STATUS)
def run(name,args):
    state['active_stage']=name;state['stages'].append(dict(name=name,state='RUNNING',started=time.time()));save()
    env=dict(os.environ,PYTHONPATH='/root/demo4_cache/env',DEMO4_GPU_FRAC='.3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with open(OUT/(name+'.log'),'a') as log:
        child=subprocess.Popen([sys.executable]+args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        state['child_pid']=child.pid;save();code=child.wait()
    state['stages'][-1].update(state='COMPLETED' if code==0 else 'ERROR',returncode=code,ended=time.time());save()
    if code:raise RuntimeError(f'{name} failed, see preserved log')
OUT.mkdir(parents=True,exist_ok=True)
try:
    base=['scripts/conditional_ranker_prepare.py','--prepared-root','/root/autodl-tmp/demo9','--out',str(OUT),'--stage-limit',str(args.samples),'--max-bytes',str(args.max_bytes),'--M','3','--classes',args.classes,'--per-class',str(args.per_class),'--held-count',str(args.held_count),'--dev-count',str(args.dev_count)]
    run('pairs',base+['--stage','pairs'])
    if args.development_gate:
        short=base.copy();short[short.index('--stage-limit')+1]='20'
        run('candidates_gate20',short+['--stage','candidates'])
        import torch
        initial=json.loads((OUT/'report.json').read_text());counts=defaultdict(lambda:[0,0,0,0]);units=0
        for entry in initial['records']:
            row=torch.load(entry['path'],weights_only=False,map_location='cpu',mmap=True)
            if row['split']=='test':continue
            iu=row['candidate_iu'];j=int(row['candidate_iou'].argmax());v=counts[row['c']]
            for k,x in enumerate([iu[0,0],iu[0,1],iu[j,0],iu[j,1]]):v[k]+=int(x)
            units+=1
        direct=sum(v[0]/max(v[1],1) for v in counts.values())/len(counts)*100
        oracle=sum(v[2]/max(v[3],1) for v in counts.values())/len(counts)*100
        state['development_headroom_gate']=dict(episodes=units,classes=len(counts),direct=direct,oracle=oracle,gap_pp=oracle-direct,test_labels_used=False,minimum_pp=2)
        save()
        if oracle-direct<2:
            state.update(state='STOPPED_INSUFFICIENT_DEVELOPMENT_CANDIDATE_HEADROOM',ended=time.time());save();sys.exit(0)
    run('candidates',base+['--stage','candidates'])
    report=json.loads((OUT/'report.json').read_text())
    if report['state']!='CANDIDATE_STAGE_COMPLETED':raise RuntimeError('Preparation budget/gate incomplete; do not train')
    index=OUT/'report.json'
    if args.controls:
        control_out=ROOT/'results'/(args.name+'_controls')
        run('controls',['scripts/conditional_ranker_controls.py','--report',str(index),'--out',str(control_out),'--limit',str(args.samples)])
        index=control_out/'report.json'
    run('train50',['scripts/conditional_ranker_train.py','--file',str(index),'--manifest',str(OUT/'manifest.json'),'--out-dir',str(ROOT/'results'/(args.name+'_train')),'--epochs','50','--arms','scalar,rq,rqdonor','--max-bytes',str(args.max_bytes)])
    state.update(state='COMPLETED',ended=time.time());save()
except BaseException as exc:
    state.update(state='ERROR',error=repr(exc),ended=time.time());save();raise
