"""Follow a healthy interior study with response-signal distillation."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/takeover_20261001_v1/distill_v1/response_distill_v1'
PREVIOUS=OUT.parent/'interior_v1/queue_status.json'


def main(resume=False):
    OUT.mkdir(exist_ok=resume)
    report=json.loads((OUT/'queue_status.json').read_text()) if resume else {'status':'WAITING_PREVIOUS_STUDY','stages':[]}
    if resume and report['status'] not in ('ERROR','COMPLETED'):raise RuntimeError('Verify the running supervisor before attempting recovery')
    report['pid']=os.getpid()
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(OUT/'queue_status.json')
    (OUT/'protocol.json').write_text(json.dumps({'hypothesis':'Predict local coordinate-response choice from one original SAM2-L decode, avoiding expensive MATH-attention JVPs at inference.','teacher':'Same two-direction JVP reconstruction, radius8, original point candidates1..3; no GT selection','evidence':'Prior fixed384 near-boundary +4.2468pp CI[2.5400,6.0682]; center no signal. This384 now development evidence for the distillation hypothesis.','data':'train256, validation128; fresh seed2032 384 excludes prior1304; DAVIS30 firstframes, void excluded','controls':'same four architectures/objectives and three seeds; head, largest, fixed tokens, physical-response teacher in preceding interior study; native dynamic comparison must use corresponding original outputs','attention':'MATH during teacher generation; no native optimized latency claim; no test tuning','storage':'compressed compact features; canonical pretrained weights reused'},indent=2)+'\n')
    save()
    while True:
        previous=json.loads(PREVIOUS.read_text())
        if previous['status'] in ('COMPLETED','COMPLETED_WITH_FAILURES','ERROR'):break
        # The current study already has a complete train/evaluate queue. Follow
        # it rather than contend for the same newly available VRAM.
        time.sleep(10)
    def export(name,subset):
        return (name,['tools/takeover_tangent_pilot.py','--subset',str(ROOT/'assets'/subset),'--output',str(OUT/(name+'_teacher.json')),'--feature-output',str(OUT/(name+'.npz')),'--scope',name+'; frozen two-direction teacher; single-view feature export; point regimes only'])
    jobs=[export('smoke24','coco_quality_seed2027_v1'),export('train256','coco_prompt_holdout_seed2029_v1'),export('validation128','coco_flip_validation_seed2028_v1'),('train_students',['tools/takeover_train_rank.py','--train',str(OUT/'train256.npz'),'--validation',str(OUT/'validation128.npz'),'--output',str(OUT/'models'),'--teacher-description','two coordinate JVP radius8 response reconstruction; point regimes only']),export('test384','coco_interior_test_seed2032_v1'),export('davis30','davis_val_firstframe_v1')]
    for name in ('test384','davis30'):
        jobs.append(('evaluate_'+name,['tools/takeover_evaluate_rank.py','--features',str(OUT/(name+'.npz')),'--models',str(OUT/'models'),'--output',str(OUT/('student_'+name+'.json'))]))
    try:
        for name,args in jobs:
            prior=[r for r in report['stages'] if r['name']==name]
            if prior and prior[-1]['status']=='DONE':continue
            if prior:
                # Keep failed attempts and restart only the affected stage.
                for suffix in ('_teacher.json','.npz','.json','.log'):
                    old=OUT/(name+suffix)
                    if old.exists():old.rename(OUT/(old.name+f'.failed_attempt{len(prior)}'))
            if args[0]=='tools/takeover_tangent_pilot.py':
                while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<6000:
                    report['status']='WAITING_VRAM';save();time.sleep(10)
            row={'name':name,'status':'RUNNING','command':[sys.executable,*args]};report['stages'].append(row);report['status']='RUNNING';save()
            print('START '+name,flush=True);start=time.monotonic()
            with (OUT/(name+'.log')).open('x') as log:
                p=subprocess.Popen(row['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);row['pid']=p.pid;save();code=p.wait()
            row.update(status='DONE' if code==0 else 'FAILED',returncode=code,wall_seconds=time.monotonic()-start);save()
            if code:raise RuntimeError(name+' failed; preserve traceback and fix this stage')
            print('DONE '+name,flush=True)
        report['status']='COMPLETED'
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true');main(p.parse_args().resume)
