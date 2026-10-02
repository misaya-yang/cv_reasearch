"""Sequential useful GPU studies, with independent failures and durable progress."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/takeover_20261001_v1/distill_v1/interior_v1'


def main():
    OUT.mkdir(exist_ok=False)
    status={'status':'RUNNING','stages':[],'policy':'GPU stays on; other sessions are permitted; do not manufacture compute after the specified studies finish.'}
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(OUT/'queue_status.json')
    excluded=set()
    for name in ('coco_quality_seed2027_v1','coco_flip_validation_seed2028_v1','coco_prompt_holdout_seed2029_v1','coco_distill_test_seed2030_v1','coco_sam2_distill_test_seed2031_v1'):
        excluded.update(i['image_id'] for i in json.loads((ROOT/'assets'/name/'manifest.json').read_text())['images'])
    assert len(excluded)==1304
    exclusion=OUT/'excluded.json';exclusion.write_text(json.dumps({'images':[{'image_id':i} for i in sorted(excluded)]})+'\n')
    test=ROOT/'assets/coco_interior_test_seed2032_v1'
    protocol={'hypothesis':'Constrain common counterfactual point probes to the original head foreground to reduce off-object boundary perturbations; distill the resulting decision into one original-view decode.','frozen_rule':'8-segment foreground ray, radius8 longest-edge1024; radius1/4/8 strong controls; boxes original head','models':'same fixed 4 ranker variants, 3 seeds, hyperparameters; no test selection','test':'384 fresh COCO images seed2032 excluding all prior1304; DAVIS30 firstframes secondary','native_control':'official SAM2 dynamic token0 stability .05/.98, plus original threecandidate head and largest','storage':'only compressed compact features and task metrics; canonical base weights reused'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def extract(name,subset):
        return (name,['tools/takeover_interior_probe.py','--subset',str(subset),'--output',str(OUT/(name+'_teacher.json')),'--feature-output',str(OUT/(name+'.npz')),'--scope',name+'; frozen interior rule and fixed radius controls; '+('development' if name in ('train256','validation128') else 'no tuning on this cohort')])
    jobs=[extract('train256',ROOT/'assets/coco_prompt_holdout_seed2029_v1'),extract('validation128',ROOT/'assets/coco_flip_validation_seed2028_v1'),('train_students',['tools/takeover_train_rank.py','--train',str(OUT/'train256.npz'),'--validation',str(OUT/'validation128.npz'),'--output',str(OUT/'models'),'--teacher-description','interior-constrained radius8 common point probes; original head on boxes']),('prepare_fresh384',['research/quality_mechanisms/prepare_coco_subset.py','--annotations-zip','/root/autodl-pub/COCO2017/annotations_trainval2017.zip','--images-zip','/root/autodl-pub/COCO2017/val2017.zip','--output-dir',str(test),'--seed','2032','--num-images','384','--exclude-manifest',str(exclusion)]),extract('test384',test),extract('davis30',ROOT/'assets/davis_val_firstframe_v1')]
    for name in ('test384','davis30'):
        jobs.append(('evaluate_'+name,['tools/takeover_evaluate_rank.py','--features',str(OUT/(name+'.npz')),'--models',str(OUT/'models'),'--output',str(OUT/('student_'+name+'.json'))]))
    save()
    failed=set()
    for name,args in jobs:
        deps={'train_students':{'train256','validation128'},'test384':{'prepare_fresh384'},'evaluate_test384':{'train_students','test384'},'evaluate_davis30':{'train_students','davis30'}}.get(name,set())
        if deps&failed:
            failed.add(name);status['stages'].append({'name':name,'status':'SKIPPED_DEPENDENCY_FAILURE','returncode':None});save();continue
        # Allow meaningful concurrent work when memory is available.
        if args[0]=='tools/takeover_interior_probe.py':
            while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<6000:
                status['status']='WAITING_VRAM';save();time.sleep(15)
        row={'name':name,'status':'RUNNING','command':[sys.executable,*args]};status['stages'].append(row);status['status']='RUNNING';save()
        start=time.monotonic();print('START '+name,flush=True)
        with (OUT/(name+'.log')).open('x') as log:
            p=subprocess.Popen(row['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);row['pid']=p.pid;save();code=p.wait()
        row.update(status='DONE' if code==0 else 'FAILED',returncode=code,wall_seconds=time.monotonic()-start);save()
        print(row['status']+' '+name,flush=True)
        if code:
            # Continue independent exports on failures, preserve tracebacks, and
            # omit dependent training/evaluation without pretending they ran.
            failed.add(name)
    status['status']='COMPLETED' if all(x['returncode']==0 for x in status['stages']) else 'COMPLETED_WITH_FAILURES';save()


if __name__=='__main__':main()
