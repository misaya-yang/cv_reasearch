"""Freeze adequate training by old validation, compare on one new shared cohort."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/takeover_20261001_v1'
V2=BASE/'sam2_local_student_v2_converge'
OUT=V2/'followup'


def main():
    OUT.mkdir(exist_ok=False)
    status={'status':'PREPARING_FRESH256_CPU','pid':os.getpid(),'stages':[],
            'protocol':'Optimization adequacy intervention only: same69760-parameter rank64 architecture/PCA/old12 train+12 teacher-MSE validation, optimizer/RNG continuation, up to2400epochs patience100. New seed2034 COCO256 excludes all2072 earlier IDs. Wait completed teacher-MSE epoch selection before either new-cohort GPU quality run; no GT model/epoch/budget selection. Actual compiled decoderFP16, SAMEFP32 encoder casts and native/fullphase controls. Frozen v1 and v2 share the identical new cohort, all0/5/10 budgets retained. DAVIS v2 is repeated external firstframe verification, not a new dataset or temporal result. No latency claim from concurrent quality runs.'}
    def save():
        t=OUT/'queue_status.tmp';t.write_text(json.dumps(status,indent=2)+'\n');t.replace(OUT/'queue_status.json')
    def run(name,cmd):
        status.update(status='RUNNING',current_stage=name)
        row={'name':name,'command':cmd,'start_unix':time.time()};status['stages'].append(row)
        with (OUT/(name+'.log')).open('x') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);row['pid']=p.pid;save();code=p.wait()
        row.update(returncode=code,end_unix=time.time());save()
        if code:raise RuntimeError(name+' failed; preserve completed stages and inspect log')
    save()
    try:
        excluded=set()
        for name in ('coco_quality_seed2027_v1','coco_flip_validation_seed2028_v1','coco_prompt_holdout_seed2029_v1','coco_distill_test_seed2030_v1','coco_sam2_distill_test_seed2031_v1','coco_interior_test_seed2032_v1','coco_sam2_local_test_seed2033_v1'):
            excluded.update(x['image_id'] for x in json.loads((ROOT/'assets'/name/'manifest.json').read_text())['images'])
        assert len(excluded)==2072
        exclusion=OUT/'excluded_manifest.json';exclusion.write_text(json.dumps({'images':[{'image_id':i} for i in sorted(excluded)]})+'\n')
        subset=ROOT/'assets/coco_sam2_converge_test_seed2034_v1'
        run('prepare_fresh256',[sys.executable,'research/quality_mechanisms/prepare_coco_subset.py','--annotations-zip','/root/autodl-pub/COCO2017/annotations_trainval2017.zip','--images-zip','/root/autodl-pub/COCO2017/val2017.zip','--output-dir',str(subset),'--seed','2034','--num-images','256','--exclude-manifest',str(exclusion)])
        status.update(status='WAITING_TRAINING_CPU_ONLY',current_stage=None);save()
        while True:
            train=json.loads((V2/'report.json').read_text())
            if train['status']=='ERROR':raise RuntimeError('Continuation failed; new cohort untouched by models')
            if train['status']=='COMPLETED_SAM2_REAL_HEAD_DEVELOPMENT':break
            time.sleep(10)
        status['frozen_v2_epoch']=train['selected_epoch'];status['plateau_before_cap']=train['plateau_before_cap'];save()
        for version,student in [('v1',BASE/'sam2_local_student_v1/deployment.pt'),('v2',V2/'deployment.pt')]:
            run('fresh256_'+version,[sys.executable,'tools/takeover_eval_sam2_local_head.py','--subset',str(subset),'--student-path',str(student),'--output',str(V2/('fresh256_compiled_fp16_'+version+'.json')),'--compiled-decoder-fp16','--minimum-free-mib','5000','--scope','Independent seed2034 COCO256 excludes2072 prior images, matched frozen v1/v2 optimization-only comparison; epoch selected by old teacherMSE before any cohort output'])
        run('davis30_v2',[sys.executable,'tools/takeover_eval_sam2_local_head.py','--subset',str(ROOT/'assets/davis_val_firstframe_v1'),'--student-path',str(V2/'deployment.pt'),'--output',str(V2/'davis30_compiled_fp16_v2.json'),'--compiled-decoder-fp16','--minimum-free-mib','5000','--scope','Repeated external DAVIS30 firstframes with frozen convergence student; void excluded, no temporal memory; v1 previous sameprecision evidence retained'])
        status.update(status='COMPLETED',current_stage=None)
    except BaseException as e:status.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
