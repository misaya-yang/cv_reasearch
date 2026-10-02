"""Prepare disjoint tests while training runs; proceed only after dev sanity."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/takeover_20261001_v1/sam2_local_student_v1'
OUT=BASE/'followup'


def main():
    OUT.mkdir(exist_ok=False)
    state={'status':'PREPARING_FRESH_COHORT_CPU','pid':os.getpid(),'stages':[],
           'protocol':'Seed2033 fresh384 excludes all1688 previous COCO images. Fixed rank64/static-skip student/budgets selected only from old24 development; new results never used for early stopping. Development sanity gate: retain native-head and native-dynamic mean losses no larger than0.5percentage points at10% in each regime before spending fresh test. This is operational scope, not success/precision/publication threshold; if unmet keep fresh test untouched and fix on development. DAVIS30 prior external firstframes, no temporal memory.'}
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUT/'queue_status.json')
    def run(name,args):
        state.update(status='RUNNING',current_stage=name);row={'name':name,'command':args,'started_unix':time.time()};state['stages'].append(row)
        with (OUT/(name+'.log')).open('x') as log:
            p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);row['pid']=p.pid;save();code=p.wait()
        row.update(returncode=code,finished_unix=time.time());save()
        if code:raise RuntimeError(name+' failed; inspect preserved evidence')
    save()
    try:
        excluded=set()
        for name in ('coco_quality_seed2027_v1','coco_flip_validation_seed2028_v1','coco_prompt_holdout_seed2029_v1','coco_distill_test_seed2030_v1','coco_sam2_distill_test_seed2031_v1','coco_interior_test_seed2032_v1'):
            excluded.update(r['image_id'] for r in json.loads((ROOT/'assets'/name/'manifest.json').read_text())['images'])
        assert len(excluded)==1688
        exclusion=OUT/'excluded_manifest.json';exclusion.write_text(json.dumps({'images':[{'image_id':i} for i in sorted(excluded)]})+'\n')
        subset=ROOT/'assets/coco_sam2_local_test_seed2033_v1'
        run('prepare_fresh384',[sys.executable,'research/quality_mechanisms/prepare_coco_subset.py','--annotations-zip','/root/autodl-pub/COCO2017/annotations_trainval2017.zip','--images-zip','/root/autodl-pub/COCO2017/val2017.zip','--output-dir',str(subset),'--seed','2033','--num-images','384','--exclude-manifest',str(exclusion)])
        state.update(status='WAITING_TRAINING_CPU_ONLY',current_stage=None);save()
        while True:
            train=json.loads((BASE/'report.json').read_text())
            if train['status']=='ERROR':raise RuntimeError('Training failed; preserve stage and repair trainer')
            if train['status']=='COMPLETED_SAM2_REAL_HEAD_DEVELOPMENT':break
            time.sleep(10)
        gate={r:{key:train['summary'][r]['0.1'][key] for key in ('mean_head_delta_iou','mean_native_dynamic_delta_iou')} for r in ('central','near_boundary','box')}
        state['development_sanity']=gate
        if any(value < -.005 for row in gate.values() for value in row.values()):
            state.update(status='NEEDS_DEVELOPMENT_FIX_FRESH_TEST_UNTOUCHED');save();return
        run('fresh384',[sys.executable,'tools/takeover_eval_sam2_local_head.py','--subset',str(subset),'--output',str(BASE/'fresh384.json'),'--scope','Fresh seed2033 COCO384 excludes all1688 previous images, frozen student and budgets; no tuning on these results'])
        run('davis30',[sys.executable,'tools/takeover_eval_sam2_local_head.py','--subset',str(ROOT/'assets/davis_val_firstframe_v1'),'--output',str(BASE/'davis30.json'),'--scope','Previously available DAVIS30 external firstframes, frozen SAM2 student, void excluded; no video memory'])
        state.update(status='COMPLETED',current_stage=None)
    except BaseException as e:state.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
