"""Independent matched fixed/learned-basis test with original response model control."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/takeover_20261001_v1'
TRAIN=BASE/'sam2_learned_basis_v1'
OUT=TRAIN/'followup'


def main():
    OUT.mkdir(exist_ok=False)
    r={'status':'PREPARING_FRESH256_CPU','pid':os.getpid(),'stages':[],
       'protocol':'New seed2036 COCO256 excludes2584 earlier COCO IDs. Same cohort for frozen previous boundary-response model, fixed-basis continuation and learned-basis model. Each model uses its old teacher-only validation epoch; no GT epoch/budget adjustment. Actual compiled FP16 decoder/FP32 encoder casts, native and complete phase controls, all0/5/10 budgets/4raw interfaces. Three concurrent quality workers use available memory; no timing claim. Follow with matched fixed/learned DAVIS30 firstframes/void exclusion/no temporal memory. Joint basis adds training degrees of freedom but deployed512x64 factor shape/storage unchanged; do not infer speed from shape or oracle.'}
    def save():
        p=OUT/'queue_status.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(OUT/'queue_status.json')
    def start(name,cmd):
        log=(OUT/(name+'.log')).open('x');p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);row={'name':name,'pid':p.pid,'command':cmd,'start_unix':time.time()};r['stages'].append(row);save();return p,log,row
    def finish(job):
        p,log,row=job;code=p.wait();log.close();row.update(returncode=code,end_unix=time.time());save();return code
    save()
    try:
        excluded=set()
        for name in ('coco_quality_seed2027_v1','coco_flip_validation_seed2028_v1','coco_prompt_holdout_seed2029_v1','coco_distill_test_seed2030_v1','coco_sam2_distill_test_seed2031_v1','coco_interior_test_seed2032_v1','coco_sam2_local_test_seed2033_v1','coco_sam2_converge_test_seed2034_v1','coco_sam2_boundary_test_seed2035_v1'):
            excluded.update(x['image_id'] for x in json.loads((ROOT/'assets'/name/'manifest.json').read_text())['images'])
        assert len(excluded)==2584
        ex=OUT/'excluded_manifest.json';ex.write_text(json.dumps({'images':[{'image_id':v} for v in sorted(excluded)]})+'\n')
        subset=ROOT/'assets/coco_sam2_basis_test_seed2036_v1'
        job=start('prepare_fresh256',[sys.executable,'research/quality_mechanisms/prepare_coco_subset.py','--annotations-zip','/root/autodl-pub/COCO2017/annotations_trainval2017.zip','--images-zip','/root/autodl-pub/COCO2017/val2017.zip','--output-dir',str(subset),'--seed','2036','--num-images','256','--exclude-manifest',str(ex)])
        if finish(job):raise RuntimeError('Cohort preparation failed')
        r['status']='WAITING_FROZEN_BASIS_MODELS_CPU_ONLY';save()
        while True:
            t=json.loads((TRAIN/'report.json').read_text())
            if t['status']=='ERROR':raise RuntimeError('Basis development failed; fresh cohort untouched')
            if t['status']=='COMPLETED_MATCHED_BASIS_DEVELOPMENT':break
            time.sleep(5)
        r['frozen_epochs']=t['selected'];r['status']='CONCURRENT_FRESH256_TASK_TESTS';save()
        models=[('previous_boundary',BASE/'sam2_boundary_objective_v1/response_boundary/deployment.pt')]+[(arm,TRAIN/arm/'deployment.pt') for arm in ('fixed_basis','learned_basis')]
        jobs=[]
        for arm,path in models:
            jobs.append(start('fresh256_'+arm,[sys.executable,'tools/takeover_eval_sam2_local_head.py','--subset',str(subset),'--student-path',str(path),'--compiled-decoder-fp16','--output',str(TRAIN/('fresh256_'+arm+'.json')),'--scope','Independent seed2036 COCO256 excludes2584 old IDs; matched frozen basis intervention and previous boundary model control; no GT tuning; concurrent task quality not latency']))
        codes=[finish(job) for job in jobs]
        if any(codes):raise RuntimeError('Fresh basis quality worker failed; preserve other completed workers')
        r['status']='CONCURRENT_DAVIS30_VERIFICATION';save();jobs=[]
        for arm in ('fixed_basis','learned_basis'):
            jobs.append(start('davis30_'+arm,[sys.executable,'tools/takeover_eval_sam2_local_head.py','--subset',str(ROOT/'assets/davis_val_firstframe_v1'),'--student-path',str(TRAIN/arm/'deployment.pt'),'--compiled-decoder-fp16','--output',str(TRAIN/('davis30_'+arm+'.json')),'--scope','Frozen fixed/learned basis matched external DAVIS30 firstframes; void exclusion, no video memory; repeated external cohort, concurrent quality only']))
        codes=[finish(job) for job in jobs]
        if any(codes):raise RuntimeError('DAVIS worker failed; preserve other completed verification')
        r['status']='COMPLETED'
    except BaseException as e:r.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
