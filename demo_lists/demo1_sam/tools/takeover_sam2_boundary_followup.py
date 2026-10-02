"""Freeze objective controls and test on a new shared cohort, two quality workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/takeover_20261001_v1'
TRAIN=BASE/'sam2_boundary_objective_v1'
OUT=TRAIN/'followup'


def main():
    OUT.mkdir(exist_ok=False)
    r={'status':'PREPARING_FRESH256_CPU','pid':os.getpid(),'stages':[],
       'protocol':'New seed2035 COCO256 excludes all2328 previous COCO IDs. Student structure/rank/fallback budgets identical; three teacher-only objective checkpoints selected by old12 validation before new output, original v1 checkpoint strong model control. SAME actual compiled decoderFP16/FP32 encoder, native/fullphase controls retained. Two concurrent QUALITY workers only, no latency interpretation; original unique weights/data reused. Each of original/coefficient/uniform/boundary scored at0/5/10% on identical cohort, no new GT tuning; two response models then matched DAVIS30 firstframe external replays, no temporal memory. Boundary weighting is a candidate mechanism requiring actual task evidence, not asserted novel or successful.'}
    def save():
        p=OUT/'queue_status.tmp';p.write_text(json.dumps(r,indent=2)+'\n');p.replace(OUT/'queue_status.json')
    def start(name,args):
        log=(OUT/(name+'.log')).open('x');p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);row={'name':name,'pid':p.pid,'command':args,'start_unix':time.time()};r['stages'].append(row);save();return p,log,row
    def finish(job):
        p,log,row=job;code=p.wait();log.close();row.update(returncode=code,end_unix=time.time());save();return code
    save()
    try:
        excluded=set()
        for name in ('coco_quality_seed2027_v1','coco_flip_validation_seed2028_v1','coco_prompt_holdout_seed2029_v1','coco_distill_test_seed2030_v1','coco_sam2_distill_test_seed2031_v1','coco_interior_test_seed2032_v1','coco_sam2_local_test_seed2033_v1','coco_sam2_converge_test_seed2034_v1'):
            excluded.update(x['image_id'] for x in json.loads((ROOT/'assets'/name/'manifest.json').read_text())['images'])
        assert len(excluded)==2328
        ex=OUT/'excluded_manifest.json';ex.write_text(json.dumps({'images':[{'image_id':v} for v in sorted(excluded)]})+'\n')
        subset=ROOT/'assets/coco_sam2_boundary_test_seed2035_v1'
        job=start('prepare_fresh256',[sys.executable,'research/quality_mechanisms/prepare_coco_subset.py','--annotations-zip','/root/autodl-pub/COCO2017/annotations_trainval2017.zip','--images-zip','/root/autodl-pub/COCO2017/val2017.zip','--output-dir',str(subset),'--seed','2035','--num-images','256','--exclude-manifest',str(ex)])
        if finish(job):raise RuntimeError('New cohort preparation failed')
        r.update(status='WAITING_FROZEN_OBJECTIVE_MODELS_CPU_ONLY');save()
        while True:
            t=json.loads((TRAIN/'report.json').read_text())
            if t['status']=='ERROR':raise RuntimeError('Matched objectives failed; preserve untouched new cohort')
            if t['status']=='COMPLETED_MATCHED_OBJECTIVE_DEVELOPMENT':break
            time.sleep(5)
        r['frozen_selected_epochs']=t['selected'];save()
        models=[('original_v1',BASE/'sam2_local_student_v1/deployment.pt')]+[(arm,TRAIN/arm/'deployment.pt') for arm in ('coefficient','response_uniform','response_boundary')]
        r.update(status='CONCURRENT_FRESH256_TASK_TESTS');save()
        for offset in (0,2):
            jobs=[]
            for arm,path in models[offset:offset+2]:
                cmd=[sys.executable,'tools/takeover_eval_sam2_local_head.py','--subset',str(subset),'--student-path',str(path),'--compiled-decoder-fp16','--output',str(TRAIN/('fresh256_'+arm+'.json')),'--scope','Independent seed2035 COCO256 excludes2328 earlier IDs; matched frozen teacher-only objective comparison, no GT tuning; concurrent quality, no speed claim']
                jobs.append(start('fresh256_'+arm,cmd))
            codes=[finish(job) for job in jobs]
            if any(codes):raise RuntimeError('Fresh quality worker failed; preserve other completed worker')
        r.update(status='CONCURRENT_DAVIS30_TASK_VERIFICATION');save();jobs=[]
        for arm in ('response_uniform','response_boundary'):
            jobs.append(start('davis30_'+arm,[sys.executable,'tools/takeover_eval_sam2_local_head.py','--subset',str(ROOT/'assets/davis_val_firstframe_v1'),'--student-path',str(TRAIN/arm/'deployment.pt'),'--compiled-decoder-fp16','--output',str(TRAIN/('davis30_'+arm+'.json')),'--scope','Matched frozen objective-model DAVIS30 firstframe external verification; old dataset reused, void excluded, no temporal memory; concurrent task quality only']))
        codes=[finish(job) for job in jobs]
        if any(codes):raise RuntimeError('External quality verification failed; preserve other completed worker')
        r['status']='COMPLETED'
    except BaseException as e:r.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
