"""Ongoing native SAM2.1 response-distillation study; compact artifacts only."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/takeover_20261001_v1/distill_v1/sam2_distill'


def main():
    OUT.mkdir(exist_ok=False)
    protocol={'method':'single original decode response-distilled ranker, native SAM2.1-L states and same original-view features','train':'previous256 COCO images','validation':'previous128 COCO images','test_primary':'new384 seed2031 excluding all prior920 images','secondary':'previous512 cohort follow-up and DAVIS30 first frames; no DAVIS training','models':'same four ranker variants and same fixed three seeds/hyperparameters as SAM1, no retuning','artifact_storage':'compressed compact features and metrics only; no full repeated masks or model-weight copies','concurrent_GPU_runs_allowed':True}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    excluded=set()
    for name in ('coco_quality_seed2027_v1','coco_flip_validation_seed2028_v1','coco_prompt_holdout_seed2029_v1','coco_distill_test_seed2030_v1'):
        excluded.update(x['image_id'] for x in json.loads((ROOT/'assets'/name/'manifest.json').read_text())['images'])
    assert len(excluded)==920
    exclusion=OUT/'excluded_manifest.json';exclusion.write_text(json.dumps({'images':[{'image_id':i} for i in sorted(excluded)]})+'\n')
    new_subset=ROOT/'assets/coco_sam2_distill_test_seed2031_v1'
    checkpoint=ROOT/'assets/checkpoints/sam2.1_hiera_large.pt'
    jobs=[('prepare_new384',['research/quality_mechanisms/prepare_coco_subset.py','--annotations-zip','/root/autodl-pub/COCO2017/annotations_trainval2017.zip','--images-zip','/root/autodl-pub/COCO2017/val2017.zip','--output-dir',str(new_subset),'--seed','2031','--num-images','384','--exclude-manifest',str(exclusion)])]
    def export(name,subset):
        return (name,['tools/takeover_sam2_quality.py','--subset',str(subset),'--checkpoint',str(checkpoint),'--config','configs/sam2.1/sam2.1_hiera_l.yaml','--output',str(OUT/(name+'_teacher.json')),'--feature-output',str(OUT/(name+'.npz'))])
    jobs += [export('train256',ROOT/'assets/coco_prompt_holdout_seed2029_v1'),export('validation128',ROOT/'assets/coco_flip_validation_seed2028_v1'),('train_students',['tools/takeover_train_rank.py','--train',str(OUT/'train256.npz'),'--validation',str(OUT/'validation128.npz'),'--output',str(OUT/'models')]),export('test384',new_subset),export('davis30',ROOT/'assets/davis_val_firstframe_v1')]
    for name in ('test384','davis30'):
        jobs.append(('evaluate_'+name,['tools/takeover_evaluate_rank.py','--features',str(OUT/(name+'.npz')),'--models',str(OUT/'models'),'--output',str(OUT/('student_'+name+'.json'))]))
    status={'status':'RUNNING','stages':[]}
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(OUT/'queue_status.json')
    save()
    try:
        for name,argv in jobs:
            if argv[0]=='tools/takeover_sam2_quality.py':
                # Resource availability, rather than existence of another GPU job.
                while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<6000:
                    status['status']='WAITING_FOR_SUFFICIENT_VRAM';save();time.sleep(15)
            status['status']='RUNNING';row={'name':name,'status':'RUNNING','command':[sys.executable,*argv]};status['stages'].append(row);save();start=time.monotonic()
            print('START '+name,flush=True)
            with (OUT/(name+'.log')).open('x') as log:
                p=subprocess.Popen(row['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);row['pid']=p.pid;save();code=p.wait()
            row.update(status='DONE' if code==0 else 'FAILED',returncode=code,wall_seconds=time.monotonic()-start);save()
            if code:raise RuntimeError(name+' failed')
            print('DONE '+name,flush=True)
        status['status']='COMPLETED'
    except BaseException as e:status.update(status='ERROR',error=str(e));raise
    finally:save()


if __name__=='__main__':main()
