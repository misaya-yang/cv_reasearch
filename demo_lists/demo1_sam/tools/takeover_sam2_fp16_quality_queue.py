"""Actual compiled decoder-FP16 task checks after the matched cost screening."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/takeover_20261001_v1/sam2_local_student_v1'
OUT=BASE/'fp16_quality_v1'


def main():
    OUT.mkdir(exist_ok=False)
    status={'status':'STARTING','pid':os.getpid(),'stages':[],
            'protocol':'Frozen same student/budgets, decoder-only FP16 with shared FP32 encoder/highres features cast. Same already tested seed2033 COCO384 and DAVIS30, no new independent-test claim. Actual Inductor fullgraph/dynamicFalse, originalFP32 and SAME-precision native/full-phase task controls, native dynamic token0 scored. 5GB free guard follows observed2.16GB small-batch native SAM2 quality context; no formal latency claim.'}
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(OUT/'queue_status.json')
    save()
    try:
        for name,subset in [('fresh384','coco_sam2_local_test_seed2033_v1'),('davis30','davis_val_firstframe_v1')]:
            cmd=[sys.executable,'tools/takeover_eval_sam2_local_head.py','--compiled-decoder-fp16','--minimum-free-mib','5000','--subset',str(ROOT/'assets'/subset),'--output',str(BASE/(name+'_compiled_decoderfp16.json')),'--scope','Same frozen method/cohort, precision and compiled-implementation validation; no refitting or new-independent-cohort claim']
            status.update(status='RUNNING',current_stage=name);row={'stage':name,'command':cmd,'start_unix':time.time()};status['stages'].append(row)
            with (OUT/(name+'.log')).open('x') as log:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);row['pid']=p.pid;save();code=p.wait()
            row.update(returncode=code,end_unix=time.time());save()
            if code:raise RuntimeError(name+' failed; inspect preserved partial metrics and traceback')
        status.update(status='COMPLETED',current_stage=None)
    except BaseException as e:status.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
