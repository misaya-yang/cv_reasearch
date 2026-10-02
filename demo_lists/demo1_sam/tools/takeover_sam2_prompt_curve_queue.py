"""Find useful prompt counts and cache amortization after the P128 comparison."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/takeover_20261001_v1/sam2_local_student_v1'
OUT=BASE/'prompt_curve_v1'


def main():
    OUT.mkdir(exist_ok=False)
    status={'status':'WAITING_PREDECESSOR_CPU_ONLY','pid':os.getpid(),'stages':[],
            'protocol':'Same frozen teacher/student, unchanged128 evidence. New P1/8/32/1024 genuine distinct image-point workloads, common MB=min(P,128), all4mask/IQ/masktoken/objscore and same seven precision controls. Determine warm decoder and per-image static cache amortization limits, not manufacture load. Independent forward/reverse formal rounds; encoder/postprocess remain excluded, no end-to-end claim.'}
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(OUT/'queue_status.json')
    save()
    try:
        while True:
            p=BASE/'formal_followup/queue_status.json'
            prior=json.loads(p.read_text())['status'] if p.exists() else 'NOT_CREATED'
            if prior in ('COMPLETED','ERROR'):break
            time.sleep(10)
        for prompts in (1,8,32,1024):
            cmd=[sys.executable,'tools/takeover_sam2_compact_perf.py','--exclusive','--prompts',str(prompts),'--microbatch',str(min(prompts,128)),'--output',str(OUT/('p'+str(prompts)))]
            status.update(status='RUNNING',current_prompts=prompts);row={'prompts':prompts,'command':cmd,'start_unix':time.time()};status['stages'].append(row)
            with (OUT/('p'+str(prompts)+'.log')).open('x') as log:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);row['pid']=p.pid;save();code=p.wait()
            row.update(returncode=code,end_unix=time.time());save()
            if code:raise RuntimeError('Prompt count '+str(prompts)+' failed; inspect preserved report')
        status.update(status='COMPLETED',current_prompts=None)
    except BaseException as e:status.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
