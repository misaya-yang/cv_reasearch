"""CPU-only predecessor wait avoids mutual CUDA exclusivity waits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/takeover_20261001_v1'
OUT=BASE/'sam2_local_student_v1/formal_followup'


def main():
    OUT.mkdir(exist_ok=False)
    state={'status':'WAITING_PREDECESSORS_CPU_ONLY','pid':os.getpid()}
    def save():
        tmp=OUT/'queue_status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUT/'queue_status.json')
    save()
    try:
        paths=[BASE/'local_student_head_v1/phase_followup_v1/queue_status.json',BASE/'sam2_local_student_v1/fp16_quality_v1/queue_status.json']
        while True:
            statuses=[json.loads(p.read_text()).get('status') if p.exists() else 'NOT_CREATED' for p in paths]
            if state.get('predecessors')!=statuses:state['predecessors']=statuses;save()
            if all(s in ('COMPLETED','ERROR') for s in statuses):break
            time.sleep(10)
        cmd=[sys.executable,'tools/takeover_sam2_compact_perf.py','--exclusive','--microbatch','128','--output',str(BASE/'sam2_local_student_v1/perf_p128_mb128_formal')]
        with (OUT/'timing.log').open('x') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);state.update(status='RUNNING',command=cmd,child_pid=p.pid);save();code=p.wait()
        state.update(status='COMPLETED' if code==0 else 'ERROR',returncode=code)
    except BaseException as e:state.update(status='ERROR',error=repr(e));raise
    finally:save()


if __name__=='__main__':main()
