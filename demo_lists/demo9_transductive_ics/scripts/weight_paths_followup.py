"""CPU continuation: consume healthy extraction once, then run fixed ablation."""
import os,json,time,subprocess
from pathlib import Path
p=Path('results/weight_paths_v1/f0');status=Path('results/weight_paths_v1/queue_status.json')
def write(d):
 t=status.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(status)
while True:
 d=json.load(open(p/'cache_status.json')) if (p/'cache_status.json').exists() else {}
 write(dict(state='WAITING_HEALTHY_CACHE',pid=os.getpid(),cache_pid=d.get('pid'),progress=d.get('completed'),total=d.get('total')))
 if d.get('state')=='COMPLETED':break
 if d.get('state')=='ERROR':write(dict(state='ERROR',reason='Existing cache failed; preserve images and inspect extraction log'));raise SystemExit(1)
 time.sleep(5)
env=dict(os.environ,DEMO4_GPU_FRAC='.45',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
f=open('results/weight_paths_v1/f0_eval.log','a')
child=subprocess.Popen(['/root/miniconda3/bin/python','-u','scripts/weight_paths_eval.py','--out',str(p)],stdout=f,stderr=subprocess.STDOUT,env=env)
write(dict(state='RUNNING',pid=os.getpid(),child_pid=child.pid));code=child.wait()
write(dict(state='COMPLETED' if code==0 else 'ERROR',pid=os.getpid(),exit_code=code))
