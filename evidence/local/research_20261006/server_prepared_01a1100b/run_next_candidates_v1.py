from pathlib import Path
import os,json,subprocess,sys,time,traceback
ROOT=Path('/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b')
CODE=ROOT/'code_4213d9a573e8'
PYTHON='/root/miniconda3/bin/python'
METHODS=['reference_absorption','reference_gaussian_density','mean_rgb_potts']
ENV=os.environ.copy();ENV.update(PYTHONPATH='/root/demo4_cache/env',CUDA_VISIBLE_DEVICES='')
STATE=ROOT/'next_candidates_state_v1.json'
def state(name,**extra):
 data=dict(state=name,pid=os.getpid(),unix=time.time(),**extra)
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(STATE)
 print(json.dumps(data),flush=True)
def run(args,phase):
 state(phase,argv=args)
 result=subprocess.run([PYTHON,str(CODE/'scripts/run_cpu_feature_candidates.py'),*args],env=ENV)
 if result.returncode:raise RuntimeError(phase+' exited '+str(result.returncode))
def infer(manifest,out,workers,threads,phase):
 run(['infer','--backend','prepared','--prepared-methods',*METHODS,'--primary-method','reference_absorption','--manifest',str(manifest),'--root','/root','--out',str(out),'--workers',str(workers),'--threads',str(threads),'--cpu-budget','8','--memory-gb','10','--exposure','reused public600 development data; no independent confirmation; absorption candidate, Gaussian density correction version, RGB Potts simple control'],phase)
try:
 bound=ROOT/'bound600_v2';runs=ROOT/'runs'
 smoke=runs/'next_candidates_smoke4_v1'
 infer(bound/'smoke4.json',smoke,2,2,'SMOKE_INFERENCE_RUNNING')
 run(['score','--out',str(smoke)],'SMOKE_SCORING_RUNNING')
 # Wait for the root's earlier12CPU inference to release before8CPU cohort.
 state('WAITING_EARLIER_INFERENCE_RESOURCE_RELEASE')
 while True:
  earlier=json.loads((ROOT/'extensions_state_v2.json').read_text())
  if earlier['state'] in ('PUBLIC600_SCORING_RUNNING','COMPLETED'):break
  if earlier['state']=='FAILED':raise RuntimeError('Earlier root inference failed; require root resolution')
  if not Path('/proc',str(earlier['pid'])).exists():raise RuntimeError('Earlier coordinator exited without resource-release state')
  time.sleep(5)
 # Predict/seal first; no method selection from smoke scores.
 cohort=runs/'next_candidates_public600_v1'
 infer(bound/'rows.json',cohort,4,2,'PUBLIC600_INFERENCE_RUNNING')
 run(['score','--out',str(cohort)],'PUBLIC600_SCORING_RUNNING')
 state('COMPLETED',out=str(cohort),new_independent_mechanisms=1,correction_versions=1,strong_simple_controls=1)
except BaseException as error:
 state('FAILED',error_type=type(error).__name__,error=str(error))
 traceback.print_exc();sys.exit(1)
