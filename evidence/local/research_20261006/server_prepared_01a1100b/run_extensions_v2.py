from pathlib import Path
import os,json,subprocess,sys,time,traceback
ROOT=Path('/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b')
CODE=ROOT/'code_4a75b20afeff'
PYTHON='/root/miniconda3/bin/python'
METHODS=['pro_reference_relations','reference_hull','constellation_local','reference_triplet_relations']
ENV=os.environ.copy();ENV.update(PYTHONPATH='/root/demo4_cache/env',CUDA_VISIBLE_DEVICES='')
STATE=ROOT/'extensions_state_v2.json'
def state(name,**extra):
 data=dict(state=name,pid=os.getpid(),unix=time.time(),**extra)
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(STATE)
 print(json.dumps(data),flush=True)
def run(args,phase):
 state(phase,argv=args)
 result=subprocess.run([PYTHON,str(CODE/'scripts/run_cpu_feature_candidates.py'),*args],env=ENV)
 if result.returncode:raise RuntimeError(phase+' exited '+str(result.returncode))
def infer(manifest,out,workers,threads,phase):
 run(['infer','--backend','prepared','--prepared-methods',*METHODS,'--primary-method','pro_reference_relations','--manifest',str(manifest),'--root','/root','--out',str(out),'--workers',str(workers),'--threads',str(threads),'--cpu-budget','12','--memory-gb','12','--exposure','reused public600 development data; no independent confirmation; ProM2 cached seed variant, local constellation correction version'],phase)
try:
 bound=ROOT/'bound600_v2';runs=ROOT/'runs'
 smoke=runs/'extensions_smoke4_v2'
 infer(bound/'smoke4.json',smoke,2,2,'SMOKE_INFERENCE_RUNNING')
 run(['score','--out',str(smoke)],'SMOKE_SCORING_RUNNING')
 # Predict/seal first; no method selection from smoke scores.
 cohort=runs/'extensions_public600_v2'
 infer(bound/'rows.json',cohort,6,2,'PUBLIC600_INFERENCE_RUNNING')
 run(['score','--out',str(cohort)],'PUBLIC600_SCORING_RUNNING')
 state('COMPLETED',out=str(cohort),new_independent_mechanisms=3,correction_versions=1)
except BaseException as error:
 state('FAILED',error_type=type(error).__name__,error=str(error))
 traceback.print_exc();sys.exit(1)
