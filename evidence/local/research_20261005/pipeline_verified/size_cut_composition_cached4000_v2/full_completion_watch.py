"""One CPU-only consumer for the already-authorized complete4000 producer."""
import json, os, subprocess, time
from pathlib import Path
ROOT=Path("/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9")
LAUNCH=ROOT/"launch/size_cut_composition_cached4000_v2"
CANDIDATE=ROOT/"outputs/uniform_tau15_sizecut4000_v2"
MEAN=ROOT/"outputs/fine_mean1200_v1"
OUT=ROOT/"outputs/size_cut_full4000_physical_verification_v1"
STATE=LAUNCH/"full_completion_watch_state.json"
env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONPATH="/root/demo4_cache/env")
def write(v):STATE.write_text(json.dumps(v,indent=2)+"\n")
write(dict(state="WAITING_FOR_COMPLETE_PRIMARY_AND_MEAN_SCORES",pid=os.getpid(),GPU=False))
while True:
    if (CANDIDATE/"score_state.json").exists() and (CANDIDATE/"sealed.json").exists() and (MEAN/"report.json").exists():break
    time.sleep(60)
write(dict(state="CPU_PHYSICAL_VERIFICATION_RUNNING",pid=os.getpid(),GPU=False))
cmd=["/root/miniconda3/bin/python","-u",str(LAUNCH/"verify_full_source.py"),"--candidate",str(CANDIDATE),"--source",str(ROOT/"outputs/frozen_subtoken4000_v1"),"--cached",str(ROOT/"outputs/size_cut_composition_cached4000_v2"),"--prior",str(ROOT/"outputs/frozen_subtoken4000_scored_v1"),"--out",str(OUT)]
r=subprocess.run(cmd,env=env)
write(dict(state="COMPLETE_PRIMARY_PHYSICAL_VERIFIED_READY_FOR_STATISTICS" if r.returncode==0 else "FAILED_PHYSICAL_VERIFICATION",pid=os.getpid(),exit_code=r.returncode,GPU=False,output=str(OUT)))
