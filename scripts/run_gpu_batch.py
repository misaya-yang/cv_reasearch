#!/usr/bin/env python3
"""Bounded complete comparison; called only inside the experiment resource guard.

Dense cache algebra and one shared-backbone C/D worker use GPU sequentially.
CPU evaluates sealed cache predictions concurrently with the C/D GPU worker.
No package installation, model download, GPU activation or shutdown occurs here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def quota():
    def read(p):
        try:return Path(p).read_text().strip()
        except OSError:return None
    raw=read('/sys/fs/cgroup/cpu.max');memory=read('/sys/fs/cgroup/memory.max')
    cpus=None
    if raw:
        q,p=raw.split();cpus=(int(q)/int(p)) if q!='max' else os.cpu_count()
    return {'cpu_quota':cpus,'memory_bytes':int(memory) if memory and memory!='max' else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    if a.out.exists():raise FileExistsError('Use a fresh run directory')
    resources=quota()
    if resources['cpu_quota'] is not None and resources['cpu_quota']<2:
        raise RuntimeError('Effective CPU quota is below 2: no-card experiment execution refused')
    if resources['memory_bytes'] is not None and resources['memory_bytes']<8*1024**3:
        raise RuntimeError('Effective memory is below 8 GiB: no-card execution refused')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv,noheader'],text=True,capture_output=True,check=True)
    foreign=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,capture_output=True,check=True)
    if foreign.stdout.strip():raise RuntimeError('Existing GPU process: defer without killing it')
    a.out.mkdir(parents=True)
    (a.out/'resources.json').write_text(json.dumps(dict(resources,gpus=gpu.stdout,initial_gpu_pids=[]),indent=2)+'\n')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',DEMO4_GPU_FRAC='0.85')
    env['PYTHONPATH']=':'.join(['/root/demo4_cache/env','/root/autodl-tmp/demo8_local_verification/crf_source/src','/root/autodl-tmp/demo8_local_verification/runtime/extensions',env.get('PYTHONPATH','')])
    cache_root='/root/autodl-tmp/demo9_extent'
    started=time.monotonic()
    def run(script,args):subprocess.run([sys.executable,str(root/'scripts'/script),*map(str,args)],cwd=root,env=env,check=True)
    run('run_mechanisms.py',['infer','--root',cache_root,'--manifest',a.manifest,'--out',a.out/'cache',
                            '--device','cuda','--threads','2','--methods','rcg','transport','structure','latent','reconstruction'])
    # The cache process has exited and owns no model. Only the next process builds DINOv3.
    log=(a.out/'cache_score.log').open('w')
    score=subprocess.Popen([sys.executable,str(root/'scripts/run_mechanisms.py'),'evaluate','--root',cache_root,'--out',str(a.out/'cache')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        run('run_intervention.py',['--manifest',a.manifest,'--out',a.out/'forward','--device','cuda','--include-multilayer'])
        rc=score.wait(timeout=60)
        if rc:raise RuntimeError(f'Cache scoring failed with exit {rc}; see cache_score.log')
    finally:
        if score.poll() is None:
            score.terminate();score.wait(timeout=10)
        log.close()
    run('score_forward_run.py',['--out',a.out/'forward','--cache-root',cache_root,'--rcg-run',a.out/'cache'])
    (a.out/'completion.json').write_text(json.dumps({'state':'COMPLETED','elapsed_seconds':time.monotonic()-started,
        'cache_report':str(a.out/'cache/report.json'),'forward_report':str(a.out/'forward/report.json'),
        'decision':'Inspect errors, batch/fold effects and controls before choosing another cohort; no automatic continuation.'},indent=2)+'\n')

if __name__=='__main__':main()
