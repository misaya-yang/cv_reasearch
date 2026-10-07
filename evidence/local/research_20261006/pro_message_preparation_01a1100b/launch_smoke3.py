"""Only three assigned smoke workers, aggregate RSS16GiB/4 threads each."""
from pathlib import Path
import json,subprocess,os,time,signal


def main():
    root=Path('/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b')
    run=root/'runs/pro_message_audit_v1';batch=run/'smoke4_fixed_v1'
    batch.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(PYTHONPATH='/root/demo4_cache/env',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                                   OPENBLAS_NUM_THREADS='4',OMP_MAX_ACTIVE_LEVELS='1',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    children=[];started=time.perf_counter();peak=0
    for index in (1,2,3):
        cmd=['/root/miniconda3/bin/python','-u',str(run/'smoke_case_worker.py'),'--root',str(root),
             '--code',str(run/'code_b9da3fd3b42c'),'--out',str(batch/f'case{index}'),'--case-index',str(index)]
        child=subprocess.Popen(cmd,stdout=(batch/f'case{index}.log').open('wb'),stderr=subprocess.STDOUT,
                               env=env,start_new_session=True)
        children.append((index,child,cmd))
    records={'children':[{'index':i,'pid':p.pid,'command':c} for i,p,c in children],
             'compute_threads':12,'group_RSS_limit_bytes':16<<30,'worker_RSS_limit_bytes':8<<30}
    (batch/'launch.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records),flush=True)
    while True:
        rss=0;states=[]
        for i,p,c in children:
            if p.poll() is None:
                status=Path(f'/proc/{p.pid}/status')
                if status.exists():
                    rss+=next(int(x.split()[1])*1024 for x in status.read_text().splitlines() if x.startswith('VmRSS:'))
            states.append({'index':i,'pid':p.pid,'exit_code':p.poll()})
        peak=max(peak,rss)
        state={'state':'RUNNING_THREE_ASSIGNED_CASES','elapsed_seconds':time.perf_counter()-started,
               'current_group_RSS_bytes':rss,'peak_group_RSS_bytes':peak,'children':states}
        if rss>16<<30:
            state['state']='GROUP_MEMORY_LIMIT_STOP'
            for _,p,_ in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            (batch/'group_state.json').write_text(json.dumps(state,indent=2)+'\n');return
        if all(p.poll() is not None for _,p,_ in children):
            state['state']='ASSIGNED_CASES_EXITED'
            (batch/'group_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True);return
        (batch/'group_state.json').write_text(json.dumps(state,indent=2)+'\n')
        time.sleep(1)


if __name__=='__main__':main()
