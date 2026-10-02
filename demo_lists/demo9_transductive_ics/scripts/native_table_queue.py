#!/usr/bin/env python3
"""CPU-only supervisor: reuse completed f0, finish f1-f3 exact-cache/table serially, delete own fold cache after use.

Does not own or mutate /root/demo9_cache or any other owner's results. f1 audit is already in flight.
M1 may read the current owned cache before deletion; it is a diagnostic, not a new method table.
"""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/demo9_transductive_ics')
OUT=ROOT/'results/native_table_v1';OUT.mkdir(parents=True,exist_ok=True)
PY='/root/miniconda3/bin/python';env=dict(os.environ,PYTHONPATH='/root/demo4_cache/env',DEMO4_GPU_FRAC='.3',HF_HUB_OFFLINE='1')
status=dict(state='RUNNING',pid=os.getpid(),completed_folds=[],scope='Existing f0 reused; official paired f1-f3 main table. No new decoder/selector rule.')
def save():
    p=OUT/'queue_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(p)
def run(cmd,log):
    with open(log,'w') as h:
        r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'Child failed code {r.returncode}: {cmd}; log={log}')
save()
try:
    original=Path('/root/autodl-tmp/demo9/results/probe_decoder_f0_400.json')
    data=json.loads(original.read_text())
    if data['episodes']!=400:raise RuntimeError('f0 incomplete, cannot reuse')
    rename={'1shot':'1shot','naive':'naive','ours':'agree:tophalf','true':'true'}
    reuse=dict(state='COMPLETED_REUSED',fold=0,episodes=400,source=str(original),
               records=[dict(e=r['e'],c=r['c'],iu={b:r['iu'][a] for a,b in rename.items()}) for r in data['records']],
               miou={b:data['miou'][a] for a,b in rename.items()},note='Exact prior f0 table reused, no model/candidate rerun. none:all absent in original f0.')
    (OUT/'e1_f0.json').write_text(json.dumps(reuse));status['completed_folds']=[0];save()
    for fold in (1,2,3):
        sub=OUT/f'f{fold}';sub.mkdir(exist_ok=True)
        audit=sub/'report.json';cache=sub/'cache'/f'episodes_f{fold}_n400.pt'
        status.update(fold=fold,phase='WAIT_AUDIT' if fold==1 else 'AUDIT');save()
        if fold==1:
            while not audit.exists() or json.loads(audit.read_text()).get('state')=='RUNNING':time.sleep(10)
        elif not audit.exists():
            run([PY,'scripts/audit_paired_interface.py','--fold',str(fold),'--n','400','--keep-cache','--out',str(sub)],OUT/f'cache_f{fold}.log')
        check=json.loads(audit.read_text())
        if check['state']!='PASSED_EXACT':raise RuntimeError(f'Exact native gate failed in fold{fold}: {audit}')
        # Optional M1 diagnostics must be fully prepared before taking the current cache; no new feature dataset.
        if fold==1:
            status['phase']='M1_PREPARE';save()
            script=ROOT/'scripts/mode_information.py'
            while not script.exists():time.sleep(10)
            status['phase']='M1_PILOT';save()
            pilot=OUT/'m1_a123_f1_pilot.json'
            if not pilot.exists():run([PY,str(script),'--file',str(cache),'--limit','30','--out',str(pilot)],OUT/'m1_a123_f1_pilot.log')
        output=OUT/f'e1_f{fold}.json';status['phase']='EVALUATE';save()
        if not output.exists():run([PY,'scripts/run_native_table.py','--file',str(cache),'--out',str(output)],OUT/f'eval_f{fold}.log')
        table=json.loads(output.read_text())
        if table['episodes']!=400:raise RuntimeError(f'Incomplete evaluation fold{fold}')
        if cache.exists():
            size=cache.stat().st_size;cache.unlink()
            (sub/'cache_cleanup.json').write_text(json.dumps(dict(state='REMOVED_AFTER_USE',path=str(cache),bytes=size,
                reason='Main table completed; f1 M1 pilot completed. Own cache only; metrics/source retained.'),indent=2))
        status['completed_folds'].append(fold);save()
    status.update(state='COMPLETED',phase='PAIRED_STATS')
    # Run only CPU analysis after all native rows; f0 is reused with identical three primary anchors.
    stats=Path('/root/autodl-tmp/demo9/scripts/stats.py')
    run([PY,str(stats),*[str(OUT/f'e1_f{x}.json') for x in range(4)],'--base','1shot','--vs','naive','--B','5000'],OUT/'paired_table.txt')
    save()
except BaseException as ex:
    status.update(state='ERROR',error=repr(ex));save();raise
