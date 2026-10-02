#!/usr/bin/env python3
"""Prepare one fixed strong-host stress control on CPU; no GPU orchestration."""
import argparse
import copy
import json
from pathlib import Path
from experiment_resource_guard import ResourceGuard,sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-dir',required=True,type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parent.parent;runtime=a.runtime_dir.resolve()
    runtime.relative_to(root)
    old=json.loads((runtime/'queue_resume2_plan.json').read_text())
    stage=copy.deepcopy(next(s for s in old['stages'] if s['name']=='E3_foris_crf_matched'))
    output=runtime/'brightness_strong_v1';output.mkdir(exist_ok=True)
    plan=output/'queue_plan.json';state=output/'queue_guard.json'
    if plan.exists():raise ValueError('Do not rewrite a frozen plan')
    base=runtime/'experiments_resume1/E8_foris_crf/report.json'
    baseline=json.loads(base.read_text());manifest=json.loads((runtime/'manifest10.json').read_text())
    task=lambda r:(r['e'],r['c'],r['support'],r['query'])
    if baseline['state']!='COMPLETED' or [task(r) for r in baseline['records']]!=[task(r) for r in manifest['frozen_episodes']]:
        raise ValueError('Completed identity control must match all ten fixed roles')
    py=old['cuda_python'];scripts=root/'scripts'
    stage.update(name='E10_FoRIS_fixed_brightness',timeout_seconds=300,
        argv=[py,str(scripts/'native_baseline_experiment.py'),'--prepared-root','/root/autodl-tmp/demo9',
              '--prepared-manifest',str(runtime/'manifest10.json'),'--host','foris','--refiner','crf',
              '--projection-basis',str(runtime/'positional_basis.pt'),'--query-brightness','.75',
              '--limit','10','--out',str(output/'changed')],
        requires=[dict(path=str(runtime/'manifest10.json'),sha256=sha256_file(runtime/'manifest10.json')),
                  dict(path=str(runtime/'positional_basis.pt'),sha256=sha256_file(runtime/'positional_basis.pt'))],
        produces=[dict(path=str(output/'changed/report.json'),json_equals=dict(state='COMPLETED'))],
        success_checks=[dict(path=str(output/'changed/report.json'),json_equals=dict(state='COMPLETED'))])
    files=sorted(list(scripts.glob('*.py'))+list((root/'tics').glob('*.py')))
    stage['code_files']=[str(x) for x in files]
    stage['cpu_artifacts']=[dict(path=str(x),sha256=sha256_file(x)) for x in files]
    for x in manifest['assets']:
        path=Path(x['path']);stat=path.stat()
        if stat.st_size!=x['size'] or stat.st_mtime_ns!=x['mtime_ns']:raise ValueError('Prepared input changed')
        stage['cpu_artifacts'].append(dict(path=str(path),size=x['size']))
    joined=output/'joined.json';analysis=output/'analysis.json'
    handoff=dict(kind='cpu',role='handoff',cwd=str(root),timeout_seconds=60,env=stage['env'])
    join=dict(handoff,name='E10_exact_task_join',
        argv=[py,str(scripts/'join_brightness_control.py'),'--base',str(base),'--changed',str(output/'changed/report.json'),'--out',str(joined)],
        requires=[dict(path=str(base),sha256=sha256_file(base)),dict(path=str(output/'changed/report.json'),json_equals=dict(state='COMPLETED'))],
        produces=[dict(path=str(joined),json_equals=dict(state='COMPLETED'))])
    analyze=dict(handoff,name='E10_paired_strong_analysis',
        argv=[py,str(scripts/'analyze_native_experiments.py'),str(joined),'--base','foris_crf','--out',str(analysis)],
        requires=[dict(path=str(joined),json_equals=dict(state='COMPLETED'))],
        produces=[dict(path=str(analysis),json_equals=dict(state='CPU_FROZEN_OUTPUT_ANALYSIS'))])
    queue={k:v for k,v in old.items() if k not in ('stages','resume_from_completed','preserved_failed_report')}
    queue.update(stages=[stage,join,analyze],sum_stage_timeout_seconds=420,
        scope='One fixed strong-host robustness control, not brightness tuning or another candidate algorithm')
    plan.write_text(json.dumps(queue,indent=2)+'\n')
    def forbidden():raise AssertionError('CPU preparation cannot probe GPU or power off')
    guard=ResourceGuard(queue,state,inventory=forbidden,poweroff=forbidden)
    if not guard.build_preflight():raise SystemExit(2)
    print(json.dumps(dict(state='CPU_PREFLIGHT_COMPLETED',stages=3,predictions=10,identity_predictions_reused=True)))


if __name__=='__main__':main()
