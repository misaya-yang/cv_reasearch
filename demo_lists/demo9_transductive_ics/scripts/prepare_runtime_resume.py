#!/usr/bin/env python3
"""CPU-only resume from a failed interface; preserve completed scientific stages."""
import argparse
import json
import re
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from experiment_resource_guard import ResourceGuard,sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-dir',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--state-file',required=True,type=Path)
    p.add_argument('--from-plan',type=Path)
    p.add_argument('--from-state',type=Path)
    p.add_argument('--tag',default='resume1')
    p.add_argument('--prior-tag',default='')
    p.add_argument('--protected-resume-checkpoint',type=Path)
    p.add_argument('--independent-first',action='store_true')
    p.add_argument('--metric-baseline-manifest',type=Path)
    a=p.parse_args(); runtime=a.runtime_dir.resolve(); root=HERE.parent
    runtime.relative_to(root)
    if not re.fullmatch(r'resume[1-9][0-9]*',a.tag) or (a.prior_tag and not re.fullmatch(r'resume[1-9][0-9]*',a.prior_tag)):
        raise ValueError('Explicit numbered resume tags required')
    from_plan=a.from_plan or runtime/'queue_plan.json'
    from_state=a.from_state or runtime/'queue_guard.json'
    old=json.loads(from_plan.read_text());state=json.loads(from_state.read_text())
    completed={v['stage'] for v in state['events'] if v.get('state')=='STAGE_COMPLETED'}
    if not completed or any(v.get('state')=='QUEUE_COMPLETED_NO_RUNNABLE_NEXT_STAGE' for v in state['events']):
        raise ValueError('Explicit incomplete/failure queue required, not a completed replay')
    if a.out.exists():raise ValueError('Fresh resume plan required')
    code=set(root.glob('scripts/*.py'))|set(root.glob('tics/*.py'))
    template=json.loads((runtime/'source_template.json').read_text())
    for source in code:template['files'][str(source.resolve())]=sha256_file(source)
    template['files'][str(runtime/'positional_basis.pt')]=sha256_file(runtime/'positional_basis.pt')
    template['resume_reason']='Preserve native basis; correct source-native structural validation and ENOEXEC helper invocation'
    source_guard=runtime/('source_guard_'+a.tag+'.json')
    if source_guard.exists():raise ValueError('Preserve existing source receipt')
    source_guard.write_text(json.dumps(template,indent=2)+'\n')
    suffix=('_'+a.prior_tag) if a.prior_tag else ''
    replacements={str(runtime/('experiments'+suffix)):str(runtime/('experiments_'+a.tag)),
        str(runtime/('source_guard'+suffix+'.json')):str(source_guard),str(from_state.resolve()):str(a.state_file.resolve()),
        str(runtime/('tensor_cleanup'+suffix+'.json')):str(runtime/('tensor_cleanup_'+a.tag+'.json'))}
    def replace(value):
        if isinstance(value,str):
            for old_value,new in replacements.items():value=value.replace(old_value,new)
            return value
        if isinstance(value,list):return [replace(v) for v in value]
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        return value
    stages=[replace(s) for s in old['stages'] if s['name'] not in completed]
    if a.metric_baseline_manifest:
        manifest=a.metric_baseline_manifest.resolve()
        if json.loads(manifest.read_text()).get('episode_protocol')!='frozen_metric_inference_cohort':raise ValueError('Explicit matched metric cohort required')
        output=runtime/('experiments_'+a.tag)/'E3_foris_crf'/'report.json'
        parent=next(s for s in old['stages'] if s['kind']=='gpu')
        stage=dict(name='E3_foris_crf_matched',kind='gpu',cwd=str(root),timeout_seconds=450,
            argv=[parent['argv'][0],str(HERE/'native_baseline_experiment.py'),'--prepared-root','/root/autodl-tmp/demo9',
                '--prepared-manifest',str(manifest),'--episode-mode','explicit_metric_cohort','--host','foris',
                '--refiner','crf','--projection-basis',str(runtime/'positional_basis.pt'),'--limit','10','--out',str(output.parent)],
            env=parent['env'],requires=[dict(path=str(manifest),sha256=sha256_file(manifest)),
                dict(path=str(runtime/'positional_basis.pt')),dict(path=str(source_guard),json_equals=dict(state='PREPARED_SOURCE'))],
            produces=[dict(path=str(output),json_equals=dict(state='COMPLETED'))],
            success_checks=[dict(path=str(output),json_equals=dict(state='COMPLETED'))])
        stages.insert(0,stage)
        final=next(s for s in stages if s['name']=='final_frozen_analysis')
        final['argv'].insert(2,str(output))
        final['requires'].append(dict(path=str(output),json_equals=dict(state='COMPLETED')))
        metric=runtime/'metric_acquisition/evaluation.json'
        joined=runtime/'metric_acquisition/evaluation_with_foris.json'
        parent_analysis=next(s for s in stages if s['name']=='E3_analysis')
        join_stage=dict(name='E3_join_strong_control',kind='cpu',role='handoff',cwd=str(root),timeout_seconds=60,
            argv=[parent['argv'][0],str(HERE/'join_metric_strong_control.py'),'--metric',str(metric),
                  '--strong',str(output),'--out',str(joined)],env=parent_analysis['env'],
            requires=[dict(path=str(metric),json_equals=dict(state='COMPLETED')),dict(path=str(output),json_equals=dict(state='COMPLETED'))],
            produces=[dict(path=str(joined),json_equals=dict(state='COMPLETED'))])
        stages.insert(stages.index(parent_analysis),join_stage)
        parent_analysis['argv']=[str(joined) if v==str(metric) else v for v in parent_analysis['argv']]
        parent_analysis['argv'][2:2]=['--base','foris_crf','--base','insid3_native','--base','native_kde']
        for check in parent_analysis['requires']:
            if check['path']==str(metric):check['path']=str(joined)
    if a.independent_first:
        # Ready independent methods are not subordinated to a numerical
        # training recovery. Preserve their own internal analysis order.
        ahead=[s for s in stages if s['name'].split('_')[0] in ('E5','E6','E9','E10')]
        stages=ahead+[s for s in stages if s not in ahead]
    if a.protected_resume_checkpoint:
        checkpoint=a.protected_resume_checkpoint.resolve();checkpoint.relative_to(root)
        if not checkpoint.is_file():raise ValueError('Explicit completed-epoch checkpoint required')
        candidates=[s for s in stages if s['name']=='E3_train_protected']
        if len(candidates)!=1:raise ValueError('Resume only the unfinished protected training stage')
        candidates[0]['argv'].extend(['--resume',str(checkpoint)])
        candidates[0]['requires'].append(dict(path=str(checkpoint),sha256=sha256_file(checkpoint)))
    # Completed reports remain the original artifacts; they are not replayed
    # into a new output directory just to satisfy a final combined analysis.
    for stage in stages:
        if stage['name']=='final_frozen_analysis':
            for original in old['stages']:
                if original['name'] in completed:
                    for artifact in original.get('produces',[]):
                        original_path=artifact['path']
                        changed=replace(original_path)
                        stage['argv']=[original_path if v==changed else v for v in stage['argv']]
                        for v in stage.get('requires',[]):
                            if v['path']==changed:v['path']=original_path
    static=old['stages'][0].get('cpu_artifacts',[])
    for value in static:
        file=Path(value['path']).resolve()
        if str(file) in template['files']:value['sha256']=template['files'][str(file)]
    known={v['path'] for v in static}
    static.extend(dict(path=name,sha256=sha) for name,sha in template['files'].items() if name not in known)
    stages[0]['cpu_artifacts']=static
    stages[0]['code_files']=[name for name in template['files'] if name.endswith('.py')]
    old.update(stages=stages,resume_from_completed=sorted(completed),
        preserved_failed_report=str(runtime/'experiments/E8_foris_bilinear/report.json'),
        sum_stage_timeout_seconds=sum(s['timeout_seconds'] for s in stages))
    a.out.write_text(json.dumps(old,indent=2)+'\n')
    def forbidden():raise AssertionError('No GPU or shutdown permitted in preparation')
    guard=ResourceGuard(old,a.state_file,inventory=forbidden,poweroff=forbidden)
    if not guard.build_preflight():raise SystemExit(2)
    print(json.dumps(dict(state='CPU_RESUME_PREFLIGHT_COMPLETED',stages=len(stages),preserved_completed=sorted(completed))))


if __name__=='__main__':main()
