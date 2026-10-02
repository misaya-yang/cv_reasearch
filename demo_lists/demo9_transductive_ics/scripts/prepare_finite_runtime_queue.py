#!/usr/bin/env python3
"""CPU-only preparation of the explicitly authorized finite E1-E8 runtime queue.

No default execution/rental/CUDA/browser/download. All native assets and source
must already have CPU receipts. E8 creates only the lawful shared native BLACK
basis; source finalization and E3 deployment exports are <=60s CPU handoffs.
These are resource dependencies, never algorithm pass/fail dependencies.
Each completed scientific negative hands off to the next ready independent
algorithm; implementation errors stop via experiment_resource_guard.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

from experiment_resource_guard import ResourceGuard,sha256_file,validate_plan

HERE=Path(__file__).resolve().parent
VARIANTS=('protected','fixed_global','unprotected')


def owned(path,root):
    path=Path(path).resolve()
    try:path.relative_to(root)
    except ValueError:raise ValueError('Runtime output must be within owned project root: '+str(path))
    return path


def artifact(path,state=None,sha=None):
    value=dict(path=str(path))
    if state is not None:value['json_equals']=dict(state=state)
    if sha is not None:value['sha256']=sha
    return value


def verify_metadata(records):
    for row in records:
        value=Path(row['path']).stat()
        if value.st_size!=row['size'] or value.st_mtime_ns!=row['mtime_ns']:
            raise ValueError('Prepared source/asset metadata drift: '+row['path'])


def read_inputs(a):
    root=a.project_root.resolve();manifest=json.loads(a.manifest.read_text());metric=json.loads(a.metric_plan.read_text())
    template=json.loads(a.source_template.read_text())
    if manifest.get('state')!='PREPARED_ASSETS' or manifest.get('seed')!=0 or manifest.get('limit')!=10:
        raise ValueError('Exactly ten frozen seed0 native episodes must be CPU-prepared')
    if len(manifest.get('frozen_episodes',[]))!=10:raise ValueError('Native manifest count mismatch')
    if metric.get('schema')!='demo9_native_metric_acquisition_v1' or metric.get('state')!='PREPARED_CPU_METADATA':
        raise ValueError('Explicit CPU-prepared E3 acquisition plan required')
    if {s:len(metric.get('rows',{}).get(s,[])) for s in ('train','development','inference')}!={'train':24,'development':8,'inference':10}:
        raise ValueError('This frozen fit pilot requires E3 train24/dev8/inference10')
    if not metric.get('isolation',{}).get('class_disjoint') or not metric.get('isolation',{}).get('all_role_photo_disjoint'):
        raise ValueError('E3 class/all-role-photo isolation must already be verified')
    if metric.get('seed')!=0 or metric.get('fold')!=manifest.get('fold'):raise ValueError('Frozen seed/fold mismatch')
    if template.get('state')!='PREPARED_SOURCE' or not template.get('files'):raise ValueError('Existing static source template required')
    for path,sha in template['files'].items():
        if not Path(path).is_absolute() or sha256_file(path)!=sha:raise ValueError('Static source snapshot differs: '+path)
    template={**template,'files':{str(Path(path).resolve()):sha for path,sha in template['files'].items()}}
    verify_metadata(manifest['assets']);verify_metadata(metric['assets'])
    output=owned(metric['output_dir'],root);owned(metric['tensor_root'],root)
    basis=owned(metric['fixed_basis']['path'],root)
    if metric['fixed_basis'].get('source')!='original_INSID3_normalized_torch_zeros_black_image_FP32_positional_basis':
        raise ValueError('E3 basis must be actual original normalized-black source')
    if basis.exists():raise ValueError('Fresh queue must create a new basis; preserve old evidence')
    return root,manifest,metric,template,output,basis


def build_plan(a):
    root,manifest,metric,template,metric_out,basis=read_inputs(a)
    runtime=owned(a.runtime_dir,root);output=owned(a.output_root,root);guard=owned(a.guard_state,root)
    source_guard=runtime/'source_guard.json'
    if any(p.exists() and any(p.iterdir()) for p in (output,metric_out)):
        raise ValueError('Use fresh algorithm and E3 output directories')
    env=dict(PYTHONPATH=a.pythonpath,DEMO4_GPU_FRAC='.3',DEMO9_CUDA_GUARD='1',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    stages=[];reports=[]
    manifest_input=artifact(a.manifest.resolve(),'PREPARED_ASSETS',sha256_file(a.manifest))
    source_input=artifact(a.source_template.resolve(),'PREPARED_SOURCE',sha256_file(a.source_template))
    basis_input=artifact(basis);guard_input=artifact(source_guard,'PREPARED_SOURCE')
    def script(name):
        path=root/'scripts'/name
        if not path.is_file():raise ValueError('Production executable source not yet ready: '+str(path))
        compile(path.read_text(),str(path),'exec')
        if str(path.resolve()) not in template['files']:raise ValueError('Production script missing frozen source template: '+str(path))
        return str(path)
    def stage(name,program,args,requires,produces,timeout,kind='gpu',report=None):
        value=dict(name=name,kind=kind,argv=[str(a.python),script(program),*map(str,args)],cwd=str(root),
            timeout_seconds=timeout,requires=requires,produces=produces,env=dict(env),code_files=[])
        if kind=='cpu':value.update(role='handoff',env={**env,'CUDA_VISIBLE_DEVICES':''})
        if report:value['success_checks']=[artifact(report,'COMPLETED')]
        stages.append(value)
    def analysis(card,report):
        target=output/(card+'_analysis.json')
        stage(card+'_analysis','analyze_native_experiments.py',[report,'--B','200','--out',target],
            [artifact(report,'COMPLETED')],[artifact(target,'CPU_FROZEN_OUTPUT_ANALYSIS')],60,'cpu')
        reports.append(report)
    baseargs=['--prepared-root',a.prepared_root,'--prepared-manifest',a.manifest,'--limit','10']
    # Basis is produced by actual unchanged INSID3, not synthetic preparation.
    setup_name=metric['fixed_basis']['producer'];report=output/'E8_insid3_bilinear'/'report.json'
    stage(setup_name,'native_baseline_experiment.py',[*baseargs,'--host','insid3','--refiner','bilinear',
        '--save-basis',basis,'--out',report.parent],[manifest_input,source_input],
        [artifact(report,'COMPLETED'),basis_input],a.baseline_timeout,report=report)
    # ALL static source/assets are verified by guard before first CUDA readiness.
    initial={str(Path(x['path']).resolve()):artifact(Path(x['path']).resolve()) for x in manifest['assets']+metric['assets']}
    initial.update({path:artifact(path,sha=sha) for path,sha in template['files'].items()})
    stages[0]['cpu_artifacts']=list(initial.values())
    stages[0]['code_files']=[path for path in template['files'] if path.endswith('.py')]
    analysis('E8_insid3_bilinear',report)
    stage('finalize_runtime_sources','finalize_runtime_sources.py',
        ['--template',a.source_template,'--basis',basis,'--out',source_guard],
        [source_input,basis_input],[guard_input],60,'cpu')
    # Released FoRIS strong reference and its bilinear companion; INSID3 CRF optional.
    baselines=[('foris','bilinear'),('foris','crf')]
    if a.insid3_crf:baselines.append(('insid3','crf'))
    for host,refiner in baselines:
        name='E8_'+host+'_'+refiner;report=output/name/'report.json'
        stage(name,'native_baseline_experiment.py',[*baseargs,'--host',host,'--refiner',refiner,
            '--foris-root',a.foris_root,'--projection-basis',basis,'--out',report.parent],
            [manifest_input,basis_input,guard_input],[artifact(report,'COMPLETED')],a.baseline_timeout,report=report)
        analysis(name,report)
    shared=[manifest_input,basis_input,guard_input]
    for card,arms in [('E1','native_raw,local_pg,kernel_pad'),('E2','native_raw,full_rewrite,full_kernel_pad')]:
        report=output/card/'report.json'
        stage(card,'native_readout_probe.py',[*baseargs,'--fold',manifest['fold'],'--projection-basis',basis,
            '--arms',arms,'--out',report.parent],shared,[artifact(report,'COMPLETED')],a.method_timeout,report=report)
        analysis(card,report)
    for card,method in [('E4','background'),('E7','lattice')]:
        report=output/card/'report.json'
        stage(card,'native_view_experiment.py',[*baseargs,'--method',method,'--fold',manifest['fold'],
            '--projection-basis',basis,'--allow-gpu','--gpu-owner',str(guard),'--out',report.parent],
            shared,[artifact(report,'COMPLETED')],a.method_timeout,report=report)
        analysis(card,report)
    # E3 asset acquisition has no scientific pass gate on E1/E2.
    acquisition=metric_out/'acquisition_report.json';training_manifest=metric_out/'training_manifest.json'
    config=metric_out/'metric_config.json';cohort=metric_out/'inference_cohort.json'
    stage('E3_acquire','native_metric_acquire.py',['--acquire','--prepared-plan',a.metric_plan,
        '--allow-gpu','--resource-guard-state',guard],
        [artifact(a.metric_plan.resolve(),'PREPARED_CPU_METADATA',sha256_file(a.metric_plan)),basis_input,guard_input],
        [artifact(acquisition,'COMPLETED'),artifact(training_manifest),artifact(config),artifact(cohort)],a.acquisition_timeout,report=acquisition)
    checkpoints=[]
    for variant in VARIANTS:
        arm=metric_out/variant;report=arm/'report.json';best=arm/'best.pt';last=arm/'last.pt';frozen=metric_out/(variant+'_frozen.pt')
        stage('E3_train_'+variant,'train_reference_metric.py',['--manifest',training_manifest,'--output-dir',arm,
            '--variant',variant,'--metric-config',config,'--epochs',a.metric_epochs,'--patience','0','--device','cuda','--allow-gpu'],
            [artifact(training_manifest),artifact(config),artifact(acquisition,'COMPLETED')],
            [artifact(report,'COMPLETED'),artifact(best),artifact(last)],a.training_timeout,report=report)
        stage('E3_export_'+variant,'train_reference_metric.py',['--export',best,'--deployment-out',frozen],
            [artifact(best),artifact(report,'COMPLETED')],[artifact(frozen)],60,'cpu')
        checkpoints.extend(['--checkpoint',variant+'='+str(frozen)])
    report=metric_out/'evaluation.json'
    stage('E3_evaluate','native_metric_cohort_experiment.py',['--cohort',cohort,*checkpoints,
        '--device','cuda','--allow-gpu','--resource-guard-state',guard,'--out',report],
        [artifact(cohort),*[artifact(metric_out/(v+'_frozen.pt')) for v in VARIANTS]],
        [artifact(report,'COMPLETED')],a.metric_evaluation_timeout,report=report)
    analysis('E3',report)
    report=output/'E5'/'report.json'
    stage('E5','native_prompt_experiment.py',[*baseargs,'--fold',manifest['fold'],'--projection-basis',basis,
        '--steps','20','--checkpoint-blocks','on','--out',report.parent],
        shared,[artifact(report,'COMPLETED')],a.prompt_timeout,report=report)
    analysis('E5',report)
    report=output/'E6'/'report.json'
    stage('E6','native_transport_experiment.py',[*baseargs,'--fold',manifest['fold'],'--projection-basis',basis,
        '--source-guard',source_guard,'--allow-gpu','--resource-guard-state',guard,'--max-iterations','200',
        '--balanced-max-iterations','2000','--out',report.parent],shared,[artifact(report,'COMPLETED')],a.transport_timeout,report=report)
    analysis('E6',report)
    diagnostic_availability={}
    for card,kind,path in [('E9','concepts',a.e9_manifest),('E10','brightness',a.e10_manifest)]:
        diagnostic=json.loads(path.read_text())
        if diagnostic.get('state')=='NOT_EVALUABLE':
            diagnostic_availability[card]=dict(state='NOT_EVALUABLE',reason='No fixed legal diagnostic fixture; no GPU stage substituted')
            continue
        if diagnostic.get('state')!='PREPARED_DIAGNOSTICS' or diagnostic.get('kind')!=kind or not 1<=diagnostic.get('actual_units',0)<=5:
            raise ValueError('Fixed CPU diagnostic manifest invalid: '+str(path))
        if diagnostic.get('source_manifest_sha256')!=sha256_file(a.manifest):raise ValueError('Diagnostic source first10 manifest differs')
        verify_metadata(diagnostic['assets'])
        diagnostic_input=artifact(path.resolve(),'PREPARED_DIAGNOSTICS',sha256_file(path))
        stages[0]['cpu_artifacts'].append(diagnostic_input)
        report=output/card/'report.json'
        stage(card,'native_diagnostic_experiment.py',['--prepared-root',a.prepared_root,
            '--prepared-manifest',path,'--source-guard',source_guard,'--projection-basis',basis,
            '--allow-gpu','--resource-guard-state',guard,'--out',report.parent],
            [diagnostic_input,basis_input,guard_input],[artifact(report,'COMPLETED')],300,report=report)
        analysis(card,report)
        diagnostic_availability[card]=dict(state='FIXED_CPU_PREPARED',actual_units=diagnostic['actual_units'])
    final=output/'combined_analysis.json'
    stage('final_frozen_analysis','analyze_native_experiments.py',[*reports,'--B','200','--out',final],
        [artifact(path,'COMPLETED') for path in reports],[artifact(final,'CPU_FROZEN_OUTPUT_ANALYSIS')],60,'cpu')
    stages[-1]['role']='finalize'
    cleanup=runtime/'tensor_cleanup.json'
    stage('cleanup_completed_metric_tensors','cleanup_runtime_tensors.py',
        ['--metric-plan',a.metric_plan,'--evaluation',metric_out/'evaluation.json','--receipt',cleanup],
        [artifact(a.metric_plan.resolve(),'PREPARED_CPU_METADATA'),artifact(metric_out/'evaluation.json','COMPLETED')],
        [artifact(cleanup,'COMPLETED')],60,'cpu')
    stages[-1]['role']='finalize'
    plan=dict(platform='autodl',cuda_python=str(a.python),stages=stages,
        protocol='Seven independent algorithms; only runtime asset dependencies; scientific negatives continue; implementation errors STOP',
        fixed_memory_fraction=.3,idle_grace_seconds=60,provider_shutdown_foreign_safe=True,
        cpu_preparation_complete_required=True,CUDA_or_billing_readiness_not_claimed=True,
        algorithms=['E1','E2','E3','E4','E5','E6','E7'],E8_released_baselines=[('insid3','bilinear'),*baselines],
        metric_pilot=dict(train=24,development=8,inference=10,epochs=a.metric_epochs,matched_variants=list(VARIANTS)),
        diagnostic_availability=diagnostic_availability,balanced_control_cap=2000,UOT_cap=200,paired_analysis_bootstrap_repetitions=200,
        sum_stage_timeout_seconds=sum(s['timeout_seconds'] for s in stages),
        strong_references_unavailable='Full RSRM/FROST not silently replaced: named adapted component controls remain inside E1/E2; standalone full implementations are not asserted')
    validate_plan(plan)
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=HERE.parent)
    p.add_argument('--prepared-root',type=Path,default=Path('/root/autodl-tmp/demo9'))
    p.add_argument('--foris-root',type=Path,default=Path('/root/autodl-tmp/demo8_local_verification/foris_source'))
    p.add_argument('--python',type=Path,default=Path('/root/miniconda3/bin/python'))
    p.add_argument('--pythonpath',default='/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions')
    for flag in ('manifest','metric-plan','source-template','runtime-dir','output-root','guard-state','out','e9-manifest','e10-manifest'):
        p.add_argument('--'+flag,type=Path)
    p.add_argument('--metric-epochs',type=int,default=10);p.add_argument('--insid3-crf',action='store_true')
    p.add_argument('--method-timeout',type=int,default=600);p.add_argument('--baseline-timeout',type=int,default=600)
    p.add_argument('--prompt-timeout',type=int,default=3600);p.add_argument('--transport-timeout',type=int,default=900)
    p.add_argument('--acquisition-timeout',type=int,default=600);p.add_argument('--training-timeout',type=int,default=600)
    p.add_argument('--metric-evaluation-timeout',type=int,default=600)
    p.add_argument('--preflight',action='store_true',help='CPU-only static code/assets receipt; no CUDA inventory or subprocess stage')
    p.add_argument('--self-check',action='store_true')
    a=p.parse_args()
    if a.self_check:self_check();return
    if any(getattr(a,k) is None for k in ('manifest','metric_plan','source_template','runtime_dir','output_root','guard_state','out','e9_manifest','e10_manifest')):
        p.error('Explicit prepared manifests/template and fresh owned runtime/output/state/plan paths are required')
    if not 1<=a.metric_epochs<=10 or not 1<=a.prompt_timeout<=3600:p.error('Pilot epoch1..10 and prompt timeout<=3600')
    if a.out.exists():raise ValueError('Preserve existing queue; use new --out')
    plan=build_plan(a);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(plan,indent=2)+'\n')
    if a.preflight:
        def forbidden():raise AssertionError('CPU preparation must never query GPU inventory')
        guard=ResourceGuard(plan,a.guard_state,inventory=forbidden,poweroff=forbidden)
        if not guard.build_preflight():raise SystemExit(2)
    print(json.dumps(dict(state='FINITE_QUEUE_CPU_PREPARED',algorithm_count=7,stages=len(plan['stages']),
        stage_timeout_budget_seconds=plan['sum_stage_timeout_seconds'],plan=str(a.out),
        CPU_preflight_recorded=a.preflight,actual_pretrained_runtime_verified=False,CUDA_calls=0)))


def self_check():
    # Pure schema/CPU guard checks. Actual source/assets still need remote no-card
    # preparation; this fixture cannot mark pretrained stages runtime-ready.
    with tempfile.TemporaryDirectory(prefix='finite_queue_schema_cpu_') as folder:
        root=Path(folder);script=root/'ready.py';script.write_text('pass\n')
        generated=root/'basis.pt'
        stages=[dict(name='native_setup_v1',kind='gpu',argv=[sys.executable,str(script)],cwd=str(root),timeout_seconds=60,
                     produces=[artifact(generated)]),
                dict(name='consume',kind='gpu',argv=[sys.executable,str(script)],cwd=str(root),timeout_seconds=60,
                     requires=[artifact(generated)])]
        plan=dict(platform='autodl',stages=stages)
        def forbidden():raise AssertionError('CPU schema check touched GPU')
        guard=ResourceGuard(plan,root/'guard.json',inventory=forbidden,poweroff=forbidden)
        assert guard.build_preflight() and guard.preflight_ready()
        assert json.loads(guard.preflight_path.read_text())['pending_produced_inputs'][0]['producer']=='native_setup_v1'
        try:owned(root.parent/'foreign.pt',root)
        except ValueError:pass
        else:raise AssertionError('Output escape accepted')
    print(json.dumps(dict(state='CPU_SCHEMA_CHECK_PASSED',actual_GPU_or_model_calls=0,
        generated_asset_pending_only=True,actual_runtime_readiness=False)))


if __name__=='__main__':main()
