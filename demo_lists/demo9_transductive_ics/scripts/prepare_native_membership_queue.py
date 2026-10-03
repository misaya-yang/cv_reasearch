#!/usr/bin/env python3
"""Prepare one complete-source decision-path batch; never initialize CUDA.

Reject partial sklearn checks and source/API-only contracts. Only a completed
actual installed-source CPU smoke, solver smoke and analysis check can arm it.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from experiment_resource_guard import ResourceGuard,sha256_file,validate_plan
from native_membership_experiment import ARMS


def artifact(path,state=None,sha=None):
    result=dict(path=str(path))
    if state is not None:result['json_equals']=dict(state=state)
    if sha:result['sha256']=sha
    return result


def owned(path,root):
    path=Path(path).resolve()
    try:path.relative_to(root)
    except ValueError:raise ValueError('Output outside owned project')
    if path==root:raise ValueError('Output cannot be project root')
    return path


def checked_receipt(path,state):
    value=json.loads(path.read_text())
    if value.get('state')!=state:raise ValueError('Incomplete CPU receipt: '+str(path))
    return value,artifact(path.resolve(),state,sha256_file(path))


def build(a):
    root=a.project_root.resolve();m=json.loads(a.manifest.read_text())
    if m.get('schema')!='native_membership_assets_v1' or m.get('state')!='PREPARED_ASSETS' or tuple(m.get('arms',[]))!=ARMS:
        raise ValueError('Fresh CPU-prepared complete-source membership manifest required')
    rows=m.get('frozen_episodes',[]);keys={(r['fold'],r['e'],r['c']) for r in rows}
    if m.get('seed')!=0 or len(rows)!=40 or len(keys)!=40 or any(sum(k[0]==f for k in keys)!=10 for f in range(4)):
        raise ValueError('Forty official frozen seed0 tasks, ten per fold required')
    if any(k[2]%4!=k[0] or not 0<=k[2]<80 for k in keys):raise ValueError('Class outside official fold')
    if any(not r.get('support') or not r.get('query') or r['support']==r['query'] for r in rows):raise ValueError('Frozen support/query roles required')
    out=owned(a.output_dir,root);state=owned(a.state_file,root);plan=owned(a.plan_out,root)
    if state.exists() or plan.exists() or out.exists() and any(out.iterdir()):raise ValueError('Fresh output/plan/state only; no resume')
    sources=m.get('source_hashes',{})
    if not sources:raise ValueError('Source freeze required')
    required=['scripts/native_membership_experiment.py','scripts/analyze_native_membership.py',
              'scripts/prepare_native_membership_queue.py','scripts/native_membership_source_cpu.py',
              'scripts/reference_kernel_svm_cpu.py','scripts/experiment_resource_guard.py',
              'tics/native_decision_trace.py','tics/reference_kernel_svm.py','tics/frost_existing_adapter.py',
              'tics/native_candidate_axis.py','tics/reference_witness_audit.py',
              'scripts/native_candidate_axis_cpu.py','scripts/reference_witness_audit_cpu.py']
    if any(str((root/p).resolve()) not in sources for p in required):raise ValueError('Missing production source freeze')
    checks=[]
    for p,h in sources.items():
        path=Path(p)
        if not path.is_absolute() or sha256_file(path)!=h:raise ValueError('Source drift: '+p)
        if path.suffix=='.py':compile(path.read_text(),str(path),'exec')
        checks.append(artifact(path,sha=h))
    for x in m.get('assets',[]):
        path=Path(x['path']);st=path.stat()
        if not path.is_absolute() or (st.st_size,st.st_mtime_ns)!=(x['size'],x['mtime_ns']):raise ValueError('Existing asset drift')
        checks.append(artifact(path))
    if not m.get('assets') or str(a.projection_basis.resolve()) not in {str(Path(x['path']).resolve()) for x in m['assets']}:
        raise ValueError('Existing native500 basis must be prepared; full FROST constructs its own250')
    native,native_input=checked_receipt(a.source_receipt,'CPU_INSTALLED_MEMBERSHIP_SOURCE_PASSED')
    if native.get('cases_completed')!=10 or native.get('actual_source_execution') is not True or native.get('CUDA_initialized') is not False:
        raise ValueError('Ten actual source CPU fixtures required, not AST/mock readiness')
    if native.get('public_FoRIS_RGB_position_cases')!=10 or native.get('public_density_cases',0)<1:
        raise ValueError('Public source/RGB path and nonfallback FROST density must be exercised')
    if native.get('production_public_preprocess_cases')!=40 or native.get('query_GT_read_for_preprocessing') is not False:
        raise ValueError('Exact production public input helper on all40 real RGB/support masks required')
    checked_keys={(r['fold'],r['e'],r['c'],r['support'],r['query']) for r in native.get('production_public_preprocess_records',[])}
    if checked_keys!={(r['fold'],r['e'],r['c'],r['support'],r['query']) for r in rows}:
        raise ValueError('Production source helper checked a different real cohort')
    if not all(row.get('source_candidate_witness_exact') and row.get('candidate_axis2_whole_public_replica_exact') and row.get('channel_candidate_full_public_executed') for row in native.get('records',[])):
        raise ValueError('Every actual source case must exercise candidate witness and full-public axis controls')
    solver,solver_input=checked_receipt(a.solver_receipt,'CPU_REFERENCE_KERNEL_SVM_PASSED')
    if solver.get('passed')!=10 or solver.get('skipped')!=0 or solver.get('sklearn_available') is not True:
        raise ValueError('Full actual solver checks required; skipped tests cannot arm a GPU')
    analysis,analysis_input=checked_receipt(a.analysis_receipt,'CPU_NATIVE_MEMBERSHIP_ANALYSIS_PASSED')
    if len(analysis.get('checks',[]))<17:raise ValueError('Complete source-witness analysis selfcheck required')
    witness,witness_input=checked_receipt(a.witness_receipt,'CPU_REFERENCE_WITNESS_AUDIT_PASSED')
    if len(witness.get('checks',[]))!=10 or witness.get('cuda_initialized') is not False:
        raise ValueError('Ten CPU dual-geometry witness checks required')
    if witness.get('source_sha256')!=sources[str((root/'tics/reference_witness_audit.py').resolve())] or witness.get('cpu_script_sha256')!=sources[str((root/'scripts/reference_witness_audit_cpu.py').resolve())]:
        raise ValueError('Witness receipt is for another code version')
    axis,axis_input=checked_receipt(a.axis_receipt,'CPU_NATIVE_CANDIDATE_AXIS_PASSED')
    if len(axis.get('cases',[]))!=10 or axis.get('CUDA_initialized') is not False:
        raise ValueError('Ten extracted-source axis checks required')
    for p,h in axis.get('source_hashes',{}).items():
        if p not in sources or sources[p]!=h:raise ValueError('Axis receipt source mismatch: '+p)
    # CPU source checks must cover the exact installed files, not a different
    # fixture source or a path merely named FoRIS/FROST.
    for p,h in native.get('source_hashes',{}).items():
        if p not in sources or sources[p]!=h:raise ValueError('CPU exercised source differs: '+p)
    if not native.get('source_hashes'):raise ValueError('Actual source test provenance missing')
    if solver.get('source_sha256')!=sources[str((root/'tics/reference_kernel_svm.py').resolve())] or solver.get('script_sha256')!=sources[str((root/'scripts/reference_kernel_svm_cpu.py').resolve())]:
        raise ValueError('Solver receipt was produced by another code version')
    report=out/'report.json';result=out/'analysis.json'
    env=dict(PYTHONPATH=a.pythonpath,DEMO9_CUDA_GUARD='1',DEMO4_GPU_FRAC='.4',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    manifest_input=artifact(a.manifest.resolve(),'PREPARED_ASSETS',sha256_file(a.manifest))
    gpu=dict(name='native_membership40',kind='gpu',cwd=str(root),timeout_seconds=1200,
        argv=[str(a.python),str(root/'scripts/native_membership_experiment.py'),
              '--prepared-manifest',str(a.manifest.resolve()),'--projection-basis',str(a.projection_basis.resolve()),
              '--prepared-root',str(a.prepared_root),'--foris-root',str(a.foris_root),
              '--resource-guard-state',str(state),'--allow-gpu','--out',str(out)],
        requires=[manifest_input,native_input,solver_input,analysis_input,witness_input,axis_input],cpu_artifacts=checks,
        code_files=[p for p in sources if p.endswith('.py')],env=env,
        produces=[artifact(report,'COMPLETED')],success_checks=[artifact(report,'COMPLETED')])
    cpu=dict(name='native_membership40_analysis',kind='cpu',role='handoff',cwd=str(root),timeout_seconds=60,
        argv=[str(a.python),str(root/'scripts/analyze_native_membership.py'),'--report',str(report),
              '--manifest',str(a.manifest.resolve()),'--out',str(result)],
        requires=[artifact(report,'COMPLETED'),manifest_input],produces=[artifact(result,'CPU_NATIVE_MEMBERSHIP_ANALYSIS')],
        success_checks=[artifact(result,'CPU_NATIVE_MEMBERSHIP_ANALYSIS')],env={**env,'CUDA_VISIBLE_DEVICES':''})
    prepared=dict(platform='autodl',cuda_python=str(a.python),stages=[gpu,cpu],fixed_memory_fraction=.4,
        idle_grace_seconds=60,provider_shutdown_foreign_safe=True,cpu_preparation_complete_required=True,
        protocol='One seven-arm full-source decision-path/normalization-axis diagnostic, not seven new methods; no retries/sweeps/queue expansion',
        actual_CUDA_readiness_not_claimed=True,sum_stage_timeout_seconds=1260)
    validate_plan(prepared);return prepared


def self_check():
    with tempfile.TemporaryDirectory(prefix='membership_guard_cpu_') as d:
        root=Path(d);script=root/'fixture.py';script.write_text('pass\n')
        report=root/'report.json'
        plan=dict(platform='autodl',stages=[dict(name='native_membership40',kind='gpu',cwd=str(root),argv=[sys.executable,str(script)],timeout_seconds=1200,produces=[artifact(report,'COMPLETED')]),dict(name='analysis',kind='cpu',role='handoff',cwd=str(root),argv=[sys.executable,str(script)],timeout_seconds=60,requires=[artifact(report,'COMPLETED')])])
        validate_plan(plan)
        def forbidden():raise AssertionError('CPU queue check touched provider/GPU')
        g=ResourceGuard(plan,root/'state.json',inventory=forbidden,poweroff=forbidden)
        assert g.build_preflight() and g.preflight_ready()
        wrong=root/'partial.json';wrong.write_text(json.dumps(dict(state='CPU_REFERENCE_KERNEL_SVM_PARTIAL_SKLEARN_UNAVAILABLE')))
        try:checked_receipt(wrong,'CPU_REFERENCE_KERNEL_SVM_PASSED')
        except ValueError:pass
        else:raise AssertionError('Skipped solver check accepted')
    print(json.dumps(dict(state='CPU_MEMBERSHIP_QUEUE_SCHEMA_CHECKED',CUDA_or_provider_used=False,partial_solver_rejected=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-check',action='store_true');p.add_argument('--preflight',action='store_true')
    p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--prepared-root',type=Path,default=Path('/root/autodl-tmp/demo9'))
    p.add_argument('--foris-root',type=Path,default=Path('/root/autodl-tmp/demo8_local_verification/foris_source'))
    p.add_argument('--python',type=Path,default=Path('/root/miniconda3/bin/python'))
    p.add_argument('--pythonpath',default='/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions')
    for key in ['manifest','output-dir','plan-out','state-file','projection-basis','source-receipt','solver-receipt','analysis-receipt','witness-receipt','axis-receipt']:p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if a.self_check:self_check();return
    if any(getattr(a,k.replace('-','_')) is None for k in ['manifest','output-dir','plan-out','state-file','projection-basis','source-receipt','solver-receipt','analysis-receipt','witness-receipt','axis-receipt']):p.error('Explicit fresh paths and complete CPU receipts required')
    plan=build(a);a.plan_out.parent.mkdir(parents=True,exist_ok=True);a.plan_out.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    if a.preflight:
        def forbidden():raise AssertionError('CPU preflight touched GPU/provider')
        if not ResourceGuard(plan,a.state_file,inventory=forbidden,poweroff=forbidden).build_preflight():raise SystemExit(2)
    print(json.dumps(dict(state='CPU_MEMBERSHIP_QUEUE_PREPARED',stages=2,preflight_recorded=a.preflight)))


if __name__=='__main__':main()
