#!/usr/bin/env python3
"""Prepare one fresh full-rank angular GPU stage and one <=60s CPU handoff; never run GPU."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from experiment_resource_guard import ResourceGuard, sha256_file, validate_plan

HERE = Path(__file__).resolve().parent


def artifact(path, state=None, sha=None):
    item = dict(path=str(path))
    if state is not None: item['json_equals'] = dict(state=state)
    if sha is not None: item['sha256'] = sha
    return item


def owned(path, root):
    path = Path(path).resolve()
    try: path.relative_to(root)
    except ValueError: raise ValueError('Output escapes owned project: '+str(path))
    if path == root: raise ValueError('Output cannot be project root')
    return path


def build_plan(a):
    root = a.project_root.resolve()
    manifest = json.loads(a.manifest.read_text())
    if manifest.get('schema') != 'native_angular_assets_v1' or manifest.get('state') != 'PREPARED_ASSETS':
        raise ValueError('CPU-prepared angular assets manifest required')
    if manifest.get('seed') != 0:
        raise ValueError('Frozen official seed0 metadata required')
    arms = manifest.get('arms', [])
    expected_arms = {'foris_crf', 'native_identity', 'paired_geometry', 'pooled_geometry',
                     'fg_aug_geometry', 'shuffled_geometry', 'frost95_geometry'}
    if len(arms) != 7 or len(set(arms)) != 7 or set(arms) != expected_arms:
        raise ValueError('Frozen seven-arm angular comparison required')
    episodes = manifest.get('frozen_episodes', [])
    keys = {(int(r['fold']), int(r['e']), int(r['c'])) for r in episodes}
    if len(episodes) != 40 or len(keys) != 40 or any(sum(k[0] == f for k in keys) != 10 for f in range(4)):
        raise ValueError('Forty unique tasks, ten each fold0..3, required')
    if {k[0] for k in keys} != {0, 1, 2, 3}:
        raise ValueError('Unexpected fold')
    if any(not 0 <= k[2] < 80 or k[2] % 4 != k[0] for k in keys):
        raise ValueError('Class is outside its official COCO-20i fold')
    if any(len({k[2] for k in keys if k[0] == f}) != 10 for f in range(4)):
        raise ValueError('Ten distinct fixed classes per fold required')
    for row in episodes:
        if not isinstance(row['support'], str) or not isinstance(row['query'], str) or row['support'] == row['query']:
            raise ValueError('Explicit different support/query role IDs required')
    output = owned(a.output_dir, root)
    state = owned(a.state_file, root)
    owned(a.plan_out, root)
    if state.exists() or a.plan_out.exists() or (output.exists() and any(output.iterdir())):
        raise ValueError('Use fresh angular output/plan/state paths; old queues are retired')
    sources = manifest.get('source_hashes')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Frozen production source hashes required')
    checks = []
    for path, sha in sources.items():
        p = Path(path)
        if not p.is_absolute() or not p.is_file() or sha256_file(p) != sha:
            raise ValueError('Prepared source drift: '+path)
        if p.suffix == '.py': compile(p.read_text(), str(p), 'exec')
        checks.append(artifact(p, sha=sha))
    canonical_sources = {str(Path(path).resolve()): sha for path, sha in sources.items()}
    if len(canonical_sources) != len(sources):
        raise ValueError('Duplicate source aliases in frozen manifest')
    sources = canonical_sources
    required = [root/'scripts/native_angular_experiment.py', root/'scripts/analyze_native_angular.py',
                root/'scripts/prepare_native_angular_queue.py', root/'scripts/global_representation_probe.py',
                root/'tics/native_assets.py', root/'tics/rice_statistics.py', root/'tics/rice_subspace.py',
                root/'tics/native_geometry_adapter.py', root/'tics/reference_nuisance_geometry.py',
                root/'tics/readout_baselines.py', root/'scripts/native_rice_core_experiment.py',
                root/'scripts/analyze_rice_core.py', root/'scripts/experiment_resource_guard.py']
    for path in required:
        if str(path.resolve()) not in sources:
            raise ValueError('Required code missing from frozen sources: '+str(path))
    assets = manifest.get('assets')
    if not isinstance(assets, list) or not assets:
        raise ValueError('Existing weights/data/basis asset metadata required')
    for row in assets:
        path = Path(row['path'])
        if not path.is_absolute(): raise ValueError('Asset paths must be absolute')
        st = path.stat()
        if st.st_size != row['size'] or st.st_mtime_ns != row['mtime_ns']:
            raise ValueError('Prepared asset metadata drift: '+str(path))
        checks.append(artifact(path))
    basis = a.projection_basis.resolve()
    if str(basis) not in {str(Path(r['path']).resolve()) for r in assets}:
        raise ValueError('Native basis must be an existing prepared asset')
    basis_meta = manifest.get('projection_basis')
    if basis_meta is not None:
        if Path(basis_meta['path']).resolve() != basis or sha256_file(basis) != basis_meta['sha256']:
            raise ValueError('Native basis differs from prepared checksum')
    receipt = json.loads(a.cpu_receipt.read_text())
    if receipt.get('state') != 'CPU_NATIVE_ANGULAR_ADAPTER_PASSED':
        raise ValueError('Actual CPU statistics checks must pass before queue preparation')
    if receipt.get('actual_source') is not True:
        raise ValueError('Actual installed FoRIS source CPU interface checks required, not just a mock')
    if manifest.get('native_host_sha') is not None and manifest['native_host_sha'] not in sources.values():
        raise ValueError('Native host source SHA missing from frozen source set')
    manifest_input = artifact(a.manifest.resolve(), 'PREPARED_ASSETS', sha256_file(a.manifest))
    cpu_input = artifact(a.cpu_receipt.resolve(), 'CPU_NATIVE_ANGULAR_ADAPTER_PASSED', sha256_file(a.cpu_receipt))
    runtime_receipt = json.loads(a.runtime_receipt.read_text())
    if runtime_receipt.get('state') != 'CPU_NATIVE_ANGULAR_RUNTIME_PASSED':
        raise ValueError('Actual angular runtime CPU checks must pass before preparation')
    extra_receipts = [artifact(a.runtime_receipt.resolve(), 'CPU_NATIVE_ANGULAR_RUNTIME_PASSED', sha256_file(a.runtime_receipt))]
    report, analysis = output/'report.json', output/'analysis.json'
    env = dict(PYTHONPATH=a.pythonpath, DEMO4_GPU_FRAC='.3', DEMO9_CUDA_GUARD='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    gpu = dict(name='native_angular40', kind='gpu', cwd=str(root), timeout_seconds=900,
        argv=[str(a.python), str(required[0]), '--prepared-root', str(a.prepared_root),
              '--foris-root', str(a.foris_root),
              '--prepared-manifest', str(a.manifest.resolve()), '--projection-basis', str(basis),
              '--allow-gpu', '--resource-guard-state', str(state), '--out', str(output)],
        requires=[manifest_input, cpu_input, artifact(basis), *extra_receipts], cpu_artifacts=checks,
        code_files=[p for p in sources if p.endswith('.py')], env=env,
        produces=[artifact(report, 'COMPLETED')], success_checks=[artifact(report, 'COMPLETED')])
    cpu = dict(name='native_angular40_analysis', kind='cpu', role='handoff', cwd=str(root), timeout_seconds=60,
        argv=[str(a.python), str(required[1]), '--report', str(report), '--manifest', str(a.manifest.resolve()), '--out', str(analysis)],
        requires=[artifact(report, 'COMPLETED'), manifest_input],
        produces=[artifact(analysis, 'CPU_FROZEN_OUTPUT_ANALYSIS')],
        success_checks=[artifact(analysis, 'CPU_FROZEN_OUTPUT_ANALYSIS')],
        env={**env, 'CUDA_VISIBLE_DEVICES': ''})
    plan = dict(platform='autodl', cuda_python=str(a.python), stages=[gpu, cpu],
        protocol='One frozen forty-task full-rank angular comparison on reused development tasks; no old matrices, retries, sweeps or automatic queue expansion',
        fixed_memory_fraction=.3, idle_grace_seconds=60, provider_shutdown_foreign_safe=True,
        cpu_preparation_complete_required=True, actual_CUDA_readiness_not_claimed=True,
        sum_stage_timeout_seconds=960, paired_analysis_bootstrap_repetitions=2000, paired_analysis_seed=2053)
    validate_plan(plan)
    return plan


def self_check():
    with tempfile.TemporaryDirectory(prefix='rice_queue_schema_cpu_') as folder:
        root = Path(folder); script = root/'fixture.py'; script.write_text('pass\n')
        produced = root/'report.json'; analysis = root/'analysis.json'
        stages = [dict(name='only_core', kind='gpu', argv=[sys.executable, str(script)], cwd=str(root),
            timeout_seconds=900, produces=[artifact(produced, 'COMPLETED')]),
            dict(name='analysis', kind='cpu', role='handoff', argv=[sys.executable, str(script)], cwd=str(root),
                timeout_seconds=60, requires=[artifact(produced, 'COMPLETED')],
                produces=[artifact(analysis, 'CPU_FROZEN_OUTPUT_ANALYSIS')])]
        def forbidden(): raise AssertionError('CPU preflight touched GPU/provider')
        guard = ResourceGuard(dict(platform='autodl', stages=stages), root/'guard.json', inventory=forbidden, poweroff=forbidden)
        assert guard.build_preflight() and guard.preflight_ready()
        try: owned(root.parent/'foreign.json', root)
        except ValueError: pass
        else: raise AssertionError('Foreign output accepted')
    # Exercise the actual preparer on fake CPU assets, not only guard schema.
    with tempfile.TemporaryDirectory(prefix='rice_queue_assets_cpu_') as folder:
        root = Path(folder)
        names = ('scripts/native_angular_experiment.py', 'scripts/analyze_native_angular.py',
                 'scripts/prepare_native_angular_queue.py', 'scripts/global_representation_probe.py',
                 'tics/native_assets.py', 'tics/rice_statistics.py', 'tics/rice_subspace.py',
                 'tics/native_geometry_adapter.py', 'tics/reference_nuisance_geometry.py',
                 'tics/readout_baselines.py', 'scripts/native_rice_core_experiment.py',
                 'scripts/analyze_rice_core.py', 'scripts/experiment_resource_guard.py')
        sources = {}
        for name in names:
            path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('pass\n')
            sources[str(path)] = sha256_file(path)
        basis = root/'native_basis.pt'; basis.write_bytes(b'CPU fixture only')
        st = basis.stat(); manifest = root/'manifest.json'; receipt = root/'statistics_cpu.json'
        manifest.write_text(json.dumps(dict(schema='native_angular_assets_v1', state='PREPARED_ASSETS', seed=0,
            arms=['foris_crf', 'native_identity', 'paired_geometry', 'pooled_geometry', 'fg_aug_geometry', 'shuffled_geometry', 'frost95_geometry'],
            frozen_episodes=[dict(fold=f, e=e, c=4*e+f, support='s'+str(f)+'_'+str(e), query='q'+str(f)+'_'+str(e))
                             for f in range(4) for e in range(10)], source_hashes=sources,
            assets=[dict(path=str(basis), size=st.st_size, mtime_ns=st.st_mtime_ns)],
            projection_basis=dict(path=str(basis), sha256=sha256_file(basis)))))
        receipt.write_text(json.dumps(dict(state='CPU_NATIVE_ANGULAR_ADAPTER_PASSED', actual_source=True)))
        runtime_receipt = root/'runtime_cpu.json'
        runtime_receipt.write_text(json.dumps(dict(state='CPU_NATIVE_ANGULAR_RUNTIME_PASSED')))
        args = argparse.Namespace(project_root=root, manifest=manifest, output_dir=root/'output',
            state_file=root/'guard.json', plan_out=root/'plan.json', projection_basis=basis,
            cpu_receipt=receipt, runtime_receipt=runtime_receipt, python=Path(sys.executable), prepared_root=root, foris_root=root,
            pythonpath='fixture_only')
        actual = build_plan(args)
        assert len(actual['stages']) == 2 and actual['stages'][0]['timeout_seconds'] == 900
        first = root/names[0]; first.write_text('pass\n# drift\n')
        try: build_plan(args)
        except ValueError: pass
        else: raise AssertionError('Frozen source drift accepted')
        first.write_text('pass\n')
        def forbidden(): raise AssertionError('CPU asset check touched GPU/provider')
        guard = ResourceGuard(actual, args.state_file, inventory=forbidden, poweroff=forbidden)
        assert guard.build_preflight() and guard.preflight_ready()
    print(json.dumps(dict(state='CPU_NATIVE_ANGULAR_QUEUE_PASSED', GPU_or_provider_calls=0, stages=2,
                         gpu_cap_seconds=900, cpu_handoff_cap_seconds=60)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=HERE.parent)
    p.add_argument('--prepared-root', type=Path, default=Path('/root/autodl-tmp/demo9'))
    p.add_argument('--foris-root', type=Path, default=Path('/root/autodl-tmp/demo8_local_verification/foris_source'))
    p.add_argument('--python', type=Path, default=Path('/root/miniconda3/bin/python'))
    p.add_argument('--pythonpath', default='/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions')
    for flag in ('manifest', 'output-dir', 'plan-out', 'state-file', 'projection-basis', 'cpu-receipt', 'runtime-receipt'):
        p.add_argument('--'+flag, type=Path)
    p.add_argument('--preflight', action='store_true', help='Existing guard CPU preflight only; no CUDA inventory')
    p.add_argument('--self-check', action='store_true')
    a = p.parse_args()
    if a.self_check: self_check(); return
    if any(getattr(a, key) is None for key in ('manifest', 'output_dir', 'plan_out', 'state_file', 'projection_basis',
                                             'cpu_receipt', 'runtime_receipt')):
        p.error('Explicit fresh plan/output/state and CPU-prepared manifest/basis/receipt required')
    plan = build_plan(a)
    a.plan_out.parent.mkdir(parents=True, exist_ok=True)
    a.plan_out.write_text(json.dumps(plan, indent=2, allow_nan=False)+'\n')
    if a.preflight:
        def forbidden(): raise AssertionError('CPU preparation touched GPU/provider')
        guard = ResourceGuard(plan, a.state_file, inventory=forbidden, poweroff=forbidden)
        if not guard.build_preflight(): raise SystemExit(2)
    print(json.dumps(dict(state='NATIVE_ANGULAR_QUEUE_CPU_PREPARED', plan=str(a.plan_out), stages=2,
        CPU_preflight_recorded=a.preflight, CUDA_or_billing_readiness_claimed=False)))


if __name__ == '__main__': main()
