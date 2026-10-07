#!/usr/bin/env python3
"""Recover an interrupted fixed-600 Astra run without changing its frozen method.

Each missing/invalid case runs in a separate subprocess. One native crash cannot
break a shared process pool; every attempt retains its exit code, signal, logs,
and partial outputs. The original directory is read-only to this program.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

SAFE = re.compile(r'^[A-Za-z0-9_.-]+$')
DRIVER = Path(__file__).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def cpu_env():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    return env


def saved_runner(source):
    """Import only the byte-verified runner and packages in the new snapshot."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(source) / 'src'))
    for name in list(sys.modules):
        if name == 'ics' or name.startswith('ics.'):
            del sys.modules[name]
    importlib.invalidate_caches()
    spec = importlib.util.spec_from_file_location('_recovery_saved_astra_runner',
                                                  Path(source) / 'scripts/run_astra300.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_paths(source, hashes):
    if 'scripts/run_astra300.py' not in hashes:
        raise ValueError('Original source hashes must bind the saved runner')
    for relative, expected in hashes.items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError('Unsafe frozen source path: ' + relative)
        path = Path(source) / rel
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError('Missing/changed original frozen source: ' + relative)
        yield relative, path


class Assets:
    """Hash each immutable input once; reject conflicting bindings and changes."""

    def __init__(self):
        self.checked = {}

    def check(self, path, expected):
        path = Path(path).resolve()
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        previous = self.checked.get(str(path))
        if previous is not None:
            if previous != (expected, fingerprint):
                raise ValueError('Input asset binding/metadata changed: ' + str(path))
        else:
            if not expected or sha(path) != expected:
                raise ValueError('Input asset SHA mismatch: ' + str(path))
            self.checked[str(path)] = (expected, fingerprint)

    def unchanged(self):
        for name, (_, fingerprint) in self.checked.items():
            stat = Path(name).stat()
            if (stat.st_size, stat.st_mtime_ns) != fingerprint:
                raise ValueError('Input asset changed during recovery: ' + name)


def arm_sources(runner, config):
    methods, controls = runner.registry(config['modules'])
    if not set(config['methods']) <= set(methods) or not set(config['controls']) <= set(controls):
        raise ValueError('Original selected arms differ from original frozen registry')
    bindings = {}
    for name in config['modules']:
        if hasattr(runner, 'module_binding'):
            import_name, source_name = runner.module_binding(name)
        else:
            import_name, source_name = 'ics.astra300.' + name, 'src/ics/astra300/' + name + '.py'
        module = importlib.import_module(import_name)
        for arm in set(module.METHODS) | set(module.CONTROLS):
            bindings[arm] = config['source_hashes'][source_name]
    return bindings


def verify_case(case, row, binding, expected_arms, sources, assets):
    """A partial, failed, altered, or differently bound receipt is never reused."""
    import numpy as np
    case = Path(case)
    record = load(case / 'receipt.json')
    if (record.get('id') != row['id'] or record.get('query_GT_read') is not False
            or record.get('source_sha256') != row['sha256']
            or set(record.get('arms', {})) != set(expected_arms)
            or record.get('row_binding') != binding):
        raise ValueError('Incomplete/differently bound case receipt: ' + row['id'])
    if load(case / 'input.json') != row or sha(case / 'input.json') != record.get('input_row_sha256'):
        raise ValueError('Case input row changed: ' + row['id'])
    assets.check(row['feature_pack'], row['sha256'])
    with np.load(row['feature_pack'], allow_pickle=False) as pack:
        producer = json.loads(pack['producer_json'].item())
        if producer != record.get('producer'):
            raise ValueError('Case producer differs from actual feature pack: ' + row['id'])
    for key in ('q_rgb', 'r_rgb', 'reference_mask'):
        if key in row:
            assets.check(row[key], row.get(key + '_sha256'))
    if record.get('provider_receipt', {}).get('query_GT_read') is not False:
        raise ValueError('Missing explicit inference-only provider receipt: ' + row['id'])
    for asset in record.get('artifact_bindings', []):
        assets.check(asset['path'], asset['sha256'])
    for arm in expected_arms:
        receipt = record['arms'][arm]
        path = case / (arm + '.npz')
        if (receipt.get('state') != 'complete' or receipt.get('query_GT_read') is not False
                or receipt.get('input_sha256') != row['sha256']
                or receipt.get('method_source_sha256') != sources[arm]
                or sha(path) != receipt.get('output_sha256')):
            raise ValueError('Incomplete/changed arm output: ' + row['id'] + '/' + arm)
        with np.load(path, allow_pickle=False) as pack:
            shape = pack['original_shape']
            mask = pack['mask_original']
            if (shape.dtype.kind not in 'iu' or shape.shape != (2,)
                    or shape.tolist() != row['original_shape'] or min(shape) <= 0
                    or mask.dtype != np.uint8 or mask.ndim != 1
                    or mask.size != (int(np.prod(shape)) + 7) // 8):
                raise ValueError('Invalid original-resolution packed mask: ' + row['id'] + '/' + arm)
            pixels = int(np.unpackbits(mask, count=int(np.prod(shape))).sum())
            if pixels != receipt.get('predicted_pixels'):
                raise ValueError('Prediction/receipt pixel count differs: ' + row['id'] + '/' + arm)
    return dict(id=row['id'], arms=record['arms'], producer=record['producer'],
                input_sha256=row['sha256'], row_binding=binding,
                receipt_sha256=sha(case / 'receipt.json'))


def case_hashes(case, arms):
    return {name: sha(Path(case) / name) for name in
            ['input.json', 'receipt.json'] + [arm + '.npz' for arm in arms]}


def host_mapping(runner, config, path):
    expected = config.get('host_manifest_sha256')
    if expected is None:
        if path is not None:
            raise ValueError('Cannot introduce a host binding absent from original config')
        return {}
    if path is None or sha(path) != expected:
        raise ValueError('Original host manifest requires its unchanged hash-bound file')
    hosts = load(path)
    runner.assert_inference_only(hosts)
    hosts = hosts.get('hosts', hosts.get('rows', hosts))
    if not isinstance(hosts, dict) or set(hosts) != {row['id'] for row in config['input_rows']}:
        raise ValueError('Original host manifest must cover exactly the original 600 IDs')
    for host in hosts.values():
        for key in ('field', 'mask'):
            if key in host:
                host[key]['path'] = str(runner.resolve_path(host[key]['path'], config.get('input_base')))
    return hosts


def child(packet_path):
    packet = load(packet_path)
    if sha(DRIVER) != packet['driver_sha256']:
        raise ValueError('Recovery driver changed before case execution')
    run = Path(packet['run'])
    config = load(run / 'config.json')
    runner = saved_runner(run / 'source')
    runner.verify_snapshots(run, config['source_hashes'])
    runner.initialize(config['threads'], str(run / 'source'))
    # Execute the old one_episode function with the old method/source/options.
    # Only its output destination changes to this retained attempt directory.
    result = runner.one_episode(packet['row'], packet['binding'], packet['output'],
                                config['modules'], config['methods'], config['controls'],
                                config['source_hashes'], config.get('input_base'),
                                config.get('save_native_fields', False), packet['host'],
                                config.get('encoder_model_dir'), config['threads'])
    runner.write(Path(packet_path).parent / 'worker_result.json', result)


def recover(args):
    original, out = args.original.resolve(), args.out.resolve()
    if original == out or original in out.parents or out in original.parents:
        raise ValueError('Recovery and original must be separate non-nested directories')
    if (original / 'sealed.json').exists() and load(original / 'sealed.json').get('all_arms_complete'):
        raise ValueError('An already complete sealed run does not need recovery')
    config, manifest = load(original / 'config.json'), load(original / 'manifest.json')
    rows = config['input_rows']
    arms = config['methods'] + config['controls']
    if (config.get('evaluation_count') != 600 or len(rows) != 600
            or len({row['id'] for row in rows}) != 600):
        raise ValueError('Recovery requires the unchanged fixed 600 unique IDs')
    if not arms or len(set(arms)) != len(arms) or config.get('threads') != 1:
        raise ValueError('Original nonempty distinct arms and single-thread budget required')
    if any(not SAFE.fullmatch(row['id']) or row['id'] in ('.', '..', 'source', 'recovery') for row in rows):
        raise ValueError('Unsafe/reserved original case identity')
    workers = args.workers if args.workers is not None else min(config['workers'], 8)
    if not 1 <= workers <= min(config['workers'], 28):
        raise ValueError('Recovery concurrency must stay within original resource budget and 28')
    initial_hashes = {name: sha(original / name) for name in ('config.json', 'manifest.json')}
    files = list(source_paths(original / 'source', config['source_hashes']))
    out.mkdir(parents=True, exist_ok=False)
    meta = out / 'recovery'
    meta.mkdir()
    for name in initial_hashes:
        shutil.copyfile(original / name, out / name)
    for relative, path in files:
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    runner = saved_runner(out / 'source')
    runner.assert_inference_only(manifest)
    runner.assert_inference_only(config)
    bound_rows = manifest if isinstance(manifest, list) else manifest['rows']
    if (bound_rows != rows or config.get('query_GT_read') is not False
            or config.get('quality_scored') is not False):
        raise ValueError('Original manifest/configuration is not the same inference-only cohort')
    sources = arm_sources(runner, config)
    hosts = host_mapping(runner, config, args.host_manifest)
    assets, ready, reused, invalid, tasks = Assets(), {}, {}, {}, deque()
    driver_sha = sha(DRIVER)
    shutil.copyfile(DRIVER, meta / 'resume_astra_run.py')
    execution = dict(schema='ASTRA_FIXED600_RECOVERY_V1', original=str(original),
                     original_config_sha256=initial_hashes['config.json'],
                     original_manifest_sha256=initial_hashes['manifest.json'],
                     driver_sha256=driver_sha, workers=workers, threads=1,
                     original_workers=config['workers'], max_case_attempts=2,
                     case_timeout_seconds=args.case_timeout,
                     config_and_source_preserved_byte_for_byte=True,
                     algorithm_execution='original frozen runner.one_episode',
                     query_GT_read=False, quality_scored=False,
                     original_stopped_required=True)
    write(meta / 'execution.json', execution)
    start = time.monotonic()
    for stub in rows:
        identity = stub['id']
        # A changed/missing input is not a recoverable method failure: stop rather
        # than silently bind a different input, skip a case, or change the cohort.
        row, binding = runner.read_input(stub, config.get('input_base'))
        try:
            result = verify_case(original / identity, row, binding, arms, sources, assets)
        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid[identity] = repr(error)
            tasks.append((row, binding, 1))
        else:
            expected = case_hashes(original / identity, arms)
            target = out / identity
            target.mkdir()
            for name in expected:
                shutil.copyfile(original / identity / name, target / name)
            if case_hashes(target, arms) != expected:
                raise ValueError('Reused case changed during copy: ' + identity)
            ready[identity] = result
            reused[identity] = dict(original=str(original / identity), file_sha256=expected)
    write(meta / 'reuse_inventory.json', dict(reused=reused, rerun_reasons=invalid,
          reused_count=len(reused), pending_count=len(tasks), total=600, query_GT_read=False))
    active, exhausted, attempts = {}, {}, []

    def status(state):
        write(out / 'status.json', dict(state=state, finished=len(ready), total=600,
              reused=len(reused), computed=len(ready)-len(reused), in_flight=len(active),
              queued=len(tasks), exhausted_fixed_ids=sorted(exhausted),
              elapsed_seconds=time.monotonic()-start, query_GT_read=False))

    try:
        while tasks or active:
            while tasks and len(active) < workers:
                row, binding, number = tasks.popleft()
                attempt = meta / 'attempts' / row['id'] / ('attempt_' + str(number))
                attempt.mkdir(parents=True)
                output = attempt / 'output'
                output.mkdir()
                packet = dict(run=str(out), row=row, binding=binding, output=str(output),
                              host=hosts.get(row['id']), driver_sha256=driver_sha)
                write(attempt / 'packet.json', packet)
                stream = (attempt / 'worker.log').open('wb')
                process = subprocess.Popen([sys.executable, str(DRIVER), '--_case', str(attempt / 'packet.json')],
                    stdout=stream, stderr=subprocess.STDOUT, env=cpu_env(), start_new_session=True)
                active[row['id']] = dict(process=process, stream=stream, row=row, binding=binding,
                    number=number, attempt=attempt, start=time.monotonic(), timed_out=False)
                write(attempt / 'launch.json', dict(id=row['id'], attempt=number, pid=process.pid,
                    threads=1, driver_sha256=driver_sha, started_unix_seconds=time.time(), query_GT_read=False))
            for identity, item in list(active.items()):
                process = item['process']
                if (args.case_timeout is not None and process.poll() is None
                        and time.monotonic()-item['start'] > args.case_timeout):
                    item['timed_out'] = True
                    os.killpg(process.pid, signal.SIGKILL)
                code = process.poll()
                if code is None:
                    continue
                process.wait()
                item['stream'].close()
                code = process.returncode
                number, attempt = item['number'], item['attempt']
                sig = -code if code < 0 else None
                outcome = dict(id=identity, attempt=number, exitcode=code, signal=sig,
                    signal_name=None if sig is None else signal.Signals(sig).name,
                    timed_out=item['timed_out'], wall_seconds=time.monotonic()-item['start'],
                    retained_attempt=str(attempt), query_GT_read=False)
                try:
                    if code != 0:
                        raise ValueError('Case subprocess exited with code ' + str(code))
                    result = verify_case(attempt / 'output' / identity, item['row'], item['binding'], arms, sources, assets)
                    worker_result = load(attempt / 'worker_result.json')
                    if any(worker_result.get(key) != result[key] for key in result):
                        raise ValueError('Worker completion and validated case receipt differ')
                    target = out / identity
                    expected = case_hashes(attempt / 'output' / identity, arms)
                    shutil.copytree(attempt / 'output' / identity, target)
                    if case_hashes(target, arms) != expected:
                        raise ValueError('Completed case changed while promoting output')
                    ready[identity] = worker_result
                    outcome['state'] = 'complete'
                except (OSError, ValueError, KeyError, TypeError) as error:
                    outcome.update(state='failed', error=repr(error), traceback=traceback.format_exc())
                    if number == 1:
                        tasks.append((item['row'], item['binding'], 2))
                        outcome['retry_queued'] = True
                    else:
                        exhausted[identity] = outcome
                write(attempt / 'exit.json', outcome)
                attempts.append(outcome)
                with (meta / 'attempt_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(outcome, allow_nan=False) + '\n')
                del active[identity]
                print(json.dumps(dict(id=identity, attempt=number, state=outcome['state'],
                                     exitcode=code, finished=len(ready), total=600)), flush=True)
            status('recovering')
            if active:
                time.sleep(.1)
    finally:
        # Preserve active scenes and their signals if this coordinator is
        # interrupted; do not leave unobserved subprocesses calculating.
        for identity, item in active.items():
            process = item['process']
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            item['stream'].close()
            write(item['attempt'] / 'exit.json', dict(id=identity, attempt=item['number'],
                  state='coordinator_interrupted', exitcode=process.returncode,
                  signal=-process.returncode if process.returncode < 0 else None, query_GT_read=False))
    runner.verify_snapshots(out, config['source_hashes'])
    list(source_paths(original / 'source', config['source_hashes']))
    assets.unchanged()
    if sha(DRIVER) != driver_sha:
        raise ValueError('Recovery executor changed during run')
    for name, expected in initial_hashes.items():
        if sha(original / name) != expected or sha(out / name) != expected:
            raise ValueError('Original/recovered configuration or manifest changed: ' + name)
    for identity, reuse in reused.items():
        if case_hashes(original / identity, arms) != reuse['file_sha256']:
            raise ValueError('Original reused case changed during recovery: ' + identity)
    if args.host_manifest is not None and sha(args.host_manifest) != config['host_manifest_sha256']:
        raise ValueError('Original host manifest changed during recovery')
    write(meta / 'result.json', dict(complete=len(ready), reused=len(reused),
          computed=len(ready)-len(reused), total=600, attempts=len(attempts),
          exhausted=exhausted, all_arms_complete=len(ready)==600 and not exhausted,
          elapsed_seconds=time.monotonic()-start, query_GT_read=False, quality_scored=False))
    if len(ready) != 600 or exhausted:
        status('failed_unsealed')
        return 2
    # Recheck all promoted/reused bytes before the original score program is
    # permitted to read any annotations. No failed fixed ID can disappear.
    for stub in rows:
        row, binding = runner.read_input(stub, config.get('input_base'))
        verified = verify_case(out / row['id'], row, binding, arms, sources, assets)
        if verified['receipt_sha256'] != ready[row['id']]['receipt_sha256']:
            raise ValueError('Final receipt changed before seal: ' + row['id'])
    outcomes = {arm: dict(complete=600, unavailable=0, failed=0, missing=0) for arm in arms}
    write(out / 'sealed.json', dict(state='sealed', receipts=[ready[row['id']] for row in rows],
          expected_arms=arms, evaluation_count=600, all_arms_complete=True, arm_outcomes=outcomes,
          config_sha256=sha(out / 'config.json'), bound_manifest_sha256=sha(out / 'manifest.json'),
          manifest_sha256=config['manifest_sha256'], source_hashes=config['source_hashes'],
          elapsed_seconds=time.monotonic()-start,
          execution_source='original frozen source in isolated CPU subprocesses; valid old cases SHA-verified and reused',
          recovery_execution_sha256=sha(meta / 'execution.json'), query_GT_read=False, quality_scored=False))
    status('sealed')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--workers', type=int, help='Default min(original workers, 8); never exceeds original budget')
    parser.add_argument('--host-manifest', type=Path, help='Required only when the original config bound a host manifest')
    parser.add_argument('--case-timeout', type=float, help='Optional explicitly recorded subprocess deadline; default no deadline')
    parser.add_argument('--_case', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.update(cpu_env())
    if args._case is not None:
        child(args._case)
        return 0
    if args.original is None or args.out is None:
        parser.error('--original and --out are required')
    if args.case_timeout is not None and args.case_timeout <= 0:
        parser.error('--case-timeout must be positive')
    try:
        return recover(args)
    except BaseException:
        if args.out.exists() and (args.out / 'recovery').is_dir():
            write(args.out / 'recovery' / 'coordinator_error.json',
                  dict(state='failed_unsealed', traceback=traceback.format_exc(), query_GT_read=False))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
