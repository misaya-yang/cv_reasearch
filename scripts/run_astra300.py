#!/usr/bin/env python3
"""Sealed CPU Astra inference; query-label scoring is a separate invocation."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
DEFAULT_MODULES = ('group_001_075', 'group_076_150', 'group_151_225', 'group_226_300')
SAFE_NAME = re.compile(r'^[A-Za-z0-9_.-]+$')
FORBIDDEN_LABEL_KEYS = {'q_gt', 'query_gt', 'query_ground_truth', 'ground_truth',
                        'truth', 'query_mask', 'evaluation_rows', 'evaluation_metadata'}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _jsonable(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        if value.size <= 128:
            return value.tolist()
        from ics.astra300.common import array_hash
        return dict(shape=list(value.shape), dtype=str(value.dtype), array_sha256=array_hash(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(_jsonable(data), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def assert_inference_only(value):
    if isinstance(value, dict):
        bad = [k for k in value if str(k).lower() in FORBIDDEN_LABEL_KEYS]
        if bad:
            raise ValueError('Query/evaluation labels are forbidden in inference input: ' + repr(bad))
        for child in value.values():
            assert_inference_only(child)
    elif isinstance(value, list):
        for child in value:
            assert_inference_only(child)


def resolve_path(value, base):
    path = Path(value)
    if path.is_absolute():
        return path
    if base is None:
        raise ValueError('Relative input path needs explicit --input-base or manifest input_base: ' + str(path))
    return Path(base) / path


def registry(modules):
    methods, controls = {}, {}
    for name in modules:
        if not SAFE_NAME.fullmatch(name) or '.' in name:
            raise ValueError('Only astra300 package module basenames are allowed')
        module = importlib.import_module('ics.astra300.' + name)
        for target, entries in ((methods, module.METHODS), (controls, module.CONTROLS)):
            if set(target) & set(entries):
                raise ValueError('Duplicate arm IDs across modules: ' + repr(set(target) & set(entries)))
            for identity, function in entries.items():
                if not SAFE_NAME.fullmatch(identity) or not callable(function):
                    raise ValueError('Safe callable method/control IDs required')
            target.update(entries)
    if set(methods) & set(controls):
        raise ValueError('Controls cannot also count as methods')
    return methods, controls


def frozen_imports(frozen):
    sys.path.insert(0, str(Path(frozen) / 'src'))
    for name in list(sys.modules):
        if name == 'ics' or name.startswith('ics.'):
            del sys.modules[name]
    importlib.invalidate_caches()


def initialize(threads, frozen=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(threads)
    if frozen is not None:
        frozen_imports(frozen)


def verify_snapshots(run, hashes):
    for relative, expected in hashes.items():
        if file_sha(Path(run) / 'source' / relative) != expected:
            raise ValueError('Frozen source changed: ' + relative)


def freeze_sources(out):
    sources = (list((ROOT / 'src/ics/astra300').glob('*.py'))
               + list((ROOT / 'src/ics/methods').glob('*.py'))
               + [ROOT / 'src/ics/cpu100/common.py', ROOT / 'src/ics/cpu100/encoder.py',
                  ROOT / 'src/ics/cpu100/__init__.py', ROOT / 'src/ics/data.py',
                  ROOT / 'src/ics/experiment.py', ROOT / 'src/ics/__init__.py', Path(__file__).resolve()])
    hashes = {}
    for source in sorted(set(sources)):
        relative = source.relative_to(ROOT)
        target = out / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[str(relative)] = file_sha(target)
    return hashes


def _normalize_row(row, base):
    row = dict(row)
    for name in ('feature_pack', 'q_rgb', 'r_rgb', 'reference_mask', 'artifact_manifest', 'artifacts_manifest'):
        if name in row:
            row[name] = str(resolve_path(row[name], base))
    return row


def read_input(stub, base):
    if 'inference_row_path' in stub:
        path = resolve_path(stub['inference_row_path'], base)
        text = path.read_text(); row = json.loads(text)
        row_sha = hashlib.sha256(text.encode()).hexdigest()
    else:
        row = dict(stub); path = None
        row_sha = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    assert_inference_only(row)
    if row['id'] != stub['id']:
        raise ValueError('Inference row identity differs from fixed cohort stub')
    normalized = _normalize_row(row, base)
    # A half-written/missing feature file is still waiting, never a new cohort.
    with Path(normalized['feature_pack']).open('rb') as stream:
        stream.read(1)
    return normalized, dict(inference_row_path=None if path is None else str(path),
                            inference_row_sha256=row_sha)


def _asset(descriptor, base, source_hashes):
    import numpy as np
    from ics.astra300.common import readonly, ArtifactUnavailable
    if not isinstance(descriptor, dict):
        raise ArtifactUnavailable('Artifacts require an explicit bound descriptor or JSON metadata')
    declared = descriptor.get('source_image_hashes')
    if declared is not None and source_hashes is not None and declared != source_hashes:
        raise ArtifactUnavailable('Artifact source image hashes differ from native source')
    array_path = descriptor.get('npz', descriptor.get('path'))
    json_path = descriptor.get('json', descriptor.get('json_path'))
    path = array_path or json_path
    if path is None:
        # Inline producer/renderer metadata, never silently interpreted as an array.
        return descriptor, None
    path = resolve_path(path, base)
    if not descriptor.get('sha256') or file_sha(path) != descriptor['sha256']:
        raise ArtifactUnavailable('Missing/changed independent artifact hash: ' + str(path))
    if json_path is not None:
        value = json.loads(path.read_text()); assert_inference_only(value)
    else:
        if 'key' not in descriptor:
            raise ArtifactUnavailable('An NPZ artifact must declare its exact array key')
        with np.load(path, allow_pickle=False) as pack:
            if set(k.lower() for k in pack.files) & FORBIDDEN_LABEL_KEYS:
                raise ArtifactUnavailable('An inference artifact pack may not contain query truth')
            value = pack[descriptor['key']].copy()
        if descriptor.get('packbits'):
            shape = tuple(descriptor['shape'])
            value = np.unpackbits(value, count=int(np.prod(shape))).reshape(shape).astype(bool)
        value = readonly(value)
    return value, dict(path=str(path), sha256=descriptor['sha256'], key=descriptor.get('key'))


def bind_artifacts(ep, row, base):
    from ics.astra300.common import as_episode, ArtifactUnavailable
    available = dict(row.get('artifacts', {})); bindings = []
    manifest_name = row.get('artifact_manifest', row.get('artifacts_manifest'))
    if manifest_name:
        expected = row.get('artifact_manifest_sha256', row.get('artifacts_manifest_sha256'))
        if not expected or file_sha(manifest_name) != expected:
            raise ArtifactUnavailable('Independent artifact manifest needs its exact bound hash')
        manifest = json.loads(Path(manifest_name).read_text()); assert_inference_only(manifest)
        declared = manifest.get('source_image_hashes')
        expected_images = ep.producer.get('source_image_hashes')
        if expected_images is not None and declared != expected_images:
            raise ArtifactUnavailable('Independent artifact manifest source images differ')
        available.update(manifest['artifacts'])
        bindings.append(dict(path=manifest_name, sha256=expected))
    loaded = {}
    for name, descriptor in available.items():
        loaded[name], binding = _asset(descriptor, base, ep.producer.get('source_image_hashes'))
        if binding is not None:
            bindings.append(dict(name=name, **binding))
    return as_episode(ep, artifacts=loaded), bindings


def _rss_bytes():
    high_water = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(high_water if sys.platform == 'darwin' else high_water * 1024)


def one_episode(row, row_binding, out, modules, selected, selected_controls, source_hashes,
                input_base=None, save_native_fields=False, host=None, model_dir=None, threads=1):
    import numpy as np
    from ics.astra300.common import (Result, load_episode, render, array_hash, ArtifactUnavailable)
    start_wall, start_cpu = time.monotonic(), time.process_time()
    case = Path(out) / row['id']; case.mkdir(exist_ok=False)
    write(case / 'input.json', row)
    ep = load_episode(dict(row, artifacts={}))
    artifact_error = None; asset_bindings = []
    try:
        ep, asset_bindings = bind_artifacts(ep, row, input_base)
    except Exception as error:
        # Native-only methods can still run; artifact-dependent methods receive
        # the original precise binding failure through an explicit provider.
        artifact_error = repr(error)
        from ics.astra300.common import as_episode
        def unavailable_provider(_ep, name):
            raise ArtifactUnavailable('Artifact binding failed for ' + name + ': ' + artifact_error)
        ep = as_episode(ep, provider=unavailable_provider)
    # The provider is lazy: native-only cards do not construct an encoder.
    from ics.astra300.providers import bind_episode
    from ics.astra300.common import as_episode
    saved_artifacts = ep.artifacts
    ep = bind_episode(ep, row, host=host, model_dir=model_dir, threads=threads)
    if saved_artifacts:
        ep = as_episode(ep, artifacts=saved_artifacts, provider=ep.provider)
    if artifact_error:
        provider = ep.provider
        original_require = provider.require
        def bound_require(episode, name):
            if name in row.get('artifacts', {}) or name in ('mean_field','p0','mask0','host_producer','host_renderer'):
                raise ArtifactUnavailable('Independent artifact binding failed: ' + artifact_error)
            return original_require(episode, name)
        provider.require = bound_require
    immutable = {name: getattr(ep, name) for name in ('q', 'r', 'wf', 'wvalid', 'q_valid', 'q_rgb', 'r_rgb', 'reference_mask')
                 if getattr(ep, name) is not None}
    immutable.update({'artifact:' + name: value for name, value in ep.artifacts.items() if isinstance(value, np.ndarray)})
    for value in immutable.values():
        value.setflags(write=False)
    array_hashes = {name: array_hash(value) for name, value in immutable.items()}
    methods, controls = registry(modules)
    arms = {name: methods[name] for name in selected}; arms.update({name: controls[name] for name in selected_controls})
    receipts = {}
    method_sources = {}
    for module_name in modules:
        owner = importlib.import_module('ics.astra300.' + module_name)
        for name in set(owner.METHODS) | set(owner.CONTROLS):
            method_sources[name] = source_hashes['src/ics/astra300/' + module_name + '.py']
    for name, function in arms.items():
        wall, cpu = time.monotonic(), time.process_time()
        before_encoder = ep.provider.cache.get('encoder')
        before = None if before_encoder is None else before_encoder.stats()
        try:
            produced = function(ep)
            if not isinstance(produced, Result):
                raise TypeError('Astra arms must return common.Result; CPU100 Result/renderer is forbidden')
            rendered = render(ep, produced); mask = rendered['original']
            target = case / (name + '.npz')
            values = dict(mask_original=np.packbits(mask), original_shape=np.asarray(ep.original_shape, np.int64))
            field_info = None
            if produced.field is not None:
                field = np.asarray(produced.field)
                field_info = dict(shape=list(field.shape), array_sha256=array_hash(field), threshold=float(produced.threshold),
                                  field_space=produced.info.get('field_space', 'native_canvas'),
                                  minimum=float(field.min()), maximum=float(field.max()))
                if save_native_fields and field.size <= 4096:
                    values['field_native'] = field.astype(np.float32); values['threshold'] = np.asarray(produced.threshold)
            np.savez_compressed(target, **values)
            receipts[name] = dict(state='complete', wall_seconds=time.monotonic()-wall,
                                  cpu_seconds=time.process_time()-cpu, peak_RSS_bytes=_rss_bytes(),
                                  RSS_scope='worker high-water since process start', predicted_pixels=int(mask.sum()),
                                  field=field_info, info=_jsonable(produced.info), output_sha256=file_sha(target),
                                  method_source_sha256=method_sources[name], input_sha256=row['sha256'], query_GT_read=False)
        except Exception as error:
            receipts[name] = dict(state='unavailable' if isinstance(error, ArtifactUnavailable) else 'failed',
                                  wall_seconds=time.monotonic()-wall, cpu_seconds=time.process_time()-cpu,
                                  peak_RSS_bytes=_rss_bytes(), error=repr(error), traceback=traceback.format_exc(),
                                  method_source_sha256=method_sources[name], input_sha256=row['sha256'], query_GT_read=False)
        after_encoder = ep.provider.cache.get('encoder')
        after = None if after_encoder is None else after_encoder.stats()
        if after is not None:
            delta_keys = ('cache_hits','cache_misses','callback_forward_attempts','callback_successful_forwards',
                          'new_encoder_forwards','forward_cpu_seconds','forward_wall_seconds')
            receipts[name]['encoder_delta'] = {k: after.get(k, 0) - (0 if before is None else before.get(k, 0))
                                                for k in delta_keys}
        provider_receipt = dict(ep.provider.receipt)
        if after is not None:
            provider_receipt['encoder_stats'] = after
        record = dict(id=row['id'], arms=receipts, source_sha256=row['sha256'], producer=ep.producer,
                      input_row_sha256=file_sha(case / 'input.json'), native_array_hashes=array_hashes,
                      artifact_bindings=asset_bindings, artifact_binding_error=artifact_error,
                      provider_receipt=provider_receipt, query_GT_read=False, row_binding=row_binding)
        write(case / 'receipt.json', record)
    for name, value in immutable.items():
        if array_hash(value) != array_hashes[name]:
            raise ValueError('An arm mutated legal input: ' + name)
    if row_binding['inference_row_path'] is not None and file_sha(row_binding['inference_row_path']) != row_binding['inference_row_sha256']:
        raise ValueError('Inference-only row changed while running')
    return dict(id=row['id'], arms=receipts, producer=ep.producer, input_sha256=row['sha256'],
                row_binding=row_binding, receipt_sha256=file_sha(case / 'receipt.json'),
                wall_seconds=time.monotonic()-start_wall, cpu_seconds=time.process_time()-start_cpu)


def frozen_worker(frozen, *args):
    """Run the saved runner implementation, not a mutable live script body."""
    import importlib.util
    name = '_astra300_saved_runner'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(frozen) / 'scripts/run_astra300.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module; spec.loader.exec_module(module)
    return sys.modules[name].one_episode(*args)


def infer(args):
    initialize(args.threads)
    raw = args.manifest.read_text(); bound = json.loads(raw); assert_inference_only(bound)
    rows = bound if isinstance(bound, list) else bound['rows']
    input_base = args.input_base or (None if isinstance(bound, list) else bound.get('input_base'))
    if input_base is not None:
        input_base = str(Path(input_base).resolve())
    host_mapping = {}; host_manifest_hash = None
    if args.host_manifest is not None:
        host_mapping = json.loads(args.host_manifest.read_text()); assert_inference_only(host_mapping)
        if 'hosts' in host_mapping:
            host_mapping = host_mapping['hosts']
        elif 'rows' in host_mapping and isinstance(host_mapping['rows'], dict):
            host_mapping = host_mapping['rows']
        if not isinstance(host_mapping, dict):
            raise ValueError('Host manifest must be an ID-to-host binding JSON mapping')
        host_manifest_hash = file_sha(args.host_manifest)
        for host in host_mapping.values():
            for spec in ('field','mask'):
                if spec in host:
                    host[spec]['path'] = str(resolve_path(host[spec]['path'], input_base))
    model_dir = None if args.encoder_model_dir is None else str(args.encoder_model_dir.resolve())
    if len(rows) != args.evaluation_count or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Exactly the fixed evaluation-count unique inference IDs are required')
    if args.host_manifest is not None and set(host_mapping) != {r['id'] for r in rows}:
        raise ValueError('Host binding must cover exactly the same fixed inference IDs')
    if any(not SAFE_NAME.fullmatch(row['id']) or row['id'] in ('.', '..') for row in rows):
        raise ValueError('Safe fixed episode IDs required')
    methods, controls = registry(args.modules)
    selected = list(methods) if args.methods == ['all'] else ([] if args.methods == ['none'] else args.methods)
    selected_controls = list(controls) if args.controls == ['all'] else ([] if args.controls == ['none'] else args.controls)
    if not set(selected) <= set(methods) or not set(selected_controls) <= set(controls):
        raise ValueError('Unknown selected Astra method/control')
    if len(set(selected)) != len(selected) or len(set(selected_controls)) != len(selected_controls):
        raise ValueError('Duplicate requested arm IDs')
    if not selected and not selected_controls:
        raise ValueError('At least one explicitly selected arm is required')
    args.out.mkdir(parents=True, exist_ok=False)
    source_hashes = freeze_sources(args.out)
    write(args.out / 'manifest.json', bound)
    manifest_hash = hashlib.sha256(raw.encode()).hexdigest()
    config = dict(modules=args.modules, methods=selected, controls=selected_controls,
                  workers=args.workers, threads=args.threads, evaluation_count=args.evaluation_count,
                  input_rows=rows, input_base=input_base, source_hashes=source_hashes,
                  manifest_sha256=manifest_hash, query_GT_read=False, quality_scored=False,
                  renderer='Astra continuous physical interpolation once then strict threshold; explicit original bool allowed',
                  wait_inputs=args.wait_inputs, save_native_fields=args.save_native_fields,
                  host_manifest_sha256=host_manifest_hash, encoder_model_dir=model_dir,
                  extra_encoder_loading='lazy CPU provider only if a selected card requests it')
    write(args.out / 'config.json', config)
    expected_arms = selected + selected_controls
    receipts = {}; queued = set(); future_rows = {}; waiting_reasons = {}; start = time.monotonic()
    frozen = args.out / 'source'
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.threads, str(frozen))) as pool:
        while len(receipts) < len(rows):
            for stub in rows:
                identity = stub['id']
                if identity in queued or identity in receipts or len(future_rows) >= 2 * args.workers:
                    continue
                try:
                    row, binding = read_input(stub, input_base)
                except (FileNotFoundError, PermissionError, json.JSONDecodeError) as error:
                    if not args.wait_inputs:
                        raise ValueError('Fixed inference input not ready: ' + identity) from error
                    waiting_reasons[identity] = repr(error); continue
                waiting_reasons.pop(identity, None)
                future = pool.submit(frozen_worker, str(frozen), row, binding, str(args.out), args.modules,
                                     selected, selected_controls, source_hashes, input_base, args.save_native_fields,
                                     host_mapping.get(identity), model_dir, args.threads)
                future_rows[future] = identity; queued.add(identity)
            if future_rows:
                done, _ = concurrent.futures.wait(future_rows, timeout=1., return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    identity = future_rows.pop(future)
                    try:
                        receipts[identity] = future.result()
                    except Exception as error:
                        receipts[identity] = dict(id=identity, state='failed', error=repr(error), traceback=traceback.format_exc(), query_GT_read=False)
                    print(json.dumps(dict(id=identity, finished=len(receipts), total=len(rows))), flush=True)
            else:
                time.sleep(1.)
            write(args.out / 'status.json', dict(state='running' if future_rows else 'waiting_inputs',
                  finished=len(receipts), total=len(rows), queued=len(queued), in_flight=len(future_rows),
                  waiting_fixed_ids=[r['id'] for r in rows if r['id'] not in queued and r['id'] not in receipts],
                  waiting_reasons=waiting_reasons, elapsed_seconds=time.monotonic()-start, query_GT_read=False))
            if args.wait_timeout is not None and time.monotonic() - start > args.wait_timeout and len(receipts) < len(rows):
                raise TimeoutError('Fixed cohort still waiting; partial run is not sealed')
    verify_snapshots(args.out, source_hashes)
    if file_sha(args.manifest) != manifest_hash:
        raise ValueError('Fixed inference manifest changed while waiting/running')
    if args.host_manifest is not None and file_sha(args.host_manifest) != host_manifest_hash:
        raise ValueError('Source-bound host manifest changed while running')
    outcomes = {arm: dict(complete=0, unavailable=0, failed=0, missing=0) for arm in expected_arms}
    for receipt in receipts.values():
        for arm in expected_arms:
            state = receipt.get('arms', {}).get(arm, {}).get('state', 'missing')
            outcomes[arm][state if state in outcomes[arm] else 'failed'] += 1
    all_complete = all(count['complete'] == len(rows) for count in outcomes.values())
    sealed = dict(state='sealed', receipts=[receipts[r['id']] for r in rows], expected_arms=expected_arms,
                  evaluation_count=len(rows), all_arms_complete=all_complete, arm_outcomes=outcomes,
                  config_sha256=file_sha(args.out / 'config.json'), bound_manifest_sha256=file_sha(args.out / 'manifest.json'),
                  manifest_sha256=manifest_hash, source_hashes=source_hashes, elapsed_seconds=time.monotonic()-start,
                  execution_source='frozen source in CPU spawned workers', query_GT_read=False, quality_scored=False)
    write(args.out / 'sealed.json', sealed)
    write(args.out / 'status.json', dict(state='sealed', finished=len(receipts), total=len(rows),
          all_arms_complete=all_complete, arm_outcomes=outcomes, elapsed_seconds=time.monotonic()-start, query_GT_read=False))


def score(args):
    run = args.run
    sealed = json.loads((run / 'sealed.json').read_text())
    if sealed['state'] != 'sealed':
        raise ValueError('Score requires a completed fixed-cohort inference seal')
    if file_sha(run / 'config.json') != sealed['config_sha256'] or file_sha(run / 'manifest.json') != sealed['bound_manifest_sha256']:
        raise ValueError('Sealed configuration/manifest changed')
    verify_snapshots(run, sealed['source_hashes'])
    config = json.loads((run / 'config.json').read_text())
    ids = [r['id'] for r in config['input_rows']]
    if len(ids) != sealed['evaluation_count'] or len(sealed['receipts']) != len(ids):
        raise ValueError('Incomplete fixed cohort cannot be scored')
    receipt_summaries = {r['id']: r for r in sealed['receipts']}; records = {}
    for identity in ids:
        path = run / identity / 'receipt.json'
        if not path.exists() or file_sha(path) != receipt_summaries[identity].get('receipt_sha256'):
            raise ValueError('Missing/changed sealed episode receipt: ' + identity)
        record = json.loads(path.read_text())
        if record['source_sha256'] != receipt_summaries[identity]['input_sha256'] or record['producer'] != receipt_summaries[identity]['producer']:
            raise ValueError('Sealed input/producer mismatch: ' + identity)
        if file_sha(run / identity / 'input.json') != record['input_row_sha256']:
            raise ValueError('Sealed inference-only row changed: ' + identity)
        records[identity] = record['arms']
    complete = set.intersection(*[{a for a, r in records[i].items() if r['state'] == 'complete'} for i in ids])
    contract_excluded = set(getattr(args, 'exclude_arms', ()) or ())
    exclusion_reason = getattr(args, 'exclusion_reason', None)
    if not contract_excluded <= set(sealed['expected_arms']):
        raise ValueError('Contract exclusions must identify actual sealed arms')
    if contract_excluded and not exclusion_reason:
        raise ValueError('Contract exclusions require an explicit source-review reason')
    complete -= contract_excluded
    excluded = {a: sealed['arm_outcomes'][a] for a in sealed['expected_arms']
                if a not in complete and a not in contract_excluded}
    if excluded and not args.allow_partial:
        raise ValueError('Missing/unavailable requested arms; score refused unless --allow-partial: ' + repr(excluded))
    baseline = args.baseline or (config['controls'][0] if config['controls'] else None)
    if baseline is None or baseline not in complete or baseline not in config['controls']:
        raise ValueError('A complete explicitly selected control is required as --baseline')
    if not complete:
        raise ValueError('No complete paired arms')
    # No evaluation metadata or query annotation is opened before all seals,
    # source identities, input records, and completeness checks above pass.
    frozen_imports(run / 'source')
    import numpy as np
    from PIL import Image
    from ics.experiment import metric, photo_groups
    evaluation = json.loads(args.evaluation_rows.read_text())
    evaluation = evaluation if isinstance(evaluation, list) else evaluation.get('rows', evaluation.get('episodes'))
    indexed = {row['key']: row for row in evaluation}
    if len(indexed) != len(evaluation) or set(indexed) != set(ids):
        raise ValueError('Independent evaluation rows must match every fixed inference ID exactly')
    rows = [indexed[i] for i in ids]
    arrays = {a: np.zeros((len(rows), 2), np.int64) for a in sorted(complete)}
    actions = {a: [] for a in complete}; details = []
    for j, row in enumerate(rows):
        annotation = args.data / 'annotations' / Path(row['query']).with_suffix('.png')
        with Image.open(annotation) as image:
            truth = np.asarray(image) == int(row['c']) + 1
        masks = {}
        for arm in sorted(complete):
            path = run / row['key'] / (arm + '.npz')
            if file_sha(path) != records[row['key']][arm]['output_sha256']:
                raise ValueError('Sealed prediction changed: ' + row['key'] + '/' + arm)
            with np.load(path, allow_pickle=False) as z:
                shape = tuple(map(int, z['original_shape']))
                if len(shape) != 2 or min(shape) <= 0 or z['mask_original'].dtype != np.uint8:
                    raise ValueError('Invalid packed original-size prediction')
                masks[arm] = np.unpackbits(z['mask_original'], count=int(np.prod(shape))).reshape(shape).astype(bool)
            if masks[arm].shape != truth.shape:
                raise ValueError('Prediction must already be actual original query resolution')
            arrays[arm][j] = [(masks[arm] & truth).sum(), (masks[arm] | truth).sum()]
        for arm, mask in masks.items():
            added, deleted = mask & ~masks[baseline], ~mask & masks[baseline]
            edit = dict(key=row['key'], c=int(row['c']), fold=row['fold'],
                        add_TP=int((added & truth).sum()), add_FP=int((added & ~truth).sum()),
                        delete_TP=int((deleted & truth).sum()), delete_FP=int((deleted & ~truth).sum()))
            if (arrays[arm][j, 0] - arrays[baseline][j, 0] != edit['add_TP'] - edit['delete_TP']
                    or arrays[arm][j, 1] - arrays[baseline][j, 1] != edit['add_FP'] - edit['delete_FP']):
                raise ValueError('Exact per-episode I/U edit closure failed')
            actions[arm].append(edit); details.append(dict(edit, arm=arm, intersection=int(arrays[arm][j,0]), union=int(arrays[arm][j,1])))
    classes = np.asarray([int(r['c']) for r in rows]); groups = photo_groups(rows); g = int(groups.max()) + 1
    draws = np.random.RandomState(0).randint(g, size=(2000, g))
    weights = np.stack([np.bincount(d, minlength=g) for d in draws])[:, groups]
    scores = {a: metric(values, classes) for a, values in arrays.items()}
    samples = {a: np.array([metric(values, classes, w) for w in weights]) for a, values in arrays.items()}
    contrasts = {a: {b: dict(gain=scores[a]-scores[b], ci95=np.percentile(samples[a]-samples[b], [2.5,97.5]).tolist())
                      for b in config['controls'] if b in arrays} for a in config['methods'] if a in arrays}
    class_actions = {}; totals = {}
    for arm in arrays:
        class_actions[arm] = {}; totals[arm] = {k: int(sum(r[k] for r in actions[arm])) for k in ('add_TP','add_FP','delete_TP','delete_FP')}
        for cls in sorted(set(classes)):
            selector = classes == cls
            initial_i, initial_u = arrays[baseline][selector].sum(0); final_i, final_u = arrays[arm][selector].sum(0)
            edits = {k: int(sum(r[k] for r in actions[arm] if r['c'] == cls)) for k in totals[arm]}
            delta = 100 * (final_i / max(final_u, 1) - initial_i / max(initial_u, 1))
            formula = 100 * ((edits['add_TP']-edits['delete_TP']) - initial_i / initial_u * (edits['add_FP']-edits['delete_FP'])) / final_u if initial_u > 0 and final_u > 0 else None
            if formula is not None and abs(formula - delta) > 1e-9:
                raise ValueError('Exact per-class IoU edit formula failed')
            class_actions[arm][str(int(cls))] = dict(**edits, baseline_I=int(initial_i), baseline_U=int(initial_u),
                final_I=int(final_i), final_U=int(final_u), gain=float(delta), exact_formula_gain=formula)
    report = dict(n=len(rows), classes=len(set(classes)), scores=scores, baseline=baseline,
                  gain_vs_controls=contrasts, actions_vs_baseline=actions, total_actions_vs_baseline=totals,
                  class_actions_vs_baseline=class_actions,
                  gain_vs_baseline={a:dict(gain=scores[a]-scores[baseline],ci95=np.percentile(samples[a]-samples[baseline],[2.5,97.5]).tolist()) for a in arrays},
                  requested_method_count=len(config['methods']), scored_method_count=len(set(config['methods']) & complete),
                  all_requested_arms_complete=not excluded, explicitly_excluded_incomplete_arms=excluded,
                  explicitly_excluded_contract_arms=sorted(contract_excluded),
                  contract_exclusion_reason=exclusion_reason,
                  original_resolution=True, renderer=config['renderer'], exposure='reused development; not independent confirmation',
                  quality_scope='synthetic/activity probe' if len(rows) <= 4 else 'fixed development measurement',
                  bootstrap=dict(draws=2000,unit='connected support/query photographs',rng='RandomState(0)',photo_groups=g),
                  sealed_sha256=file_sha(run/'sealed.json'), config_sha256=sealed['config_sha256'],
                  evaluation_rows_sha256=file_sha(args.evaluation_rows), method_source_hashes=sealed['source_hashes'],
                  score_runner_sha256=file_sha(__file__), query_GT_first_read_stage='score after seal verification')
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out/'report.json', report); write(args.out/'episode_metrics.json', details)
    np.savez_compressed(args.out/'counts.npz', **arrays, classes=classes)
    np.save(args.out/'bootstrap_photo_draws.npy', draws)
    print(json.dumps(dict(n=len(rows), scores=scores)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    listing = stages.add_parser('list'); listing.add_argument('--modules', nargs='+', default=list(DEFAULT_MODULES))
    inference = stages.add_parser('infer')
    inference.add_argument('--manifest', type=Path, required=True); inference.add_argument('--evaluation-count', type=int, default=600)
    inference.add_argument('--modules', nargs='+', default=list(DEFAULT_MODULES)); inference.add_argument('--methods', nargs='+', default=['all'])
    inference.add_argument('--controls', nargs='+', default=['all']); inference.add_argument('--workers', type=int, default=3)
    inference.add_argument('--threads', type=int, default=2); inference.add_argument('--out', type=Path, required=True)
    inference.add_argument('--input-base', type=Path); inference.add_argument('--wait-inputs', action='store_true')
    inference.add_argument('--host-manifest', type=Path); inference.add_argument('--encoder-model-dir', type=Path)
    inference.add_argument('--wait-timeout', type=float); inference.add_argument('--save-native-fields', action='store_true')
    scoring = stages.add_parser('score'); scoring.add_argument('--run', type=Path, required=True)
    scoring.add_argument('--evaluation-rows', type=Path, required=True); scoring.add_argument('--data', type=Path, required=True)
    scoring.add_argument('--out', type=Path, required=True); scoring.add_argument('--baseline')
    scoring.add_argument('--allow-partial', action='store_true', help='Explicitly exclude arms incomplete on any fixed episode')
    scoring.add_argument('--exclude-arms', nargs='+', default=[], help='Explicit source-contract invalid arms; does not filter episodes')
    scoring.add_argument('--exclusion-reason', help='Source-review reason for excluding invalid algorithm versions')
    args = parser.parse_args()
    if args.stage == 'list':
        methods, controls = registry(args.modules); print(json.dumps(dict(methods=list(methods),controls=list(controls)),indent=2))
    elif args.stage == 'infer':
        if min(args.workers,args.threads,args.evaluation_count) < 1 or args.workers*args.threads > 28:
            raise ValueError('Positive CPU resource bound workers*threads <=28 required')
        infer(args)
    else:
        score(args)


if __name__ == '__main__':
    main()
