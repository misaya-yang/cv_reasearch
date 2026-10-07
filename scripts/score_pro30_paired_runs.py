#!/usr/bin/env python3
"""Score sealed primary-only Pro30 predictions against saved fixed-600 baselines.

Inference is never rerun. All selected predictions and their sources are
verified before evaluation metadata or query annotations are opened.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile

SAFE = re.compile(r'^[A-Za-z0-9_.-]+$')
BASELINES = {'PRO30_B_R', 'insid3.release_bilinear.control'}
NATIVE_ARRAYS = ('q', 'r', 'wf', 'wvalid', 'q_valid', 'q_rgb', 'r_rgb', 'reference_mask')
LABEL_KEYS = {'q_gt', 'query_gt', 'query_ground_truth', 'ground_truth', 'truth',
              'query_mask', 'evaluation_rows', 'evaluation_metadata'}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def inference_only(value):
    if isinstance(value, dict):
        if LABEL_KEYS & {str(key).lower() for key in value}:
            raise ValueError('Evaluation/query-label input in inference binding')
        for child in value.values(): inference_only(child)
    elif isinstance(value, list):
        for child in value: inference_only(child)


class CheckedFiles:
    def __init__(self):
        self.files = {}
        self.image_shapes = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        old = self.files.get(str(path))
        if old is None:
            actual = file_sha(path)
            self.files[str(path)] = (actual, fingerprint)
        else:
            actual, previous = old
            if previous != fingerprint:
                raise ValueError('A scoring source changed: '+str(path))
        if expected is not None and actual != expected:
            raise ValueError('Missing/changed SHA-bound scoring source: '+str(path))
        return actual

    def unchanged(self):
        for name, (_, fingerprint) in self.files.items():
            stat = Path(name).stat()
            if (stat.st_size, stat.st_mtime_ns) != fingerprint:
                raise ValueError('Source changed during scoring: '+name)


def source_path(run, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe frozen source path')
    return run/'source'/path


def input_row(stub, base, checked):
    if 'inference_row_path' in stub:
        path = Path(stub['inference_row_path'])
        if not path.is_absolute():
            if base is None: raise ValueError('Relative row requires original input_base')
            path = Path(base)/path
        raw = path.read_text()
        row = json.loads(raw)
        binding = dict(inference_row_path=str(path), inference_row_sha256=hashlib.sha256(raw.encode()).hexdigest())
        checked.check(path, binding['inference_row_sha256'])
    else:
        row = dict(stub)
        binding = dict(inference_row_path=None,
                       inference_row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest())
    inference_only(row)
    if row['id'] != stub['id']: raise ValueError('Native row identity differs from fixed stub')
    for name in ('feature_pack', 'q_rgb', 'r_rgb', 'reference_mask', 'artifact_manifest', 'artifacts_manifest'):
        if name in row:
            path = Path(row[name])
            if not path.is_absolute():
                if base is None: raise ValueError('Relative native source requires original input_base')
                path = Path(base)/path
            row[name] = str(path)
    return row, binding


def verify_native(row, record, checked, packs):
    """Validate current native pack metadata without decompressing its features."""
    import numpy as np
    pack_key = (str(Path(row['feature_pack']).resolve()), row['sha256'])
    checked.check(row['feature_pack'], row['sha256'])
    if pack_key not in packs:
        with np.load(row['feature_pack'], allow_pickle=False) as pack:
            if set(pack.files) != {'q', 'r', 'foreground_weight', 'valid_weight',
                                   'query_geometry_json', 'producer_json'}:
                raise ValueError('Not an inference-only native feature pack')
            producer = json.loads(pack['producer_json'].item())
            geometry = json.loads(pack['query_geometry_json'].item())
        # Inspect NPY headers instead of materializing 32 MiB of R/Q each time.
        shapes = {}
        with zipfile.ZipFile(row['feature_pack']) as archive:
            for name in ('q', 'r'):
                with archive.open(name+'.npy') as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                    shape, _, dtype = reader(stream)
                    shapes[name] = (shape, dtype)
        if (producer.get('kind') != 'frozen_DINOv3_FP32_native_final_LN_patches'
                or producer.get('FoRIS_Part1_applied') is not False
                or producer.get('model_input_side') != 1024
                or any(shape != (4096, 1024) or dtype != np.float32 for shape, dtype in shapes.values())):
            raise ValueError('Current pre-unit FP32 native1024 source required; processed cache forbidden')
        packs[pack_key] = (producer, geometry)
    producer, geometry = packs[pack_key]
    if record['producer'] != producer or row.get('query_geometry', geometry) != geometry:
        raise ValueError('Recorded native producer/geometry differs from actual pack')
    model = producer.get('model_assets', producer)
    checkpoint = model.get('checkpoint_sha256')
    config_sha = model.get('config_sha256', model.get('model_config_sha256'))
    if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) for value in (checkpoint, config_sha)):
        raise ValueError('Actual checkpoint/config identities required')
    source_images = producer.get('source_image_hashes')
    if not isinstance(source_images, list) or len(source_images) != 2:
        raise ValueError('Ordered original R/Q image hash binding required')
    for name, expected in (('r_rgb', source_images[0]), ('q_rgb', source_images[1]),
                           ('reference_mask', producer.get('reference_mask_sha256'))):
        if row.get(name+'_sha256') != expected or not expected:
            raise ValueError('Native source row/producer identity differs: '+name)
        checked.check(row[name], expected)
        path = str(Path(row[name]).resolve())
        if path not in checked.image_shapes:
            from PIL import Image
            with Image.open(path) as image:
                checked.image_shapes[path] = (image.height, image.width)
    q_shape = checked.image_shapes[str(Path(row['q_rgb']).resolve())]
    r_shape = checked.image_shapes[str(Path(row['r_rgb']).resolve())]
    mask_shape = checked.image_shapes[str(Path(row['reference_mask']).resolve())]
    if list(q_shape) != row['original_shape'] or r_shape != mask_shape:
        raise ValueError('Native original R/Q/MR physical geometry differs')
    for physical in (geometry, row['reference_geometry']):
        if (physical.get('view_side') != 1024 or physical.get('resized_hw') != [1024, 1024]
                or physical.get('padding_top_left') != [0, 0]):
            raise ValueError('Source native1024 transform geometry differs')
    if row['reference_geometry'].get('original_hw', list(r_shape)) != list(r_shape):
        raise ValueError('Native original reference geometry differs from bound RGB')
    core_hashes = record.get('native_array_hashes', {})
    if not all(isinstance(core_hashes.get(name), str) and re.fullmatch('[0-9a-f]{64}', core_hashes[name]) for name in NATIVE_ARRAYS):
        raise ValueError('Complete native array provenance required')
    return dict(feature_pack_sha256=row['sha256'], source_image_hashes=source_images,
                reference_mask_sha256=producer['reference_mask_sha256'], producer=producer,
                query_geometry=geometry, reference_geometry=row['reference_geometry'],
                original_shape=row['original_shape'],
                native_array_hashes={name: core_hashes[name] for name in NATIVE_ARRAYS})


def renderer_label(arm, info, source_hashes, run, cache):
    if arm == 'insid3.release_bilinear.control':
        host = info.get('host_config', {})
        if (info.get('method') != 'complete released insid3' or info.get('refiner') != 'bilinear'
                or host.get('mask_refiner') != 'bilinear' or host.get('resize_to_orig_size') is not True
                or host.get('image_size') != 1024):
            raise ValueError('INSID3 baseline is not the complete recorded original bilinear pipeline')
        relative = 'src/ics/astra300/complete_baselines.py'
        if relative not in cache:
            tree = ast.parse(source_path(run, relative).read_text())
            hashes = next(ast.literal_eval(node.value) for node in tree.body
                          if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCE_HASHES' for t in node.targets))
            cache[relative] = hashes['insid3/models/insid3.py']
        original = cache[relative]
        if info.get('third_party_source_hashes', {}).get('insid3/models/insid3.py') != original:
            raise ValueError('Recorded INSID3 original renderer source identity differs')
        return dict(label='released INSID3 bilinear: native64 boolean -> FP32 bilinear original -> strict >0.5; no CRF',
                    evidence='recorded full host configuration and frozen embedded original source',
                    original_source_sha256=original,
                    adapter_source_sha256=source_hashes[relative])
    label = info.get('renderer')
    if not isinstance(label, (str, dict)) or not label:
        raise ValueError('Actual per-arm renderer label missing: '+arm)
    if arm == 'PRO30_B_R':
        if (info.get('contract') != 'Pro30_B_R_exact_CPU100_reference_fit' or info.get('Huber') is not True) and info.get('degeneration') not in (
                'empty_reference_target', 'full_foreground_reference_single_class_support'):
            raise ValueError('B_R baseline is not the source-defined Huber/degenerate readout')
    return dict(label=label, evidence='sealed per-arm execution receipt',
                recorded_geometry={key: info[key] for key in ('work_mask_shape', 'original_mask_shape',
                                   'output_field_hw', 'field_space', 'source_renderer_parity') if key in info})


def verify_run(run, selected, checked, packs):
    import numpy as np
    run = run.resolve()
    seal = load(run/'sealed.json')
    seal_sha = checked.check(run/'sealed.json')
    if seal.get('state') != 'sealed' or seal.get('evaluation_count') != 600 or seal.get('query_GT_read') is not False:
        raise ValueError('A complete fixed600 inference-only seal is required: '+str(run))
    checked.check(run/'config.json', seal['config_sha256'])
    checked.check(run/'manifest.json', seal['bound_manifest_sha256'])
    config, manifest = load(run/'config.json'), load(run/'manifest.json')
    inference_only(config); inference_only(manifest)
    hashes = seal['source_hashes']
    if hashes != config['source_hashes'] or seal['manifest_sha256'] != config['manifest_sha256']:
        raise ValueError('Sealed source/config/manifest binding differs')
    if config.get('query_GT_read') is not False or config.get('quality_scored') is not False:
        raise ValueError('Original run was not sealed as inference-only')
    for relative, expected in hashes.items(): checked.check(source_path(run, relative), expected)
    if 'src/ics/experiment.py' not in hashes: raise ValueError('Frozen statistical source missing')
    rows = config['input_rows']
    ids = [row['id'] for row in rows]
    manifest_rows = manifest if isinstance(manifest, list) else manifest['rows']
    if (len(ids) != 600 or len(set(ids)) != 600 or rows != manifest_rows
            or config.get('evaluation_count') != 600
            or any(not SAFE.fullmatch(identity) or identity in ('.', '..') for identity in ids)):
        raise ValueError('Original manifest/config must bind all600 unique safe IDs')
    expected_arms = config['methods']+config['controls']
    if seal['expected_arms'] != expected_arms or len(set(expected_arms)) != len(expected_arms):
        raise ValueError('Sealed arm selection differs from config')
    if not set(selected) <= set(expected_arms): raise ValueError('Unknown selected sealed arm')
    summaries = seal['receipts']
    if [entry['id'] for entry in summaries] != ids:
        raise ValueError('Complete sealed receipt order must equal all600 configured IDs')
    modules = set()
    for name in config['modules']:
        package, basename = ('pro30', name[6:]) if name.startswith('pro30.') else ('astra300', name)
        modules.add(hashes['src/ics/'+package+'/'+basename+'.py'])
    native, predictions, renderers = {}, {arm: [] for arm in selected}, {arm: Counter() for arm in selected}
    cases, renderer_cache = {}, {}
    outcomes = {arm: dict(complete=0, unavailable=0, failed=0, missing=0) for arm in expected_arms}
    for stub, summary in zip(rows, summaries):
        identity = stub['id']; case = run/identity
        checked.check(case/'receipt.json', summary['receipt_sha256'])
        record = load(case/'receipt.json')
        checked.check(case/'input.json', record['input_row_sha256'])
        row, binding = input_row(stub, config.get('input_base'), checked)
        if (load(case/'input.json') != row or record.get('id') != identity
                or record.get('source_sha256') != row['sha256'] or summary.get('input_sha256') != row['sha256']
                or record.get('producer') != summary.get('producer') or record.get('row_binding') != binding
                or record.get('arms') != summary.get('arms')
                or record.get('query_GT_read') is not False or set(record.get('arms', {})) != set(expected_arms)
                or record.get('provider_receipt', {}).get('query_GT_read') is not False):
            raise ValueError('Changed/incomplete sealed input or receipt: '+identity)
        for arm in expected_arms:
            state = record['arms'][arm].get('state', 'missing')
            outcomes[arm][state if state in outcomes[arm] else 'failed'] += 1
        native[identity] = verify_native(row, record, checked, packs)
        cases[identity] = row
        for arm in selected:
            receipt = record['arms'][arm]
            if (receipt.get('state') != 'complete' or receipt.get('query_GT_read') is not False
                    or receipt.get('input_sha256') != row['sha256']
                    or receipt.get('method_source_sha256') not in modules):
                raise ValueError('Selected arm incomplete/unbound on a fixed case: '+identity+'/'+arm)
            path = case/(arm+'.npz')
            checked.check(path, receipt['output_sha256'])
            with np.load(path, allow_pickle=False) as pack:
                shape = pack['original_shape'].copy(); bits = pack['mask_original'].copy()
                if not set(pack.files) <= {'mask_original', 'original_shape', 'field_native', 'threshold'}:
                    raise ValueError('Unexpected prediction pack fields')
            if (shape.shape != (2,) or shape.dtype.kind not in 'iu' or shape.tolist() != row['original_shape']
                    or min(shape) <= 0 or bits.dtype != np.uint8 or bits.ndim != 1
                    or bits.size != (int(np.prod(shape))+7)//8):
                raise ValueError('Invalid complete original-size packed prediction')
            mask = np.unpackbits(bits)
            size = int(np.prod(shape))
            if np.any(mask[size:]) or int(mask[:size].sum()) != receipt.get('predicted_pixels'):
                raise ValueError('Packed mask and receipt disagree')
            predictions[arm].append((bits, tuple(map(int, shape))))
            label = renderer_label(arm, receipt.get('info', {}), hashes, run, renderer_cache)
            renderers[arm][json.dumps(label, sort_keys=True)] += 1
    if (outcomes != seal['arm_outcomes']
            or seal.get('all_arms_complete') != all(values['complete'] == 600 for values in outcomes.values())):
        raise ValueError('Sealed per-arm completion counts differ from actual600 receipts')
    provenance = dict(run=str(run), sealed_sha256=seal_sha, config_sha256=seal['config_sha256'],
                      manifest_sha256=seal['bound_manifest_sha256'], source_hashes=hashes,
                      config_declared_renderer=config.get('renderer'),
                      actual_arm_renderers={arm: [dict(json.loads(label), cases=count) for label, count in values.items()]
                                            for arm, values in renderers.items()},
                      selected_arms=selected, selected_complete_cases=600,
                      all_original_requested_arms_complete=seal.get('all_arms_complete'),
                      unselected_arms=[arm for arm in expected_arms if arm not in selected])
    return dict(run=run, config=config, ids=ids, native=native, predictions=predictions,
                cases=cases, provenance=provenance)


def statistical_functions(runs):
    """Reuse exact frozen metric/group functions; never import inference models."""
    import numpy as np
    first = None; sources = {}
    for run in runs:
        path = source_path(run['run'], 'src/ics/experiment.py')
        text = path.read_text(); tree = ast.parse(text)
        nodes = [node for name in ('metric', 'photo_groups') for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(nodes) != 2: raise ValueError('Exact metric/photo_groups functions missing')
        identity = ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
        if first is not None and identity != first:
            raise ValueError('Frozen baseline/primary statistical protocols differ')
        first = identity
        sources[str(path)] = dict(file_sha256=file_sha(path), function_source_sha256={
            node.name: hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest() for node in nodes})
    namespace = {'np': np, 'Path': Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['metric'], namespace['photo_groups'], sources


def relative_data(data, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts: raise ValueError('Evaluation photo path must be relative to --data')
    return data/path


def score(args):
    import numpy as np
    from PIL import Image
    start = time.monotonic(); checked, packs = CheckedFiles(), {}
    driver_sha = checked.check(__file__)
    if args.out.exists(): raise ValueError('Preserve existing score output; choose a new directory')
    for run in [args.run]+args.baseline_run:
        if run.resolve() in args.out.resolve().parents or run.resolve() == args.out.resolve():
            raise ValueError('Score output must be separate from every immutable original run')
    primary_config = load(args.run/'config.json')
    if not primary_config['methods']:
        raise ValueError('Primary run must contain actual methods')
    methods = args.methods or list(primary_config['methods'])
    if len(set(methods)) != len(methods) or not set(methods) <= set(primary_config['methods']):
        raise ValueError('Choose actual distinct primary methods, never a control')
    if len(args.baseline_run) != len(args.baseline_arm) or not args.baseline_run:
        raise ValueError('Pair each --baseline-run with one --baseline-arm')
    if len(set(args.baseline_arm)) != len(args.baseline_arm) or not set(args.baseline_arm) <= BASELINES:
        raise ValueError('Only distinct real saved B_R/INSID3-bilinear controls are supported')
    if set(methods) & set(args.baseline_arm): raise ValueError('A primary cannot be its own baseline')
    if any(run.resolve() == args.run.resolve() for run in args.baseline_run):
        raise ValueError('A method-only primary run cannot double as a control run')
    primary = verify_run(args.run, methods, checked, packs)
    baselines = []
    for path, arm in zip(args.baseline_run, args.baseline_arm):
        baseline = verify_run(path, [arm], checked, packs)
        if arm not in baseline['config']['controls']:
            raise ValueError('Baseline must have been selected as an actual control')
        if baseline['ids'] != primary['ids'] or baseline['native'] != primary['native']:
            raise ValueError('Baseline/primary ordered fixed600 native R/Q/MR/model/geometry sources differ')
        baselines.append(baseline)
    metric, photo_groups, statistic_sources = statistical_functions([primary]+baselines)
    checked.unchanged()
    # Every selected original-size prediction, input and source above passed.
    # Only now may the independent evaluation mapping or annotations be read.
    evaluation_sha = checked.check(args.evaluation_rows, args.evaluation_rows_sha256)
    evaluation = load(args.evaluation_rows)
    rows = evaluation if isinstance(evaluation, list) else evaluation.get('rows', evaluation.get('episodes'))
    if not isinstance(rows, list) or [row['key'] for row in rows] != primary['ids']:
        raise ValueError('Evaluation must bind the exact ordered600 inference IDs')
    for row in rows:
        identity = row['key']; native = primary['native'][identity]
        if 'e' in row and identity != '{}_{}_{}'.format(int(row['fold']), int(row['e']), int(row['c'])):
            raise ValueError('Evaluation class/fold/episode does not match fixed source identity')
        if int(row['c']) < 0 or int(row['fold']) < 0: raise ValueError('Invalid evaluation class/fold')
        checked.check(relative_data(args.data, row['support']), native['source_image_hashes'][0])
        checked.check(relative_data(args.data, row['query']), native['source_image_hashes'][1])
    checked.unchanged()
    bundles = {arm: primary['predictions'][arm] for arm in methods}
    bundles.update({arm: run['predictions'][arm] for arm, run in zip(args.baseline_arm, baselines)})
    arrays = {arm: np.zeros((600, 2), np.int64) for arm in bundles}
    edits = {arm: {baseline: [] for baseline in args.baseline_arm} for arm in methods}
    gt_sources = {}
    for index, row in enumerate(rows):
        annotation = args.data/'annotations'/Path(row['query']).with_suffix('.png')
        gt_sources[str(annotation.resolve())] = checked.check(annotation)
        with Image.open(annotation) as image: truth = np.asarray(image) == int(row['c'])+1
        masks = {}
        for arm, predictions in bundles.items():
            bits, shape = predictions[index]
            if shape != truth.shape: raise ValueError('Prediction and original annotation geometry differ')
            mask = np.unpackbits(bits, count=int(np.prod(shape))).reshape(shape).astype(bool)
            masks[arm] = mask
            arrays[arm][index] = [int((mask & truth).sum()), int((mask | truth).sum())]
        for arm in methods:
            for baseline in args.baseline_arm:
                added, deleted = masks[arm] & ~masks[baseline], masks[baseline] & ~masks[arm]
                edit = dict(key=row['key'], c=int(row['c']), fold=int(row['fold']),
                            add_TP=int((added & truth).sum()), add_FP=int((added & ~truth).sum()),
                            delete_TP=int((deleted & truth).sum()), delete_FP=int((deleted & ~truth).sum()))
                delta_i, delta_u = arrays[arm][index]-arrays[baseline][index]
                if delta_i != edit['add_TP']-edit['delete_TP'] or delta_u != edit['add_FP']-edit['delete_FP']:
                    raise ValueError('Exact per-episode I/U edit closure failed')
                edits[arm][baseline].append(edit)
    classes = np.asarray([int(row['c']) for row in rows]); groups = photo_groups(rows)
    count = int(groups.max())+1
    draws = np.random.RandomState(0).randint(count, size=(2000, count))
    weights = np.stack([np.bincount(draw, minlength=count) for draw in draws])[:, groups]
    scores = {arm: metric(values, classes) for arm, values in arrays.items()}
    samples = {arm: np.asarray([metric(values, classes, weight) for weight in weights]) for arm, values in arrays.items()}
    contrasts = {arm: {baseline: dict(gain=scores[arm]-scores[baseline],
                    ci95=np.percentile(samples[arm]-samples[baseline], [2.5, 97.5]).tolist())
                    for baseline in args.baseline_arm} for arm in methods}
    class_edits, totals = {}, {}
    for arm in methods:
        class_edits[arm], totals[arm] = {}, {}
        for baseline in args.baseline_arm:
            class_edits[arm][baseline] = {}
            totals[arm][baseline] = {key: sum(edit[key] for edit in edits[arm][baseline])
                                     for key in ('add_TP', 'add_FP', 'delete_TP', 'delete_FP')}
            for cls in sorted(set(classes)):
                selector = classes == cls
                initial_i, initial_u = map(int, arrays[baseline][selector].sum(0))
                final_i, final_u = map(int, arrays[arm][selector].sum(0))
                edit = {key: sum(item[key] for item in edits[arm][baseline] if item['c'] == cls)
                        for key in totals[arm][baseline]}
                gain = 100*(final_i/max(final_u, 1)-initial_i/max(initial_u, 1))
                formula = (100*((edit['add_TP']-edit['delete_TP'])-initial_i/initial_u*(edit['add_FP']-edit['delete_FP']))/final_u
                           if initial_u > 0 and final_u > 0 else None)
                if formula is not None and abs(formula-gain) > 1e-9: raise ValueError('Exact class-IoU edit formula failed')
                class_edits[arm][baseline][str(int(cls))] = dict(edit, baseline_I=initial_i, baseline_U=initial_u,
                    final_I=final_i, final_U=final_u, gain=gain, exact_formula_gain=formula)
    checked.unchanged()
    source_files = {name: value[0] for name, value in checked.files.items()}
    report = dict(schema='PRO30_SEALED_PAIRED_RUN_SCORE_V1', n=600, classes=len(set(classes)),
                  methods=methods, baseline_arms=args.baseline_arm, scores=scores, gain_vs_baselines=contrasts,
                  total_actions_vs_baselines=totals, class_actions_vs_baselines=class_edits,
                  original_resolution=True, identical_ordered_native_inputs=True,
                  renderer_comparison='each complete method retains its separately recorded actual renderer; renderer equality is not asserted',
                  runs=[run['provenance'] for run in [primary]+baselines],
                  bootstrap=dict(draws=2000, rng='RandomState(0)', unit='connected support/query photographs', photo_groups=count),
                  exposure='reused development; not independent confirmation',
                  quality_scope='synthetic fixture' if all(native['producer'].get('synthetic_contract_fixture') for native in primary['native'].values()) else 'fixed development measurement',
                  mechanism_controls='not evaluated by this primary-versus-saved-baselines scoring invocation',
                  all_selected_arms_complete=True, inference_recomputed=False, failed_cases_filtered=0,
                  evaluation_rows_sha256=evaluation_sha, score_runner_sha256=driver_sha,
                  statistical_source_hashes=statistic_sources, all_score_source_files_sha256=source_files,
                  query_annotation_source_files_sha256=gt_sources,
                  query_GT_first_read_stage='after both run seals, config, full sources, all600 receipts/outputs, paired native identities and evaluation photo mapping verified',
                  wall_seconds=time.monotonic()-start)
    if args.out.exists(): raise ValueError('Preserve existing score output; choose a new directory')
    args.out.mkdir(parents=True)
    write(args.out/'report.json', report); write(args.out/'episode_edits.json', edits)
    np.savez_compressed(args.out/'counts.npz', **arrays, classes=classes, photo_groups=groups)
    np.save(args.out/'bootstrap_photo_draws.npy', draws)
    shutil.copyfile(__file__, args.out/'score_pro30_paired_runs.py')
    write(args.out/'scoring_source_manifest.json', dict(score_runner_sha256=report['score_runner_sha256'],
          statistical_source_hashes=statistic_sources, prediction_sources=[run['provenance'] for run in [primary]+baselines]))
    print(json.dumps(dict(n=600, scores=scores, gain_vs_baselines=contrasts)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--methods', nargs='+', help='Default all primary methods; no episode filtering')
    p.add_argument('--baseline-run', nargs='+', type=Path, required=True)
    p.add_argument('--baseline-arm', nargs='+', required=True)
    p.add_argument('--evaluation-rows', type=Path, required=True)
    p.add_argument('--evaluation-rows-sha256', help='Optional additional binding to the independently fixed evaluation file')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    score(args)


if __name__ == '__main__':
    main()
