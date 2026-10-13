"""S2: 1000 unused cached single-reference tasks, with unscored baselines.

The pool contains preexisting photographs, not a new-photo holdout. Sampling
uses protocol metadata only. All inference runs on CPU with a cache-only
encoder adapter, and refuses current-query annotation reads. No scoring entry
point exists here. Runtime dependencies are frozen outside the active checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from functools import lru_cache

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '2'
sys.dont_write_bytecode = True

REPO = Path('/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch')
ASSETS = REPO.parent / 'cv_data'
OUT = ASSETS / 'a/x1012s2_confirm1000'
R5 = ASSETS / 'a/x1012_R5_validation'
P0 = ASSETS / 'a/x1012_P0'
RUNTIME = OUT / 'runtime'
SEED = 2026101301
PER_DOMAIN = 100
TARGET_N = 1000
WORKERS = 1
DOMAINS = ('coco', 'lvis', 'pascal_part', 'paco_part', 'suim', 'isaid', 'isic', 'lung', 'fundus', 'deepglobe_road')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def freeze_runtime():
    """Copy only executable dependencies; never restore deleted workspace files."""
    snapshot = OUT / 'runtime_sources.json'
    if snapshot.exists():
        for item in read(snapshot):
            assert sha(item['snapshot']) == item['sha256'], 'Frozen runtime changed'
        return
    sources = {str(p.relative_to(R5 / 'runtime')): [p] for p in sorted((R5 / 'runtime').rglob('*.py'))}
    records = []
    for relative, candidates in sources.items():
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            raise FileNotFoundError('Missing frozen dependency: ' + relative)
        target = RUNTIME / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == sha(source), 'Preserve different partial runtime'
        else:
            shutil.copyfile(source, target)
        records.append(dict(source=str(source), snapshot=str(target), sha256=sha(target)))
    for relative in ('ics/__init__.py', 'ics/methods/__init__.py'):
        target = RUNTIME / relative
        if not target.exists():
            target.write_text('"""S2 isolated replay package."""\n')
        if not any(r['snapshot'] == str(target) for r in records):
            records.append(dict(source='S2 minimal package marker', snapshot=str(target), sha256=sha(target)))
    frozen = OUT / 'frozen/x1012s2_confirm.py'
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), frozen)
    records.append(dict(source=str(Path(__file__).resolve()), snapshot=str(frozen), sha256=sha(frozen)))
    write(snapshot, records)


def modules():
    sys.path.insert(0, str(RUNTIME))
    lab = importlib.import_module('lab')
    # The original lab derives its root from __file__; the frozen copy is relocated.
    lab.ROOT, lab.ASSETS = REPO, ASSETS
    lab.BINDINGS = ASSETS / 'a/consensus_reference_calibration6380_v1_20261011/source_bindings.json'
    lab.BASIS = ASSETS / 'native_assets/positional_basis.pt'
    pc = importlib.import_module('plan_common')
    # plan_common prepends the active checkout; use snapshots for all runtime imports.
    sys.path.insert(0, str(RUNTIME))
    pm = importlib.import_module('p0_manifest')
    pm.OUT = OUT
    return lab, pc, pm


def signature_of_binding(binding, pm):
    if binding.get('pair_signature'):
        return binding['pair_signature']
    b = pm.normalized(binding)
    return pm.pair_signature(pm.observation(b, 'reference'), pm.observation(b, 'query'))


def input_task(reference_input, reference_mask_hash, query_input):
    """Inference identity deliberately excludes fold/class and query annotation."""
    return digest([reference_input, reference_mask_hash, query_input])


@lru_cache(None)
def canonical_reference_hash(path, native_hash):
    """Hash a lawful reference input only; never inspect query foreground GT."""
    import numpy as np
    import torch
    from PIL import Image
    from ics.official_data import array_hash
    with Image.open(path) as im:
        m = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
    assert array_hash(m) == native_hash, 'Reference annotation identity changed'
    m = torch.nn.functional.interpolate(torch.from_numpy(m)[None, None].float(),
                                         (1024, 1024), mode='nearest')[0, 0].numpy().astype(np.uint8)
    return hashlib.sha256(m.tobytes()).hexdigest()


def exclusion_registry(lab, pm, observations):
    """Read task identities from existing manifests, never outcome fields."""
    excluded = set(read(R5 / 'excluded_pairs.json'))
    tasks = set()
    old_masks = defaultdict(dict)
    rgb_inputs = defaultdict(set)
    for o in observations:
        rgb_inputs[(o['dataset'], o['view']['rgb_hash'])].add(o['raw']['input_sha256'])
    sources = {}
    recognized = Counter()

    def add_task(rids, mask_hash, mask_path, qids):
        if not mask_hash or not mask_path:
            return 0
        n = 0
        for ri in rids:
            for qi in qids:
                tasks.add(input_task(ri, mask_hash, qi))
                old_masks[(ri, qi)][mask_hash] = mask_path
                n += 1
        return n

    def consume(b):
        if not isinstance(b, dict):
            return 0
        if isinstance(b.get('binding'), dict):
            b = b['binding']
        row = b.get('row', b)
        if not isinstance(row, dict):
            return 0
        if b.get('pair_signature'):
            excluded.add(b['pair_signature'])
        fit = b.get('fit', {})
        if not isinstance(fit, dict):
            fit = {}
        if 'reference_raw' in fit and 'query_raw_identity' in fit:
            def raw_id(name):
                v = fit[name]
                return v.get('input_sha256') or pm.raw_metadata(v['payload_path'])['input_sha256']
            try:
                rids, qids = {raw_id('reference_raw')}, {raw_id('query_raw_identity')}
            except (FileNotFoundError, KeyError, TypeError, AttributeError):
                domain = b.get('dataset', row.get('dataset'))
                rids = rgb_inputs.get((domain, row.get('reference_rgb_hash')), set())
                qids = rgb_inputs.get((domain, row.get('query_rgb_hash')), set())
            if (not b.get('pair_signature') and b.get('dataset') and
                    all(k in row for k in ('query_mask_hash', 'fold', 'global_class_id', 'loader_class_id', 'episode_id')) and
                    all(isinstance(fit[k], dict) and Path(fit[k].get('payload_path', '/nonexistent')).is_file()
                        for k in ('reference_raw', 'query_raw_identity'))):
                excluded.add(signature_of_binding(b, pm))
        elif isinstance(b.get('raw_binding'), dict):
            raw = b['raw_binding']
            if not all(k in raw for k in ('reference', 'query')):
                return 0
            rids = {raw['reference']['input_tensor_hash']}
            qids = {raw['query']['input_tensor_hash']}
        else:
            domain = b.get('dataset', row.get('dataset'))
            rids = rgb_inputs.get((domain, row.get('reference_rgb_hash')), set())
            qids = rgb_inputs.get((domain, row.get('query_rgb_hash')), set())
        count = add_task(rids, row.get('reference_mask_hash'), row.get('reference_mask_path'), qids)
        for ref in b.get('references', []):
            if isinstance(ref, dict) and 'raw' in ref and 'view' in ref:
                count += add_task({ref['raw']['input_sha256']}, ref['view']['mask_hash'], ref['view']['mask_path'], qids)
                qid = b.get('source_observations', {}).get('query')
                if qid and 'id' in ref:
                    excluded.add(digest([ref['id'], qid]))
        variants = b.get('variants', {})
        for variant in variants.values() if isinstance(variants, dict) else []:
            if isinstance(variant, dict):
                count += add_task(rids, variant.get('array_hash'), variant.get('path'), qids)
        return count

    originals = lab.bindings() + lab.ext_bindings()
    for b in originals:
        recognized['original_lab_bindings'] += bool(consume(b))
    paths = {P0 / (n + '.json') for n in ('explore', 'confirm', 'kshot')}
    paths.update({R5 / 'manifest.json', ASSETS / 'a/reference_mask_task_trace_20261010/pairs.json'})
    for pattern in ('*/manifest*.json', '*/*/manifest*.json', '*/*/*/manifest*.json',
                    '*/*bindings*.json', '*/*/*bindings*.json', '*/pairs.json', '*/*/pairs.json'):
        paths.update((ASSETS / 'a').glob(pattern))
    paths = sorted(p for p in paths if OUT not in p.parents and p.is_file())
    for path in paths:
        value = read(path)
        if isinstance(value, list):
            entries = value
        elif isinstance(value, dict):
            entries = next((value[k] for k in ('rows', 'bindings', 'pairs', 'episodes', 'tasks')
                            if isinstance(value.get(k), list)), [])
        else:
            entries = []
        count = sum(bool(consume(b)) for b in entries)
        recognized[str(path)] = count
        sources[str(path)] = dict(sha256=sha(path), rows=len(entries), recognized_task_rows=count)
    return excluded, tasks, old_masks, sources, dict(recognized)


def eligible_pairs(pool, excluded, excluded_tasks, group):
    """Enumerate metadata-eligible pairs, then seed-shuffle without outcomes."""
    pool = sorted(pool, key=lambda o: o['id'])
    rng = random.Random(int(digest([SEED, group])[:16], 16))
    pairs, seen = [], set()
    candidates = ((r, q) for r in pool for q in pool)
    for r, q in candidates:
        sig = digest([r['id'], q['id']])
        task = input_task(r['raw']['input_sha256'], r['view']['mask_hash'], q['raw']['input_sha256'])
        if (r['photo'] == q['photo'] or r['view']['rgb_hash'] == q['view']['rgb_hash'] or
                r['raw']['input_sha256'] == q['raw']['input_sha256'] or sig in excluded or
                task in excluded_tasks or task in seen):
            continue
        pairs.append((r, q)); seen.add(task)
    rng.shuffle(pairs)
    return pairs


def prepare():
    began = time.monotonic()
    freeze_runtime()
    lab, pc, pm = modules()
    pc.setup('cpu')
    import torch
    torch.set_num_threads(2)
    if (OUT / 'config.json').exists():
        config = checked_config()
        print(dict(state='S2_CONFIRM_ALREADY_FROZEN', n=config['n'], manifest=str(OUT / 'manifest.json')), flush=True)
        return
    observations = read(P0 / 'observations.json')
    assert len(observations) == 11291
    excluded, excluded_tasks, old_masks, sources, exclusion_counts = exclusion_registry(lab, pm, observations)
    write(OUT / 'exclusion_sources.json', sources)
    pools = defaultdict(list)
    for o in observations:
        pools[(o['dataset'], str(o['fold']), str(o['class_id']))].append(o)
    rows, stats, selected_tasks = [], {}, set()
    for domain in DOMAINS:
        groups = sorted(g for g in pools if g[0] == domain)
        random.Random(int(digest([SEED, domain, 'class_order'])[:16], 16)).shuffle(groups)
        proposals = {g: eligible_pairs(pools[g], excluded, excluded_tasks, g) for g in groups}
        selected, offsets, rejected = [], dict.fromkeys(groups, 0), Counter()
        while len(selected) < PER_DOMAIN:
            any_added = False
            for g in groups:
                chosen = None
                while offsets[g] < len(proposals[g]):
                    r, q = proposals[g][offsets[g]]; offsets[g] += 1
                    reference_hash = canonical_reference_hash(r['view']['mask_path'], r['view']['mask_hash'])
                    ri, qi = r['raw']['input_sha256'], q['raw']['input_sha256']
                    canonical_task = input_task(ri, reference_hash, qi)
                    if canonical_task in selected_tasks:
                        rejected['duplicate_selected_canonical_task'] += 1
                        continue
                    duplicate_old = False
                    for old_hash, old_path in old_masks.get((ri, qi), {}).items():
                        if not Path(old_path).is_file():
                            duplicate_old = True
                            rejected['old_same_RGB_pair_reference_input_unavailable'] += 1
                            break
                        if reference_hash == canonical_reference_hash(old_path, old_hash):
                            duplicate_old = True
                            rejected['old_canonical_reference_task'] += 1
                            break
                    if duplicate_old:
                        continue
                    chosen = r, q, canonical_task, reference_hash
                    break
                if chosen is None:
                    continue
                r, q, canonical_task, reference_hash = chosen
                # make_pair may write only iSAID ignore protocol masks, to pm.OUT.
                b = pm.make_pair(copy.deepcopy(r), copy.deepcopy(q), 's2_confirm', 'old_photo_new_inference_task')
                b['index'] = 'cf13_' + b['pair_signature'][:20]
                b['episode_id'] = b['row']['episode_id'] = 'x1012s2_confirm1000/' + b['index']
                b['row']['query_gt_role'] = 'scoring only after candidate freeze and all natural predictions sealed'
                b['cohort'] = 'x1012s2_confirm1000'
                b['inference_task_signature'] = canonical_task
                b['reference_mask_canonical1024_sha256'] = reference_hash
                b['exposure'].update(baseline_predictions_previously_scored=False,
                                     new_inference_task=True,
                                     validation_scope='new R-input/reference-mask1024/Q-input task of preexisting photographs; not photo holdout')
                assert b['pair_signature'] not in excluded
                for role in ('reference', 'query'):
                    assert Path(b['row'][role + '_mask_path']).is_file()
                    assert (ASSETS / b['row'][role + '_path']).is_file()
                for name in ('reference_raw', 'query_raw_identity'):
                    assert Path(b['fit'][name]['payload_path']).is_file()
                    assert Path(b['fit'][name]['entry_path']).is_file()
                selected_tasks.add(canonical_task)
                selected.append(b); any_added = True
                if len(selected) == PER_DOMAIN:
                    break
            if not any_added:
                raise RuntimeError('Insufficient new directed pairs for ' + domain)
        rows.extend(selected)
        slots = Counter((str(b['row']['fold']), str(b['row']['global_class_id'])) for b in selected)
        stats[domain] = dict(n=len(selected), eligible_metadata_pairs=sum(len(proposals[g]) for g in groups),
                             available_slots=sum(bool(proposals[g]) for g in groups),
                             canonical_identity_rejections=dict(rejected),
                             selected_slots=len(slots), slot_counts={f'{f}/{c}': n for (f, c), n in sorted(slots.items())},
                             unique_reference_photos=len({str((ASSETS / b['row']['reference_path']).resolve()) for b in selected}),
                             unique_query_photos=len({str((ASSETS / b['row']['query_path']).resolve()) for b in selected}))
        print(dict(prepared_domain=domain, n=len(selected), eligible_metadata_pairs=stats[domain]['eligible_metadata_pairs'],
                   canonical_identity_rejections=dict(rejected)), flush=True)
    assert len(rows) == TARGET_N and len({b['pair_signature'] for b in rows}) == TARGET_N
    assert len({b['index'] for b in rows}) == len(selected_tasks) == TARGET_N
    write(OUT / 'excluded_pairs.json', sorted(excluded))
    write(OUT / 'excluded_input_tasks.json', sorted(excluded_tasks))
    write(OUT / 'manifest.json', rows)
    external_sources = [P0 / (n + '.json') for n in ('observations', 'explore', 'confirm', 'kshot')]
    external_sources += [lab.BINDINGS, lab.BASIS,
                         ASSETS / 'third_party/foris_official/models/foris.py',
                         ASSETS / 'third_party/foris_official/utils/data.py',
                         ASSETS / 'third_party/foris_official/utils/clustering.py',
                         ASSETS / 'third_party/foris_official/utils/refinement.py',
                         ASSETS / 'runtime/macos/crf/ics_permutohedral_m4.so',
                         ASSETS / 'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so']
    config = dict(state='S2_CONFIRM_FROZEN_UNSCORED', n=TARGET_N, seed=SEED, per_domain=PER_DOMAIN,
                  sample_rule='seeded per-(fold,class) candidate order; seeded slot round-robin; no scores or target sizes consulted',
                  independence='unused directed observation pairs plus actual R-input/reference-mask1024/Q-input identity; all photographs preexisting',
                  inference=dict(workers=WORKERS, threads_per_worker=2, device='cpu', new_encoder_calls=0, query_GT_in_inference=False,
                                 optional_mean='exact frozen MEAN_a0.25_l16, FP16 feature handoff; no CRF, as historical complete control'),
                  query_GT_scores_read=0, exclusion_counts=exclusion_counts, excluded_unique_pairs=len(excluded),
                  excluded_native_mask_input_tasks=len(excluded_tasks),
                  exclusion_sources_sha256=sha(OUT / 'exclusion_sources.json'),
                  excluded_input_tasks_sha256=sha(OUT / 'excluded_input_tasks.json'),
                  stats=stats, source_sha256={str(p): sha(p) for p in external_sources},
                  runtime_sources_sha256=sha(OUT / 'runtime_sources.json'),
                  frozen_main_sha256=sha(OUT / 'frozen/x1012s2_confirm.py'),
                  manifest_sha256=sha(OUT / 'manifest.json'), excluded_pairs_sha256=sha(OUT / 'excluded_pairs.json'),
                  prepare_seconds=time.monotonic() - began,
                  ignore_preparation='Only official iSAID ignore convention is prepared when an original reference becomes query; no foreground scoring')
    write(OUT / 'config.json', config)
    print(dict(state=config['state'], n=TARGET_N, seed=SEED, stats=stats,
               excluded_unique_pairs=len(excluded), manifest_sha256=config['manifest_sha256'],
               prepare_seconds=config['prepare_seconds']), flush=True)


def checked_config():
    config = read(OUT / 'config.json')
    assert sha(OUT / 'manifest.json') == config['manifest_sha256']
    assert sha(OUT / 'excluded_pairs.json') == config['excluded_pairs_sha256']
    assert sha(OUT / 'runtime_sources.json') == config['runtime_sources_sha256']
    for record in read(OUT / 'runtime_sources.json'):
        assert sha(record['snapshot']) == record['sha256'], 'Runtime source changed'
    for path, value in config['source_sha256'].items():
        assert sha(path) == value, 'Frozen external input changed: ' + path
    return config


def init_worker():
    global LAB, PC, PM, HOST, ADAPTER, RUN, LOAD, RM
    LAB, PC, PM = modules()
    PC.setup('cpu')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    HOST, ADAPTER = PC.host_cpu()
    from ics.foris import run_foris
    from ics.official_data import load_inputs
    RUN, LOAD = run_foris, load_inputs
    try:
        RM = importlib.import_module('repo_methods')
    except Exception:
        RM = None


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def infer(binding):
    import numpy as np
    import torch
    began = time.monotonic()
    name = PC.key(binding)
    record_path = OUT / 'baseline/records' / (name + '.json')
    if record_path.exists():
        record = read(record_path)
        for prefix in ('prediction', 'field', 'mean_prediction', 'mean_field'):
            if record.get(prefix + '_path'):
                assert sha(record[prefix + '_path']) == record[prefix + '_sha256']
        return record
    prediction = OUT / 'baseline/predictions' / (name + '.npz')
    field = OUT / 'baseline/fields' / (name + '.npz')
    if prediction.exists() or field.exists():
        raise RuntimeError('Preserve incomplete inference output: ' + name)
    ADAPTER.bind(binding)
    with PC.inference_only(binding):
        reference, mask, query = LOAD(binding['row'], ASSETS)
        final, got, _, _ = RUN(HOST, reference, mask, query)
        score = got['score'].cpu().numpy().copy()
        cli = final.cpu().numpy()
        original_hw = tuple(binding['row']['query_size_hw'])
        original = torch.nn.functional.interpolate(final.float()[None, None], size=original_hw,
                        mode='bilinear', align_corners=False)[0, 0] > .5
        save_npz(prediction, **{'cli/foris.crf': PC.pack(cli), 'original/foris.crf': PC.pack(original.numpy()),
                               'original_hw': np.asarray(original_hw, np.int32)})
        save_npz(field, score=score)
        record = dict(index=name, episode_id=binding['row']['episode_id'], dataset=binding['dataset'],
                      pair_signature=binding['pair_signature'], prediction_path=str(prediction),
                      prediction_sha256=sha(prediction), field_path=str(field), field_sha256=sha(field),
                      foris_seconds=time.monotonic() - began, new_encoder_calls=ADAPTER.encoder_calls,
                      query_GT_reads=0, query_GT_scored=False, worker_pid=os.getpid(),
                      mean_status='PENDING')
        # Preserve completed FoRIS even if the optional MEAN computation fails.
        write(record_path, record)
        mean_began = time.monotonic()
        try:
            if RM is None:
                raise RuntimeError('MEAN module unavailable; FoRIS complete')
            b = copy.deepcopy(binding)
            b['score64'] = dict(path=str(field), key='score', sha256=record['field_sha256'])
            mean = RM.mean_field(b)
            mean_cli = RM.field_mask(mean).numpy()
            mean_original = torch.nn.functional.interpolate(torch.from_numpy(np.ascontiguousarray(mean))[None, None],
                                size=original_hw, mode='bilinear', align_corners=False)[0, 0].numpy() > .5
            mean_prediction = OUT / 'baseline/mean_predictions' / (name + '.npz')
            mean_field = OUT / 'baseline/mean_fields' / (name + '.npz')
            save_npz(mean_prediction, **{'cli/mean': PC.pack(mean_cli), 'original/mean': PC.pack(mean_original),
                                        'original_hw': np.asarray(original_hw, np.int32)})
            save_npz(mean_field, field=mean)
            record.update(mean_status='COMPLETE', mean_prediction_path=str(mean_prediction),
                          mean_prediction_sha256=sha(mean_prediction), mean_field_path=str(mean_field),
                          mean_field_sha256=sha(mean_field), mean_seconds=time.monotonic() - mean_began)
        except Exception as exc:
            record.update(mean_status='FAILED_OPTIONAL', mean_error=repr(exc), mean_traceback=traceback.format_exc(),
                          mean_seconds=time.monotonic() - mean_began)
        assert ADAPTER.encoder is None and ADAPTER.encoder_calls == 0
    record['seconds'] = time.monotonic() - began
    write(record_path, record)
    ADAPTER.memo.clear()
    del got
    return record


def produce():
    config = checked_config()
    if (OUT / 'baseline/sealed.json').exists():
        sealed = read(OUT / 'baseline/sealed.json')
        print(dict(state=sealed['state'], n=sealed['n'], already_sealed=True), flush=True)
        return
    rows = read(OUT / 'manifest.json')
    for name in ('predictions', 'fields', 'mean_predictions', 'mean_fields', 'records'):
        (OUT / 'baseline' / name).mkdir(parents=True, exist_ok=True)
    began, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'), initializer=init_worker) as pool:
        futures = {pool.submit(infer, b): b['index'] for b in rows}
        for count, future in enumerate(as_completed(futures), 1):
            r = future.result(); records.append(r)
            if count <= 10 or count % 25 == 0:
                progress = dict(completed=count, total=len(rows), dataset=r['dataset'],
                                seconds=round(time.monotonic() - began, 3), mean_status=r['mean_status'],
                                new_encoder_calls=0, query_GT_reads=0)
                write(OUT / 'baseline/progress.json', progress)
                print(progress, flush=True)
    indexed = {r['index']: r for r in records}
    bound = copy.deepcopy(rows)
    for b in bound:
        r = indexed[b['index']]
        b['old_arms']['B'] = dict(path=r['prediction_path'], sha256=r['prediction_sha256'],
                                  cli_key='cli/foris.crf', original_key='original/foris.crf')
        b['score64'] = dict(path=r['field_path'], sha256=r['field_sha256'], key='score')
        if r['mean_status'] == 'COMPLETE':
            b['old_arms']['G'] = dict(path=r['mean_prediction_path'], sha256=r['mean_prediction_sha256'],
                                      cli_key='cli/mean', original_key='original/mean')
            b['mean64'] = dict(path=r['mean_field_path'], sha256=r['mean_field_sha256'], key='field')
    write(OUT / 'manifest.baselines.json', bound)
    sealed = dict(state='S2_CONFIRM_BASELINES_SEALED_UNSCORED', n=len(rows),
                  seconds=time.monotonic() - began, foris_complete=len(records),
                  mean_complete=sum(r['mean_status'] == 'COMPLETE' for r in records),
                  new_encoder_calls=0, query_GT_reads=0, query_GT_scored=False,
                  manifest_sha256=config['manifest_sha256'], config_sha256=sha(OUT / 'config.json'),
                  manifest_baselines_sha256=sha(OUT / 'manifest.baselines.json'),
                  records=[indexed[b['index']] for b in rows])
    assert len(records) == TARGET_N and all(r['new_encoder_calls'] == 0 and r['query_GT_reads'] == 0 for r in records)
    write(OUT / 'baseline/sealed.json', sealed)
    print({k: v for k, v in sealed.items() if k != 'records'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'produce'))
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else produce()
