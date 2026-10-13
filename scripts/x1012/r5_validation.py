"""R5: 500 unused directed cached pairs and unscored complete baselines.

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

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'
sys.dont_write_bytecode = True

REPO = Path('/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch')
ASSETS = REPO.parent / 'cv_data'
OUT = ASSETS / 'a/x1012_R5_validation'
P0 = ASSETS / 'a/x1012_P0'
RUNTIME = OUT / 'runtime'
SEED = 2026101203
PER_DOMAIN = 50
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
    c1 = ASSETS / 'a/x1012_C1/frozen'
    b4 = ASSETS / 'a/x1012_B4/seed150/frozen'
    sources = {
        'lab.py': [REPO / 'scripts/x1012/lab.py', c1 / '8_lab.py'],
        'plan_common.py': [REPO / 'scripts/x1012/plan_common.py', c1 / '6_plan_common.py'],
        'p0_manifest.py': [REPO / 'scripts/x1012/p0_manifest.py'],
        'cached_dino.py': [REPO / 'scripts/cached_dino.py'],
        'raw_feature_cache.py': [REPO / 'scripts/raw_feature_cache.py', c1 / '10_raw_feature_cache.py'],
        'ics/foris.py': [REPO / 'src/ics/foris.py'],
        'ics/official_data.py': [REPO / 'src/ics/official_data.py'],
        'ics/native_basis.py': [REPO / 'src/ics/native_basis.py'],
        'ics/m4_crf.py': [REPO / 'src/ics/m4_crf.py'],
        'ics/m4_crf_cached.py': [REPO / 'src/ics/m4_crf_cached.py', b4 / '7_m4_crf_cached.py'],
        'ics/methods/rcg.py': [REPO / 'src/ics/methods/rcg.py'],
        'ics/methods/mean_control.py': [b4 / '6_mean_control.py'],
        'repo_methods.py': [b4 / '5_repo_methods.py'],
    }
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
            target.write_text('"""R5 isolated replay package."""\n')
        records.append(dict(source='R5 minimal package marker', snapshot=str(target), sha256=sha(target)))
    frozen = OUT / 'frozen/r5_validation.py'
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


def eligible_pairs(pool, excluded, group):
    """At most 50 deterministic proposals per slot, enough for class round-robin."""
    pool = sorted(pool, key=lambda o: o['id'])
    rng = random.Random(int(digest([SEED, group])[:16], 16))
    pairs, seen = [], set()
    if len(pool) <= 64:
        candidates = [(r, q) for r in pool for q in pool if r['photo'] != q['photo']]
        rng.shuffle(candidates)
    else:
        candidates = ((pool[i], pool[j]) for i, j in
                      (rng.sample(range(len(pool)), 2) for _ in range(max(10000, len(pool) * 100))))
    for r, q in candidates:
        sig = digest([r['id'], q['id']])
        if r['photo'] == q['photo'] or sig in excluded or sig in seen:
            continue
        pairs.append((r, q)); seen.add(sig)
        if len(pairs) == PER_DOMAIN:
            break
    return pairs


def prepare():
    began = time.monotonic()
    freeze_runtime()
    lab, pc, pm = modules()
    if (OUT / 'config.json').exists():
        config = checked_config()
        print(dict(state='R5_ALREADY_FROZEN', n=config['n'], manifest=str(OUT / 'manifest.json')), flush=True)
        return
    observations = read(P0 / 'observations.json')
    assert len(observations) == 11291
    originals = lab.bindings() + lab.ext_bindings()
    excluded, exclusion_counts = set(), {}
    for origin, entries in [('original_lab_bindings', originals),
                            ('P0_explore', read(P0 / 'explore.json')),
                            ('P0_confirm', read(P0 / 'confirm.json'))]:
        signatures = {signature_of_binding(b, pm) for b in entries}
        excluded.update(signatures)
        exclusion_counts[origin] = dict(rows=len(entries), unique_pairs=len(signatures))
    kshot = read(P0 / 'kshot.json')
    k_pairs = {digest([r['id'], b['source_observations']['query']]) for b in kshot for r in b['references']}
    excluded.update(k_pairs)
    exclusion_counts['P0_kshot_all_references'] = dict(queries=len(kshot), unique_pairs=len(k_pairs))
    pools = defaultdict(list)
    for o in observations:
        pools[(o['dataset'], str(o['fold']), str(o['class_id']))].append(o)
    rows, stats = [], {}
    for domain in DOMAINS:
        groups = sorted(g for g in pools if g[0] == domain)
        random.Random(int(digest([SEED, domain, 'class_order'])[:16], 16)).shuffle(groups)
        proposals = {g: eligible_pairs(pools[g], excluded, g) for g in groups}
        selected, round_index = [], 0
        while len(selected) < PER_DOMAIN:
            any_added = False
            for g in groups:
                if round_index >= len(proposals[g]):
                    continue
                r, q = proposals[g][round_index]
                # make_pair may write only iSAID ignore protocol masks, to pm.OUT.
                b = pm.make_pair(copy.deepcopy(r), copy.deepcopy(q), 'r5_validation', 'old_photo_new_directed_pair')
                b['index'] = 'r5_' + b['pair_signature'][:20]
                b['episode_id'] = b['row']['episode_id'] = 'x1012_R5_validation/' + b['index']
                b['row']['query_gt_role'] = 'scoring only after candidate freeze and all natural predictions sealed'
                b['cohort'] = 'x1012_R5_validation_500'
                b['exposure'].update(baseline_predictions_previously_scored=False,
                                     validation_scope='new directed pair of preexisting observations; not photo holdout')
                assert b['pair_signature'] not in excluded
                selected.append(b); any_added = True
                if len(selected) == PER_DOMAIN:
                    break
            if not any_added:
                raise RuntimeError('Insufficient new directed pairs for ' + domain)
            round_index += 1
        rows.extend(selected)
        slots = Counter((str(b['row']['fold']), str(b['row']['global_class_id'])) for b in selected)
        stats[domain] = dict(n=len(selected), available_slots=sum(bool(proposals[g]) for g in groups),
                             selected_slots=len(slots), slot_counts={f'{f}/{c}': n for (f, c), n in sorted(slots.items())},
                             unique_reference_photos=len({str((ASSETS / b['row']['reference_path']).resolve()) for b in selected}),
                             unique_query_photos=len({str((ASSETS / b['row']['query_path']).resolve()) for b in selected}))
    assert len(rows) == 500 and len({b['pair_signature'] for b in rows}) == 500
    assert len({b['index'] for b in rows}) == 500
    write(OUT / 'excluded_pairs.json', sorted(excluded))
    write(OUT / 'manifest.json', rows)
    external_sources = [P0 / (n + '.json') for n in ('observations', 'explore', 'confirm', 'kshot')]
    external_sources += [lab.BINDINGS, lab.BASIS,
                         ASSETS / 'third_party/foris_official/models/foris.py',
                         ASSETS / 'third_party/foris_official/utils/data.py',
                         ASSETS / 'third_party/foris_official/utils/clustering.py',
                         ASSETS / 'third_party/foris_official/utils/refinement.py',
                         ASSETS / 'runtime/macos/crf/ics_permutohedral_m4.so',
                         ASSETS / 'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so']
    config = dict(state='R5_FROZEN_UNSCORED', n=500, seed=SEED, per_domain=PER_DOMAIN,
                  sample_rule='seeded per-(fold,class) candidate order; seeded slot round-robin; no scores or target sizes consulted',
                  independence='unused directed observation pairs; all photographs preexisting and potentially previously exposed',
                  inference=dict(workers=2, threads_per_worker=4, device='cpu', new_encoder_calls=0, query_GT_in_inference=False,
                                 optional_mean='exact frozen MEAN_a0.25_l16, FP16 feature handoff; no CRF, as historical complete control'),
                  query_GT_scores_read=0, exclusion_counts=exclusion_counts, excluded_unique_pairs=len(excluded),
                  stats=stats, source_sha256={str(p): sha(p) for p in external_sources},
                  runtime_sources_sha256=sha(OUT / 'runtime_sources.json'),
                  frozen_main_sha256=sha(OUT / 'frozen/r5_validation.py'),
                  manifest_sha256=sha(OUT / 'manifest.json'), excluded_pairs_sha256=sha(OUT / 'excluded_pairs.json'),
                  prepare_seconds=time.monotonic() - began,
                  ignore_preparation='Only official iSAID ignore convention is prepared when an original reference becomes query; no foreground scoring')
    write(OUT / 'config.json', config)
    print(dict(state=config['state'], n=500, seed=SEED, stats=stats,
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
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'), initializer=init_worker) as pool:
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
    sealed = dict(state='R5_BASELINES_SEALED_UNSCORED', n=len(rows),
                  seconds=time.monotonic() - began, foris_complete=len(records),
                  mean_complete=sum(r['mean_status'] == 'COMPLETE' for r in records),
                  new_encoder_calls=0, query_GT_reads=0, query_GT_scored=False,
                  manifest_sha256=config['manifest_sha256'], config_sha256=sha(OUT / 'config.json'),
                  manifest_baselines_sha256=sha(OUT / 'manifest.baselines.json'),
                  records=[indexed[b['index']] for b in rows])
    assert len(records) == 500 and all(r['new_encoder_calls'] == 0 and r['query_GT_reads'] == 0 for r in records)
    write(OUT / 'baseline/sealed.json', sealed)
    print({k: v for k, v in sealed.items() if k != 'records'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'produce'))
    args = parser.parse_args()
    prepare() if args.mode == 'prepare' else produce()
