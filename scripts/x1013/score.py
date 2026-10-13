"""Scoring-only interfaces, strict tied-level oracles, and W02 aggregation."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
import time
import numpy as np
from scipy import sparse
import common as c

ARMS = {
    'T0': ('candidate',),
    'T1': ('foris', 'foris_noCRF', 'mean', 'T', 'oracle_candidate', 'oracle_foris'),
    'T2': tuple(f'A{i}' for i in range(8)),
    'T3': ('VP_no_fallback', 'G1'),
    'T4': ('F6a', 'F6b', 'F6c'),
    'T5': ('F1', 'F1b'),
    'T6': ('S-self', 'S-avg', 'S-ref'),
    'T7': ('C1', 'C2'),
    'appendix': ('MU_control',),
}


def strict_oracle(field, gt, valid):
    """Diagnostic maximum IoU over whole equal-score groups (no tied splits)."""
    values = c.up(field)[valid].ravel()
    labels = gt[valid].ravel()
    if not values.size:
        return [0, 0, 0]
    order = np.argsort(-values)
    ranked = values[order]
    cumulative = np.cumsum(labels[order], dtype=np.int64)
    ends = np.flatnonzero(np.r_[ranked[:-1] != ranked[1:], True])
    intersections = cumulative[ends]
    areas = ends + 1
    unions = int(labels.sum()) + areas - intersections
    if not len(ends) or not labels.any():
        return [0, int(labels.sum()), 0]
    selected = int(np.argmax(intersections / np.maximum(unions, 1)))
    return [int(intersections[selected]), int(unions[selected]), int(areas[selected])]


def t1_identity_xor(binding, archive):
    if 'mean_xor' in archive:
        return int(archive['mean_xor'])
    if 'G' not in binding['old_arms']:
        return -1
    import repo_methods
    recomputed = repo_methods.field_mask(archive['mean']).numpy()
    cached = c.rc.lab.arm_cli(binding, 'G').numpy()
    return int(np.count_nonzero(recomputed ^ cached))


def score_one(job):
    task, binding = job
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    record = c.metadata(binding, fields)
    record.update(gt=int(gt.sum()), touched=int(fields['touched']),
                  reference_fraction=float(fields['reference_fraction']))
    with np.load(c.prediction_path(task, binding), allow_pickle=False) as archive:
        for name in archive.files:
            record[name] = c.iu(c.unpack(archive[name]), gt, valid)
    if task == 'T0':
        record['foris'] = c.iu(c.rc.lab.arm_cli(binding, 'B').numpy(), gt, valid)
    elif task == 'T1':
        import repo_methods
        record['oracle_candidate'] = strict_oracle(c.foreground_field(fields), gt, valid)
        record['oracle_foris'] = strict_oracle(repo_methods.foris_score(binding), gt, valid)
        with np.load(c.task_path('T1') / 'fields' / (c.key(binding) + '.npz'), allow_pickle=False) as archive:
            mismatch = t1_identity_xor(binding, archive)
        record['mean_xor'] = None if mismatch < 0 else mismatch
        if 'G' in binding['old_arms']:
            cached = c.rc.lab.arm_cli(binding, 'G').numpy()
            record['mean_cached'] = c.iu(cached, gt, valid)
    elif task == 'T6':
        with np.load(c.task_path('T6') / 'fields' / (c.key(binding) + '.npz'), allow_pickle=False) as archive:
            record['rewritten'] = bool(archive['rewritten'])
    return record


def read_records(task):
    path = c.task_path(task) / 'iu.jsonl.gz'
    if not path.exists():
        return []
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def save_records(path, records):
    path = c.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(temporary, 'wt') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def score_task(task, workers=6):
    destination = c.task_path(task)
    complete = c.read(destination / 'complete.json')
    bindings = c.rows()
    if complete['n'] != len(bindings) or any(not c.prediction_path(task, b).exists() for b in bindings):
        raise RuntimeError(f'{task}: predictions must all exist before scoring')
    if (destination / 'scored.json').exists():
        return read_records(task)
    # Preserve the held-out T half until the revised E2 instruction explicitly authorizes it.
    scored_bindings = [b for b in bindings if task not in ('T1', 'T6') or c.half(b) == 'S']
    print(f'{task} score: {len(scored_bindings)} ({len(bindings)} predictions available)', flush=True)
    started, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=c.initialize) as pool:
        for number, record in enumerate(pool.map(score_one, ((task, b) for b in scored_bindings), chunksize=8), 1):
            records.append(record)
            if number % 1000 == 0 or number == len(scored_bindings):
                print(f'{task} score: {number}/{len(scored_bindings)}, {time.monotonic() - started:.1f}s', flush=True)
    if task == 'T0':
        with gzip.open(c.ROOT / 'evidence/research/w02_scale13613_iu.jsonl.gz', 'rt') as stream:
            old = {str(record['index']): record for record in map(json.loads, stream)}
        failures = []
        if len(old) != len(records):
            failures.append({'old_n': len(old), 'new_n': len(records)})
        for record in records:
            previous = old.get(record['index'])
            if previous is None or record['candidate'] != previous['un50'] or record['foris'] != previous['foris']:
                failures.append({'index': record['index'], 'new': record['candidate'],
                                 'old': None if previous is None else previous['un50']})
        if failures:
            c.write(destination / 'gate_failure.json', {'gate': 'all 13613 integer I/U equal W02',
                                                        'failures': failures})
            raise RuntimeError(f'T0 identity gate failed on {len(failures)} entries; stop without redefining the method')
    identity_gate = 'PASS' if task in ('T0', 'T2') else 'N/A'
    if task == 'T1':
        c.initialize()
        mismatches = []
        for binding in bindings:
            with np.load(destination / 'fields' / (c.key(binding) + '.npz'), allow_pickle=False) as archive:
                value = t1_identity_xor(binding, archive)
            if value >= 0:
                mismatches.append((binding['dataset'], value))
        values = np.asarray([value for _, value in mismatches], dtype=np.int64)
        identity = {'scope': 'all cached masks; label-free', 'n_cached': int(values.size),
                    'n_over_4_pixels': int(np.count_nonzero(values > 4)),
                    'xor_pixels': ({f'p{q}': float(np.percentile(values, q)) for q in (50, 90, 99)}
                                   | {'max': int(values.max())}) if values.size else {}}
        cached_s = [record for record in records if 'mean_cached' in record]
        identity['cached_S_n'] = len(cached_s)
        identity['miou_delta_recomputed_minus_cached_S'] = {
            domain: (miou([r for r in cached_s if r['dataset'] == domain], 'mean') -
                     miou([r for r in cached_s if r['dataset'] == domain], 'mean_cached'))
            for domain in c.DOMAINS
            if any(r['dataset'] == domain for r in cached_s)
        }
        c.write(destination / 'identity_summary.json', identity)
        identity_gate = 'REPORT_ONLY'
    save_records(destination / 'iu.jsonl.gz', records)
    c.write(destination / 'scored.json', {'n': len(records), 'pool_n': len(bindings),
            'scope': 'S' if task in ('T1', 'T6') else 'all', 'seconds': time.monotonic() - started,
            'identity_gate': identity_gate})
    return records


def merged_records(include_t9=True):
    base = read_records('T0')
    if not base:
        return []
    indexed = {record['index']: record for record in base}
    for task in (*(f'T{i}' for i in range(1, 8)), *(('T9', 'appendix') if include_t9 else ())):
        for record in read_records(task):
            indexed[record['index']].update(record)
    return base


def miou(records, arm):
    slots = {}
    for record in records:
        values = record[arm]
        slot = (record['fold'], record['cls'])
        total = slots.setdefault(slot, [0, 0])
        total[0] += values[0]
        total[1] += values[1]
    return float(np.mean([100 * i / max(u, 1) for i, u in slots.values()])) if slots else None


def episode_iou(record, arm):
    return 100 * record[arm][0] / max(record[arm][1], 1)


def area_band(record):
    fraction = record['gt'] / 1048576
    return '<2%' if fraction < .02 else ('2-20%' if fraction <= .2 else '>20%')


def domain_scores(records, arm):
    return {domain: miou([record for record in records if record['dataset'] == domain], arm)
            for domain in c.DOMAINS}


def paired_ci(records, a, b, rng):
    photos = list(dict.fromkeys(record['photo'] for record in records))
    slots = list(dict.fromkeys((record['fold'], record['cls']) for record in records))
    p_index = {photo: i for i, photo in enumerate(photos)}
    s_index = {slot: i for i, slot in enumerate(slots)}
    p = [p_index[record['photo']] for record in records]
    s = [s_index[(record['fold'], record['cls'])] for record in records]
    shape = (len(photos), len(slots))
    weights = np.stack([np.bincount(rng.integers(len(photos), size=len(photos)),
                                  minlength=len(photos)) for _ in range(500)])
    presence = sparse.csr_matrix((np.ones(len(records)), (p, s)), shape=shape)
    represented = weights @ presence > 0
    result = []
    for arm in (a, b):
        i = sparse.csr_matrix(([record[arm][0] for record in records], (p, s)), shape=shape)
        u = sparse.csr_matrix(([record[arm][1] for record in records], (p, s)), shape=shape)
        values = 100 * (weights @ i) / np.maximum(weights @ u, 1)
        result.append((values * represented).sum(1) / represented.sum(1))
    return np.percentile(result[0] - result[1], [2.5, 97.5]).tolist()
