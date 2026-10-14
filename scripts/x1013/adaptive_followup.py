"""Offline S-half comparison of three scale-gated compositions.

Reads frozen T0/T2/T5 caches and E1 C1 masks only. It does not run an encoder,
and it writes outputs under ../cv_data/a/x1013_adaptive_followup/.

    python -B scripts/x1013/adaptive_followup.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT.parent / 'cv_data'
OUT = ASSETS / 'a/x1013_adaptive_followup'
EVIDENCE = ROOT / 'evidence/research/x1013_adaptive_followup_iu.jsonl.gz'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c
import w02_scale_eval as w02

DOMAINS = ('coco', 'lvis', 'pascal_part', 'paco_part', 'suim',
           'isaid', 'isic', 'lung', 'fundus', 'deepglobe_road')
PX = 1024 * 1024
SEED = 20261013
BOOTSTRAPS = 300


def records(path: Path) -> dict[str, dict]:
    with gzip.open(path, 'rt') as stream:
        return {str(row['index']): row for row in map(json.loads, stream)}


def key(index: object) -> str:
    value = str(index)
    return f'{int(value):06d}' if value.isdigit() else value


def packed(path: Path, arm: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        return c.unpack(z[arm])


def score_mask(mask: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> list[int]:
    pred = np.asarray(mask, dtype=bool) & valid
    return [int(np.count_nonzero(pred & gt)),
            int(np.count_nonzero(pred | gt)),
            int(np.count_nonzero(pred))]


def hash_array(hasher, index: str, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    header = json.dumps([index, name, list(array.shape), array.dtype.str],
                        separators=(',', ':')).encode()
    hasher.update(len(header).to_bytes(4, 'big'))
    hasher.update(header)
    hasher.update(array.tobytes())


def normalized_up(field: np.ndarray) -> np.ndarray:
    value = c.rc.lab.up(torch.as_tensor(field, dtype=torch.float32).reshape(64, 64))
    return c.rc.lab.minmax(value).cpu().numpy()


def solve_a3(fields: dict[str, np.ndarray]) -> np.ndarray:
    """Recreate the frozen T2 A3 score from saved arrays; no feature/model call."""
    if float(fields['reference_rho'][0]) > .9:
        primary = fields['c32'].reshape(64, 64)
    else:
        primary = fields['c_mod'].reshape(64, 64) + fields['addition_P'].reshape(64, 64)
    graph = c.sparse.csr_matrix((fields['L_data'], fields['L_indices'], fields['L_indptr']),
                                shape=(4096, 4096))
    return c.rc.solve(primary, fields['explanation'], graph)[0]


def f1b_safe(fields: dict[str, np.ndarray], labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """F1b with .22 score-SD and .35 area gates; unsafe clusters keep point scores."""
    field = fields['VP_UN'].astype(np.float32).reshape(64, 64)
    flat = field.ravel().astype(np.float64)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float64)
    sums = np.bincount(labels, weights=flat, minlength=len(counts))
    squares = np.bincount(labels, weights=flat * flat, minlength=len(counts))
    means = sums / np.maximum(counts, 1)
    std = np.sqrt(np.maximum(squares / np.maximum(counts, 1) - means * means, 0))
    safe = (std < .22) & (counts < .35 * flat.size)
    replacement = np.where(safe[labels], means[labels], flat).reshape(64, 64).astype(np.float32)
    mask = c.readout(.5 * field + .5 * replacement)
    return mask, {'n_clusters': int(len(counts)), 'n_safe': int(safe.sum()),
                  'largest_cluster_fraction': float(counts.max() / flat.size),
                  'max_cluster_std': float(std.max())}


def miou(rows: list[dict], arm: str, weights: np.ndarray | None = None) -> float:
    slots: dict[tuple[object, object], list[int]] = {}
    for i, row in enumerate(rows):
        weight = 1 if weights is None else int(weights[i])
        slot = (row['fold'], row['cls'])
        total = slots.setdefault(slot, [0, 0])
        total[0] += weight * row[arm][0]
        total[1] += weight * row[arm][1]
    values = [100 * i / u for i, u in slots.values() if u > 0]
    return float(np.mean(values)) if values else 0.0


def domain_table(rows: list[dict], arms: tuple[str, ...], baseline: str) -> dict[str, dict]:
    rng = np.random.default_rng(SEED)
    result = {}
    for domain in DOMAINS:
        items = [r for r in rows if r['dataset'] == domain]
        photo_ids = sorted({r['photo'] for r in items})
        photo_ix = {photo: i for i, photo in enumerate(photo_ids)}
        index = np.asarray([photo_ix[r['photo']] for r in items])
        draws = [np.bincount(rng.integers(0, len(photo_ids), len(photo_ids)),
                             minlength=len(photo_ids))[index]
                 for _ in range(BOOTSTRAPS)]
        base = [miou(items, baseline, weight) for weight in draws]
        table = {}
        for arm in arms:
            delta = miou(items, arm) - miou(items, baseline)
            diffs = [miou(items, arm, weight) - value for weight, value in zip(draws, base)]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            table[arm] = {'delta_pp': float(delta), 'ci95_pp': [float(lo), float(hi)],
                          'n': len(items)}
        result[domain] = table
    return result


def main() -> None:
    # Project cache initializer exposes only the frozen CPU readout helpers;
    # it has no encoder constructor or inference fallback.
    c.initialize()
    assert getattr(c.rc.pc, 'encoder', None) is None
    t0 = records(ASSETS / 'a/x1013_T0/iu.jsonl.gz')
    t2 = records(ASSETS / 'a/x1013_T2/iu.jsonl.gz')
    t5 = records(ASSETS / 'a/x1013_T5/iu.jsonl.gz')
    bindings = {str(b['index']): b for b in w02.rows_all(0)}
    selected = [t0[index] for index in sorted(t0, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
                if t0[index]['split'] == 'S']
    if len(selected) != 6752:
        raise RuntimeError(f'Expected the frozen 6,752-example S half; found {len(selected)}')

    out: list[dict] = []
    field_hash = hashlib.sha256()
    cluster_hash = hashlib.sha256()
    c1_hash = hashlib.sha256()
    large_n = 0
    large_by_domain: dict[str, int] = {}
    for number, base in enumerate(selected, 1):
        index = str(base['index'])
        if index not in bindings or index not in t2 or index not in t5:
            raise RuntimeError(f'Missing cached inputs for index {index}')
        a3_record = t2[index]['A3']
        if t2[index]['A4'] != base['candidate']:
            raise RuntimeError(f'T0/T2 candidate identity mismatch at {index}')
        is_large = base['candidate'][2] / PX > .25
        row = {k: base[k] for k in ('index', 'dataset', 'split', 'fold', 'cls', 'photo',
                                    'rho', 'gt', 'foris', 'candidate')}
        row['S1_noV_large'] = a3_record if is_large else base['candidate']
        if is_large:
            large_n += 1
            large_by_domain[base['dataset']] = large_by_domain.get(base['dataset'], 0) + 1
            binding = bindings[index]
            fields = c.load_fields(binding)
            gt, valid = c.truth(binding)

            if float(base['rho']) <= .9:
                for name in ('c32', 'c_mod', 'addition_P', 'explanation',
                             'VP_UN', 'L_data', 'L_indices', 'L_indptr'):
                    hash_array(field_hash, index, name, fields[name])
                labels_path = ASSETS / 'a/x1013_T5/clusters' / (key(index) + '.npz')
                with np.load(labels_path, allow_pickle=False) as archive:
                    labels_source = archive['labels'].copy()
                labels = labels_source.astype(np.int64)
                hash_array(cluster_hash, index, 'labels', labels_source)
                safe_mask, safe_summary = f1b_safe(fields, labels)
                row['S2_F1b_safe_large'] = score_mask(safe_mask, gt, valid)
                row['S2_safe_clusters'] = safe_summary
                a3_field = solve_a3(fields)
                a3_half = c.readout(a3_field)
                # Cache identity check uses T2's already scored binary A3.
                cached_a3 = packed(c.prediction_path('T2', binding), 'A3')
                if not np.array_equal(a3_half, cached_a3):
                    raise RuntimeError(f'Replayed A3 at .5 differs from cached T2 at {index}')
                a3_core = normalized_up(a3_field) > .65
                c1_path = c.prediction_path('E1', binding)
                with np.load(c1_path, allow_pickle=False) as archive:
                    c1_bits = archive['C1'].copy()
                hash_array(c1_hash, index, 'C1_packed', c1_bits)
                c1_mask = c.unpack(c1_bits)
                s3_mask = c1_mask | a3_core
                row['S3_C1_or_A3_core_large'] = score_mask(s3_mask, gt, valid)
            else:
                row['S2_F1b_safe_large'] = base['candidate']
                row['S2_safe_clusters'] = {'fallback': True}
                row['S3_C1_or_A3_core_large'] = base['candidate']
        else:
            row['S2_F1b_safe_large'] = base['candidate']
            row['S2_safe_clusters'] = {'used': False}
            row['S3_C1_or_A3_core_large'] = base['candidate']
        out.append(row)
        if number % 250 == 0 or number == len(selected):
            print(f'S half: {number}/{len(selected)}; large={large_n}', flush=True)

    arms = ('S1_noV_large', 'S2_F1b_safe_large', 'S3_C1_or_A3_core_large')
    summary = {
        'state': 'S_HALF_OFFLINE_COMPOSITIONS_COMPLETE',
        'split': 'S only; T untouched', 'n': len(out), 'large_gate_n': large_n,
        'large_gate_by_domain': large_by_domain,
        'new_dinov3_inference_calls': 0,
        'new_model_construction': False,
        'replayed_a3_sparse_solves': sum(r['rho'] <= .9 and r['candidate'][2] / PX > .25
                                         for r in out),
        'thresholds': {'predicted_area_fraction_gt': .25,
                       'f1b_safe_cluster_std_lt': .22,
                       'f1b_safe_cluster_area_fraction_lt': .35,
                       'a3_core_normalized_score_gt': .65},
        'bootstrap': {'unit': 'query photo cluster', 'replicates': BOOTSTRAPS, 'seed': SEED,
                      'metric': 'CLI-1024; (fold,class) pooled I/U then equal-slot mIoU'},
        'inputs': {},
        'results_vs_foris': domain_table(out, arms, 'foris'),
        'results_vs_candidate': domain_table(out, arms, 'candidate'),
    }
    for label, path in (('T0_iu', ASSETS / 'a/x1013_T0/iu.jsonl.gz'),
                        ('T2_iu', ASSETS / 'a/x1013_T2/iu.jsonl.gz'),
                        ('T5_iu', ASSETS / 'a/x1013_T5/iu.jsonl.gz')):
        summary['inputs'][label] = hashlib.sha256(path.read_bytes()).hexdigest()
    index_blob = '\n'.join(str(r['index']) for r in out).encode()
    summary['inputs']['S_index_order_sha256'] = hashlib.sha256(index_blob).hexdigest()
    summary['inputs']['large_gate_replayed_fields_sha256'] = field_hash.hexdigest()
    summary['inputs']['large_gate_cluster_labels_sha256'] = cluster_hash.hexdigest()
    summary['inputs']['large_gate_C1_packed_masks_sha256'] = c1_hash.hexdigest()
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['source_sha256'] = source_sha

    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / 'iu.jsonl.gz', EVIDENCE):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        with gzip.open(temp, 'wt') as stream:
            for record in out:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        temp.replace(path)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False,
                                                  indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
