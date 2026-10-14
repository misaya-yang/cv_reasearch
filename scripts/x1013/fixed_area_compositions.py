"""S-half evaluation of the user-proposed R1/R2/R3 scale-gated compositions.

Reads frozen T0/T2/T5 fields and masks plus the already produced E1 C1 masks.
Lambda-64 fields are replayed from cached T0 arrays. No encoder/model is called.
R3 uses cached A3 without CRF (one of the requested choices) to keep this pass
offline and CPU-bounded.

    aidemo/bin/python -B scripts/x1013/fixed_area_compositions.py
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
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT.parent / 'cv_data'
OUT = ASSETS / 'a/x1013_scale_compositions'
EVIDENCE = ROOT / 'evidence/research/x1013_scale_compositions_iu.jsonl.gz'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c
import w02_scale_eval as w02
import w09_integration_lambda as w09

DOMAINS = ('coco', 'lvis', 'pascal_part', 'paco_part', 'suim',
           'isaid', 'isic', 'lung', 'fundus', 'deepglobe_road')
PX = 1024 * 1024
BOOTSTRAPS = 500
SEED = 20261014
ARMS = ('R1_fix_l64', 'R2_fix_core_union', 'R3_fix_l64', 'R3_fix_core_union')


def read_records(path: Path) -> dict[str, dict]:
    with gzip.open(path, 'rt') as stream:
        return {str(r['index']): r for r in map(json.loads, stream)}


def key(index: str) -> str:
    return f'{int(index):06d}' if index.isdigit() else index


def mask_from(path: Path, arm: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        return c.unpack(z[arm])


def area_u(field: np.ndarray) -> np.ndarray:
    up = c.rc.lab.up(torch.as_tensor(field, dtype=torch.float32).reshape(64, 64))
    return c.rc.lab.minmax(up).cpu().numpy()


def solve_l64(fields: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lap = sparse.csr_matrix((fields['L_data'], fields['L_indices'], fields['L_indptr']),
                            shape=(4096, 4096))
    field = w09.solve(fields['primary'], fields['log_e'], lap, 64).reshape(64, 64)
    score = area_u(field)
    return score > .5, score


def f1b_score(fields: dict[str, np.ndarray], labels: np.ndarray) -> np.ndarray:
    value = fields['VP_UN'].astype(np.float32).reshape(64, 64)
    flat_labels = labels.astype(np.int64).ravel()
    flat_value = value.ravel()
    count = np.bincount(flat_labels)
    total = np.bincount(flat_labels, weights=flat_value)
    mean = total / np.maximum(count, 1)
    averaged = mean[flat_labels].reshape(64, 64).astype(np.float32)
    return area_u(.5 * value + .5 * averaged)


def score(mask: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> list[int]:
    pred = mask & valid
    y = gt & valid
    i = int(np.count_nonzero(pred & y))
    u = int(np.count_nonzero(pred | y))
    return [i, u, int(pred.sum())]


def miou(rows: list[dict], arm: str, weights: np.ndarray | None = None) -> float:
    slots: dict[tuple[object, object], list[int]] = {}
    for i, row in enumerate(rows):
        w = 1 if weights is None else int(weights[i])
        slot = (row['fold'], row['cls'])
        pair = slots.setdefault(slot, [0, 0])
        pair[0] += w * row[arm][0]
        pair[1] += w * row[arm][1]
    return float(np.mean([100 * i / max(u, 1) for i, u in slots.values() if u > 0]))


def domain_bootstrap(rows: list[dict], baseline: str, arms: tuple[str, ...]) -> dict:
    rng = np.random.default_rng(SEED)
    results = {}
    for domain in DOMAINS:
        selected = [r for r in rows if r['dataset'] == domain]
        photos = sorted({r['photo'] for r in selected})
        pindex = {photo: i for i, photo in enumerate(photos)}
        ix = np.asarray([pindex[r['photo']] for r in selected])
        weights = [np.bincount(rng.integers(0, len(photos), len(photos)),
                               minlength=len(photos))[ix] for _ in range(BOOTSTRAPS)]
        base = [miou(selected, baseline, w) for w in weights]
        domain_result = {'n': len(selected), 'arms': {}}
        for arm in arms:
            diffs = [miou(selected, arm, w) - b for w, b in zip(weights, base)]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            domain_result['arms'][arm] = {
                'delta_pp': miou(selected, arm) - miou(selected, baseline),
                'ci95_pp': [float(lo), float(hi)],
            }
        results[domain] = domain_result
    return results


def array_digest(hasher, index: str, arm: str, arr: np.ndarray) -> None:
    value = np.ascontiguousarray(arr)
    header = json.dumps([index, arm, list(value.shape), value.dtype.str],
                        separators=(',', ':')).encode()
    hasher.update(len(header).to_bytes(4, 'big'))
    hasher.update(header)
    hasher.update(value.tobytes())


def main() -> None:
    c.initialize()
    assert getattr(c.rc.pc, 'encoder', None) is None
    t0 = read_records(ASSETS / 'a/x1013_T0/iu.jsonl.gz')
    t2 = read_records(ASSETS / 'a/x1013_T2/iu.jsonl.gz')
    t5 = read_records(ASSETS / 'a/x1013_T5/iu.jsonl.gz')
    bindings = {str(b['index']): b for b in w02.rows_all(0)}
    selected = [t0[i] for i in sorted(t0, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
                if t0[i]['split'] == 'S']
    if len(selected) != 6752:
        raise RuntimeError(f'Expected 6,752 S examples, found {len(selected)}')

    rows = []
    split_counts: dict[str, list[int]] = {d: [0, 0, 0] for d in DOMAINS}
    field_digest, label_digest, c1_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for number, base in enumerate(selected, 1):
        index = str(base['index'])
        binding = bindings[index]
        t2row, t5row = t2[index], t5[index]
        if t2row['A4'] != base['candidate']:
            raise RuntimeError(f'T0/T2 candidate identity mismatch at {index}')
        area = base['candidate'][2] / PX
        area_bin = 0 if area < .03 else (1 if area <= .25 else 2)
        split_counts[base['dataset']][area_bin] += 1
        gt, valid = c.truth(binding)
        predpath = c.prediction_path('T0', binding)
        a4 = mask_from(predpath, 'candidate')
        c1path = c.prediction_path('E1', binding)
        with np.load(c1path, allow_pickle=False) as z:
            c1bits = z['C1'].copy()
        c1_digest.update(index.encode() + b'\0')
        c1_digest.update(c1bits.tobytes())
        c1 = c.unpack(c1bits)
        a3 = mask_from(c.prediction_path('T2', binding), 'A3')

        fields = c.load_fields(binding)
        ufield = fields['c32'].reshape(64, 64) if float(base['rho']) > .9 else fields['VP_UN'].reshape(64, 64)
        u = area_u(ufield)
        if not np.array_equal(u > .5, a4):
            raise RuntimeError(f'Candidate score/mask identity mismatch at {index}')

        fallback = float(base['rho']) > .9
        if area > .25 and not fallback:
            l64, _ = solve_l64(fields)
            labels_path = ASSETS / 'a/x1013_T5/clusters' / (key(index) + '.npz')
            with np.load(labels_path, allow_pickle=False) as archive:
                labels_source = archive['labels'].copy()
            label_digest.update(index.encode() + b'\0')
            label_digest.update(labels_source.tobytes())
            f1b = f1b_score(fields, labels_source)
            cached_f1b = mask_from(c.prediction_path('T5', binding), 'F1b')
            if not np.array_equal(f1b > .5, cached_f1b):
                raise RuntimeError(f'Reconstructed F1b score differs from T5 mask at {index}')
            core_union = (u > .5) | ((f1b > .5) & (u > .30))
            for name in ('VP_UN', 'primary', 'log_e', 'L_data', 'L_indices', 'L_indptr'):
                array_digest(field_digest, index, name, fields[name])
        else:
            l64 = a4
            core_union = a4

        # R1: C1 through 25%; raw lambda-64 readout above 25%; unsupported
        # high-rho fallback remains the frozen candidate.
        r1 = c1 if area <= .25 else l64
        # R2: C1 through 25%; conservative F1b support expansion above 25%.
        r2 = c1 if area <= .25 else core_union
        # R3: preserve tiny objects with C1; use cached A3 at medium scale;
        # choose the same two large-target alternatives for a paired check.
        r3_l64 = c1 if area < .03 else (a3 if area <= .25 else l64)
        r3_core = c1 if area < .03 else (a3 if area <= .25 else core_union)

        row = {'index': index, 'dataset': base['dataset'], 'split': 'S',
               'fold': base['fold'], 'cls': base['cls'], 'photo': base['photo'],
               'rho': float(base['rho']), 'pred_area': float(area), 'gt': int(gt.sum()),
               'foris': score(c.rc.lab.arm_cli(binding, 'B').cpu().numpy().astype(bool), gt, valid),
               'candidate': score(a4, gt, valid),
               'R1_fix_l64': score(r1, gt, valid),
               'R2_fix_core_union': score(r2, gt, valid),
               'R3_fix_l64': score(r3_l64, gt, valid),
               'R3_fix_core_union': score(r3_core, gt, valid)}
        rows.append(row)
        if number % 500 == 0 or number == len(selected):
            print(f'S-half compositions: {number}/{len(selected)}', flush=True)

    summary = {
        'state': 'S_HALF_SCALE_COMPOSITIONS_COMPLETE', 'n': len(rows),
        'new_dinov3_inference_calls': 0, 'new_model_construction': False,
        'T_scored': False, 'R1_large_branch': 'lambda=64 min-max upsampled score > .5; no CRF',
        'R2_large_branch': '(u>.50) OR ((F1b_score>.50) AND (u>.30)); F1b is continuous min-max upsampled blend',
        'R3_mid_branch': 'cached T2 A3 without CRF; user proposed CRF(A3) or A3',
        'area_bins': {'R1_R2': '<=.25 C1; >.25 large branch',
                      'R3': '<.03 C1; [.03,.25] A3; >.25 large branch'},
        'fallback': 'rho>.9 preserves candidate where graph/F1b branch is unavailable',
        'area_counts_by_domain_small_mid_large': split_counts,
        'inputs': {
            'T0_iu_sha256': hashlib.sha256((ASSETS / 'a/x1013_T0/iu.jsonl.gz').read_bytes()).hexdigest(),
            'T2_iu_sha256': hashlib.sha256((ASSETS / 'a/x1013_T2/iu.jsonl.gz').read_bytes()).hexdigest(),
            'T5_iu_sha256': hashlib.sha256((ASSETS / 'a/x1013_T5/iu.jsonl.gz').read_bytes()).hexdigest(),
            'S_index_order_sha256': hashlib.sha256('\n'.join(r['index'] for r in rows).encode()).hexdigest(),
            'large_gate_fields_sha256': field_digest.hexdigest(),
            'large_gate_cluster_labels_sha256': label_digest.hexdigest(),
            'large_gate_C1_packed_masks_sha256': c1_digest.hexdigest(),
        },
        'bootstrap': {'unit': 'query photo cluster', 'replicates': BOOTSTRAPS, 'seed': SEED,
                      'metric': 'CLI-1024 class mIoU; (fold,class) pooled I/U then equal-slot mean'},
        'results_vs_foris': domain_bootstrap(rows, 'foris', ARMS),
        'results_vs_candidate': domain_bootstrap(rows, 'candidate', ARMS),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for path in (OUT / 'iu.jsonl.gz', EVIDENCE):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        with gzip.open(temp, 'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        temp.replace(path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
