"""Offline structural and reference-pixel diagnostics from sealed predictions.

No model is constructed and no DINOv3 inference is run. Reads the already sealed
R2 Fundus/Road outputs and the x1013 S-half A4/C1 outputs; never scores x1013 T.

    aidemo/bin/python -B scripts/x1013/structure_pixel_diagnostic.py
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
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT.parent / 'cv_data'
OUT = ASSETS / 'a/x1013_mechanism_diagnostics'
EVIDENCE = ROOT / 'evidence/research/x1013_mechanism_diagnostics_iu.jsonl.gz'
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import lab
import w02_scale_eval as w02

PX = 1024 * 1024
BOOTSTRAPS = 1000
SEED = 20261014
CONNECTIVITY8 = np.ones((3, 3), dtype=bool)


def unpack(path: Path, arm: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return np.unpackbits(archive[arm])[:PX].reshape(1024, 1024).astype(bool)


def zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Deterministic 8-neighbour Zhang-Suen thinning for binary 2D masks."""
    image = np.asarray(mask, dtype=bool).copy()
    while True:
        changed = False
        for phase in (0, 1):
            padded = np.pad(image, 1)
            center = padded[1:-1, 1:-1]
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            ring = (p2, p3, p4, p5, p6, p7, p8, p9)
            neighbors = sum(x.astype(np.uint8) for x in ring)
            transitions = sum((~ring[i] & ring[(i + 1) % 8]).astype(np.uint8)
                              for i in range(8))
            eligible = center & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1)
            if phase == 0:
                eligible &= ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                eligible &= ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            if eligible.any():
                image[eligible] = False
                changed = True
        if not changed:
            return image


def component_counts(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred_labels, n_pred = ndimage.label(pred, structure=CONNECTIVITY8)
    gt_labels, n_gt = ndimage.label(gt, structure=CONNECTIVITY8)
    overlap = pred & gt
    pairs = np.unique(np.column_stack((gt_labels[overlap], pred_labels[overlap])), axis=0)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)] if pairs.size else pairs.reshape(0, 2)
    gt_to_pred: dict[int, set[int]] = {}
    pred_to_gt: dict[int, set[int]] = {}
    for g, p in pairs:
        gt_to_pred.setdefault(int(g), set()).add(int(p))
        pred_to_gt.setdefault(int(p), set()).add(int(g))
    return {
        'gt_components': int(n_gt), 'pred_components': int(n_pred),
        'missed_gt_components': int(sum(g not in gt_to_pred for g in range(1, n_gt + 1))),
        'spurious_pred_components': int(sum(p not in pred_to_gt for p in range(1, n_pred + 1))),
        'split_gt_components': int(sum(len(ps) > 1 for ps in gt_to_pred.values())),
        'merged_pred_components': int(sum(len(gs) > 1 for gs in pred_to_gt.values())),
    }


def mask_metrics(pred: np.ndarray, gt: np.ndarray, gt_skeleton: np.ndarray | None = None) -> dict:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if gt_skeleton is None:
        gt_skeleton = zhang_suen(gt)
    pred_skeleton = zhang_suen(pred)
    intersection = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    skel_precision = float(np.count_nonzero(pred_skeleton & gt) / max(int(pred_skeleton.sum()), 1))
    skel_recall = float(np.count_nonzero(gt_skeleton & pred) / max(int(gt_skeleton.sum()), 1))
    cldice = 0.0 if skel_precision + skel_recall == 0 else (
        2 * skel_precision * skel_recall / (skel_precision + skel_recall))
    result = {'I': intersection, 'U': union,
              'iou': float(intersection / max(union, 1)),
              'skeleton_precision': skel_precision, 'skeleton_recall': skel_recall,
              'cldice_style': float(cldice)}
    result.update(component_counts(pred, gt))
    return result


def gt_for(binding: dict) -> np.ndarray:
    return lab.mask1024(binding['row']['query_mask_path']).cpu().numpy().astype(bool)


def row_base(binding: dict, cohort: str, gt: np.ndarray) -> dict:
    row = binding['row']
    return {'index': str(binding['index']), 'dataset': binding['dataset'],
            'cohort': cohort, 'fold': row['fold'], 'cls': row['global_class_id'],
            'photo': str(row.get('query_photo_id') or row['query_path']),
            'gt_area': int(gt.sum())}


def run_r2(bindings: dict[str, dict]) -> list[dict]:
    sealed = json.loads((ASSETS / 'a/x1012_R2/sealed.json').read_text())
    if sealed['state'] != 'ALL_NATURAL_OUTPUTS_SEALED' or sealed['n'] != 1360:
        raise RuntimeError('R2 outputs are not fully sealed at the expected 1,360 examples')
    by_episode = {b['row']['episode_id']: b for b in bindings.values()}
    out = []
    for record in sealed['records']:
        if record['dataset'] not in ('fundus', 'deepglobe_road'):
            continue
        index = str(record['index'])
        # R2 prediction files use their own string index namespace (for
        # example ``000000``), while the shared evaluation bindings use the
        # original episode index (for example ``0``). Bind by frozen episode
        # identity, then retain the R2 index only for its prediction archive.
        binding = by_episode.get(record['episode_id'])
        if binding is None or binding['dataset'] != record['dataset']:
            raise RuntimeError(f'R2/P0 binding identity mismatch at {index}')
        gt = gt_for(binding)
        gt_skeleton = zhang_suen(gt)
        pred_path = Path(record['prediction_path'])
        if hashlib.sha256(pred_path.read_bytes()).hexdigest() != record['prediction_sha256']:
            raise RuntimeError(f'R2 prediction hash mismatch at {index}')
        arms = {name: unpack(pred_path, name) for name in
                ('VP_identity', 'pixel', 'pixel_permuted', 'patch_constant')}
        arms['foris'] = lab.arm_cli(binding, 'B').cpu().numpy().astype(bool)
        base = row_base(binding, 'R2_1360_exposed', gt)
        base['arms'] = {name: mask_metrics(mask, gt, gt_skeleton) for name, mask in arms.items()}
        out.append(base)
        if len(out) % 25 == 0:
            print(f'R2 structural audit: {len(out)}', flush=True)
    return out


def run_x1013_s(bindings: dict[str, dict]) -> list[dict]:
    t0_path = ASSETS / 'a/x1013_T0/iu.jsonl.gz'
    records = [json.loads(line) for line in gzip.open(t0_path, 'rt')]
    out = []
    for base in records:
        if base['split'] != 'S' or base['dataset'] not in ('fundus', 'deepglobe_road'):
            continue
        index = str(base['index'])
        binding = bindings.get(index)
        if binding is None:
            raise RuntimeError(f'Missing x1013 binding at {index}')
        gt = gt_for(binding)
        gt_skeleton = zhang_suen(gt)
        pred_path = ASSETS / 'a/x1013_T0/predictions' / (f'{int(index):06d}.npz' if index.isdigit() else index + '.npz')
        e1_path = ASSETS / 'a/x1013_E1/predictions' / (f'{int(index):06d}.npz' if index.isdigit() else index + '.npz')
        a4 = unpack(pred_path, 'candidate')
        c1 = unpack(e1_path, 'C1')
        foris = lab.arm_cli(binding, 'B').cpu().numpy().astype(bool)
        with np.load(ASSETS / 'a/x1013_T0/fields' / (f'{int(index):06d}.npz' if index.isdigit() else index + '.npz'), allow_pickle=False) as z:
            field = z['c32'].reshape(64, 64) if float(z['reference_rho'][0]) > .9 else z['VP_UN'].reshape(64, 64)
            score = lab.minmax(lab.up(torch.as_tensor(field, dtype=torch.float32))).cpu().numpy()
        row = row_base(binding, 'x1013_S', gt)
        row['gt_skeleton_score'] = {
            'mean': float(score[gt_skeleton].mean()) if gt_skeleton.any() else 0.0,
            'p10': float(np.percentile(score[gt_skeleton], 10)) if gt_skeleton.any() else 0.0,
            'fraction_below_candidate_cut': float(np.mean(score[gt_skeleton] <= .5)) if gt_skeleton.any() else 0.0,
        }
        row['arms'] = {name: mask_metrics(mask, gt, gt_skeleton) for name, mask in
                       (('A4', a4), ('C1', c1), ('FoRIS', foris))}
        out.append(row)
        if len(out) % 25 == 0:
            print(f'x1013 S structural audit: {len(out)}', flush=True)
    return out


def mean_metric(rows: list[dict], arm: str, metric: str) -> float:
    values = [r['arms'][arm][metric] for r in rows]
    return float(np.mean(values)) if values else 0.0


def photo_bootstrap(rows: list[dict], domain: str, a: str, b: str, metric: str) -> list[float]:
    selected = [r for r in rows if r['dataset'] == domain]
    photos = sorted({r['photo'] for r in selected})
    by_photo: dict[str, list[dict]] = {p: [] for p in photos}
    for row in selected:
        by_photo[row['photo']].append(row)
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(photos, size=len(photos), replace=True)
        sample = [r for p in sampled for r in by_photo[str(p)]]
        deltas.append(float(np.mean([r['arms'][a][metric] - r['arms'][b][metric] for r in sample])))
    return [float(x) for x in np.percentile(deltas, [2.5, 97.5])]


def summarize(rows: list[dict], cohort: str, arms_by_domain: dict[str, tuple[str, ...]]) -> dict:
    summary = {'cohort': cohort, 'domains': {}}
    for domain, arms in arms_by_domain.items():
        selected = [r for r in rows if r['dataset'] == domain]
        fields = ('iou', 'cldice_style', 'skeleton_precision', 'skeleton_recall',
                  'gt_components', 'pred_components', 'missed_gt_components',
                  'spurious_pred_components', 'split_gt_components', 'merged_pred_components')
        domain_data = {'n': len(selected), 'arms': {}}
        for arm in arms:
            domain_data['arms'][arm] = {field: mean_metric(selected, arm, field) for field in fields}
        compare = []
        if cohort == 'R2_1360_exposed':
            compare = [('pixel', 'VP_identity'), ('pixel', 'pixel_permuted')]
        else:
            compare = [('C1', 'A4')]
        domain_data['paired_deltas'] = {}
        for a, b in compare:
            metrics = ('iou', 'cldice_style', 'skeleton_precision', 'skeleton_recall',
                       'missed_gt_components', 'spurious_pred_components',
                       'split_gt_components', 'merged_pred_components')
            domain_data['paired_deltas'][f'{a}-{b}'] = {
                m: {'mean': float(np.mean([r['arms'][a][m] - r['arms'][b][m] for r in selected])),
                    'ci95_photo_bootstrap': photo_bootstrap(rows, domain, a, b, m)}
                for m in metrics
            }
        summary['domains'][domain] = domain_data
    return summary


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    lab.DEV = torch.device('cpu')
    torch.set_num_threads(1)
    bindings = {str(b['index']): b for b in w02.rows_all(0)}
    r2 = run_r2(bindings)
    current = run_x1013_s(bindings)
    summary = {
        'state': 'OFFLINE_STRUCTURE_PIXEL_DIAGNOSTIC_COMPLETE',
        'new_dinov3_inference_calls': 0,
        'new_model_construction': False,
        'x1013_T_scored': False,
        'skeleton': 'Zhang-Suen 8-neighbour thinning; foreground components use 8-connectivity',
        'cldice_style': 'harmonic mean of skeleton precision and skeleton recall',
        'bootstrap': {'unit': 'query photo cluster', 'replicates': BOOTSTRAPS, 'seed': SEED},
        'inputs': {
            'R2_sealed': sha256(ASSETS / 'a/x1012_R2/sealed.json'),
            'x1013_T0_iu': sha256(ASSETS / 'a/x1013_T0/iu.jsonl.gz'),
        },
        'R2_1360_exposed': summarize(r2, 'R2_1360_exposed', {
            'fundus': ('foris', 'VP_identity', 'pixel', 'pixel_permuted', 'patch_constant'),
            'deepglobe_road': ('foris', 'VP_identity', 'pixel', 'pixel_permuted', 'patch_constant')}),
        'x1013_S': summarize(current, 'x1013_S', {
            'fundus': ('FoRIS', 'A4', 'C1'), 'deepglobe_road': ('FoRIS', 'A4', 'C1')}),
        'x1013_skeleton_score': {
            d: {'n': sum(r['dataset'] == d for r in current),
                'mean_score': float(np.mean([r['gt_skeleton_score']['mean'] for r in current if r['dataset'] == d])),
                'p10_score': float(np.mean([r['gt_skeleton_score']['p10'] for r in current if r['dataset'] == d])),
                'fraction_below_0_5': float(np.mean([r['gt_skeleton_score']['fraction_below_candidate_cut'] for r in current if r['dataset'] == d]))}
            for d in ('fundus', 'deepglobe_road')
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    evidence_rows = [{'cohort': 'R2_1360_exposed', **r} for r in r2] + [{'cohort': 'x1013_S', **r} for r in current]
    for path in (OUT / 'episodes.jsonl.gz', EVIDENCE):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + '.tmp')
        with gzip.open(temp, 'wt') as stream:
            for row in evidence_rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        temp.replace(path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
