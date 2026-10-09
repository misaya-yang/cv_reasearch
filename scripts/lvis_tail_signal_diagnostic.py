#!/usr/bin/env python3
"""Diagnose existing1400 source scores inside already sealed final masks.

No encoding, candidate mask, coefficient selection, or new mIoU. GT weights
are used only here after the original1400 prediction seal, never in inference.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '2')
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from ics.official_data import array_hash

ROOT = REPO.parent / 'cv_data/a/lvis_atomic1400_20261009'
OUT = REPO.parent / 'cv_data/a/lvis_tail_signal1400_20261009'
FIELDS = ['sf', 'score', 'mean_guide', 'mean_unary']
ROLES = ['global', 'foris.crf', 'mean']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_auc(scores, foreground, background):
    """Exact weighted pair ordering with half credit for tied scores."""
    order = np.argsort(scores, kind='stable')
    values = scores[order]
    fg, bg = foreground[order], background[order]
    fm, bm = float(fg.sum()), float(bg.sum())
    if not fm or not bm:
        return None
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    wf, wb = np.add.reduceat(fg, starts), np.add.reduceat(bg, starts)
    return float(np.sum(wf * (np.cumsum(wb) - .5 * wb)) / (fm * bm))


def summary(rows):
    result = {'n': len(rows), 'regions': {}}
    for role in ROLES:
        rr = [r for r in rows if r['regions'][role]['foreground_mass'] > 0
              and r['regions'][role]['background_mass'] > 0]
        result['regions'][role] = {
            'n_auc_valid': len(rr),
            'n_no_foreground': sum(r['regions'][role]['foreground_mass'] == 0 for r in rows),
            'n_no_background': sum(r['regions'][role]['background_mass'] == 0 for r in rows),
            'macro_weighted_AUC': {field: None if not rr else
                float(np.mean([r['regions'][role]['auc'][field] for r in rr]))
                for field in FIELDS}}
    return result


def main():
    torch.set_num_threads(2)
    seal = json.loads((ROOT / 'sealed.json').read_text())
    assert seal['n'] == 1400
    for filename, key in [('config.json', 'config_sha256'), ('manifest.json', 'manifest_sha256'),
                          ('inference.jsonl', 'inference_index_sha256')]:
        assert sha(ROOT / filename) == seal[key]
    rows = json.loads((ROOT / 'manifest.json').read_text())
    index = {r['episode_id']: r for r in map(json.loads, (ROOT / 'inference.jsonl').read_text().splitlines())}
    assert len(rows) == len(index) == 1400
    result = []
    for row in rows:
        rec = index[row['episode_id']]
        field, pred = ROOT / 'fields' / rec['filename'], ROOT / 'predictions' / rec['filename']
        assert sha(field) == rec['field_sha256'] and sha(pred) == rec['prediction_sha256']
        with Image.open(row['query_mask_path']) as image:
            raw = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
        assert array_hash(raw) == row['query_mask_hash']
        truth = F.interpolate(torch.from_numpy(raw)[None, None].float(), (1024, 1024),
                              mode='nearest')[0, 0].numpy() > .5
        # Foreground/background are intersected with the *actual pixel mask*
        # before pooling. Product of separately pooled masks would be wrong.
        with np.load(pred) as masks:
            regions = {'global': np.ones((1024, 1024), dtype=bool)}
            regions.update({arm: np.unpackbits(masks['cli/' + arm], count=1024 * 1024)
                            .reshape(1024, 1024).astype(bool) for arm in ROLES[1:]})
        with np.load(field) as source:
            values = {key: source[key].ravel().astype(np.float64) for key in FIELDS}
        item = dict(episode_id=row['episode_id'], fold=row['fold'],
                    class_id=row['loader_class_id'], pilot=rec['pilot'],
                    query_fraction=float(truth.mean()), regions={})
        for role, region in regions.items():
            fg = (truth & region).reshape(64, 16, 64, 16).sum((1, 3), dtype=np.float64).ravel()
            bg = (~truth & region).reshape(64, 16, 64, 16).sum((1, 3), dtype=np.float64).ravel()
            item['regions'][role] = dict(foreground_mass=float(fg.sum()), background_mass=float(bg.sum()),
                auc={key: weighted_auc(value, fg, bg) for key, value in values.items()})
        result.append(item)
    report = dict(state='COMPLETE_LABEL_DIAGNOSTIC_ONLY', n=1400,
                  source_seal_sha256=sha(ROOT / 'sealed.json'), producer_sha256=sha(__file__),
                  definition='GT pixel mass per original16x16 token; weighted score ordering inside actual sealed masks; ties half credit; no selected cutoff',
                  new_encoder_calls=0, new_prediction_masks=0, new_miou=0,
                  groups={'all1400': summary(result), 'design1300': summary([r for r in result if not r['pilot']]),
                          'exposed_pilot100': summary([r for r in result if r['pilot']]),
                          'query_lt1pct': summary([r for r in result if r['query_fraction'] < .01])},
                  limitations='Token-constant ordering diagnostic, not pixel interpolation AUC, classification accuracy or final mIoU; mask-conditioned comparisons exclude zero-role cases and report their counts')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in result))
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
