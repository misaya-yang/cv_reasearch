"""Post-seal GT diagnostic: maximum ranking obtainable with fixed token cells.

The ratio TP/(TP+FP) is an oracle score, never a method input. Compare cell
resolutions on the same saved FoRIS pixels without images, raw reads or DINO.
"""
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'evidence/local/false_alarm_floor_20261010'))
import inmask_evidence as base


def main():
    parser = base.arguments(['analyze'])
    parser.add_argument('--joint-source', required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    source, _ = base.opened(args)
    prior = Path(args.joint_source)
    if not prior.is_absolute():
        prior = base.ROOT / 'inmask' / prior
    config, seal = base.read(prior / 'config.json'), base.read(prior / 'sealed.json')
    report = base.read(prior / 'report.json')
    assert base.sha(prior / 'sealed.json') == report['seal_sha256']
    assert base.sha(prior / 'config.json') == seal['config_sha256']
    assert base.sha(source.folder / 'manifest.json') == config['manifest_sha256']
    rows = source.rows[:len(config['ids'])]
    assert [source.describe(r)['id'] for r in rows] == config['ids']
    for i, row in enumerate(rows):
        assert base.sha(prior / 'fields' / f'{i:06d}.npz') == seal['fields'][row['episode_id']]
    items = []
    grids = [config['grid'], 64, 128]
    assert len(set(grids)) == len(grids)
    for row in rows:
        mask = source.prediction(row)
        truth, ignore = source.truth(row, mask.shape)
        valid = np.ones(mask.shape, bool) if ignore is None else ~ignore
        mask, truth = mask & valid, truth & valid
        item = dict(id=row['episode_id'], dataset=row['dataset'], cls=source.describe(row)['cls'],
                    hit=int((mask & truth).sum()), false_alarm=int((mask & ~truth).sum()), grids={})
        for grid in grids:
            hit = base.token_mass(mask & truth, grid)
            fa = base.token_mass(mask & ~truth, grid)
            oracle = hit / np.maximum(hit + fa, 1.)
            values = base.separation(oracle, hit, fa)
            mixed = (hit > 0) & (fa > 0)
            values.update(mixed_tokens=int(mixed.sum()),
                          false_alarm_in_mixed=float(fa[mixed].sum()),
                          hit_in_mixed=float(hit[mixed].sum()))
            item['grids'][str(grid)] = values
        items.append(item)
    summary = {}
    for dataset in sorted({r['dataset'] for r in items}):
        part = [r for r in items if r['dataset'] == dataset]
        result = {}
        for grid in grids:
            values = [r['grids'][str(grid)] for r in part if r['grids'][str(grid)]['auc'] is not None]
            total_fa = sum(r['false_alarm'] for r in part)
            total_hit = sum(r['hit'] for r in part)
            result[str(grid)] = dict(n_separable=len(values),
                oracle_auc=float(np.mean([r['auc'] for r in values])) if values else None,
                oracle_k95=sum(r['kept95_mass'] for r in values) / max(sum(r['false_alarm'] for r in values), 1.),
                mixed_false_alarm_share=sum(r['grids'][str(grid)]['false_alarm_in_mixed'] for r in part) / max(total_fa, 1.),
                mixed_hit_share=sum(r['grids'][str(grid)]['hit_in_mixed'] for r in part) / max(total_hit, 1.))
        summary[dataset] = result
    output = base.ROOT / 'execution' / f'token_ceiling_{prior.name}.json'
    output.parent.mkdir(exist_ok=True)
    base.write(output, dict(state='POST_SEAL_GT_ORACLE_DIAGNOSTIC',
        recorded_utc=datetime.now(timezone.utc).isoformat(), source=str(prior), n=len(items),
        seal_sha256=base.sha(prior / 'sealed.json'), frame=report['frame'],
        diagnostic='Same saved FoRIS pixels; GT-derived within-cell TP/(TP+FP), optimal weighted ROC ordering',
        limits='Oracle labels are unavailable to inference. Finer grid is a measurement control, not evidence of finer observed features or deployable gain.',
        encoder_calls=0, raw_cache_reads=0, raw_cache_writes=0, datasets=summary, episodes=items))
    print(dict(path=str(output), datasets=summary), flush=True)


if __name__ == '__main__':
    main()
