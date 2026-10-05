#!/usr/bin/env python3
"""One fixed CPU read: can query contrast choose among five reference-named concepts?

Uses only an existing frozen dino.txt bank and FoRIS packets. No encoding, GPU,
query-label use in selection, or parameter search. Scores are before FoRIS CRF.
"""
import argparse
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--packets', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from lang_common import COCO, binarise, class_miou, fuse, iu, paired, photo_groups

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    rows = json.loads(a.manifest.read_text())['episodes']
    with np.load(a.bank / 'text.npz') as t:
        names = [str(x) for x in t['names']]
        e_patch = F.normalize(torch.from_numpy(t['e_patch'].astype(np.float32)), dim=-1)
        e_full = F.normalize(torch.from_numpy(t['e_full'].astype(np.float32)), dim=-1)
        scale = float(t['scale'])
    arms = ('host', 'source_top1', 'query_only', 'source_plus_query', 'top5_oracle')
    records = []
    started = time.monotonic()
    with torch.inference_mode():
        for row in rows:
            key = f"{row['fold']}_{row['e']}_{row['c']}"
            with np.load(a.bank / (key + '.npz')) as b, np.load(a.packets / (key + '.npz')) as z:
                c_cls = torch.from_numpy(b['c_cls'].astype(np.float32))
                c_tok = torch.from_numpy(b['c_tok'].astype(np.float32))
                c_cov = torch.from_numpy(b['c_cov'].astype(np.float32))
                pooled = (c_cov[..., None] * c_tok).sum((0, 1)) / c_cov.sum().clamp_min(1e-6)
                source = scale * (F.normalize(torch.cat((c_cls, pooled)), dim=0) @ e_full.T)
                top = source.topk(5).indices
                q = torch.from_numpy(b['q_b'].astype(np.float32))
                h, w = q.shape[:2]
                logits = scale * F.normalize(q.reshape(-1, q.shape[-1]), dim=-1) @ e_patch[top].T
                logits = logits.reshape(h, w, 5).permute(2, 0, 1)[None]
                maps = F.interpolate(logits, (64, 64), mode='bilinear', align_corners=False)[0]
                native_score = torch.from_numpy(z['score'].astype(np.float32)).reshape(64, 64)
                flat = native_score.flatten()
                core = flat.topk(41).indices
                outside = flat.topk(1024, largest=False).indices
                contrast = maps.flatten(1)[:, core].mean(1) - maps.flatten(1)[:, outside].mean(1)
                src5 = source[top]
                zscore = lambda v: (v - v.mean()) / v.std(unbiased=False).clamp_min(1e-6)
                source_choice = 0
                query_choice = int(contrast.argmax())
                joint_choice = int((zscore(src5) + zscore(contrast)).argmax())
                masks = [binarise(fuse(native_score, maps[i], 0.5), (1024, 1024)) for i in range(5)]
                native = binarise(native_score, (1024, 1024))
                cached_pre = torch.from_numpy(np.unpackbits(z['pre'])[:1024 * 1024].reshape(1024, 1024).astype(bool))
                if not torch.equal(native, cached_pre):
                    raise RuntimeError('FoRIS pre-mask reproduction failed: ' + key)
                truth = torch.from_numpy(np.unpackbits(z['truth'])[:1024 * 1024].reshape(1024, 1024).astype(bool))
                candidate_iu = [iu(m, truth) for m in masks]
                candidate_iou = [i / max(u, 1) for i, u in candidate_iu]
                best = int(np.argmax(candidate_iou))  # label-assisted diagnostic; never selects a method mask
                selections = (native, masks[source_choice], masks[query_choice], masks[joint_choice], masks[best])
                records.append(dict(key=key, fold=row['fold'], c=row['c'], support=row['support'], query=row['query'],
                                    selected=dict(source_top1=source_choice, query_only=query_choice,
                                                  source_plus_query=joint_choice, top5_oracle=best),
                                    top5=[names[i] for i in top.tolist()], source_logits=src5.tolist(),
                                    query_contrast=contrast.tolist(), iu={name: iu(mask, truth) for name, mask in zip(arms, selections)}))
            if len(records) % 40 == 0:
                print(json.dumps(dict(episodes=len(records), seconds=round(time.monotonic() - started, 1))), flush=True)

    cls = np.array([x['c'] for x in records])
    folds = np.array([x['fold'] for x in records])
    groups = photo_groups(rows)
    values = {name: np.array([x['iu'][name] for x in records], dtype=np.float64) for name in arms}
    compare = lambda x, y: dict(paired(values[x], values[y], cls, groups),
                                fold_gain=[class_miou(values[x][folds == f], cls[folds == f]) -
                                           class_miou(values[y][folds == f], cls[folds == f]) for f in range(4)])
    chosen = np.array([x['selected']['source_plus_query'] for x in records])
    best = np.array([x['selected']['top5_oracle'] for x in records])
    report = dict(state='COMPLETED', dataset='COCO-20i', split='exposed DEV241', episodes=len(rows), seed=0,
                  resolution='1024 model image, before CRF', host='cached public FoRIS pre-mask',
                  selector='source crop top5 + query top41-versus-bottom1024 text-logit contrast; within-top5 z-score sum; fixed lambda 0.5',
                  same_information_control='source crop top1 with identical candidate maps and lambda',
                  host_miou=class_miou(values['host'], cls),
                  source_top1_vs_host=compare('source_top1', 'host'),
                  query_only_vs_host=compare('query_only', 'host'),
                  source_plus_query_vs_host=compare('source_plus_query', 'host'),
                  source_plus_query_vs_source_top1=compare('source_plus_query', 'source_top1'),
                  top5_oracle_vs_host=compare('top5_oracle', 'host'),
                  joint_matches_top5_oracle=float((chosen == best).mean()),
                  source_matches_top5_oracle=float((best == 0).mean()),
                  query_matches_top5_oracle=float(np.mean([x['selected']['query_only'] == x['selected']['top5_oracle'] for x in records])),
                  true_name_in_top5_exact=float(np.mean([COCO[x['c']] in x['top5'] for x in records])),
                  photo_groups=len(groups), bootstrap_draws=2000, seconds=time.monotonic() - started,
                  unverified='Not the complete CRF pipeline; no CONFIRM600; top5 oracle uses query labels and is not a method.')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    a.out.with_suffix('.episodes.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in records))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
