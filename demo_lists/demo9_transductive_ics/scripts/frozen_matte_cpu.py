#!/usr/bin/env python3
"""Original-resolution CPU replay of one frozen matte refiner and its controls.

The encoder/public FoRIS forward is reused from its existing cache. No CUDA,
fresh encoding, extra CRF, parameter selection, or CONFIRM600 read occurs here.
"""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--packets', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from scipy import ndimage
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest['episodes']
    if args.smoke:
        rows = [next(r for r in rows if r['fold'] == f) for f in range(4)]
    yy, xx = np.mgrid[0:64, 0:64]
    positions = np.stack([yy.ravel(), xx.ravel()], 1).astype(np.float32)
    records = []
    started = time.monotonic()

    def i_u(mask, truth):
        return [int((mask & truth).sum()), int((mask | truth).sum())]

    for row in rows:
        key = f"{row['fold']}_{row['e']}_{row['c']}"
        q = torch.load(Path(args.features) / (key + '.pt'), map_location='cpu', weights_only=True)['q'].float()
        with np.load(Path(args.packets) / (key + '.npz')) as packet:
            score = torch.from_numpy(packet['score'].astype(np.float32)).flatten()
            score -= score.min()
            score /= score.max().clamp_min(1e-6)
            mask = (score > .5).view(64, 64).numpy()
            st = np.ones((3, 3), bool)
            fg = ndimage.binary_erosion(mask, st, iterations=2)
            bg = ~ndimage.binary_dilation(mask, st, iterations=2)
            if not fg.any():
                fg = np.zeros(4096, bool)
                fg[(score * torch.from_numpy(mask.ravel()).float()).topk(max(1, int(mask.sum()) // 4)).indices.numpy()] = True
                fg = fg.reshape(64, 64)
            ui = np.flatnonzero((~fg & ~bg).ravel())
            endpoints = []
            for zone in (fg, bg):
                indices = np.flatnonzero(zone.ravel())
                if not len(indices):
                    break
                distances = ((positions[ui, None] - positions[indices][None]) ** 2).sum(-1)
                count = min(6, len(indices))
                neighbors = (np.argpartition(distances, count - 1, axis=1)[:, :count]
                             if count < len(indices) else np.tile(np.arange(len(indices)), (len(ui), 1)))
                endpoints.append(q[torch.from_numpy(indices[neighbors])].mean(1))
            field = score.clone()
            if len(endpoints) == 2:
                g, b = endpoints
                f = q[torch.from_numpy(ui)]
                direction = g - b
                alpha = (((f - b) * direction).sum(1) / (direction ** 2).sum(1).clamp_min(1e-6)).clamp(0, 1)
                field[torch.from_numpy(ui)] = (score[torch.from_numpy(ui)] + alpha) / 2
            matte = F.interpolate(field.view(1, 1, 64, 64), size=(1024, 1024), mode='bilinear', align_corners=False)[0, 0] > .5
            bit = lambda name: torch.from_numpy(np.unpackbits(packet[name])[:1024 ** 2].reshape(1024, 1024).astype(bool))
            native, pre = bit('native'), bit('pre')
            outputs = dict(native=native, pre=pre, matte=matte, delete_only=matte & native)
            # Both outputs and all test-time endpoints are frozen before labels enter.
            annotation = Path(manifest['annotation_root']) / Path(row['query']).with_suffix('.png')
            truth = torch.from_numpy((np.asarray(Image.open(annotation)) == row['c'] + 1).copy())
            truth_model = F.interpolate(truth[None, None].float(), size=(1024, 1024), mode='nearest')[0, 0].bool()
            if not torch.equal(truth_model, bit('truth')):
                raise RuntimeError('Original annotation differs from the frozen episode: ' + key)
            original = {name: F.interpolate(value[None, None].float(), size=truth.shape,
                        mode='bilinear', align_corners=False)[0, 0] > .5 for name, value in outputs.items()}
            records.append(dict(key=key, fold=row['fold'], c=row['c'], support=row['support'], query=row['query'],
                                original_hw=list(truth.shape), original={name: i_u(value, truth) for name, value in original.items()},
                                model={name: i_u(value, truth_model) for name, value in outputs.items()}))
        if len(records) % 40 == 0 or args.smoke:
            print(json.dumps(dict(event='CPU_REPLAY_PROGRESS', episodes=len(records), seconds=round(time.monotonic() - started, 2))), flush=True)

    # Match the existing class-pooled, connected-photo-group bootstrap convention.
    classes = np.array([r['c'] for r in records])
    folds = np.array([r['fold'] for r in records])
    parent, owner = list(range(len(records))), {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, row in enumerate(records):
        for role in ('support', 'query'):
            value = row[role]
            if value in owner:
                parent[find(i)] = find(owner[value])
            else:
                owner[value] = i
    group_map = {}
    for i in range(len(records)):
        group_map.setdefault(find(i), []).append(i)
    groups = [np.array(indices, int) for indices in group_map.values()]

    def miou(values, pick=None):
        if pick is None:
            pick = np.arange(len(values))
        values, cls = values[pick], classes[pick]
        return 100 * float(np.mean([values[cls == c, 0].sum() / max(values[cls == c, 1].sum(), 1) for c in np.unique(cls)]))

    def paired(a, b):
        rng = np.random.default_rng(0)
        differences = []
        for _ in range(2000):
            pick = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            differences.append(miou(a, pick) - miou(b, pick))
        return dict(miou=miou(a), gain=miou(a) - miou(b), ci95=np.quantile(differences, [.025, .975]).tolist(),
                    fold_gain=[miou(a, np.flatnonzero(folds == f)) - miou(b, np.flatnonzero(folds == f)) for f in range(4)],
                    up=int((a[:, 0] / np.maximum(a[:, 1], 1) > b[:, 0] / np.maximum(b[:, 1], 1) + 1e-9).sum()),
                    down=int((a[:, 0] / np.maximum(a[:, 1], 1) < b[:, 0] / np.maximum(b[:, 1], 1) - 1e-9).sum()))

    values = lambda level, name: np.array([r[level][name] for r in records], float)
    report = dict(state='SMOKE_PASSED' if args.smoke else 'COMPLETED', dataset='COCO-20i', split='exposed DEV241',
                  episodes=len(records), seed=0, resolution='original annotation resolution',
                  pipeline='Reuse unchanged cached complete public FoRIS forward/CRF; replace final refinement with one frozen matte arm.',
                  source_precision='Cached q tokens are float16, evaluated in float32; public FoRIS score/masks are unchanged.',
                  original_native_miou=miou(values('original', 'native')),
                  matte_vs_native=paired(values('original', 'matte'), values('original', 'native')),
                  delete_only_vs_native=paired(values('original', 'delete_only'), values('original', 'native')),
                  matte_vs_delete_only=paired(values('original', 'matte'), values('original', 'delete_only')),
                  model_native_miou=miou(values('model', 'native')), model_matte_miou=miou(values('model', 'matte')),
                  photo_groups=len(groups), bootstrap_draws=2000, threads=args.threads,
                  seconds=time.monotonic() - started,
                  unverified='Fresh forward/cache equality of the proposed refiner, matte followed by identical CRF, CONFIRM600 and finer features.')
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    output.with_suffix('.episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
