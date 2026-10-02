#!/usr/bin/env python3
"""Same-pool ICS centroid-bank control. Not original TPA: no text, calibrated host posterior or late fusion.

Reads the supplied paired cache without modifying it. All bank rows reuse its debiased DINO coordinates,
not TPA's external DINOv2 encoder. Pseudo masks come from the same one-shot INSID3. Query is excluded from
its bank. No cross-image verification, iteration, tuning, or model/data download. Small pilot only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def bank_mask(features, masks, target, fallback, labels=None, min_anchors=5):
    """Per-class per-image anchor floor; sum unit patch vectors, then normalize one FG/BG centroid."""
    import torch
    import torch.nn.functional as F
    sums = torch.zeros(2, target.shape[-1], device=target.device, dtype=target.dtype)
    counts = [0, 0]
    for f, m in zip(features, masks):
        for c, selected in enumerate((~m, m)):
            n = int(selected.sum())
            if n >= min_anchors:
                sums[c] += f[selected].sum(0)
                counts[c] += n
    if min(counts) == 0 or bool((sums.norm(dim=1) == 0).any()):
        return fallback.clone(), counts
    prototypes = F.normalize(sums, dim=1)
    margin = target @ (prototypes[1] - prototypes[0])
    if labels is not None:
        k = int(labels.max()) + 1
        margin = (torch.bincount(labels, weights=margin, minlength=k)
                  / torch.bincount(labels, minlength=k).clamp_min(1))[labels]
    return margin > 0, counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared-root', default='/root/autodl-tmp/demo9')
    ap.add_argument('--file', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=30)
    ap.add_argument('--M', type=int, default=15)
    ap.add_argument('--pool-seed', type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.prepared_root) / 'scripts'))
    import _paths
    import numpy as np
    import torch
    from tics import ImageSet, one_shot
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC', '.3')))
    d = torch.load(a.file, weights_only=False)
    cls = np.array(d['cls']); names = d['names']; n = len(names) // 2
    rng = np.random.default_rng(a.pool_seed)
    records = []; t0 = time.time(); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), Path(a.prepared_root)/'tics/imageset.py']
    report = dict(state='RUNNING', file=a.file, fold=d['fold'], args=vars(a),
                  feature_space='FP16 cached orthogonal-complement coordinates, renormalized FP32; same as supplied ICS reader',
                  adaptation='Confidence-free FG/BG centroid readout; no dense host posterior, no fusion. Not a TPA reproduction.',
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}, records=records)
    def write():
        acc = {}
        for r in records:
            for key, (i, u) in r['iu'].items():
                row = acc.setdefault(key, {}).setdefault(r['c'], [0., 0.]); row[0] += i; row[1] += u
        report['miou'] = {k:100*float(np.mean([i/max(u, 1) for i, u in rows.values()])) for k, rows in acc.items()}
        report['episodes'] = len(records); report['elapsed_s'] = time.time()-t0
        tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(report)); tmp.replace(out)
    write()
    try:
        for e in range(min(a.limit, n)):
            c = int(cls[2*e]); ref, q = 2*e, 2*e+1
            candidates = [2*o+1 for o in range(n) if o != e and cls[2*o] == c and names[2*o+1] not in (names[ref], names[q])]
            seen = set(); candidates = [x for x in candidates if not (names[x] in seen or seen.add(names[x]))]
            pool = [int(x) for x in rng.permutation(candidates)[:a.M]]
            idx = [ref, q]+pool
            # Inference object gets only the legally annotated support. Hold truth outside that object.
            gt64 = torch.zeros_like(d['gt64'][idx]); gt64[0] = d['gt64'][ref]
            gb = np.zeros_like(d['gt_bits'][idx]); gb[0] = d['gt_bits'][ref]
            s = ImageSet(dict(c=c, names=[names[x] for x in idx], fq=d['fq'][idx], lab=d['lab'][idx],
                              Po=[d['Po'][x] for x in idx], gt64=gt64, gt_bits=gb, S=d['S']))
            j = list(range(1, s.n)); p = one_shot(s, j); donors = list(range(2, s.n)); gold = s.gt64[0]
            predictions = {'1shot':p[1], 'naive':s.predict(1, [0]+donors, [gold]+[p[x] for x in donors], backward='majority')}
            counts = {}; times = {}
            for tag, sources in [('bank_support', [0]), ('bank_pool', donors), ('bank_support_pool', [0]+donors)]:
                for clustered in (False, True):
                    name = tag+('_cluster' if clustered else '_patch')
                    torch.cuda.synchronize(); start = time.time()
                    predictions[name], counts[name] = bank_mask([s.fd[x] for x in sources], [p[x] for x in sources],
                                                              s.fd[1], p[1], s.lab[1] if clustered else None)
                    torch.cuda.synchronize(); times[name] = time.time()-start
            # All deployable predictions are frozen before loading donor/query labels for diagnostics and scoring.
            real = d['gt64'][idx].to(s.fd.device).reshape(s.n, -1)
            predictions['true'] = s.predict(1, [0]+donors, [gold]+[real[x] for x in donors], backward='pooled', k=5)
            truth = torch.from_numpy(np.unpackbits(d['gt_bits'][q])).to(s.fd.device).bool().reshape(s.S, s.S)
            iu = {}
            for key, m in predictions.items():
                full = s.up(m); iu[key] = [float((full & truth).sum()), float((full | truth).sum())]
            records.append(dict(e=e, c=c, query=names[q], support=names[ref], pool=len(pool),
                                donor_ids=[names[x] for x in pool], iu=iu, anchors=counts, bank_readout_s=times))
            del s
            write()
            if (e+1)%5 == 0: print(e+1, round(time.time()-t0, 1), report['miou'], flush=True)
        report['state'] = 'COMPLETED'; write()
    except BaseException as ex:
        report.update(state='ERROR', error=repr(ex)); write(); raise

if __name__ == '__main__':
    main()
