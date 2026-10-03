#!/usr/bin/env python3
"""Is the extent learnable without knowing the class? A small convolutional read-out fitted on three folds and
scored on the fourth, whose classes it never saw. Inputs are relations only (the FoRIS score and its stages, best
foreground and background similarity, similarity to the query's own confident core and sure background, and how
different adjacent patches are), so nothing in it can name a class.

  python scripts/extent_decoder_probe.py --run results/extent_v1/run --maps results/evidence_v1/maps \
      --out results/self_support_v0/decoder_probe.json

A premise test on about 180 training episodes per fold: patch level, no refinement, not a method score.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_extent import load, miou  # noqa: E402

SETS = dict(score_only=("sn",), score_stages=("sn", "s2", "s3"),
            relations=("sn", "s2", "s3", "fg", "bg", "grp_core", "grp_out", "edge_r", "edge_d"))


def norm(x):
    x = x - x.min()
    return x / max(float(x.max()), 1e-6)


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--maps", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs = load(a.run)
    n = len(recs)
    cls, folds = np.array([r["c"] for r in recs]), np.array([r["fold"] for r in recs])
    chan, tf = {k: [] for k in SETS["relations"]}, []
    for r in recs:
        name = "%d_%d_%d" % (r["fold"], r["e"], r["c"])
        z, m = np.load(a.run / "packets" / (name + ".npz")), np.load(a.maps / (name + ".npz"))
        g = lambda v: np.asarray(v, np.float32).reshape(64, 64)
        chan["sn"].append(norm(g(z["score"])))
        chan["s2"].append(norm(g(z["s2"])))
        chan["s3"].append(norm(g(z["s3"])))
        chan["fg"].append(g(z["fg_max"]))
        chan["bg"].append(g(z["bg_max"]))
        chan["grp_core"].append(g(m["grp_core"]))
        chan["grp_out"].append(g(m["grp_out"]))
        chan["edge_r"].append(np.pad(1 - z["aff_r"].astype(np.float32), ((0, 0), (0, 1))))
        chan["edge_d"].append(np.pad(1 - z["aff_d"].astype(np.float32), ((0, 1), (0, 0))))
        tf.append(np.unpackbits(z["truth"])[:1 << 20].reshape(64, 16, 64, 16).mean((1, 3)).astype(np.float32))
    tf = np.stack(tf)
    Y = torch.from_numpy(tf).to(dev)
    sn = np.stack(chan["sn"])

    def iu(pred):  # [n, 64, 64] bool
        i = (tf * pred).sum((1, 2))
        return np.stack([i, tf.sum((1, 2)) + pred.sum((1, 2)) - i], 1)
    base = iu(sn > 0.5)
    w = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=2000).astype(float)
    score = lambda t, m=slice(None): float(miou(t[m, 0], t[m, 1], cls[m])[0])
    out = dict(state="PROBED", episodes=n, native_patch_miou=score(base), arms={},
               scope="patch level, no refinement; classes of the scored fold unseen; about 180 training episodes")
    for arm, names in SETS.items():
        X = torch.from_numpy(np.stack([np.stack(chan[k]) for k in names], 1)).to(dev)  # [n, d, 64, 64]
        prob = np.zeros((n, 64, 64), np.float32)
        for f in range(4):
            shut = {x for r in recs if r["fold"] == f for x in (r["support"], r["query"])}
            tr = torch.tensor([j for j, r in enumerate(recs) if r["fold"] != f and r["support"] not in shut and r["query"] not in shut], device=dev)
            te = torch.tensor(np.nonzero(folds == f)[0], device=dev)
            mu, sd = X[tr].mean((0, 2, 3), keepdim=True), X[tr].std((0, 2, 3), keepdim=True) + 1e-6
            for seed in range(a.seeds):
                torch.manual_seed(seed)
                layers, d = [], len(names)
                for dil in (1, 2, 4, 1):
                    layers += [nn.Conv2d(d, 48, 3, padding=dil, dilation=dil), nn.GroupNorm(8, 48), nn.ReLU()]
                    d = 48
                net = nn.Sequential(*layers, nn.Dropout2d(0.1), nn.Conv2d(48, 1, 1)).to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
                for _ in range(a.epochs):
                    perm = tr[torch.randperm(len(tr), device=dev)]
                    for s in range(0, len(perm), 32):
                        idx = perm[s:s + 32]
                        x, y = (X[idx] - mu) / sd, Y[idx]
                        if torch.rand(()) < 0.5:
                            x, y = x.flip(3), y.flip(2)
                        logit = net(x)[:, 0]
                        pr = logit.sigmoid()
                        soft_iou = 1 - (pr * y).sum((1, 2)) / (pr + y - pr * y).sum((1, 2)).clamp_min(1e-6)
                        loss = F.binary_cross_entropy_with_logits(logit, (y > 0.5).float()) + soft_iou.mean()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                net.eval()
                with torch.no_grad():
                    prob[te.cpu().numpy()] += net((X[te] - mu) / sd)[:, 0].sigmoid().cpu().numpy() / a.seeds
        t = iu(prob > 0.5)
        d = miou(t[:, 0], t[:, 1], cls, w) - miou(base[:, 0], base[:, 1], cls, w)
        out["arms"][arm] = dict(patch_miou=score(t), gain=score(t) - score(base), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                                by_fold={int(f): score(t, folds == f) - score(base, folds == f) for f in range(4)}, inputs=list(names))
        print(arm, json.dumps(out["arms"][arm]), flush=True)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
