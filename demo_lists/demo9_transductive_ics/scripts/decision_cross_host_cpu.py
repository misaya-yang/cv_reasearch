#!/usr/bin/env python3
"""Cross-host decision, CPU: the same read-out on a different host's mask (PLAN 2026-10-04, method M3).

The read-out is fitted once on the training episodes with FoRIS's mask as the host (`convctx:agnostic`: the host's
binary mask, the relations to the reference, the layer margins, and how each patch relates to the host's mask
through the query's own feature graph). Here the identical fitted model is applied to another host's binary mask on
the same episodes - the pair evidence (relations, margins, graph) stays FoRIS's, so what changes is only which
decision the read-out starts from and re-decides. Patch level, no refinement (a premise reading, not the method's
pipeline score).

  python scripts/decision_cross_host_cpu.py --cache cache/decision_v1 \
      --model results/decision_cpu_20261004/fit_agnostic/models/convctx_agnostic.pt \
      --sam3 /root/autodl-tmp/demo9_transductive_ics/results/sam3_handover_v3 --set confirm \
      --out results/decision_cpu_20261004/cross_host_confirm.json

Rows: FoRIS native; SAM3 native; the read-out on FoRIS's mask; the read-out on SAM3's mask; plus the read-out's
removal-only variant on each. Paired class-mIoU intervals over 2000 draws, per fold included.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def sam3_mask(npz_path, h, w, key="visual"):
    """The stored SAM3 mask (packed bits + shape) resampled to the patch grid the way the cache builds targets."""
    import torch
    import torch.nn.functional as F
    with np.load(npz_path) as z:
        packed = np.asarray(z[key]).reshape(-1)
        shape = tuple(int(v) for v in z["shape"])
    bits = np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape)
    t = torch.from_numpy(bits.astype(np.float32))[None, None]
    pooled = F.interpolate(t, (h, w), mode="area")[0, 0]
    return (pooled > 0.5).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="a fitted convctx:agnostic export")
    p.add_argument("--sam3", type=Path, required=True, help="directory with A_dev/masks and A_confirm/masks")
    p.add_argument("--set", default="confirm", choices=("dev", "confirm", "test"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    import torch
    from decision_fit import load, table
    from tics.decision_heads import agnostic, restore
    folder = {"dev": "test", "test": "test", "confirm": "confirm"}[a.set]
    d = load(a.cache / folder)
    n = len(d["fold"]) if not a.limit else a.limit
    x = torch.from_numpy(d["x"][:n].astype(np.float32))
    idx = torch.from_numpy(d["idx"][:n])
    sim = torch.from_numpy(d["sim"][:n])
    tf = d["tf"][:n]
    fold, cls, names = d["fold"][:n], d["cls"][:n], d["names"][:n]
    saved = torch.load(a.model, map_location="cpu", weights_only=False)
    assert saved["channels"] == "agnostic", "this script expects the agnostic channels, got %s" % saved["channels"]
    models = {f: restore(e, "cpu") for f, e in saved["models"].items()}
    mask_dir = a.sam3 / ("A_dev" if a.set in ("dev", "test") else "A_confirm") / "masks"
    h, w = d["x"].shape[-2:]

    ag_foris, ag_sam3, sam3_native = [], [], []
    missing = 0
    for i in range(n):
        xi = x[i:i + 1]
        ag_foris.append(agnostic(xi, idx[i:i + 1], sim[i:i + 1]))
        path = mask_dir / (names[i] + ".npz")
        if path.is_file():
            m = sam3_mask(path, h, w)
        else:
            missing += 1
            m = None
        if m is None:
            ag_sam3.append(ag_foris[-1])  # a missing case falls back to the FoRIS mask and is counted
            sam3_native.append(xi[:, 0].numpy() > 0.5)
        else:
            x2 = xi.clone()
            x2[:, 0] = torch.from_numpy(m.astype(np.float32))[None]
            ag_sam3.append(agnostic(x2, idx[i:i + 1], sim[i:i + 1]))
            sam3_native.append(m[None])
    ag_foris = torch.cat(ag_foris)
    ag_sam3 = torch.cat(ag_sam3)
    sam3_native = np.concatenate(sam3_native)

    prob_foris, prob_sam3 = np.zeros((n,) + (h, w), np.float32), np.zeros((n,) + (h, w), np.float32)
    for f in range(4):
        te = fold == f
        if not te.any():
            continue
        prob_foris[te] = models[f](ag_foris[torch.from_numpy(np.flatnonzero(te))]).numpy()
        prob_sam3[te] = models[f](ag_sam3[torch.from_numpy(np.flatnonzero(te))]).numpy()
    foris_native = x[:, 0].numpy() > 0.5
    read_foris, read_sam3 = prob_foris > 0.5, prob_sam3 > 0.5

    row = table(tf, cls, fold)
    rows = {
        "foris_native": row(foris_native, foris_native),
        "sam3_native": row(sam3_native > 0.5, foris_native),
        "readout_on_foris": row(read_foris, foris_native),
        "readout_on_sam3": row(read_sam3, foris_native),
        "readout_on_foris_removal_only": row(read_foris & foris_native, foris_native),
        "readout_on_sam3_removal_only": row(read_sam3 & (sam3_native > 0.5), foris_native),
        "sam3_or_readout": row((sam3_native > 0.5) | read_sam3, foris_native),
    }
    rep = dict(state="COMPLETED", set=a.set, episodes=int(n), patch_grid=int(h * w), missing_sam3=int(missing),
               model=str(a.model), sam3_root=str(a.sam3), rows=rows, scope="patch level, no refinement")
    for k, v in rows.items():
        print("%-32s %6.2f  %+6.2f [%+.2f, %+.2f]  folds %s" % (
            k, v["miou"], v["over_foris"], *v["ci95"], " ".join("%+.1f" % v_ for v_ in v["per_fold"])), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(dict(state="COMPLETED", rows={k: round(v["over_foris"], 3) for k, v in rows.items()})))


if __name__ == "__main__":
    main()