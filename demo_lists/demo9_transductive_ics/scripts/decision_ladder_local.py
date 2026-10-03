#!/usr/bin/env python3
"""Is the decision learnable across classes? The ladder of tics/decision_heads.py on the 241 saved episodes, each
fold scored by rungs fitted on the other three folds (its classes are never seen). Files on this machine, no encoder.

  python scripts/decision_ladder_local.py --run results/extent_v1/run --maps results/evidence_v1/maps \
      --out results/self_support_v0/decision_ladder.json

Inputs: the 12 class-free maps that can be rebuilt from the saved packets (FoRIS score and its two stage scores, the
rank, nearest reference foreground and background similarity, the label and the round trip of the nearest reference
patch, similarity to the query's own confident core and sure background, the two neighbour edges). Patch level, no
refinement. A premise test with about 180 training episodes per fit, not a method score.

Card, written 2026-10-03 before the run. Reference: FoRIS 56.99; a plain convolutional read-out on the same number
of episodes lost 3.4 (results/self_support_v0/decoder_probe.json); perfect removal +19.99, true share +9.07.
  Assumption: with the host's mask as the starting point and the stopping epoch chosen on held-out classes, a
    read-out of class-free evidence improves on FoRIS for unseen classes even with 180 episodes.
  Prediction: cut within +-0.5 of FoRIS; pixel +0.5 to +1.5; mlp and context +0.5 to +2; conv and convctx between
    -1 and +2; the removal variant (read-out intersected with FoRIS's mask) not below the read-out by more than 0.5;
    gains grow from 60 to 180 training episodes for at least the per-patch rungs.
  Match: the decision is learnable class-free and limited by data; the server run with 1800 episodes per fit is
    predicted at about twice the best gain here, and the conv rungs are expected to overtake the per-patch rungs.
  Mismatch: no rung above +0.5 and a flat curve: these 12 maps do not carry the decision; the server run must lead
    with the wider inputs (other layers, feature components), not with more episodes of the same maps.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from analyze_extent import load, miou  # noqa: E402
from evidence_audit import normalise  # noqa: E402
from tics.decision_heads import RUNGS, fit  # noqa: E402

NAMES = ("sn", "s2", "s3", "rank", "fg", "bg", "nn_label", "cycle", "grp_core", "grp_out", "edge_h", "edge_v")
SETS = dict(score=("sn",), evidence=NAMES)


def inputs(run, maps, r):
    name = "%d_%d_%d" % (r["fold"], r["e"], r["c"])
    z, m = np.load(run / "packets" / (name + ".npz")), np.load(maps / (name + ".npz"))
    tf = np.unpackbits(z["truth"])[:1 << 20].reshape(64, 16, 64, 16).mean((1, 3))
    sn = normalise(z["score"].astype(np.float32))
    g = lambda k: m[k].astype(np.float32).reshape(64, 64)
    cov, fwd, back = z["cov"].ravel(), z["fwd_idx"].astype(np.int64), z["back_idx"].astype(np.int64)
    pos, there = np.arange(4096), back[fwd]
    close = np.hypot(there // 64 - pos // 64, there % 64 - pos % 64) <= 2
    er, ed = np.pad(1 - z["aff_r"], ((0, 0), (0, 1))), np.pad(1 - z["aff_d"], ((0, 1), (0, 0)))
    edge_h = (er + np.pad(er, ((0, 0), (1, 0)))[:, :-1]) / 2  # both horizontal neighbours, so a flip keeps the map
    edge_v = (ed + np.pad(ed, ((1, 0), (0, 0)))[:-1]) / 2
    x = dict(sn=sn, s2=normalise(z["s2"].astype(np.float32)), s3=normalise(z["s3"].astype(np.float32)),
             rank=(sn.ravel().argsort().argsort() / 4095.0).reshape(64, 64).astype(np.float32), fg=g("fg"), bg=g("bg"),
             nn_label=cov[fwd].reshape(64, 64), cycle=(cov[fwd] * close).reshape(64, 64), grp_core=g("grp_core"),
             grp_out=g("grp_out"), edge_h=edge_h, edge_v=edge_v)
    return np.stack([x[k] for k in NAMES]).astype(np.float32), tf.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--maps", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--curve", default="60,120")
    p.add_argument("--rungs", default=",".join(RUNGS))
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    recs = load(a.run)
    n = len(recs)
    both = [inputs(a.run, a.maps, r) for r in recs]
    X, Y = torch.from_numpy(np.stack([b[0] for b in both])), torch.from_numpy(np.stack([b[1] for b in both]))
    cls, fold = np.array([r["c"] for r in recs]), np.array([r["fold"] for r in recs])
    pair = (NAMES.index("edge_h"), NAMES.index("edge_v"))
    host = (X[:, 0] > 0.5).numpy()
    tf = Y.numpy()

    def score(mask):
        i = (mask * tf).sum((1, 2))
        return np.stack([i, tf.sum((1, 2)) + mask.sum((1, 2)) - i], 1)
    base = score(host)
    w = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=2000).astype(float)
    ref = miou(base[:, 0], base[:, 1], cls)[0]
    out = dict(state="RUNNING", episodes=n, scope="patch level, no refinement; each fold scored by rungs fitted on the other three",
               foris_patch_miou=float(ref), inputs=list(NAMES), rows={}, curve={})

    def row(mask):
        t = score(mask)
        d = miou(t[:, 0], t[:, 1], cls, w) - miou(base[:, 0], base[:, 1], cls, w)
        per = [float(miou(t[fold == f, 0], t[fold == f, 1], cls[fold == f])[0] - miou(base[fold == f, 0], base[fold == f, 1], cls[fold == f])[0]) for f in range(4)]
        return dict(over_foris=float(miou(t[:, 0], t[:, 1], cls)[0] - ref), ci95=[float(v) for v in np.percentile(d, [2.5, 97.5])], per_fold=per)

    def held_out(rung, chans, limit=None):
        prob, notes = np.zeros((n, 64, 64), np.float32), []
        for f in range(4):
            tr = np.flatnonzero(fold != f)
            if limit:
                tr = np.sort(np.random.default_rng(0).permutation(tr)[:limit])
            sel = [NAMES.index(k) for k in chans]
            pr = tuple(sel.index(v) for v in pair) if all(v in sel for v in pair) else None
            model, note = fit(rung, X[tr][:, sel], Y[tr], torch.from_numpy(cls[tr]), sel.index(0), epochs=a.epochs,
                              seeds=a.seeds, pair=pr, dev=a.device)
            prob[fold == f] = model(X[fold == f][:, sel]).numpy()
            notes.append({k: note[k] for k in ("epochs", "weights")})
        return prob, notes
    for rung in a.rungs.split(","):
        for tag, chans in SETS.items():
            if tag == "score" and rung not in ("cut", "conv"):
                continue
            prob, notes = held_out(rung, chans)
            name = "%s|%s" % (rung, tag)
            out["rows"][name] = dict(row(prob > 0.5), removal_only=row((prob > 0.5) & host), fits=notes)
            v = out["rows"][name]
            print("%-18s %+6.2f [%+.2f,%+.2f]  folds %s  removal only %+6.2f  epochs %s" % (
                name, v["over_foris"], *v["ci95"], " ".join("%+.1f" % x for x in v["per_fold"]), v["removal_only"]["over_foris"],
                [k["epochs"] for k in notes]), flush=True)
            if a.out:
                a.out.write_text(json.dumps(out, indent=1))
    for limit in [int(v) for v in a.curve.split(",") if v]:
        for rung in ("pixel", "context", "convctx"):
            if rung in a.rungs.split(","):
                prob, _ = held_out(rung, NAMES, limit)
                out["curve"]["%s|%d" % (rung, limit)] = row(prob > 0.5)
                print("curve %-10s %4d episodes  %+6.2f" % (rung, limit, out["curve"]["%s|%d" % (rung, limit)]["over_foris"]), flush=True)
    out["state"] = "ANALYSED"
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
