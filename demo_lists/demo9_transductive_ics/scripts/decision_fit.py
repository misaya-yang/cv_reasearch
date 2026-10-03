#!/usr/bin/env python3
"""Is the decision learnable across classes, from which evidence, and how fast does it grow with base data?

  python scripts/decision_fit.py --cache cache/decision_v1 --out results/decision_v1/fit        # under the guard
  python scripts/decision_fit.py --cache DIR --out OUT --arms mlp:relations --curve "" --unguarded --device cpu

Every read-out of tics/decision_heads.py starts as FoRIS's own mask. A model that scores fold f is fitted on the
training episodes of the other three folds (60 classes), its number of epochs chosen on a fifth of those classes.
Three readings, in this order:
  development   every arm on the 241 development episodes; the arm for the next step is chosen here and written to
                selection.json before anything else is opened
  curve         the chosen kinds of arm refitted on 150, 300, 600, 1200 training episodes per model
  confirmation  the chosen arm, its score-only control and FoRIS on the confirmation episodes; their labels are read
                from the annotation files only now
Patch level (64 x 64), no refinement: a premise test of the read-out, not the method's score.

Card, written 2026-10-03 before the run. Measured before on the 241 development episodes, patch level: FoRIS 56.99;
true share +9.07; every wrongly included patch removed +19.99; on 180 training episodes (files on the laptop,
results/self_support_v0/decision_ladder.json) cut +0.29, pixel +0.04, mlp +1.67 [+0.53, +2.96], conv on the score
alone +1.59, conv on 12 maps +1.64, each fold fitted on the other three.
  Assumption: where FoRIS's evidence should be cut in a query is decided the same way for every class, so it can
    be learned on base classes from class-free pair evidence and used on unseen classes.
  Prediction (1800 training episodes per model, development): cut +0.2 to +0.6; pixel 0 to +1; mlp on relations +2
    to +3.5; conv and convctx on relations +2.5 to +5; middle-layer margins add 0 to +1.5; principal components
    add -1 to +1.5; unroll adds +0.5 to +2 over convctx; removal only (read-out inside FoRIS's mask) within 0.7 of
    the read-out; the best conv rung gains at least +0.4 per doubling between 300 and 1800 episodes; conv on the
    score alone +1.5 to +3 (how much is a re-shaping of the score rather than new evidence). Patch-level removal by
    a reference read-out: block-12 mean margin -8 to -3, final-layer mean margin -1 to 0.
  Match: the decision is learnable class-free and limited by base data: build the train2014 cache (standard
    protocol), keep the chosen rung, read at original resolution with refinement on 1000 episodes per fold.
  Mismatch: every rung under +1.5 with a flat curve: these inputs do not carry the decision at this scale, lead
    with raw features or stop the trained line; conv rungs under mlp: capacity outruns the data, shrink and add
    data; principal components far above relations: the knowledge sits in raw features, build the decoder on
    them; evidence arms not above conv on the score alone: the gain is re-shaping, not evidence.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from analyze_extent import miou  # noqa: E402

GATE = dict(go_direct=4.0, go_scale=2.5, slope=0.4, folds=3)
RELATIONS = ("sn", "s2", "s3", "rank", "fg", "bg", "nn_label", "cycle", "grp_core", "grp_out", "grp_core5", "grp_out5",
             "proto_core", "proto_out", "edge_h", "edge_v")
ARMS = ("cut:score", "conv:score", "pixel:relations", "mlp:relations", "context:relations", "conv:relations",
        "convctx:relations", "mlp:layers", "convctx:layers", "convctx:features", "unroll:relations", "unroll:layers")
CURVE_ARMS = ("mlp:relations", "convctx:relations", "unroll:layers")
SETS = dict(score=slice(0, 1), relations=slice(0, 16), layers=slice(0, 24), features=slice(0, 40))


def load(folder, rows=None, targets=True):
    """Arrays of a cache folder (all of it, or the episodes of `rows`). Channels: 16 relations (the two edge maps made
    symmetric), 8 layer margins, then the 16 principal components of each of the 4 models (kept apart, a model sees
    only its own)."""
    files = sorted(folder.glob("*.npz")) if rows is None else [folder / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])) for r in rows]
    z = [np.load(f, allow_pickle=False) for f in files]
    maps = np.stack([x["maps"] for x in z]).astype(np.float32)
    er, ed = maps[:, 14], maps[:, 15]  # 1 - similarity to the right and to the lower neighbour, zero at the border
    maps[:, 14] = (er + np.pad(er, ((0, 0), (0, 0), (1, 0)))[:, :, :-1]) / 2
    maps[:, 15] = (ed + np.pad(ed, ((0, 0), (1, 0), (0, 0)))[:, :-1]) / 2
    out = dict(x=np.concatenate([maps, np.stack([x["layers"] for x in z]).astype(np.float32)], 1).astype(np.float16),
               pca=np.stack([x["pca"] if "pca" in x.files else np.zeros((4, 16) + maps.shape[-2:], np.float16) for x in z]),
               idx=np.stack([x["nbr_idx"] for x in z]), sim=np.stack([x["nbr_sim"] for x in z]),
               fold=np.array([int(f.stem.split("_")[0]) for f in files]), cls=np.array([int(f.stem.split("_")[2]) for f in files]),
               names=[f.stem for f in files], mask_path=[str(x["query_mask_path"]) for x in z], shape=[tuple(int(v) for v in x["model_shape"]) for x in z])
    if targets:
        out["tf"] = np.stack([x["tf"] for x in z]).astype(np.float32)
    return out


def read_targets(d):
    """The labels of the confirmation queries, on the patch grid, exactly as the cache makes them for the other sets."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    tf = []
    for path, shape, c in zip(d["mask_path"], d["shape"], d["cls"]):
        truth = torch.from_numpy((np.asarray(Image.open(path)) == c + 1).copy())
        truth = F.interpolate(truth[None, None].float(), shape, mode="nearest")
        tf.append(F.avg_pool2d(truth, shape[0] // d["x"].shape[-1])[0, 0].numpy())
    return np.stack(tf).astype(np.float32)


def table(tf, cls, fold, draws=2000):
    """Returns a function mask -> row (gain over FoRIS with interval and per-fold gains); FoRIS is the mask sn > 0.5."""
    n = len(tf)
    w = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=draws).astype(float)

    def iu(mask):
        i = (mask * tf).sum((1, 2))
        return np.stack([i, tf.sum((1, 2)) + mask.sum((1, 2)) - i], 1)

    def row(mask, base):
        t, b = iu(mask), iu(base)
        d = miou(t[:, 0], t[:, 1], cls, w) - miou(b[:, 0], b[:, 1], cls, w)
        per = [float(miou(t[fold == f, 0], t[fold == f, 1], cls[fold == f])[0] - miou(b[fold == f, 0], b[fold == f, 1], cls[fold == f])[0]) for f in range(4)]
        return dict(miou=float(miou(t[:, 0], t[:, 1], cls)[0]), over_foris=float(miou(t[:, 0], t[:, 1], cls)[0] - miou(b[:, 0], b[:, 1], cls)[0]),
                    ci95=[float(v) for v in np.percentile(d, [2.5, 97.5])], per_fold=per)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--curve", default="150,300,600,1200")
    p.add_argument("--curve-arms", default=",".join(CURVE_ARMS))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, help="use only this many training episodes (checks)")
    p.add_argument("--train-manifest", help="fit only on the episodes of this manifest (default: every cached training episode)")
    p.add_argument("--unguarded", action="store_true")
    p.add_argument("--no-confirm", action="store_true", help="checks: never open the confirmation labels")
    a = p.parse_args()
    import torch
    torch.backends.cudnn.benchmark = True
    from tics.decision_heads import fit
    if a.device == "cuda" and not a.unguarded and os.environ.get("DEMO9_CUDA_GUARD") != "1":
        raise SystemExit("run under scripts/experiment_resource_guard.py, or pass --unguarded")
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "report.json").exists():
        raise SystemExit("fresh output directory required")
    start = time.monotonic()
    tr = load(a.cache / "train", json.loads(Path(a.train_manifest).read_text())["episodes"] if a.train_manifest else None)
    dv = load(a.cache / "test")
    if a.limit:
        keep = np.sort(np.random.default_rng(0).permutation(len(tr["fold"]))[:a.limit])
        tr = {k: ([v[i] for i in keep] if isinstance(v, list) else v[keep]) for k, v in tr.items()}
    pair = (14, 15)
    T = lambda v: torch.from_numpy(v)
    Xtr, Ytr, Gtr = T(tr["x"]), T(tr["tf"]), (T(tr["idx"]), T(tr["sim"]))
    report = dict(state="RUNNING", train=len(tr["fold"]), development=len(dv["fold"]), scope="patch level, no refinement",
                  arms={}, curve={}, reference={}, fits={})

    def save():
        report["elapsed_s"] = time.monotonic() - start
        tmp = a.out / "report.json.tmp"
        tmp.write_text(json.dumps(report, indent=1))
        tmp.replace(a.out / "report.json")

    def inputs(d, f, name):
        """Maps of the episodes for model f (held-out fold f): the common channels, then model f's own components."""
        x = T(d["x"])
        if name == "features":
            x = torch.cat([x, T(d["pca"][:, f])], 1)
        return x[:, SETS[name]]

    def fitted(arm, limit=None):
        """Held-out probabilities on the development episodes, and the four models."""
        rung, name = arm.split(":")
        prob, models, notes, exports = np.zeros(dv["tf"].shape, np.float32), {}, [], {}
        for f in range(4):
            rows = np.flatnonzero(tr["fold"] != f)
            if limit:
                rows = np.sort(np.random.default_rng(0).permutation(rows)[:limit])
            x = inputs(tr, f, name)[rows]
            graph = (Gtr[0][rows].to(a.device), Gtr[1][rows].to(a.device)) if rung == "unroll" else None
            model, note = fit(rung, x, Ytr[rows], T(tr["cls"][rows]), 0, epochs=a.epochs, batch=a.batch, seeds=a.seeds,
                              pair=pair if x.shape[1] > 15 else None, dev=a.device, graph=graph)
            models[f] = model
            exports[f] = note["export"]
            notes.append({k: note[k] for k in ("epochs", "weights")})
            te = dv["fold"] == f
            if te.any():
                prob[te] = apply(model, dv, f, name, rung, te).numpy()
        if not limit:  # the models that score fold f, for inference at original resolution later
            (a.out / "models").mkdir(exist_ok=True)
            torch.save(dict(arm=arm, channels=name, models=exports), a.out / "models" / (arm.replace(":", "_") + ".pt"))
        return prob, models, notes

    def apply(model, d, f, name, rung, rows):
        g = (T(d["idx"][rows]), T(d["sim"][rows])) if rung == "unroll" else None
        return model(inputs(d, f, name)[torch.from_numpy(np.flatnonzero(rows))], g)
    row = table(dv["tf"], dv["cls"], dv["fold"])
    host = dv["x"][:, 0].astype(np.float32) > 0.5
    truth = dv["tf"] > 0.5
    count = truth.reshape(len(truth), -1).sum(1)
    order = np.argsort(-dv["x"][:, 0].astype(np.float32).reshape(len(truth), -1), 1)
    share = np.zeros(order.shape, bool)
    for j in range(len(truth)):
        share[j, order[j, :max(int(count[j]), 1)]] = True
    report["reference"] = {"FoRIS": row(host, host), "FoRIS, true share  LABELS": row(share.reshape(host.shape), host),
                           "every wrongly included patch removed  LABELS": row(host & truth, host),
                           "removal by block-12 mean margin": row(host & (dv["x"][:, 18].astype(np.float32) > 0), host),
                           "removal by final-layer mean margin": row(host & (dv["x"][:, 22].astype(np.float32) > 0), host)}
    for k, v in report["reference"].items():
        print("%-48s %6.2f  %+6.2f [%+.2f, %+.2f]" % (k, v["miou"], v["over_foris"], *v["ci95"]), flush=True)
    save()
    kept = {}
    for arm in a.arms.split(","):
        prob, models, notes = fitted(arm)
        kept[arm] = models
        report["arms"][arm] = dict(row(prob > 0.5, host), removal_only=row((prob > 0.5) & host, host), addition_only=row((prob > 0.5) | host, host), fits=notes)
        v = report["arms"][arm]
        print("%-20s %+6.2f [%+.2f, %+.2f]  folds %s  removal only %+6.2f  addition only %+6.2f  epochs %s" % (
            arm, v["over_foris"], *v["ci95"], " ".join("%+.1f" % x for x in v["per_fold"]), v["removal_only"]["over_foris"],
            v["addition_only"]["over_foris"], [k["epochs"] for k in notes]), flush=True)
        save()
    # The arm for the next step: the best on development among the arms that use evidence; frozen before anything else.
    cands = [k for k in report["arms"] if not k.endswith(":score")]
    chosen = max(cands, key=lambda k: report["arms"][k]["over_foris"]) if cands else None
    control = "conv:score" if "conv:score" in report["arms"] else None
    (a.out / "selection.json").write_text(json.dumps(dict(state="SELECTION_FROZEN", chosen=chosen, control=control,
                                                          development=report["arms"].get(chosen)), indent=1))
    report["chosen"] = chosen
    save()
    for limit in [int(v) for v in a.curve.split(",") if v]:
        for arm in a.curve_arms.split(","):
            if arm in report["arms"]:
                prob, _, _ = fitted(arm, limit)
                report["curve"]["%s|%d" % (arm, limit)] = row(prob > 0.5, host)
                print("curve %-20s %5d  %+6.2f" % (arm, limit, report["curve"]["%s|%d" % (arm, limit)]["over_foris"]), flush=True)
                save()
    if chosen and not a.no_confirm and (a.cache / "confirm").is_dir() and any((a.cache / "confirm").glob("*.npz")):
        cf = load(a.cache / "confirm", targets=False)
        cf["tf"] = read_targets(cf)  # the first time these labels are opened
        crow = table(cf["tf"], cf["cls"], cf["fold"])
        chost = cf["x"][:, 0].astype(np.float32) > 0.5
        report["confirmation"] = dict(episodes=len(cf["fold"]), FoRIS=crow(chost, chost))
        for arm in [chosen] + ([control] if control else []):
            rung, name = arm.split(":")
            prob = np.zeros(cf["tf"].shape, np.float32)
            for f in range(4):
                te = cf["fold"] == f
                if te.any():
                    prob[te] = apply(kept[arm][f], cf, f, name, rung, te).numpy()
            report["confirmation"][arm] = dict(crow(prob > 0.5, chost), removal_only=crow((prob > 0.5) & chost, chost))
            v = report["confirmation"][arm]
            print("confirmation %-20s %+6.2f [%+.2f, %+.2f]  folds %s  removal only %+6.2f" % (
                arm, v["over_foris"], *v["ci95"], " ".join("%+.1f" % x for x in v["per_fold"]), v["removal_only"]["over_foris"]), flush=True)
        got = report["confirmation"][chosen]
        ok = got["ci95"][0] > 0 and sum(x > 0 for x in got["per_fold"]) >= GATE["folds"]
        report["confirmation"]["verdict"] = ("GO_DIRECT" if ok and got["over_foris"] >= GATE["go_direct"] else
                                             "GO_SCALE" if ok and got["over_foris"] >= GATE["go_scale"] else "BELOW_GATE")
    report["state"] = "COMPLETED"
    save()
    print(json.dumps(dict(state="COMPLETED", chosen=chosen, verdict=report.get("confirmation", {}).get("verdict"),
                          elapsed_s=report["elapsed_s"])), flush=True)


if __name__ == "__main__":
    main()
