#!/usr/bin/env python3
"""Can the extent be learned without the class? Heads fitted on base-class episodes of three folds, scored on the
fourth fold's novel classes.

  python scripts/extent_head.py --cache cache/extent_head_v0 --run results/extent_v1/run --out results/extent_head_v0/report.json
  python scripts/extent_head.py --selfcheck

Heads (inputs are the relation maps of tics/relations.py):
  dense       a small convolutional net gives the mask directly
  dense_area  the same net only says how large the target is; the mask is the FoRIS level set of that area
  size        a regressor of the log area; the mask is the FoRIS level set of that area
  quality     a net that sees a candidate cut of the FoRIS score and predicts its IoU; the best of 8 cuts is taken
Input sets: the score alone (control: what a fitted read-out gets from the same information), all relation maps,
and relation maps plus 16 principal components of the query features (not class-free, to see what raw features add).
Upper bound: the FoRIS level set of the true area. Patch level, no refinement; a premise test, not a method score.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from analyze_extent import load, miou  # noqa: E402

# Written 2026-10-03 before the run. Reference points (241 episodes, patch level unless stated): native 56.99; the
# level set of the true area +8.7 (model size); FoRIS's own log-area error 0.67; a read-out fitted on about 180
# episodes per fold loses 3.4 with relation inputs and gains 0.15 with the score alone.
CARD = dict(
    assumption="how large the target is in the query, and where it ends, does not depend on the class and can be "
               "learned from base-class episodes",
    prediction="with about 1800 training episodes per fold: relation inputs give the dense head +2 to +5 over native "
               "with the interval above 0 and at least +2 over the score-only control; log-area error of the size "
               "estimate 0.45 or less (FoRIS's own 0.67); the score-only control within 1 of native",
    match="extent is learnable across classes: train on train2014 with the standard protocol, add refinement, run "
          "1000 episodes per fold",
    mismatch="relation inputs not above the score-only control: these inputs do not hold the size, change the inputs "
             "(other layers, raw features), not the task; principal components far above relations: the knowledge "
             "sits in raw features, build the head on them; training fit high and held-out low: more base data")
GATE = dict(gain=4.0, folds=4)
LEVELS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # candidate cuts of the normalised FoRIS score for the quality head


def load_dir(folder, rows=None):
    files = sorted(folder.glob("*.npz")) if rows is None else [folder / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])) for r in rows]
    z = [np.load(f) for f in files]
    fold = np.array([int(f.stem.split("_")[0]) for f in files])
    return (np.stack([x["maps"] for x in z]).astype(np.float32), np.stack([x["pca"] for x in z]).astype(np.float32),
            np.stack([x["tf"] for x in z]).astype(np.float32), np.array([float(x["ref_area"]) for x in z], np.float32), fold)


def level_of_area(sn, k):
    """The FoRIS level set with k patches: the k highest scores. sn: [n, P]; k: [n] ints."""
    order = np.argsort(-sn, 1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(sn.shape[1])[None].repeat(len(sn), 0), 1)
    return rank < np.clip(k, 1, sn.shape[1])[:, None]


def fit(a, Xtr, Ttr, Xte, d, size_head, dev):
    """Returns held-out outputs averaged over seeds, and the fit on a sample of the training episodes."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    mu, sd = Xtr.mean((0, 2, 3), keepdim=True), Xtr.std((0, 2, 3), keepdim=True) + 1e-6
    target = torch.log(Ttr.mean((1, 2)).clamp_min(1e-5))
    sample = torch.arange(0, len(Xtr), max(1, len(Xtr) // 200), device=dev)
    held, seen = 0, 0
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        if size_head:
            layers, c = [], d
            for _ in range(4):
                layers += [nn.Conv2d(c, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.ReLU(), nn.AvgPool2d(2)]
                c = 48
            net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
        else:
            layers, c = [], d
            for dil in (1, 2, 4, 8, 1, 1):
                layers += [nn.Conv2d(c, 64, 3, padding=dil, dilation=dil), nn.GroupNorm(8, 64), nn.ReLU()]
                c = 64
            net = nn.Sequential(*layers, nn.Dropout2d(0.1), nn.Conv2d(64, 1, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        steps = a.epochs * ((len(Xtr) + a.batch - 1) // a.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        net.train()
        for _ in range(a.epochs):
            perm = torch.randperm(len(Xtr), device=dev)
            for s in range(0, len(perm), a.batch):
                idx = perm[s:s + a.batch]
                x, y = (Xtr[idx] - mu) / sd, Ttr[idx]
                if torch.rand(()) < 0.5:
                    x, y = x.flip(3), y.flip(2)
                if size_head:
                    loss = F.smooth_l1_loss(net(x)[:, 0], target[idx])
                else:
                    logit = net(x)[:, 0]
                    pr = logit.sigmoid()
                    soft = 1 - (pr * y).sum((1, 2)) / (pr + y - pr * y).sum((1, 2)).clamp_min(1e-6)
                    loss = F.binary_cross_entropy_with_logits(logit, (y > 0.5).float()) + soft.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        with torch.no_grad():
            run = lambda x: torch.cat([net((x[i:i + 128] - mu) / sd)[:, 0] for i in range(0, len(x), 128)])
            post = (lambda v: v) if size_head else torch.sigmoid
            held = held + post(run(Xte)).cpu().numpy() / a.seeds
            seen = seen + post(run(Xtr[sample])).cpu().numpy() / a.seeds
    return held, seen, sample.cpu().numpy()


def fit_quality(a, Xtr, Ttr, Xte, d, dev):
    """Predicted IoU of each candidate cut for the held-out episodes: [n, len(LEVELS)]. Channel 0 is the score."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    mu, sd = Xtr.mean((0, 2, 3), keepdim=True), Xtr.std((0, 2, 3), keepdim=True) + 1e-6
    levels = torch.tensor(LEVELS, device=dev)
    held = 0
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        layers, c = [], d + 1
        for _ in range(4):
            layers += [nn.Conv2d(c, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.ReLU(), nn.AvgPool2d(2)]
            c = 48
        net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * ((len(Xtr) + a.batch - 1) // a.batch))
        net.train()
        for _ in range(a.epochs):
            perm = torch.randperm(len(Xtr), device=dev)
            for s in range(0, len(perm), a.batch):
                idx = perm[s:s + a.batch]
                x, y = Xtr[idx], Ttr[idx]
                cand = (x[:, 0] > levels[torch.randint(len(LEVELS), (len(idx),), device=dev)][:, None, None]).float()
                inter = (y * cand).sum((1, 2))
                iou = inter / (y.sum((1, 2)) + cand.sum((1, 2)) - inter).clamp_min(1e-6)
                loss = F.binary_cross_entropy_with_logits(net(torch.cat([(x - mu) / sd, cand[:, None]], 1))[:, 0], iou)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        with torch.no_grad():
            cols = [torch.cat([net(torch.cat([(Xte[i:i + 128] - mu) / sd, (Xte[i:i + 128, :1] > t).float()], 1))[:, 0]
                               for i in range(0, len(Xte), 128)]).sigmoid() for t in LEVELS]
            held = held + torch.stack(cols, 1).cpu().numpy() / a.seeds
    return held


def evaluate(a, train, test, recs, dev, log=print):
    import torch
    Mtr, Ptr, Ttr, _, Ftr = train
    Mte, Pte, Tte, _, Fte = test
    n, side = len(Mte), Mte.shape[-1]
    P = side * side
    cls = np.array([r["c"] for r in recs])
    tf, sn = Tte.reshape(n, -1), Mte[:, 0].reshape(n, -1)

    def iu(pred):
        pred = pred.reshape(n, -1)
        i = (tf * pred).sum(1)
        return np.stack([i, tf.sum(1) + pred.sum(1) - i], 1)
    base = iu(sn > 0.5)
    w = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=2000).astype(float)
    score = lambda t, m=slice(None): float(miou(t[m, 0], t[m, 1], cls[m])[0])

    def row(pred, ref=base):
        t = iu(pred)
        d = miou(t[:, 0], t[:, 1], cls, w) - miou(ref[:, 0], ref[:, 1], cls, w)
        by = {int(f): score(t, Fte == f) - score(ref, Fte == f) for f in np.unique(Fte)}
        return dict(patch_miou=score(t), gain=score(t) - score(ref), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                    by_fold=by, folds_up=int(sum(v > 0 for v in by.values())))
    true_log = np.log(np.maximum(tf.mean(1), 1e-5))
    err = lambda est: dict(log_error_std=float(np.std(est - true_log)), log_error_bias=float(np.mean(est - true_log)),
                           log_error_median_abs=float(np.median(np.abs(est - true_log))))
    cuts = np.stack([sn > t for t in LEVELS], 1)  # [n, levels, P]
    cut_iou = (tf[:, None] * cuts).sum(2) / np.maximum(tf.sum(1)[:, None] + cuts.sum(2) - (tf[:, None] * cuts).sum(2), 1e-6)
    out = dict(native_patch_miou=score(base), true_area_level=row(level_of_area(sn, np.round(tf.sum(1)).astype(int))),
               best_of_candidate_cuts=row(cuts[np.arange(n), cut_iou.argmax(1)]),
               foris_own_area=err(np.log(np.maximum((sn > 0.5).mean(1), 1e-5))), arms={})
    sets = dict(score_only=(Mtr[:, :1], Mte[:, :1]), relations=(Mtr, Mte),
                relations_pca=(np.concatenate([Mtr, Ptr], 1), np.concatenate([Mte, Pte], 1)))
    for arm, (xtr, xte) in sets.items():
        Xtr, Xte, T = torch.from_numpy(xtr).to(dev), torch.from_numpy(xte).to(dev), torch.from_numpy(Ttr).to(dev)
        prob, seen_gain = np.zeros((n, side, side), np.float32), []
        size, qual = np.zeros(n, np.float32), np.zeros((n, len(LEVELS)), np.float32)
        for f in np.unique(Fte):
            tr = torch.from_numpy(np.nonzero(Ftr != f)[0]).to(dev)
            te = np.nonzero(Fte == f)[0]
            held, seen, sample = fit(a, Xtr[tr], T[tr], Xte[torch.from_numpy(te).to(dev)], xtr.shape[1], False, dev)
            prob[te] = held
            ttr, strn = Ttr[tr.cpu().numpy()][sample].reshape(len(sample), -1), xtr[tr.cpu().numpy()][sample][:, 0].reshape(len(sample), -1)
            fit_iou = lambda p: float(np.mean((ttr * p).sum(1) / np.maximum(ttr.sum(1) + p.sum(1) - (ttr * p).sum(1), 1e-6)))
            seen_gain.append(100 * (fit_iou(seen.reshape(len(sample), -1) > 0.5) - fit_iou(strn > 0.5)))
            if arm == "relations":
                size[te] = fit(a, Xtr[tr], T[tr], Xte[torch.from_numpy(te).to(dev)], xtr.shape[1], True, dev)[0]
            if arm != "relations_pca":
                qual[te] = fit_quality(a, Xtr[tr], T[tr], Xte[torch.from_numpy(te).to(dev)], xtr.shape[1], dev)
        area = prob.reshape(n, -1).sum(1)
        res = dict(dense=row(prob > 0.5), dense_area=dict(row(level_of_area(sn, np.round(area).astype(int))), **err(np.log(np.maximum(area / P, 1e-5)))),
                   training_fit_gain_mean_iou=float(np.mean(seen_gain)), inputs=int(xtr.shape[1]))
        if arm == "relations":
            res["size"] = dict(row(level_of_area(sn, np.round(np.exp(size) * P).astype(int))), **err(size))
        if arm != "relations_pca":
            res["quality"] = dict(row(cuts[np.arange(n), qual.argmax(1)]),
                                  rank_correlation_with_true_iou=float(np.mean([np.corrcoef(qual[j], cut_iou[j])[0, 1] for j in range(n) if cut_iou[j].std() > 0])))
        if arm != "score_only":
            res["dense_over_score_only"] = row(prob > 0.5, iu(out["arms"]["score_only"]["_mask"]))
        res["_mask"] = prob > 0.5
        out["arms"][arm] = res
        log(arm + " " + json.dumps({k: ({x: (round(y, 2) if isinstance(y, float) else y) for x, y in v.items() if x != "by_fold"} if isinstance(v, dict) else v)
                                    for k, v in res.items() if k != "_mask"}))
    for res in out["arms"].values():
        del res["_mask"]
    rel = out["arms"]["relations"]
    best = max((rel[k] for k in ("dense", "dense_area", "size", "quality")), key=lambda r: r["gain"])
    out["verdict"] = dict(best_relation_head_gain=best["gain"], ci95=best["ci95"], folds_up=best["folds_up"],
                          passes=bool(best["gain"] >= GATE["gain"] and best["ci95"][0] > 0 and best["folds_up"] >= GATE["folds"]))
    return out


def main_run(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs = load(a.run)
    cache = Path(a.cache)
    start = time.monotonic()
    train, test = load_dir(cache / "train"), load_dir(cache / "test", recs)
    res = dict(state="COMPLETED", card=CARD, gate=GATE, train_episodes=int(len(train[0])), test_episodes=int(len(test[0])),
               train_per_fold={int(f): int((train[4] == f).sum()) for f in np.unique(train[4])}, epochs=a.epochs, seeds=a.seeds,
               scope="patch level, no refinement; test classes unseen in fitting; premise test, not a method score",
               **evaluate(a, train, test, recs, dev))
    res["elapsed_s"] = time.monotonic() - start
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(state="COMPLETED", native=res["native_patch_miou"], true_area_level=res["true_area_level"]["gain"], verdict=res["verdict"])))


def selfcheck():
    """A world where the score overshoots the target by an attached region that only a second map tells apart."""
    rng = np.random.default_rng(0)
    side, K = 16, 3

    def make(m, fold):
        maps, tf = np.zeros((m, K, side, side), np.float32), np.zeros((m, side, side), np.float32)
        for j in range(m):
            x0, w0 = int(rng.integers(1, 5)), int(rng.integers(3, 7))
            extra = int(rng.integers(0, 5))  # width of the attached region; zero in some episodes
            tf[j, 4:12, x0:x0 + w0] = 1
            s = rng.uniform(0.0, 0.3, (side, side))
            s[4:12, x0:x0 + w0] = rng.uniform(0.6, 1.0, (8, w0))
            s[4:12, x0 + w0:x0 + w0 + extra] = rng.uniform(0.55, 0.95, (8, extra))  # overlaps the target's scores
            maps[j, 0] = (s - s.min()) / (s.max() - s.min())
            maps[j, 1] = tf[j] + 0.1 * rng.normal(size=s.shape)  # the map that tells the attached region apart
            maps[j, 2] = rng.normal(size=s.shape)
        return maps, np.zeros((m, 2, side, side), np.float32), tf, np.ones(m, np.float32), np.full(m, fold)
    cat = lambda parts: tuple(np.concatenate(x) for x in zip(*parts))
    train, test = cat([make(50, f) for f in range(4)]), cat([make(8, f) for f in range(4)])
    recs = [dict(c=int(f) * 5 + j % 3) for j, f in enumerate(test[4])]
    a = argparse.Namespace(epochs=25, seeds=1, batch=32)
    res = evaluate(a, train, test, recs, "cpu", log=lambda s: None)
    sn = np.array([[0.9, 0.2, 0.7, 0.1]])
    out = dict(level_of_area_takes_top_scores=bool((level_of_area(sn, np.array([2])) == np.array([[True, False, True, False]])).all()),
               quality_head_scores_every_cut=bool(np.isfinite(res["arms"]["relations"]["quality"]["patch_miou"])),
               relation_head_beats_native=bool(res["arms"]["relations"]["dense"]["gain"] > 5),
               relation_head_beats_score_only=bool(res["arms"]["relations"]["dense_over_score_only"]["gain"] > 3),
               size_head_reduces_area_error=bool(res["arms"]["relations"]["size"]["log_error_std"] < res["foris_own_area"]["log_error_std"]),
               verdict_reads_it=bool(res["verdict"]["passes"]))
    out.update(checks=len(out), passed=int(sum(out.values())), state="PASSED" if all(out.values()) else "FAILED",
               gains={k: round(v["dense"]["gain"], 2) for k, v in res["arms"].items()})
    print(json.dumps(out, indent=1))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache")
    p.add_argument("--run")
    p.add_argument("--out")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args()
    if a.selfcheck:
        out = selfcheck()
        if a.out:
            Path(a.out).write_text(json.dumps(out, indent=1))
        raise SystemExit(0 if out["state"] == "PASSED" else 1)
    main_run(a)


if __name__ == "__main__":
    main()
