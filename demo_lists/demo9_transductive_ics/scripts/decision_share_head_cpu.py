#!/usr/bin/env python3
"""The share head (PLAN 2026-10-04, method M2), CPU only, streaming: predict how much of the query is target from
class-free statistics, then cut the host's own score field at that quantile.

Bayes says the decision level is -logit(share); the host fixes it (FoRIS: the min-max midpoint; FROST: zero under
equal priors). The true share is worth +9.09 on DEV241 at patch level. Four label-free share estimators from the
score field alone miss the true log-area by about 0.66; this fits a small class-free regression (image statistics of
the evidence maps, no class input) on training episodes and reads it on development and confirmation.

  python scripts/decision_share_head_cpu.py --cache cache/decision_v1 \
      --train-manifest .../extent_head_t1_isolated_v1/train_episodes.json \
      --out results/decision_cpu_20261004/share_head.json --sets test,confirm

Rows (patch level, class mIoU, paired 2000-draw intervals): host native; host cut at the predicted quantile; host cut
at the true share (oracle); host cut at the training-median share (fixed-level control). Fit: ridge on standardised
features, per held-out collection fold, lambda chosen on held-out training classes. One archive is open at a time
(the no-card container has a 2 GB memory cap).
"""
import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ("sn", "s2", "s3", "rank", "fg", "bg", "nn_label", "cycle", "grp_core", "grp_out", "grp_core5", "grp_out5",
         "proto_core", "proto_out", "edge_h", "edge_v")


def files_of(folder, rows=None, limit=None):
    if rows is None:
        out = sorted(Path(folder).glob("*.npz"))
    else:
        out = [Path(folder) / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])) for r in rows]
    return out[:limit] if limit else out


def features_of(path, want_tf=False):
    """Class-free per-episode statistics of one cached episode; returns (feature vector, names)."""
    with np.load(path, allow_pickle=False) as z:
        x = np.concatenate([z["maps"].astype(np.float32), z["layers"].astype(np.float32)]).reshape(-1, 64, 64)
        sn = x[0]
        flat = sn.reshape(-1)
        qs = np.percentile(flat, [10, 25, 50, 75, 90])
        f = [sn.mean(), sn.std(), sn.min(), sn.max(), *qs, (sn > 0.5).mean(), (sn > 0.25).mean(), (sn > 0.75).mean()]
        names = ["sn_mean", "sn_std", "sn_min", "sn_max", "sn_q10", "sn_q25", "sn_q50", "sn_q75", "sn_q90",
                 "host_share50", "host_share25", "host_share75"]
        for idx in (4, 5, 8, 9, 12, 13):  # fg, bg, grp_core, grp_out, proto_core, proto_out
            m = x[idx].reshape(-1)
            f += [m.mean(), m.std(), np.percentile(m, 90)]
            names += ["%s_mean" % NAMES[idx], "%s_std" % NAMES[idx], "%s_q90" % NAMES[idx]]
        for idx in range(16, 24):
            m = x[idx].reshape(-1)
            f += [m.mean(), np.percentile(np.abs(m), 90)]
            names += ["layer%d_mean" % idx, "layer%d_absq90" % idx]
        if "cov" in z.files:
            f.append(float(np.asarray(z["cov"]).mean()))
            names.append("reference_coverage")
        score = sn.reshape(-1).copy()
        tf = z["tf"].astype(np.float32) if (want_tf and "tf" in z.files) else None
    return np.array(f, np.float64), names, score, tf


def ridge_fit(Xtr, ytr, Xva=None, yva=None, lambdas=(0.01, 0.1, 1.0, 10.0)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], 1)
    best = None
    for lam in lambdas:
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ ytr)
        if Xva is None:
            return dict(mu=mu, sd=sd, w=w, lam=lam)
        pred = np.concatenate([(Xva - mu) / sd, np.ones((len(Xva), 1))], 1) @ w
        err = float(np.mean(np.abs(pred - yva)))
        if best is None or err < best[2]:
            best = (dict(mu=mu, sd=sd, w=w, lam=lam), lam, err)
    return best[0]


def ridge_apply(fit, X):
    return np.concatenate([(X - fit["mu"]) / fit["sd"], np.ones((len(X), 1))], 1) @ fit["w"]


def quantile_mask(scores, share):
    n, hw = scores.shape
    k = np.clip(np.round(share * hw).astype(int), 1, hw - 1)
    order = np.argsort(-scores, 1)[:, : k.max()]
    grid = np.arange(k.max())[None, :] < k[:, None]
    out = np.zeros((n, hw), bool)
    np.put_along_axis(out, order, grid, axis=1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sets", default="test,confirm")
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from decision_fit import table

    rows = json.loads(a.train_manifest.read_text())["episodes"]
    tr_x, tr_y, tr_fold, tr_cls, names = [], [], [], [], None
    for row in files_of(a.cache / "train", rows, a.train_limit or None):
        f, names, _, tf = features_of(row, want_tf=True)
        tr_x.append(f)
        tr_y.append(np.log(max(float((tf > 0.5).mean()), 1e-4)))
        parts = row.stem.split("_")
        tr_fold.append(int(parts[0]))
        tr_cls.append(int(parts[2]))
    Xtr, ytr = np.stack(tr_x), np.array(tr_y)
    tr_fold, tr_cls = np.array(tr_fold), np.array(tr_cls)
    rep = dict(state="COMPLETED", features=names, train=int(len(Xtr)), sets={})
    print("training episodes %d, features %d" % (len(Xtr), len(names)), flush=True)
    for name in a.sets.split(","):
        X, tf, fold, cls, score = [], [], [], [], []
        for row in files_of(a.cache / name, None, a.limit or None):
            f, _, sc, t = features_of(row, want_tf=True)
            X.append(f)
            score.append(sc)
            tf.append(t.reshape(-1))
            parts = row.stem.split("_")
            fold.append(int(parts[0]))
            cls.append(int(parts[2]))
        X, score = np.stack(X), np.stack(score)
        tf, fold, cls = np.stack(tf), np.array(fold), np.array(cls)
        share_true = (tf > 0.5).mean(1)
        host = score > 0.5
        pred = np.zeros(len(X))
        for f in range(4):
            tr_rows = np.flatnonzero(tr_fold != f)
            te = np.flatnonzero(fold == f)
            if not len(te):
                continue
            ids = np.unique(tr_cls[tr_rows])
            ids = ids[np.random.default_rng(0).permutation(len(ids))]
            val_ids = ids[: max(1, len(ids) // 5)]
            is_val = np.isin(tr_cls[tr_rows], val_ids)
            fit = ridge_fit(Xtr[tr_rows][~is_val], ytr[tr_rows][~is_val], Xtr[tr_rows][is_val], ytr[tr_rows][is_val])
            pred[te] = ridge_apply(fit, X[te])
        share_hat = np.clip(np.exp(pred), 0.01, 0.95)
        row = table(tf.reshape(-1, 64, 64), cls, fold)
        rows_out = {
            "host_native": row(host.reshape(-1, 64, 64), host.reshape(-1, 64, 64)),
            "host_at_predicted_share": row(quantile_mask(score, share_hat), host.reshape(-1, 64, 64)),
            "host_at_true_share_LABELS": row(quantile_mask(score, share_true), host.reshape(-1, 64, 64)),
            "host_at_median_share_control": row(quantile_mask(score, np.full(len(X), float(np.exp(np.median(ytr))))),
                                                host.reshape(-1, 64, 64)),
        }
        err = np.abs(pred - np.log(np.clip(share_true, 1e-4, 1.0)))
        rep["sets"][name] = dict(
            episodes=int(len(X)), rows=rows_out,
            share=dict(true_mean=float(share_true.mean()), pred_mean=float(share_hat.mean()),
                       log_error_mean=float(err.mean()), log_error_median=float(np.median(err)),
                       frac_within_0p35=float((err < 0.35).mean()),
                       correlation=float(np.corrcoef(pred, np.log(np.clip(share_true, 1e-4, 1)))[0, 1])))
        print("== %s (%d episodes)" % (name, len(X)), flush=True)
        for k, v in rows_out.items():
            print("  %-30s %6.2f  %+6.2f [%+.2f, %+.2f]  folds %s" % (
                k, v["miou"], v["over_foris"], *v["ci95"], " ".join("%+.1f" % z for z in v["per_fold"])), flush=True)
        s = rep["sets"][name]["share"]
        print("  share: true %.3f pred %.3f  log-err mean %.3f median %.3f  frac<0.35 %.2f  corr %.2f" % (
            s["true_mean"], s["pred_mean"], s["log_error_mean"], s["log_error_median"], s["frac_within_0p35"],
            s["correlation"]), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(dict(state="COMPLETED", sets=list(rep["sets"]))))


if __name__ == "__main__":
    main()