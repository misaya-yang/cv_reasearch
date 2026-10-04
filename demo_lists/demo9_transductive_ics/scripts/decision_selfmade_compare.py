#!/usr/bin/env python3
"""D1 verdict: is the constructed-pair fit the same rule as the label fit, and how well does the decision estimate
the query's share? CPU only, on saved models and the decision cache.

  python scripts/decision_selfmade_compare.py --label results/decision_v1/fit/models/pixel_relations.pt \
      --made results/decision_selfmade_v1/fit/models/pixel_relations.pt --dev-cache cache/decision_v1/test \
      --out results/decision_selfmade_v1/compare.json

The linear rung is a 1x1 convolution on standardised maps, so a coefficient is comparable only in raw-map units:
w_raw = w / sd and b_raw = b - sum(w * mu / sd). The cosine of the two raw coefficient vectors is the D1 criterion
(>= 0.9 match). If --dev-cache is given, the share diagnostic reports, per episode, the true foreground share
against the share implied by the host's own mask and by the read-out's mask: mean |log ratio| and correlation.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def raw_coefficients(path, fold=0):
    """The linear rung's coefficients of one fold model, in raw-map units (a comparison space both fits share)."""
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    export = saved["models"][fold] if "models" in saved else saved
    if export.get("rung") != "pixel":
        raise SystemExit("this comparison expects the linear rung, got %s" % export.get("rung"))
    state = export["states"][0]
    w = state["body.weight"].reshape(-1).double()          # [C]
    b = float(state["body.bias"].reshape(-1)[0])
    mu = export["mu"].reshape(-1).double()
    sd = export["sd"].reshape(-1).double()
    return dict(w_raw=(w / sd).numpy(), b_raw=float(b - float((w * mu / sd).sum())),
                w_per_sd=w.numpy(), channels=int(w.shape[0]), arm=saved.get("arm"), fold=fold)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", type=Path, required=True)
    p.add_argument("--made", type=Path, required=True)
    p.add_argument("--dev-cache", type=Path, help="a decision cache folder with targets (the development set)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    lab, made = raw_coefficients(a.label), raw_coefficients(a.made)
    per_fold = []
    for f in range(4):
        try:
            lf, mf = raw_coefficients(a.label, f), raw_coefficients(a.made, f)
        except (KeyError, IndexError):
            continue
        if lf["channels"] != mf["channels"]:
            raise SystemExit("channel counts differ: %d vs %d" % (lf["channels"], mf["channels"]))
        per_fold.append(float(np.dot(lf["w_raw"], mf["w_raw"]) / (np.linalg.norm(lf["w_raw"]) * np.linalg.norm(mf["w_raw"]))))
    if lab["channels"] != made["channels"]:
        raise SystemExit("channel counts differ: %d vs %d" % (lab["channels"], made["channels"]))
    cos = float(np.mean(per_fold)) if per_fold else float(np.dot(lab["w_raw"], made["w_raw"]) /
                                                          (np.linalg.norm(lab["w_raw"]) * np.linalg.norm(made["w_raw"])))
    cos_sd = float(np.dot(lab["w_per_sd"], made["w_per_sd"]) /
                   (np.linalg.norm(lab["w_per_sd"]) * np.linalg.norm(made["w_per_sd"])))
    rep = dict(state="COMPLETED", channels=lab["channels"], cosine_raw=cos, cosine_per_fold=per_fold,
               cosine_per_sd=cos_sd, label=dict(w_raw=lab["w_raw"].tolist(), b_raw=lab["b_raw"]),
               made=dict(w_raw=made["w_raw"].tolist(), b_raw=made["b_raw"]),
               criterion=dict(cosine_min=0.9, dev_gain_min=2.0), verdict=None)
    print("cosine(raw coefficients) mean %.3f  folds %s   cosine(per-sd) %.3f   channels %d" % (
        cos, " ".join("%.3f" % v for v in per_fold), cos_sd, lab["channels"]))
    if a.dev_cache:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from decision_fit import load
        d = load(a.dev_cache)
        n = len(d["fold"]) if not a.limit else a.limit
        keep = slice(0, n)
        tf = d["tf"][keep]
        share_true = (tf > 0.5).reshape(n, -1).mean(1)
        share_host = (d["x"][keep, 0].astype(np.float32) > 0.5).reshape(n, -1).mean(1)
        eps = 1e-6
        err = lambda s: float(np.mean(np.abs(np.log((s + eps) / (share_true + eps)))))
        rep["share"] = dict(episodes=n, true_mean=float(share_true.mean()),
                            host_log_error=err(share_host), host_corr=float(np.corrcoef(share_host, share_true)[0, 1]))
        print("share diagnostic: %d episodes, true %.3f, host log-error %.3f, host corr %.3f" % (
            n, share_true.mean(), rep["share"]["host_log_error"], rep["share"]["host_corr"]))
    rep["verdict"] = "MATCH" if cos >= 0.9 else ("BORDERLINE" if cos >= 0.6 else "MISMATCH")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(dict(state=rep["state"], verdict=rep["verdict"], cosine=cos)))


if __name__ == "__main__":
    main()