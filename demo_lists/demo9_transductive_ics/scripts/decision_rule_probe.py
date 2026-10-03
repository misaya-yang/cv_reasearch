#!/usr/bin/env python3
"""What did the linear read-out learn, and does one fixed formula work? CPU only, cached evidence, saved read-outs.

  CUDA_VISIBLE_DEVICES="" python scripts/decision_rule_probe.py --cache cache/decision_v1 --fit results/decision_v1/fit --out results/decision_v1/rule_probe.json

The `pixel` read-out is logit = sum_k a_k * map_k + c over the 16 class-free relation maps. Printed: the
coefficients of the four fold models in units of one standard deviation of each map, how alike the four are, and
the patch-level gain over FoRIS on development and confirmation of (1) each fold scored by its own held-out model,
(2) ONE formula for all folds, the average of the four (its constants have then seen every class once: a rule with
fixed constants, not a class-held-out estimate), (3) the same formula keeping only its largest terms.
Confirmation labels were opened before for another arm; these are later readings and are reported as such.
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
from decision_fit import RELATIONS, table  # noqa: E402


def light(folder):
    """Only the 16 relation maps and the targets, one archive at a time (the machine may have 2 GB and half a core)."""
    import torch.nn.functional as F
    from PIL import Image
    files = sorted(folder.glob("*.npz"))
    x, tf = np.zeros((len(files), 16, 64, 64), np.float16), np.zeros((len(files), 64, 64), np.float32)
    for i, f in enumerate(files):
        with np.load(f, allow_pickle=False) as z:
            m = z["maps"].astype(np.float32)
            er, ed = m[14].copy(), m[15].copy()
            m[14] = (er + np.pad(er, ((0, 0), (1, 0)))[:, :-1]) / 2
            m[15] = (ed + np.pad(ed, ((1, 0), (0, 0)))[:-1]) / 2
            x[i] = m
            if "tf" in z.files:
                tf[i] = z["tf"]
            else:  # confirmation: the label file, reduced to the patch grid exactly as the cache does it
                shape, c = tuple(int(v) for v in z["model_shape"]), int(f.stem.split("_")[2])
                truth = torch.from_numpy((np.asarray(Image.open(str(z["query_mask_path"]))) == c + 1).copy())
                tf[i] = F.avg_pool2d(F.interpolate(truth[None, None].float(), shape, mode="nearest"), shape[0] // 64)[0, 0].numpy()
    return dict(x=x, tf=tf, fold=np.array([int(f.stem.split("_")[0]) for f in files]), cls=np.array([int(f.stem.split("_")[2]) for f in files]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--export", type=Path, help="write the one formula as a models file for scripts/decision_infer.py, then stop")
    a = p.parse_args()
    saved = torch.load(a.fit / "models" / "pixel_relations.pt", map_location="cpu", weights_only=False)["models"]
    std, raw, const = [], [], []
    for f in range(4):
        e = saved[f]
        mu, sd = e["mu"].flatten().numpy(), e["sd"].flatten().numpy()
        w = np.mean([s["body.weight"].flatten().numpy() for s in e["states"]], 0)
        b = np.mean([float(s["body.bias"]) + float(s["shift"]) for s in e["states"]])
        scale = np.mean([float(s["scale"]) for s in e["states"]])
        coef = w / sd
        coef[0] += scale  # the host's score enters twice: through its own logit and through the fusion
        std.append(coef * sd)
        raw.append(coef)
        const.append(b - 0.5 * scale - float((w * mu / sd).sum()))
    std, raw, const = np.array(std), np.array(raw), np.array(const)
    unit = std / np.linalg.norm(std, axis=1, keepdims=True)
    print("coefficients per standard deviation of each map (rows: the models that score folds 0-3; last row: their mean)")
    print(" " * 6 + " ".join("%10s" % k for k in RELATIONS))
    for f in range(4):
        print("fold %d " % f + " ".join("%+10.2f" % v for v in std[f]))
    print("mean   " + " ".join("%+10.2f" % v for v in std.mean(0)))
    alike = float(np.mean([unit[i] @ unit[j] for i in range(4) for j in range(i)]))
    print("cosine between the four coefficient vectors, mean over pairs: %.3f" % alike)
    out = dict(state="ANALYSED", maps=list(RELATIONS), per_sd=std.tolist(), raw=raw.tolist(), constant=const.tolist(), cosine=alike, rows={})
    if a.export:  # the same 17 constants for every fold, in the format of a fitted `pixel` read-out (no standardisation, no host logit)
        from tics.decision_heads import restore
        state = {"scale": torch.zeros(()), "shift": torch.zeros(()), "body.weight": torch.tensor(raw.mean(0), dtype=torch.float32).view(1, 16, 1, 1),
                 "body.bias": torch.tensor([const.mean()], dtype=torch.float32)}
        one = dict(rung="pixel", channels=16, host=0, width=48, steps=2, mu=torch.zeros(1, 16, 1, 1), sd=torch.ones(1, 16, 1, 1), states=[state])
        x = torch.rand(3, 16, 64, 64)
        want = torch.einsum("nkhw,k->nhw", x, torch.tensor(raw.mean(0), dtype=torch.float32)) + float(const.mean())
        differ = float((torch.logit(restore(one)(x, None).clamp(1e-6, 1 - 1e-6)) - want).abs().max())
        if differ > 1e-3:
            raise SystemExit("the exported read-out does not reproduce the formula: %.5f" % differ)
        torch.save(dict(arm="formula:relations", channels="relations", models={f: one for f in range(4)}), a.export)
        print("exported %s; largest difference from the formula on random maps %.6f" % (a.export, differ))
        return
    torch.set_num_threads(1)
    dv, cf = light(a.cache / "test"), light(a.cache / "confirm")
    order = np.argsort(-np.abs(std.mean(0)))
    for name, d in (("development", dv), ("confirmation", cf)):
        x = d["x"][:, :16].astype(np.float32)
        host = x[:, 0] > 0.5
        row = table(d["tf"], d["cls"], d["fold"])
        logit = lambda coef, c: np.einsum("nkhw,k->nhw", x, coef.astype(np.float32)) + c
        held = np.zeros(host.shape, bool)
        for f in range(4):
            held[d["fold"] == f] = logit(raw[f], const[f])[d["fold"] == f] > 0
        rows = {"each fold by its held-out model": row(held, host), "one formula (mean of the four)": row(logit(raw.mean(0), const.mean()) > 0, host)}
        for m in (2, 3, 5, 8):
            keep = np.zeros(16)
            keep[order[:m]] = 1
            mu = np.mean([saved[f]["mu"].flatten().numpy() for f in range(4)], 0)
            c = const.mean() + float((raw.mean(0) * (1 - keep) * mu).sum())  # dropped maps are held at their mean
            rows["one formula, largest %d terms (%s)" % (m, ", ".join(RELATIONS[k] for k in order[:m]))] = row(logit(raw.mean(0) * keep, c) > 0, host)
        out["rows"][name] = rows
        print("\n%s, %d episodes, patch level, gain over FoRIS" % (name, len(host)))
        for k, v in rows.items():
            print("  %-95s %+6.2f [%+.2f, %+.2f]  folds %s" % (k, v["over_foris"], *v["ci95"], " ".join("%+.1f" % t for t in v["per_fold"])))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
