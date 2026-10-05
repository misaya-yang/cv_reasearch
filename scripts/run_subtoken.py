#!/usr/bin/env python3
"""Sub-token boundary resolution for a sealed 64 x 64 field (GPU).

The RCG mask is the bilinear upsampling of a 64 x 64 field, so its boundary is blurred over half a token. This run
re-encodes the query four times, shifted by (+-4, +-4) pixels, which gives one frozen-encoder feature per 8 x 8 cell
(128 x 128), and upsamples the sealed field with those features as the guide:

  z_fine(p) = sum_q ks(p, q) kf(f_p, f_q) z(q) / sum_q ks kf     q: the 64 x 64 tokens within WINDOW of p

  feat[s,t]      guide = shifted-grid features against the cached token features, kf = exp((cos - 1) / t)
  rgb[s,c]       control: guide = mean colour of the cell against mean colour of the token (no extra encoder pass)
  rcg_crf        control: the FoRIS finish (band CRF on colour) applied to the RCG mask
  *+crf          the same finish after the guided upsampling

Inference reads no query truth; truth is opened only to count. Selection is nested over folds.

  python scripts/run_subtoken.py --root outputs/fresh600_root --run outputs/recheck_fresh600_v1 --out outputs/claude_subtoken_fresh600
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SIGMAS, TAUS, COLOURS, WINDOW = (0.75, 1.25), (0.03, 0.07, 0.15), (0.03, 0.07, 0.15), 2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--run", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--host-manifest", type=Path, default=Path("outputs/claude_official/batch0.json"))
    p.add_argument("--host-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent")); p.add_argument("--demo4-root", default="/root/autodl-tmp/demo4")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ics.experiment import packet, photo_groups, summarize, unpack
    sys.path.insert(0, str(a.host_root)); sys.path.insert(0, str(a.host_root / "scripts"))
    from extent_experiment import build_host
    man = json.loads(a.host_manifest.read_text()); rows = json.loads((a.run / "manifest.json").read_text())[:a.limit]
    data = Path(man["data_root"]); dev = torch.device("cuda")
    a.out.mkdir(parents=True, exist_ok=False); (a.out / "predictions").mkdir()
    variants = [("feat", s, t) for s in SIGMAS for t in TAUS] + [("rgb", s, c) for s in SIGMAS for c in COLOURS]
    vname = lambda v: "%s[s%g,%s%g]" % (v[0], v[1], "t" if v[0] == "feat" else "c", v[2])
    arms = ["native", "rcg", "rcg_crf"] + [vname(v) + x for v in variants for x in ("", "+crf")]
    iu = {k: np.zeros((len(rows), 2), np.int64) for k in arms}
    band = {k: np.zeros((len(rows), 2), np.int64) for k in arms}                # errors within 8 px of the true boundary: false, missed
    # geometry of the guided upsampling: fine cell (I, J) has centre (8I+4, 8J+4); token (i, j) has centre (16i+8, 16j+8)
    fi = torch.arange(128, device=dev); base = ((fi * 8 + 4 - 8).float() / 16).round().long()        # nearest token index along one axis
    off = torch.arange(-WINDOW, WINDOW + 1, device=dev)
    ti = base[:, None] + off[None]                                                                   # [128, K]
    dist = ((ti * 16 + 8) - (fi * 8 + 4)[:, None]).float() / 16; ok1 = (ti >= 0) & (ti < 64); ti = ti.clamp(0, 63)
    K = len(off)
    ny = ti[:, None, :, None].expand(128, 128, K, K); nx = ti[None, :, None, :].expand(128, 128, K, K)
    nb = (ny * 64 + nx).reshape(128 * 128, K * K)                                                    # token neighbours of each cell
    d2 = (dist[:, None, :, None] ** 2 + dist[None, :, None, :] ** 2).reshape(128 * 128, K * K)
    okn = (ok1[:, None, :, None] & ok1[None, :, None, :]).reshape(128 * 128, K * K)
    begin = time.monotonic()
    with torch.inference_mode():
        host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), man, "cuda")
        from utils.data import denormalize                                   # FoRIS source, on the path after build_host
        ring = torch.ones(1, 1, 17, 17, device=dev)
        for n, row in enumerate(rows):
            feat = torch.load(a.root / "cache/evidence_v1/feat" / (row["key"] + ".pt"))
            q = F.normalize(feat["q"].to(dev).float(), dim=1)                                        # [4096, C], the tokens RCG read
            with np.load(a.run / "fields" / (row["key"] + ".npz"), allow_pickle=False) as z:
                field = torch.from_numpy(z["rcg"]).to(dev).float()
            with np.load(a.run / "predictions" / (row["key"] + ".npz"), allow_pickle=False) as z:
                rcg = torch.from_numpy(unpack(z["RCG"])).to(dev)
            with np.load(packet(a.root, row), allow_pickle=False) as z:
                native, truth = (torch.from_numpy(unpack(z[k])).to(dev) for k in ("native", "truth"))
            tgt = host._transform(Image.open(data / row["query"]).convert("RGB")).to(dev)            # [3, 1024, 1024]
            pad = F.pad(tgt[None], (4, 4, 4, 4), mode="reflect")[0]
            fine = torch.zeros(128, 128, q.shape[1], device=dev)
            for ay, ax in itertools.product((0, 1), (0, 1)):                                         # shift -4 serves even cells, +4 odd cells
                sy, sx = (-4, 4)[ay], (-4, 4)[ax]
                x = pad[:, 4 + sy:4 + sy + 1024, 4 + sx:4 + sx + 1024]
                f = F.normalize(host._extract_features(x[None, None]), p=2, dim=2)
                if bool(feat["debiased"]):
                    f = host._debias_features(f)
                fine[ay::2, ax::2] = F.normalize(f[0, 0], dim=0).permute(1, 2, 0)
            fine = fine.reshape(128 * 128, -1)
            img = denormalize(tgt).clamp(0, 1)
            c_fine = F.avg_pool2d(img[None], 8)[0].flatten(1).T; c_tok = F.avg_pool2d(img[None], 16)[0].flatten(1).T
            cos = torch.einsum("pc,pkc->pk", fine, q[nb]); dc = (c_fine[:, None] - c_tok[nb]).square().sum(-1)
            zq = field.flatten()[nb]
            masks = dict(native=native, rcg=rcg, rcg_crf=host._finalize_mask(rcg, tgt[None]).reshape(1024, 1024).bool())
            for v in variants:
                w = torch.exp(-d2 / (2 * v[1] ** 2)) * (torch.exp((cos - 1) / v[2]) if v[0] == "feat" else torch.exp(-dc / (2 * v[2] ** 2))) * okn
                zf = ((w * zq).sum(1) / w.sum(1).clamp_min(1e-12)).reshape(1, 1, 128, 128)
                m = F.interpolate(zf, (1024, 1024), mode="bilinear", align_corners=False)[0, 0] > .5
                masks[vname(v)] = m; masks[vname(v) + "+crf"] = host._finalize_mask(m, tgt[None]).reshape(1024, 1024).bool()
            t = truth.float()[None, None]; s = F.conv2d(t, ring, padding=8)[0, 0]; near = (s > 0) & (s < ring.numel())
            for k, m in masks.items():
                iu[k][n] = int((m & truth).sum()), int((m | truth).sum())
                band[k][n] = int((m & ~truth & near).sum()), int((~m & truth & near).sum())
            np.savez_compressed(a.out / "predictions" / (row["key"] + ".npz"), **{k: np.packbits(m.cpu().numpy()) for k, m in masks.items() if k not in ("native", "rcg")})
            if (n + 1) % 25 == 0:
                print(json.dumps(dict(n=n + 1, total=len(rows), seconds=round(time.monotonic() - begin, 1))), flush=True)
    np.savez_compressed(a.out / "counts.npz", **{"iu:" + k: v for k, v in iu.items()}, **{"band:" + k: v for k, v in band.items()})
    classes = np.array([r["c"] for r in rows]); folds = np.array([r["fold"] for r in rows]); groups = photo_groups(rows)

    def miou(x, ix):
        return float(np.mean([x[ix][classes[ix] == c, 0].sum() / max(x[ix][classes[ix] == c, 1].sum(), 1) for c in np.unique(classes[ix])]) * 100)

    def nested(names):
        out, picks = np.zeros((len(rows), 2), np.int64), {}
        for f in np.unique(folds):
            held = folds == f; shared = np.isin(groups, groups[held]); fit = np.flatnonzero(~held & ~shared)
            best = max(names, key=lambda k: miou(iu[k], fit)); out[held] = iu[best][held]; picks[str(int(f))] = best
        return out, picks
    arrays, selection = dict(iu), {}
    arrays["rcg_crf.control"] = arrays.pop("rcg_crf")
    for tag, names in {"feat.nested": [k for k in arms if k.startswith("feat") and "+crf" not in k], "feat+crf.nested": [k for k in arms if k.startswith("feat") and "+crf" in k],
                       "rgb.nested.control": [k for k in arms if k.startswith("rgb") and "+crf" not in k], "rgb+crf.nested.control": [k for k in arms if k.startswith("rgb") and "+crf" in k]}.items():
        arrays[tag], selection[tag] = nested(names)
    corr = {k: [dict(key=r["key"], c=r["c"], fold=r["fold"], batch=str(r.get("batch", "unspecified")), add_TP=0, delete_FP=0, delete_TP=0, add_FP=0) for r in rows] for k in arrays}
    s, _ = summarize(rows, arrays, corr)
    for k in ("corrections_vs_native", "corrections_by_class", "corrections_by_batch"):
        s.pop(k, None)
    s["selection"] = selection; s["band_errors_within_8px"] = {k: dict(false=int(v[:, 0].sum()), missed=int(v[:, 1].sum())) for k, v in band.items()}
    (a.out / "report.json").write_text(json.dumps(s, indent=1, default=float) + "\n")
    order = sorted(arrays, key=lambda k: -s["scores"][k])
    L = ["# Sub-token boundary resolution: %d episodes, class mIoU at 1024" % len(rows), "",
         "| arm | mIoU | vs FoRIS | vs RCG | vs RCG+CRF | false / missed within 8 px (M) |", "|---|---:|---|---|---|---|"]
    g = lambda k, b: "" if b not in s["contrasts"].get(k, {}) else "%+.2f [%+.2f, %+.2f]" % (s["contrasts"][k][b]["gain"], *s["contrasts"][k][b]["ci95"])
    for k in order:
        e = band.get(k if k != "rcg_crf.control" else "rcg_crf")
        L.append("| %s | %.2f | %s | %s | %s | %s |" % (k, s["scores"][k], g(k, "native"), g(k, "rcg"), g(k, "rcg_crf.control"), "" if e is None else "%.2f / %.2f" % (e[:, 0].sum() / 1e6, e[:, 1].sum() / 1e6)))
    L += ["", "Selections: " + json.dumps(selection)]
    (a.out / "report.md").write_text("\n".join(L) + "\n"); print("\n".join(L), flush=True); print("SUBTOKEN_DONE", flush=True)


if __name__ == "__main__":
    main()
