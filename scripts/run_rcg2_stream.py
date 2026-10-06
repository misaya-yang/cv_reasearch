#!/usr/bin/env python3
"""The frozen combined version on a new group of episodes, streamed: nothing but masks' counts and 64 x 64 fields is kept (GPU).

Per episode: complete public FoRIS (the comparator) -> RCG on its final score and matched features (the delivered solve,
dense parts on the CPU) -> shifted-grid feature-guided readout -> cut level by the size of the mask at 0.5. Every setting
is read from the frozen file declared before this group was opened. Truth is read only to count, after the masks exist.

  python scripts/run_rcg2_stream.py --episodes .../train_episodes.json --start 150 --count 150 --frozen outputs/claude_rcg2_frozen.json --out outputs/claude_rcg2_groupB
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LEVELS = np.round(np.arange(0.30, 0.7001, 0.0125), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, required=True); p.add_argument("--start", type=int, required=True); p.add_argument("--count", type=int, required=True)
    p.add_argument("--frozen", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--limit", type=int)
    p.add_argument("--host-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent")); p.add_argument("--demo4-root", default="/root/autodl-tmp/demo4")
    a = p.parse_args()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(a.host_root)); sys.path.insert(0, str(a.host_root / "scripts"))
    from extent_experiment import build_host, run_foris
    from run_rcg2 import solve
    man = json.loads(a.episodes.read_text()); fz = json.loads(a.frozen.read_text()); rows = []
    for f in sorted({e["fold"] for e in man["episodes"]}):
        rows += [dict(e, key="%d_%d_%d" % (e["fold"], e["e"], e["c"])) for e in man["episodes"] if e["fold"] == f][a.start:a.start + a.count]
    rows = rows[:a.limit]; data, ann = Path(man["data_root"]), Path(man["annotation_root"]); dev = torch.device("cuda")
    sigma, tau, W = fz["readout"]["sigma"], fz["readout"]["tau"], fz["readout"]["window"]; edges, levels = np.array(fz["size_cut"]["mask_area_at_half_edges"]), fz["size_cut"]["levels"]
    a.out.mkdir(parents=True, exist_ok=False); lv = torch.from_numpy(LEVELS).float().to(dev)[:, None, None]
    arms = ("native", "rcg", "rcg_fine", "rcg2"); iu = {k: np.zeros((len(rows), 2), np.int64) for k in arms}; iuo = {k: np.zeros((len(rows), 2), np.int64) for k in arms}
    Il, Ul = (np.zeros((len(rows), len(LEVELS)), np.int64) for _ in range(2)); fields = np.zeros((len(rows), 64, 64), np.float16); picked = np.zeros(len(rows), np.float32); T = np.zeros(len(rows), np.int64)
    fi = torch.arange(128, device=dev); base = ((fi * 8 + 4 - 8).float() / 16).round().long(); off = torch.arange(-W, W + 1, device=dev)
    ti = base[:, None] + off[None]; dist = ((ti * 16 + 8) - (fi * 8 + 4)[:, None]).float() / 16; ok1 = (ti >= 0) & (ti < 64); ti = ti.clamp(0, 63); K = len(off)
    nb = (ti[:, None, :, None].expand(128, 128, K, K) * 64 + ti[None, :, None, :].expand(128, 128, K, K)).reshape(-1, K * K)
    ks = torch.exp(-(dist[:, None, :, None] ** 2 + dist[None, :, None, :] ** 2).reshape(-1, K * K) / (2 * sigma ** 2)) * (ok1[:, None, :, None] & ok1[None, :, None, :]).reshape(-1, K * K)
    begin = time.monotonic()
    with torch.inference_mode():
        host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), man, "cuda"); torch.set_num_threads(8)
        for n, row in enumerate(rows):
            sp, qp = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == row["c"] + 1).copy())
            native, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
            deb = got["deb"][0]; q, r = F.normalize(deb[-1].flatten(1).T.float(), dim=1), F.normalize(deb[0].flatten(1).T.float(), dim=1)
            score = got["score"].float(); cov = F.interpolate(ref_mask[None, None].float(), tuple(score.shape), mode="area")[0, 0].cpu().numpy()
            # the cached runs matched in float16 features; the same rounding is applied here
            z = solve(q.half().float().cpu(), r.half().float().cpu(), cov, score.cpu().numpy().astype(np.float32), 16., "cpu"); fields[n] = z; zt = torch.from_numpy(z).to(dev)
            pad = F.pad(tgt[None], (4, 4, 4, 4), mode="reflect")[0]; fine = torch.zeros(128, 128, q.shape[1], device=dev); debiased = bool((F.normalize(got["raw"][0], dim=1) - deb).abs().max() > 1e-4)
            for ay, ax in itertools.product((0, 1), (0, 1)):
                sy, sx = (-4, 4)[ay], (-4, 4)[ax]
                f = F.normalize(host._extract_features(pad[None, None, :, 4 + sy:4 + sy + 1024, 4 + sx:4 + sx + 1024]), p=2, dim=2)
                if debiased:
                    f = host._debias_features(f)
                fine[ay::2, ax::2] = F.normalize(f[0, 0], dim=0).permute(1, 2, 0)
            w = ks * torch.exp((torch.einsum("pc,pkc->pk", fine.reshape(128 * 128, -1), q.half().float()[nb]) - 1) / tau)
            zf = ((w * zt.flatten()[nb]).sum(1) / w.sum(1).clamp_min(1e-12)).reshape(1, 1, 128, 128); up = F.interpolate(zf, (1024, 1024), mode="bilinear", align_corners=False)[0, 0]
            m_fine = up > .5; level = levels[int(np.searchsorted(edges, float(m_fine.float().mean()), side="right") - 1)]; picked[n] = level
            masks = dict(native=native, rcg=F.interpolate(zt[None, None], (1024, 1024), mode="bilinear", align_corners=False)[0, 0] > .5, rcg_fine=m_fine, rcg2=up > level)
            # counting only from here on
            t_o = torch.from_numpy((np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == row["c"] + 1).copy()).to(dev)
            truth = F.interpolate(t_o[None, None].float(), (1024, 1024), mode="nearest")[0, 0].bool(); T[n] = int(truth.sum())
            for k, m in masks.items():
                iu[k][n] = int((m & truth).sum()), int((m | truth).sum())
                o = F.interpolate(m[None, None].float(), tuple(t_o.shape), mode="bilinear", align_corners=False)[0, 0] > .5; iuo[k][n] = int((o & t_o).sum()), int((o | t_o).sum())
            ml = up[None] > lv; Il[n] = (ml & truth).flatten(1).sum(1).cpu().numpy(); Ul[n] = (ml | truth).flatten(1).sum(1).cpu().numpy()
            if (n + 1) % 50 == 0:
                print(json.dumps(dict(n=n + 1, total=len(rows), seconds=round(time.monotonic() - begin, 1))), flush=True)
    (a.out / "rows.json").write_text(json.dumps([dict(key=r["key"], c=r["c"], fold=r["fold"], support=r["support"], query=r["query"]) for r in rows]) + "\n")
    np.savez_compressed(a.out / "counts.npz", levels=LEVELS, truth=T, picked=picked, fields=fields, I_fine=Il, U_fine=Ul, **{"iu:" + k: v for k, v in iu.items()}, **{"orig:" + k: v for k, v in iuo.items()})
    print("STREAM_DONE", flush=True)


if __name__ == "__main__":
    main()
