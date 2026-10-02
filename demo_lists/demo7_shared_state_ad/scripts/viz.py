#!/usr/bin/env python3
"""contact sheet: image | ground truth | score map (bank given by --map) for some bad test images of one category"""
import argparse, os, sys
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from adx.common import CACHE
ap = argparse.ArgumentParser()
ap.add_argument("--cat", default="vial"); ap.add_argument("--tag", default=""); ap.add_argument("--map", default="dB")
ap.add_argument("--cond", default="regular"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--start", type=int, default=0)
ap.add_argument("--out", default="/root/demo7_cache/results/viz.jpg"); ap.add_argument("--w", type=int, default=260)
a = ap.parse_args()
R = torch.load(f"{CACHE}/results/{a.cat}{a.tag}.pt", weights_only=False)
meta, gt = R["meta"], R["gt"].numpy(); size = gt.shape[1:]
s = F.interpolate(R[a.map].mean(1)[:, None].float(), size=size, mode="bilinear", align_corners=False)[:, 0].numpy()
calib = np.array([m["kind"] in ("thr", "val") for m in meta])
lo, hi = np.percentile(s[calib], 50), np.percentile(s[calib], 95) * 1.421 * 1.6
idx = [i for i, m in enumerate(meta) if m["kind"] == "bad" and m["cond"] == a.cond][a.start:a.start + a.n]
h = int(a.w * size[0] / size[1]); rows = []
for i in idx:
    im = np.asarray(Image.open(meta[i]["path"]).convert("RGB").resize((a.w, h)))
    g = np.asarray(Image.fromarray((gt[i] * 255).astype(np.uint8)).resize((a.w, h)))
    v = np.clip((s[i] - lo) / (hi - lo), 0, 1)
    v = np.asarray(Image.fromarray((v * 255).astype(np.uint8)).resize((a.w, h)))
    heat = np.stack([v, (v * 0.6).astype(np.uint8), 255 - v], -1)
    over = im.copy(); over[g > 127] = (0.5 * over[g > 127] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    rows.append(np.concatenate([im, over, heat], 0))
Image.fromarray(np.concatenate(rows, 1)).save(a.out, quality=85)
print("saved", a.out, [meta[i]["scene"] for i in idx])
