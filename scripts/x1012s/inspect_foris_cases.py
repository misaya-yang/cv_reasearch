"""Inspect three existing FoRIS failures; no encoder, fitting or candidate inference.

Cases were selected visually from existing v01 sheets. Reconstruct the official
pre-CRF readout, compare the saved final output, and compute a tied-score-respecting
GT oracle only to locate the loss. All numbers are selected-case diagnostics.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/x1012"))
import lab

OUT = ROOT.parent / "cv_data/a/x1012s_case_review"
CASES = ("dev_s1/coco/0/85", "dev_s1/coco/1/89", "isic/-1/34")


def metrics(mask, gt):
    tp = int(np.count_nonzero(mask & gt))
    fp = int(np.count_nonzero(mask & ~gt))
    fn = int(np.count_nonzero(~mask & gt))
    return dict(tp=tp, fp=fp, fn=fn, area=int(mask.sum()),
                iou=tp / max(tp + fp + fn, 1))


def strict_oracle(score, gt):
    order = np.argsort(-score.ravel(), kind="stable")
    s = score.ravel()[order]
    tp = np.cumsum(gt.ravel()[order], dtype=np.int64)
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    iou = tp[ends] / np.maximum(ends + 1 + gt.sum() - tp[ends], 1)
    j = ends[int(np.argmax(iou))]
    threshold = float(s[j])
    mask = score >= threshold
    m = metrics(mask, gt)
    assert np.isclose(m["iou"], iou.max(), atol=1e-12)
    return mask, threshold, m


def overlay(rgb, mask, color):
    x = rgb.copy().astype(float)
    x[mask] = .55 * x[mask] + .45 * np.array(color)
    return x.astype(np.uint8)


def main():
    torch.set_num_threads(2)
    lab.DEV = torch.device("cpu")
    identity = json.loads((OUT / "v01_identity_map.json").read_text())
    bindings = {e["episode_id"]: e for e in identity["entries"]}
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    rows = []
    for k, task in enumerate(CASES):
        b = bindings[task]
        assert b["row"].get("query_crop") is None
        assert b["row"].get("reference_crop") is None
        e = b["score64"]
        with np.load(e["path"], allow_pickle=False) as z:
            s = torch.from_numpy(z[e["key"]].reshape(64, 64).astype(np.float32))
        z64 = (s - s.min()) / (s.max() - s.min()).clamp_min(1e-6)
        z1024 = lab.up(z64).numpy()
        raw1024 = lab.up(s).numpy()
        pre = z1024 > .5
        final = lab.foris_cli(b).numpy()
        gt = lab.mask1024(b["row"]["query_mask_path"]).numpy()
        oracle, threshold, om = strict_oracle(raw1024, gt)
        p = torch.from_numpy(pre.astype(np.float32))[None, None]
        yy, xx = torch.meshgrid(torch.arange(-5, 6), torch.arange(-5, 6), indexing="ij")
        kernel = (xx.square() + yy.square() <= 25).float()[None, None]
        counts = F.conv2d(p, kernel, padding=5)[0, 0].numpy()
        dilated = counts > 0
        eroded = counts == float(kernel.sum())
        band = dilated & ~eroded
        source_paths = [e["path"], b["old_arms"]["B"]["path"],
                        b["row"]["query_mask_path"], b["row"]["reference_mask_path"]]
        row = dict(episode_id=task, index=b["index"], dataset=b["dataset"],
                   score_min=float(s.min()), score_max=float(s.max()),
                   raw_midpoint=float((s.min()+s.max())/2),
                   strict_oracle_threshold=threshold,
                   strict_oracle_threshold_normalized=(threshold-float(s.min())) /
                       max(float(s.max()-s.min()), 1e-6),
                   pre=metrics(pre, gt), final=metrics(final, gt), strict_oracle=om,
                   final_pre_xor=int(np.count_nonzero(final != pre)),
                   final_pre_xor_outside_band=int(np.count_nonzero((final != pre) & ~band)),
                   oracle_pre_xor=int(np.count_nonzero(oracle != pre)),
                   oracle_pre_xor_outside_band=int(np.count_nonzero((oracle != pre) & ~band)),
                   input_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths})
        rows.append(row)
        rgb = np.array(Image.open(b["query_rgb_absolute"]).convert("RGB").resize((512,512)))
        ref = np.array(Image.open(b["reference_rgb_absolute"]).convert("RGB").resize((512,512)))
        rm = lab.mask1024(b["row"]["reference_mask_path"]).numpy()[::2,::2]
        panels = [(overlay(ref, rm, (0,255,0)), task+"\nReference"),
                  (overlay(rgb, gt[::2,::2], (255,0,0)), "Query GT (diagnostic)"),
                  (z1024[::2,::2], "Existing score, minmax"),
                  (overlay(rgb, pre[::2,::2], (0,80,255)), f"Reconstructed pre-CRF {row['pre']['iou']:.3f}"),
                  (overlay(rgb, final[::2,::2], (0,80,255)), f"Saved full FoRIS {row['final']['iou']:.3f}"),
                  (overlay(rgb, oracle[::2,::2], (255,220,0)), f"GT threshold oracle {om['iou']:.3f}\nt={row['strict_oracle_threshold_normalized']:.3f}")]
        for ax, (im, title) in zip(axes[k], panels):
            ax.imshow(im, cmap="viridis", vmin=0 if im.ndim==2 else None,
                      vmax=1 if im.ndim==2 else None)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "foris_three_cases.png", dpi=120)
    plt.close(fig)
    result = dict(scope="Three visually selected existing failures, not a benchmark or a new method",
                  new_encoder_calls=0, new_candidate_inference=False,
                  source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  oracle_rule="Include complete equal-score groups; >= threshold; query GT diagnostic only",
                  rows=rows)
    (OUT / "foris_three_cases.json").write_text(json.dumps(result, indent=2)+"\n")
    for r in rows:
        print(json.dumps({k:v for k,v in r.items() if k!='input_sha256'}))


if __name__ == "__main__":
    main()
