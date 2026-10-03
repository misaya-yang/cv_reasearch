#!/usr/bin/env python3
"""CPU checks of tics/extent_cut.py: constructed cases with a known answer, and a replay of the saved FoRIS traces.

  python scripts/extent_selfcheck_cpu.py [--traces results/native_membership_v1/causal_v3/experiment] [--out FILE]

The end-to-end check of the runner is `scripts/extent_experiment.py --fixture DIR`.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tics.extent_cut import (LEVELS, binarise, choose, edge_signature, level_sets, level_statistics,  # noqa: E402
                             neighbour_affinity, normalise, paste, to_original, zoom_box)


def constructed():
    """A core object with a score shoulder around it. The midpoint cut takes the shoulder; the object ends at
    the core. Query features: object / shoulder / far background, shoulder and background alike."""
    h = w = 24
    g = torch.Generator().manual_seed(0)
    basis = torch.linalg.qr(torch.randn(16, 16, generator=g))[0]
    obj, sh, bg = basis[0], F.normalize(basis[1] + 0.05 * basis[3], dim=0), F.normalize(basis[1] + 0.3 * basis[2], dim=0)
    core = torch.zeros(h, w, dtype=torch.bool)
    core[9:15, 9:15] = True
    shoulder = torch.zeros(h, w, dtype=torch.bool)
    shoulder[5:19, 5:19] = True
    shoulder &= ~core
    score = torch.full((h, w), 0.1)
    score[shoulder] = 0.6
    score[core] = 1.0
    score += 0.01 * torch.rand(h, w, generator=g)
    fq = bg.expand(h, w, 16).clone()
    fq[shoulder] = sh
    fq[core] = obj
    # Reference: the object fills the left part, next to the same shoulder material, then background.
    cov = torch.zeros(h, w)
    cov[:, :8] = 1.0
    fs = bg.expand(h, w, 16).clone()
    fs[:, 8:14] = sh
    fs[:, :8] = obj
    sn = normalise(score)
    aff_r, aff_d = neighbour_affinity(fq)
    sim = fq.flatten(0, 1) @ fs.flatten(0, 1).T
    sig = edge_signature(fq, fs, cov)
    st = level_statistics(sn, aff_r, aff_d, sim.argmax(0), cov, sig)
    R = level_sets(sn)
    out = {}
    mid = sn > 0.5
    out["midpoint_takes_shoulder"] = bool((mid & shoulder).sum() == shoulder.sum())
    for rule in ("boundary", "round_trip", "signature"):
        k = choose(st, rule)
        out[rule + "_is_core"] = bool(k is not None and torch.equal(R[k], core))
    # The score alone cannot tell: its largest drop is at the shoulder's outer edge, by construction.
    k = choose(st, "contrast")
    out["contrast_follows_score_only"] = bool(k is not None and torch.equal(R[k], core | shoulder))
    k = choose(st, "boundary_inverted")
    out["inverted_is_not_core"] = bool(k is not None and not torch.equal(R[k], core))
    # No valid level: constant score, nothing to decide.
    flat = level_statistics(torch.zeros(h, w), aff_r, aff_d, sim.argmax(0), cov, sig)
    out["constant_score_declines"] = choose(flat, "boundary") is None
    # A reference mask with no boundary gives no signature and the rule declines.
    none = level_statistics(sn, aff_r, aff_d, sim.argmax(0), torch.ones(h, w), edge_signature(fq, fs, torch.ones(h, w)))
    out["no_boundary_declines"] = choose(none, "signature") is None
    # The exposed threshold reproduces the source binarisation at 0.5.
    s = score - score.min()
    src = F.interpolate((s / s.max().clamp_min(1e-6))[None, None], size=(96, 96), mode="bilinear", align_corners=False)[0, 0] > 0.5
    out["binarise_matches_source_formula"] = bool(torch.equal(binarise(score, 0.5, (96, 96)), src))
    return out


def geometry():
    out = {}
    m = torch.zeros(1024, 1024, dtype=torch.bool)
    m[400:500, 300:420] = True
    box = zoom_box(m)
    out["box_square_inside_frame"] = box is not None and box[2] - box[0] == box[3] - box[1] == 256 and min(box) >= 0 and max(box) <= 1024
    out["box_contains_mask"] = box[0] <= 300 and box[1] <= 400 and box[2] >= 420 and box[3] >= 500
    big = torch.zeros(1024, 1024, dtype=torch.bool)
    big[100:700, 100:700] = True
    out["large_target_not_zoomed"] = zoom_box(big) is None
    out["empty_not_zoomed"] = zoom_box(torch.zeros(1024, 1024, dtype=torch.bool)) is None
    corner = torch.zeros(1024, 1024, dtype=torch.bool)
    corner[0:40, 990:1024] = True
    cb = zoom_box(corner)
    out["corner_box_clamped"] = cb == (768, 0, 1024, 256)
    obox, mbox = to_original(box, (1024, 1024), (640, 427))
    out["original_box_valid"] = 0 <= obox[0] < obox[2] <= 640 and 0 <= obox[1] < obox[3] <= 427
    # Paste back what a perfect crop prediction would be and compare with the mask.
    crop = F.interpolate(m[mbox[1]:mbox[3], mbox[0]:mbox[2]][None, None].float(), size=(1024, 1024), mode="nearest")[0, 0] > 0.5
    back = paste(crop, mbox, (1024, 1024))
    out["paste_roundtrip_iou"] = float((back & m).sum() / (back | m).sum())
    out["paste_roundtrip_ok"] = out["paste_roundtrip_iou"] > 0.97
    return out


def replay(traces):
    """Saved complete-FoRIS traces: the cut replica against the stored mask, and the score-only rule's level."""
    files = sorted(glob.glob(os.path.join(traces, "trace_*.npz")))
    agree, own, con, orc = [], [], [], []
    for f in files:
        z = np.load(f)
        score = torch.from_numpy(z["part4_score"]).float()
        pre = torch.from_numpy(z["pre_refinement"])
        agree.append(float((binarise(score, 0.5, tuple(pre.shape)) == pre).float().mean()))
        t64 = torch.from_numpy(z["truth_model"]).view(64, 16, 64, 16).sum((1, 3)).float()
        tsum = t64.sum()
        sn = normalise(score)
        R = level_sets(sn)
        iou = lambda m: float(t64[m].sum() / (tsum + (256 - t64[m]).sum())) if m.any() else 0.0
        zero = torch.zeros(64, 64)
        st = level_statistics(sn, zero[:, :-1], zero[:-1], torch.zeros(4096, dtype=torch.long), zero, None)
        k = choose(st, "contrast")
        own.append(iou(sn > 0.5))
        con.append(iou(R[k]) if k is not None else own[-1])
        orc.append(max(iou(R[j]) for j in range(len(LEVELS))))
    return dict(traces=len(files), cut_replica_pixel_agreement_min=min(agree), cut_replica_pixel_agreement_mean=float(np.mean(agree)),
                patch_level_midpoint=100 * float(np.mean(own)), patch_level_contrast=100 * float(np.mean(con)),
                patch_level_oracle_on_levels=100 * float(np.mean(orc)),
                note="CPU bilinear against stored GPU masks; exploratory development cohort, not a method score")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", default=str(Path(__file__).resolve().parent.parent / "results/native_membership_v1/causal_v3/experiment"))
    p.add_argument("--out")
    a = p.parse_args()
    torch.set_num_threads(1)  # the no-GPU server mode has half a core; the default thread count stalls there
    out = dict(constructed=constructed(), geometry=geometry())
    if glob.glob(os.path.join(a.traces, "trace_*.npz")):
        out["replay"] = replay(a.traces)
        out["replay_ok"] = out["replay"]["cut_replica_pixel_agreement_min"] > 0.9995
    checks = [v for part in (out["constructed"], out["geometry"]) for v in part.values() if isinstance(v, bool)]
    checks.append(out.get("replay_ok", True))
    out.update(checks=len(checks), passed=int(sum(checks)), state="PASSED" if all(checks) else "FAILED")
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    raise SystemExit(0 if out["state"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
