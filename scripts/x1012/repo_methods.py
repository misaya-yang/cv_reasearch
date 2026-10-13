"""Exact re-entry into the repository's existing arms from the cached O24 features and FoRIS score.

foris_features : FoRIS Part-1 output ("deb"): unit features, projected off the positional basis iff s_sem < 0.8
mean_field     : the locked MEAN control (src/ics/methods/mean_control.py), same FP16 hand-off as the archive
"""
import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import lab
from ics.methods import mean_control


def foris_features(b):
    r, q = lab.unit(lab.feats(b, "reference")), lab.unit(lab.feats(b, "query"))
    m = lab.mask1024(b["row"]["reference_mask_path"])
    fg = m[::16, ::16].flatten().to(r.device)
    apply = True
    if fg.any():
        tgt = q.mean(0)
        apply = bool(torch.dot(lab.unit(r[fg].mean(0)), tgt) / (tgt.norm() + 1e-6) < 0.8)
    if apply:
        B = lab.basis()
        r, q = lab.unit(r - (r @ B) @ B.T), lab.unit(q - (q @ B) @ B.T)
    return q, r, m, apply


def foris_score(b):
    e = b["score64"]
    return np.load(e["path"])[e["key"]].reshape(64, 64).astype(np.float32)


def mean_field(b, feats=None):
    q, r, m, _ = feats or foris_features(b)
    cov = lab.coverage64(m).numpy()
    return mean_control.predict(q.half().cpu(), r.half().cpu(), cov, foris_score(b))[0]


def field_mask(z):
    return F.interpolate(torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))[None, None], (1024, 1024),
                         mode="bilinear", align_corners=False)[0, 0] > 0.5


if __name__ == "__main__":
    rows = lab.subset(3)
    for b in rows:
        z = mean_field(b)
        e = b["mean64"]
        ref = np.load(e["path"])[e["key"]].reshape(64, 64)
        old = lab.arm_cli(b, "G")
        print(f"{b['dataset']:12s} {b['index']:5d}  max|field diff| {np.abs(z - ref).max():.2e}   mask pixels differing {int((field_mask(z) != old).sum())}")
