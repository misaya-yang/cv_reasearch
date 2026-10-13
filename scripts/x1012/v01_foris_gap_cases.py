"""Contact sheets of the episodes where complete FoRIS is furthest below the GT-optimal level
of its own score field (diagnostic only; query GT used to pick and draw cases)."""
import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 12
rows = lab.subset(300) + lab.ext_subset(300, ("isic", "isaid"))


def img(p):
    q = lab.ASSETS / p
    return np.array(Image.open(q if q.exists() else p).convert("RGB").resize((512, 512)))


def score(b):
    e = b["score64"]
    return torch.from_numpy(np.load(e["path"])[e["key"]].reshape(64, 64).astype(np.float32))


by = {}
for b in rows:
    try:
        gt = lab.mask1024(b["row"]["query_mask_path"]); s = lab.up(score(b)); f = lab.foris_cli(b)
    except Exception:
        continue
    i, u = lab.iu(f, gt); io, uo, t = lab.best_threshold_iou(s, gt)
    by.setdefault(b["dataset"], []).append((io / max(uo, 1) - i / max(u, 1), i / max(u, 1), io / max(uo, 1), t, b))
for d, L in by.items():
    L.sort(key=lambda x: -x[0])
    # evenly spaced over the worst third, so the sheet is not only the extreme tail
    pick = [L[k] for k in np.linspace(0, len(L) // 3, n_show).astype(int)]
    print(d, len(L), "mean gap %.3f" % np.mean([x[0] for x in L]), "share gap>0.1: %.2f" % np.mean([x[0] > 0.1 for x in L]))
    for page in range(0, n_show, 6):
        fig, ax = plt.subplots(6, 6, figsize=(18, 18))
        for r, (gap, nat, orc, t, b) in enumerate(pick[page:page + 6]):
            R, Q = img(b["row"]["reference_path"]), img(b["row"]["query_path"])
            m = lab.mask1024(b["row"]["reference_mask_path"]).numpy()[::2, ::2]
            gt = lab.mask1024(b["row"]["query_mask_path"]).numpy()[::2, ::2]
            s = lab.up(score(b)); f = lab.foris_cli(b).numpy()[::2, ::2]
            Rm = R.copy(); Rm[m] = (0.5 * Rm[m] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            Qg = Q.copy(); Qg[gt] = (0.5 * Qg[gt] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            Qf = Q.copy(); Qf[f] = (0.5 * Qf[f] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            o = (s >= t).numpy()[::2, ::2]
            Qo = Q.copy(); Qo[o] = (0.5 * Qo[o] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)
            for c, (im, ti) in enumerate([(Rm, "ref"), (Q, "query"), (Qg, "GT"), (Qf, f"FoRIS {nat:.2f}"),
                                          (lab.minmax(s).numpy()[::2, ::2], "score"), (Qo, f"oracle level {orc:.2f}")]):
                ax[r, c].imshow(im, cmap="viridis"); ax[r, c].set_title(ti, fontsize=10); ax[r, c].axis("off")
        plt.tight_layout(); fig.savefig(out / f"{d}_{page // 6}.png", dpi=60); plt.close(fig)
