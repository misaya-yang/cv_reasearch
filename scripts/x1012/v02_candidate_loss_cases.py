"""Contact sheets of the development episodes where the frozen independent candidate (R1 VP_UN)
adds the most error pixels relative to complete FoRIS (diagnostic only; query GT picks and draws)."""
import sys, json
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
doms = sys.argv[2].split(",")
R1 = lab.ASSETS / "a/x1012_R1"
rows = json.loads((R1 / "manifest.json").read_text())
key = lambda b: b["index"] if isinstance(b["index"], str) else format(b["index"], "06d")
bits = lambda a: torch.from_numpy(np.unpackbits(a)[: 1024 * 1024].reshape(1024, 1024).astype(bool))


def img(p):
    q = lab.ASSETS / p
    return np.array(Image.open(q if q.exists() else p).convert("RGB").resize((512, 512)))


def tint(im, m, col):
    o = im.copy(); o[m] = (0.45 * o[m] + 0.55 * np.array(col)).astype(np.uint8); return o


by = {}
for b in rows:
    if b["dataset"] not in doms:
        continue
    gt = lab.mask1024(b["row"]["query_mask_path"]); f = lab.foris_cli(b)
    u = bits(np.load(R1 / "predictions" / f"{key(b)}.npz")["VP_UN"])
    ef, eu = int((f ^ gt).sum()), int((u ^ gt).sum())
    by.setdefault(b["dataset"], []).append((eu - ef, b, f, u, gt))
for d, L in by.items():
    L.sort(key=lambda x: -x[0])
    tot = sum(x[0] for x in L); pos = sum(x[0] for x in L if x[0] > 0)
    print(d, len(L), f"net extra error px {tot:.3g}; top12 share of positive part {sum(x[0] for x in L[:12]) / max(pos, 1):.2f};",
          f"episodes worse/better {sum(x[0] > 0 for x in L)}/{sum(x[0] < 0 for x in L)}")
    for page in range(2):
        fig, ax = plt.subplots(6, 5, figsize=(15, 18))
        for r, (de, b, f, u, gt) in enumerate(L[page * 6: page * 6 + 6]):
            R, Q = img(b["row"]["reference_path"]), img(b["row"]["query_path"])
            m = lab.mask1024(b["row"]["reference_mask_path"]).numpy()[::2, ::2]
            fld = np.load(R1 / "fields" / f"{key(b)}.npz")
            s = lab.minmax(lab.up(torch.from_numpy(fld["VP_UN"]))).numpy()[::2, ::2]
            g, ff, uu = gt.numpy()[::2, ::2], f.numpy()[::2, ::2], u.numpy()[::2, ::2]
            iou = lambda p: (p & g).sum() / max((p | g).sum(), 1)
            for c, (im, ti) in enumerate([(tint(R, m, (0, 255, 0)), f"ref rho={float(fld['reference_rho'][0]):.2f}"), (tint(Q, g, (255, 0, 0)), "query + GT"),
                                          (tint(Q, ff, (0, 0, 255)), f"FoRIS {iou(ff):.2f}"), (tint(Q, uu, (255, 0, 255)), f"VP_UN {iou(uu):.2f}"), (s, "VP_UN field")]):
                ax[r, c].imshow(im, cmap="viridis"); ax[r, c].set_title(ti, fontsize=10); ax[r, c].axis("off")
        plt.tight_layout(); fig.savefig(out / f"un_{d}_{page}.png", dpi=60); plt.close(fig)
