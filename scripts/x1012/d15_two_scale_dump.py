"""Two-scale observation dump: does a second, finer observation add usable evidence?

Global view: cached O24 at 1024 (64x64 tokens).  Fine view: 2x2 tiles of the same 1024 frame,
each re-encoded at 1024 (-> 128x128 tokens of 8 px), for query and reference.
All features: positional projection, centred by the global reference mean, unit norm.
Saved score maps (no labels used):
  gg  global query tokens  x global reference prototype          (64)
  fg  fine   query tokens  x global reference prototype          (128)
  ff  fine   query tokens  x fine   reference prototype          (128)
  gf  global query tokens  x fine   reference prototype          (64)
"""
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
import lab, enc
from lab import *

N, OUT = int(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
rows = lab.subset(N, offset=2)
tf = enc.transform()


def tiles(x):
    """(3,1024,1024) normalised image -> 128x128 fine tokens (16384, C) from four 2x-zoomed tiles."""
    grid = torch.zeros(128, 128, 1024, device=DEV)
    for ty in range(2):
        for tx in range(2):
            t = x[:, ty * 512:(ty + 1) * 512, tx * 512:(tx + 1) * 512]
            t = F.interpolate(t[None], size=(1024, 1024), mode="bilinear", align_corners=False)[0]
            f = enc.layers(t, [24])[24].reshape(64, 64, 1024)
            grid[ty * 64:(ty + 1) * 64, tx * 64:(tx + 1) * 64] = f
    return grid.reshape(-1, 1024)


t0 = time.time()
for kk, b in enumerate(rows):
    f = OUT / f"{b['index']:06d}.npz"
    if f.exists():
        continue
    mr = lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)
    y64 = lab.coverage64(mr).flatten()
    y128 = F.avg_pool2d(mr[None, None].float(), 8)[0, 0].flatten()
    rg, qg = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    rf = lab.debias(tiles(tf(enc.rgb(b, "reference")).to(DEV)))
    qf = lab.debias(tiles(tf(enc.rgb(b, "query")).to(DEV)))
    mu = rg.mean(0)
    muf = rf.mean(0)

    def proto(r, y, m):
        w = y if y.sum() > 1e-6 else (y >= y.max()).float()
        return lab.unit((r * w[:, None]).sum(0) / w.sum() - m)

    pg, pf, pff = proto(rg, y64, mu), proto(rf, y128, mu), proto(rf, y128, muf)
    np.savez(f,
             gg=(lab.unit(qg - mu) @ pg).cpu().numpy(), fg=(lab.unit(qf - mu) @ pg).cpu().numpy(),
             ff=(lab.unit(qf - mu) @ pf).cpu().numpy(), gf=(lab.unit(qg - mu) @ pf).cpu().numpy(),
             ff_own=(lab.unit(qf - muf) @ pff).cpu().numpy(),
             y64=y64.cpu().numpy())
    lab.free()
    if kk % 20 == 0:
        print(kk, len(rows), round(time.time() - t0, 1), flush=True)
print("done", time.time() - t0)
