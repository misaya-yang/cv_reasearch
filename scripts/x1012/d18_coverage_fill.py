"""Sub-token extent: coverage transfer + within-token pixel allocation (CPU, cached O24 + RGB).

Token level: expected coverage c_j = sum_i softmax(<q_j, r_i>/tau) y_i  (y = soft reference coverage).
Pixel level: within each 16x16 token, the top c_j fraction of pixels by a colour log-likelihood
ratio fitted on the reference pixels (raw + local-contrast colour, Gaussian FG vs BG).
Solid targets (coverage ~ 1) are unaffected by construction; thin targets become representable.
"""
import sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import lab

lab.DEV = torch.device("cpu")
torch.set_num_threads(8)
SPEC = dict(a.split("=") for a in sys.argv[1:])          # e.g. fundus=60 deepglobe_road=60 coco=30
rows = []
for d, n in SPEC.items():
    rows += lab.ext_subset(int(n), (d,)) if d in lab.EXT_DOMAINS else lab.subset(int(n), (d,))
ALL = list(SPEC)


def rgb1024(b, role):
    row = b["row"]
    p = Path(row[role + "_path"])
    with Image.open(p if p.is_absolute() else lab.ASSETS / p) as im:
        x = np.asarray(im.convert("RGB")).copy()
    crop = row.get(role + "_crop")
    if crop is not None:
        x0, y0, x1, y1 = crop
        x = x[y0:y1, x0:x1]
    im = Image.fromarray(x).resize((1024, 1024), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0


def colour_feat(x):
    """Raw colour + local contrast (colour minus a 33-px box mean): (6, H, W)."""
    blur = F.avg_pool2d(F.pad(x[None], (16, 16, 16, 16), mode="reflect"), 33, stride=1)[0]
    return torch.cat([x, x - blur], 0)


def gauss(v):
    mu = v.mean(0)
    c = torch.cov(v.T) + 1e-4 * torch.eye(v.shape[1])
    return mu, torch.linalg.inv(c), torch.logdet(c)


def llr_map(fr, mr, fq):
    """Per-pixel Gaussian log-likelihood ratio FG vs BG, fitted on reference pixels."""
    v = fr.reshape(6, -1).T
    m = mr.flatten()
    if m.sum() < 16 or (~m).sum() < 16:
        return torch.zeros(1024, 1024)
    out = 0
    for sign, sel in ((1, m), (-1, ~m)):
        vv = v[sel]
        vv = vv[:: max(1, len(vv) // 200000)]
        mu, P, ld = gauss(vv)
        d = fq.reshape(6, -1).T - mu
        out = out + sign * (-0.5 * ((d @ P) * d).sum(1) - 0.5 * ld)
    return out.reshape(1024, 1024)


def allocate(c64, llr):
    """Within each token keep the top c fraction of its 256 pixels by llr."""
    blocks = llr.reshape(64, 16, 64, 16).permute(0, 2, 1, 3).reshape(4096, 256)
    k = (c64.flatten().clamp(0, 1) * 256).round().long()
    rank = blocks.argsort(1, descending=True).argsort(1)
    keep = rank < k[:, None]
    return keep.reshape(64, 64, 16, 16).permute(0, 2, 1, 3).reshape(1024, 1024)


M = {}
t0 = time.time()
for kk, b in enumerate(rows):
    gt = lab.mask1024(b["row"]["query_mask_path"])
    mr = lab.mask1024(b["row"]["reference_mask_path"])
    ig = lab.ignore1024(b)
    y = lab.coverage64(mr).flatten()
    r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
    mu = 0.5 * (r.mean(0) + q.mean(0))
    rc, qc = lab.unit(r - mu), lab.unit(q - mu)
    c = torch.softmax((qc @ rc.T) / 0.05, dim=1) @ y
    llr = llr_map(colour_feat(rgb1024(b, "reference")), mr, colour_feat(rgb1024(b, "query")))
    cg = lab.coverage64(gt)

    def add(name, pred):
        g = gt
        if ig is not None:
            pred, g = pred & ~ig, gt & ~ig
        M.setdefault(name, lab.Meter()).add(b, *lab.iu(pred, g))

    add("FoRIS complete (CRF)", lab.foris_cli(b))
    add("ceiling: hard 64-grid tokens", lab.up(cg) > 0.5)
    add("ceiling: GT coverage + colour fill", allocate(cg, llr))
    add("nw coverage > 0.5 (bilinear)", lab.up(c.reshape(64, 64)) > 0.5)
    add("nw coverage minmax 0.5", lab.minmax(lab.up(c.reshape(64, 64))) > 0.5)
    add("nw coverage + colour fill", allocate(c.reshape(64, 64), llr))
    # coverage rescaled so that its top level equals the reference's top coverage level
    scale = torch.quantile(y, 0.995).clamp_min(1e-3) / torch.quantile(c, 0.995).clamp_min(1e-3)
    add("nw coverage (ref-top matched) + fill", allocate((c * scale).clamp(0, 1).reshape(64, 64), llr))
    if kk % 40 == 0:
        print(kk, len(rows), round(time.time() - t0, 1), flush=True)

print("\nclass-pooled mIoU (per-episode mean IoU), CLI-1024")
print(f"{'method':38s}" + "".join(f"{d[:10]:>16s}" for d in ALL))
for name, m in M.items():
    r = m.row(ALL)
    print(f"{name:38s}" + "".join(f"{r[d][0]:8.1f} ({r[d][1]:5.1f})" for d in ALL))
