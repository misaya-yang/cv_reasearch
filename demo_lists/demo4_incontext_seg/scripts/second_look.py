"""Second look: re-encode each candidate region as an image of its own and compare with the reference object encoded the same way.

Candidates = the K tree nodes with the highest bidirectional F1. Each candidate (and the reference mask) is cropped to its bounding
box plus a margin and resized to --res; variants: plain crop, crop with everything outside the region greyed out.
Scores saved per candidate: cosine of CLS tokens, cosine of patch features averaged inside the region.

  python scripts/second_look.py --fold 0 --n 300 --out results/look_f0_300
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from icx.offline2 import accumulate, f1, leaves_of
from utils.data import load_image, load_mask, downsample_mask
from sklearn.cluster import AgglomerativeClustering

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=300); ap.add_argument("--out", required=True)
ap.add_argument("--K", type=int, default=12); ap.add_argument("--res", type=int, default=448); ap.add_argument("--margin", type=float, default=0.15)
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
model = build_model(); tf = model._transform; S_ = model.image_size; enc = model.encoder.m; U = model.positional_basis
proj = lambda X: F.normalize(X - (X @ U) @ U.T, dim=-1)

def crop(img, mask, grey):
    """img (1,3,S,S) normalised, mask (S,S) bool -> (3,res,res) crop, (res,res) mask."""
    ys, xs = torch.where(mask)
    if len(ys) == 0: ys = xs = torch.tensor([0, S_ - 1], device=img.device)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1; my, mx = int(a.margin * (y1 - y0)) + 4, int(a.margin * (x1 - x0)) + 4
    y0, y1, x0, x1 = max(0, y0 - my), min(S_, y1 + my), max(0, x0 - mx), min(S_, x1 + mx); im = img[0, :, y0:y1, x0:x1]; m = mask[y0:y1, x0:x1]
    if grey: im = im * m[None]                                             # zero in normalised space = mean colour
    im = F.interpolate(im[None], size=(a.res, a.res), mode="bilinear", align_corners=False)[0]; m = F.interpolate(m[None, None].float(), size=(a.res, a.res), mode="nearest")[0, 0] > 0.5
    return im, m

@torch.no_grad()
def embed(ims, ms):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16): tok = enc.forward_features(torch.stack(ims))
    tok = tok.float(); cls = F.normalize(tok[:, 0], dim=1); P = tok[:, enc.num_prefix_tokens:]; hh = int(math.sqrt(P.shape[1]))
    mm = torch.stack([F.avg_pool2d(m[None, None].float(), a.res // hh)[0, 0].reshape(-1) for m in ms])        # (B, N) soft mask on the crop grid
    pool = F.normalize((mm[:, :, None] * P).sum(1) / mm.sum(1, keepdim=True).clamp(min=1e-6), dim=1); return cls, pool, proj(pool)

eps, class_ids, base = coco_episodes(a.fold, a.n); out = []; t0 = time.time()
for e, (c, tn, rn) in enumerate(eps):
    timg, tmask = coco_load(base, tn, c); rimg, rmask = coco_load(base, rn[0], c)
    ti, _ = load_image(timg, tf, DEV); ri, _ = load_image(rimg, tf, DEV); rm = load_mask(rmask, S_, DEV)
    gt = F.interpolate(tmask[None, None].float().to(DEV), size=(S_, S_), mode="nearest")[0, 0] > 0.5
    f = F.normalize(model._extract_features(torch.cat([ri, ti], 0).unsqueeze(0)).float(), p=2, dim=2)[0]; C = f.shape[1]; R, Tt = f[0].reshape(C, -1).T, f[1].reshape(C, -1).T; h = int(math.sqrt(len(Tt))); n = h * h
    ac = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=float(1.0 - model.tau)).fit((1.0 - (Tt @ Tt.T).clamp(-1, 1)).cpu().numpy())
    ch = ac.children_.astype(np.int64); g = F.avg_pool2d(gt.float()[None, None], S_ // h)[0, 0].reshape(-1).cpu().numpy(); mf = downsample_mask(rm.unsqueeze(1), h, h).reshape(-1)
    Sd = proj(Tt) @ proj(R).T; back = mf[Sd.argmax(1)].float().cpu().numpy(); fwd = torch.bincount(Sd.argmax(0)[mf], minlength=n).float().cpu().numpy()
    X = accumulate(ch, np.stack([np.ones(n, np.float32), g, back, fwd])); Fs = f1(X[2] / X[0], X[3] / max(int(mf.sum()), 1)); io = X[1] / (X[0] + X[1, -1] - X[1] + 1e-9)
    top = np.argsort(-Fs)[:a.K]; ims, ms = [], []
    for grey in (False, True):
        im, m = crop(ri, rm[0], grey); ims.append(im); ms.append(m)
        for k in top:
            nm = torch.zeros(n, dtype=torch.bool, device=DEV); nm[torch.from_numpy(leaves_of(ch, n, k)).to(DEV)] = True
            nm = F.interpolate(nm.reshape(1, 1, h, h).float(), size=(S_, S_), mode="nearest")[0, 0] > 0.5; im, m = crop(ti, nm, grey); ims.append(im); ms.append(m)
    cls, pool, poold = embed(ims, ms); B = 1 + a.K; rec = dict(e=e, c=c, top=top, F1=Fs[top], iou=io[top], area=X[0][top], fg=X[1][top], G=float(X[1, -1]), best=float(io.max()))
    for j, tag in enumerate(("plain", "grey")):
        sl = slice(j * B, (j + 1) * B)
        for nm_, Z in (("cls", cls[sl]), ("pool", pool[sl]), ("poold", poold[sl])): rec[f"{tag}_{nm_}"] = (Z[1:] @ Z[0]).cpu().numpy()
    out.append(rec)
    if (e + 1) % 50 == 0: print(e + 1, f"{time.time() - t0:.0f}s", flush=True)
torch.save(out, a.out + ".look.pt"); print("done", round(time.time() - t0))
