"""Which DINOv3 layer gives the best correspondence evidence for choosing a tree node?  (COCO-20i, one shot)

The target tree is INSID3's (last layer, original features). For every layer in --layers, with and without positional debiasing
(basis estimated per layer the way INSID3 does for the last one), nearest-neighbour maps in both directions are saved:
  back = nearest reference patch is foreground;  fwd / bfwd = number of reference FG / BG patches whose nearest target patch is this one.
"avg" = nearest neighbours under the similarity averaged over the listed layers.

  python scripts/layers.py --fold 0 --n 300 --out results/layers_f0_300
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx import *
from icx.dissect import dissect, up
from utils.data import load_image, load_mask, downsample_mask
from sklearn.cluster import AgglomerativeClustering
import einops

ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=300); ap.add_argument("--out", required=True)
ap.add_argument("--layers", default="9,12,15,18,20,22,23")
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); LS = [int(x) for x in a.layers.split(",")]
model = build_model(); tf = model._transform; S_ = model.image_size; enc = model.encoder.m

@torch.no_grad()
def feats(x):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = enc.forward_intermediates(x, indices=LS, norm=True, output_fmt="NCHW", intermediates_only=True)
    return [F.normalize(o.float(), dim=1) for o in out]                      # list of (B, C, h, w)

from torchvision.transforms.functional import normalize as tvnorm
black = tvnorm(torch.zeros(1, 3, S_, S_), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(DEV); basis = []
with torch.no_grad():
    for o in enc.forward_intermediates(black, indices=LS, norm=True, output_fmt="NCHW", intermediates_only=True):
        E = einops.rearrange(F.normalize(o.float(), dim=1), "b c h w -> c (b h w)"); E = E - E.mean(1, keepdim=True); U = torch.linalg.svd(E, full_matrices=False)[0]; basis.append(U[:, :500].contiguous())
print("last-layer basis equals INSID3's:", float((basis[-1] @ basis[-1].T - model.positional_basis @ model.positional_basis.T).abs().max()), flush=True)
deb = lambda X, Bm: F.normalize(X - (X @ Bm) @ Bm.T, dim=1)                   # X (N, C)

eps, class_ids, base = coco_episodes(a.fold, a.n); out = []; t0 = time.time()
for e, (c, tn, rn) in enumerate(eps):
    timg, tmask = coco_load(base, tn, c); rimg, rmask = coco_load(base, rn[0], c)
    ti, _ = load_image(timg, tf, DEV); ri, _ = load_image(rimg, tf, DEV); rm = load_mask(rmask, S_, DEV)
    gt = F.interpolate(tmask[None, None].float().to(DEV), size=(S_, S_), mode="nearest")[0, 0] > 0.5
    fs = feats(torch.cat([ri, ti], 0)); h = fs[0].shape[-1]; n = h * h; C = fs[0].shape[1]
    Tl = fs[-1][1].reshape(C, -1).T; D = (1.0 - (Tl @ Tl.T).clamp(-1, 1)).cpu().numpy()
    ac = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=float(1.0 - model.tau)).fit(D)
    g = F.avg_pool2d(gt.float()[None, None], S_ // h)[0, 0].reshape(-1); mf = downsample_mask(rm.unsqueeze(1), h, h).reshape(-1)
    ev = {}; acc = {"deb": 0, "raw": 0}
    def maps(Sm, name):
        nt = Sm.argmax(1); nr = Sm.argmax(0); ev[name + "/back"] = mf[nt].cpu().numpy(); ev[name + "/fwd"] = torch.bincount(nr[mf], minlength=n).to(torch.int16).cpu().numpy()
        ev[name + "/bfwd"] = torch.bincount(nr[~mf], minlength=n).to(torch.int16).cpu().numpy()
    for l, fl, Bm in zip(LS, fs, basis):
        R, Tt = fl[0].reshape(C, -1).T, fl[1].reshape(C, -1).T; Sr = Tt @ R.T; Sdb = deb(Tt, Bm) @ deb(R, Bm).T
        maps(Sr, f"L{l}raw"); maps(Sdb, f"L{l}deb"); acc["raw"] = acc["raw"] + Sr; acc["deb"] = acc["deb"] + Sdb
    maps(acc["raw"], "avgraw"); maps(acc["deb"], "avgdeb")
    out.append(dict(e=e, c=c, children=ac.children_.astype(np.int16), dist=ac.distances_.astype(np.float16), g=g.half().cpu().numpy(), n_reffg=int(mf.sum()), n_refbg=int((~mf).sum()), ev=ev))
    if (e + 1) % 50 == 0: print(e + 1, f"{time.time() - t0:.0f}s", flush=True)
torch.save(out, a.out + ".layers.pt"); print("done", round(time.time() - t0))
