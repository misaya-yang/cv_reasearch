"""CPU only. How much mIoU can a labelling that is constant per s x s cell reach on ADE20K val (short side 512)?
hard: majority class per cell, nearest upsampling.  soft: class fractions per cell, bilinear upsampling, argmax."""
import glob, torch, numpy as np, torch.nn.functional as F, sys
from PIL import Image
torch.set_num_threads(16)
C = 150; files = sorted(glob.glob("/root/demo2_cache/ADEChallengeData2016/annotations/validation/*.png"))
strides = [4, 8, 16, 32]; names = [f"{k}{s}" for s in strides for k in ("hard", "soft")]
tp = {n: torch.zeros(C, dtype=torch.float64) for n in names}; pr = {n: torch.zeros(C, dtype=torch.float64) for n in names}; gt_n = torch.zeros(C, dtype=torch.float64)
area_frac = []  # (class, fraction of image, cells of 16) per (image, class)
for i, f in enumerate(files):
    y = torch.from_numpy(np.array(Image.open(f))).long() - 1   # -1 = ignore
    H, W = y.shape; sc = 512 / min(H, W); h, w = int(round(H * sc / 32)) * 32, int(round(W * sc / 32)) * 32
    y = F.interpolate(y[None, None].float(), (h, w), mode="nearest")[0, 0].long(); valid = y >= 0
    oh = F.one_hot(y.clamp(min=0), C).permute(2, 0, 1).float() * valid[None]
    gt_n += oh.flatten(1).sum(1).double()
    cnt = oh.flatten(1).sum(1); pres = cnt > 0
    for c in pres.nonzero()[:, 0].tolist(): area_frac.append((c, float(cnt[c] / valid.sum()), float(cnt[c] / 256)))
    for s in strides:
        p = F.avg_pool2d(oh[None], s)[0]
        hard = F.interpolate(p.argmax(0)[None, None].float(), (h, w), mode="nearest")[0, 0].long()
        soft = F.interpolate(p[None], (h, w), mode="bilinear", align_corners=False)[0].argmax(0)
        for k, m in (("hard", hard), ("soft", soft)):
            n = f"{k}{s}"; pv = m[valid]; gv = y[valid]
            pr[n] += torch.bincount(pv, minlength=C).double(); tp[n] += torch.bincount(gv[pv == gv], minlength=C).double()
    if i % 400 == 0: print(i, flush=True)
ok = gt_n > 0
for n in names:
    iou = tp[n] / (gt_n + pr[n] - tp[n]); print(n, "mIoU %.2f" % (float(iou[ok].mean()) * 100), "pixel acc %.2f" % (float(tp[n].sum() / gt_n.sum()) * 100), "classes with IoU<0.5: %d" % int((iou[ok] < 0.5).sum()))
a = torch.tensor(area_frac)
print("image-class pairs:", len(a), " per image: %.2f" % (len(a) / len(files)))
for t in (1, 4, 16, 64): print(f"  pairs covering < {t} cells of 16x16: {float((a[:, 2] < t).float().mean()) * 100:.1f}%")
for t in (0.001, 0.005, 0.01, 0.05): print(f"  pairs covering < {t*100:.1f}% of the image: {float((a[:, 1] < t).float().mean()) * 100:.1f}%")
torch.save(dict(tp=tp, pr=pr, gt=gt_n, area=a), "/root/autodl-tmp/demo2_pilot/grid_oracle.pt")
