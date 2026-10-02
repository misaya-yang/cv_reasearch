"""Where does the mIoU loss sit?  (CPU only, from the per-image statistics that eval_fusion.py saves)

  python scripts/budget.py results/ade20k/ens_segformer-b5.perimage.pt [variant ...]
An (image, class) pair is one class in one image. For the argmax prediction:
  1. pairs present in the ground truth, binned by the share of the image they cover: how many are missed entirely,
     pixel recall, and the mIoU if every pair of that bin were recalled perfectly (unions left unchanged, so a lower bound of the gain);
  2. pairs that are predicted but absent from the ground truth, binned by predicted area: mIoU if they were removed;
  3. the classes with the lowest IoU, with the number of validation images that contain them.
Extra variants (second opinions, fusions) are listed next to argmax in 3 and their miss rate is given per bin in 1.
"""
import os, sys, torch
d = torch.load(sys.argv[1], map_location="cpu"); C, N, V = d["C"], d["images"], d["variants"]; extra = [v for v in sys.argv[2:] if v in V]
names = [l.rstrip("\n").split("\t")[-1].split(",")[0].strip() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "objectInfo150.txt"))][1:] if C == 150 else [str(i) for i in range(C)]
def stats(n):
    t = torch.zeros(N, 3, C, dtype=torch.float64); idx = V[n]["idx"].long(); t[idx[:, 0], :, idx[:, 1]] = V[n]["val"].double(); return t
T = stats("argmax"); tp, pr, g = T[:, 0], T[:, 1], T[:, 2]; s = T.sum(0); iou = s[0] / (s[1] + s[2] - s[0]); ok = s[2] > 0
mi = lambda x: float(x[ok].mean()) * 100
area = g.sum(1, keepdim=True); frac, pfrac = g / area, pr / area; pres = g > 0; hall = (pr > 0) & ~pres
E = {v: stats(v) for v in extra}
print(f"{sys.argv[1]}: {N} images, mIoU {mi(iou):.2f}; {int(pres.sum())} present pairs ({pres.sum() / N:.2f} per image), {int(((tp == 0) & pres).sum())} missed entirely; {int(hall.sum())} hallucinated pairs ({hall.sum() / N:.2f} per image)")
edges = [0, 0.0005, 0.002, 0.01, 0.05, 0.2, 1.01]; bins = list(zip(edges[:-1], edges[1:]))
print("\n1. Present pairs by share of the image\n   share of image     pairs   missed entirely   pixel recall   mIoU if recalled perfectly" + "".join(f"   missed with {v}" for v in extra))
for lo, hi in bins:
    m = pres & (frac >= lo) & (frac < hi); iou2 = (s[0] + ((g - tp) * m).sum(0)) / (s[1] + s[2] - s[0])
    print(f"   {lo * 100:5.2f}%-{min(hi, 1) * 100:6.2f}%   {float(m.sum() / pres.sum()) * 100:5.1f}%   {float(((tp == 0) & m).sum() / m.sum()) * 100:5.1f}%           {float((tp * m).sum() / (g * m).sum()) * 100:5.1f}%         {mi(iou2):.2f} ({mi(iou2) - mi(iou):+.2f})"
          + "".join(f"   {float(((E[v][:, 0] == 0) & m).sum() / m.sum()) * 100:5.1f}%" for v in extra))
m = pres & (tp == 0); iou2 = (s[0] + (g * m).sum(0)) / (s[1] + s[2] - s[0]); print(f"   every pair that was missed entirely, recalled perfectly: {mi(iou2):.2f} ({mi(iou2) - mi(iou):+.2f})")
print("\n2. Hallucinated pairs by predicted share of the image\n   share of image     pairs   mIoU if removed")
for lo, hi in bins:
    m = hall & (pfrac >= lo) & (pfrac < hi); iou2 = s[0] / (s[1] - (pr * m).sum(0) + s[2] - s[0])
    print(f"   {lo * 100:5.2f}%-{min(hi, 1) * 100:6.2f}%   {float(m.sum() / hall.sum()) * 100:5.1f}%   {mi(iou2):.2f} ({mi(iou2) - mi(iou):+.2f})")
iou2 = s[0] / (s[1] - (pr * hall).sum(0) + s[2] - s[0]); print(f"   all removed: {mi(iou2):.2f} ({mi(iou2) - mi(iou):+.2f})")
nimg = pres.sum(0); order = torch.argsort(torch.where(ok, iou, torch.ones_like(iou) * 9))
print("\n3. Lowest classes\n   class            val images   IoU argmax" + "".join(f"   {v[-14:]:>14s}" for v in extra) + "   recall   precision")
EI = {v: (lambda t: t[0] / (t[1] + t[2] - t[0]))(E[v].sum(0)) for v in extra}
for c in order[:30].tolist():
    print(f"   {names[c][:16]:16s} {int(nimg[c]):6d}       {float(iou[c]) * 100:5.1f}   " + "".join(f"   {float(EI[v][c]) * 100:14.1f}" for v in extra) + f"   {float(s[0][c] / s[2][c]) * 100:5.1f}    {float(s[0][c] / s[1][c]) * 100:5.1f}")
few = ok & (nimg < 30); print(f"\n   classes with fewer than 30 validation images: {int(few.sum())}; their mean IoU {float(iou[few].mean()) * 100:.1f} vs {float(iou[ok & ~few].mean()) * 100:.1f} for the rest; "
                              f"they hold {float(((1 - iou)[few]).sum() / ((1 - iou)[ok]).sum()) * 100:.0f}% of the total IoU deficit")
