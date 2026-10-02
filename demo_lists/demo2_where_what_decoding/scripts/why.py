"""Why does a second opinion help, and do different ways of attaching it fix the same things?  (CPU only, from per-image statistics)

  python scripts/why.py results/ade20k/ens_segformer-b5.perimage.pt
Per class: IoU of the segmenter, of each second opinion on its own, and of each combination. Then
  1. is the gain of a combination explained by where the second opinion is better than the segmenter on its own?
  2. do two ways of attaching the same encoder (pixel product vs region recogniser) gain on the same classes?
  3. where does the gain sit: frequent or rare classes, classes the segmenter is good or bad at?
  4. how large is the per-class change when nothing but the input mirror changes (the noise reference)?
"""
import sys, torch
d = torch.load(sys.argv[1]); C, N, V = d["C"], d["images"], d["variants"]
def stats(n):
    t = torch.zeros(N, 3, C, dtype=torch.float64); idx = V[n]["idx"].long(); t[idx[:, 0], :, idx[:, 1]] = V[n]["val"].double(); return t.sum(0)
def iou(n): tp, p, g = stats(n); return tp / (g + p - tp)
base = iou("argmax"); G = stats("argmax")[2]; ok = ~torch.isnan(base) & (G > 0)
def corr(a, b):
    m = ok & ~torch.isnan(a) & ~torch.isnan(b); a, b = a[m] - a[m].mean(), b[m] - b[m].mean(); return float((a * b).sum() / (a.norm() * b.norm()))
def rank(x): r = torch.empty_like(x); r[x.argsort()] = torch.arange(len(x), dtype=x.dtype); return r
def spear(a, b):
    m = ok & ~torch.isnan(a) & ~torch.isnan(b); return corr_raw(rank(a[m]), rank(b[m]))
def corr_raw(a, b): a, b = a - a.mean(), b - b.mean(); return float((a * b).sum() / (a.norm() * b.norm()))
mi = lambda x: float(torch.nanmean(x[ok])) * 100
print(f"{sys.argv[1]}: {N} images, {int(ok.sum())} classes, segmenter mIoU {mi(base):.2f}")
have = lambda n: n in V
combos = [n for n in ("ext/fuse^2", "ext/relabel^2", "lin/ens^1", "lin/ens^0.5", "lin/region_vote/fuse^1", "upernet/ens^1", "eomt/ens^1", "m2fl/ens^1", "b5/ens^1", "v:flip/mean",
                      "v:all/mean", "both/fuse^2", "dino/fuse^2", "sig/fuse^2", "own_gt/fuse^2", "dino_gt/fuse^2") if have(n)]
alone = [n for n in ("lin/alone", "upernet/alone", "eomt/alone", "m2fl/alone", "b5/alone", "v:flip/alone") if have(n)]
gain = {n: iou(n) - base for n in combos}; adv = {n: iou(n) - base for n in alone}

print("\n1. Is the gain explained by where the second opinion is better on its own?  (per class; gain of the combination vs advantage of the second opinion alone)")
for a in alone:
    src = a.split("/")[0]
    for c in combos:
        if c.startswith(src + "/") or (src == "lin" and c.startswith("ext/")):
            print(f"   {c:26s} gain {mi(gain[c]):+5.2f} | {a:14s} alone {mi(adv[a]):+6.2f} | Pearson r = {corr(gain[c], adv[a]):+.2f}, Spearman = {spear(gain[c], adv[a]):+.2f}")

print("\n2. Do different attachments gain on the same classes?  (per-class gain vs per-class gain)")
for i, a in enumerate(combos):
    for b in combos[i + 1:]:
        print(f"   {a:26s} vs {b:26s} Pearson r = {corr(gain[a], gain[b]):+.2f}   both up {int(((gain[a] > 0.005) & (gain[b] > 0.005) & ok).sum()):3d}  only first {int(((gain[a] > 0.005) & ~(gain[b] > 0.005) & ok).sum()):3d}  only second {int((~(gain[a] > 0.005) & (gain[b] > 0.005) & ok).sum()):3d}")

print("\n3. Where the gain sits (mean IoU change in points; classes split into thirds)")
for title, keyv in (("by pixel frequency (rare -> frequent)", G), ("by the segmenter's own IoU (bad -> good)", base.nan_to_num(0))):
    order = torch.argsort(torch.where(ok, keyv, torch.full_like(keyv, float("inf")))); n = int(ok.sum()); thirds = [order[:n // 3], order[n // 3:2 * n // 3], order[2 * n // 3:n]]
    print(f"   {title}:  segmenter IoU per third = " + " / ".join(f"{float(torch.nanmean(base[t])) * 100:.1f}" for t in thirds))
    for c in combos: print(f"      {c:26s} " + " / ".join(f"{float(torch.nanmean(gain[c][t])) * 100:+6.2f}" for t in thirds) + f"   share of total gain: " + " / ".join(f"{float(gain[c][t].nansum() / gain[c][ok].nansum()) * 100:4.0f}%" for t in thirds))

print("\n4. Concentration and noise")
for c in combos:
    gsorted = gain[c][ok].nan_to_num(0).sort(descending=True).values; tot = gsorted.sum()
    print(f"   {c:26s} top 10 classes carry {float(gsorted[:10].sum() / tot) * 100:4.0f}% of the gain; {int((gain[c][ok] < -0.005).sum()):3d} classes get worse by more than 0.5 point; per-class sd of change {float(gain[c][ok].std()) * 100:.2f}")
if have("v:flip/alone"): print(f"   mirror image alone (no information added): mIoU change {mi(adv['v:flip/alone']):+.2f}, per-class sd of change {float(adv['v:flip/alone'][ok].std()) * 100:.2f}, |change| > 2 points in {int((adv['v:flip/alone'][ok].abs() > 0.02).sum())} classes")
