"""Paired bootstrap over images for the mIoU difference between variants of one run.

  python scripts/bootstrap.py results/ade20k/ens_eomt-large.perimage.pt argmax ext/fuse^2 lin/ens^0.5 ...
Prints, for every listed variant, mIoU and the difference to the first one with a 95% interval (2000 resamples of the image set).
With no variants listed, prints the names available.
"""
import sys, torch
d = torch.load(sys.argv[1]); C, N, V = d["C"], d["images"], d["variants"]; names = sys.argv[2:]
if not names: print("\n".join(V)); raise SystemExit
def dense(name):
    v = V[name]; t = torch.zeros(N, 3, C, dtype=torch.float64); idx = v["idx"].long(); t[idx[:, 0], :, idx[:, 1]] = v["val"].double(); return t
def miou(t):                                                   # t: (..., 3, C) summed counts -> dataset-level mIoU
    tp, p, g = t[..., 0, :], t[..., 1, :], t[..., 2, :]; return torch.nanmean(tp / (g + p - tp), -1) * 100
T = {n: dense(n) for n in names}; torch.manual_seed(0); B = 2000
w = torch.zeros(B, N, dtype=torch.float64).scatter_add_(1, torch.randint(0, N, (B, N)), torch.ones(B, N, dtype=torch.float64))   # resampling weights
boot = {n: miou(torch.einsum("bn,nkc->bkc", w, t)) for n, t in T.items()}; base = names[0]
print(f"{'variant':44s} {'mIoU':>7s}  {'diff to ' + base:>18s}   95% interval      P(diff<=0)")
for n in names:
    m = float(miou(T[n].sum(0))); df = boot[n] - boot[base]; lo, hi = (float(v) for v in torch.quantile(df, torch.tensor([0.025, 0.975], dtype=torch.float64)))
    print(f"{n:44s} {m:7.2f}  {m - float(miou(T[base].sum(0))):+18.2f}   [{lo:+.2f}, {hi:+.2f}]   {float((df <= 0).double().mean()):.3f}")
print(f"spread of the base mIoU itself under resampling: sd = {float(boot[base].std()):.2f}")
