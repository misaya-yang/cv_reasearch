"""Train a region recogniser (MLP on pooled features) from files written by extract_regions.py.

  python scripts/train_recogniser.py --train rf/ade20k_train_X_own.pt --kinds gt --out rec/X_own_gt.pt --val rf/ade20k_val_X_own.pt
  python scripts/train_recogniser.py --train rf/ade20k_val_X_own.pt --kinds gt,pred --crossfit --out rec/X_own_cf.pt
`--kinds`: gt = ground-truth class regions (hard labels); pred = predicted regions (soft labels = ground-truth class histogram).
`--crossfit`: two models, each trained on the images of one parity; `<out>.f0` is trained on odd images and is to be applied to even ones.
Paths are relative to $DEMO2_CACHE unless absolute.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wwd import *

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True, nargs="+"); ap.add_argument("--val", default=None); ap.add_argument("--out", required=True)
ap.add_argument("--kinds", default="gt"); ap.add_argument("--crossfit", action="store_true"); ap.add_argument("--C", type=int, default=150)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--hidden", type=int, default=2048)
ap.add_argument("--drop", type=float, default=0.3); ap.add_argument("--min-purity", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dims", default=None, help="keep only feature columns lo:hi (ablation)")
ap.add_argument("--concat", default=[], nargs="+", help="per --train file, a `+`-joined list of files whose features are appended column-wise")
ap.add_argument("--val-concat", default=None, help="the same for --val")
a = ap.parse_args(); C = a.C
P = lambda p: p if p.startswith("/") else f"{CACHE}/{p}"
kinds = [dict(gt=0, pred=1)[k] for k in a.kinds.split(",")]
lo, hi = (int(v) for v in a.dims.split(":")) if a.dims else (0, None)


def load(paths, kinds, min_purity=0.0, concat=()):
    ds = [torch.load(P(p)) for p in paths]; off = 0; X, T, Y, I, K, PC = [], [], [], [], [], []
    for j, d in enumerate(ds):
        if concat:                                            # same regions in the same order, other sources
            for q in concat[j].split("+"):
                e = torch.load(P(q)); assert torch.equal(e["img"], d["img"]) and torch.equal(e["y"], d["y"]), q; d["x"] = torch.cat([d["x"], e["x"]], 1)
        keep = torch.isin(d["kind"], torch.tensor(kinds, dtype=torch.uint8)) & (d["purity"] >= min_purity)
        t = torch.zeros(int(keep.sum()), C); tc, tf = d["tk_cls"][keep], d["tk_frac"][keep]
        t.scatter_add_(1, tc.clamp(min=0), tf * (tc >= 0)); t = t / t.sum(1, keepdim=True)
        X.append(d["x"][keep][:, lo:hi]); T.append(t); Y.append(d["y"][keep]); I.append(d["img"][keep].long() + off); K.append(d["kind"][keep]); PC.append(d["pc"][keep])
        off += d["files"]
    return torch.cat(X), torch.cat(T), torch.cat(Y), torch.cat(I), torch.cat(K), torch.cat(PC)


def fit(X, T, seed):
    """X half (N, d) on CPU, T soft targets (N, C). Returns checkpoint dict."""
    torch.manual_seed(seed); mu = X.float().mean(0, keepdim=True); sd = X.float().std(0, keepdim=True) + 1e-6
    Xd, Td = ((X.float() - mu) / sd).to(DEV), T.to(DEV); net = make_mlp(Xd.shape[1], C, a.hidden, a.drop).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=a.wd); steps = a.epochs * (len(Xd) // 512 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps); Ts = Td * 0.9 + 0.1 / C
    for ep in range(a.epochs):
        net.train(); p = torch.randperm(len(Xd), device=DEV)
        for i in range(0, len(p), 512):
            b = p[i:i + 512]; loss = -(Ts[b] * torch.log_softmax(net(Xd[b]), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    return dict(state=net.state_dict(), mu=mu, sd=sd, dim=Xd.shape[1], prior=T.mean(0), args=vars(a)), net


def report(net, ck, X, Y, K, PC, tag):
    with torch.no_grad(): pr = net(((X.float() - ck["mu"]) / ck["sd"]).to(DEV)).argmax(1).cpu()
    out = {}
    for k, name in ((0, "gt_regions"), (1, "pred_regions")):
        m = K == k
        if m.any():
            out[f"{tag}/{name}/n"] = int(m.sum()); out[f"{tag}/{name}/rec_acc"] = round(float((pr[m] == Y[m]).float().mean()), 4)
            if k == 1: out[f"{tag}/{name}/seg_acc"] = round(float((PC[m] == Y[m]).float().mean()), 4)
    return out


X, T, Y, I, K, PC = load(a.train, kinds, a.min_purity, a.concat); res = dict(args=vars(a), n_train=len(X), dim=X.shape[1]); t0 = time.time()
os.makedirs(os.path.dirname(P(a.out)), exist_ok=True)
if a.crossfit:
    Xa, Ta, Ya, Ia, Ka, PCa = load(a.train, [0, 1], 0.0, a.concat)
    for fold in (0, 1):                                       # model f<fold> is trained on images of the other parity
        tr = I % 2 != fold; ck, net = fit(X[tr], T[tr], a.seed); torch.save(ck, P(a.out) + f".f{fold}")
        te = Ia % 2 == fold; res.update(report(net, ck, Xa[te], Ya[te], Ka[te], PCa[te], f"crossfit_f{fold}"))
else:
    hold = I % 10 == 0; ck, net = fit(X[~hold], T[~hold], a.seed)           # 10% of images held out, only to report accuracy
    res.update(report(net, ck, X[hold], Y[hold], K[hold], PC[hold], "train_heldout"))
    ck, net = fit(X, T, a.seed); torch.save(ck, P(a.out))
    if a.val:
        Xv, Tv, Yv, Iv, Kv, PCv = load([a.val], [0, 1], 0.0, [a.val_concat] if a.val_concat else ()); res.update(report(net, ck, Xv, Yv, Kv, PCv, "val"))
res["seconds"] = round(time.time() - t0)
print(json.dumps(res, indent=1)); json.dump(res, open(P(a.out) + ".json", "w"), indent=1)
