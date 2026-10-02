"""Analysis only (cross-fitted on validation images): how accurate can a region-level decision get when the recogniser
sees several sources at once, with or without the segmenter's own label?
  python scripts/stack_report.py <segtag>
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wwd import *
tag = sys.argv[1]; C = 150
A = torch.load(f"{CACHE}/rf/ade20k_val_{tag}_dinov2l.pt"); B = torch.load(f"{CACHE}/rf/ade20k_val_{tag}_own.pt")
assert torch.equal(A["img"], B["img"]) and torch.equal(A["y"], B["y"])
kind, y, pc, img, area = A["kind"], A["y"], A["pc"], A["img"].long(), A["area"]
T = torch.zeros(len(y), C); T.scatter_add_(1, A["tk_cls"].clamp(min=0), A["tk_frac"] * (A["tk_cls"] >= 0)); T = T / T.sum(1, keepdim=True)
onehot = torch.zeros(len(y), C); onehot[pc >= 0, pc[pc >= 0]] = 1


def fit_predict(X, tr, te, seed=0):
    torch.manual_seed(seed); mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True) + 1e-6
    Xd, Td = ((X - mu) / sd).to(DEV), (T * 0.9 + 0.1 / C).to(DEV); net = make_mlp(X.shape[1], C).to(DEV); idx = tr.nonzero()[:, 0].to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); E = 30; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=E * (len(idx) // 512 + 1))
    for ep in range(E):
        net.train(); p = idx[torch.randperm(len(idx), device=DEV)]
        for i in range(0, len(p), 512):
            b = p[i:i + 512]; loss = -(Td[b] * torch.log_softmax(net(Xd[b]), 1)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad(): return net(Xd[te.to(DEV)]).argmax(1).cpu()


def run(name, X, train_kinds):
    pred = torch.full_like(y, -1); te_all = kind == 1
    for fold in (0, 1):
        tr = torch.isin(kind, torch.tensor(train_kinds, dtype=torch.uint8)) & (img % 2 != fold); te = te_all & (img % 2 == fold); pred[te] = fit_predict(X, tr, te)
    ok = (pred == y)[te_all].float(); sg = (pc == y)[te_all].float(); ar = area[te_all]
    s = f"{name:34s} rec_acc={float(ok.mean()):.3f} (seg {float(sg.mean()):.3f})  area-weighted {float((ok * ar).sum() / ar.sum()):.3f} (seg {float((sg * ar).sum() / ar.sum()):.3f})"
    for lo, hi in ((0, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 2)):
        m = (ar >= lo) & (ar < hi); s += f" | [{lo:g},{hi:g}) {float(ok[m].mean()):.3f}"
    print(s, flush=True)


a, b = A["x"].float(), B["x"].float(); la = area.clamp(min=1e-6).log()[:, None]
run("dinov2l (gt+pred)", a, [0, 1]); run("own (gt+pred)", b, [0, 1]); run("dinov2l+own (gt+pred)", torch.cat([a, b], 1), [0, 1])
run("dinov2l (pred)", a, [1]); run("dinov2l + seg label (pred)", torch.cat([a, onehot, la], 1), [1]); run("own + seg label (pred)", torch.cat([b, onehot, la], 1), [1])
run("dinov2l+own + seg label (pred)", torch.cat([a, b, onehot, la], 1), [1])
