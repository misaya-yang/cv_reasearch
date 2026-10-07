"""Does the reference BACKGROUND tell where the target stops? Token level (64x64), frozen DINOv3-L at 1024, positional debias.
Baseline B = stored RCG field cut at 0.5 (the full method). Probe = keep a B token only if its nearest reference token is foreground.
Run on PACO-Part / PASCAL-Part / SUIM packs and on COCO fresh600 (harm check). No fitting, no constants tuned."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from PIL import Image
B = "/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9"; O = B + "/outputs"; sys.path.insert(0, B + "/src")
from ics.data import TimmDINOv3
dev = "cuda"; net = TimmDINOv3().to(dev).eval()
basis = torch.load("/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt", map_location=dev)
if isinstance(basis, dict): print("basis keys", list(basis)); basis = [v for v in basis.values() if torch.is_tensor(v) and v.ndim == 2][0]
basis = basis.float(); basis = basis if basis.shape[0] == 1024 else basis.T; print("basis", tuple(basis.shape))
MEAN, STD = torch.tensor([.485, .456, .406], device=dev).view(3, 1, 1), torch.tensor([.229, .224, .225], device=dev).view(3, 1, 1)

@torch.no_grad()
def enc(path):
    x = torch.from_numpy(np.asarray(Image.open(path).convert("RGB").resize((1024, 1024), Image.BICUBIC))).to(dev).permute(2, 0, 1).float() / 255
    with torch.autocast("cuda", dtype=torch.bfloat16): t = net.get_intermediate_layers(((x - MEAN) / STD)[None], n=1, reshape=False)[0][0]
    t = F.normalize(t.float(), dim=1); return F.normalize(t - (t @ basis) @ basis.T, dim=1)
cov = lambda m: np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((64, 64), Image.BOX), np.float32).reshape(-1) / 255
unit = lambda x: (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)

def sets():
    for name, packs in (("PACO-Part", [f"paco_part_f{i}" for i in range(4)]), ("PASCAL-Part", ["pascal_part"]), ("SUIM", ["suim"])):
        E = []
        for p in packs:
            z = np.load(f"{O}/claude_rcg2_{p}/counts.npz"); rows = json.load(open(f"{O}/claude_rcg2_{p}/rows.json")); P = f"{O}/claude_packs/{p}"
            for i, r in enumerate(rows):
                E.append(dict(cls=p + ":" + str(r["c"]), ref=f"{P}/data/{r['support']}", qry=f"{P}/data/{r['query']}",
                              rm=lambda P=P, r=r: np.asarray(Image.open(f"{P}/ann/{r['support']}").convert("L")) > 0,
                              tm=lambda P=P, r=r: cov((np.asarray(Image.open(f"{P}/ann/{r['query']}").convert("L")) > 0).astype(np.float32)) > .5,
                              field=z["fields"][i].astype(np.float32).reshape(-1)))
        yield name, E
    Z = np.load(O + "/claude_order_fresh600/tokens.npz"); rows = json.load(open(O + "/claude_order_fresh600/rows.json")); C = "/root/demo4_cache/data/COCO2014/"
    yield "COCO fresh600", [dict(cls=str(r["c"]), ref=C + r["support"], qry=C + r["query"],
                                 rm=lambda r=r: np.asarray(Image.open(C + "annotations/" + r["support"][:-4] + ".png")) == r["c"] + 1,
                                 tm=lambda i=i: Z["truth"][i] > .5, field=Z["rcg"][i].astype(np.float32)) for i, r in enumerate(rows)]

def miou(I, U, cls): return np.mean([I[cls == c].sum() / max(U[cls == c].sum(), 1e-9) for c in np.unique(cls)]) * 100
for name, E in sets():
    R = {k: [] for k in ("B", "L4", "NN", "B&NN", "B&NN5")}; ED = []; cls = np.array([e["cls"] for e in E])
    for e in E:
        q, r = enc(e["qry"]), enc(e["ref"]); c = torch.from_numpy(cov(e["rm"]().astype(np.float32))).to(dev); t = torch.from_numpy(e["tm"]()).to(dev)
        fg = c >= .5; fg = fg if fg.any() else c >= c.max().clamp_min(1e-6); bg = c == 0
        if not bg.any(): bg = ~fg
        s = q @ r.T; dfg, dbg = s[:, fg].max(1).values, s[:, bg].max(1).values
        k = lambda m: s[:, m].topk(min(5, int(m.sum())), dim=1).values.mean(1)
        mu = F.normalize(r[fg].mean(0), dim=0); b = r[~fg]; hard = b[(b @ mu).topk(max(1, int(.2 * b.shape[0]))).indices].mean(0); n = F.normalize(F.normalize(hard, dim=0) - (F.normalize(hard, dim=0) @ mu) * mu, dim=0)
        Bm = torch.from_numpy(e["field"]).to(dev) > .5; nn = dfg > dbg; nn5 = k(fg) > k(bg)
        M = {"B": Bm, "L4": unit(unit(q @ mu) - .55 * unit(q @ n)) > .5, "NN": nn, "B&NN": Bm & nn, "B&NN5": Bm & nn5}
        for kk, m in M.items(): R[kk].append(((m & t).sum().item(), (m | t).sum().item()))
        d = Bm & ~nn; ED.append((int((d & t).sum()), int((d & ~t).sum()), int((Bm & ~t).sum()), int(t.sum())))
    R = {k: np.array(v, float) for k, v in R.items()}; ED = np.array(ED, float); rng = np.random.RandomState(0); N = len(E)
    print(f"\n{name}: n={N}, classes={len(np.unique(cls))}  (token-level class mIoU; B = stored RCG field cut 0.5)")
    for kk, v in R.items():
        g = [miou(v[ix, 0], v[ix, 1], cls[ix]) - miou(R['B'][ix, 0], R['B'][ix, 1], cls[ix]) for ix in (rng.randint(0, N, N) for _ in range(1000))]
        print(f"  {kk:6s} {miou(v[:, 0], v[:, 1], cls):6.2f}   vs B {miou(v[:, 0], v[:, 1], cls) - miou(R['B'][:, 0], R['B'][:, 1], cls):+6.2f} [{np.percentile(g, 2.5):+.2f}, {np.percentile(g, 97.5):+.2f}]")
    print(f"  tokens deleted from B by the NN rule: true {ED[:, 0].sum():.0f} ({ED[:, 0].sum() / ED[:, 3].sum() * 100:.1f}% of truth), false {ED[:, 1].sum():.0f} ({ED[:, 1].sum() / max(ED[:, 2].sum(), 1) * 100:.1f}% of B's false area); purity of deletion {ED[:, 1].sum() / max(ED[:, :2].sum(), 1) * 100:.0f}%")
