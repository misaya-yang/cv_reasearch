"""W01: is the task extent a within-image quantity that the reference can calibrate?

Cross-image similarity is used only to rank (seeds).  Extent is read from the within-image
cosine affinity to the seeds, the same computation in both images:
    A_Q(q) = mean_s cos(q, q_s)        A_R(r) = mean_s cos(r, r_s)
On the reference the answer is known, so t_R = IoU-optimal level of A_R against M_R (legal).
Rule: cut A_Q at t_R.  Controls: best pooled constant, per-episode oracle (diagnostic),
the cross-image field c at minmax 0.5 and at its oracle level, complete FoRIS.

Pre-stated predictions: P1 oracle(A_Q) >= oracle(c) on whole-object domains; P2 corr(t_R, t*_Q)
>= 0.4; P3 natural(t_R) beats the best constant by >= 2 points on average.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import DEV

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rows = lab.subset(N) + lab.ext_subset(N)
DOMS = [d for d in lab.ALL_DOMAINS if any(b["dataset"] == d for b in rows)]
SPACES = ("raw", "deb")
CONST = np.round(np.arange(0.20, 0.96, 0.025), 3)


def soft_level(s, y):
    """Soft-IoU-optimal level of token scores s against soft coverage y. Returns (t, iou)."""
    order = torch.argsort(s, descending=True)
    inter = torch.cumsum(y[order], 0)
    union = torch.arange(1, s.numel() + 1, device=s.device, dtype=torch.float32) + y.sum() - inter
    k = int((inter / union).argmax())
    lo = s[order[min(k + 1, s.numel() - 1)]]
    return float(0.5 * (s[order[k]] + lo)), float(inter[k] / union[k])


M = {k: lab.Meter() for k in ["foris", "c_mm", "c_orc", "seedprec"]}
for sp in SPACES:
    for k in ["orc", "tR", "tR_rel", "tR_frac"]:
        M[f"{sp}_{k}"] = lab.Meter()
    for t in CONST:
        M[f"{sp}_const{t}"] = lab.Meter()
stat = {sp: {d: [] for d in DOMS} for sp in SPACES}
prec = {d: [] for d in DOMS}
t0 = time.time()
for kk, b in enumerate(rows):
    d = b["dataset"]
    gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
    ig = lab.ignore1024(b)
    keep = None if ig is None else ~ig.to(DEV)
    iu = lambda p: lab.iu(p if keep is None else p & keep, gt if keep is None else gt & keep)
    y = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
    yq = lab.coverage64(gt).flatten()
    R, Q = lab.unit(lab.feats(b, "reference")), lab.unit(lab.feats(b, "query"))
    r, q = lab.debias(R), lab.debias(Q)
    mu = 0.5 * (r.mean(0) + q.mean(0))
    rc, qc = lab.unit(r - mu), lab.unit(q - mu)
    sim = qc @ rc.T
    c = torch.softmax(sim / 0.05, dim=1) @ y
    M["foris"].add(b, *lab.iu(lab.foris_cli(b).to(DEV), gt) if keep is None else iu(lab.foris_cli(b).to(DEV)))
    cu = lab.up(c.reshape(64, 64))
    M["c_mm"].add(b, *iu(lab.minmax(cu) > 0.5))
    M["c_orc"].add(b, *lab.best_threshold_iou(cu if keep is None else cu.masked_fill(~keep, -1), gt if keep is None else gt & keep)[:2])
    # seeds: highest closed-set posterior among tokens the reference foreground explains best
    fgw = y / y.sum().clamp_min(1e-6)
    best = (sim * (y > 0.5)).max(1).values if (y > 0.5).any() else sim @ fgw
    seed_score = c * lab.minmax(best)
    sq = torch.topk(seed_score, K).indices
    prec[d].append(float(yq[sq].mean()))
    # reference seeds: the foreground tokens the query seeds correspond to
    back = (rc @ lab.unit(qc[sq].mean(0)))
    pure = y >= 0.9 if int((y >= 0.9).sum()) >= K else (y >= torch.topk(y, K).values[-1])
    sr = torch.topk(back.masked_fill(~pure, -2), K).indices
    for sp, (XR, XQ) in {"raw": (R, Q), "deb": (r, q)}.items():
        AQ, AR = (XQ @ XQ[sq].T).mean(1), (XR @ XR[sr].T).mean(1)
        tR, iouR = soft_level(AR, y)
        selfQ, selfR = float((XQ[sq] @ XQ[sq].T).mean()), float((XR[sr] @ XR[sr].T).mean())
        fracR = (tR - float(AR.min())) / max(float(AR.max() - AR.min()), 1e-6)
        U = lab.up(AQ.reshape(64, 64))
        io, uo, tq = lab.best_threshold_iou(U if keep is None else U.masked_fill(~keep, -2), gt if keep is None else gt & keep)
        M[f"{sp}_orc"].add(b, io, uo)
        M[f"{sp}_tR"].add(b, *iu(U > tR))
        M[f"{sp}_tR_rel"].add(b, *iu(U > tR * selfQ / max(selfR, 1e-6)))
        M[f"{sp}_tR_frac"].add(b, *iu(lab.minmax(U) > fracR))
        for t in CONST:
            M[f"{sp}_const{t}"].add(b, *iu(U > float(t)))
        stat[sp][d].append((tR, tq, iouR, float(y.mean()), float(yq.mean()), selfR, selfQ))
    lab.free()
    if kk % 200 == 0:
        print(kk, len(rows), round(time.time() - t0, 1), flush=True)


def line(tag, key):
    v = [M[key].miou(d) for d in DOMS]
    print(f"{tag:34s}" + "".join(f"{x:8.1f}" for x in v) + f"{np.mean(v):8.1f}")


print(f"\nclass-pooled mIoU, no CRF except FoRIS; seeds K={K}; n per domain: " + ", ".join(f"{d[:6]}={len(prec[d])}" for d in DOMS))
print(" " * 34 + "".join(f"{d[:7]:>8s}" for d in DOMS) + "    mean")
line("FoRIS complete (CRF)", "foris")
line("c | minmax 0.5", "c_mm")
line("c | oracle level", "c_orc")
print(f"{'seed precision (GT coverage)':34s}" + "".join(f"{100 * np.mean(prec[d]):8.1f}" for d in DOMS))
for sp in SPACES:
    print(f"--- within-image affinity, {sp} features")
    line("  oracle level (diagnostic)", f"{sp}_orc")
    line("  t_R (reference-calibrated)", f"{sp}_tR")
    line("  t_R x self-affinity ratio", f"{sp}_tR_rel")
    line("  reference range fraction", f"{sp}_tR_frac")
    best = max(CONST, key=lambda t: np.mean([M[f"{sp}_const{t}"].miou(d) for d in DOMS]))
    line(f"  best pooled constant ({best})", f"{sp}_const{best}")
    wst = max(CONST, key=lambda t: min(M[f"{sp}_const{t}"].miou(d) - M["foris"].miou(d) for d in DOMS))
    line(f"  best worst-domain constant ({wst})", f"{sp}_const{wst}")
    S = {d: np.array(stat[sp][d]) for d in DOMS}
    print("  corr(t_R, t*_Q)       " + " " * 12 + "".join(f"{np.corrcoef(S[d][:, 0], S[d][:, 1])[0, 1]:8.2f}" for d in DOMS))
    print("  corr(t*_Q, log areaQ) " + " " * 12 + "".join(f"{np.corrcoef(S[d][:, 1], np.log(S[d][:, 4] + 1e-5))[0, 1]:8.2f}" for d in DOMS))
    print("  corr(t_R, log areaR)  " + " " * 12 + "".join(f"{np.corrcoef(S[d][:, 0], np.log(S[d][:, 3] + 1e-5))[0, 1]:8.2f}" for d in DOMS))
    print("  median |t_R - t*_Q|   " + " " * 12 + "".join(f"{np.median(np.abs(S[d][:, 0] - S[d][:, 1])):8.3f}" for d in DOMS))
    print("  reference IoU at t_R  " + " " * 12 + "".join(f"{100 * np.mean(S[d][:, 2]):8.1f}" for d in DOMS))
    print("  mean t_R              " + " " * 12 + "".join(f"{np.mean(S[d][:, 0]):8.2f}" for d in DOMS))
    print("  mean t*_Q             " + " " * 12 + "".join(f"{np.mean(S[d][:, 1]):8.2f}" for d in DOMS))
    print("  corr(log areaR, log areaQ)" + " " * 8 + "".join(f"{np.corrcoef(np.log(S[d][:, 3] + 1e-5), np.log(S[d][:, 4] + 1e-5))[0, 1]:8.2f}" for d in DOMS))
