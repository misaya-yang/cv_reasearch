"""Inverse prediction as extent selector: does a query mask hypothesis explain the reference mask?

Ranking is held fixed (s0 = jointly centred prototype contrast). For a nested family of query
masks S_t we predict the reference labels from (Q, S_t) with two-sided query-prototype
competition and score J(t) = IoU(pred_R, M_R). The legal rule picks argmax_t J(t).
Diagnostics: J for the GT query mask and the selected-vs-oracle extent IoU.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import lab
from lab import *
from lab import kmeans, minmax

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
rows = lab.subset(N)
QS = torch.tensor([0.5, 0.6, 0.7, 0.76, 0.81, 0.85, 0.88, 0.905, 0.925, 0.94, 0.953, 0.963, 0.971, 0.978, 0.984, 0.989, 0.993, 0.996, 0.998])


def reverse_iou(qc, rc, fgq, cr, KF, KB):
    """Predict reference labels from the query hypothesis; token-level soft IoU with M_R."""
    if fgq.sum() < 1 or (~fgq).sum() < 1:
        return 0.0
    PF, PB = kmeans(qc, KF, iters=4, w=fgq.float()), kmeans(qc, KB, iters=4, w=(~fgq).float())
    pred = ((rc @ PF.T).max(1).values - (rc @ PB.T).max(1).values) > 0
    inter = (pred.float() * cr).sum()
    return float(inter / (pred.float().sum() + cr.sum() - inter).clamp_min(1e-6))


if __name__ == "__main__":
    names = ["minmax.5", "oracle", "inv_1_1", "inv_4_16", "inv_2_32"]
    M = {k: Meter() for k in names}
    diag = lab.defaultdict(list)
    pb = lab.defaultdict(list)
    t0 = time.time()
    for k, b in enumerate(rows):
        d = b["dataset"]
        gt = lab.mask1024(b["row"]["query_mask_path"]).to(DEV)
        cr = lab.coverage64(lab.mask1024(b["row"]["reference_mask_path"]).to(DEV)).flatten()
        r, q = lab.debias(lab.feats(b, "reference")), lab.debias(lab.feats(b, "query"))
        w = cr if cr.sum() > 1e-6 else (cr >= cr.max()).float()
        mu = 0.5 * (r.mean(0) + q.mean(0))
        rc, qc = lab.unit(r - mu), lab.unit(q - mu)
        s0 = qc @ lab.unit((r * w[:, None]).sum(0) / w.sum() - mu)
        S = up(s0.reshape(64, 64))
        M["minmax.5"].add(b, *iu(minmax(S) > 0.5, gt))
        i, u, _ = lab.best_threshold_iou(S, gt)
        M["oracle"].add(b, i, u)
        ts = torch.quantile(s0, QS.to(DEV))
        ious = torch.tensor([iu(S > t, gt) for t in ts]).float()
        ious = ious[:, 0] / ious[:, 1].clamp_min(1)
        for KF, KB in ((1, 1), (4, 16), (2, 32)):
            J = torch.tensor([reverse_iou(qc, rc, s0 > t, cr, KF, KB) for t in ts])
            j = int(J.argmax())
            M[f"inv_{KF}_{KB}"].add(b, *iu(S > ts[j], gt))
            if (KF, KB) == (4, 16):
                diag[d].append((float(J.max()), float(ious.max()), float(ious[j]),
                                float(np.corrcoef(J.numpy(), ious.numpy())[0, 1]) if J.std() > 0 and ious.std() > 0 else float("nan"),
                                reverse_iou(qc, rc, lab.coverage64(gt).flatten() > 0.5, cr, KF, KB)))
        yq = lab.coverage64(gt).flatten()
        simRQ = rc @ qc.T
        for tau in (0.02, 0.05, 0.1):
            pull = torch.softmax(simRQ / tau, dim=1) @ yq
            pb[(d, tau)].append((float((pull * cr).sum() / cr.sum().clamp_min(1e-6)),
                                 float((pull * (1 - cr)).sum() / (1 - cr).sum().clamp_min(1e-6)), float(yq.mean())))
        lab.free()
        if k % 150 == 0:
            print(k, round(time.time() - t0, 1), flush=True)

    print(f"\nclass-pooled mIoU (per-episode mean), fixed ranking s0, n={N}/domain, no CRF")
    print(f"{'extent rule':12s}" + "".join(f"{d:>18s}" for d in DOMAINS))
    for k in names:
        r = M[k].row()
        print(f"{k:12s}" + "".join(f"{r[d][0]:9.1f} ({r[d][1]:5.1f})" for d in DOMAINS))
    print("\ninv_4_16 diagnostics per domain: mean maxJ | corr(J, IoU) over thresholds | J of GT query mask")
    for d in DOMAINS:
        a = np.array(diag[d])
        print(f"{d:12s} maxJ {a[:,0].mean():.3f}  corr {np.nanmean(a[:,3]):+.3f}  J(GT_Q) {a[:,4].mean():.3f}")
    print("\npullback of the GT query mask through reference->query attention: mean on ref FG | ref BG | (query FG area)")
    for d in DOMAINS:
        print(f"{d:12s}" + "".join(f"  tau={tau}: {np.mean([x[0] for x in pb[(d, tau)]]):.3f} | {np.mean([x[1] for x in pb[(d, tau)]]):.3f}" for tau in (0.02, 0.05, 0.1)) + f"  | area {np.mean([x[2] for x in pb[(d, 0.02)]]):.3f}")
