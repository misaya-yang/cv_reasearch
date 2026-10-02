"""The coverage factor in INSID3's aggregation (score = cross * intra * coverage > 0.2): what does it cost, what does removing it do?
(CPU; budget.py tables; IoU at feature resolution with soft ground truth)

  official        cross * intra * coverage            > thr      (coverage of the seed set to 1)
  candidate_gate  cross * intra * [coverage > 0]      > thr      (the single-factor intervention)
  no_gate         cross * intra                       > thr
Also: share of true foreground lying in non-seed clusters with coverage <= 0.2 (these can never be selected by the official rule),
and the true-positive / false-positive area that the intervention adds.

  python scripts/gate.py results/budget_coco_f0.tables.pt ...
"""
import sys, torch, numpy as np
def run(path):
    T = torch.load(path); acc = {}; locked = []; add_tp = add_fp = base_tp = base_fp = 0.0
    def add(name, t, s):
        a, p = t["area"].float(), t["pur"].float(); fg = a * p; s = s.float(); i = float((s * fg).sum()); u = float((s * a).sum() + ((1 - s) * fg).sum()); x = acc.setdefault(name, {}).setdefault(t["c"], [0.0, 0.0]); x[0] += i; x[1] += u
    for t in T:
        K = t["K"]; a, p = t["area"].float(), t["pur"].float(); fg = a * p; seed = t["seed"]
        if seed < 0:
            for n in ("official", "candidate_gate", "no_gate"): add(n, t, torch.zeros(K))
            for thr in (0.25, 0.3, 0.35, 0.4): add(f"candidate_gate thr {thr}", t, torch.zeros(K)); add(f"no_gate thr {thr}", t, torch.zeros(K))
            locked.append(1.0); continue
        aw = t["aw"].float().clone(); aw[seed] = 1.0; q = t["cross"].float() * t["intra"].float()
        off = q * aw > 0.2; cg = q * (aw > 0).float() > 0.2; ng = q > 0.2; add("official", t, off); add("candidate_gate", t, cg); add("no_gate", t, ng)
        for thr in (0.25, 0.3, 0.35, 0.4): add(f"candidate_gate thr {thr}", t, q * (aw > 0).float() > thr); add(f"no_gate thr {thr}", t, q > thr)
        lock = (aw <= 0.2); lock[seed] = False; locked.append(float(fg[lock].sum() / fg.sum().clamp(min=1e-6)))
        new = cg & ~off; add_tp += float(fg[new].sum()); add_fp += float((a - fg)[new].sum()); base_tp += float(fg[off].sum()); base_fp += float((a - fg)[off].sum())
        add("oracle", t, p > 0.5)
    m = {k: 100 * float(np.mean([i / max(u, 1e-6) for i, u in v.values()])) for k, v in acc.items()}
    m["_locked_fg_share"] = 100 * float(np.mean(locked)); m["_added_tp_over_base_tp"] = 100 * add_tp / base_tp; m["_added_fp_over_base_fp"] = 100 * add_fp / max(base_fp, 1e-6); m["_added_tp_share"] = 100 * add_tp / max(add_tp + add_fp, 1e-6)
    return m
R = [run(p) for p in sys.argv[1:]]
for k in R[0]: print(f"  {k:28s} " + "  ".join(f"{r[k]:7.2f}" for r in R) + f"   mean {np.mean([r[k] for r in R]):7.2f}")
