"""Ideal-observation diagnostic for mass_decision; NOT a deployable result.

An observation reveals an atom's TRUE foreground mass. This measures the value
of additional information and checks the controller independently of a vision
verifier. It must never be presented as actual segmentation performance.
"""
import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import torch
from mass_decision import MassDecision, common_atoms, iou_from_masses

torch.set_num_threads(2)
BUDGETS = (0, 1, 2, 4, 8, 12)


def candidates(t, count=12):
    n = len(t["g"])
    m = t["r_mask"]
    nt, nr = t["nn_t"].astype(int), t["nn_r"].astype(int)
    x = np.zeros((3, 2 * n - 1))
    x[:, :n] = np.stack([np.ones(n), m[nt], np.bincount(nr[m], minlength=n)])
    for k, (left, right) in enumerate(t["children"]):
        x[:, n + k] = x[:, left] + x[:, right]
    precision = x[1] / x[0]
    recall = x[2] / max(m.sum(), 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    top = np.argsort(-f1)[:count]
    masks = []
    for node in top:
        mask = np.zeros(n, dtype=bool)
        stack = [int(node)]
        while stack:
            v = stack.pop()
            if v < n:
                mask[v] = True
            else:
                stack.extend(t["children"][v - n])
        masks.append(mask)
    masks.append(t["insid3"].astype(bool))
    return np.stack(masks)


def run_policy(membership, areas, true_mass, policy, seed):
    initial = membership[0].astype(float) * areas
    engine = MassDecision(membership, areas, initial, np.zeros_like(areas), areas)
    rng = np.random.default_rng(seed)
    history, queried = {}, []
    for spent in range(max(BUDGETS) + 1):
        if spent in BUDGETS:
            certified, gap = engine.certificate()
            pick = certified if gap <= .01 else engine.recommendation()
            history[spent] = {"pick": pick, "certificate_gap": gap, "queries": len(queried)}
        if spent == max(BUDGETS):
            break
        unseen = np.flatnonzero(engine.upper - engine.lower > 1e-10)
        if len(unseen) == 0:
            continue
        if policy == "decision_gap":
            atom = engine.next_atom()
        elif policy == "largest_area":
            atom = int(unseen[np.argmax(areas[unseen])])
        else:
            atom = int(rng.choice(unseen))
        # Oracle access is restricted to the action already chosen above.
        answer = float(true_mass[atom])
        engine.observe(atom, answer, answer, answer)
        queried.append(atom)
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows, totals, means, certs = [], {}, {}, {}
    def record(key, cls, intersection, union):
        value = totals.setdefault(key, {}).setdefault(cls, [0., 0.])
        value[0] += intersection
        value[1] += union
        means.setdefault(key, []).append(intersection / max(union, 1e-9))
    for fold in range(4):
        tables = torch.load(args.cache / f"l3b_f{fold}.l3.pt", map_location="cpu", weights_only=False)
        for t in tables:
            masks = candidates(t)
            membership, areas, labels = common_atoms(masks)
            # Used only by the oracle and evaluator; not provided to the controller.
            gt = t["g"].astype(float)
            true_mass = np.bincount(labels, weights=gt, minlength=len(areas))
            quality = iou_from_masses(membership, areas, true_mass)
            def counts(pick):
                inter = float(membership[pick] @ true_mass)
                union = float(membership[pick] @ areas + (~membership[pick]) @ true_mass)
                return inter, union
            for key, pick in (("insid3", len(masks)-1), ("F1", 0), ("candidate_oracle", int(quality.argmax()))):
                record(key, int(t["c"]), *counts(pick))
            result = {"fold": fold, "episode": int(t["e"]), "class": int(t["c"]), "atoms": len(areas)}
            for policy in ("decision_gap", "largest_area", "random"):
                history = run_policy(membership, areas, true_mass, policy, seed=fold*10000+int(t["e"]))
                result[policy] = history
                for budget, state in history.items():
                    key = f"{policy}/{budget}"
                    record(key, int(t["c"]), *counts(state["pick"]))
                    certs.setdefault(key, []).append(state["certificate_gap"] <= .01)
                    if state["certificate_gap"] <= .01:
                        assert quality.max() - quality[state["pick"]] <= .01 + 1e-9
            rows.append(result)
        print("fold", fold, "complete; episodes", len(rows), flush=True)
    summary = {key: {"class_miou": 100 * float(np.mean([i/max(u,1e-9) for i,u in per_class.values()])),
                     "mean_episode_iou": 100 * float(np.mean(means[key])),
                     "certified_fraction": float(np.mean(certs[key])) if key in certs else None}
               for key, per_class in totals.items()}
    output = {"status": "ORACLE information-value diagnostic; true query masses revealed",
              "n": len(rows), "summary": summary, "rows": rows,
              "mean_atoms": float(np.mean([r["atoms"] for r in rows])), "gpu_used": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps({k:v for k,v in output.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
