"""Empirical Adversarial Stress Suite for M2 Methodology Design.

This harness tests:
1. Mathematical and tensor shape consistency in M-CTA and G-MDE.
2. Parameter count and FLOPs accounting consistency vs M2 Table 2.3.
3. Absent target (G=0) failure modes, division-by-zero, and lack of null candidate.
4. Atom partition scaling: testing whether K <= 2M holds for non-laminar proposals,
   and measuring atom fragmentation (connected components and bbox dispersion).
5. Tied challenger intervals and myopic sensitivity deadlock in IT-ATS.
"""

import json
import math
import sys
import numpy as np
from scipy import ndimage


def test_parameter_accounting():
    """Verify trainable parameter counts against M2 Table 2.3 claims."""
    # Table 2.3 claims:
    # W_Q, W_K: 524,288
    # W_V, W_O: 131,072
    # Relative Coord MLP: 321
    # Atomic Readout Head: 16,514
    # Total: 672,195 (~0.67M)

    w_q = 1024 * 256  # 262,144
    w_k = 1024 * 256  # 262,144
    w_v = 1024 * 256  # 262,144
    w_o = 256 * 256   # 65,536

    # Rel pos MLP: Linear(3, 64) -> GELU -> Linear(64, 1)
    mlp_layer1 = 3 * 64 + 64   # weights + bias = 256
    mlp_layer2 = 64 * 1 + 1    # weights + bias = 65
    mlp_total = mlp_layer1 + mlp_layer2  # 321

    # Atomic head: Linear(256, 64) -> GELU -> Linear(64, 2)
    head_layer1 = 256 * 64 + 64  # 16,448
    head_layer2 = 64 * 2 + 2     # 130
    head_total = head_layer1 + head_layer2  # 16,578
    head_nobias1 = (256 * 64) + (64 * 2 + 2)  # 16,514

    true_wv_wo = w_v + w_o  # 327,680
    table_wv_wo = 131072

    total_table_claimed = 672195
    total_true_shared_wk = (w_q + w_k) + true_wv_wo + mlp_total + head_total  # 868,867
    total_true_separate_wk_bg = total_true_shared_wk + (1024 * 256)  # 1,131,011

    results = {
        "claimed_table_total": total_table_claimed,
        "true_wv_wo": true_wv_wo,
        "table_wv_wo": table_wv_wo,
        "wv_wo_discrepancy": true_wv_wo - table_wv_wo,
        "head_total_with_bias": head_total,
        "head_claimed": 16514,
        "head_missing_bias": head_total - head_nobias1,
        "total_true_shared_wk_bg": total_true_shared_wk,
        "total_true_separate_wk_bg": total_true_separate_wk_bg,
        "undercount_shared": total_true_shared_wk - total_table_claimed,
        "undercount_separate": total_true_separate_wk_bg - total_table_claimed,
    }
    return results


def test_absent_target_failure():
    """Test absent target (G=0) behavior across G-MDE, bounds, and regret."""
    # When G=0: true mask is all False
    # Non-empty candidates will have true IoU = 0.0
    # Empty candidate (if exists) would have IoU = 1.0
    rng = np.random.default_rng(101)

    # 5 non-empty candidates (e.g. from hierarchical clustering)
    M = 5
    areas = np.array([1000.0, 2500.0, 5000.0, 8000.0, 12000.0])
    # Say candidates induce K=6 atoms
    K = 6
    atom_areas = np.array([500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0])
    membership = rng.integers(0, 2, (M, K)).astype(float)
    # Ensure each candidate has at least one atom
    membership[:, 0] = 1.0

    # True foreground mass mu is identically 0 for all atoms
    true_mu = np.zeros(K)
    G = float(true_mu.sum())

    # Case 1: Model has non-zero uncertainty [0, 0.2 * area]
    lower = np.zeros(K)
    upper = 0.1 * atom_areas  # small false positive noise in upper bound

    # Compute bounds
    cand_area = membership @ atom_areas
    lower_denom = cand_area + (1 - membership) @ upper
    upper_denom = cand_area + (1 - membership) @ lower
    lo = np.divide(membership @ lower, lower_denom, out=np.zeros(M), where=lower_denom > 0)
    hi = np.divide(membership @ upper, upper_denom, out=np.zeros(M), where=upper_denom > 0)

    # True IoUs
    true_num = membership @ true_mu
    true_denom = cand_area + (1 - membership) @ true_mu
    true_iou = np.divide(true_num, true_denom, out=np.zeros(M), where=true_denom > 0)

    # Incumbent selection
    incumbent = int(np.argmax(lo))  # lo is all 0.0! argmax picks 0
    regret_certificate = float(hi.max() - lo[incumbent])

    # True regret:
    # If the set C does not contain the empty mask, the best achievable IoU in C is 0.0!
    # BUT if the task evaluates open-set / negative rejection, ideal output is empty mask (IoU=1.0)
    # The actual IoU of the emitted candidate is 0.0, incurring true regret = 1.0!

    # Lipschitz bound check:
    estimate = 0.05 * atom_areas
    l1_mass_error = float(np.abs(estimate - true_mu).sum())

    lipschitz_bound_div_by_zero = False
    try:
        theoretical_regret_bound = 2.0 * l1_mass_error / G
    except ZeroDivisionError:
        lipschitz_bound_div_by_zero = True
        theoretical_regret_bound = float("inf")

    return {
        "G": G,
        "all_lower_bounds_zero": bool(np.all(lo == 0.0)),
        "arbitrary_argmax_index": incumbent,
        "regret_certificate_within_C": regret_certificate,
        "emitted_candidate_true_iou": float(true_iou[incumbent]),
        "lipschitz_bound_div_by_zero": lipschitz_bound_div_by_zero,
        "theoretical_regret_bound": theoretical_regret_bound,
    }


def test_atom_scaling_and_fragmentation(num_trials=200):
    """Test atom count K vs 2M claim and check spatial fragmentation."""
    rng = np.random.default_rng(2026)

    # We will test two proposal regimes:
    # 1. Pure hierarchical laminar trees (where K <= M+1)
    # 2. Multi-seed / promptable proposals (SAM-like boxes/ellipses), M in [8, 16, 24]

    records = []

    for M in [6, 10, 16, 24]:
        k_over_2m_violations = 0
        max_k = 0
        total_k = 0
        disconnected_atom_ratios = []
        bbox_inflation_ratios = []

        for _ in range(num_trials):
            masks = np.zeros((M, 256, 256), dtype=bool)
            # Generate overlapping proposals
            for m in range(M):
                cy, cx = rng.integers(40, 216, size=2)
                ry, rx = rng.integers(15, 60, size=2)
                yy, xx = np.ogrid[:256, :256]
                mask = ((yy - cy)**2 / ry**2 + (xx - cx)**2 / rx**2) <= 1.0
                masks[m] = mask

            flat = masks.reshape(M, -1).T
            signatures, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
            K = len(signatures)
            max_k = max(max_k, K)
            total_k += K
            if K > 2 * M:
                k_over_2m_violations += 1

            # Analyze spatial connectivity for a subset of trials
            if _ < 20:
                labels_2d = inverse.reshape(256, 256)
                num_disconnected = 0
                inflations = []
                for k in range(K):
                    atom_binary = (labels_2d == k)
                    area = counts[k]
                    labeled_cc, num_cc = ndimage.label(atom_binary)
                    if num_cc > 1:
                        num_disconnected += 1
                    # Bounding box area
                    rows = np.any(atom_binary, axis=1)
                    cols = np.any(atom_binary, axis=0)
                    if rows.any() and cols.any():
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        bbox_area = (ymax - ymin + 1) * (xmax - xmin + 1)
                        inflations.append(bbox_area / max(area, 1))
                disconnected_atom_ratios.append(num_disconnected / max(K, 1))
                if inflations:
                    bbox_inflation_ratios.append(float(np.mean(inflations)))

        records.append({
            "M": M,
            "2M_threshold": 2 * M,
            "trials": num_trials,
            "violations_count": k_over_2m_violations,
            "violation_rate": k_over_2m_violations / num_trials,
            "avg_K": total_k / num_trials,
            "max_K": max_k,
            "avg_disconnected_atom_ratio": float(np.mean(disconnected_atom_ratios)) if disconnected_atom_ratios else 0.0,
            "avg_bbox_inflation_ratio": float(np.mean(bbox_inflation_ratios)) if bbox_inflation_ratios else 1.0,
        })

    return records


def test_tied_challengers_myopic_deadlock(num_trials=500):
    """Test tied challengers causing IT-ATS decision sensitivity deadlock."""
    # Scenario: There are M candidates. Incumbent is b.
    # There are multiple tied leading challengers c_1, c_2, ... having identical J_hi.
    # The heuristic picks c = argmax_{m != b} J_hi, picking only c_1.
    # We measure how often updating the atom selected by next_atom FAILS to reduce
    # the global regret certificate max_m J_hi(C_m) - J_lo(b) because c_2 maintains the maximum.

    rng = np.random.default_rng(777)
    deadlock_count = 0
    total_tied_trials = 0

    for _ in range(num_trials):
        M = 8
        K = 12
        atom_areas = rng.uniform(100, 1000, K)

        # Design candidate membership such that candidates 1 and 2 share similar structure
        # but differ on distinct disjoint atoms
        membership = rng.integers(0, 2, (M, K)).astype(float)

        # Make candidates 1 and 2 share incumbent base but have disjoint disputed atoms
        membership[1] = membership[0].copy()
        membership[2] = membership[0].copy()
        membership[1, 1] = 1.0
        membership[1, 2] = 0.0
        membership[2, 1] = 0.0
        membership[2, 2] = 1.0
        atom_areas[1] = 300.0
        atom_areas[2] = 300.0

        lower = np.zeros(K)
        upper = 0.5 * atom_areas
        upper[1] = 200.0
        upper[2] = 200.0

        # Candidate areas
        c_areas = membership @ atom_areas
        lo_denom = c_areas + (1 - membership) @ upper
        hi_denom = c_areas + (1 - membership) @ lower
        lo = np.divide(membership @ lower, lo_denom, out=np.zeros(M), where=lo_denom > 0)
        hi = np.divide(membership @ upper, hi_denom, out=np.zeros(M), where=hi_denom > 0)

        # Check if challengers 1 and 2 have nearly tied J_hi
        incumbent = int(np.argmax(lo))
        hi_cand = hi.copy()
        hi_cand[incumbent] = -np.inf
        top_challenger = int(np.argmax(hi_cand))
        top_val = hi_cand[top_challenger]

        # Count tied challengers
        tied = np.where(np.abs(hi_cand - top_val) < 1e-4)[0]
        if len(tied) >= 2:
            total_tied_trials += 1
            init_delta_C = float(hi.max() - lo[incumbent])

            # Run the M2 next_atom heuristic:
            costs = np.ones(K)
            uncertain = (upper - lower) > 1e-10
            pair = membership[[incumbent, top_challenger]]
            scores = np.full(K, -np.inf)
            estimate = 0.5 * (lower + upper)

            for atom in np.flatnonzero(uncertain):
                low_est = estimate.copy()
                high_est = estimate.copy()
                low_est[atom] = lower[atom]
                high_est[atom] = upper[atom]

                # IoU from masses
                def calc_iou(m_pair, masses):
                    num = m_pair @ masses
                    den = m_pair @ atom_areas + (1 - m_pair) @ masses
                    return np.divide(num, den, out=np.zeros(len(m_pair)), where=den > 0)

                jl = calc_iou(pair, low_est)
                ju = calc_iou(pair, high_est)
                gap_swing = abs((ju[1] - ju[0]) - (jl[1] - jl[0]))
                scores[atom] = gap_swing / costs[atom]

            selected_atom = int(np.argmax(scores))

            # Now simulate targeted observation of selected_atom:
            # Oracle reveals true value or contracts interval by factor rho=0.2
            new_lower = lower.copy()
            new_upper = upper.copy()
            new_upper[selected_atom] = new_lower[selected_atom] + 0.2 * (upper[selected_atom] - lower[selected_atom])

            # Recompute certificate
            new_lo_denom = c_areas + (1 - membership) @ new_upper
            new_hi_denom = c_areas + (1 - membership) @ new_lower
            new_lo = np.divide(membership @ new_lower, new_lo_denom, out=np.zeros(M), where=new_lo_denom > 0)
            new_hi = np.divide(membership @ new_upper, new_hi_denom, out=np.zeros(M), where=new_hi_denom > 0)
            new_incumbent = int(np.argmax(new_lo))
            new_delta_C = float(new_hi.max() - new_lo[new_incumbent])

            # Check if global regret certificate Delta_C decreased:
            if abs(new_delta_C - init_delta_C) < 1e-6:
                deadlock_count += 1

    return {
        "total_tied_trials": total_tied_trials,
        "deadlock_count": deadlock_count,
        "deadlock_rate": deadlock_count / max(total_tied_trials, 1),
    }


def main():
    print("Running Empirical Adversarial Stress Suite for M2 Methodology...")
    res_params = test_parameter_accounting()
    print("\n--- Parameter Accounting Results ---")
    print(json.dumps(res_params, indent=2))

    res_absent = test_absent_target_failure()
    print("\n--- Absent Target (G=0) Results ---")
    print(json.dumps(res_absent, indent=2))

    res_atoms = test_atom_scaling_and_fragmentation(num_trials=200)
    print("\n--- Atom Scaling and Fragmentation Results ---")
    print(json.dumps(res_atoms, indent=2))

    res_tied = test_tied_challengers_myopic_deadlock(num_trials=500)
    print("\n--- Tied Challengers Deadlock Results ---")
    print(json.dumps(res_tied, indent=2))

    summary = {
        "parameter_accounting": res_params,
        "absent_target": res_absent,
        "atom_scaling": res_atoms,
        "tied_challengers": res_tied,
    }
    with open("tests/verification/m2_adversarial_stress_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved full results to tests/verification/m2_adversarial_stress_results.json")


if __name__ == "__main__":
    main()
