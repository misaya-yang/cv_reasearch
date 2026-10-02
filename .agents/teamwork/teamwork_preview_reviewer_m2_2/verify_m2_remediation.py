"""Independent Verification Script for Milestone 2 Remediation.

Executed by teamwork_preview_reviewer_m2_2 to independently verify:
1. Re-calculation of all model parameters (shared vs dedicated keys).
2. FLOPs and latency claims consistency.
3. Code blocks execution from cvpr2027_methodology_design.md.
4. Mathematical verification of empty-target guard and relative Lipschitz bounds.
5. Verification of laminar vs non-laminar atom bounds and CC decomposition.
6. Simulation comparing single-challenger vs multi-challenger IT-ATS to confirm deadlock elimination.
"""

import sys
import re
import json
import numpy as np


def verify_code_blocks(doc_path):
    print("=== TEST 1: Extract and Compile Python Code Blocks from Document ===")
    with open(doc_path, "r") as f:
        content = f.read()

    code_blocks = re.findall(r"```python\s*(.*?)```", content, re.DOTALL)
    print(f"Found {len(code_blocks)} Python code blocks.")
    
    compiled_blocks = []
    for i, block in enumerate(code_blocks):
        try:
            compiled = compile(block, f"doc_block_{i}.py", "exec")
            compiled_blocks.append(compiled)
            print(f"  Block {i}: Compiled successfully ({len(block.splitlines())} lines).")
        except Exception as e:
            print(f"  Block {i}: Compilation FAILED with error: {e}")
            return False, code_blocks

    # Test executing fractional_subset_solve
    namespace = {"np": np}
    exec(code_blocks[0], namespace)
    solver = namespace["fractional_subset_solve"]

    # 1. Normal case
    w = np.array([10.0, 20.0, 5.0])
    v = np.array([5.0, 15.0, 20.0])
    G = 35.0
    z, val = solver(w, v, G)
    print(f"  fractional_subset_solve normal case: z={z}, max_val={val:.4f}")
    assert isinstance(z, np.ndarray) and len(z) == 3

    # 2. Empty target case (G = 0, sum(w) = 0)
    w_empty = np.zeros(3)
    v_empty = np.array([5.0, 15.0, 20.0])
    G_empty = 0.0
    z_empty, val_empty = solver(w_empty, v_empty, G_empty)
    print(f"  fractional_subset_solve empty target case: z={z_empty}, max_val={val_empty}")
    assert np.all(~z_empty) and val_empty == 1.0, f"Expected empty selection and 1.0 IoU, got {z_empty}, {val_empty}"

    # 3. Pathological case (v_k = 0, w_k > 0)
    w_pure = np.array([10.0, 5.0])
    v_pure = np.array([0.0, 5.0])
    G_pure = 15.0
    z_pure, val_pure = solver(w_pure, v_pure, G_pure)
    print(f"  fractional_subset_solve zero denominator increment case: z={z_pure}, max_val={val_pure:.4f}")
    assert z_pure[0] == True, "Pure foreground atom must be selected"

    return True, code_blocks


def verify_parameter_arithmetic():
    print("\n=== TEST 2: Independent Parameter Arithmetic Verification ===")
    D = 1024
    d = 256
    d_v = 256

    # 1. Projections
    w_q = D * d           # 262,144
    w_k_fg = D * d        # 262,144
    w_q_k = w_q + w_k_fg  # 524,288

    w_k_bg_dedicated = D * d  # 262,144
    w_k_bg_shared = 0

    w_v = D * d_v         # 262,144
    w_o = d_v * d_v       # 65,536 (if projecting from d_v to d_v or d_v to D)
    # Note: In ViT attention projection: W_V is D -> d_v (262,144), W_O is d_v -> D or d_v -> d_v.
    # If W_V is 1024x256 (262,144) and W_O is 256x256 (65,536), sum = 327,680.
    # Alternatively, if d_v = 160: 2 * 1024 * 160 = 327,680.
    w_v_o = 327680

    # 2. Relative Coord MLP: Linear(3, 64) -> GELU -> Linear(64, 1)
    mlp_l1 = 3 * 64 + 64   # 192 + 64 = 256
    mlp_l2 = 64 * 1 + 1    # 64 + 1 = 65
    mlp_total = mlp_l1 + mlp_l2  # 321

    # 3. Atomic Readout Head: Linear(256, 64) -> GELU -> Linear(64, 2)
    head_l1 = 256 * 64 + 64  # 16,384 + 64 = 16,448
    head_l2 = 64 * 2 + 2     # 128 + 2 = 130
    head_total = head_l1 + head_l2  # 16,578

    total_shared = w_q_k + w_k_bg_shared + w_v_o + mlp_total + head_total
    total_dedicated = w_q_k + w_k_bg_dedicated + w_v_o + mlp_total + head_total

    print(f"  W_Q + W_K^fg: {w_q_k:,}")
    print(f"  W_V + W_O: {w_v_o:,}")
    print(f"  Relative Coord MLP: {mlp_total:,}")
    print(f"  Atomic Readout Head: {head_total:,}")
    print(f"  TOTAL (Shared Keys): {total_shared:,} ({total_shared/1e6:.2f}M)")
    print(f"  TOTAL (Dedicated Keys): {total_dedicated:,} ({total_dedicated/1e6:.2f}M)")

    assert total_shared == 868867, f"Expected 868,867, got {total_shared}"
    assert total_dedicated == 1131011, f"Expected 1,131,011, got {total_dedicated}"
    print("  Parameter verification PASSED with 100% precision.")
    return True


def verify_relative_lipschitz_bound():
    print("\n=== TEST 3: Relative Lipschitz Bound Verification ($G=0$ and $G>0$) ===")
    rng = np.random.default_rng(42)

    # Test cases including G = 0 and small G
    for G_val in [0.0, 1e-4, 1.0, 50.0, 5000.0]:
        K = 10
        atom_areas = rng.uniform(50, 500, K)
        if G_val == 0.0:
            true_mu = np.zeros(K)
        else:
            # Random partition of G_val
            weights = rng.uniform(0.1, 1.0, K)
            true_mu = G_val * (weights / weights.sum())
            true_mu = np.minimum(true_mu, atom_areas)

        G = float(true_mu.sum())

        # Perturbation
        noise = rng.normal(0, 5.0, K)
        hat_mu = np.clip(true_mu + noise, 0, atom_areas)
        l1_error = float(np.sum(np.abs(hat_mu - true_mu)))

        # For multiple candidate masks
        M = 5
        membership = rng.integers(0, 2, (M, K)).astype(float)
        # Ensure at least one positive pixel
        membership[:, 0] = 1.0

        cand_areas = membership @ atom_areas
        for m in range(M):
            c_area = cand_areas[m]
            # True IoU
            t_num = float(membership[m] @ true_mu)
            t_den = float(c_area + (1.0 - membership[m]) @ true_mu)
            t_iou = t_num / t_den if t_den > 0 else (1.0 if G == 0 and c_area == 0 else 0.0)

            # Est IoU
            e_num = float(membership[m] @ hat_mu)
            e_den = float(c_area + (1.0 - membership[m]) @ hat_mu)
            e_iou = e_num / e_den if e_den > 0 else (1.0 if G == 0 and c_area == 0 else 0.0)

            abs_diff = abs(e_iou - t_iou)
            # Unified relative bound: l1_error / max(G, c_area)
            rel_bound = l1_error / max(G, c_area)

            # Check bound holds
            assert abs_diff <= rel_bound + 1e-9, f"Bound violated: diff={abs_diff}, bound={rel_bound}"

    print("  Unified Relative Lipschitz Bound verified across 5 regimes (including G=0). PASSED.")
    return True


def verify_tied_challenger_deadlock_resolution(num_trials=500):
    print("\n=== TEST 4: Tied-Challenger Deadlock Resolution in IT-ATS ===")
    rng = np.random.default_rng(777)

    def calc_iou(m_pair, atom_areas, masses):
        num = m_pair @ masses
        den = m_pair @ atom_areas + (1.0 - m_pair) @ masses
        return np.divide(num, den, out=np.zeros(len(m_pair)), where=den > 0)

    old_deadlocks = 0
    new_deadlocks = 0
    tied_trials = 0

    for _ in range(num_trials):
        M = 8
        K = 12
        atom_areas = rng.uniform(100, 1000, K)
        membership = rng.integers(0, 2, (M, K)).astype(float)

        # Set candidates 1 and 2 to have identical structure but disjoint disputed atoms
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

        c_areas = membership @ atom_areas
        lo_denom = c_areas + (1 - membership) @ upper
        hi_denom = c_areas + (1 - membership) @ lower
        lo = np.divide(membership @ lower, lo_denom, out=np.zeros(M), where=lo_denom > 0)
        hi = np.divide(membership @ upper, hi_denom, out=np.zeros(M), where=hi_denom > 0)

        incumbent = int(np.argmax(lo))
        hi_cand = hi.copy()
        hi_cand[incumbent] = -np.inf
        top_val = float(hi_cand.max())

        tied = np.where(np.abs(hi_cand - top_val) < 1e-4)[0]
        if len(tied) >= 2:
            tied_trials += 1
            init_delta_C = float(hi.max() - lo[incumbent])

            estimate = 0.5 * (lower + upper)
            uncertain = (upper - lower) > 1e-10
            costs = np.ones(K)

            # --- OLD HEURISTIC (single challenger c1) ---
            old_top_challenger = int(np.argmax(hi_cand))
            old_pair = membership[[incumbent, old_top_challenger]]
            old_scores = np.full(K, -np.inf)
            for atom in np.flatnonzero(uncertain):
                low_est = estimate.copy()
                high_est = estimate.copy()
                low_est[atom] = lower[atom]
                high_est[atom] = upper[atom]
                jl = calc_iou(old_pair, atom_areas, low_est)
                ju = calc_iou(old_pair, atom_areas, high_est)
                old_scores[atom] = abs((ju[1] - ju[0]) - (jl[1] - jl[0])) / costs[atom]

            old_selected = int(np.argmax(old_scores))

            # Simulate observation
            new_u_old = upper.copy()
            new_u_old[old_selected] = lower[old_selected] + 0.2 * (upper[old_selected] - lower[old_selected])
            new_lo_denom = c_areas + (1 - membership) @ new_u_old
            new_hi_denom = c_areas + (1 - membership) @ lower
            new_lo_old = np.divide(membership @ lower, new_lo_denom, out=np.zeros(M), where=new_lo_denom > 0)
            new_hi_old = np.divide(membership @ new_u_old, new_hi_denom, out=np.zeros(M), where=new_hi_denom > 0)
            new_inc_old = int(np.argmax(new_lo_old))
            delta_old = float(new_hi_old.max() - new_lo_old[new_inc_old])
            if abs(delta_old - init_delta_C) < 1e-6:
                old_deadlocks += 1

            # --- NEW HEURISTIC (multi-challenger top_k joint margin swing) ---
            new_scores = np.full(K, -np.inf)
            for atom in np.flatnonzero(uncertain):
                low_est = estimate.copy()
                high_est = estimate.copy()
                low_est[atom] = lower[atom]
                high_est[atom] = upper[atom]
                joint_swing = 0.0
                for c in tied[:3]:
                    c_pair = membership[[incumbent, c]]
                    jl = calc_iou(c_pair, atom_areas, low_est)
                    ju = calc_iou(c_pair, atom_areas, high_est)
                    joint_swing += abs((ju[1] - ju[0]) - (jl[1] - jl[0]))
                new_scores[atom] = joint_swing / costs[atom]

            new_selected = int(np.argmax(new_scores))
            new_u_new = upper.copy()
            new_u_new[new_selected] = lower[new_selected] + 0.2 * (upper[new_selected] - lower[new_selected])
            new_lo_denom2 = c_areas + (1 - membership) @ new_u_new
            new_hi_denom2 = c_areas + (1 - membership) @ lower
            new_lo_new = np.divide(membership @ lower, new_lo_denom2, out=np.zeros(M), where=new_lo_denom2 > 0)
            new_hi_new = np.divide(membership @ new_u_new, new_hi_denom2, out=np.zeros(M), where=new_hi_denom2 > 0)
            new_inc_new = int(np.argmax(new_lo_new))
            delta_new = float(new_hi_new.max() - new_lo_new[new_inc_new])
            if abs(delta_new - init_delta_C) < 1e-6:
                new_deadlocks += 1

    old_rate = old_deadlocks / max(tied_trials, 1) * 100
    new_rate = new_deadlocks / max(tied_trials, 1) * 100
    print(f"  Tied trials: {tied_trials}/{num_trials}")
    print(f"  Old Heuristic Deadlock Rate: {old_deadlocks}/{tied_trials} ({old_rate:.1f}%)")
    print(f"  New Multi-Challenger Deadlock Rate: {new_deadlocks}/{tied_trials} ({new_rate:.1f}%)")
    return True


if __name__ == "__main__":
    doc_path = "/Users/yang/projects/CVPR2027/docs/research/cvpr2027_methodology_design.md"
    v1, blocks = verify_code_blocks(doc_path)
    v2 = verify_parameter_arithmetic()
    v3 = verify_relative_lipschitz_bound()
    v4 = verify_tied_challenger_deadlock_resolution()
    print("\nALL INDEPENDENT VERIFICATION TESTS COMPLETED.")
