# Project: CVPR 2027 Solid-Accept Research Paper Plan & Verification

## Architecture & System Overview
The project establishes a premier-tier (CVPR 2027 Solid Accept / Oral competitive) research paper framework: **Multi-granularity Token-conditioned Active Perception & Global Mass-Decision Network (M-TAP & G-MDN)** for In-Context Dense Prediction and Visual Reasoning.

### Core Architecture Components:
1. **Multi-granularity Cross-View Token Attention (M-CTA)**:
   - Preserves 2D spatial arrangement and token-level correspondences from frozen visual foundation models (DINOv3-L).
   - Resolves the 0.45 (FG) vs 0.46 (BG) similarity overlap via support background keys, relative 2D coordinate embeddings, and contrastive affinity suppression.
   - Specifically parameterizes cross-view token affinities to estimate atom-level mass intervals $[L_k, U_k]$ and global load $\mu(C^c)$ for linear-fractional decision making.
2. **Global Foreground Mass & Quality Decision Engine (G-MDE)**:
   - Overcomes isolated crop external mass blindness by formulating exact global IoU via additive local foreground mass:
     $$J(C) = \frac{\mu(C)}{|C| + \mu(C^c)}$$
   - Applies the local incremental update rule: a region $D$ improves IoU if and only if $p_D > \frac{J}{1 + J}$.
   - Solves the linear-fractional robust subset programming problem in $O(K \log K)$ via sorted prefix scanning over atomic common refinements.
3. **Information-Theoretic Active Test-Time Scaling (IT-ATS)**:
   - Dynamically schedules fine-grained patch observations using a decision-gap acquisition heuristic conditioned on interval regret bound certificates.
   - Bounded by finite lattice stopping time and calibrated conformal empirical coverage under bounded distribution shift.
   - Evaluated against a mandatory Max-Area baseline to verify compute-allocation value.
4. **Rigorous Verification & Falsifiable Guardrail**:
   - 0-GPU self-contained Python numerical test suite verifying 100% of mathematical theorems across $\ge 1000$ test cases.
   - Dual-isolated fresh800 evaluation protocol with strictly aligned compute budgets and paired bootstrap confidence intervals.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | 2025-2026 Paradigm Shift & Literature Mapping | Systematic analysis of DINOv3, in-context segmentation, and test-time scaling across 22 papers | M1 | Survey 2 |
| F2 | SOTA Baseline Autopsy & Blind Spot Formulation | Deep theoretical & empirical dissection of INSID3, FoRIS, FROST, and REBASE | M1 | Survey 2 |
| F3 | Historical Negative Results Root Cause Diagnosis | Mathematical proof of scalar selector collapse and crop verifier external mass blindness | M1 | Survey 1 |
| F4 | 4 Fundamental Mechanism Differentiators | Clear, non-cosmetic formulation of 4 core mechanism departures from existing SOTA | M1 | Survey 2 |
| F5 | M-TAP & G-MDN System Architecture Specification | Full architectural blueprint, dataflow, tensor specs, and multi-granularity patch conditioning | M2 | Survey 1, 2 |
| F6 | Token-Conditioned Local-Global Interaction Engine | Formal interaction mechanics replacing unconditioned crop similarity with dense cross-view token grounding | M2 | Survey 1, 2 |
| F7 | Linear-Fractional Robust Subset Decision Programming | $O(K \log K)$ sorted prefix optimization for global IoU maximization over atom partitions | M2 | Survey 1, 3 |
| F8 | Adaptive Test-Time Compute Scaling Logic | Ambiguity-driven observation scheduling with strict latency/FLOPs budget allocation | M2 | Survey 2, 3 |
| F9 | Formal Problem Setting & Common Refinement Atoms | Rigorous measure space formulation with zero hidden pixel-independence assumptions | M3 | Survey 3 |
| F10 | Exact Algebraic Jaccard & 3 Core Lemmas Proofs | Full proofs for monotonic fraction bounding, mass decomposition, and $J/(1+J)$ threshold | M3 | Survey 1, 3 |
| F11 | 4 Main Theorems & Regret Bounds Proofs | Regret upper bound certificate $\le \Delta_\mathcal{C}$, perturbation Lipschitz bounds, and stopping convergence | M3 | Survey 1, 3 |
| F12 | Conformal Risk Calibration & $G=0$ Boundary Handling | CRC interval guarantees and absent-target safe fallback | M3 | Survey 3 |
| F13 | 0-GPU Python Numerical Verification Suite | Comprehensive verification program covering $\ge 1000$ test cases (standard, pathological, adversarial, stability) | M3 | Survey 1, 3 |
| F14 | Numerical Verification Execution & Log Preservation | Full execution logs demonstrating 100% test pass rate preserved in repository | M3 | Survey 3 |
| F15 | Elimination of Historical COCO-20i Benchmark Biases | Unified benchmark protocol fixing mask scales, multi-label fold leaks, and CRF confounders | M4 | Survey 3 |
| F16 | Fair SOTA Comparison Matrix under Matched Compute | Fair evaluation matrix vs FoRIS (512/1024+CRF), INSID3, FROST, REBASE under FLOPs/latency parity | M4 | Survey 1, 2, 3 |
| F17 | Fresh800 Dual-Isolation Evaluation Pipeline | Class- and image-isolated fresh800 evaluation protocol with seed 2040 and val image exclusion | M4 | Survey 1, 3 |
| F18 | Low-Cost 200-Episode Probe Kill Criteria | Falsifiable early-stopping / kill rule: $\text{CI}_{0.025}(\Delta) \le 0 \implies$ halt & fallback | M4 | Survey 3 |
| F19 | Paired Bootstrap CI & Compute Profiling Protocol | Cluster bootstrap ($B=1500$) calculation spec and detailed profiling matrix | M4 | Survey 1, 3 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Frontier Paradigm Analysis & Gap Formulation | Features F1, F2, F3, F4 -> docs/research/cvpr2027_frontier_and_gap.md | Survey | DONE |
| M2 | Core Methodological Innovation for CVPR 2027 | Features F5, F6, F7, F8 -> docs/research/cvpr2027_methodology_design.md | M1 | PLANNED |
| M3 | Theoretical Derivation & Numerical Verification Suite | Features F9, F10, F11, F12, F13, F14 -> docs/research/cvpr2027_theoretical_derivation.md, tests/verification/test_mass_decision_theory.py, and verification_results.json | M2 | PLANNED |
| M4 | Experimental Protocol & Falsifiable Verification Plan | Features F15, F16, F17, F18, F19 -> docs/research/cvpr2027_experimental_protocol.md | M2 | PLANNED |

---

## Interface Contracts
### M1 -> M2
- M1 established the 4 mechanism differentiators, the resolution of the 0.45 vs 0.46 similarity overlap via negative keys and relative positions, the structural failure of score gating, and the operational contract of mass intervals $[L_k, U_k]$ and global load $\mu(C^c)$.
### M2 -> M3
- M2 defines the formal decision engine, multi-granularity token interaction, and atomic common refinement; M3 provides the rigorous mathematical proofs, regret certificates, and the $\ge 1000$ cases verification suite.
### M2 & M3 -> M4
- M2 specifies latency/FLOPs requirements and inference scaling logic; M3 provides kill criteria formulas and regret bounds; M4 operationalizes them into the benchmark matrix, probe rules, and fresh800 pipeline.

---

## Code Layout
```
/Users/yang/projects/CVPR2027/
├── docs/
│   └── research/
│       ├── cvpr2027_frontier_and_gap.md          # Milestone 1 deliverable (DONE)
│       ├── cvpr2027_methodology_design.md        # Milestone 2 deliverable
│       ├── cvpr2027_theoretical_derivation.md   # Milestone 3 deliverable
│       └── cvpr2027_experimental_protocol.md     # Milestone 4 deliverable
├── tests/
│   └── verification/
│       ├── test_mass_decision_theory.py          # Milestone 3 verification suite (>=1000 cases, 0-GPU)
│       └── verification_results.json             # Execution output & validation log
├── demo_lists/
│   ├── demo8_local_verification/
│   │   ├── fresh800_seed2040_manifest.json       # Frozen fresh800 test set
│   │   └── foris_control.py                      # Replicated baseline control
│   └── research_decision_20261001/
│       ├── HANDOFF.md                            # Historical theory notes
│       └── mass_decision.py                      # Reference prototype
└── .agents/
    └── teamwork/                                 # Multi-agent coordination metadata
```
