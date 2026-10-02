# Orchestrator Soft Handoff — Generation 1 to Generation 2

**Date & Time**: 2026-10-02T09:47:00Z  
**Current Orchestrator Workspace**: `/Users/yang/projects/CVPR2027/.agents/teamwork/orchestrator_1`  
**Parent Conversation ID**: `52f1a343-4884-46c6-b7fb-4808ee541135`  
**Cumulative Spawn Count**: 17 (Succession Threshold: 16 reached)  

---

## 1. Milestone State
| Milestone | Name | Status | Key Deliverables & Artifacts |
|-----------|------|--------|------------------------------|
| Phase 0 | Survey & Global Scoping | **DONE** | `survey_local_report.md`, `survey_literature_report.md`, `survey_theory_eval_report.md`, `PROJECT.md` |
| Milestone 1 | Frontier Paradigm Analysis & Gap Formulation (R1) | **DONE** | `docs/research/cvpr2027_frontier_and_gap.md` (passed Gate Iteration 2: Reviewer APPROVE, Challenger APPROVE, Auditor CLEAN) |
| Milestone 2 | Core Methodological Innovation for CVPR 2027 (R2) | **IN-PROGRESS (Needs Revision)** | `docs/research/cvpr2027_methodology_design.md` authored by worker_m2_1 (733 lines); Reviewer APPROVE, Auditor CLEAN, Challenger REQUEST_CHANGES on 5 technical issues |
| Milestone 3 | Theoretical Derivation & Numerical Verification (R3) | **PLANNED** | Survey blueprint ready in `survey_theory_eval_report.md`; requires `docs/research/cvpr2027_theoretical_derivation.md` and `tests/verification/test_mass_decision_theory.py` (>=1000 cases, 0 GPU) |
| Milestone 4 | Experimental Protocol & Falsifiable Verification Plan (R4) | **PLANNED** | Protocol blueprint ready in `survey_theory_eval_report.md`; requires `docs/research/cvpr2027_experimental_protocol.md` |

---

## 2. Active Subagents
- **None running**: All 17 subagents spawned in Generation 1 have delivered their handoff reports and are idle.

---

## 3. Pending Decisions & Immediate Next Steps for Successor (Gen 2)

### Immediate Step 1: Milestone 2 Remediation (Iteration 4)
Spawn a Worker (`worker_m2_2`) to update `/Users/yang/projects/CVPR2027/docs/research/cvpr2027_methodology_design.md` addressing the 5 concrete issues documented in `/Users/yang/projects/CVPR2027/.agents/teamwork/teamwork_preview_challenger_m2_1/handoff.md`:
1. **Parameter Count Correction**: Correct Table 2.3 parameter accounting. Readout head is ~0.87M params (with shared $W_K$) or ~1.13M params (with separate $W_K^{\text{fg}}, W_K^{\text{bg}}$); include $W_V+W_O = 327,680$ and biases.
2. **Compute & Latency Per-Episode Clarity**: Clarify that single-image forward pass is ~305.3 GFLOPs, but per-episode (Support + Query) is ~609 GFLOPs. Decouple FoRIS latency properly: FoRIS 512 is 1,065.2s / 1,200 episodes (~887 ms/ep) and FoRIS 1024 is 3,223.9s / 1,200 episodes (~2,686 ms/ep) with DenseCRF.
3. **Absent Target ($G=0$) Robustness**: Add explicit null candidate $C_\emptyset$ with zero mass to $\mathcal{C}$. Condition Theorem 3's Lipschitz bound on $G > 0$ and specify safe fallback when unobserved load $\mu(C^c) \le \delta_{\text{empty}}$.
4. **Atom Cardinality & Spatial Disconnection**: Clarify that $K \le 2M$ applies strictly to laminar hierarchical tree proposals; general proposals induce $K$ up to ~300. Handle spatial disconnection via connected-component sub-atom decomposition.
5. **IT-ATS Tied Challenger Min-Margin**: Update the sensitivity acquisition heuristic to use a tie-aware min-margin gap across top challengers ($\text{gap}(A_k) = \min_{C' \in \mathcal{C}_{\text{top}}} \Delta(C^*, C')$) to prevent deadlock.

### Immediate Step 2: Milestone 2 Gate Re-check
Verify `docs/research/cvpr2027_methodology_design.md` with Reviewer, Challenger, and Auditor. Upon all-pass, update `PROJECT.md` Milestone 2 to **DONE**.

### Immediate Step 3: Milestone 3 Execution (R3)
- Dispatch Worker to author `/Users/yang/projects/CVPR2027/docs/research/cvpr2027_theoretical_derivation.md` with:
  - Formal measure-theoretic setting on canonical common refinement atoms (zero pixel-independence assumption).
  - Algebraic Jaccard identity, Lemma 1 (monotonic fraction), Lemma 2 (mass decomposition), Lemma 3 ($J/(1+J)$ local threshold).
  - Theorem 1 (per-image regret certificate $\le \Delta_{\mathcal{C}}$), Theorem 2 ($O(K \log K)$ linear-fractional subset programming), Theorem 3 (Lipschitz bound $\le 2e/G$), Theorem 4 (geometric contraction / finite lattice stopping).
  - Conformal risk calibration on unseen classes under bounded Wasserstein drift.
- Dispatch Test Writer / Worker to implement self-contained Python numerical test suite at:
  `/Users/yang/projects/CVPR2027/tests/verification/test_mass_decision_theory.py`
  Must cover $\ge 1,000$ test cases across 4 groups (Standard Stochastic, Pathological Distributions, Adversarial Edge Cases, Numerical Stability) with 0 GPU requirement.
- Execute test suite, ensure 100% pass, and save output logs to `/Users/yang/projects/CVPR2027/tests/verification/verification_results.json`.
- Gate-check Milestone 3.

### Immediate Step 4: Milestone 4 Execution (R4)
- Dispatch Worker to author `/Users/yang/projects/CVPR2027/docs/research/cvpr2027_experimental_protocol.md` with:
  - Unified COCO-20i benchmark eliminating historical scale/leak/CRF biases.
  - Fair comparison matrix vs FoRIS (512/1024+CRF), INSID3, FROST, REBASE under FLOPs/latency parity.
  - Fresh800 dual-isolation evaluation pipeline (seed 2040, 80 classes, 1600 images, strictly excluding 40,504 val images).
  - Low-cost 200-episode probe kill criteria ($\text{CI}_{0.025}(\Delta) \le 0 \implies$ halt & fallback).
  - Paired Bootstrap CI calculation protocol ($B=1500$).
- Gate-check Milestone 4.

### Immediate Step 5: Final Verification & Completion Report
- Dispatch Challengers & Forensic Auditor across the complete research package.
- Send detailed completion report to Parent (`52f1a343-4884-46c6-b7fb-4808ee541135`).

---

## 4. Key Artifacts Index
- `/Users/yang/projects/CVPR2027/.agents/teamwork/ORIGINAL_REQUEST.md` — Authoritative User Request
- `/Users/yang/projects/CVPR2027/PROJECT.md` — Authoritative Project Plan & Feature Inventory
- `/Users/yang/projects/CVPR2027/docs/research/cvpr2027_frontier_and_gap.md` — Milestone 1 Deliverable (Approved & Verified)
- `/Users/yang/projects/CVPR2027/docs/research/cvpr2027_methodology_design.md` — Milestone 2 Deliverable (Drafted, Needs Challenger Remediation)
- `/Users/yang/projects/CVPR2027/.agents/teamwork/teamwork_preview_challenger_m2_1/handoff.md` — Challenger report with exact 5 fixes
- `/Users/yang/projects/CVPR2027/.agents/teamwork/orchestrator_1/GATE_STATUS.md` — Gate verdicts history
- `/Users/yang/projects/CVPR2027/.agents/teamwork/orchestrator_1/progress.md` — Progress tracker
- `/Users/yang/projects/CVPR2027/.agents/teamwork/orchestrator_1/BRIEFING.md` — Orchestrator memory & identity
