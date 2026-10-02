## 2026-10-02T09:48:31Z
You are a teamwork_preview_worker. Your working directory is: /Users/yang/projects/CVPR2027/.agents/teamwork/worker_m2_2.
MANDATORY: You MUST read the authoritative user request at: /Users/yang/projects/CVPR2027/.agents/teamwork/ORIGINAL_REQUEST.md before doing anything else.

DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Context and Feedback:
The Milestone 2 deliverable (/Users/yang/projects/CVPR2027/docs/research/cvpr2027_methodology_design.md) was reviewed. Reviewers approved and Auditor confirmed CLEAN, but Challenger 2 issued REQUEST_CHANGES with 5 specific, rigorous, empirically verified defects. You must read the challenger report and test results at:
- /Users/yang/projects/CVPR2027/.agents/teamwork/teamwork_preview_challenger_m2_1/handoff.md
- /Users/yang/projects/CVPR2027/tests/verification/m2_adversarial_stress_results.json

File Ownership:
You exclusively own and may edit:
- /Users/yang/projects/CVPR2027/docs/research/cvpr2027_methodology_design.md
Do NOT modify any other files in the project.

Objective:
Revise /Users/yang/projects/CVPR2027/docs/research/cvpr2027_methodology_design.md to completely resolve all 5 issues:

1. **Parameter Count Defect**:
   - Correct Table 2.3 parameter accounting.
   - Accurately account for W_V in R^{D x d_v} and W_O in R^{d_v x D} (for D=1024, d_v=160, this is 2 * 1024 * 160 = 327,680 params, not 131,072).
   - Accurately include W_K^{bg} and layer norms/biases.
   - State the true parameter count: ~0.87M parameters (with shared key projection) or ~1.13M parameters (with dedicated foreground/background key projections).

2. **Compute & Latency Per-Episode Clarity**:
   - Clarify FLOPs: single query forward pass is ~305.3 GFLOPs; full episode (support image + query image) is ~609 GFLOPs.
   - Fairly explain FoRIS runtime numbers: FoRIS 512 total time of 1,065.2s across 1,200 episodes is ~887 ms/episode, and FoRIS 1024 total time of 3,223.9s across 1,200 episodes is ~2,686 ms/episode (including dataloading, GPU forward, and CPU DenseCRF on a shared server). Do not compare theoretical GPU forward FLOPs against total wall-clock time without explicit qualification.

3. **Absent Target (G=0) Failure**:
   - Add explicit null candidate C_\emptyset = \emptyset (with \mu(C_\emptyset)=0, |C_\emptyset|=0, J(C_\emptyset)=0) to candidate set C.
   - Add an empty-target guard in G-MDE: if total predicted foreground mass across the query image falls below a threshold \tau_{empty}, output C_\emptyset.
   - Explicitly condition Theorem 3's Lipschitz bound on G > 0, and formulate the boundary case for G = 0.

4. **Atom Scaling (K <= 2M) Falsified & Spatial Disconnection**:
   - Clarify that K <= 2M applies strictly to laminar / hierarchical tree proposals.
   - For general non-laminar proposals (SAM, multi-threshold seeds), the atom count scales as K <= min(2^M, |\Omega|), reaching up to ~300 in practice.
   - Formulate connected-component atom decomposition: since 21-25% of intersection atoms consist of multiple disconnected components, the local refinement and re-observation stage decomposes atoms into spatially connected components, preventing inflated bounding boxes.

5. **IT-ATS Tied Challenger Deadlock**:
   - Fix the myopic sensitivity heuristic to use a tie-aware min-margin gap across the top challenger set:
     \Delta_{\mathcal{C}}(A_k) = \min_{C' \in \mathcal{C}_{top} \setminus \{C^*\}} (J(C^*) - J(C')).
   - When multiple challengers tie, select the atom that maximizes joint margin separation across all tied contenders.

Completion Criteria:
- docs/research/cvpr2027_methodology_design.md is updated with all 5 remediations.
- Write handoff.md in your working directory (/Users/yang/projects/CVPR2027/.agents/teamwork/worker_m2_2/handoff.md).
- Send a message to parent when completed.
