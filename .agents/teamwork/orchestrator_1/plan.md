# Plan — CVPR 2027 Paper Research Project

## Objective
Establish a complete, solid-accept quality CVPR 2027 research paper foundation and verification package meeting requirements R1-R4:
- R1: Frontier Paradigm Analysis & Gap Formulation
- R2: Core Methodological Innovation for CVPR 2027
- R3: Theoretical Derivation & Computational Verification
- R4: Experimental Protocol & Falsifiable Verification Plan

## Phase 0: Survey & Scoping (Current)
- Dispatch 3 Explorers in parallel:
  - Explorer 1 (Local Evidence & Codebase): Inspect RESEARCH_STATUS.md, HANDOFF.md, mass_decision.py, fresh800_seed2040_manifest.json, foris_control.py.
  - Explorer 2 (Literature & Baselines): Inspect exa-results/cvpr2027-direction-review-2026-10-02.csv, SOTA baselines (INSID3, FoRIS, FROST, DINOv3, test-time scaling).
  - Explorer 3 (Theoretical & Mathematical Foundation): Inspect theoretical foundations of mass decision, IoU regret bounds, information gain, and numerical verification requirements.
- Synthesize findings into `PROJECT.md` at project root with Feature Inventory, Milestones, and Interface Contracts.

## Phase 1: Execution of Milestones
- Execute Milestones with Explorer -> Worker -> Reviewers -> Challengers -> Auditor loop.
- Deliverables:
  - docs/research/cvpr2027_frontier_and_gap.md (R1)
  - docs/research/cvpr2027_methodology_design.md (R2)
  - docs/research/cvpr2027_theoretical_derivation.md & verification suite + logs (R3)
  - docs/research/cvpr2027_experimental_protocol.md (R4)
- Maintain strict integrity enforcement (zero tolerance for hardcoding or facades).

## Phase 2: Verification & Review Gate
- Run Challengers & Forensic Auditor.
- Consolidate handoff and report to parent orchestrator.
