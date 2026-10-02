## 2026-10-02T08:37:57Z
Sender: 7fe97960-d0b9-46be-a8bc-5305ae36c0d8 (parent)
Priority: MESSAGE_PRIORITY_HIGH

You are a teamwork_preview_explorer. Your working directory is: /Users/yang/projects/CVPR2027/.agents/teamwork/teamwork_preview_explorer_survey_3.
MANDATORY: You MUST read the authoritative user request at: /Users/yang/projects/CVPR2027/.agents/teamwork/ORIGINAL_REQUEST.md before doing anything else.

Objective:
Perform a deep survey of the theoretical formulation requirements (R3) and experimental verification protocol requirements (R4):
1. Inspect and analyze:
   - Requirements R3 and R4 in ORIGINAL_REQUEST.md.
   - Mathematical structures in /Users/yang/projects/CVPR2027/demo_lists/research_decision_20261001/HANDOFF.md and mass_decision.py.
   - Requirements for self-contained Python numerical verification suite: >=1000 test cases, 0 GPU requirement, rigorous test coverage (pathological distributions, adversarial cases, numerical stability).
   - Formal theoretical proof requirements: notation table, explicit assumptions without hidden independence, lemmas, main theorems, local-global consistency interval convergence, per-image/class IoU regret strict upper bounds, information gain guarantees.
   - Experimental protocol requirements: benchmark alignment (eliminating COCO-20i historical biases), fair comparison matrix vs FoRIS/INSID3/FROST, fresh800 evaluation pipeline, low-cost probe kill criteria, compute budget matching, paired bootstrap CI calculation.
2. Synthesize:
   - Blueprint for the mathematical derivation document.
   - Architecture and test plan for the Python numerical verification suite (>=1000 cases).
   - Experimental protocol specification and kill criteria formulation.
3. Output: Write a comprehensive, structured report to:
   /Users/yang/projects/CVPR2027/.agents/teamwork/teamwork_preview_explorer_survey_3/survey_theory_eval_report.md
   and write a standard handoff.md in your working directory.
4. When complete, send a message to parent with the summary and path to your report. Do not modify any codebase files.
