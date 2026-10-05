# CVPR 2027 — single-reference segmentation

**No paper method claim is established yet.** We seek a complete inference method that improves on complete
FoRIS under the same frozen-DINOv3, single-reference information budget. The supervised-readout and SAM3
results belong to separate resource settings.

## Start here

Codex and Claude Code share [AGENTS.md](AGENTS.md) directly. Read [STATUS](docs/harness/STATUS.md), then
[CLAIM](docs/research/CLAIM.md). [PLAN](docs/research/PLAN.md) is the only pending-work list.
Project Claude auto memory remains disabled; shared lessons are in [LESSONS](docs/research/LESSONS.md).

## Layout

```
src/ics/        shared data/encoder, complete FoRIS entry, native basis and paired statistics
scripts/        run_foris.py and experiment_resource_guard.py
evidence/       recorded results and reports; no active experiment queues
  local/        local evidence; research_20261005 holds the prepared six-mechanism batch
  insid3/       historical INSID3 diagnostics
  dots-2026-10-05/  supplied research synthesis and portable cloud evidence
docs/
  research/     CLAIM, PLAN, LESSONS
  harness/      STATUS, SERVER, closed-direction history
```

The former demo4 was an INSID3 exploration; only its shared data/encoder helpers remain in code.
The former demo9 mixed unrelated experiments; those implementations are retired from the working tree.
The user deleted local demo8, and it has not been recreated. See [closed-direction history](docs/harness/ARCHIVE.md).

| Need | Entry |
|---|---|
| Current claim, controls and evidence limits | [CLAIM](docs/research/CLAIM.md) |
| Local results and failed constructions | [Local evidence ledger](evidence/local/RESULTS.md) |
| INSID3 diagnostic history | [INSID3 evidence](evidence/insid3/RESULTS.md) |
| Dots contribution/theory review | [Imported record](evidence/dots-2026-10-05/IMPORT.md) |
| What the retained code does | [Code and baseline use](scripts/README.md) |
| Existing server paths and resource policy | [SERVER](docs/harness/SERVER.md) |

Old source is available from Git/history or the recorded local backup, not copied into another active code
tree. Scientific evidence remains inspectable. A stored result or a historical plan is not authorization to run.

Current preparation: [six complete mechanisms, controls and finite GPU launch](evidence/local/research_20261005/README.md).
The user stopped CPU experiments and will enable GPU mode. The batch has no new accuracy result yet.
