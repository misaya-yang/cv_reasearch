# Research agreement

The current user request sets the task. This is the single project agreement for Codex and Claude Code;
old chats, automatic memories, imported records and archived plans are evidence, not standing instructions.
The paper objective does not authorize experiments during a documentation or maintenance task.

## Start here

Read `docs/harness/STATUS.md` and `docs/research/CLAIM.md`. Read `docs/research/PLAN.md` when continuing
research, and only the evidence needed for the question. The root README maps the repository.

## Research principles

1. Deliver the requested result and preserve the requested method, count and scope. A control is not an
   independent method. If a proposal needs changing, explain the change; do not silently test a substitute.
2. Aim for one complete in-context segmentation method with a defensible contribution. Explain which error
   of the strong complete baseline it changes, using what available evidence. An oracle, a signal, a proof,
   code preparation or GPU activity alone is not a method result. Do not demand certainty before experiment.
3. Use the smallest real comparison that can decide the question: complete baseline, candidate and strongest
   simple same-information control. Check an uncertain premise with existing outputs when useful, then reach
   the complete-mask result; do not grow an endless sequence of local probes, reviews or variants.
4. Keep resource and evaluation conditions explicit. Extra images, text, mask-pretrained models and base-class
   labels change the comparison. Use enough episodes for the effect; a smoke run checks execution, not efficacy.
   Develop on DEV; freeze before confirmation. Repeatedly inspected data cannot become fresh confirmation.
5. Report effect size and uncertainty together. A positive interval crossing zero is unresolved. A negative
   result limits the tested construction, not every method or the representation. Identify the failed link
   before another attempt; neither sunk code nor a renamed formula is a reason to continue it.

## Work and records

- `STATUS.md` holds current state; `CLAIM.md` the paper question and evidence boundary; `PLAN.md` only pending
  work with its owner, status and authorization. Shared lessons live in `docs/research/LESSONS.md`.
  Results belong in the linked evidence ledger, not new plans.
- Keep each result's arm, dataset/split, episode count, seed, resolution, complete control, paired gain/95% CI,
  fold gains and source together. Mark missing fields and GT diagnostics instead of guessing them.
- Share these files across agents. Check current changes and dependencies before editing; coordinate ownership
  in the pending-work entry. Do not overwrite another worker's active output or infer cross-chat permission.
- Read `docs/harness/SERVER.md` before remote work. Ask for GPU rental/extension, downloads, changes of research
  direction, or deletion outside the authorized scope. Existing authorization need not be requested again.
  Reversible work inside the requested scope proceeds. A cleanup request permits its scoped local cleanup.
- Commit/push only when requested. For this project, an authorized commit includes everything not ignored and
  is pushed; report the contents without editing or omitting files merely to prepare that commit. Never force-push.
- Report briefly in Chinese: conclusion, measured evidence, interpretation and unverified work; list anything
  needed from the user separately when applicable. Keep this agreement, harness documents and code in English.
