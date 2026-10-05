# AGENTS.md

The user's latest message overrides this file.

**Goal.** One method paper for CVPR 2027 (deadline 2026-11-16 AoE) that earns a solid accept: a method that
beats a CVPR 2026 oral (INSID3) and its best follow-ups in in-context segmentation, with reproducible
evidence. An analysis paper does not count.

**Read first.** `docs/harness/STATUS.md` (what is true now and the one next action), then the live direction's
`PLAN.md` (experiments not yet run). Before proposing an idea: the failure ledger in the direction's
`README.md`. Before the first `ssh`: `docs/harness/SERVER.md`.

## Research

1. **Error before mechanism.** Before writing code for an idea, show from saved outputs which decision of the
   strong host is wrong, what it costs in mIoU, and a test-time signal that separates those cases, with its
   measured separation. An oracle gap is not that signal. Signal unmeasured: measure it; do not build.
2. **One script to the first number.** A new idea gets one script and one run against the host and the
   same-information control. No interfaces, fixtures, proofs, review passes or variants before that number.
   Several methods means several independent methods, each with its own number.
3. **Enough episodes to see the effect.** Real effects here are 2 to 4 mIoU. Measured half-widths of the
   paired 95% interval: 40 episodes, 3 to 9; DEV241, 1.8; CONFIRM600, 1.5 against the same host and 3
   against a different host. Choose on DEV241, confirm once on CONFIRM600. No method probes on 40 episodes.
4. **What counts.** A method counts when it beats the strongest baseline of its column (training-free;
   fitted on base classes; built on a foundation segmenter) and the same-information control, on CONFIRM600,
   at original resolution, inside the complete pipeline. An added input (an image pool, a class name) is not
   a contribution. No tables and no extra benchmarks before a first place in some column is in sight.
5. **Reading a result.** Paired gain, 95% interval, gain per fold, episodes up and down; every number keeps
   its arm name. A positive gain whose interval crosses zero is unresolved, not a failure. A failure limits
   the construction and the setting that were tested and is not a ceiling: write one ledger row and name the
   link that failed.

## Run

1. The GPU is rented by the hour. Before queueing, state the total GPU cost and the column in which the
   result would be first. Prepare code and the queue in no-card mode. A queue starts with a smoke stage (the
   real pipeline on a few real episodes, minutes) and ends with `/usr/bin/shutdown`. A passed smoke is the
   acceptance test; the full stages follow at once.
2. Queues go through `scripts/experiment_resource_guard.py`: it waits for foreign GPU jobs and powers off.
3. The server and the repository are shared with other agents. Look at `nvidia-smi` first. Stop only your own
   processes, by PID. Leave the base Python environment and other agents' files, directories and caches alone.
4. Delete your own caches and raw outputs once the run they feed has been read. Keep small JSON and logs.

## Ask the user first

Renting a GPU, or keeping one on beyond the approved session; downloading models, weights or datasets;
deleting or overwriting what you did not create; committing or pushing; changing the research direction.
Everything else that is reversible and inside the plan proceeds without asking.

## Repository

- `docs/harness/`: `STATUS.md` (at most 40 lines: now, measured, unverified, needed from the user; replaced,
  never appended to), `SERVER.md`, `ARCHIVE.md` (closed directions).
- `demo_lists/<direction>/`: `README.md` (numbers with provenance, failure ledger), `HANDOFF.md` (the method,
  its reasoning, pitfalls), `PLAN.md` (only experiments not yet run: question, arms, data, command, cost, what
  each outcome changes; an entry is deleted once it is read), `scripts/`, the package, `results/`.
- When an experiment ends: its numbers go to the README and a failure gets one ledger row.
- A commit the user asks for takes everything git does not ignore, whoever wrote it, as one commit, and is
  pushed. `.gitignore` alone decides what stays out; no file is edited, deleted or left out to prepare a commit.
- Every number carries dataset, split, episode count, seed, control, interval and result file.
- This file states the project's rules; agents do not append to it. It, `docs/harness/` and code are in
  English.

## Report

Chinese, conclusion first, numbers with their controls and intervals, three tiers (measured, inferred,
unverified), what is needed from the user listed separately. No guessed numbers or odds. Never "it cannot be
done", and stopping is not offered as an option: give the measured obstacle and the next move.
