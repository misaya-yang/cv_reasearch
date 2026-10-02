# AGENTS.md

Rules for every agent in this repository. The user's latest instruction overrides this file.

**Goal.** One method paper for CVPR 2027 (deadline 2026-11-16 AoE) that earns a solid accept: stronger than a
CVPR 2026 oral and its best follow-ups in that oral's field, with independent, reproducible evidence. An
analysis paper does not count.

**Start here.** Read `docs/harness/STATUS.md` (what is live, who runs what), then the live direction's
`HANDOFF.md` and `PLAN.md`. Do all research work by `.claude/skills/measure-first-research/SKILL.md`.
Everything else is indexed in `docs/harness/INDEX.md`.

## Think

1. Measure before building. Size the problem on a strong baseline with ground-truth (oracle) experiments.
   Build only what the measurement says is worth at least 5 points.
2. Predict before running. Write four lines: the assumption tested, the predicted numbers, what a match
   means, what a mismatch refutes and how the method changes. No fourth line, no run.
3. Bracket every result with three rows: the baseline, the naive method using the same information, the
   oracle upper bound.
4. Believe only paired differences with an interval, on all folds. A subset, a single fold, a smoke test or an
   oracle number is never the method's score.
5. When a key test fails, stop the queue, find the cause in data already on disk, then propose one run.
6. When a stop rule is met, stop. Closed directions stay closed unless new measured evidence exists and the
   user agrees. Read `docs/harness/ARCHIVE.md` before proposing an idea.
7. Plans and proofs without a measurement are not progress. Test the premise with the cheapest run instead.

## Run

1. Prepare first. Before taking the GPU: code smoke-tested on about 10 samples, caches built, four lines
   written, the next job ready.
2. Latest user resource policy (2026-10-02): prepare code, CPU checks and the finite queue before GPU rental.
   Continue an immediately executable valuable next stage, or power off the AutoDL instance when the queue
   ends/fails. Do not keep a paid GPU on for coding, literature or planning, and do not manufacture load.
   The prepared resource guard interrupts only its own stalled jobs after 60 seconds of confirmed zero GPU
   activity; foreign GPU jobs must be protected. AutoDL's instance command is `/usr/bin/shutdown` (no `-h now`).
3. Get a signal in minutes: small sample, results streamed per unit. If the first few units show nothing, stop
   and say so. Complete multi-hour runs are for final tables only.
4. The server is shared. Check `nvidia-smi` and `STATUS.md` first, cap your GPU memory, run one job per agent,
   and do not repeat a matrix another agent is running.
5. Stop only your own processes, by PID. Do not touch the base Python environment, or another agent's
   directories and caches. Details and pitfalls: `docs/harness/SERVER.md`.
6. Clean as you go. Delete models, datasets, caches and raw outputs in the same work session unless they
   feed the paper. Keep small JSON and log files.
7. Ask the user before downloading models, weights or datasets. Gated, licence-bound or Google-Drive-only
   files come from the user.

## Write

1. One directory per direction under `demo_lists/`, with `README.md` (numbers), `HANDOFF.md` (method,
   reasoning, pitfalls) and `PLAN.md` (next experiments with commands and stop rules). Layout rules:
   `docs/harness/REPO.md`.
2. Every number carries its dataset, split, sample count, seed, control, interval, script and result file.
3. Write claims in three tiers: measured, inferred, unverified.
4. When a direction changes state, update `docs/harness/STATUS.md` in the same change. When it closes, add
   five lines to `ARCHIVE.md` and delete its directory.
5. No diaries, scratch folders, generated planning documents or third-party checkouts in the repository.
6. This file, `docs/harness/` and skills are in English. Commit only when the user asks.

## Report

Reply to the user in Chinese, conclusion first, with numbers and their controls. No result: say so. A failure:
say so and give the numbers. List what you need from the user separately. Give short progress notes while
working. Never answer "it cannot be done"; report the measured obstacle and the next move.
