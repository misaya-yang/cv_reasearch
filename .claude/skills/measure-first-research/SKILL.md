---
name: measure-first-research
description: Working method for empirical ML/CV research that has to end in a publishable method. Covers choosing an arena, sizing the problem with oracle experiments, finding an unused information source, pre-registering predictions, minute-scale tests with strong controls, diagnosing failures, killing dead directions early, hardening a result, and handing work to another agent. Use when picking or judging a research direction, designing or running an experiment, reading a surprising or failed result, deciding whether to continue or stop, or writing a plan or hand-off.
---

# Measure-first research

Do not build a method and hope. Measure where a strong baseline loses its points and what information would
recover them; build only what the measurement says is worth building; predict the number before each run. Then
every result, including a failure, says what to do next.

The steps below are in order. Skipping one is how the failures in `references/worked-examples.md` happened.

## 1. Pick the arena before the idea

All four must hold:

- A strong recent paper (an oral, or widely followed) with public code. It proves the field cares and gives a
  baseline nobody can call weak. A method that scores 90, not 100, can be improved; the job is to find the 10.
- Its headline number reproduces on your hardware within a day.
- One evaluation of a change takes minutes, or can be made to by caching.
- Ground truth is available on the evaluation set, so oracle experiments are possible.

## 2. Reproduce, then make the loop fast

- Run the released code first. Record your number next to the paper's. Never rank your numbers against paper
  numbers; compare only runs made under one protocol on one machine.
- Cache everything frozen (features, clusterings, baseline outputs) so that a variant evaluates in seconds
  without the encoder.
- Prove the fast loop is exact: compare its outputs with the released code item by item (share of identical
  predictions), not only the mean metric. A mean that is "about the same" hides real differences.

## 3. Build the error ledger

For each decision the baseline makes, replace that one decision with ground truth and re-score. One row per
intervention: what was fixed, the metric, the gain. The ledger says where the points are and which way the
errors lean (misses or false alarms).

Then check that the largest entry is removable. Drop it if it is:

- an annotation convention (near-synonym or parent/child classes);
- evaluation noise (classes with a handful of images; know the bootstrap spread of the metric);
- already gone on a stronger model;
- absent in the benchmark (measure the premise itself: does the condition you want to fix cost anything?).

Leave the arena if the largest removable entry is under about 5 points or under 3 times the noise.

## 4. Find the information, then the mechanism

An oracle gain shows the points exist, not that they are reachable. The oracle used labels. Ask what it knows
that the method does not, and whether any label-free signal at test time carries part of it.

- **Is the information in the current inputs?** Use a learned scorer as a measuring device: train it with
  supervision on every available feature, with held-out classes and images. If it recovers only a small part of
  the gap, the evidence is insufficient, and no hand-written rule over the same evidence will do better. Stop
  writing rules.
- **Where can more evidence come from for free?** List what the deployment setting provides and the benchmark
  protocol ignores: other test inputs, other views, neighbours in time, text, the model's own outputs on other
  inputs. Size each source with its labelled version first (the upper bound), then test the unlabeled version.
- **Write the mechanism.** "The error exists because ..." plus the assumptions it relies on, each tied to a
  quantity that can be measured alone. Test the assumptions, cheapest first, before assembling the method.
- **Check prior work under adjacent names** before investing more than a day. Write down the nearest three to
  five works and the difference.

## 5. Write four lines before every run

1. Which assumption this run tests.
2. The predicted numbers, derived from quantities already measured.
3. What a match means and what comes next.
4. What a mismatch means: which assumption is refuted and how the method changes.

If line 4 cannot be written, do not run it. The run would not be able to teach anything.

## 6. Design the run to answer in minutes

- Use the smallest sample that can show the effect, and stream results per unit (per category, per 50
  episodes). If the first few units show nothing, there is nothing. Stop and say so.
- Put three anchor rows in the same run: the baseline, the naive method with the same information, and the
  oracle upper bound. The naive row separates "new information helps" from "my mechanism helps". The oracle row
  shows how far there is to go.
- For a selection or weighting mechanism add random and inverted selection. If top, random and bottom score
  the same, the selection carries no information.
- Never queue hours of complete runs to answer a yes/no question. Complete runs are for final tables.
- Prepare before taking the GPU: code smoke-tested on about 10 samples, caches built, four lines written. While
  a job runs, prepare the next one. An idle GPU and a GPU busy with an uninformative job are both waste.

## 7. Read the result against the prediction

- **Match:** go to the next card.
- **Mismatch:** stop the queue first. Diagnose from data already on disk: per-class and per-unit breakdowns,
  precision against recall, an oracle fix for each error type. Only then propose the next run, one at a time,
  stating the question it answers.
- **Several rules on the same evidence fail:** conclude about the evidence, not about the rules (step 4).
- **A stop criterion written in advance is met:** stop. Record the measured reason and what is reusable.

## 8. Harden before claiming

- Paired differences on the same units, bootstrap over independent units, an interval, and the count of units
  that went up and down. `scripts/paired_bootstrap.py` does this.
- Every fold and dataset. No threshold tuned on one fold. A sign seen on 60 or 300 samples is not a result.
- Measure the variance of the protocol itself; choose the protocol that gives paired, standard units.
- Controls get the same information and are trained or tuned to convergence. Give the baseline every trick
  that is not yours.
- Apply the method as a black box on a second, stronger base method. A gain that transfers is a separate step;
  one that does not is a patch for one baseline.
- Attack the assumption that makes the method work (for example, mix in inputs that violate it).
- Translate the gain into a cost the reader understands (for example, how many extra labels it equals).

## 9. Record so that someone else can continue

- Sort every claim into three tiers: measured, inferred from measurements, unverified. An oracle result, a
  smoke test or a single-fold number is never the method's score.
- Give each number its provenance: dataset, split, sample count, seed, control, interval, script, result file.
- Keep the list of what was tried and failed, with numbers. It is what stops the next agent repeating it.
- A plan handed to another agent carries a stop rule for every experiment.
- Delete heavy artefacts when a line ends. Keep small result files and the conclusion.

## Red flags

| Thought | What to do instead |
|---|---|
| "I need the complete run to know." | The test is badly designed. Find the smallest unit that shows the effect. |
| "The oracle shows +20, so there is room." | The oracle used labels. Check that the information is reachable (step 4). |
| "It beats the baseline." | Compare with the naive same-information control, trained properly. |
| "It passed on the subset." | Compute the paired interval on all folds. |
| "One more rule or threshold." | After three fail on the same evidence, measure whether the evidence is enough. |
| "The key test failed, but the queue is already running." | Kill the queue. Diagnose from data on disk. |
| "The math is verified, so it will help." | Correct algebra is not task gain. Test the modelling assumption on real data. |
| "The cached pipeline gives about the same mean." | Check item-level identity with the released code. |
| "The plan needs more detail first." | A plan or proof with no measurement behind it is not progress. Run the cheapest test of its premise. |
| "This cannot be done." | Report the measured obstacle and the next move. |

## Files

- `references/worked-examples.md`: five cases with numbers. Read it when a step above seems optional.
- `references/templates.md`: the four-line card, ledger, result table, plan entry, hand-off and closing note.
- `scripts/paired_bootstrap.py`: paired bootstrap interval for mean metrics and for ratio metrics such as
  class-wise IoU. Standard library only.
