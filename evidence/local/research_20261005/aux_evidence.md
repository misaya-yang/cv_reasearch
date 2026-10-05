# The sweep: every proposal and every kind of label-free evidence as edits of the host mask

Owner: Claude Code, 2026-10-05. Status: **code prepared; self-check and a 20-episode local execution smoke passed;
not launched; no efficacy result.** Remote execution is held with the rest of [PLAN](../../../docs/research/PLAN.md).
An earlier version of this file proposed five hand-picked auxiliaries; the user corrected that: with A + B +
auxiliary as the working formula, the candidates are to be swept in bulk, not guessed.

## What is swept

The host mask H stays. A proposer adds P & ~H and deletes H & ~P; the two sets never overlap, so with good = added
true or deleted false pixels and bad = the side effect, `I = I0 + add_good - delete_bad`, `U = U0 + add_bad -
delete_good` per episode: the class mIoU of any composition is exact from counts
([edit_counts.py](../../../src/ics/edit_counts.py)). Evidence is one byte per token; additions are accepted from
the top, deletions from the bottom; a rule is one threshold meeting a label-free pixel budget, per episode (`rank`)
or pooled over the fitting folds (`absolute`).

**Rows = proposers x {add, delete} x maps.** One row took about 1 ms on 20 episodes; the cost grows with the number of episodes.

- Proposers: every sealed arm (RCG, D, transition only, concat; the suite's E, fixed-slot, two-slot and B kernel
  proposals; the Astra replay), read from their sealed files and never recomputed, **and the universe**: every pixel
  outside H for addition, every pixel inside H for deletion, so that each map is also graded as a proposal of its
  own. Budgets: 10% to 100% of a proposer's edit set; 2% to 50% of the host's area for the universe.
- Maps ([aux_evidence.py](../../../src/ics/methods/aux_evidence.py)), enumerated as family x token space x parameter:

| Family | Maps per space | What varies |
|---|---:|---|
| R reference matching (cross-image spaces) | 10 | nearest / top-10 / prototype / 4-part margins, one-sided scores, label transfer by 1, 5, 20 neighbours, cycle consistency |
| Q same-image anchors | 24 | anchors (host mask, or its confident core against confident background); k = 1, 5, 20; pairs closer than 0, 2, 4 tokens excluded; prototype margin; one-sided scores |
| G grouping | 11 | k-means segments 8, 16, 32, 64 (host cover and mean host score of the rest of the segment); diffusion of the host mask over the 10-NN graph for 1, 3, 10 steps |
| H host and geometry | 10 in all | host score and stage scores, cached reference maxima and their margin (Astra's gate signal), local score contrast, signed distance to the host |
| V other proposals | 1 + one per proposal | how many proposals take the token; each proposal's own field |
| N placebo | 24 in all | seeded noise at three spatial scales |

Spaces: the cached last layer; with the saved layers of the D run, layers 16 and 24 and their difference (197 maps);
with `--forward-layers 4,8,12,16,20,22,24`, one extra paired forward per episode gives 14 spaces and 666 maps,
positional bases built from the black image as the host builds its own, nothing stored but the maps. With ten
proposers that is about 14,600 rows. Fixed constants throughout; nothing is fitted.

## What a sweep this wide needs, and has

- **Placebo calibration.** The 24 noise maps go through the identical sweep. Per operator the report gives the
  placebo maximum and 95th percentile of the gain and the placebo range of selectivity, and counts the real rows
  outside them. A real map counts only if it clears what noise reaches under the same search. (In the 20-episode
  smoke a noise map tops one leaderboard, as it should at that size.)
- **Two statistics per row.** Within-episode selectivity (probability that a good edit pixel outranks a bad one;
  0.5 = no selection) does not involve choosing a threshold. The nested gain has its budget chosen on three folds
  and read on the fourth, with a paired interval, and for a sealed proposer is also reported minus the same nested
  choice between the whole edit set and none of it.
- **Composition with matched controls.** Additions of one rule and deletions of another, the whole choice nested
  over folds: `edit`, against the same search with no auxiliary (`edit.unfiltered.control`), with noise maps only
  (`edit.placebo.control`) and with the simple readings only (`edit.simple_auxiliary.control`: host score, proposal
  fields, Astra's gate signal). The chosen masks are rendered, checked against the counts and sealed.
- **Still development data.** Every row is read on DEV241, which has been read many times. What survives the placebo
  line is a candidate for episodes not yet read, not a result.

## Run

```bash
cd WORKSPACE; PY=/root/miniconda3/bin/python
$PY scripts/prepare_aux_evidence_batch.py --workspace . --run-name sweep241_v1 \
    --suite outputs/SUITE_RUN --astra outputs/ASTRA_RUN --forward-layers 4,8,12,16,20,22,24   # each optional
$PY scripts/experiment_pipeline.py --plan launch/sweep241_v1/plan.json \
    --state-file launch/sweep241_v1/pipeline.json --log-file launch/sweep241_v1/events.jsonl      # add --run
```

`aux_infer` is one GPU process (or `--device cpu` with workers); `aux_score` is CPU only, uses `--workers` processes
for its counting pass and can run beside other GPU work. Without `--suite`, `--astra` and `--forward-layers` the
batch needs only what is already sealed on the server. Outputs: `outputs/<run-name>/{evidence/, sealed.json,
report.json, report.md, predictions/}`; `report.json` holds every row.

## Checked and not checked

- `PYTHONPATH=src python3 -m ics.edit_counts`: counts equal rendered masks for both operators, sealed and universe
  proposers, both rule kinds and every budget; noise has selectivity 0.5 and perfect evidence 1.
- Local execution smoke, 20 cached episodes (5 per fold), cached last layer: 84 maps, four sealed proposers plus
  the universe, 840 rows; both stages finish, every rendered mask equals its counts, the unfiltered search
  returns RCG exactly. Library 1.1 s per episode with 2 threads; the sweep itself about 1 s. No accuracy is read
  from 20 episodes.
- Layer spaces ran only on stand-ins: perturbed cached tokens for the saved-layer path (197 maps, 3.3 s per
  episode), a random stand-in encoder for the forward path (666 maps, 14 spaces). They show the code executes and
  nothing about real layers.
- Not checked: CUDA, the real encoder taps and bases, the suite's `proposals/` and Astra's `frozen/` as sources
  (their seal lookup is written from the file formats), runtime and memory on the server, and whether any row
  clears the placebo line.
