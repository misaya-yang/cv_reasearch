# demo9 plan

Only experiments that have not been run. An entry is deleted once its result is read: numbers go to
`README.md`, a failure gets one ledger row there. Method and theory: `HANDOFF.md`.

## Next GPU session: two readings that decide what the paper is

    cd /root/autodl-tmp/demo9_extent && nohup bash scripts/decision_boot_pivot.sh > results/decision_pivot_v1.log 2>&1 &

About one hour (an estimate from rates measured on the previous instance: 110 episodes a minute with six
workers; the log prints the minutes of each stage). No main table. The instance powers off at the end;
`POWEROFF=0` keeps it on. Each stage runs under the resource guard; a failed stage does not stop the other
reading. The transfer packs are built on the CPU beside D1 if they are missing.

### D1: can the read-out be fitted without any annotation?

- Question: is the decision rule a property of the host's evidence, so that pairs whose mask is known by
  construction teach it?
- Arms: `pixel:relations` and `convctx:layers`, each fitted only on constructed pairs from 300 unlabelled
  images x 4 regions. Three training sets: `same` (two windows of one image), `paste` (the region pasted on
  another image), `all`. Controls: FoRIS (gain 0 by definition) and the label fit (linear +2.19,
  `convctx:layers` +4.12).
- Data: DEV241 at patch level chooses. On PASS or PARTIAL the chosen fit is read once on CONFIRM600 inside
  the complete pipeline (`infer_d1_confirm`); that reading is the result.
- Rule (`scripts/decision_pivot_verdict.py`): PASS = a `convctx:layers` fit at +2.0 or more with the
  interval's lower end above 0. PARTIAL = the best arm at +1.0 or more with the lower end above 0. Otherwise
  FAIL.
- Cost: 35 to 40 GPU minutes.

### T2: does the COCO-fitted read-out transfer without refitting?

- Question: does a rule fitted on COCO objects hold on other vocabularies, on parts and on medical images?
- Arms: FoRIS, removal only, and the frozen `convctx:layers` read-out (`results/decision_v1/fit/models`).
- Data: 300-episode packs of LVIS-92i, PASCAL-Part, PACO-Part, SUIM and lung X-ray
  (`scripts/decision_transfer_pack.py`), original resolution, complete pipeline. The packs are not INSID3's
  own evaluation lists; each comparison is paired on the pack's episodes.
- Rule: PASS = three or more packs at +1.5 or more with the lower end above 0, and no pack whose whole
  interval is below 0. FAIL = fewer than two packs with the lower end above 0. Otherwise PARTIAL.
- Cost: about 15 GPU minutes.

### What each outcome changes

| Outcome | The paper | Next session |
|---|---|---|
| D1 PASS or PARTIAL, confirmed on CONFIRM600 | an annotation-free decision rule on a frozen host; first in the training-free column | the 4 x 1000 main table with that fit, then the transfer benchmarks in full |
| D1 FAIL, T2 PASS | a 104k-weight class-free read-out fitted once on COCO that improves the strongest training-free host on other benchmarks | the main table with the label fit, then the transfer benchmarks in full |
| D1 FAIL, T2 FAIL | none on this line: the read-out is a calibration of COCO objects | choose the next object from what is measured by then; the two largest measured levers are the name gap of SAM3 (about 12 points) and the union of two hosts (+3.05) |
| a PARTIAL in T2 with D1 FAIL | undecided | enlarge the packs that gained before deciding |
| INCOMPLETE in either | nothing yet: a fit or a pack did not produce its report | rerun the stages listed under `missing` in `pivot.json`, then read again |

## Held (another chat's; not part of the session above)

- Pro regional batch, 40 episodes (`scripts/pro_regional_experiment.py`): 40 episodes resolve effects of
  about 8 points; if it is run, run it on DEV241.
- Host reliability diagnostic (`scripts/host_reliability_probe.py`): it selects between two hosts' outputs;
  it is a diagnostic, not a method.

## Not planned

The 4 x 1000 main table, other benchmark tables and any new mechanism wait for D1 and T2.
