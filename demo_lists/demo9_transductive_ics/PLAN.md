# demo9 plan

Codex's dated plans are in `docs/codex_plan/MM_DD.md` (the newest one is current). This file lists experiments
prepared here and not yet run. `docs/harness/STATUS.md` names the one next action.

## 2026-10-05 (Claude): boundary from finer tokens of the query

**Error, cost, signal (DEV241, measured, README).** FoRIS's own boundary is off by a few pixels on small targets: true
pixels within 16 px of it are worth +15.8 mIoU (within 8 px: +9.8), with FoRIS's coarse decision kept. Its CRF takes
+0.5; colour cues take nothing more. A mixing share (a band token's position between the means of its nearest sure
object and sure surround tokens, in the query's own feature space) follows true coverage (Spearman 0.67; FoRIS's score
0.47) and gives +0.88 [+0.50, +1.71] on the 64 x 64 tokens, where small targets have no sure tokens at all.
**Question.** Does the same mixing share on a 128 x 128 token grid of the same frozen model take a larger part of it?
**Bank** (`scripts/hires_bank.py`, GPU): query tokens on a stride-8 grid (four shifted encodings interleaved) and of the
view enlarged to 2048; the reference's mean token in the same basis. **Reading** (`cpu/hmatte.py` on the server's CPUs):
bands of 1 to 3 fine tokens, alpha alone and averaged with FoRIS's score, against FoRIS after its CRF; diagnostics with
the truth's trimap. **Known risk:** re-running FoRIS on zoomed crops did not sharpen boundaries (+0.15).

**Session** (`/root/autodl-tmp/demo9_lang`, GPU mode, both preflights passed in no-card mode; ends with shutdown):

    cd /root/autodl-tmp/demo9_lang && (RUN_LANG=1 nohup bash session_all.sh > session_all.log 2>&1 < /dev/null &)

Part 1, `plan_main.json`: smoke on 4 real episodes of every script (bank, layer probe, both readings, the complete
pipeline) -> `hires_dev` bank -> `layers_dev` maps -> `cpu/layer_read.py` -> `cpu/hmatte.py` (writes
`cpu/hmatte_gate.json`; PASS = nested gain >= +2.0 with the interval above 0, against FoRIS after its CRF) ->
`hires_pipe.py` on DEV241 at original resolution (skips itself unless PASS; `go` = +1.5 with the interval above 0) ->
the same on CONFIRM600 (skips itself unless `go`). Part 2, only with `RUN_LANG=1`: the text-head plan below.
`scripts/layer_probe.py` stores, per layer (8 to 24), for the raw last layer and for both images encoded on one canvas,
only the one-vector score map; FoRIS's whole use of the reference equals that one vector (README).

| Stage | What each outcome settles |
|---|---|
| `hmatte` diagnostics (truth trimap, alpha at 128 x 128) | above the 64 x 64 value (90.1): finer tokens carry sub-token coverage; not above: token features do not hold the boundary at any grid, the band needs a mask-trained decoder |
| `hmatte` label-free arms | PASS: a boundary method for this column; FAIL with good diagnostics: the loss is in the trimap (FoRIS's coarse mask), not in the features |
| `layer_read` | a view BETTER than FoRIS's tokens (best cut +2.0): the one-vector reading should move there; none: layers, the raw last layer and joint encoding are closed for the one-vector reading with one row |
| text-head gate | see the verdict table below |

**Cost.** Part 1 about 35 minutes when the gate fails, about 1.5 hours when both pipeline stages run (caps sum to
5.5 h); part 2 about 20 minutes on FAIL, 1 hour on PASS. Disk: 4 GB for the token bank, deleted by the script once it
is read; about 7 GB for the text-head bank (the script skips part 2 under 10 GB free). Column: training-free on the
frozen DINOv3 (FoRIS 60.9). No stage runs CONFIRM600 unless its DEV241 gate passed; nothing runs the whole list.

## 2026-10-05 (Claude): the language channel on the frozen DINOv3, one gate (held; adds the backbone's text head)

**Why.** The measured errors are confusions between neighbouring categories (README, 2026-10-05); a category
boundary is what a name supplies, and it is the lever confirmed on SAM3 (exemplar 68.9, self-named and routed 73.1,
true name 78.9). DINOv3 ViT-L has one released head that carries names: dino.txt (image-text pairs, no masks). The
user objected on 2026-10-05 that this adds a component; it runs only if the user wants it.

**Method under test.** Reference region -> a posterior over a fixed public vocabulary (cleaned LVIS names + 43
background words); each query patch -> its probability under those words; that evidence, on a 0..1 range, is added
to FoRIS's final score; FoRIS's own cut and CRF follow unchanged. `scripts/lang_common.py` holds the arms.

**Work mode (answers "the GPU decides every idea").** The GPU only fills a bank of text-head tokens for DEV241
(`lang_bank.py`) and later confirms one frozen arm. All 96 label-free arms (3 token views x 8 ways to carry the
concept x 4 weights) are read from the bank and FoRIS's saved packets by `lang_gate.py`; the reported gain is nested
over folds, so the choice among arms cannot inflate it. Checked without a GPU on 2026-10-05: the gate reproduces FoRIS
from the packets exactly (DEV241, cut before the CRF: 58.56, 0 differing pixels; contested-area AUC 0.656); a
synthetic bank built from the labels gives PASS (+32.8), one built from noise gives FAIL (-0.9 [-1.5, -0.4]); bank,
gate and pipeline ran end to end on the no-card server with small random models.

**Verdict, fixed before any number** (DEV241, FoRIS's cut at model size before the CRF, paired, photograph groups):

| Verdict | Condition | What follows |
|---|---|---|
| PASS | nested label-free gain >= +2.0, interval above 0 | the arm chosen on all of DEV241 is frozen; `lang_pipe.py` reads it inside the complete pipeline at original resolution on DEV241; if that is >= +1.5 with the interval above 0, once on CONFIRM600 |
| NAMING | true class name >= +3.0 (interval above 0), label-free below the bar | the channel is good, our reading of the word loses it: work on the naming from the bank, CPU only |
| FAIL | true class name < +3.0 | this head does not carry the category where FoRIS errs: one ledger row; a failure of this construction, not a ceiling for the column |

**Session** (`/root/autodl-tmp/demo9_lang`, `plan.json` through the guard, ends with shutdown): smoke on 4 real
episodes (bank, gate, complete pipeline) -> DEV241 bank -> gate -> pipeline on DEV241 -> pipeline on CONFIRM600 ->
name posteriors of the CONFIRM600 references (a by-product: it lets a CPU replay test an independent namer for the
SAM3 route) -> a 50 s window to copy the reports. The two pipeline stages skip themselves unless their bar is met.

    RUN_LANG=1 bash session_all.sh        # part 2 of the session above; preflight passed in no-card mode

**Cost.** About 1 hour of GPU if the gate passes, about 20 minutes if it fails. FoRIS's 1.7 s per episode is
measured; the text head's time is not (the smoke stage prints it). Stage caps sum to 2.5 hours. Column: training-free
on the frozen DINOv3 (FoRIS 60.9 is the bar). The weights (3.47 GB: the text head and the official backbone
file) are on the server, hashes verified; public copy `PIA-SPACE-LAB/dinov3_vitl16_dinotxt_vision_head_and_text_encoder` on Hugging Face, official
hashes a442d8f5 and 8aa4cbdd in the file names; the official route is Meta's request form.

**Not in this plan on purpose:** whole-list runs, other benchmarks, any new arm on the GPU. A reviewer may count
the text head as an added component: the paper would state "DINOv3 with its own text head, no mask supervision".

## Held (another chat's)

- Pro regional batch, 40 episodes (`scripts/pro_regional_experiment.py`): 40 episodes resolve effects of
  about 8 points; if it is run, run it on DEV241.
- Host reliability diagnostic (`scripts/host_reliability_probe.py`): it selects between two hosts' outputs.
