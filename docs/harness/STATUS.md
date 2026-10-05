# Status

Updated 2026-10-05 13:10 CST. Replace, never append. At most 40 lines; history goes to the ledger in the direction's README.

## Now

- Live: `demo_lists/demo9_transductive_ics`. Server `ssh -p 48002 root@connect.westd.seetacloud.com`, no-card mode, nothing runs.
- **The one next action:** in GPU mode start `session_all.sh` (PLAN.md): finer query tokens + layer probe for DEV241, read on
  the server's CPUs; pipeline and CONFIRM600 stages skip themselves unless the DEV241 gate passes. Both preflights passed.
- Work mode: the GPU fills a bank and confirms one frozen arm; arms are chosen on a CPU. The text-head gate is ready, held.

## Measured

COCO-20i 1-shot, standard list (4 x 1000, seed 0), class mIoU at original resolution.

| Method | Whole list (4000) | CONFIRM600 | 915 fresh |
|---|---:|---:|---:|
| FoRIS (public code; DINO only) | not run | 59.78 | not run |
| FoRIS + read-out fitted on base classes | not run | 63.33, +3.54 [+1.96, +4.96] | not run |
| SAM3 exemplar, proposals >= 0.7 x top score | 68.86 | 67.12 | 68.76 |
| SAM3 exemplar or self-named, routed by confidence | 73.14 [71.85, 74.41] | 72.62 | 73.82 |
| SAM3, true class name (privileged) | not run | 78.34 | 78.92 |

- FoRIS's errors: wrongly included regions resemble the reference more (0.53) than missed target parts (0.44); the true
  share is worth +8.67 and four label-free estimates of it fail; contested-area AUC 0.664, best label-free reading 0.724.
- CPU replay 2026-10-05 (DEV241, before the CRF, FoRIS 58.52; about 30 readings, README): none above +1. FoRIS uses the
  reference as one mean vector (its other reference terms are inert). True pixels within 16 px of FoRIS's boundary: +15.8;
  its CRF +0.5; mixing share on 64 x 64 tokens +0.88 [+0.50, +1.71]. Wrong objects far away: +12, untouched by any reading.
- Published: INSID3 57.6, FoRIS 60.9 (DINOv3 only); UINO-FSS 64.5 (trained); FSS-SAM3 66.1; CG-ICS 72.3 (SAM3 + MLLM);
  SegIC 76.1, UNICL-SAM 77.8 (trained on COCO).

## Unverified

- Whether the text head's patch tokens carry the category where FoRIS errs (gate held).

## Needed from the user

- GPU mode: about 35 min if the boundary gate fails, 1.5 h if it passes; plus 20 to 60 min with the text-head gate (RUN_LANG=1).
- Optional: results of the GPT chat/agent runs on the uploaded DEV241 archives; candidates are re-read on all 241 here.
