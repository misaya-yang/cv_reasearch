# Public4000 streaming resource audit

2026-10-05. Two completed read-only remote sampling windows, each40 seconds with
21 readings at2-second spacing. Main process PID45702/start953851020 remained
unchanged. CPU percentages use100%=one core. GPU means/fractions summarize
sampled `nvidia-smi` readings, not continuous kernel occupancy.

**CLS concurrency increased GPU utilization but substantially reduced the main
task's throughput. It was not cost-free use of idle GPU gaps.**

| Measurement | Main alone | Main + bounded CLS |
|---|---:|---:|
| Sampled GPU utilization mean | 71.8% | 95.8% |
| Readings below50% utilization | 23.8% | 4.8% |
| Readings at least90% utilization | 47.6% | 90.5% |
| Whole-device memory used | 2482 MiB | 5785–6085 /32760 MiB |
| Main process average CPU | 103.51% | 101.93% |
| Two solver workers' combined CPU | 35.15% | 19.10% |
| Main seconds per newly encoded episode | 2.448 | 4.002 |
| Main newly encoded episodes/minute | 24.5 | 15.0 |

The standalone reference aggregates three20-case log intervals:49.640,48.692,
48.544 seconds. Immediately preceding the shared interval, three other20-case
intervals took48.023,47.796,50.067 seconds. With CLS active, encoded1620→1640
took80.039748 seconds. This is approximately **63.5% more time per case and
38.8% lower main throughput** than the earlier2.448-second reference.

The time relationship supports a shared-GPU competition interpretation, but
these windows contain different consecutive episodes. They are **not a
same-sample controlled causal comparison**. No code or parameter change to
the main v1 was made during this audit, and there is no measured breakdown of
encoder, FoRIS decoder, CRF, dispatch or synchronization time.

The main's pending fine-RAM queue remained0 and completion lag remained3 at
the retained progress events. Solver workers frequently waited on
`pipe_read`/futex. In the first window physical reads were0, worker physical
writes were0, and main physical writes averaged about0.073 MB/s (peak0.170);
no D-state was observed. This does not support adding solver workers or treating
physical disk writing as the bottleneck. The earlier candidate optimization
remains a future Part1-only extractor, gated by two-pair q/r, debias decision,
lambda16 field and stored-mask parity; its speedup is unmeasured.

CLS used supervisor48562/start954245670 and child48569/start954245687, nice10,
seven threads. Actual `model_setup_smoke.json` records FP32/no autocast,
tokens `[3,4101,1024]`, and a PyTorch CUDA allocated peak of1,866,449,408 bytes
(about1.74 GiB). This process-specific setup peak is distinct from whole-device
memory use. Whole-device memory peaked at6085/32760 MiB (18.6%), with no
material memory risk observed in the sampled interval.

The auditor notified root immediately of the slowdown and made no job changes.
Root reported CLS66/85 complete, leaving19 fixed cases out of the367-view
test, and chose to finish that bounded remainder without expanding parallel
GPU work. Future CPU analysis may run concurrently; long GPU tasks should be
queued serially. The healthy main v1 remains unchanged. This decision/status
is root-reported; the auditor did not perform another poll or claim final CLS
completion.

Sources: `nvidia-smi`, `/proc` deltas, main
`launch/frozen_subtoken4000_v1/events.jsonl.stage-0.log`, CLS
`launch/object_cls_dev241_v1/state.json`, and
`outputs/object_cls_dev241_v1/model_setup_smoke.json`, under the recorded remote
workspace. Exact retained summaries and progress intervals are in
[report.json](report.json). Full raw records and the complete second-window
per-reading series were not persisted; missing values and absolute sampling
start timestamps have not been guessed. No additional sampling, GPU compute,
process action, cleanup or shared-document edit was performed after the save
request.
