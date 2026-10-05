# Part1 producer two-pair benchmark

Owned, frozen source snapshot and one GPU stage for root to combine into the post4000 pipeline. No GPU stage was launched during preparation.

The existing pipeline runner waits until the visible GPU inventory is known and every compute PID has released the GPU. Its default sharing list is empty. Do not pass `--share-gpu-with-state`, adopt the main job, reuse the main state/log files, or launch this one-stage plan independently before root composes the post4000 stages.

The stage uses the real host, reference RGB/mask, query RGB, FP32 paired extraction, actual Part1 gate/projection, actual full producer and independent stored native/RCG controls. It does not download assets or replace a missing runtime. `remote_assets.json` records the actual read-only runtime/source/asset checks, with CUDA hidden during dependency imports.

Success requires report/state `TWO_PAIR_PART1_PRODUCER_PARITY_AND_TIMING_COMPLETE`, n=2, query_truth_opened=false, plus config/evidence files. Source hashes, external source/model/basis hashes, selected cache/RGB/reference-mask hashes and prior seals are explicit requires. Missing or changed inputs fail before GPU launch.

`validation_command.json` contains the exact remote validate-only command. Only root may add `--run` after combining stages. GPU numerical parity and time saving remain unmeasured until the benchmark succeeds.
