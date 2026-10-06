# Read-only direct-MEAN1200 / peer-C resource window

Observed 2026-10-06 01:17:53–01:18:21 UTC, 28.113 seconds, 15 samples at 2-second target spacing. NVIDIA telemetry and procfs only; no GPU work, signals, queue changes, or cleanup.

GPU utilization averaged 100% (all 15 samples 100%; none below50%). Used memory averaged6064.2MiB, range6049–6277 of32760MiB. This is a bounded repeated observation, not a claim about every instant or a free parallel speedup.

Direct-MEAN child71246/start955120027 used107.44% CPU and13 threads; peer-C71178/start955119997 used124.98% CPU and139 threads. Both endpoint snapshots had one running thread and all remaining threads sleeping; no D-state thread occurred in any sampled process. Parents56334/start954783172 and63234/start954947392 used0% CPU. Direct-MEAN physical read bytes were0 (reads served from cache), physical writes33.214MB/28s ≈1.19MB/s. Peer-C physical read/write bytes were0. These observations provide no evidence of a write bottleneck; they do not identify each GPU process's device share.

MEAN's latest complete20-case log interval220→240 took30.358s:1.518s/case (39.53cases/min). Its synchronized internal stage grew29.838s:1.492s/case. That timer includes feature extraction, readout, CPU transfers/checks, kernel preparation and device synchronization; it is not kernel-only time and may include waits caused by the peer. Logs advanced230→240 during the observation, but logging every10 means that count difference is not the exact28-second throughput. At240/1200, query_truth_opened remainedfalse.

Peer-C remained50/600 at cumulative240.7s in the sampled logs. Its start300/count150 is applied separately per four folds, yielding600 draws, not150 total. Cumulative4.814s/draw includes model setup; there was no second C log point in this window to estimate concurrent short-window throughput. The preceding B group's different draws and stages cannot serve as a same-sample causal control.

Scopes remain the original jobs: direct-MEAN reconstructs four FP32 shifted grids/case, verifies the old fine16 field/mask, adds only fine.mean16.control on1200 and retains reusable raw FP32 kernels. Fine64 stays the primary, and six original arms are preserved. Peer-C runs native, RCG, uniform-tau fine readout and frozen size-cut on its own600 draws. This audit changed no job. The missing2000 uniform-tau successor is the controller's separate queue decision; no duplicate1200 encoder was launched here.

[samples.json](samples.json) contains all raw telemetry and before/after log tails; [report.json](report.json) contains derived values. There is no serial counterfactual in this window, so higher utilization alone cannot establish lower combined latency or cost-free co-running.
