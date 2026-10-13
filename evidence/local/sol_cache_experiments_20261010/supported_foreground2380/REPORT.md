# Frozen supported-foreground composition: 2380 exposed cases

Protocol: [`consensus_supported_foreground_protocol.json`](../../insid3_foris_generalization_20261010/consensus_supported_foreground_protocol.json), SHA-256 `5fd1f4be5abc6473af2b59c60d75a521fb10211269820c093e2949893166f2e2`. Cases are ordered DeepGlobe 100, COCO 200, PACO 600, LVIS 1400, PASCAL-Part 40, SUIM 40. The LVIS 40 from the earlier 1020-case pool are contained within LVIS 1400, not counted twice. This is an exposed development evaluation, not unseen confirmation or the five complete official benchmarks.

The fixed candidate is `N = C | (U & (H > 0))`, where `C` is the previously sealed majority of `B,R,G`, `U = ~B & ~R & G`, and `H` is the previously fitted, clipped reference ridge field. The same-K control adds exactly `K = |U & (H > 0)|` pixels to `C` from `U`, ranked by descending `abs(2 * FP32_bilinear(mean64, frame) - 1)` with row-major ties. The candidate and control were both saved in CLI 1024 and original-image frames. No encoder, raw producer, new fit, or parameter search was run for this experiment. The producer sealed all 9520 new masks before the scorer read any query ground truth.

Official fixed class/fold slots, including zeros for missing classes, give these mIoU percentages. The table compares the new candidate against the existing shared-edits consensus `C` and the same-K control; CLI is primary. `N-C` and `N-ownK` are percentage-point differences.

| Dataset | Cases | Frame | C | N | ownK | N-C | N-ownK |
|---|---:|---|---:|---:|---:|---:|---:|
| DeepGlobe | 100 | CLI | 30.225795 | 30.376570 | 30.370820 | +0.150775 | +0.005750 |
| COCO | 200 | CLI | 63.888679 | 63.962037 | 63.949968 | +0.073358 | +0.012069 |
| PACO | 600 | CLI | 39.492237 | 39.517861 | 39.510933 | +0.025624 | +0.006928 |
| LVIS | 1400 | CLI | 44.398755 | 44.317619 | 44.319063 | **-0.081136** | -0.001443 |
| PASCAL-Part | 40 | CLI | 36.379279 | 36.509369 | 36.511559 | +0.130090 | -0.002190 |
| SUIM | 40 | CLI | 62.193643 | 62.389209 | 62.415551 | +0.195566 | -0.026342 |
| DeepGlobe | 100 | Original | 30.225795 | 30.376570 | 30.370820 | +0.150775 | +0.005750 |
| COCO | 200 | Original | 64.080288 | 64.140667 | 64.127715 | +0.060379 | +0.012952 |
| PACO | 600 | Original | 39.664615 | 39.684486 | 39.678407 | +0.019871 | +0.006079 |
| LVIS | 1400 | Original | 44.519548 | 44.438002 | 44.435575 | **-0.081546** | +0.002428 |
| PASCAL-Part | 40 | Original | 36.352448 | 36.389254 | 36.388528 | +0.036806 | +0.000726 |
| SUIM | 40 | Original | 62.533676 | 62.715986 | 62.746631 | +0.182310 | -0.030645 |

The positive five-domain `N-C` signs do not transfer to the full LVIS pool. The matched LVIS600 strong-control subset confirms the limitation: CLI `N=31.470808`, `C=31.595094`, same-K `31.465452`, and the pre-existing fast region control `33.718963`; original-frame values are `31.553474`, `31.671297`, `31.546009`, and `33.916394`. Thus this gate does not beat the matched strong control. The ownK arm also equals or exceeds N in LVIS CLI, PASCAL CLI, and SUIM in both frames, so these scores do not establish a unique ridge-evidence ranking effect.

The independent scorer replayed the saved masks from sealed fields and a separate lexicographic same-K selection; its four reconstruction mismatch counts are zero. It also rechecked prior parent intersection/union counts, the official class/fold evaluator, and the matched LVIS600 source scores. On the exact earlier 1020-case subset, all six CLI `N-C` values equal the independently recorded analytic additions to numerical precision; LVIS40 is `+0.001256` pp even though LVIS1400 is negative. Full scorer records, class-slot details, paired-photo intervals, all comparison arms, checks, and source hashes are in [`report.json`](../../../../../cv_data/a/consensus_supported_foreground2380_v2_20261011/report.json) and [`scored_episodes.json`](../../../../../cv_data/a/consensus_supported_foreground2380_v2_20261011/scored_episodes.json). `verification.json` records the final cross-check and file digests.

The initial `v1` output directory contains only a configuration and manifest: production stopped before any mask because of a manifest key typo. The corrected, exclusive `v2` directory is the sealed scored result. Existing masks and fitted fields were not changed.
