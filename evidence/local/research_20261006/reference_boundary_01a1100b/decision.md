# Rejected construction and execution correction

The current user asked chat 01a1100b to solve the main complete-method problem,
then correctly objected to its return to boundary joint optimization. The run
reduced the main problem to reference-conditioned edge regularization without
establishing a substantive difference from the already failed boundary approaches.
Do not count it as a main-line research achievement or automatically resume it.

Frozen inference used cached reference/query DINOv3 features, reference-mask edge
statistics and native Part2 contrast. No query GT was used by inference. The native
query aggregation stages were absent; final binary-token masks were bilinearly
rendered at 1024. Existing complete FoRIS/RCG/MEAN were comparisons. The integer
minimum-cut solver passed exhaustive six-node tests, but solver correctness does
not validate semantic assumptions or final mIoU.

Full reused DEV241 results, class-summed mIoU:

| Method | mIoU |
|---|---:|
| Boundary joint | 51.882497 |
| Same-mass affinity control | 51.815606 |
| Reference-only unary control | 51.174138 |
| Complete native FoRIS | 59.074825 |
| Locked RCG | 61.019669 |
| Locked MEAN | 60.680117 |

Boundary minus affinity: +0.066890 [-0.032783, +0.240805].
Boundary minus native: -7.192328 [-9.689911, -5.542657].
Intervals use 2,000 connected-photo resamples, RandomState(0). All predictions
were completed before scoring. This comparison does not establish a novel useful
signal. The target method remains unsolved.

Execution used four CPU workers and existing caches; no encoder/GPU calls,
downloads of weights/data, feature deletions, commits, pushes or peer-process
changes. The run completed; no owned queue remains. Original outputs remain in
`/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/launch/reference_boundary_01a1100b_v1`.
No variants or confirmation are scheduled. Feature-changing proposals considered
after this result were also withdrawn before implementation or execution.
