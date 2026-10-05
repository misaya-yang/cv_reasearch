# Mechanism A: paired reference contrast

Owner: `mechanism_transport`. Status: code prepared; all execution stopped after the user's instruction to prepare a batch before enabling the GPU. Earlier label-free real-data smoke is recorded below; efficacy remains untested. No remote work, downloads, model forwards, query truth or class identity have been used by this worker.

## Mechanism and prior-failure distinction

The starting perspective was assignment with rejection/capacity. The archived unbalanced-OT experiment has no incremental result. The archived reference-coverage objective has an explicit missing-part counterexample: a background witness can be rewarded for covering a foreground appearance absent from the query. Neither is being rerun. There is no Sinkhorn, mass budget, source coverage reward, native/RCG mask gate, or assumed query area in this implementation.

The gap is identity/ranking, not an oracle threshold: existing source-only ridge, QDA and joint candidate-selection results show that source fit can coexist with severe complete-mask failure. There is no prior measured evidence that this new reference-pair operation fixes the gap. Its first complete comparison is therefore decisive for development, not confirmation.

Current V2 changes the individual matching comparison. Each labelled reference witness is paired with its four most similar opposite-role reference appearances. A query patch scores the witness minus the mean of these fixed counterparts. This asks whether the query expresses a locally discriminative foreground appearance instead of merely being close to foreground in absolute cosine. Foreground/background top-four evidence compete to produce one full probability field. It can both add and remove pixels; its intended gain is recovering rare foreground appearances while rejecting common appearance shared with background. That is a hypothesis, not a demonstrated benefit.

The matched simple control uses query-chosen top-four opposite-role neighbors instead of fixed reference-chosen counterparts. It receives the same full feature matrices, same eligible witness set, same top-four operator and renderer. Its direct kNN contrast can make the more complex construction unnecessary.

## Fixed specification before first score

- Inputs: normalized cached final-layer query/reference patch features, reference coverage, query grid shape only. Native score values are ignored. No extra images, names, fitted base-class labels, intermediate layers or encoder calls.
- Reference foreground: coverage at least 0.5.
- A reference witness is eligible only when its top-four same-role mean similarity outside a Chebyshev radius of two patches exceeds its top-four opposite-role mean similarity. This is reference-only support filtering, shared by method and control.
- Opposite-role counterparts: top four by reference cosine; average their feature vectors.
- Query evidence: top-four mean of witness/counterpart contrast within each role; foreground minus background; sigmoid temperature 0.05.
- Complete renderer: shared runner bilinear interpolation, `align_corners=False`, to 1024 x 1024, then strict `> 0.5`. No CRF is claimed. The actual native complete mask remains the principal comparator.
- Class identity and query truth are forbidden inference arguments. The class-summed metric and error decomposition belong to evaluation only.
- Missing supported reference role is explicitly reported; no host fallback occurs. It is a likely failure mode, not silently replaced by an alternate method.

## Failed first construction and observed repair

V1 calibrated per-witness absolute acceptance levels as the midpoint of within-reference same-role and opposite-role similarities, with a null/reject expert. On the first real episode `0_11_0`, both candidate and role-global-threshold control rejected **every query token**, despite 911 eligible foreground and 2930 eligible background reference atoms. No query truth was opened. This directly falsifies usable absolute source-to-query calibration in that example and motivates changing the comparison to relative witness/counterpart contrast. V1 is retained as `rejected_absolute_predict` and `rejected_absolute_control`; it is not expanded or renamed into a successful result.

Real CPU one-thread smoke, `/tmp/DEV241_DINO20_export_xknuc38h`, episode `0_11_0`:

| Frozen implementation | Seconds | Patch foreground fraction | Null fraction | Result |
|---|---:|---:|---:|---|
| V1 local rejection | 0.423 | 0 | 1 | Label-free operational failure |
| V1 global rejection control | 0.231 | 0 | 1 | Same failure |
| V2 paired reference contrast | 0.566 | 0.6304 | Not used | Finite 64 x 64 field; efficacy unknown |
| V2 direct query kNN contrast | 0.430 | 0.6111 | Not used | Finite 64 x 64 field; efficacy unknown |

These are patch-field smoke observations, not scored 1024-mask results. Parameters were not chosen using the field's overlap with truth. No twenty-episode run was started. The matrices are on the requested torch device; no model is instantiated. The user subsequently stopped CPU testing and asked for prepared code before enabling the GPU; that instruction supersedes any earlier run estimate or allocation.

Static CUDA-path inspection: feature normalization, pairwise source similarity, cross-image similarity, source-neighborhood masking, counterpart construction, top-k and logits all remain on the selected torch device. Only the final 64 x 64 output is moved to CPU. Diagnostic `.item()` calls synchronize scalar values; they do not move heavy matrix operations to CPU. CUDA execution, GPU memory and wall time remain unverified. The source spatial-distance temporary is quadratic in reference tokens (about 256 MiB for the initial 4096 x 4096 x 2 int64 coordinate difference); matrices add memory but no model copy. A single model owner outside this module is required for any new cached feature extraction.

## Cross-agent challenges and required result

Agent F correctly predicted that within-instance source acceptance radii could be much narrower than cross-instance target variation. The first real example manifested exactly that failure. Agent E's query/reference latent mixture and F's local simplex reconstruction are independent of this comparison operation. A challenge to F is that coordinate-split reconstruction depends on embedding basis; the split is not independent statistical evidence. No extra experiment was launched from this discussion.

The complete-mask report must retain native, RCG and the matched kNN control on the same actual cohort; paired class-summed gains, photo-component RandomState(0) 2000-draw intervals, folds, batches, changed examples, recovered true foreground, removed false foreground, deleted true foreground and added false foreground. The exposed 220 development episodes and the local selected20 are not fresh confirmation. The current construction must not be promoted from source separation, map visual appearance, or absence of execution errors.

Next decision after the user enables GPU execution: score the frozen V2 and matched control through the shared runner. No automatic run is pending from this worker. Severe drift toward full foreground would locate a relative-score calibration/identity failure rather than validate matching. No claim or +2 target achievement exists yet.

## GPU smoke4: measured complete-mask failure and next decision

The user subsequently enabled GPU work. The coordinator ran the unchanged V2 through the shared complete-mask runner. This worker did not launch a remote process or change code while the coordinator's first20 run was active. Sources: `runs/outputs/gpu_smoke_v1/cache/report.json`, `audits.json`, and `episode_metrics.json`, relative to this note's directory. The cohort contains four episodes, four classes, four photograph components, one per fold and per listed batch; it is development, not independent confirmation. All masks are 1024 x 1024. Intervals below use 2000 connected-photo bootstrap draws with RandomState(0).

| Arm | Class-summed mIoU | Gain vs native [95% interval] | Up/down |
|---|---:|---|---:|
| Native | 66.0181 | — | — |
| RCG | 67.0743 | +1.0562 [-1.3409, +3.4533] | 2/2 |
| Paired reference contrast V2 | 58.7386 | -7.2795 [-10.8986, -3.6605] | 0/4 |
| Direct query kNN contrast control | 61.7729 | -4.2453 [-9.2232, +0.7326] | 2/2 |

V2 minus its control is **-3.0343 [-5.7377, -0.3308]**, 1 up and 3 down. These four cases demonstrate a concrete failure of this frozen construction, not a general rejection of correspondence mechanisms or reliable generalization intervals from a representative large sample.

| Episode / fold / batch | V2 gain vs native | Control gain vs native | V2 add TP | V2 delete FP | V2 delete TP | V2 add FP |
|---|---:|---:|---:|---:|---:|---:|
| `0_11_0` / 0 / old20 | -13.6933 | -11.6619 | 25,854 | 0 | 86 | 131,595 |
| `1_38_5` / 1 / new40 | -1.7644 | +0.1251 | 4,788 | 2,717 | 2,959 | 10,353 |
| `2_21_78` / 2 / new60 | -8.1039 | +1.3402 | 38 | 25,536 | 2,382 | 32,930 |
| `3_0_75` / 3 / arrived100_exposed | -5.5566 | -6.7844 | 17,484 | 29,319 | 4,821 | 108,198 |
| **V2 total** | | | **48,164** | **57,572** | **10,248** | **283,076** |
| **Control total** | | | **41,771** | **43,054** | **17,161** | **210,442** |

The dominant observed defect is foreground acceptance: only 14.5% of V2 additions are true foreground, while 84.9% of its deletions remove false foreground. Relative to native, V2 has a net +37,916 intersection pixels but +225,504 union pixels. Relative to the control's corresponding net changes, V2 retains 13,306 more true pixels while introducing 58,116 more net false pixels. These are pooled pixel accounting values, not the class-summed mIoU estimand and not direct pairwise V2/control transition counts.

This evidence does **not** justify saying V2 has no recovery ability: the `2_21_78` control deletes 9,917 true pixels whereas V2 deletes 2,382, but V2 adds 32,930 false pixels versus the control's 506. Conversely, aggregate four-action counts cannot distinguish near-boundary extent expansion from accepting a separate incorrect object. That geometric attribution is unverified until spatial predictions are inspected or a separately labelled diagnostic measures it.

The audit exposes a more specific inference problem. V2's positive foreground-evidence fraction is 1.0 in all four episodes; its positive background-evidence fraction is 1.0 in three and 0.99976 in the fourth. The direct kNN control does not have this saturation. Both role scores use `topk_j q·(r_j-b_j)`, where `b_j` is a background/foreground counterpart chosen by **reference** similarity. Maximizing the difference can select a witness because its counterpart is unusually far from the query, without requiring the witness itself to be among that query's nearest appearances. Positive role evidence is therefore not a sound acceptance condition. This algebraic possibility and near-total sign saturation are observed; their precise overlap with newly added false pixels has not been measured.

Measured CUDA wall times (candidate/control, seconds): `0_11_0` 0.1853/0.0612; `1_38_5` 0.0533/0.0447; `2_21_78` 0.0485/0.0460; `3_0_75` 0.0480/0.0393. Initial-call overhead is visible. These times include no model forwards and establish that additional CPU prototypes are unnecessary for this frozen-feature mechanism on the enabled GPU.

### Decision prepared before reading first20

Read the unchanged first20 aggregate, per-fold/per-batch gains, four actions, and per-episode changes; retain all cases regardless of whether they favor the mechanism. If the extra false-positive expansion relative to kNN persists across cases/batches, change the **witness selection operation**, not a temperature, global threshold, host gate or mask union. If the dominant failure instead changes to deleting correct foreground or one exceptional case, revise that diagnosis before coding the proposed repair. A wide interval alone is not a reason to reject or to claim success.

Proposed independent next version, not yet coded or run: select each role's four reference witnesses using raw query/reference similarity **first**, then evaluate the already fixed source-counterpart contrast only on those witnesses. Current V2 selects witnesses *by the contrast being maximized*. The change prevents a remote witness from winning solely through an even more remote opponent. It preserves the ability to use local reference role contrasts for recovering difficult foreground while tying the evidence to actual query matches. It does not guarantee removal of incorrect objects or solve cross-instance calibration by construction.

The same-input control remains direct query top-four foreground/background contrast with the same witness eligibility, features, resource budget and complete renderer. The proposed method is not equivalent to the control because its counterpart vectors remain reference-conditioned; the control chooses the opposite neighbors directly for each query. If this extra reference conditioning continues to lose, remove it rather than adding more calibrations. Any code must be a new independently named module/version, leaving the running `transport.py` unchanged. No parameter grid, RCG gate, two-mask fusion, or fresh-confirmation claim is planned.

## First20 result and the independent V3 operator test

The coordinator approved a single operator change after first20. V2 remains unchanged; V3 is implemented separately as `src/ics/methods/transport_query_witness.py` with the same `CONFIG`, `predict` and `control` API. Source helpers and the exact unchanged control are imported from the frozen V2 module. No remote execution or new model forward was launched by this worker. Python syntax compilation passed; real V3 execution is assigned to the coordinator's next unified GPU run.

First20 source: `runs/outputs/gpu_first20_v1/cache/{report,audits,episode_metrics}.json` and `fields/*.npz`. The 20 development episodes contain 18 classes and 20 photo components. Native is 55.0962, RCG 55.6279, V2 41.9625 and control 43.6969 class-summed mIoU. V2 minus native is **-13.1336 [-20.4529, -5.1908]**, 3 up / 16 down / 1 tie. V2 minus control is **-1.7344 [-4.7977, +0.9482]**, 6 up / 13 down / 1 tie. The latter interval spans zero; the complete-mask loss versus native remains practically large.

| First20 breakdown | V2 gain vs native | Control gain vs native |
|---|---:|---:|
| Fold 0, n=5 | -14.2389 | -9.2915 |
| Fold 1, n=5 | -27.8856 | -24.9945 |
| Fold 2, n=5 | -8.8922 | -10.7501 |
| Fold 3, n=5 | -0.8907 | -0.0096 |
| old20, n=4 | -12.6091 | -18.3604 |
| new40, n=4 | -0.6891 | +3.5343 |
| new60, n=4 | -29.8939 | -22.3710 |
| arrived100_exposed, n=8 | -9.9770 | -7.1570 |

| First20 pixel actions vs native | Add TP | Delete FP | Delete TP | Add FP |
|---|---:|---:|---:|---:|
| V2 | 188,491 | 237,867 | 49,150 | 1,717,117 |
| Control | 185,875 | 185,651 | 42,191 | 1,518,956 |

V2 addition precision falls to **9.89%**, while deletion precision is 82.88%. Unlike the four-case sample, V2 has fewer net retained true pixels than the control: net intersection gains of 139,341 versus 143,684. Its net false-pixel increase is 1,479,250 versus 1,333,305. This supports the prewritten false-acceptance diagnosis without relying on an interval-based rejection rule.

Saved fields were inspected for both destructive and beneficial cases. The catastrophic `1_21_65` case adds 485,305 false pixels and gains only 1,792 true pixels (V2 gain -70.4048). Both V2 and control label exactly 59.692% of patch positions foreground; their median probability is 0.595 and 0.762 respectively. This strongly indicates a **shared complete-scope failure**, not just the identified contrast-maximization defect. Beneficial cases remain real: `3_1_3` gains +7.1831 with 73,814 added TP; `2_0_74` gains +19.8350 mainly by deleting 18,276 FP; `2_38_50` gains +5.1246 with 35,981 added TP. Their V2 patch-foreground fractions are 0.5698, 0.0464 and 0.1758. The existing failures do not justify discarding all recovery behavior, but the successes do not offset the new60 collapse.

### Exactly what changes, and what does not

For each role, V2 forms every fixed source-conditioned contrast `c_j(q)=q·r_j-q·b_j` and averages the largest four `c_j`. V3 first finds the four largest **raw** `q·r_j` among the same eligible reference witnesses, and averages `c_j` only at those indices. This removes the option to promote a distant positive witness solely because its source opponent is farther away. Reference eligibility, counterpart construction, top-k count, temperature, feature layer, model-forward budget, native-score exclusion, output interpolation and threshold remain unchanged. No parameters were selected from first20 truth.

The control is an exact call to V2's direct-query kNN control, so numerical discrepancies between its first20 and V3-run output indicate a reproducibility problem rather than a method change. The candidate remains different: its opposing appearance is conditioned on the chosen reference witness, whereas the control chooses the opposite appearance directly from query similarity.

**Scope limitation:** V2 and V3 both replace the complete host readout. They do not preserve FoRIS positional/background-suppression/clustering scope inference or its CRF. Therefore V3 can repair the documented witness-selection defect while still losing badly to complete native, especially where the control also expands wrong foreground. No claim that this operator change restores host scope is made. New complete-native and RCG comparisons remain mandatory; improvement over V2 alone is not sufficient progress toward the requested method. If V3 does not materially improve the false-acceptance problem and its complete-mask tradeoff, do not extend this construction through temperature searches or extra gates.
