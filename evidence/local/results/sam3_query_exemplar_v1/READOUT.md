# C2 query-exemplar probe: completed, not expanded

COCO-20i val2014, old exposed DEV40 (ten per fold), seed0, legal single semantic-mask component, same frozen official SAM3, FP32/TF32off, original-resolution class-summed IoU. The original241 global index and source box were preserved. Forty paired photograph groups;2000 common paired-bootstrap draws, seed0. This is exposed premise evidence, not an independent method result or an exact published FSS-SAM3 episode reproduction.

Latest interpretation correction: a paired interval crossing0 does not erase a positive measured aggregate gain. A on oldCONF600 measured61.0908 versus FoRIS59.7830 (+1.3078pp); it is an observed aggregate win with uncertainty, not a failed SAM3 line. C2's primary is below its same-input cheapOR in the observed aggregate, which is a separate negative mechanism comparison. The new invariant-cue route has no real-model result.

Read-only independent source audit: FSS-SAM3 commit b82ba838678ae8c2f27f618d9500a50c171e223e, `model/evaluate.py`160–244 uses the same COCO1-shot1008/60%-Source lower canvas, cxcywh normalisation, complete retained-mask union and nearest original mapping as the adaptation. `data/dataset_tool.py`71–88/137–141 instead samples1000 episodes/fold and a noncrowd instance with annotation area>1000, while our legal semantic-component sampling differs. Static alignment is not a real-model independent public-evaluator parity check; FP32 instance-only MLP versus the author's bundled numerical backend also remains unverified. No newly identified score-changing bug or official benchmark reproduction is claimed.

| Row | Class mIoU | Delta over source-only SAM3, pp | Connected-photo95% interval |
|---|---:|---:|---|
| Source-only SAM3 |62.492|0|—|
| Complete FoRIS |65.161|+2.669|[-5.366,10.989]|
| Source + FoRIS query box, primary |63.557|+1.065|[-7.539,9.691]|
| Source + self-predicted query box |61.113|-1.379|[-3.124,.598]|
| Source + location-random query box |42.051|-20.440|[-29.213,-11.461]|
| Query-FoRIS box alone |60.971|-1.521|[-10.415,7.615]|
| Cheap OR of the two original masks |67.418|+4.926|[-2.873,13.081]|
| Cheap AND |59.706|-2.786|[-3.868,-1.547]|
| Source + true query box, privileged input |77.334|+14.842|[6.810,22.529]|

Primary minus cheap OR=-3.861[-8.385,1.184]. Primary fold changes over SAM3 are[-16.718,2.843,7.047,11.089]. Primary minus random=+21.506[12.652,30.000], but primary minus self=+2.445[-6.095,10.987]. Random matches FoRIS-box size/aspect/count in all40; self uses fewer extra prompts in9 cases, explicitly reported. No prompt-count equivalence is claimed for those9 self cases.

The registered primary failed its2pp/positive-lower-bound prediction and did not beat cheap OR. This fixed query-box re-prompting construction is not expanded. Correct query boxes are useful under this action, but their GT gain does not establish an observable locator or a reachable information guarantee. Random boxes damage the concept condition; that does not prove which semantic element causes the primary's failures. Cheap OR is a strong naive control to check on the existing larger cohorts, not a novel method or a stable40-case gain.

All40 original SAM3 masks/I-U and complete FoRIS I-U matched exactly; prompt-state NoOps were exact. One image encoding was used for all legal arms per case. Every legal prediction froze before any query label was opened; the privileged40-case bbox phase paid for40 new encodings. No training, new model download, image pool, threshold/box sweep or text class-name input was added. Source-mask reconstruction from query-only prompts remains diagnostic and never selected the primary. Code: scripts/sam3_query_exemplar_probe.py, SHA74623926; CPU helper SHA9643920e; manifest SHA d01aab6c6f27bacf8ac8c6d9477a5faa1f4e7d372a9c2efc75add91f76440d86.

The finite guard completed legal40 at56.926s, GT-bbox40 at99.229s and CPU score at101.417s; shutdown was requested at101.851s. E62 p48vc3ykfm-afe2efc4, weste:12376 independently showed STOPPED afterward. E69 had0 free host GPUs, so same-price32GB/E62 was cloned with both disks; the provider's copy/provisioning time precedes the execution clock. These shared-server elapsed times are neither isolated speed evidence nor100% GPU-duty measurements. A separate no-card session retrieved the final files after the fast guard shutdown raced the original export; no GPU inference was repeated.

Complete evidence is retained in completed_evidence_20261003.tgz:512,254bytes,99members, both-copy SHA8d895f8b9683b62e12a2c8b6fa206d882483abf80a2629bf58455671f9ea5f45. Per-case original I/U,80 small packed predictions, prompt/freeze metadata, code snapshots and guard logs are protected. The abandoned0-byte capture was deleted only after complete archive validation. No retired F4/SAM1/info cache was rebuilt; fresh160 remains unopened. The research-paper objective remains unfinished.

Cached full-cohort control: `scripts/fixed_host_union_cpu.py` on oldCONF600 gives SAM61.091/FoRIS59.783/OR62.829/AND56.868. OR−SAM+1.738[−.297,5.381] fails its registered premise; OR−FoRIS+3.046[1.816,4.394] is useful as a naive control. All841 native masks/I-U are exact;2000 connected-photo paired draws, seed0. Reports and per-case records: `results/fixed_host_union_v1/`. No GPU repeat or new independent confirmation was performed.

`source_reconstruction_diagnostic.json` records legal Source-mask reconstruction by query-only prompts. Its association with Query quality is .589[.300,.783] across40 episodes, one per class. This is neither within-task candidate selection nor a method gain. CG-ICS already selects concepts by full Source-mask reconstruction; M2C optimises concept embeddings. The unresolved question is whether a different legal observation can disambiguate concepts that fit the same Source, not whether Source fit alone proves transfer.
