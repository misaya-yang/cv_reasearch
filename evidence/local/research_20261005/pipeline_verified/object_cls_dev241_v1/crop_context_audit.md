# Fixed object CLS crop context audit

Pure post-hoc geometry on all197 unchanged crops. No fit, crop change, sample filtering, subgroup AUROC, sign reversal, threshold search or encoder.

| fixed negative cohort | n | contains target GT | target in unexpanded bbox | target only after square/context | CLS>0 FP: containing | CLS>0 FP: empty |
|---|---:|---:|---:|---:|---:|---:|
| All83 stray | 83 | 14 | 2 | 12 | 8/14 | 51/69 |
| Primary31 negative | 31 | 4 | 1 | 3 | 3/4 | 17/27 |

All14 GT-containing stray crops, including the4 primary negative crops, contain target pixels outside the selected prediction component in the original1024 diagnostic frame. No selected stray component overlaps target GT. Two all-cohort cases already contain target in the component bounding rectangle; the other12 acquire target through the fixed square/context expansion (primary:1 and3).

The primary20 fixed0-margin false positives consist of3 GT-containing and17 target-GT-empty crops. Thus85% of primary false positives remain without any target GT in the crop. Crop-content/region-ownership confusion is present, but target context alone cannot explain most fixed-rule errors. The GT groups are descriptive; no subgroup performance is promoted to a method result.

## Target GT occupancy

| cohort | mean crop fraction | median | maximum | mean among containing crops | median among containing crops | pixel-pooled fraction |
|---|---:|---:|---:|---:|---:|---:|
| All83 stray | 0.996% | 0.000% | 35.193% | 5.904% | 2.273% | 4.196% |
| Primary31 negative | 0.528% | 0.000% | 12.346% | 4.093% | 1.853% | 0.150% |

Fractions count original-annotation target pixels within the actual integer square RGB crop before Resize1024, divided by the full padded crop area. Out-of-image padding contributes zero GT. Per-crop fractions over valid RGB pixels are retained separately in JSON. Target-GT-empty means no pixels of the episode target class, not an empty image or absence of other objects.

Component ownership is evaluated in its original1024 connected-region frame, using working pixel centers inside the fixed original-coordinate crop. This avoids inventing a native-resolution prediction component. Original-annotation and working-frame GT-presence labels agree on all83 negatives.

All85 original annotation masks reproduce existing1024 truth, all197 crops match the frozen10percent mapping, and all85 sealed descriptor hashes plus all197 CLS scores/fixed0 signs match (maximum score difference0). Source hashes, per-crop counts and score links are in crop_context_audit.json. The original18-episode primary CLS AUROC remains0.592 [0.389,0.788], unresolved; no subgroup AUROC was computed.
