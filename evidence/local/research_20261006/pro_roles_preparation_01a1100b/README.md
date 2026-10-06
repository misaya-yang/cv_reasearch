# Pro M3 deployable cache implementation

`src/ics/methods/pro_role_prediction.py` implements the fixed heldout/all-role/mean/zero rows from Pro report lines523–666. True adjacent Ward costs are updated after each merge; there is no score-based candidate filter. Held-out query assignments see only relation fingerprints and only afterwards receive their cross-image validation margin. Original dummy, five refinements, V clipping, κ=log(1+N), tiny-object bias and tree-DP tie order are retained.

Run independently on a bound cache manifest:

```sh
python scripts/run_pro_role_prediction.py infer --manifest rows.json --root /actual/cache/root --out /new/run/path --workers 4 --threads 1
python scripts/run_pro_role_prediction.py score --out /new/run/path
```

Manifest rows use `c,fold,support,query,feature_export,packet_export` and optional canonical `support_photo_id/query_photo_id` plus complete `evaluation_controls`. `feature_export` contains q/r4096×1024, and `packet_export` contains cov/raw full-FoRIS preCRF score64×64. Inference sees only these arrays. If the reference does not furnish any component with ≥4 nonempty roles, it lazily opens the bound full-FoRIS packed native1024 mask. It never fabricates CRF-native from score. Scoring alone reads packed `truth` after all predictions are sealed. All repeated source occurrences remain in the output.

Four packed1024 masks and token fields are emitted under the names in `method.json`; receipts bind cache hashes, actual Hungarian calls, Ward/matching/complete wall time, process RSS high-water, dummy/degenerate-direction counts, template role counts, and fallback reason. Untruncated node validation maxima and an evaluated-node mask are saved in `fields`; skipped whole nodes still have every descendant evaluated. The shared score function reports class-summed I/U, canonical-photo paired bootstrap and four pixel edit types.

Two small check scripts are included. `checks.json` verifies dynamic Ward against an independent naive merger on25 tiny problems and DP against enumeration on24 trees, plus dummy/heldout/fallback boundaries. `runner_check.json` verifies two CPU workers,4 repeated synthetic occurrences, native fallback masks/scoring and an unreadable GT sentinel. The active structural branch is checked only on a2×4 toy grid; no full4096 active tree or natural-image experiment was run. Actual full cost remains unknown.

Specific contract resolutions/limits:

- Pro does not give an initial farthest-point seed. Minimum original token ID is fixed for the initial cluster seed and first template, then exact farthest-point with minimum-ID ties; no data fitting or search occurs.
- BG uses the complement of effective coarse foreground (coverage≥.5 with maximum-coverage tiny-reference fallback).
- This cache adapter outputs/scores1024 work masks. Pro's second interpolation to original H/W is not implemented as an unannounced scoring change; bind original dimensions and declare that evaluation stage separately.
- The source of q/r and score must be bound externally. The adapter does not certify an old FP16/projection cache as Pro's FP32 native producer.
- Full-native fallback is mandatory when triggered; no background/query-GT label is inferred from a missing asset.
- `peak_owned_rss_bytes` in the seal is an explicitly labelled conservative worker-high-water estimate for scorer compatibility, not measured simultaneous RSS. Process high-water values and complete timings remain available in each receipt.
- Nine current candidate methods, their shared entry, PLAN, STATUS and the Pro original are untouched.
