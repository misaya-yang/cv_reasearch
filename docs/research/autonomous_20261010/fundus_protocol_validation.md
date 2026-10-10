# Independent Fundus released-protocol audit — 2026-10-10

The completed Fundus baseline matches the pinned released loader's default200 one-shot seed0 draws and released CLI1024 evaluation geometry. No segmentation rerun, query labels, scoring output or prediction-mask pixels were read. The compute agent independently owns full hashes and I/U verification.

Evidence: `evidence/local/autonomous_20261010/scene_model_validation/fundus_protocol_audit.json`; reproducer: `fundus_protocol_audit.py` in that directory. Completed run: `cv_data/a/fundus_foris_seed0_20261010/`. The run and shared code were not edited.

The source is FoRIS commit `1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e`; frozen Fundus loader SHA256 `f5c3b51a682f96f8631407d56db45427a1796e53b897576e70fffc99a211e919`. Source, external dependency, manifest, config, profile and basis hashes passed. Manifest SHA256 is `5e89b914773c4bc32d3e819fee5b929fbf606ae710335caaeb8910ef691f937f`.

## Episode and model protocol

The unchanged `DatasetFundus` defaults to `test/Original/*.png` paired by filename with `test/Ground truth/*.png`. Its default episode override is `None`, so the verified200-image paired pool gives length200. Calling its actual `sample_episode` after NumPy seed0 reproduced every frozen query/reference identity exactly. The one reference differs from the query filename. Natural repeated query draws remain:200 episodes cover120 distinct query photographs, rather than exhaustively visiting every photograph once. All class IDs are0; the adapter's fold-1 denotes this single nonfold dataset without changing the pooled statistic.

The baseline uses DINOv3-L/16, frozen FP32 MPS batch2 raw O24,64×64 patch grid and1024 channels. All400 recorded R/Q request metadata agree with the common profile, payload/tensor identities and173 unique keys. Frozen encoder/observer/preprocessing source digests match the profile; native U500 basis and tau.6 are unchanged. Actual full weight/payload verification belongs to the independent compute audit.

The baseline calls original `set_reference`, `set_target`, `segment`; source stage-observation wrappers return each original result unchanged. The CPU cache adapter has no encoder fallback. Query labels are denied during producer and CPU inference; all outputs seal before scoring.

## Exact CLI and additional-frame geometry

| Step | Released/current behavior |
| --- | --- |
| RGB input | Original RGB square resize to1024; tensor conversion and ImageNet normalization |
| Reference mask | Released Fundus loader thresholds L values at128, then model nearest-resizes the binary mask to1024 |
| CLI prediction | Model factory explicitly sets `resize_to_orig_size=False`; full FoRIS/CRF result stays1024×1024 |
| CLI GT | Released `inference.py` directly nearest-resizes the raw binary GT to the prediction shape, then uses `>.5` |
| Additional original output | Binary CLI mask bilinear-resized with `align_corners=False` to2048×2048, then `>.5` |
| Additional original GT | Raw binary GT directly resized to2048×2048; reported separately |

Every saved NPZ header has exactly131,072 uint8 packed bytes for CLI1024² and524,288 for original2048², plus `original_hw=[2048,2048]`. Only headers and the small geometry array were read; mask bit arrays were not decoded. All200 manifest query RGB and GT shape metadata are2048×2048. The Fundus loader supplies no ignore map.

The single-class metric is pooled foreground intersection divided by `max(pooled union,1)`, consistent with released `AverageMeter`; it is not the arithmetic mean of episode IoUs. No real I/U was recomputed here.

Relevant frozen code locations are `datasets/fundus.py:42` for episode length, `:75` for sampling, `models/__init__.py:97` for CLI size, `inference.py:83` for direct GT resize, and frozen `scripts/run_extended_foris.py:236` for the additional original renderer.

## Scope and limits

The actual model factory takes precedence over README's general original-resolution note. Bare CLI defaults do not enable CRF; this complete FoRIS baseline explicitly enables the paper's `--crf-mask-refinement` choice and retains the original solver/settings through the local CPU backend.

The generic released Fundus loader does not itself pin an archive hash or immutable FIVES image list. Separate source-pool receipts identify and verify the local FIVES v1 test200 cohort. Exact released-source/default-draw/geometry compatibility therefore establishes this controlled baseline, not reproduction of the paper's table value or CUDA numerical equivalence.

Audit guards recorded0 annotation/scoring accesses and0 completed-run writes. The checked config/seal/manifest/inference hashes remained unchanged. No mask pixels, feature arrays, encoder or segmentation pipeline were executed by this complementary audit.
