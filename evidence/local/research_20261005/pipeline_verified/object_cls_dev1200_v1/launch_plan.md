# Frozen CLS1200: root-owned serial launch plan

CPU preparation and independent checks are complete; no new encoder/GPU run has been launched. Root owns the post4000 serial schedule. Use stage_plan.json for exact argv, environment, source hashes, prerequisite seals and expected state.

Full sealed1200 provides332 eligible cases/736 regions(378 missed GT,358 stray), with77 paired episodes/296 regions. Reuse the original85 cases'367 compact vectors exactly; encode247 new cases,494 reference views and539 query crops(1033 images),323 batch-at-most4 calls. Store1400 compact CLS vectors(5,734,400 bytes before container overhead). Same weights, FP32/no autocast, transform, masks, square crop plus10percent context and score orientation remain frozen. All1200 source masks/IU/RGB headers/reference coverage were verified; every legacy region/crop identity matches and every preserved CLS margin reconstructs exactly.

The measured original367-view run took153.29 seconds alongside the then-active pipeline and peaked at2,080,604,672 CUDA bytes. New serial throughput remains unmeasured. This preparation opens only the fully sealed1200 sources and the completed original241 diagnostic, never unsealed4000 GT.

After the producer benchmark finishes on the released GPU:

```sh
PYTHONPATH=/root/demo4_cache/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/bin/python -u /root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/launch/object_cls_dev1200_v1/snapshot/scripts/diagnose_object_cls_dev1200.py infer --out /root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/object_cls_dev1200_v1
```

After ALL_CLS_DESCRIPTORS_SEALED:

```sh
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/root/demo4_cache/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/bin/python -u /root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/launch/object_cls_dev1200_v1/snapshot/scripts/diagnose_object_cls_dev1200.py score --out /root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/object_cls_dev1200_v1
```

Full1200 compares CLS and stored nearest-reference margin on the identical77 paired episodes. The original241 raw/projected controls and CLS remain paired on their original18 episodes and are explicitly separate. No full1200 raw control exists, and no cross-cohort estimator comparison is made. GT boxes are privileged diagnostics, not legal proposals, independent confirmation or complete method results. The original241 dafa29 snapshot is immutable; this new snapshot only generalizes cohort counts and copies sealed legacy vectors.
