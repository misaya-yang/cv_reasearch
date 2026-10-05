# Prepared fixed object-crop CLS diagnostic

CPU preparation only. No encoder construction/forward or GPU run was performed. Root owns launch and GPU scheduling. The GT crop geometry is privileged; this is not a deployable proposal generator or a complete method.

85 encoded episodes,170 supplied-reference descriptors,197 fixed query region descriptors,367 image forwards. Fixed maximum batch4 within each case gives114 encoder calls. The first actual batch contains3 images; subsequent full4 batches and every batch check (B,4101,1024), FP32, finite descriptors and L2 normalization. Primary statistics remain the original18 paired episodes with40 missed and31 stray regions. All other126 regions are retained only for the supplement.

Actual CLS interface: TimmDINOv3.m is timm.models.eva.Eva. Use forward_features(x)[:,0,:] after the final model norm. Index0 is CLS; the four next prefix tokens are registers. The default avg pooling head is bypassed. Existing weights and model configuration remain unchanged.

Each square query crop maps the fixed1024-space box to original RGB coordinates, adds10percent context per edge and pads outside the image to ImageNet mean rounded for PIL RGB. Query crops are unmasked. The positive reference crop erases background; the negative full reference view erases foreground. Both reference masks reproduce the original sealed cov exactly in CPU preparation. Existing host build_transform(1024) is used.

## Root launch commands

Run only after root reviews the preparation seal and chooses the resource schedule:

```sh
PYTHONPATH=/root/demo4_cache/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 /root/miniconda3/bin/python -u /root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/launch/object_cls_dev241_v1/snapshot/scripts/diagnose_object_cls_dev241.py infer --out /root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/object_cls_dev241_v1
```

After ALL_CLS_DESCRIPTORS_SEALED:

```sh
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/root/demo4_cache/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/bin/python -u /root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/launch/object_cls_dev241_v1/snapshot/scripts/diagnose_object_cls_dev241.py score --out /root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs/object_cls_dev241_v1
```

No automatic GPU queue or launch is armed. Inference records model_setup_smoke.json, state.json and then sealed.json. Every compact descriptor is retained. Score refuses an incomplete seal, verifies descriptors/labels, reproduces the old controls, and computes paired2000 RandomState(0) connected-photo intervals. Existing partial or scored outputs are never overwritten.

## Resource estimate and unverified limits

At most4 FP32 1024-square transformed RGB tensors plus one case of PIL crops are held; no pixel or patch archive is written. Compact CLS storage is1,503,232 bytes before container overhead. CPU torch threads2; BLAS/OpenMP1. GPU memory, throughput and interference with the active4000 stream have not been measured. If a math-attention backend materializes B×heads×tokens×tokens in FP32, one batch4 score matrix is about4.30 GB, with possible additional temporary buffers and1.21 GB weights. The first actual GPU batch records peak allocation; root decides whether concurrency is acceptable and may pause only this diagnostic.

## Frozen dependencies

```json
{
  "wrapper": {
    "path": "/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/launch/object_cls_dev241_v1/snapshot/src/ics/data.py",
    "sha256": "7664bfcc0de9a9183296273a3ea57390e00147b82afe626f47bde1bf2383b165"
  },
  "model_config": {
    "path": "/root/demo4_cache/models/dinov3-vitl16-timm/config.json",
    "sha256": "a71f705b0074e173540d0bdbd3aa940fa8d7d3c6c7f020a683004c46ca605b24"
  },
  "model_weights": {
    "path": "/root/demo4_cache/models/dinov3-vitl16-timm/model.safetensors",
    "sha256": "45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941"
  },
  "timm_eva": {
    "path": "/root/demo4_cache/env/timm/models/eva.py",
    "sha256": "a8c9807ef5e8dabc725c1e2a89439760a900d6af6cdafefea18fc121173643a2"
  },
  "host_transform": {
    "path": "/root/autodl-tmp/demo8_local_verification/foris_source/utils/data.py",
    "sha256": "37211b2f55391694023cad806cdc9ec210944d32346ac1de0f9502df39731b53"
  }
}
```

Implementation SHA256: dafa29dd7922494c1a8d4514bf4662a986efd791e02292359ba8a8737caaa985
