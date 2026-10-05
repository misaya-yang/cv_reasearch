# Public FoRIS protocol: verified contract and paper/code mismatch

Owner: receiving controller, 2026-10-05. Read-only official-source inspection through
authenticated gh CLI. No model/data download or new public benchmark inference.

Official repository revision checked:
`1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e`.

| Contract | Observed primary source |
|---|---|
| COCO episodes | `datasets/coco.py:19`:1000; length returns this number. Four class folds, c=fold+4*v. |
| Sampling | `datasets/coco.py:82`:uniform class, then target from that class's metadata, then a distinct reference. Preserve RNG call order and metadata order. |
| Seed | `opts.py:109`:default0; `inference.py:28` seeds torch,numpy,Python. DataLoader worker configuration must also be preserved. |
| Encoder/input | Large DINOv3,1024,one shot,500 positional components,tau.6; CRF enabled in the published command. |
| Metric | `utils/metrics.py`:sum foreground I/U by class, then mean class IoU; same as the project's class-summed estimand. |
| Official CLI renderer | `models/__init__.py:97` explicitly sets `resize_to_orig_size=False`. `inference.py:83` resizes GT by nearest to the prediction shape. This is a1024 prediction-grid evaluation. |
| Paper text | AppendixA describes original-resolution interpolation and CRF. The current model code applies CRF before optional original-size interpolation. |

Sources: [paper Appendix A](https://arxiv.org/html/2609.03384v1#A1),
[COCO loader](https://github.com/Xi-Mu-Yu/FoRIS/blob/1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e/datasets/coco.py),
[model factory](https://github.com/Xi-Mu-Yu/FoRIS/blob/1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e/models/__init__.py),
[evaluation entry](https://github.com/Xi-Mu-Yu/FoRIS/blob/1aa02a11ef5f6673ed7a8a666ccf7d5586998d9e/inference.py).

Do not resolve this mismatch by assuming that original resolution is the sole public
protocol or by directly subtracting a subset score from the published60.9. Retain
separately named4000-episode CLI-grid and original-resolution readouts, using identical
episode identities and frozen complete methods. Reproduce the paired FoRIS baseline
in each renderer. Any claim of matching the published number needs the associated
code/version/renderer agreement, not just the same number of episodes.

The deployed FoRIS source directory contains the model utilities but lacks the public
evaluation entry and COCO loader. Existing assets can still support a streaming adapter,
but official episode-generation parity and encoder/runtime parity must be verified
with a small real run before full evaluation. This is preparation, not a score result.

Store bounded in-flight episode features only; retain sealed packed final masks and
per-episode I/U/provenance. The candidate's original-resolution renderer must be frozen
before evaluation. A new CRF applied to a fixed Astra output is a separately named
method change, rather than silently replacing its supplied1024 prediction rule.

Audit overlap with every previously inspected cohort, including DEV241 and existing600.
The standard4000 benchmark can contain exposed episodes/photos; report this explicitly
and separate any genuinely unseen photograph-isolated readout. Do not choose parameters
or operator paths on either confirmation readout.
