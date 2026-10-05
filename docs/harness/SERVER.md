# Server operations

This is the only server runbook. It describes the recorded environment, not live availability or authorization.
Last recorded endpoint (2026-10-05): `ssh -p 48002 root@connect.westd.seetacloud.com`.
Last recorded mode: no-card. Neither was rechecked during repository cleanup; instances and ports can change.

## One resource policy

Use the budget and lifecycle approved for the current session. There is no standing “keep on forever” rule
and no permission to shut down another worker's session. Before an authorized queue, state the bounded total
GPU time/cost and scientific comparison; unknown price/runtime stays unknown, not an invented estimate.
Prepare on CPU/no-card, then use a short real-pipeline smoke followed immediately by the authorized stages.
Run through `scripts/experiment_resource_guard.py`.

The guard waits for foreign GPU jobs. With explicit shutdown authorization it invokes `/usr/bin/shutdown`
at the end of the finite session, subject to foreign-job checks and the shared KEEP_ON hold.
A recent `/root/autodl-tmp/KEEP_ON` holds shutdown for 15 minutes; it is not permission to extend paid rental.
A shutdown request is not billing-stop confirmation. No-card operation does not justify starting a GPU.

Check `nvidia-smi` before remote compute. Stop only owned PIDs; do not edit a running queue or use broad
process-name killing. Keep the base Python environment and shared assets unchanged. Use an agreed CPU/memory
quota; no-card was last recorded as only 0.5 core / 2 GB, unlike the GPU-mode machine.

## Recorded dependencies

| Path | Role |
|---|---|
| `/root/autodl-tmp/demo9_extent` | Shared FoRIS runner, feature packets and evidence; read only unless owned by this task |
| `/root/autodl-tmp/demo9_lang` | Claude's 2026-10-05 queue and reports |
| `/root/autodl-tmp/demo9_transductive_ics` | Image-isolated manifests and SAM3 results |
| `/root/autodl-tmp/datasets/ics` | Shared datasets |
| `/root/autodl-tmp/demo4/INSID3` | INSID3 code and dataset loaders |
| `/root/autodl-tmp/demo8_local_verification` | Shared FoRIS/CRF sources and runtime extensions |
| `/root/autodl-tmp/sam3_preparation` | SAM3 code/checkpoint |
| `/root/demo4_cache` | Shared Python packages, weights and COCO masks |
| `/root/autodl-tmp/cvpr_*` | Other chats' held work; not this task's outputs |

Recorded Python: `/root/miniconda3/bin/python`; FoRIS runtime search path:

    /root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions

Availability must be checked for the selected run. Old paths are not permission to download missing assets.
The user deleted local `demo8_local_verification`. It was not recreated. The remote paths above are
historical inventory only; this cleanup did not check or modify them. A future run needs an existing
FoRIS checkout and CRF runtime explicitly configured through its manifest/environment.

## Known execution pitfalls

- A real smoke must check input paths and CRF; synthetic fixtures previously passed before a zero-episode run.
- bfloat16 separate versus paired encoding can change INSID3 clustering. Preserve the released batch composition.
- FoRIS and INSID3 share top-level names `models` and `utils`; load FoRIS first in the existing runner.
- Close `np.load` handles. Cache prefixes may contain only fold 0.
- Use the owned job's exit status and expected output; a generic “Error” string also matches harmless warnings.
- An old preflight receipt is tied to its plan, code and state-file path; “prepared” does not mean a real run passed.
