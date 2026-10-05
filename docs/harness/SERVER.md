# Server operations

This is the only server runbook. It describes the recorded environment, not live availability or authorization.
Live endpoint checked in the 2026-10-05 preparation task: `ssh -p 48002 root@connect.westd.seetacloud.com`.
Preparation mode was no-card, cgroup `cpu.max=50000 100000` (0.5 CPU), `memory.max=2147483648` (2 GiB).
Host-level nproc/free are misleading for this container. Its no-card nvidia-smi is a stub, so the expected
future GPU/12-core/60GB configuration remains unverified. SSH initially timed out, then succeeded without
changing the endpoint. No experiment process was observed in the read-only process inventory.
All 241 manifest reference/query RGB, mask, feature and packet paths exist; DINOv3 model.safetensors is
1,212,347,640 bytes. Existence/size is not model-content or runtime verification.
Full receipt: [server inventory](../../evidence/local/research_20261005/server_inventory.json).
The user subsequently stopped all CPU testing; only code preparation and static checks continue.

## Active GPU session

The user enabled GPU mode and resumed experiments. Live inventory reports RTX 4080 SUPER, 32,760 MiB,
12 CPU quota (`1200000 100000`), 66,571,993,088 memory bytes and no foreign GPU jobs at launch.
The original hostname occasionally fails through the local fake-IP resolver. For this session, its live
AliDNS answer was 36.103.198.204; using `HostName` with the original host-key alias and strict checking
restored access. This is a temporary transport observation, not a permanent DNS or SSH configuration.
Owned workspace: `/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9`.
The user subsequently removed artificial duration/round/idle cutoffs; the owned pipeline used
`scripts/experiment_pipeline.py` without those limits. Shutdown was not armed. After the GPU-idle
correction all owned queues finished; the user later paused and powered off for network repair. The current user has now resumed
pipeline tracking and authorized running the prepared work after the existing GPU is enabled.
Ordinary hostname SSH works again; the latest check still reports no-card 0.5 CPU / 2 GiB.
The isolated root queue is `launch/resume_dev241_20261005_v1`; do not confuse other live workers
with its owned processes, and do not retry from a mere SSH observation timeout.
This inventory is historical. Do not alter another worker's processes or assets.

## One resource policy

Use the budget and lifecycle approved for the current session. There is no standing “keep on forever” rule
and no permission to shut down another worker's session. Before an authorized queue, state its scientific
comparison and measured resource expectations; unknown price/runtime stays unknown. The current task
removed artificial time/round/idle caps; do not restore them from this older runbook.
Prepare on CPU/no-card, then use a short real-pipeline smoke followed immediately by the authorized stages.
The current pipeline is `scripts/experiment_pipeline.py`; the finite resource guard is historical.
Neither runner is permission to resume held compute.

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
