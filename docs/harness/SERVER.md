# Server

One rented AutoDL instance at a time, shared by every agent session. The current address is in `STATUS.md`;
instances are recreated often, so do not trust an address written anywhere else.

## Limits

- One GPU with 32 GB, time-sliced: N busy processes each get about 1/N of it.
- No-card mode: 0.5 CPU core, 2 GB of memory. Only light CPU work fits; switching modes restarts the instance
  and kills background jobs.
- No access to Google Drive. Hugging Face only through `hf-mirror.com`. SAM3 is public on ModelScope.

## Layout

| Path | What | Rule |
|---|---|---|
| `/root/autodl-tmp/demo9_extent` | FoRIS runner, feature caches and packets every reading uses | read only |
| `/root/autodl-tmp/demo9_lang` | the session prepared on 2026-10-05 (scripts, CPU readers, banks, guard plans) | Claude's |
| `/root/autodl-tmp/demo9_transductive_ics` | image-isolated manifests (`results/extent_head_t1_isolated_v1`) | read only |
| `/root/autodl-tmp/datasets/ics` | COCO-20i, LVIS-92i, PASCAL-Part, PACO-Part, SUIM, lung X-ray | read only |
| `/root/autodl-tmp/demo4/INSID3` | INSID3 checkout; its dataset loaders build the transfer packs | read only |
| `/root/autodl-tmp/demo8_local_verification` | FoRIS and CRF sources used by every run | read only |
| `/root/autodl-tmp/sam3_preparation` | SAM3 checkpoint | read only |
| `/root/demo4_cache` | shared Python packages, DINOv3 weights, COCO masks | read only; never delete |
| `/root/autodl-tmp/cvpr_*` | another chat's held experiments | not yours |

Python is `/root/miniconda3/bin/python`. Never install into the base environment; use `pip install --target`
into your own directory. The live line runs with

    PYTHONPATH=/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions

## Running jobs

- Start detached: `nohup <cmd> > log 2>&1 < /dev/null &`. A foreground job dies with the connection.
- Queues run through `scripts/experiment_resource_guard.py`. It returns `DEFER_FOREIGN_GPU` while another
  job holds the GPU; the queue waits and retries. It issues `/usr/bin/shutdown` at the end.
- Stop only your own processes, by PID. `pkill -f` and `pgrep -f` match the ssh shell that runs them.
- Do not edit a bash queue script while it runs.
- Cap GPU memory (`DEMO4_GPU_FRAC`, 0.3 to 0.45). One GPU job per agent.

## Pitfalls that cost time

- A first run that completes zero cases is almost always a path: run the smoke stage first.
- Wait loops must not grep for "Error" (a harmless `httpcore` warning contains it). Grep for your own marker
  or for "Traceback".
- INSID3's encoder runs in bfloat16: encoding an image alone or in a pair changes its clustering. Cache with
  the batch composition of the released code and compare masks episode by episode.
- FoRIS and INSID3 both have top-level modules named `models` and `utils`. Import FoRIS first.
- Thousands of open `np.load` handles exceed the limit of 1024 open files. Close them.
- Caches are written in fold order: a prefix of a cache is fold 0 only.
- Keep at least 5 GB free on each disk. Before deleting anything outside your own directory, grep the other
  directories for the path.
