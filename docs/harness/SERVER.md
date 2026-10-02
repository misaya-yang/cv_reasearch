# Server

One rented machine, shared by every agent session.

```
ssh -p 57510 root@connect.westc.seetacloud.com
```

## Limits

- One GPU with 32 GB, time-sliced: N busy processes each get about 1/N of it.
- System disk `/` 30 GB. Data disk `/root/autodl-tmp` 50 GB. Memory 90 GB (not what `free` shows). 25 CPU cores.
- No access to Google Drive. Hugging Face only through `hf-mirror.com`; its API needs a `User-Agent` header.

## Layout

| Path | What | Rule |
|---|---|---|
| `/root/autodl-pub/` | public datasets (COCO14, COCO2017, ADE20K, VOC, ImageNet, ...) | read only |
| `/root/demo4_cache/` | shared by the live direction: `env/` (Python packages), `models/` (DINOv3 ViT-L), `data/` (COCO-20i masks) | read only; do not delete |
| `/root/autodl-tmp/<direction>/` | code mirror of a repository directory | owned by the session that created it |
| `/root/<direction>_cache/` | feature caches of one direction | delete when the run is done |

Python: `/root/miniconda3/bin/python` (`python3` is not on `PATH`). Never install into the base environment.
Use the shared packages with `PYTHONPATH=/root/demo4_cache/env`, or install into your own folder with
`pip install --target`.

Folders as of 2026-10-02:

| Folder | Size | State |
|---|---:|---|
| `/root/autodl-tmp/demo9`, `/root/autodl-tmp/demo4` | 14 MB | live code mirror used by `PLAN.md` |
| `/root/autodl-tmp/demo9_transductive_ics` | 2.0 GB | live, Codex session (open-pool runs) |
| `/root/autodl-tmp/demo8_local_verification` | 49 MB | closed, but `foris_source/` is used by `run_blackbox.py`; keep |
| `/root/demo4_cache` | 8.3 GB | shared assets; keep |
| `/root/autodl-tmp/datasets/ics/` | growing | benchmark data in INSID3's layout, built by `scripts/get_data.sh`; shared, read only |
| `/root/demo2_cache`, `/root/autodl-tmp/demo2_pilot`, `demo2_where_what_decoding` | 2.6 GB | closed direction; check references, then delete |
| `/root/autodl-tmp/gic_validation`, `demo5_*`, `demo6_*`, `demo7` | 0.2 GB | closed directions; check references, then delete |

## Running jobs

- Look first: `nvidia-smi`, then `ps -eo pid,etimes,args | grep scripts/`. Read `STATUS.md` for who owns what.
- Cap your memory (`torch.cuda.set_per_process_memory_fraction`; the live code reads `DEMO4_GPU_FRAC`, 0.3 to
  0.45). One GPU job per agent.
- Start detached: `ssh -n ... 'cd <dir> && nohup <cmd> > log 2>&1 < /dev/null &'`. A foreground job dies with
  the connection.
- Wait with a remote loop (`until grep -q <marker> log; do sleep 20; done`), not with local `sleep`.
- Stop only your own processes, by PID, after reading the command line. `pkill -f` and `pgrep -f` match the
  ssh shell that runs them.
- Under contention each GPU-to-host sync (`.item()`, `.cpu()`, boolean indexing) costs a scheduling round
  trip. Batch them.

## Pitfalls that cost time

- Wait loops must not grep for "Error": a harmless `httpcore` warning contains it. Grep for your own marker
  or for "Traceback".
- `np.trapz` is gone in the installed numpy; use `np.trapezoid`.
- INSID3's encoder runs in bfloat16. Encoding an image alone or in a (reference, query) pair changes the
  features enough to change its clustering. Cache features with the same batch composition as the released
  code and check item-level identity.
- FoRIS and INSID3 both have top-level modules named `models` and `utils`. Import FoRIS first.
- Inline Python through nested ssh quoting breaks on f-strings. Copy result files and compute locally.

## Cleanup

- Delete caches, weights and datasets in the same session they stop being needed. Keep small JSON and logs.
- Before deleting anything outside your own folder: `grep -rl <path> /root/autodl-tmp/demo*`. Other sessions
  reference shared caches.
- Keep at least 5 GB free on each disk.
