#!/usr/bin/env python3
"""Build the decision cache with several processes on one GPU: the principal components first, then the shards.

  python scripts/decision_cache_parallel.py --workers 4 -- --suite DIR --packets P --out cache/decision_v1 [--manifest M --kinds train]
Everything after `--` goes to scripts/decision_cache.py. The common report is written only when every shard has
finished and every episode of the manifests is on disk.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    a = p.parse_args()
    rest = [x for x in a.rest if x != "--"]
    get = lambda flag, default=None: rest[rest.index(flag) + 1] if flag in rest else default
    out, suite, extra = Path(get("--out")), Path(get("--suite")), get("--manifest")
    kinds = get("--kinds", "train,dev,confirm").split(",")
    base = [sys.executable, str(HERE / "decision_cache.py")] + rest
    start = time.monotonic()
    if not (out / "pca.pt").is_file():
        only = [x for i, x in enumerate(rest) if not (x == "--kinds" or (i and rest[i - 1] == "--kinds"))]
        subprocess.run([sys.executable, str(HERE / "decision_cache.py")] + only + ["--kinds", "none"], check=True)
    jobs = [subprocess.Popen(base + ["--resume", "--shard", "%d/%d" % (i, a.workers)]) for i in range(a.workers)]
    codes = [j.wait() for j in jobs]
    want = {}
    if "train" in kinds:
        want["train"] = json.loads(Path(extra or suite / "train_episodes.json").read_text())["episodes"]
    if "dev" in kinds:
        want["test"] = json.loads((suite / "dev_episodes.json").read_text())["episodes"]
    if "confirm" in kinds:
        want["confirm"] = json.loads((suite / "confirm_episodes.json").read_text())["episodes"]
    missing = {k: sum(not (out / k / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"]))).is_file() for r in rows) for k, rows in want.items()}
    tag = "_" + Path(extra).stem if extra else ""
    shards = [json.loads(f.read_text()) for f in sorted(out.glob("report%s_shard*.json" % tag))]
    state = "COMPLETED" if not any(codes) and not any(missing.values()) and all(s["state"] == "COMPLETED" for s in shards) else "ERROR"
    report = dict(state=state, returncodes=codes, missing=missing, episodes={k: len(v) for k, v in want.items()},
                  native_checks=sum(s["native_checks"] for s in shards), native_bit_identical=sum(s["native_bit_identical"] for s in shards),
                  middle_layers_captured=sum(s["middle_layers_captured"] for s in shards), written=sum(s["train"] + s["test"] + s["confirm"] for s in shards),
                  peak_bytes=max([s.get("peak_bytes", 0) for s in shards] + [0]), workers=a.workers, elapsed_s=time.monotonic() - start,
                  layers=shards[0]["layers"] if shards else None)
    (out / ("report%s.json" % tag)).write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)
    sys.exit(0 if state == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
