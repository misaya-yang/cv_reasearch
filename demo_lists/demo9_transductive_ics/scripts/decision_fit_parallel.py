#!/usr/bin/env python3
"""scripts/decision_fit.py with the arms spread over several processes on one GPU (the fits are bound by one CPU core
each), then one process that merges the development rows, freezes the choice and reads the confirmation episodes.

  python scripts/decision_fit_parallel.py --cache cache/decision_v1 --out results/decision_v1/fit [--workers 6] [arguments of decision_fit.py]
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decision_fit import ARMS, CURVE_ARMS  # noqa: E402

COST = dict(cut=1, pixel=2, mlp=2, context=2, conv=3, convctx=3, unroll=5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--curve", default="150,300,600,1200")
    p.add_argument("--curve-arms", default=",".join(CURVE_ARMS))
    a, rest = p.parse_known_args()
    if (a.out / "report.json").exists():
        raise SystemExit("fresh output directory required")
    arms, curved = a.arms.split(","), set(a.curve_arms.split(",")) if a.curve else set()
    cost = {k: COST[k.split(":")[0]] * (2.2 if k in curved else 1) for k in arms}
    groups = [[] for _ in range(min(a.workers, len(arms)))]
    for k in sorted(arms, key=lambda k: -cost[k]):  # the heaviest first, each into the lightest group
        min(groups, key=lambda g: sum(cost[x] for x in g)).append(k)
    base = [sys.executable, str(HERE / "decision_fit.py"), "--cache", a.cache] + rest
    jobs = []
    for i, g in enumerate(groups):
        mine = [k for k in g if k in curved]
        cmd = base + ["--out", str(a.out / ("part%d" % i)), "--arms", ",".join(g), "--no-confirm",
                      "--curve", a.curve if mine else "", "--curve-arms", ",".join(mine) if mine else ",".join(g)]
        (a.out / ("part%d" % i)).mkdir(parents=True, exist_ok=True)
        jobs.append(subprocess.Popen(cmd, stdout=open(a.out / ("part%d.log" % i), "w"), stderr=subprocess.STDOUT))
    codes = [j.wait() for j in jobs]
    if any(codes):
        raise SystemExit("a part failed: %s" % codes)
    last = base + ["--out", str(a.out), "--select-from", ",".join(str(a.out / ("part%d" % i)) for i in range(len(groups)))]
    sys.exit(subprocess.call(last))


if __name__ == "__main__":
    main()
