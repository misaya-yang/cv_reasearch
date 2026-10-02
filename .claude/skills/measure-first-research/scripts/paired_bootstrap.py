#!/usr/bin/env python3
"""Paired bootstrap interval for the difference between two methods evaluated on the same units.

Input: a JSON file holding a list of units (or {"units": [...]}). Every unit is a dict with the methods as keys:

  mean metric   {"id": "img_17", "A": 0.61, "B": 0.66}
                metric = mean over units
  ratio metric  {"id": "ep_3", "group": "person", "A": [inter, union], "B": [inter, union]}
                metric = mean over groups of sum(numerator) / sum(denominator); this is class-wise IoU

Optional key "stratum" (for example the fold): units are resampled inside each stratum and the metric is
averaged over strata, so every replicate keeps the original number of units per stratum.

  python paired_bootstrap.py units.json --base A --vs B [--draws 2000] [--seed 0] [--scale 100]

Prints both metrics, the paired difference with a 95% percentile interval, the share of replicates above zero,
and how many groups (or units, for a mean metric) moved up and down by more than --margin.
Standard library only.
"""
import argparse, json, random
from collections import defaultdict


def metric(units, key):
    """mean over groups of sum(num)/sum(den) for ratio values, plain mean for scalar values"""
    if isinstance(units[0][key], (list, tuple)):
        num, den = defaultdict(float), defaultdict(float)
        for u in units:
            g = u.get("group", "all"); num[g] += u[key][0]; den[g] += u[key][1]
        vals = [num[g] / den[g] for g in num if den[g] > 0]
        return sum(vals) / len(vals)
    return sum(u[key] for u in units) / len(units)


def stratified(strata, key):
    return sum(metric(s, key) for s in strata) / len(strata)


def per_group(units, key):
    if isinstance(units[0][key], (list, tuple)):
        num, den = defaultdict(float), defaultdict(float)
        for u in units:
            g = (u.get("stratum", 0), u.get("group", "all")); num[g] += u[key][0]; den[g] += u[key][1]
        return {g: num[g] / den[g] for g in num if den[g] > 0}
    return {i: u[key] for i, u in enumerate(units)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--base", required=True); ap.add_argument("--vs", required=True)
    ap.add_argument("--draws", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the metric, e.g. 100 for percent")
    ap.add_argument("--margin", type=float, default=0.02, help="change (unscaled) that counts as up or down")
    a = ap.parse_args()
    data = json.load(open(a.file)); units = data["units"] if isinstance(data, dict) else data
    by = defaultdict(list)
    for u in units: by[u.get("stratum", 0)].append(u)
    strata = [by[k] for k in sorted(by, key=str)]
    base, new = stratified(strata, a.base), stratified(strata, a.vs)
    rng = random.Random(a.seed); diffs = []
    for _ in range(a.draws):
        res = [[s[rng.randrange(len(s))] for _ in s] for s in strata]
        diffs.append(stratified(res, a.vs) - stratified(res, a.base))
    diffs.sort(); lo, hi = diffs[int(0.025 * a.draws)], diffs[min(int(0.975 * a.draws), a.draws - 1)]
    gb, gn = per_group(units, a.base), per_group(units, a.vs)
    up = sum(gn[g] - gb[g] > a.margin for g in gb if g in gn); down = sum(gb[g] - gn[g] > a.margin for g in gb if g in gn)
    k = a.scale
    print(f"units {len(units)}  strata {len(strata)}  draws {a.draws}")
    print(f"{a.base:>12s} {k * base:8.2f}")
    print(f"{a.vs:>12s} {k * new:8.2f}")
    print(f"  difference {k * (new - base):+8.2f}   95% interval [{k * lo:+.2f}, {k * hi:+.2f}]   "
          f"share of draws above zero {sum(d > 0 for d in diffs) / a.draws:.3f}")
    print(f"  up by more than {k * a.margin:g}: {up}   down: {down}   of {len(gb)}")


if __name__ == "__main__":
    main()
