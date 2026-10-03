#!/usr/bin/env python3
"""One table of everything the decision queue has produced so far (CPU, reads small JSON files only).

  python scripts/decision_summary.py --root results/decision_v1 [--main results/decision_main] [--scale results/scale_align_v0]
"""
import argparse
import json
from pathlib import Path


def get(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def line(name, v, key="over_foris"):
    folds = " ".join("%+.1f" % x for x in v["per_fold"]) if "per_fold" in v else ""
    return "| %s | %+.2f | [%+.2f, %+.2f] | %s |" % (name, v[key], v["ci95"][0], v["ci95"][1], folds)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--main", type=Path)
    p.add_argument("--scale", type=Path)
    a = p.parse_args()
    for tag in ("fit", "fit2", "fit_agnostic"):
        d = get(a.root / tag / "report.json")
        if not d:
            continue
        print("\n### %s: %s, %d training episodes, %d development episodes (patch level)\n" % (tag, d["state"], d["train"], d["development"]))
        print("| row | over FoRIS | 95% interval | folds 0-3 |\n|---|---|---|---|")
        for k, v in d["reference"].items():
            if k != "FoRIS":
                print(line(k, v))
        for k, v in sorted(d["arms"].items(), key=lambda kv: -kv[1]["over_foris"]):
            print(line("%s (removal only %+.2f, addition only %+.2f; epochs %s; %d weights)" % (
                k, v["removal_only"]["over_foris"], v["addition_only"]["over_foris"], [f["epochs"] for f in v["fits"]], v["fits"][0]["weights"]), v))
        for k, v in d["curve"].items():
            print(line("curve " + k, v))
        c = d.get("confirmation")
        if c:
            print("\nchosen on development: %s. Confirmation, %d episodes, FoRIS %.2f, verdict %s\n" % (d.get("chosen"), c["episodes"], c["FoRIS"]["miou"], c.get("verdict")))
            print("| row | over FoRIS | 95% interval | folds 0-3 |\n|---|---|---|---|")
            for k, v in c.items():
                if isinstance(v, dict) and "over_foris" in v and k != "FoRIS":
                    print(line("%s (removal only %+.2f)" % (k, v["removal_only"]["over_foris"]), v))
    sets = [(a.root / ("infer_%s_%s" % (t, s)), "read-out %s, %s" % (t, s)) for t in ("1", "2") for s in ("dev", "confirm")]
    if a.main:
        sets.append((a.main / "infer", "main table, standard 4 x 1000"))
        d = get(a.main / "fit" / "report.json")
        if d:
            print("\n### main-table fit: %s, %d training episodes; development rows (patch level)\n" % (d["state"], d["train"]))
            for k, v in d["arms"].items():
                print(line(k, v))
            for k, v in d["curve"].items():
                print(line("curve " + k, v))
    rows = [(get(f / "report.json"), name) for f, name in sets]
    if any(r for r, _ in rows):
        print("\n### inside the pipeline: class mIoU at original resolution, FoRIS's refinement\n")
        print("| set | episodes | FoRIS | arm | gain | 95% interval | folds 0-3 | up / down | lose > 10 | before refinement |\n|---|---|---|---|---|---|---|---|---|---|")
        for d, name in rows:
            if d:
                for k, v in d["rows"].items():
                    print("| %s (%s) | %d | %.2f | %s | %+.2f | [%+.2f, %+.2f] | %s | %d / %d | %d | %+.2f |" % (
                        name, d["arm"], d["episodes"], d["class_miou"]["native"], k, v["gain"], *v["ci95"],
                        " ".join("%+.1f" % x for x in v["per_fold"]), v["up"], v["down"], v["lose_more_than_10"], d["before_refinement"][k]["gain"]))
    if a.scale:
        d = get(a.scale / "run" / "analysis.json") or get(a.scale / "analysis.json")
        if d:
            print("\n### scale alignment\n")
            print(json.dumps({k: d[k] for k in d if k in ("state", "episodes", "verdict", "overall", "gate")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
