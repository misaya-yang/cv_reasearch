#!/usr/bin/env python3
"""The two readings that decide what the paper is, as one table and one verdict each (CPU, small JSON files only).

  python scripts/decision_pivot_verdict.py --root results/decision_pivot_v1 --stage d1        # after the three fits
  python scripts/decision_pivot_verdict.py --root results/decision_pivot_v1 --stage all       # at the end

D1, annotation-free fit: the read-out fitted ONLY on constructed pairs (three training sets: `same`, `paste`, `all`),
read on DEV241 at patch level like the label fit was (label fit: linear +2.19, convctx:layers +4.12; the bare host
is 0 by definition). Rule, written before the run:
  PASS     a convctx:layers fit reaches +2.0 with the lower end of its interval above 0
  PARTIAL  the best arm reaches +1.0 with the lower end above 0 (still first in the training-free column, by less)
  FAIL     otherwise
On PASS or PARTIAL the best (set, arm) is written to d1_choice.json and is read once on CONFIRM600 inside the
complete pipeline; that reading, not the development number, is the result.
The cosine with the label-fitted constants is reported but does not decide: a rule that only keeps the host's score
already has cosine 0.81 with the label fit in raw units.

T2, transfer without refitting: the read-out fitted on COCO base classes, read inside the complete pipeline at
original resolution on packs of other benchmarks. Rule: PASS when at least three of the packs gain +1.5 or more with
the lower end above 0 and no pack loses with its whole interval below 0; FAIL when fewer than two packs gain with the
lower end above 0; PARTIAL otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

SETS, ARMS = ("same", "paste", "all"), ("pixel:relations", "convctx:layers")
PACKS = ("lvis", "pascal_part", "paco_part", "suim", "lung")


def get(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def d1(root):
    rows, best = [], None
    for s in SETS:
        rep = get(root / ("fit_" + s) / "report.json")
        cos = (get(root / ("compare_%s.json" % s)) or {}).get("cosine_raw")
        for arm in ARMS:
            v = (rep or {}).get("arms", {}).get(arm)
            if not v:
                continue
            row = dict(set=s, arm=arm, gain=v["over_foris"], ci95=v["ci95"], per_fold=v.get("per_fold"), train=rep.get("train"),
                       epochs=[f.get("epochs") for f in v.get("fits", [])], cosine=cos if arm == ARMS[0] else None,
                       model=str(root / ("fit_" + s) / "models" / (arm.replace(":", "_") + ".pt")))
            rows.append(row)
            if row["ci95"][0] > 0 and (best is None or row["gain"] > best["gain"]):
                best = row
    conv = [r for r in rows if r["arm"] == ARMS[1] and r["gain"] >= 2.0 and r["ci95"][0] > 0]
    verdict = "NOT_RUN" if not rows else "PASS" if conv else "PARTIAL" if best and best["gain"] >= 1.0 else "FAIL"
    return dict(verdict=verdict, rows=rows, choice=best if verdict in ("PASS", "PARTIAL") else None,
                reference=dict(label_fit_linear=2.19, label_fit_convctx_layers=4.12, scope="DEV241, patch level, each fold by a model that saw no source image of that fold"))


def pipeline_row(path):
    rep = get(path)
    if not rep or "rows" not in rep or "readout" not in rep["rows"]:
        return None
    v = rep["rows"]["readout"]
    return dict(state=rep.get("state"), episodes=rep.get("episodes"), foris=rep["class_miou"].get("native"), gain=v["gain"], ci95=v["ci95"],
                up=v.get("up"), down=v.get("down"), lose_more_than_10=v.get("lose_more_than_10"), removal_only=rep["rows"].get("removal", {}).get("gain"))


def t2(root):
    rows = {k: pipeline_row(root / ("transfer_" + k) / "report.json") for k in PACKS}
    got = {k: v for k, v in rows.items() if v}
    strong = sum(v["gain"] >= 1.5 and v["ci95"][0] > 0 for v in got.values())
    positive = sum(v["ci95"][0] > 0 for v in got.values())
    harmed = sum(v["ci95"][1] < 0 for v in got.values())
    verdict = "NOT_RUN" if not got else "PASS" if strong >= 3 and not harmed else "FAIL" if positive < 2 else "PARTIAL"
    return dict(verdict=verdict, packs=rows, scope="original resolution, complete FoRIS pipeline, class mIoU on each pack's own episodes; read-out fitted on COCO only")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("d1", "all"), default="all")
    a = p.parse_args()
    out = dict(d1=d1(a.root))
    if out["d1"]["choice"]:
        (a.root / "d1_choice.json").write_text(json.dumps(out["d1"]["choice"], indent=1))
    print("D1 (fit on constructed pairs only; DEV241 patch level; label fit: linear +2.19, convctx +4.12): %s" % out["d1"]["verdict"])
    for r in out["d1"]["rows"]:
        print("  %-5s %-16s %+6.2f [%+.2f, %+.2f]  train %s  epochs %s%s" % (r["set"], r["arm"], r["gain"], *r["ci95"], r["train"], r["epochs"],
                                                                         "  cosine %.3f" % r["cosine"] if r["cosine"] is not None else ""))
    if a.stage == "all":
        out["d1"]["confirmation"] = pipeline_row(a.root / "infer_d1_confirm" / "report.json")
        c = out["d1"]["confirmation"]
        if c:
            print("  chosen fit on CONFIRM600 inside the pipeline: FoRIS %.2f, %+.2f [%+.2f, %+.2f], %s lose more than 10" % (c["foris"], c["gain"], *c["ci95"], c["lose_more_than_10"]))
        out["t2"] = t2(a.root)
        print("T2 (COCO-fitted read-out on other benchmarks, no refitting; original resolution): %s" % out["t2"]["verdict"])
        for k, v in out["t2"]["packs"].items():
            print("  %-12s %s" % (k, "not run" if not v else "FoRIS %.2f  %+.2f [%+.2f, %+.2f]  %s episodes, removal only %s" % (
                v["foris"], v["gain"], *v["ci95"], v["episodes"], "%+.2f" % v["removal_only"] if v["removal_only"] is not None else "-")))
        (a.root / "pivot.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(d1=out["d1"]["verdict"], t2=out.get("t2", {}).get("verdict"))))
    sys.exit(0)


if __name__ == "__main__":
    main()
