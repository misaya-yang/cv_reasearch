"""Aggregate a completed shared W02 run with explicit matched MEAN subsets.

Reads saved I/U only. Refuses partial runs, duplicate record keys and changed
cohort membership. Does not launch inference or modify the shared run.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "cv_data/a"
OUT = DATA / "x1012s_case_review"
DOMAINS = ("coco", "lvis", "pascal_part", "paco_part", "suim", "isaid",
           "isic", "lung", "fundus", "deepglobe_road")


def read(path):
    return json.loads(path.read_text())


def miou(rows, arm):
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        x = groups[(row["fold"], row["cls"])]
        x[0] += row[arm][0]
        x[1] += row[arm][1]
    return 100 * sum(i / max(u, 1) for i, u in groups.values()) / len(groups) if groups else None


def main():
    expected = {}
    for part in ("explore", "confirm"):
        for b in read(DATA / f"x1012_P0/{part}.json"):
            if "B" not in b.get("old_arms", {}):
                continue
            key = str(b["index"])
            assert key not in expected, key
            expected[key] = b
    path = DATA / "x1013/w02/episodes.jsonl"
    payload = path.read_bytes()
    lines = payload.decode().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    keys = [r["index"] for r in records]
    assert len(keys) == len(set(keys)), "Duplicate result keys"
    if set(keys) != set(expected):
        print(json.dumps(dict(state="WAITING_FOR_COMPLETE_SHARED_RUN",
                              complete=len(records), expected=len(expected))))
        return
    dev = {str(b["index"]) for b in read(DATA / "x1012_R1/manifest.json")}
    for r in records:
        b = expected[r["index"]]
        assert r["dataset"] == b["dataset"]
        assert (r["fold"], r["cls"]) == (b["row"]["fold"], b["row"]["global_class_id"])
        assert (r["index"] in dev) == (r["split"] == "dev1360")
        assert ("mean" in r) == ("G" in b.get("old_arms", {}))
    table = []
    for scope in ("all_exposed", "outside_dev1360_exposed"):
        for domain in DOMAINS:
            rows = [r for r in records if r["dataset"] == domain and
                    (scope == "all_exposed" or r["index"] not in dev)]
            matched = [r for r in rows if "mean" in r]
            scores = {a: miou(rows, a) for a in ("foris", "un50", "un475")}
            row = dict(scope=scope, dataset=domain, n=len(rows),
                       class_slots=len({(r["fold"], r["cls"]) for r in rows}),
                       **scores,
                       un50_minus_foris=scores["un50"]-scores["foris"],
                       un475_minus_foris=scores["un475"]-scores["foris"],
                       matched_mean_n=len(matched),
                       matched_mean={a: miou(matched, a) for a in ("foris", "mean", "un50", "un475")})
            table.append(row)
    result = dict(state="COMPLETE_READING_OF_SHARED_RUN", n=len(records),
                  source_path=str(path), source_sha256=hashlib.sha256(payload).hexdigest(),
                  source_bytes=len(payload),
                  interpretation="Exposed tasks; outside_dev1360 is not unseen-photo or blind validation. MEAN scores use their own exact common subset.",
                  new_encoder_calls=0, new_predictions=False,
                  table=table)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "w02_matched_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    for scope in ("all_exposed", "outside_dev1360_exposed"):
        print(scope)
        for r in table:
            if r["scope"] == scope:
                print(f"{r['dataset']:16s} n={r['n']:5d} FoRIS={r['foris']:.3f} "
                      f"UN.5={r['un50']:.3f} delta={r['un50_minus_foris']:+.3f} "
                      f"UN.475={r['un475']:.3f} delta={r['un475_minus_foris']:+.3f} "
                      f"MEAN common n={r['matched_mean_n']}")


if __name__ == "__main__":
    main()
