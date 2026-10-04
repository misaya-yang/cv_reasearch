#!/usr/bin/env python3
"""Decompose the saved 0.5 -> 0.3 threshold gain by empty/nonempty baseline cases.

CPU-only; consumes saved per-episode original-resolution I/U and truth-area fields.
It opens no image, mask, or annotation files.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def load_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    out = {}
    for row in rows:
        key = (row["fold"], row["e"], row["c"])
        if key in out:
            raise ValueError("duplicate per-episode key: " + repr(key))
        out[key] = row
    return out


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.project_root
    sys.path.insert(0, str(root / "scripts"))
    import sam3_stitch as S

    specifications = {
        "dev": (241, root / "results/sam3_keep_v1/dev/episodes_shard0.jsonl",
                root / "results/sam3_relative_v1/dev.json.episodes.jsonl"),
        "confirm": (600, root / "results/sam3_keep_v1/confirm/episodes_shard0.jsonl",
                    root / "results/sam3_relative_v1/confirm.json.episodes.jsonl"),
        "fresh": (915, root / "results/sam3_keep_v1/rest/episodes_shard0.jsonl",
                  root / "results/sam3_relative_v1/fresh.json.episodes.jsonl"),
    }
    result = dict(state="COMPLETED", dataset="COCO-20i", seed=0,
                  metric="original-resolution class mIoU; paired connected support/query photo-group bootstrap",
                  draws=2000, query_annotation_files_opened=False,
                  source="saved scored episode fields only", splits={})
    for split, (expected, all_path, sidecar_path) in specifications.items():
        all_rows, sidecar = load_jsonl(all_path), load_jsonl(sidecar_path)
        if len(sidecar) != expected or not set(sidecar).issubset(all_rows):
            raise ValueError("saved score sidecar does not match the episode run for " + split)
        records = []
        for key, scored in sidecar.items():
            full = all_rows[key]
            truth_area = int(full["truth_area"])
            p05 = sidecar[key]["rule_iu"]["absolute_0.5"]["original"]
            p03 = sidecar[key]["rule_iu"]["absolute_0.3"]["original"]
            area05 = int(p05[0] + p05[1] - truth_area)
            area03 = int(p03[0] + p03[1] - truth_area)
            if area05 < 0 or area03 < 0:
                raise ValueError("I/U and truth-area identity produced a negative predicted area: " + repr(key))
            is_empty05 = area05 == 0
            records.append(dict(fold=key[0], e=key[1], c=key[2], support=full["support"], query=full["query"],
                                fixed05=list(p05), fixed03=list(p03), empty05=is_empty05,
                                pred_area05=area05, pred_area03=area03))

        strategies = {
            "all_0.3": lambda r: r["fixed03"],
            "only_if_0.5_empty_use_0.3": lambda r: r["fixed03"] if r["empty05"] else r["fixed05"],
            "only_if_0.5_nonempty_use_0.3": lambda r: r["fixed03"] if not r["empty05"] else r["fixed05"],
        }
        base = lambda r: r["fixed05"]
        split_out = dict(episodes=len(records),
                         baseline_empty_count=sum(r["empty05"] for r in records),
                         baseline_nonempty_count=sum(not r["empty05"] for r in records),
                         mean_pred_area_0p5=sum(r["pred_area05"] for r in records) / len(records),
                         mean_pred_area_0p3=sum(r["pred_area03"] for r in records) / len(records),
                         strategies={})
        for name, get in strategies.items():
            paired = S.paired(records, get, base)
            per_fold = [S.paired([r for r in records if r["fold"] == f], get, base, draws=1)["gain"]
                        for f in sorted({r["fold"] for r in records})]
            values = [get(r)[0] / max(get(r)[1], 1) - base(r)[0] / max(base(r)[1], 1) for r in records]
            split_out["strategies"][name] = dict(miou=paired["miou"], gain=paired["gain"], ci95=paired["ci95"],
                                                  groups=paired["groups"], largest_group=paired["largest_group"],
                                                  per_fold=per_fold, episodes_up=sum(x > 0 for x in values),
                                                  episodes_down=sum(x < 0 for x in values))
        result["splits"][split] = split_out

    result["inputs"] = {name: {"scored_episodes_sha256": sha256(paths[1]), "rule_sidecar_sha256": sha256(paths[2])}
                         for name, paths in specifications.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    for split, rows in result["splits"].items():
        print(split, "baseline empty", rows["baseline_empty_count"], "/", rows["episodes"])
        for name, metric in rows["strategies"].items():
            print("  %s: %.2f, %+.2f [%+.2f, %+.2f], folds %s" %
                  (name, metric["miou"], metric["gain"], *metric["ci95"],
                   " ".join("%+.2f" % x for x in metric["per_fold"])))
    print("saved", a.out)


if __name__ == "__main__":
    main()
