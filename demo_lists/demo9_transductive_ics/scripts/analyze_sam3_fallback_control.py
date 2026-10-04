#!/usr/bin/env python3
"""Paired, photo-group-bootstrap comparison of relative_0.7 and fallback_top1.

Consumes only frozen SAM3 score-ledger JSONL files and their already-scored
rule-IOU sidecars. It does not open images or query annotations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sam3_stitch as S


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def key(row):
    return row["fold"], row["e"], row["c"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(root, source_root, draws):
    pairs = {
        "dev": (source_root / "dev/episodes_shard0.jsonl", root / "dev.json.episodes.jsonl"),
        "confirm": (source_root / "confirm/episodes_shard0.jsonl", root / "confirm.json.episodes.jsonl"),
        "fresh": (source_root / "rest/episodes_shard0.jsonl", root / "fresh.json.episodes.jsonl"),
    }
    rel = lambda row: row["rule_iu"]["relative_0.7"]["original"]
    fallback = lambda row: row["rule_iu"]["fallback_top1"]["original"]
    results = {}
    for split, (source_path, sidecar_path) in pairs.items():
        source, sidecar = read_jsonl(source_path), read_jsonl(sidecar_path)
        by_key = {key(row): row for row in sidecar}
        if len(by_key) != len(sidecar):
            raise ValueError(f"{split}: duplicate sidecar episode keys")
        joined = []
        for row in source:
            if key(row) in by_key:
                item = dict(row)
                item["rule_iu"] = by_key[key(row)]["rule_iu"]
                joined.append(item)
        observed = {key(row) for row in joined}
        if observed != set(by_key):
            raise ValueError(f"{split}: sidecar keys do not exactly join to scored records")
        if any(not row.get("support") or not row.get("query") for row in joined):
            raise ValueError(f"{split}: missing support/query group identifiers")

        paired = S.paired(joined, rel, fallback, draws=draws)
        case_delta = [
            rel(row)[0] / max(rel(row)[1], 1) - fallback(row)[0] / max(fallback(row)[1], 1)
            for row in joined
        ]
        fold_gain = {}
        for fold in sorted({row["fold"] for row in joined}):
            subset = [row for row in joined if row["fold"] == fold]
            fold_gain[str(fold)] = S.paired(subset, rel, fallback, draws=1)["gain"]
        results[split] = dict(
            episodes=len(joined),
            relative_miou=paired["miou"],
            fallback_top1_miou=paired["miou"] - paired["gain"],
            gain=paired["gain"],
            ci95=paired["ci95"],
            bootstrap_unit=paired["bootstrap_unit"],
            groups=paired["groups"],
            largest_group=paired["largest_group"],
            draws=paired["draws"],
            per_fold_gain=fold_gain,
            episodes_up=sum(value > 0 for value in case_delta),
            episodes_down=sum(value < 0 for value in case_delta),
            source_records=dict(path=str(source_path), sha256=sha256(source_path)),
            score_sidecar=dict(path=str(sidecar_path), sha256=sha256(sidecar_path)),
        )
    return dict(
        state="COMPLETED",
        comparison="relative_0.7 vs fallback_top1",
        scope="original-resolution class mIoU; paired connected-photo-group bootstrap; seed 0",
        query_annotations_opened=False,
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/sam3_relative_v1"))
    parser.add_argument("--source-root", type=Path, default=Path("results/sam3_keep_v1"))
    parser.add_argument("--out", type=Path, default=Path("results/sam3_relative_v1/fallback_control.json"))
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    result = analyze(args.root, args.source_root, args.draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    for split, row in result["results"].items():
        print("%s n=%d relative=%.2f fallback=%.2f diff=%+.2f [%+.2f,%+.2f] folds=%s up=%d down=%d groups=%d" % (
            split, row["episodes"], row["relative_miou"], row["fallback_top1_miou"], row["gain"],
            row["ci95"][0], row["ci95"][1],
            ",".join("%+.2f" % row["per_fold_gain"][fold] for fold in sorted(row["per_fold_gain"])),
            row["episodes_up"], row["episodes_down"], row["groups"]))
    print("saved", args.out)


if __name__ == "__main__":
    main()
