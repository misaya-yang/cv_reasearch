#!/usr/bin/env python3
"""Measure whether the top-20 score-ordered candidate-prefix family has IoU headroom.

CPU diagnostic only. It uses already-scored DEV/CONFIRM/Fresh candidate masks and query
annotations; the per-episode best-prefix result is an oracle ceiling, never a method score.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def read_jsonl(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        key = (row["fold"], row["e"], row["c"])
        if key in out:
            raise ValueError("duplicate episode key " + repr(key))
        out[key] = row
    return out


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.project_root
    sys.path.insert(0, str(root / "scripts"))
    import numpy as np
    from PIL import Image
    import sam3_stitch as S

    suite = root / "results/extent_head_t1_isolated_v1"
    rest_manifest_path = root / "results/sam3_keep_v1/rest/rest_episodes.json"
    rest_manifest = json.loads(rest_manifest_path.read_text())
    rest_rows = {(r["fold"], r["e"], r["c"]): r for r in rest_manifest["episodes"]}
    specs = {
        "dev": (root / "results/sam3_keep_v1/dev", suite / "dev_episodes.json",
                root / "results/sam3_relative_v1/dev.json.episodes.jsonl"),
        "confirm": (root / "results/sam3_keep_v1/confirm", suite / "confirm_episodes.json",
                    root / "results/sam3_relative_v1/confirm.json.episodes.jsonl"),
        "fresh": (root / "results/sam3_keep_v1/rest", rest_manifest_path,
                  root / "results/sam3_relative_v1/fresh.json.episodes.jsonl"),
    }
    output = dict(state="COMPLETED", dataset="COCO-20i", seed=0,
                  diagnostic="best IoU over all prefixes of the frozen top-20 proposal score order",
                  query_annotations_opened=True,
                  metric="original-resolution class mIoU; connected support/query photo-group bootstrap",
                  draws=2000, splits={})
    for split, (run, manifest_path, sidecar_path) in specs.items():
        manifest = json.loads(manifest_path.read_text())
        rows_by_key = {(r["fold"], r["e"], r["c"]): r for r in manifest["episodes"]}
        if split == "fresh":
            sidecar = read_jsonl(sidecar_path)
            rows_by_key = {key: rest_rows[key] for key in sidecar if key in rest_rows}
        else:
            sidecar = read_jsonl(sidecar_path)
        if len(sidecar) == 0 or not set(sidecar).issubset(rows_by_key):
            raise ValueError("scored sidecar does not match its manifest for " + split)

        records = []
        # Load the frozen scored run once; it contains the exact support/query IDs and truth areas.
        full_path = run / "episodes_shard0.jsonl"
        full_rows = read_jsonl(full_path)
        ann_root = Path(manifest["annotation_root"])
        for key, scored in sidecar.items():
            row = rows_by_key[key]
            full = full_rows.get(key)
            if full is None or full.get("support") != row["support"] or full.get("query") != row["query"]:
                raise ValueError("frozen score record does not match manifest UIDs: " + repr(key))
            query_path = Path(ann_root) / Path(row["query"]).with_suffix(".png")
            with Image.open(query_path) as image:
                truth = np.asarray(image) == int(row["c"]) + 1
            if list(truth.shape) != list(full["query_shape"]):
                raise ValueError("query annotation shape mismatch: " + repr(key))
            n, h, w = full["proposal_shape"]
            q = [i for i, proposal in enumerate(full["proposals"]) if proposal[1] > 0]
            with np.load(run / full["candidate_file"], allow_pickle=False) as z:
                raw = np.unpackbits(z["proposal_query"], axis=1)[:, :h * w].reshape(n, h, w).astype(bool)

            prefix_iu = []
            union = np.zeros((h, w), dtype=bool)
            for k, index in enumerate(q, 1):
                union |= raw[index]
                full_pred = np.asarray(Image.fromarray(union.astype(np.uint8)).resize(truth.shape[::-1], Image.NEAREST)) > 0
                inter = int((full_pred & truth).sum())
                area_union = int((full_pred | truth).sum())
                prefix_iu.append([inter, area_union])
            if q:
                ious = [i / max(u, 1) for i, u in prefix_iu]
                oracle_k = max(range(len(ious)), key=lambda idx: (ious[idx], -idx)) + 1
                oracle_iu = prefix_iu[oracle_k - 1]
            else:
                oracle_k, oracle_iu = 0, [0, int(truth.sum())]
            scores = [float(p[0]) for p in full["proposals"]]
            active_scores = [scores[i] for i in q]
            rel_k = sum(score >= 0.7 * max(active_scores) for score in active_scores) if active_scores else 0
            base_iu = scored["rule_iu"]["relative_0.7"]["original"]
            records.append(dict(fold=key[0], e=key[1], c=key[2], support=full["support"], query=full["query"],
                                oracle_prefix=list(oracle_iu), relative_0p7=list(base_iu),
                                oracle_k=oracle_k, relative_k=rel_k, candidate_count=len(q)))

        oracle_get = lambda r: r["oracle_prefix"]
        rel_get = lambda r: r["relative_0p7"]
        paired = S.paired(records, oracle_get, rel_get)
        per_fold = [S.paired([r for r in records if r["fold"] == f], oracle_get, rel_get, draws=1)["gain"]
                    for f in sorted({r["fold"] for r in records})]
        output["splits"][split] = dict(episodes=len(records), candidate_count_mean=float(np.mean([r["candidate_count"] for r in records])),
                                      relative_k_mean=float(np.mean([r["relative_k"] for r in records])),
                                      oracle_k_mean=float(np.mean([r["oracle_k"] for r in records])),
                                      oracle_k_differs_from_relative=sum(r["oracle_k"] != r["relative_k"] for r in records),
                                      oracle_prefix_miou=paired["miou"], relative_0p7_miou=S.paired(records, rel_get, oracle_get)["miou"],
                                      oracle_headroom=paired["gain"], ci95=paired["ci95"], groups=paired["groups"],
                                      largest_group=paired["largest_group"], per_fold=per_fold,
                                      scored_run_sha256=file_sha(full_path), sidecar_sha256=file_sha(sidecar_path))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=1) + "\n")
    for split, row in output["splits"].items():
        print("%s: relative %.2f, oracle-prefix %.2f, headroom +%.2f [%+.2f,%+.2f]; k mean %.2f -> %.2f; folds %s" %
              (split, row["relative_0p7_miou"], row["oracle_prefix_miou"], row["oracle_headroom"], *row["ci95"],
               row["relative_k_mean"], row["oracle_k_mean"], " ".join("%+.2f" % x for x in row["per_fold"])))
    print("saved", a.out)


if __name__ == "__main__":
    main()
