#!/usr/bin/env python3
"""Index the existing 13 public4000 mask families plus the fixed DEV241 raw origin."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def identity(row):
    return (int(row["fold"]), int(row["e"]), int(row["c"]),
            Path(row["support"]).name, Path(row["query"]).name)


def check(job):
    path, expected_file_sha, entries = job
    if sha(path) != expected_file_sha:
        raise ValueError("Changed sealed prediction: " + path)
    with np.load(path, allow_pickle=False) as z:
        for entry in entries:
            mask = z[entry["key"]]
            if mask.dtype != np.uint8 or mask.shape != (131072,):
                raise ValueError("Require packed complete1024 mask")
            if hashlib.sha256(mask.tobytes()).hexdigest() != entry["packed_array_sha256"]:
                raise ValueError("Changed complete mask: " + path + ":" + entry["key"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--consumer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    started = time.monotonic()
    public_path = a.consumer / "manifest_public4000.json"
    dev_path = a.consumer / "manifest_dev241.json"
    expected = {
        public_path: "a7cb8f550e1f1b6a9fd0db7acb2177070e27014f6f9f7dba0452c2f7902a96db",
        dev_path: "01b90f4923b63dbabc35164071f9fa33e7c1ce19ab6de6e6e2db58ef06f19025",
    }
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError("Changed immutable v4 consumer")
    public = json.loads(public_path.read_text())
    dev = json.loads(dev_path.read_text())
    if len(public["rows"]) != 4000 or len(dev["rows"]) != 241:
        raise ValueError("Wrong complete cohort")
    arms = []
    seen = set()
    for arm in public["arms"]:
        if arm["producer_id"] not in seen:
            seen.add(arm["producer_id"])
            arms.append(copy.deepcopy(arm))
    if len(arms) != 13:
        raise ValueError("Require the exact existing13 public mask families")
    raw_id = "frozen_fine_raw_dev241_v1::model.raw_nn"
    raw = copy.deepcopy(next(arm for arm in dev["arms"] if arm["id"] == raw_id))
    raw["fixed_common_origin"] = True
    arms.append(raw)
    indexed = {}
    for row in public["rows"]:
        indexed.setdefault(identity(row), []).append(row)
    rows, jobs, matches = [], {}, []
    for old in dev["rows"]:
        candidates = indexed.get(identity(old), [])
        if len(candidates) != 1:
            raise ValueError("Missing/ambiguous exact public draw: " + old["key"])
        source = candidates[0]
        row = {k: copy.deepcopy(v) for k, v in old.items() if k != "masks"}
        row["masks"] = {}
        row["public4000_draw_key"] = source["key"]
        for arm in arms:
            aid = arm["id"]
            if aid == raw_id:
                entry = old["masks"][aid]
            else:
                entry = source["masks"][aid]
                dev_aid = aid.replace("claude_official_4000_existing_masks::",
                                      "claude_official_projected_DEV241::")
                if old["masks"][dev_aid]["packed_array_sha256"] != entry["packed_array_sha256"]:
                    raise ValueError("Public/DEV mask projection disagreement")
            row["masks"][aid] = copy.deepcopy(entry)
            token = (entry["path"], entry["sha256"])
            jobs.setdefault(token, {})[entry["key"]] = entry
        rows.append(row)
        matches.append(dict(dev_key=old["key"], public_key=source["key"],
                            exact_fold_episode_class_support_query=True))
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(check, [(path, digest, list(entries.values()))
                              for (path, digest), entries in jobs.items()]))
    for arm in arms:
        digest = hashlib.sha256()
        for row in rows:
            digest.update(bytes.fromhex(row["masks"][arm["id"]]["packed_array_sha256"]))
        if arm["id"] != raw_id:
            arm["public4000_arm_id"] = arm["id"]
            arm["public4000_producer_id"] = arm["producer_id"]
        arm["producer_id"] = "mask_method:" + digest.hexdigest()
    def count(flag):
        return len({x["producer_id"] for x in arms if x[flag]})
    source_ids = {arm["source_id"] for arm in arms}
    sources = [copy.deepcopy(s) for s in public["sources"] if s["id"] in source_ids]
    sources += [copy.deepcopy(s) for s in dev["sources"] if s["id"] == raw["source_id"]]
    for source in sources:
        retained = {row["masks"][arm["id"]]["path"] for row in rows
                    for arm in arms if arm["source_id"] == source["id"]}
        source["records"] = [r for r in source.get("records", [])
                             if r.get("prediction_path") in retained]
    result = dict(state="CONSUMABLE_RESTRICTED_SHARED_MASK_LIBRARY", cohort="DEV241",
                  n=241, resolution=1024, encoding=dev["encoding"], common_origin="model.raw_nn",
                  origin_available=True, origin_arm_id=raw_id, requested_common_origin="model.raw_nn",
                  canonical_source=dev["canonical_source"],
                  canonical_manifest_sha256=dev["canonical_manifest_sha256"],
                  arms=arms, rows=rows, sources=sources, source_arm_count=len(arms),
                  distinct_mask_sequences=len({x["producer_id"] for x in arms}),
                  strict_distinct_method_count=count("strict_eligible"),
                  label_fitted_extended_distinct_method_count=count("label_fitted_extended_eligible"),
                  public4000_consumer_path=str(public_path), public4000_consumer_sha256=expected[public_path],
                  full_dev241_consumer_path=str(dev_path), full_dev241_consumer_sha256=expected[dev_path],
                  selection_scope="Only existing13 public4000 distinct families projected by exact draw plus unchanged rawNN O; not the full187 family",
                  full_library_distinct_count=dev["distinct_mask_sequences"],
                  full_library_source_arm_count=dev["source_arm_count"],
                  full187_library_exhaustive_search_claim=False,
                  full_library_not_selected_arm_ids=[x["id"] for x in dev["arms"]
                      if x["id"] not in {arm["id"].replace("claude_official_4000_existing_masks::",
                                                         "claude_official_projected_DEV241::") for arm in arms}],
                  gt_metadata=dict(query_truth_opened=False), no_GT_opened=True,
                  no_features_opened=True, no_new_masks=True,
                  public_to_dev_array_hash_differences=0, exact_draw_matches=matches,
                  checked_prediction_files=len(jobs), workers=a.workers, seconds=time.monotonic()-started,
                  builder_sha256=sha(Path(__file__)))
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError("Immutable consumer changed during indexing")
    a.out.mkdir(parents=True, exist_ok=False)
    target = a.out / "manifest_dev241.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    receipt = {k: result[k] for k in ["state", "n", "source_arm_count", "distinct_mask_sequences",
               "strict_distinct_method_count", "label_fitted_extended_distinct_method_count",
               "checked_prediction_files", "seconds", "builder_sha256", "no_GT_opened",
               "no_features_opened", "no_new_masks", "public_to_dev_array_hash_differences"]}
    receipt.update(manifest_path=str(target), manifest_sha256=sha(target),
                   full187_library_exhaustive_search_claim=False)
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
