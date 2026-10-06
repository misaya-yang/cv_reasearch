#!/usr/bin/env python3
"""Continue the existing official seed0 sampler; reject prior episode photographs only."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle
import re
import time

PHOTO = re.compile(r"(?:COCO_(?:val|train)(?:2014|2017)_)?(\d{12})\.(?:jpg|jpeg|png)", re.I)
ALIAS = re.compile(r"COCO:(\d+)$", re.I)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids(value):
    if isinstance(value, (list, tuple)):
        return set().union(*(ids(x) for x in value)) if value else set()
    if not isinstance(value, str):
        return set()
    match = PHOTO.search(value) or ALIAS.fullmatch(value)
    return {int(match.group(1))} if match else set()


def episode_photos(value):
    """Recognize actual paired episode/role records, never class-pool references."""
    photos, count = set(), 0
    stack = [value]
    while stack:
        row = stack.pop()
        if isinstance(row, dict):
            for query, support in [("query", "support"), ("query_path", "support_path"),
                                   ("target", "reference"), ("tgt_name", "ref_names")]:
                q, s = ids(row.get(query)), ids(row.get(support))
                if q and s:
                    photos.update(q | s)
                    count += 1
                    break
            for query, support in [("query_photo_id", "support_photo_id"),
                                   ("query_id", "support_id")]:
                q, s = row.get(query), row.get(support)
                if isinstance(q, int) and isinstance(s, int):
                    photos.update([q, s])
                    count += 1
                    break
            # roles_photo_ids uses the same query/support record parser above.
            stack.extend(x for x in row.values() if isinstance(x, (dict, list)))
        elif isinstance(row, list):
            if len(row) == 3 and isinstance(row[0], int) and ids(row[1]) and ids(row[2]):
                photos.update(ids(row[1]) | ids(row[2]))
                count += 1
            else:
                stack.extend(x for x in row if isinstance(x, (dict, list)))
    return photos, count


def collect(roots, exclude):
    photos, sources, ignored, errors = set(), [], [], []
    paths = sorted({p.resolve() for root in roots if Path(root).exists()
                    for p in Path(root).rglob("*") if p.is_file()
                    and p.suffix in (".json", ".jsonl", ".csv", ".tsv")
                    and not p.resolve().is_relative_to(exclude)})
    for path in paths:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(dict(path=str(path), reason="non_UTF8_metadata"))
            continue
        # Avoid decoding large pool-only metadata or metric arrays without episode roles.
        if path.suffix not in (".csv", ".tsv") and not any(token in text for token in
                ['"query"', '"support"', '"query_path"', '"target"', '"tgt_name"',
                 '"episodes"', '"query_photo_id"', '"query_id"']):
            if PHOTO.search(text):
                ignored.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                    reason="COCO_references_without_paired_episode_roles"))
            continue
        try:
            if path.suffix == ".jsonl":
                values = [json.loads(line) for line in text.splitlines() if line.strip()]
            elif path.suffix in (".csv", ".tsv"):
                values = list(csv.DictReader(text.splitlines(), delimiter="\t" if path.suffix == ".tsv" else ","))
            else:
                values = json.loads(text)
        except (ValueError, csv.Error) as error:
            errors.append(dict(path=str(path), reason=str(error)))
            continue
        got, count = episode_photos(values)
        if got:
            photos.update(got)
            sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                paired_records=count, photo_ids=sorted(got)))
        elif PHOTO.search(text):
            ignored.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                reason="COCO_references_without_valid_paired_episode_roles"))
    return dict(photo_ids=sorted(photos), sources=sources, ignored_pool_or_unpaired_references=ignored,
                parse_errors=errors, scanned_metadata_files=len(paths), query_mask_pixels_opened=False,
                rule="union of paired query/support episode records and explicit COCO role IDs; includes registered training/development/evaluation/prepared episodes conservatively; class/image pools are not episodes")


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan-root", action="append", default=[])
    p.add_argument("--extra-exposure", type=Path)
    p.add_argument("--local-exposure-only", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--host-manifest", type=Path)
    p.add_argument("--prior-public", type=Path)
    p.add_argument("--sampler-source", type=Path)
    p.add_argument("--per-fold", type=int, default=300)
    p.add_argument("--max-source-draws", type=int, default=100000)
    a = p.parse_args()
    started = time.monotonic()
    if a.out.exists():
        raise FileExistsError("Immutable sampling output already exists")
    exposure = collect(a.scan_root, a.out.resolve())
    if a.extra_exposure:
        extra = json.loads(a.extra_exposure.read_text())
        exposure["photo_ids"] = sorted(set(exposure["photo_ids"]) | set(extra["photo_ids"]))
        exposure["extra_exposure"] = dict(path=str(a.extra_exposure), sha256=sha(a.extra_exposure),
                                          sources=extra["sources"], parse_errors=extra.get("parse_errors", []))
    a.out.mkdir(parents=True, exist_ok=False)
    write(a.out / "exposure.json", exposure)
    if a.local_exposure_only:
        print(json.dumps(dict(blocked_photo_count=len(exposure["photo_ids"]),
                              source_files=len(exposure["sources"]), parse_errors=len(exposure["parse_errors"]))))
        return
    import numpy as np
    freeze_sha = sha(a.freeze)
    if freeze_sha != "5fec4eca46825876d40617e8f7c53a38abfd4fc0716305014767cda0477f42c7":
        raise ValueError("Changed pre-sampling frozen strict12 method")
    host = json.loads(a.host_manifest.read_text())
    base, ann = Path(host["data_root"]), Path(host["annotation_root"])
    prior = json.loads(a.prior_public.read_text())
    prior = prior if isinstance(prior, list) else prior.get("episodes", prior.get("rows", []))
    prefix = {(int(r["fold"]), int(r["e"])): r for r in prior}
    if len(prefix) != 4000:
        raise ValueError("Require existing complete official4000 prefix")
    blocked = set(exposure["photo_ids"])
    selected, fold_audits, metadata, pool, parity = [], {}, [], [], 0
    with (a.out / "source_draws.jsonl").open("w") as draws:
        for fold in range(4):
            path = base / "splits/val" / ("fold%d.pkl" % fold)
            meta = pickle.loads(path.read_bytes())
            classes = [fold + 4 * v for v in range(20)]
            metadata.append(dict(fold=fold, path=str(path), sha256=sha(path),
                                 class_ids=classes, pool_order_preserved=True))
            availability, valid = {}, {}
            for c in classes:
                names = [str(x) for x in meta[c]]
                for name in names:
                    if name not in valid:
                        exists = (base / name).is_file() and (ann / Path(name).with_suffix(".png")).is_file()
                        valid[name] = exists and next(iter(ids(name))) not in blocked
                    pool.append(dict(fold=fold, c=c, name=name, photo_id=next(iter(ids(name))), eligible=valid[name]))
                available = {name for name in names if valid[name]}
                availability[str(c)] = dict(metadata_images=len(names), unseen_legal_images=len(available),
                                            distinct_ordered_pairs=len(available) * max(0, len(available) - 1))
            rng = np.random.RandomState(0)
            current, reasons, end = [], Counter(), -1
            for e in range(a.max_source_draws):
                c = int(rng.choice(classes, 1, replace=False)[0])
                q = str(rng.choice(meta[c], 1, replace=False)[0])
                reference_attempts = 0
                while True:
                    s = str(rng.choice(meta[c], 1, replace=False)[0])
                    reference_attempts += 1
                    if q != s:
                        break
                if e < 1000:
                    old = prefix[(fold, e)]
                    if (c, q, s) != (int(old["c"]), old["query"], old["support"]):
                        raise ValueError("Official4000 prefix sampling parity failed")
                    parity += 1
                    continue
                end = e
                qid, sid = next(iter(ids(q))), next(iter(ids(s)))
                cause = []
                if qid in blocked:
                    cause.append("exposed_query_photo")
                if sid in blocked:
                    cause.append("exposed_support_photo")
                if not cause and (not valid[q] or not valid[s]):
                    cause.append("missing_existing_RGB_or_annotation_file")
                rec = dict(fold=fold, e=e, c=c, query=q, support=s, query_photo_id=qid,
                           support_photo_id=sid, reference_attempts=reference_attempts,
                           accepted=not cause, rejection_reasons=cause)
                draws.write(json.dumps(rec) + "\n")
                if cause:
                    reasons["any_rejected"] += 1
                    reasons.update(cause)
                else:
                    row = dict(rec, key="confirm_photo1200:%d_%d_%d" % (fold, e, c),
                               source_key="%d_%d_%d" % (fold, e, c), batch="photo_disjoint_confirm1200_v1")
                    current.append(row)
                if len(current) == a.per_fold:
                    break
                if e == 1000 and not any(x["unseen_legal_images"] >= 2 for x in availability.values()):
                    break
            selected.extend(current)
            fold_audits[str(fold)] = dict(selected=len(current), last_source_draw=end,
                post1000_draws_scanned=max(0, end - 999), rejection_counts=dict(reasons),
                class_draw_counts={str(c): sum(r["c"] == c for r in current) for c in classes},
                available_by_class=availability,
                final_rng_state_sha256=hashlib.sha256(pickle.dumps(rng.get_state())).hexdigest())
    unique_photos = {r[k] for r in selected for k in ["support_photo_id", "query_photo_id"]}
    if unique_photos & blocked:
        raise ValueError("Exposed photograph in confirmation")
    complete = len(selected) == 4 * a.per_fold
    manifest = dict(state="PHOTO_DISJOINT_CONFIRMATION_PREPARED" if complete else "INSUFFICIENT_SELECTED_CONFIRMATION_DRAWS",
        episodes=selected, **{k: host[k] for k in ["data_root", "annotation_root", "foris_root", "projection_basis"]},
        protocol=dict(source_seed=0, start_source_draw=1000, target_per_fold=a.per_fold,
            worker_rng="serial numpy RandomState(0); original class/query/distinct-reference choices in metadata order",
            official_public1000_per_fold_SOTA_protocol=False, natural_repeats_preserved=True,
            photo_disjoint_from_existing_registered_episodes=True, query_truth_opened=False,
            parameter_or_recipe_selection_on_confirmation=False), freeze_path=str(a.freeze), freeze_sha256=freeze_sha)
    write(a.out / "manifest.json", manifest)
    write(a.out / "image_pool.json", dict(state="METADATA_POOL_ONLY_NOT_EXPOSURE", rows=pool))
    if sha(a.freeze) != freeze_sha:
        raise ValueError("Frozen method changed during preparation")
    receipt = dict(state=manifest["state"], n=len(selected), target=4 * a.per_fold,
        source_seed=0, per_fold=a.per_fold, source_start_draw=1000, source_scan_limit=a.max_source_draws,
        folds=fold_audits, metadata=metadata, official_prefix_parity_draws=parity,
        official_prefix_manifest=dict(path=str(a.prior_public), sha256=sha(a.prior_public)),
        sampler_source=dict(path=str(a.sampler_source), sha256=sha(a.sampler_source)),
        host_manifest=dict(path=str(a.host_manifest), sha256=sha(a.host_manifest)),
        freeze_sha256=freeze_sha, blocked_photo_count=len(blocked), confirmation_unique_photos=len(unique_photos),
        overlap_count=0, classes=len({r["c"] for r in selected}),
        absent_classes=sorted(set(range(80)) - {r["c"] for r in selected}),
        natural_repeated_episode_identities=len(selected)-len({(r["c"],r["query"],r["support"]) for r in selected}),
        no_query_GT_pixels=True, no_RGB_pixels=True, no_GPU=True,
        selection_rule="first300/fold accepted seed0 continuation draws; reject only registered episode photo overlap or absent existing files; selected photos not added to blocked",
        estimand_limit="photo rejection changes class/image sampling; independent confirmation cohort, not public first1000/fold benchmark",
        exposure_parse_errors=exposure["parse_errors"], seconds=time.monotonic()-started,
        source_code_sha256=sha(__file__), hashes={name: sha(a.out/name) for name in
            ["manifest.json", "source_draws.jsonl", "image_pool.json", "exposure.json"]})
    write(a.out / "receipt.json", receipt)
    print(json.dumps({k:receipt[k] for k in ["state", "n", "blocked_photo_count", "confirmation_unique_photos", "classes", "absent_classes", "overlap_count", "official_prefix_parity_draws", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
