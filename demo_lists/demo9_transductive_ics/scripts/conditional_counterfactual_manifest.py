#!/usr/bin/env python3
"""CPU-only feasibility audit; never edits a source manifest or invokes a model.

Only train target classes and train-role photos are used. The clean-patch rule
is fixed at >=16 patches with >=90% foreground in a 64x64 area grid; it is a
diagnostic purity rule, not INSID3's bilinear mask sampling interface.
"""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


def photo_id(name):
    match = re.search(r"COCO_(?:train|val)2014_(\d+)", name)
    return "COCO:" + str(int(match[1])) if match else Path(name).stem


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--min-clean-patches", type=int, default=16)
    ap.add_argument("--candidate-dir", help="Optional CURRENT pilot masks only; CPU oracle for existing clean pairs")
    ap.add_argument("--out")
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    rows = manifest["records"]
    active = rows[:args.limit]
    train = [r for r in active if r["split"] == "train"]
    classes = sorted({r["c"] for r in rows if r["split"] == "train"})
    excluded = {photo_id(n) for r in rows if r["split"] != "train" for n in r["image_ids"]}
    names = sorted({n for r in train for n in r["image_ids"]})
    assert not {photo_id(n) for n in names} & excluded
    stats = {}
    for name in names:
        ann = np.asarray(Image.open(Path(args.annotations) / Path(name).with_suffix(".png")))
        stats[name] = {}
        for c in classes:
            mask = ann == c + 1
            ys, xs = np.nonzero(mask)
            if not len(ys):
                continue
            purity = np.asarray(Image.fromarray(mask.astype(np.float32)).resize((64, 64), Image.Resampling.BOX))
            stats[name][c] = dict(area_pixels=int(mask.sum()), area_fraction=float(mask.mean()),
                foreground_patch_mass=float(purity.sum()), clean_patches=int((purity >= .9).sum()),
                centroid_xy=[float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0])])

    def eligible(name, c):
        return stats[name].get(c, {}).get("clean_patches", 0) >= args.min_clean_patches

    def item(s, q, a, b, e=None):
        ds = [float(np.linalg.norm(np.array(stats[s][c]["centroid_xy"]) - stats[q][c]["centroid_xy"])) for c in (a, b)]
        ratios = [stats[q][c]["area_fraction"] / stats[s][c]["area_fraction"] for c in (a, b)]
        return dict(source_episode=e, support=s, query=q, classes=[a, b],
            support_stats={str(c): stats[s][c] for c in (a, b)},
            query_stats={str(c): stats[q][c] for c in (a, b)},
            area_ratio_support_to_query=ratios, centroid_displacements=ds,
            direct_oracle="NOT_MEASURED: requires shared fixed candidate bank scored against both training masks")

    paired = []
    present_pairs = 0
    for r in train:
        s, q, a = r["support"], r["query"], r["c"]
        for b in classes:
            if b == a:
                continue
            if b in stats[s] and b in stats[q]:
                present_pairs += 1
            if all(eligible(n, c) for n in (s, q) for c in (a, b)):
                paired.append(item(s, q, a, b, r["e"]))
    if args.candidate_dir:
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(2)
        for row in paired[:10]:
            path = Path(args.candidate_dir) / f"episode_{row['source_episode']:04d}.pt"
            if not path.exists():
                row["direct_oracle"] = dict(state="NOT_ACQUIRED")
                continue
            payload = torch.load(path, weights_only=False, map_location="cpu")
            if payload.get("candidate_state") != "COMPLETE":
                row["direct_oracle"] = dict(state="PAIR_ONLY")
                continue
            size = payload["candidate_native_mask_size"]
            masks = np.unpackbits(payload["candidate_native_mask_bits"], axis=-1)[:, :size * size].reshape(-1, size, size)
            ann = np.asarray(Image.open(Path(args.annotations) / Path(row["query"]).with_suffix(".png")))
            prediction = F.interpolate(torch.from_numpy(masks.copy()).float()[:, None], size=ann.shape,
                mode="bilinear", align_corners=False)[:, 0].numpy() > .5
            score = {}
            for c in row["classes"]:
                gt = ann == c + 1
                inter = (prediction & gt).sum((1, 2))
                union = (prediction | gt).sum((1, 2))
                iou = inter / np.maximum(union, 1)
                score[str(c)] = dict(iou=iou.tolist(), argmax=int(iou.argmax()),
                    oracle=float(iou.max()), direct=float(iou[0]))
            row["direct_oracle"] = dict(state="CPU_FIXED_EXISTING_BANK", scores=score,
                same_candidate_bank=True, new_semantic_pass=False, resolution="original_HxW")
            del payload
    cooccurring = {}
    recombined = []
    for a, b in combinations(classes, 2):
        usable = [n for n in names if eligible(n, a) and eligible(n, b)]
        cooccurring[f"{a},{b}"] = usable
        for s, q in combinations(usable, 2):
            recombined.append(item(s, q, a, b))
    # One photo group for all tasks sharing a photo, including transitively.
    parent = {n: n for n in names}
    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    for r in train:
        ns = r["image_ids"]
        for n in ns[1:]:
            parent[find(n)] = find(ns[0])
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    # Balance target classes and photo reuse without candidate/model quality.
    remaining = list(recombined)
    selected, uses, class_uses, pair_uses = [], Counter(), Counter(), Counter()
    while remaining and len(selected) < 10:
        best = min(range(len(remaining)), key=lambda i: (
            max(class_uses[c] for c in remaining[i]["classes"]),
            sum(class_uses[c] for c in remaining[i]["classes"]),
            pair_uses[tuple(remaining[i]["classes"])],
            max(uses[remaining[i]["support"]], uses[remaining[i]["query"]]),
            uses[remaining[i]["support"]] + uses[remaining[i]["query"]], i))
        row = remaining.pop(best)
        # Derive a single donor list from existing train records. Never condition
        # donor selection on the current target or donor mask quality.
        source = next(r for r in train if row["support"] in r["image_ids"])
        pool = list(dict.fromkeys(source["donors"] + groups[find(row["support"])]))
        donors = [n for n in pool if n not in (row["support"], row["query"])][:3]
        assert len(donors) == 3 and all(n in names for n in donors)
        row.update(donors=donors, direct_index=0,
            source_train_episode=source["e"],
            source_identity_use="e/classes/IDs are audit/generation metadata, NEVER model input",
            both_condition_fixed_fields=["candidate_union", "donor_P1_union", "provenance", "donors", "direct_index"])
        selected.append(row)
        uses.update((row["support"], row["query"]))
        class_uses.update(row["classes"])
        pair_uses.update((tuple(row["classes"]),))
    geometry = Counter()
    for row in selected:
        a, b = row["classes"]
        for target in (a, b):
            s = row["support_stats"][str(target)]
            choices = [row["query_stats"][str(c)] for c in (a, b)]
            area = [abs(np.log(q["area_fraction"] / s["area_fraction"])) for q in choices]
            centroid = [float(np.linalg.norm(np.array(q["centroid_xy"]) - s["centroid_xy"])) for q in choices]
            geometry["tasks"] += 1
            geometry["area_correct"] += (a, b)[int(np.argmin(area))] == target
            geometry["centroid_correct"] += (a, b)[int(np.argmin(centroid))] == target
    augmented_parent = dict(parent)
    def augmented_find(n):
        while augmented_parent[n] != n:
            augmented_parent[n] = augmented_parent[augmented_parent[n]]
            n = augmented_parent[n]
        return n
    for row in selected:
        roles = [row["support"], row["query"]] + row["donors"]
        for n in roles[1:]:
            augmented_parent[augmented_find(n)] = augmented_find(roles[0])
    augmented_groups = {}
    for n in names:
        augmented_groups.setdefault(augmented_find(n), []).append(n)
    result = dict(source_manifest=args.manifest, limit=args.limit,
        active_split_counts=dict(Counter(r["split"] for r in active)), train_classes=classes,
        excluded_classes=sorted(set(manifest["classes"]) - set(classes)),
        train_role_photo_count=len(names), forbidden_photo_overlap=0,
        clean_patch_rule=dict(grid=64, foreground_fraction=.9, min_clean_patches=args.min_clean_patches),
        existing_pair_alternative_present=present_pairs, existing_pair_clean_count=len(paired),
        existing_pairs=paired, cooccurring_train_photos=cooccurring,
        recombined_train_pair_count=len(recombined), minimal_recombined_examples=selected,
        geometry_shortcut_diagnostic=dict(geometry,
            scope="Annotation-only: choose between two TRUE query masks by support/query area or centroid; not legal candidate performance"),
        selected_photo_reuse=dict(uses),
        selected_class_reuse=dict(class_uses),
        all_role_connected_photo_groups=list(groups.values()),
        counterfactual_augmented_photo_groups=list(augmented_groups.values()),
        condition_contract="same RGB, same frozen candidate bank, same P1/provenance/donors for both tasks; only support mask/task truth changes",
        generation_limit="No alternate semantic pass run. New union bank needs both semantic passes before freezing; current oracle is not its ceiling.")
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
