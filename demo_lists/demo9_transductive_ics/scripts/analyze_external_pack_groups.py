#!/usr/bin/env python3
"""Identify exact repeated RGB photos in a packed external episode manifest.

This deliberately hashes decoded RGB pixels, not copied filenames. It reads only
support/query images; annotations and query labels are never opened.

  python scripts/analyze_external_pack_groups.py \
      --manifest transfer_v1/paco_part/episodes.json \
      --out results/sam3_relative_v1/paco_part_photo_groups.json
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def photo_hash(path):
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        payload = (rgb.width.to_bytes(4, "big") + rgb.height.to_bytes(4, "big") + rgb.tobytes())
    return hashlib.sha256(payload).hexdigest()


def analyze(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    data_root = Path(manifest["data_root"])
    if not data_root.is_absolute():
        candidates = [Path.cwd() / data_root, manifest_path.parent / data_root,
                      manifest_path.parent.parent / data_root]
        data_root = next((candidate.resolve() for candidate in candidates if candidate.is_dir()), None)
        if data_root is None:
            raise FileNotFoundError("Could not resolve manifest data_root: " + manifest["data_root"])
    rows = manifest["episodes"]
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    owners = defaultdict(list)
    episode_rows = []
    n_images = 0
    for i, row in enumerate(rows):
        ids = {}
        for role in ("support", "query"):
            ids[role] = photo_hash(data_root / row[role])
            for owner in owners[ids[role]]:
                union(i, owner)
            owners[ids[role]].append(i)
            n_images += 1
        episode_rows.append(dict(fold=row["fold"], e=row["e"], c=row["c"],
                                 support_rgb_sha256=ids["support"], query_rgb_sha256=ids["query"]))

    clusters = defaultdict(list)
    for i, row in enumerate(episode_rows):
        clusters[find(i)].append(i)
    stable = sorted(clusters.values(), key=lambda group: min(group))
    group_id = {index: group for group, members in enumerate(stable) for index in members}
    for i, row in enumerate(episode_rows):
        row["photo_group"] = group_id[i]

    sizes = [len(group) for group in stable]
    duplicate_hashes = sum(len(indices) > 1 for indices in owners.values())
    return dict(
        state="COMPLETED",
        dataset=manifest.get("dataset", "unknown"),
        split="packed transfer episodes",
        episodes=len(rows),
        rgb_images=n_images,
        unique_exact_rgb=len(owners),
        duplicate_rgb_hashes=duplicate_hashes,
        duplicate_image_copies=n_images - len(owners),
        connected_photo_groups=len(stable),
        groups_with_multiple_episodes=sum(size > 1 for size in sizes),
        singleton_episodes=sum(size == 1 for size in sizes),
        largest_group=max(sizes, default=0),
        identity_method="SHA-256(decoded RGB pixels with width and height); exact duplicates only",
        query_annotation_opened=False,
        episodes_with_photo_groups=episode_rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "episodes_with_photo_groups"}, sort_keys=True))


if __name__ == "__main__":
    main()
