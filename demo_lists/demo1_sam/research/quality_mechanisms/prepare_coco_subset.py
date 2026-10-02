"""Prepare a seeded small real COCO subset without extracting whole archives.

Requires official pycocotools and NumPy. Sampling depends on annotations only,
never predictions. No network access, training, GPU, or package installation.
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np


def erode3(mask):
    h, w = mask.shape
    pad = np.pad(mask, 1, constant_values=False)
    return np.logical_and.reduce([pad[y:y+h, x:x+w] for y in range(3) for x in range(3)])


def prompts_from_mask(mask, rng):
    """XY integer pixel centers and exclusive upper tight box coordinates."""
    if mask.ndim != 2 or not mask.any():
        raise ValueError("need nonempty two-dimensional GT mask")
    remaining = mask.copy()
    last = remaining
    depth = 0
    while remaining.any():
        last = remaining
        remaining = erode3(remaining)
        depth += 1
    central_yx = np.argwhere(last)
    central = central_yx[rng.integers(len(central_yx))][::-1]
    band = mask & ~erode3(mask)
    near_yx = np.argwhere(band)
    near = near_yx[rng.integers(len(near_yx))][::-1]
    ys, xs = np.where(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
    return central.astype(float).tolist(), near.astype(float).tolist(), box, depth


def decode_coco_annotation(annotation, height, width):
    from pycocotools import mask as mask_utils
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        raise ValueError("invalid COCO segmentation")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(-1)
    if decoded.shape != (height, width):
        raise ValueError("decoded annotation has wrong image shape")
    return decoded.astype(bool)


def locate_member(archive, suffix):
    names = [name for name in archive.namelist() if name == suffix or name.endswith("/" + suffix)]
    if len(names) != 1:
        raise ValueError(f"archive requires exactly one {suffix}, found {len(names)}")
    return names[0]


def prepare(annotations_zip, images_zip, output_dir, seed=2027, num_images=24,
            max_instances=4, min_area=1024, decoder=decode_coco_annotation,
            excluded_image_ids=None):
    if num_images < 1 or max_instances < 1 or min_area < 1:
        raise ValueError("counts and min_area must be positive")
    output_dir = Path(output_dir)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError("existing manifest: use a new output directory to preserve selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "ground_truth").mkdir(exist_ok=True)
    with zipfile.ZipFile(annotations_zip) as archive:
        source = json.loads(archive.read(locate_member(archive, "instances_val2017.json")))
    image_by_id = {int(image["id"]): image for image in source["images"]}
    excluded = set(excluded_image_ids or ())
    eligible = {}
    for annotation in source["annotations"]:
        bbox = annotation.get("bbox", [])
        if (annotation.get("iscrowd", 0) == 0 and annotation.get("area", 0) >= min_area
                and annotation.get("segmentation") and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0
                and int(annotation["image_id"]) in image_by_id
                and int(annotation["image_id"]) not in excluded):
            eligible.setdefault(int(annotation["image_id"]), []).append(annotation)
    rng = np.random.default_rng(seed)
    candidate_ids = rng.permutation(sorted(eligible)).tolist()
    selected, annotation_subset, skipped = [], [], []
    with zipfile.ZipFile(images_zip) as image_archive:
        for image_id in candidate_ids:
            image = image_by_id[image_id]
            height, width = int(image["height"]), int(image["width"])
            annotations = sorted(eligible[image_id], key=lambda item: item["id"])
            masks, objects = [], []
            # Consider all eligible annotations in random order until four valid ones.
            for index in rng.permutation(len(annotations)):
                annotation = annotations[int(index)]
                try:
                    mask = decoder(annotation, height, width)
                    if not mask.any():
                        raise ValueError("empty decoded mask")
                    central, near, box, depth = prompts_from_mask(mask, rng)
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    skipped.append({"annotation_id": annotation["id"], "reason": str(error)})
                    continue
                masks.append(mask)
                objects.append({"annotation_id": int(annotation["id"]), "category_id": int(annotation["category_id"]),
                                "annotation_area": float(annotation["area"]), "decoded_area": int(mask.sum()),
                                "gt_index": len(masks)-1, "central_xy": central, "near_boundary_xy": near,
                                "tight_box_xyxy": box, "central_erosion_depth": depth})
                annotation_subset.append(annotation)
                if len(masks) == max_instances:
                    break
            if not masks:
                continue
            image_name = Path(image["file_name"]).name
            if image_name != image["file_name"]:
                raise ValueError("COCO file_name must be a plain filename")
            jpeg = image_archive.read(locate_member(image_archive, image_name))
            image_relative = f"images/{image_name}"
            (output_dir / image_relative).write_bytes(jpeg)
            gt_relative = f"ground_truth/{image_id:012d}.npz"
            np.savez_compressed(output_dir / gt_relative, gt=np.stack(masks),
                                annotation_ids=np.array([item["annotation_id"] for item in objects], dtype=np.int64))
            selected.append({"image_id": image_id, "file_name": image_name, "image_file": image_relative,
                             "gt_file": gt_relative, "height": height, "width": width, "objects": objects})
            if len(selected) == num_images:
                break
    if len(selected) != num_images:
        raise ValueError(f"requested {num_images} images, only {len(selected)} valid images available")
    manifest = {"schema_version": 1, "dataset": "COCO2017 val", "seed": seed,
                "selection": "seeded annotation-only eligible image permutation, then per-image annotation permutation",
                "model_outputs_used_for_selection": False,
                "num_images": len(selected), "num_instances": sum(len(item["objects"]) for item in selected),
                "max_instances_per_image": max_instances, "min_annotation_area": min_area,
                "point_coordinates": "original-image XY integer pixel indices, no additional half-pixel offset",
                "near_boundary_rule": "uniform positive pixel in mask minus one 3x3 zero-padded erosion",
                "central_rule": "uniform pixel in last nonempty repeated 3x3 erosion; Chebyshev depth",
                "box_rule": "tight decoded-mask XYXY box with exclusive xmax/ymax",
                "regimes": ["central", "near_boundary", "box"], "images": selected, "skipped_annotations": skipped,
                "asset_inputs": {"annotations_zip": str(annotations_zip), "images_zip": str(images_zip)}}
    manifest["excluded_image_ids"] = sorted(excluded)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "selected_annotations.json").write_text(json.dumps({"images": [image_by_id[item["image_id"]] for item in selected],
                                                                     "annotations": annotation_subset, "categories": source.get("categories", [])}) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-images", type=int, default=24)
    parser.add_argument("--max-instances", type=int, default=4)
    parser.add_argument("--min-area", type=float, default=1024)
    parser.add_argument("--exclude-manifest", type=Path,
                        help="Exclude every image in this development manifest before seeded sampling")
    args = parser.parse_args()
    excluded = None if args.exclude_manifest is None else [x["image_id"] for x in json.loads(args.exclude_manifest.read_text())["images"]]
    result = prepare(args.annotations_zip, args.images_zip, args.output_dir, args.seed,
                     args.num_images, args.max_instances, args.min_area,
                     excluded_image_ids=excluded)
    print(json.dumps({key: result[key] for key in ("num_images", "num_instances", "seed")}, indent=2))
