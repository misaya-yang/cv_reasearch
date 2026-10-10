"""Validate/publicize newly downloaded exact Chest and FIVES loader pools.

Own staging only; refuse to overwrite an existing dataset destination. No
encoder/inference calls. Official source loaders are verified separately by
the benchmark-extension agent after these integrity receipts become ready.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parents[4].parent / "cv_data"
DOWNLOADS = ASSETS / "setup/asset_downloads"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def chest_source(root):
    complete = root / "download_complete.json"
    if not complete.exists():
        return None
    if json.loads(complete.read_text())["state"] != "COMPLETE_SOURCE_POOL":
        raise ValueError("Chest source download not validated")
    stage = root / "staging/LungSegmentation"
    records = [json.loads(line) for line in (root / "source_files.jsonl").read_text().splitlines()]
    for r in records:
        path = stage / r["relative_path"]
        if not path.is_file() or path.stat().st_size != r["bytes"] or digest(path) != r["sha256"]:
            raise ValueError("Chest source file changed")
    if len(list((stage / "CXR_png").glob("*.png"))) != 800 or len(list((stage / "masks").glob("*.png"))) != 704:
        raise ValueError("Chest full cohort mismatch")
    return stage, dict(dataset="Chest X-ray masks and labels", dataset_version=1,
        dataset_reference="nikhilpandey360/chest-xray-masks-and-labels",
        source_url="https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels",
        source_foRIS_doc_sha256=digest(ASSETS / "third_party/foris_official/docs/data.md"),
        license="CC0: Public Domain, as returned by Kaggle dataset metadata",
        retrieval="All 800 CXR_png and 704 masks from original documented Lung Segmentation ZIP root, byte-range extraction with original member size/CRC/SHA; duplicate archive data not fetched",
        source_integrity_receipt=str(complete.relative_to(ASSETS)),
        downloaded_full_archive=False)


def fundus_source(root):
    archive = root / "FIVES_v1.rar"
    if not archive.exists():
        return None
    meta = json.loads((root / "source_figshare_metadata.json").read_text())
    expected = meta["files"][0]
    if archive.stat().st_size != expected["size"] or digest(archive, "md5") != expected["computed_md5"]:
        raise ValueError("Official FIVES original archive MD5/size mismatch")
    names = subprocess.check_output(["bsdtar", "-tf", str(archive)], text=True).splitlines()
    (root / "archive_members.txt").write_text("\n".join(names) + "\n")
    selected = []
    for name in names:
        parts = Path(name).parts
        if name.startswith("/") or ".." in parts:
            raise ValueError("Unsafe original archive member path")
        for i, part in enumerate(parts):
            if part == "test" and len(parts) == i + 3 and parts[i + 1].lower() in ("original", "ground truth") and parts[-1].endswith(".png"):
                selected.append(name)
    if len(selected) != 400:
        raise ValueError("FIVES original test cohort must contain 200 image/mask pairs: " + str(len(selected)))
    extract = root / "source_extracted"
    extract.mkdir(exist_ok=True)
    subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(extract), *selected], check=True)
    stage = root / "prepared/Fundus"
    for name in selected:
        original = extract / name
        role = "Original" if "original" in [p.lower() for p in Path(name).parts] else "Ground truth"
        destination = stage / "test" / role / original.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != digest(original):
                raise ValueError("Existing own prepared FIVES file changed")
        else:
            destination.hardlink_to(original)
    return stage, dict(dataset="FIVES: Fundus Image Dataset for AI-based Vessel Segmentation",
        dataset_reference="10.6084/m9.figshare.19688169.v1", dataset_version=1,
        dataset_article_id=19688169, source_file_id=34969398,
        source_url="https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1",
        dataset_paper="https://doi.org/10.1038/s41597-022-01564-3",
        protocol_identification="FoRIS Appendix F cites Jin2022 FIVES; pinned fundus.py defaults to original test split with same-name PNG image/mask pairs",
        license=meta["license"], original_archive=dict(path=str(archive.relative_to(ASSETS)),
            bytes=archive.stat().st_size, md5=digest(archive, "md5"), sha256=digest(archive),
            supplied_md5=expected["computed_md5"], supplied_md5_passed=True),
        extraction="Original test200 pairs only, bsdtar/RAR checksum verification; filename casing relocated to loader Ground truth; PNG bytes unchanged; no resize/crop",
        downloaded_full_archive=True, default_split="test", original_train_split_count=600,
        test_pool_count=200)


def validate(stage, name):
    if name == "Chest":
        images = sorted((stage / "CXR_png").glob("*.png"))
        masks = sorted((stage / "masks").glob("*.png"))
        pairs = [(p, stage / "CXR_png" / (p.name if "MCUCXR" in str(p) else p.name[:-9] + ".png")) for p in masks]
    else:
        images = sorted((stage / "test/Original").glob("*.png"))
        masks = sorted((stage / "test/Ground truth").glob("*.png"))
        pairs = [(p, stage / "test/Original" / p.name) for p in masks]
        if len(images) != 200 or len(masks) != 200:
            raise ValueError("FIVES test cohort mismatch")
    missing = [str(q) for p, q in pairs if not q.is_file()]
    if missing:
        raise ValueError("Missing exact official image/mask mapping")
    geometries = {}
    for path in images:
        with Image.open(path) as im:
            im.load()
            geometries[path.name] = dict(image_size_wh=list(im.size), image_mode=im.mode)
    pair_info = []
    for mask, image in pairs:
        with Image.open(mask) as im:
            im.load()
            array = np.asarray(im.convert("L"))
            shape = list(im.size)
            foreground = int((array >= 128).sum())
        if foreground == 0:
            raise ValueError("Empty foreground in exact source mask: " + mask.name)
        pair_info.append(dict(mask=mask.name, image=image.name, image_size_wh=geometries[image.name]["image_size_wh"],
            mask_size_wh=shape, geometry_mismatch=shape != geometries[image.name]["image_size_wh"],
            foreground_pixels=foreground))
    return dict(images=len(images), masks=len(masks), sampling_pairs=len(pairs),
                missing_pairs=0, empty_masks=0,
                image_mask_geometry_mismatches=sum(p["geometry_mismatch"] for p in pair_info),
                pairs=pair_info, all_images_decoded=True, all_masks_decoded=True,
                mask_threshold="Exact official loader convertL then >=128; source masks untouched")


def prepare(name):
    root = DOWNLOADS / (name + "_autonomous_20261010")
    ready = root / "preparation_receipt.json"
    if ready.exists() and json.loads(ready.read_text())["status"] == "READY_FOR_OFFICIAL_LOADER":
        return True
    dataset = "LungSegmentation" if name == "Chest" else "Fundus"
    destination = ASSETS / "datasets/ics" / dataset
    if destination.exists():
        raise FileExistsError("Refuse to overwrite existing pool: " + str(destination))
    loaded = chest_source(root) if name == "Chest" else fundus_source(root)
    if loaded is None:
        return False
    stage, receipt = loaded
    pool = validate(stage, name)
    files = [dict(path=str((destination / p.relative_to(stage)).relative_to(ASSETS)),
                  bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(stage.rglob("*.png"))]
    pool.pop("pairs", None)
    stage.rename(destination)
    receipt.update(status="READY_FOR_OFFICIAL_LOADER", dataset_root=str(destination.relative_to(ASSETS)),
        files=files, sampling_pool=pool, files_sha256=digest_bytes(files),
        created_utc=datetime.now(timezone.utc).isoformat(), pixel_transform="none",
        model_forwards=0, existing_assets_overwritten=False,
        note="Readiness validates exact released-loader source pool; not a reproduction of published model scores.")
    write(ready, receipt)
    print(json.dumps(dict(asset=name, status=receipt["status"], files=len(files), sampling_pool=pool)), flush=True)
    return True


def digest_bytes(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    completed = set()
    while len(completed) < 2:
        for name in ("Chest", "Fundus"):
            if name not in completed and prepare(name):
                completed.add(name)
        if not args.wait:
            break
        if len(completed) < 2:
            time.sleep(20)


if __name__ == "__main__":
    main()
