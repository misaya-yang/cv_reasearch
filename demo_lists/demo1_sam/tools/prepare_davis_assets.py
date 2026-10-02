#!/usr/bin/env python3
"""Validate and optionally extract the official DAVIS2017 trainval 480p ZIP.

Standard library only. No download, prompt generation, image/mask decoding,
model execution or GPU use. Indexed GT PNG bytes are copied without conversion.

Example (after the official download finishes):
  python tools/prepare_davis_assets.py \
    --archive assets/datasets/archives/DAVIS-2017-trainval-480p.zip \
    --output-parent assets/datasets --extract
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BYTES = 832766765
SOURCE_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
DEFAULT_RECEIPT = ROOT / "assets/manifests/davis_archive_receipt.json"
CHUNK_BYTES = 1024 * 1024


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def safe_member(info):
    name = info.filename
    original = getattr(info, "orig_filename", name)
    if "\x00" in original or "\\" in name or ":" in name or name.startswith("/"):
        raise ValueError(f"unsafe archive member: {original!r}")
    segments = name.rstrip("/").split("/")
    if not segments or segments[0] != "DAVIS" or any(part in ("", ".", "..") for part in segments):
        raise ValueError(f"member is not a safe DAVIS-relative path: {name!r}")
    if len(segments) == 1 and not info.is_dir():
        raise ValueError("DAVIS root must be a directory")
    file_type = stat.S_IFMT(info.external_attr >> 16)
    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"nonregular/symlink ZIP member rejected: {name}")
    if (file_type == stat.S_IFDIR and not info.is_dir()) or (file_type == stat.S_IFREG and info.is_dir()):
        raise ValueError(f"inconsistent file/directory ZIP member: {name}")
    if info.flag_bits & 1:
        raise ValueError(f"encrypted member is not supported: {name}")
    return PurePosixPath(*segments)


def checked_members(archive):
    members, kinds = [], {}
    for info in archive.infolist():
        relative = safe_member(info)
        key = str(relative)
        if key in kinds:
            raise ValueError(f"duplicate archive path: {key}")
        kinds[key] = info.is_dir()
        members.append((info, relative))
    for _info, relative in members:
        for ancestor in relative.parents:
            if str(ancestor) in kinds and not kinds[str(ancestor)]:
                raise ValueError(f"archive file is an ancestor of another member: {ancestor}")
    return members


def split_sequences(archive, files, split):
    name = f"DAVIS/ImageSets/2017/{split}.txt"
    if name not in files:
        raise ValueError(f"missing official split list: {name}")
    raw = archive.read(files[name])  # CRC checked by zipfile.
    values = [line.strip() for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"empty or duplicate sequences in {name}")
    if any(value in (".", "..") or "/" in value or "\\" in value or ":" in value for value in values):
        raise ValueError(f"invalid sequence name in {name}")
    return values


def frame_record(info):
    return {"path": info.filename, "frame_id": PurePosixPath(info.filename).stem,
            "bytes": info.file_size, "crc32_hex": f"{info.CRC:08x}"}


def inventory_archive(archive, members):
    files = {str(relative): info for info, relative in members if not info.is_dir()}
    train, val = split_sequences(archive, files, "train"), split_sequences(archive, files, "val")
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise ValueError(f"official train/val overlap: {overlap}")
    images, ground_truth = {}, {}
    for info, relative in members:
        parts = relative.parts
        if info.is_dir() or len(parts) != 5 or parts[2] != "480p":
            continue
        if parts[1] == "JPEGImages" and relative.suffix.lower() in (".jpg", ".jpeg"):
            images.setdefault(parts[3], []).append(info)
        elif parts[1] == "Annotations" and relative.suffix.lower() == ".png":
            ground_truth.setdefault(parts[3], []).append(info)
    sequences = []
    all_names = sorted(set(images) | set(ground_truth) | set(train) | set(val))
    for name in all_names:
        image_files = sorted(images.get(name, []), key=lambda info: info.filename)
        gt_files = sorted(ground_truth.get(name, []), key=lambda info: info.filename)
        if not image_files or not gt_files:
            raise ValueError(f"sequence lacks image frames or GT frames: {name}")
        image_stems = [PurePosixPath(info.filename).stem for info in image_files]
        gt_stems = [PurePosixPath(info.filename).stem for info in gt_files]
        if len(image_stems) != len(set(image_stems)) or len(gt_stems) != len(set(gt_stems)):
            raise ValueError(f"duplicate frame stem in sequence: {name}")
        first_image, first_gt = frame_record(image_files[0]), frame_record(gt_files[0])
        paired = first_image["frame_id"] == first_gt["frame_id"]
        if not paired:
            raise ValueError(f"first image/GT frame not paired: {name}")
        missing_gt = sorted(set(image_stems) - set(gt_stems))
        missing_image = sorted(set(gt_stems) - set(image_stems))
        sequences.append({"sequence": name, "split": "train" if name in train else ("val" if name in val else "unlisted"),
                          "image_frame_count": len(image_files), "gt_frame_count": len(gt_files),
                          "first_image": first_image, "first_gt": first_gt, "first_frame_paired": paired,
                          "all_image_gt_frame_stems_match": not missing_gt and not missing_image,
                          "image_stems_missing_gt": missing_gt, "gt_stems_missing_image": missing_image})
    return {"train_sequences": train, "val_sequences": val, "train_count": len(train), "val_count": len(val),
            "train_val_disjoint": True, "sequence_count": len(sequences), "sequences": sequences,
            "image_frame_total": sum(item["image_frame_count"] for item in sequences),
            "gt_frame_total": sum(item["gt_frame_count"] for item in sequences),
            "sequences_not_in_2017_splits": sorted(set(all_names) - set(train) - set(val))}


def safe_destination(parent, relative):
    destination = parent.joinpath(*relative.parts)
    current = parent
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"existing symlink in extraction destination: {current}")
    if not destination.resolve(strict=False).is_relative_to(parent):
        raise ValueError(f"destination escapes output parent: {destination}")
    return destination


def file_crc(path):
    crc, size = 0, 0
    with Path(path).open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            size += len(chunk)
            crc = zlib.crc32(chunk, crc)
    return size, crc & 0xffffffff


def existing_matches(path, info):
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != info.file_size:
        raise FileExistsError(f"existing file differs in type/size; not overwritten: {path}")
    size, crc = file_crc(path)
    if size != info.file_size or crc != info.CRC:
        raise FileExistsError(f"existing file CRC differs; not overwritten: {path}")
    return True


def prepare_archive(archive_path, output_parent, receipt_path=DEFAULT_RECEIPT, extract=False,
                    _expected_bytes=EXPECTED_BYTES):
    archive_path, parent = Path(archive_path).resolve(), Path(output_parent).resolve()
    receipt = {"schema_version": 1, "status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "archive": str(archive_path), "expected_bytes": _expected_bytes,
               "expected_size_origin": "fixed official dataset source metadata" if _expected_bytes == EXPECTED_BYTES else "synthetic test override",
               "official_source_url": SOURCE_URL, "output_parent": str(parent), "dataset_root": str(parent / "DAVIS"),
               "extract_requested": extract, "archive_crc_verified": False,
               "annotation_byte_policy": "copy indexed PNG bytes exactly; no mask/image decoding or conversion",
               "scope": "asset inventory/preparation only; no prompts/model/metrics/GPU/network"}
    temporary = None
    try:
        actual_bytes = archive_path.stat().st_size
        receipt["actual_bytes"] = actual_bytes
        if actual_bytes != _expected_bytes:
            raise ValueError(f"archive byte count {actual_bytes} != expected {_expected_bytes}; download may be incomplete")
        with zipfile.ZipFile(archive_path) as archive:
            members = checked_members(archive)
            receipt["inventory"] = inventory_archive(archive, members)
            receipt["archive_file_count"] = sum(not info.is_dir() for info, _relative in members)
            receipt["archive_directory_count"] = sum(info.is_dir() for info, _relative in members)
            receipt["total_uncompressed_bytes"] = sum(info.file_size for info, _relative in members)
            # Check every existing destination before creating any dataset file.
            identical = set()
            if extract:
                for info, relative in members:
                    destination = safe_destination(parent, relative)
                    for ancestor in destination.parents:
                        if ancestor == parent:
                            break
                        if ancestor.exists() and not ancestor.is_dir():
                            raise FileExistsError(f"existing ancestor is not a directory: {ancestor}")
                    if info.is_dir():
                        if destination.exists() and not destination.is_dir():
                            raise FileExistsError(f"existing directory destination has different type: {destination}")
                    elif existing_matches(destination, info):
                        identical.add(str(relative))
                parent.mkdir(parents=True, exist_ok=True)
            created, reused, verified = 0, 0, 0
            for info, relative in members:
                if info.is_dir():
                    if extract:
                        safe_destination(parent, relative).mkdir(parents=True, exist_ok=True)
                    continue
                destination = safe_destination(parent, relative) if extract else None
                already_present = str(relative) in identical
                output_handle = None
                if extract and not already_present:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    output_handle = tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                        prefix=f".{destination.name}.", suffix=".partial", delete=False)
                    temporary = Path(output_handle.name)
                size, crc = 0, 0
                try:
                    with archive.open(info) as source:
                        while chunk := source.read(CHUNK_BYTES):
                            size += len(chunk)
                            crc = zlib.crc32(chunk, crc)
                            if output_handle is not None:
                                output_handle.write(chunk)
                finally:
                    if output_handle is not None:
                        output_handle.close()
                if size != info.file_size or (crc & 0xffffffff) != info.CRC:
                    raise zipfile.BadZipFile(f"decompressed size/CRC mismatch: {info.filename}")
                verified += 1
                if extract:
                    if already_present:
                        reused += 1
                    else:
                        # Hard-link commit creates only if absent, never replaces a
                        # conflicting file appearing after the preflight check.
                        try:
                            os.link(temporary, destination)
                            created += 1
                        except FileExistsError:
                            existing_matches(destination, info)
                            reused += 1
                        temporary.unlink()
                        temporary = None
            receipt["archive_crc_verified"] = True
            receipt["crc_verified_file_count"] = verified
            receipt["extraction"] = {"performed": extract, "created_files": created, "reused_identical_files": reused,
                                      "bytes_preserved": extract, "all_archive_members_processed": True}
        receipt["status"] = "READY_EXTRACTED" if extract else "VERIFIED_ARCHIVE_NOT_EXTRACTED"
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(receipt_path, receipt)
        return receipt
    except Exception as error:
        receipt["status"] = "ERROR"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_json(receipt_path, receipt)
        raise
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--extract", action="store_true", help="copy every DAVIS-root member without decoding GT")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    try:
        result = prepare_archive(args.archive, args.output_parent, args.receipt, args.extract)
    except Exception as error:
        print(f"DAVIS preparation failed: {error}", file=sys.stderr)
        return 2
    inventory = result["inventory"]
    print(json.dumps({"status": result["status"], "sequences": inventory["sequence_count"],
                      "train": inventory["train_count"], "val": inventory["val_count"],
                      "image_frames": inventory["image_frame_total"], "gt_frames": inventory["gt_frame_total"],
                      "receipt": str(args.receipt)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
