"""Retrieve the contiguous original ZIP mask block with all original CRC checks.

704 small masks are contiguous in the fixed documented Kaggle v1 ZIP. A single
range read avoids 1,408 per-member HTTP requests. The main cohort downloader
independently revalidates every staged member before writing its source receipt.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import zlib


ASSETS = Path(__file__).resolve().parents[4].parent / "cv_data"
ROOT = ASSETS / "setup/asset_downloads/Chest_autonomous_20261010"


def main():
    spec = importlib.util.spec_from_file_location("chest_exact_source", ROOT / "download_exact_pool.py")
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    inventory = sorted(json.loads((ROOT / "archive_members.json").read_text()), key=lambda m: m["header_offset"])
    members = [m for m in inventory if m["name"].startswith("Lung Segmentation/masks/") and m["name"].endswith(".png")]
    if len(members) != 704:
        raise ValueError("Exact original mask pool differs")
    first = members[0]["header_offset"]
    end = next(m["header_offset"] for m in inventory if m["header_offset"] > members[-1]["header_offset"])
    within = [m for m in inventory if first <= m["header_offset"] < end]
    if within != members:
        raise ValueError("Original mask block is not contiguous")
    value = downloader.fetch(first, end - 1)
    decoded = []
    for member in members:
        offset = member["header_offset"] - first
        header = value[offset:offset + 30]
        fields = struct.unpack("<IHHHHHIIIHH", header)
        signature, flags, compression = fields[0], fields[2], fields[3]
        name_length, extra_length = fields[-2:]
        if signature != 0x04034B50 or flags & 1 or compression != 8:
            raise ValueError("Unexpected original ZIP member header")
        name = value[offset + 30:offset + 30 + name_length].decode("utf-8")
        if name != member["name"]:
            raise ValueError("Original ZIP member name differs")
        begin = offset + 30 + name_length + extra_length
        compressed = value[begin:begin + member["compressed_bytes"]]
        png = zlib.decompress(compressed, -15)
        if len(png) != member["bytes"] or f"{zlib.crc32(png):08x}" != member["crc32"]:
            raise ValueError("Original ZIP mask CRC or size differs")
        decoded.append((member, png))
    stage = ROOT / "staging/LungSegmentation/masks"
    stage.mkdir(parents=True, exist_ok=True)
    for member, png in decoded:
        destination = stage / Path(member["name"]).name
        if destination.exists():
            if destination.read_bytes() != png:
                raise ValueError("Existing staged original mask differs")
            continue
        temporary = destination.with_suffix(".mask_prefill.part")
        temporary.write_bytes(png)
        temporary.replace(destination)
    receipt = dict(status="ALL_ORIGINAL_MASK_MEMBERS_CRC_VERIFIED", source_url=downloader.URL,
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   original_archive_byte_range=[first, end - 1],
                   range_bytes=len(value), range_sha256=hashlib.sha256(value).hexdigest(),
                   original_members_verified=len(decoded),
                   decoded_png_bytes=sum(len(png) for _, png in decoded),
                   model_forwards=0,
                   note="Transport optimization only; main downloader independently checks all original members before final source receipt")
    downloader.write(ROOT / "source_mask_group_range.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
