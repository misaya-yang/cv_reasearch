"""Link the complete publisher-verified FIVES archive to the immutable test pool.

The released FoRIS test pool was prepared from complete, checksum-validated
original RAR members. This separate receipt closes the optional whole-source
archive MD5 check without changing the pool receipt or any frozen manifest.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ASSETS = Path(__file__).resolve().parents[4].parent / "cv_data"
ROOT = ASSETS / "setup/asset_downloads/Fundus_autonomous_20261010"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    receipt_path = ROOT / "preparation_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    metadata = json.loads((ROOT / "source_figshare_metadata.json").read_text())
    source = next(f for f in metadata["files"] if f["id"] == 34969398)
    archive = ROOT / "FIVES_v1.rar"
    if archive.stat().st_size != source["size"]:
        raise ValueError("Original Figshare archive size differs")
    prefix = receipt["original_source_prefix"]
    remaining_prefix = prefix["bytes"]
    md5, archive_sha, prefix_sha = hashlib.md5(), hashlib.sha256(), hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            md5.update(block)
            archive_sha.update(block)
            if remaining_prefix:
                prefix_sha.update(block[:remaining_prefix])
                remaining_prefix = max(0, remaining_prefix - len(block))
    if md5.hexdigest() != source["computed_md5"]:
        raise ValueError("Original Figshare publisher MD5 differs")
    if remaining_prefix or prefix_sha.hexdigest() != prefix["sha256"]:
        raise ValueError("Complete original archive differs from the prepared source prefix")
    for record in receipt["files"]:
        path = ASSETS / record["path"]
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Prepared test-pool file differs: " + record["path"])
    value = dict(
        status="VERIFIED_ORIGINAL_ARCHIVE_AND_IMMUTABLE_TEST_POOL",
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_url="https://ndownloader.figshare.com/files/34969398",
        source_article_id=19688169,
        source_file_id=34969398,
        source_dataset_version=1,
        archive=dict(path=str(archive.relative_to(ASSETS)), bytes=archive.stat().st_size,
                     md5=md5.hexdigest(), sha256=archive_sha.hexdigest(),
                     supplied_md5=source["computed_md5"], supplied_md5_passed=True),
        prepared_source_prefix=dict(bytes=prefix["bytes"], sha256=prefix_sha.hexdigest(),
                                    matches_immutable_preparation_receipt=True),
        preparation_receipt=dict(path=str(receipt_path.relative_to(ASSETS)),
                                 sha256=sha256(receipt_path), unchanged=True),
        pool_files_rehashed=len(receipt["files"]),
        pool_bytes=sum(r["bytes"] for r in receipt["files"]),
        existing_assets_overwritten=False,
        model_forwards=0,
    )
    destination = ROOT / "archive_verification.json"
    if destination.exists():
        raise FileExistsError("Refuse to overwrite an existing archive verification receipt")
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps({"status": value["status"], "archive_sha256": archive_sha.hexdigest(),
                      "pool_files_rehashed": value["pool_files_rehashed"]}))


if __name__ == "__main__":
    main()
