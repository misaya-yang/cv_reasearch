#!/usr/bin/env python3
"""Read-only asset audit. No model imports, installation, GPU or experiment work."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--verify-hashes", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = a.root.resolve()
    spec = json.loads((r / "assets/manifests/asset_inventory_spec.json").read_text())
    receipt = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "root": str(r), "scope": "Asset integrity only; no inference/quality/speed validation",
               "checkpoints": [], "source_archives": [], "download_receipts": {},
               "sam3_checkpoint": spec["sam3_checkpoint"]}
    for x in spec["checkpoints"]:
        f = r / "assets/checkpoints" / x["name"]
        row = dict(x, path=str(f), status="MISSING_OR_IN_PROGRESS")
        if f.exists():
            row["actual_bytes"] = f.stat().st_size
            row["status"] = "SIZE_MATCH_HASH_NOT_CHECKED" if row["actual_bytes"] == x["bytes"] else "INCOMPLETE_OR_WRONG_SIZE"
            if a.verify_hashes and row["actual_bytes"] == x["bytes"]:
                row["actual_sha256"] = sha256(f)
                row["status"] = "BYTES_VERIFIED" if row["actual_sha256"] == x["expected_sha256"] else "HASH_MISMATCH"
        receipt["checkpoints"].append(row)
    for x in spec["source_archives"]:
        f = r / x["archive"]
        row = dict(x, status="MISSING")
        if f.exists():
            row["actual_sha256"] = sha256(f)
            row["status"] = "BYTES_VERIFIED" if row["actual_sha256"] == x["sha256"] else "HASH_MISMATCH"
        receipt["source_archives"].append(row)
    paths = list((r / "assets/manifests").glob("*download_status.json"))
    paths += list((r / "assets/checkpoints").glob("public_assets_status*.json"))
    for f in paths:
        # A failed earlier route stays in the receipt alongside the later successful route.
        receipt["download_receipts"][str(f.relative_to(r))] = json.loads(f.read_text())
    versions = {}
    for name in ("torch", "torchvision", "numpy", "hydra-core", "omegaconf", "iopath", "timm", "tqdm", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    receipt["base_environment_versions_without_importing_models"] = versions
    receipt["data_disk_free_bytes"] = shutil.disk_usage(r).free
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"checkpoints": {x["name"]: x["status"] for x in receipt["checkpoints"]},
                      "source_archives": len(receipt["source_archives"]),
                      "output": str(a.output)}, indent=2))


if __name__ == "__main__":
    main()
