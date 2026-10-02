#!/usr/bin/env python3
"""Official public checkpoints only. Resumable byte ranges; no Torch/GPU import."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

ASSETS = [
    {"name": "sam2.1_hiera_small.pt", "bytes": 184416285,
     "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"},
    {"name": "sam2.1_hiera_large.pt", "bytes": 898083611,
     "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"},
    {"name": "sam2.1_hq_hiera_large.pt", "bytes": 898844313,
     "url": "https://huggingface.co/lkeab/hq-sam/resolve/09b02a333b37772133eff3997bdba997867374b7/sam2.1_hq_hiera_large.pt",
     "sha256": "03b8be4b4f282c3d7365715a033918d9ac67cbc597660ccc274448aa1c8cdb05"},
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate(path, asset):
    if path.stat().st_size != asset["bytes"]:
        raise ValueError(f"Wrong size; existing file retained: {path}")
    sha = digest(path)
    if asset.get("sha256") and sha != asset["sha256"]:
        raise ValueError(f"Official SHA mismatch; file retained: {path}")
    blob_sha = None
    if asset.get("git_blob_sha1"):
        blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                blob.update(block)
        blob_sha = blob.hexdigest()
        if blob_sha != asset["git_blob_sha1"]:
            raise ValueError(f"Official Git blob mismatch; file retained: {path}")
    with zipfile.ZipFile(path) as archive:
        if asset.get("kind", "torch_zip") == "torch_zip" and not any(n.endswith("/data.pkl") for n in archive.namelist()):
            raise ValueError("Checkpoint is not the expected PyTorch archive")
        if archive.testzip() is not None:
            raise ValueError("Checkpoint archive CRC failed")
    return {"status": "READY", "path": str(path), "bytes": path.stat().st_size, "sha256": sha,
            "downloaded_bytes": path.stat().st_size,
            "git_blob_sha1": blob_sha,
            "validated": "expected size, archive CRC; official SHA additionally checked where published"}


def part(asset, folder, index, count):
    start = asset["bytes"] * index // count
    end = asset["bytes"] * (index + 1) // count - 1
    path = folder / f"part_{index:02d}.bin"
    expected = end - start + 1
    failures = 0
    while failures < 4:
        current = path.stat().st_size if path.exists() else 0
        if current == expected:
            return path
        if current > expected:
            raise ValueError(f"Oversize chunk retained: {path}")
        # Short requests reconnect instead of leaving one slow TCP stream to
        # transfer 100+ MiB. The persisted part boundaries remain unchanged.
        request_end = min(end, start + current + 8 * 1024 * 1024 - 1)
        request_bytes = request_end - start - current + 1
        header = folder / f"part_{index:02d}.headers"
        with path.open("ab") as out, (folder / f"part_{index:02d}.log").open("a") as log:
            p = subprocess.run(["curl", "-fLsS", "--retry", "0", "--connect-timeout", "15",
                "--max-time", "120", "--max-filesize", str(request_bytes),
                "--range", f"{start + current}-{request_end}", "--dump-header", str(header),
                asset.get("resolved_url", asset["url"])], stdout=out, stderr=log)
        ranges = re.findall(r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)", header.read_text(errors="replace") if header.exists() else "")
        if not ranges and p.returncode and path.stat().st_size == current:
            failures += 1
            time.sleep(2 * failures)
            continue
        if not ranges or tuple(map(int, ranges[-1])) != (start + current, request_end, asset["bytes"]):
            raise ValueError(f"Range contract failed; chunk retained for inspection: {path}")
        if p.returncode == 0 and path.stat().st_size == current + request_bytes:
            failures = 0
            continue
        if path.stat().st_size > current:
            failures = 0
        else:
            failures += 1
            time.sleep(2 * failures)
    raise RuntimeError(f"Incomplete chunk; rerun resumes it: {path}")


def fetch(asset, root, count):
    final = root / asset["name"]
    if final.exists():
        return validate(final, asset)
    folder = root / ".chunks" / asset["name"]
    folder.mkdir(parents=True, exist_ok=True)
    partition = folder / "partition.json"
    partition_data = {"url": asset["url"], "bytes": asset["bytes"], "parts": count}
    if partition.exists() and json.loads(partition.read_text()) != partition_data:
        raise ValueError(f"Existing chunk partition differs; retain it and use the original --parts: {folder}")
    partition.write_text(json.dumps(partition_data, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=count) as pool:
        chunks = list(pool.map(lambda i: part(asset, folder, i, count), range(count)))
    assembling = final.with_suffix(final.suffix + ".assembling")
    with assembling.open("wb") as out:
        for chunk in chunks:
            with chunk.open("rb") as f:
                shutil.copyfileobj(f, out, 1024 * 1024)
    result = validate(assembling, asset)
    assembling.replace(final)
    return {**result, "path": str(final)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--parts", type=int, default=4)
    p.add_argument("--names", nargs="+",
                   help="Download only selected checkpoints; avoids another transfer's targets.")
    p.add_argument("--asset-list", type=Path, help="Curated JSON list of public assets: name, bytes, url, kind, optional sha256.")
    p.add_argument("--url-map", type=Path, help="Optional ephemeral official CDN redirects keyed by asset name. Never stored in public manifests.")
    p.add_argument("--status-file", type=Path, help="Explicit separate progress receipt for this asset batch.")
    a = p.parse_args()
    if a.parts < 1:
        p.error("parts must be positive")
    a.directory.mkdir(parents=True, exist_ok=True)
    assets = json.loads(a.asset_list.read_text()) if a.asset_list else ASSETS
    if a.names and set(a.names) - {x["name"] for x in assets}:
        p.error("Unknown selected asset name")
    selected = [dict(x) for x in assets if a.names is None or x["name"] in a.names]
    if not selected or any(Path(x["name"]).name != x["name"] or x["bytes"] <= 0 for x in selected):
        p.error("Asset list must contain safe basenames and positive byte counts")
    if a.url_map:
        redirects = json.loads(a.url_map.read_text())
        for x in selected:
            if x["name"] in redirects:
                x["resolved_url"] = redirects[x["name"]]
    state = {x["name"]: {**{k: v for k, v in x.items() if k != "resolved_url"}, "status": "DOWNLOADING"} for x in selected}
    # Separate partial runs do not overwrite each other's manifests.
    suffix = "_" + "_".join(x["name"].split(".")[1] for x in selected) if a.names else ""
    manifest = a.status_file or a.directory / f"public_assets_status{suffix}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {x["name"]: pool.submit(fetch, x, a.directory, a.parts) for x in selected}
        while pending:
            for name, future in list(pending.items()):
                if future.done():
                    try:
                        state[name].update(future.result())
                    except Exception as e:
                        state[name].update(status="FAILED_RESUMABLE", error=str(e))
                    del pending[name]
                else:
                    folder = a.directory / ".chunks" / name
                    state[name]["downloaded_bytes"] = sum(f.stat().st_size for f in folder.glob("part_*.bin"))
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(manifest)
            print(json.dumps({k: {x: v.get(x) for x in ("status", "downloaded_bytes", "bytes")}
                              for k, v in state.items()}), flush=True)
            if pending:
                time.sleep(15)
    if any(x["status"] != "READY" for x in state.values()):
        raise RuntimeError("Some downloads incomplete; inspect manifest. Rerun resumes, never silently marks ready.")


if __name__ == "__main__":
    main()
