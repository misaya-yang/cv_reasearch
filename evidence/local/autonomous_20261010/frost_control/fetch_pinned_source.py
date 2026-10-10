"""Fetch inspected official FROST text source, never install or execute it."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


PIN = "b9ece69d7495a698c298e7cc3d16efacd4497a43"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DEST = REPO.parent / "cv_data/third_party/frost_official"
TREE = json.loads((HERE / "official_tree.json").read_text())
PATHS = [x for x in TREE["tree"] if x["type"] == "blob" and not x["path"].startswith("figure/")]


def fetch(entry):
    relative = entry["path"]
    url = f"https://raw.githubusercontent.com/jhpark-ai/FROST/{PIN}/{relative}"
    request = urllib.request.Request(url, headers={"User-Agent": "source-provenance-audit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()
    git_blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    if git_blob != entry["sha"] or len(content) != entry["size"]:
        raise ValueError(f"Git blob or byte count differs at {relative}")
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != content:
        raise ValueError(f"Refuse to replace existing source at {target}")
    target.write_bytes(content)
    return dict(path=str(target), relative=relative, url=url, bytes=len(content),
                git_blob_sha1=git_blob, sha256=hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    if TREE["truncated"]:
        raise ValueError("Repository tree is truncated")
    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(fetch, PATHS))
    receipt = dict(state="SHA_PINNED_OFFICIAL_TEXT_SOURCE_FETCHED", repository="https://github.com/jhpark-ai/FROST",
                   commit=PIN, time_utc=datetime.now(timezone.utc).isoformat(),
                   acquisition="GitHub raw commit URLs; each byte count and Git blob hash matched official tree API",
                   source_files=files, omitted="figure PNGs are not needed to inspect or validate the feature head",
                   git_clone_completed=False,
                   git_clone_error="RPC failed curl18 partial transfer; early EOF; invalid index-pack output",
                   installs=0, package_execution=False, backbone_execution=False)
    (HERE / "source_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(state=receipt["state"], commit=PIN, files=len(files))))
