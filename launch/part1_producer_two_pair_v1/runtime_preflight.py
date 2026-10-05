"""Read-only remote dependency receipt; hides CUDA before importing packages."""
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path("/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9")
assets = {}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add(path):
    path = Path(path)
    assets[str(path)] = {"exists": path.is_file()}
    if path.is_file():
        assets[str(path)].update(bytes=path.stat().st_size, sha256=digest(path))


static = [
    "/root/miniconda3/bin/python",
    "/root/autodl-tmp/demo9_extent/scripts/extent_experiment.py",
    "/root/autodl-tmp/demo9_extent/tics/__init__.py",
    "/root/autodl-tmp/demo9_extent/tics/native_assets.py",
    "/root/autodl-tmp/demo4/icx/__init__.py",
    "/root/autodl-tmp/demo4/icx/common.py",
    "/root/autodl-tmp/demo8_local_verification/foris_source/models/__init__.py",
    "/root/autodl-tmp/demo8_local_verification/foris_source/models/foris.py",
    "/root/autodl-tmp/demo8_local_verification/foris_source/utils/__init__.py",
    "/root/autodl-tmp/demo8_local_verification/foris_source/utils/refinement.py",
    "/root/autodl-tmp/demo8_local_verification/foris_source/utils/clustering.py",
    "/root/autodl-tmp/demo8_local_verification/foris_source/utils/data.py",
    "/root/autodl-tmp/demo8_local_verification/runtime/extensions/Permutohedral.cpython-312-x86_64-linux-gnu.so",
    "/root/autodl-tmp/demo8_local_verification/runtime/extensions/Permutohedral_gpu.cpython-312-x86_64-linux-gnu.so",
    "/root/demo4_cache/models/dinov3-vitl16-timm/config.json",
    "/root/demo4_cache/models/dinov3-vitl16-timm/model.safetensors",
    "/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt",
]
for base in (
    "/root/autodl-tmp/demo8_local_verification/crf_source/src/CRF",
    "/root/autodl-tmp/demo8_local_verification/crf_source/src/PermutohedralFiltering",
):
    static.extend(str(p) for p in Path(base).rglob("*.py") if "__pycache__" not in p.parts)
for path in static:
    add(path)
for block in range(7):
    for name in ("manifest.json", "sealed.json"):
        add(ROOT / f"outputs/claude_official/run{block}" / name)
for name in ("manifest.json", "config.json", "sealed.json"):
    add(ROOT / "outputs/frozen_subtoken1200_v1" / name)
add(ROOT / "outputs/claude_subtoken_fresh600/report.json")
add(ROOT / "outputs/claude_official/batch0.json")
config = json.loads((ROOT / "outputs/frozen_subtoken1200_v1/config.json").read_text())
rows = json.loads((ROOT / "outputs/frozen_subtoken1200_v1/manifest.json").read_text())
selected = [rows[0], next(r for r in rows if r["fold"] != rows[0]["fold"])]
for name in ("manifest.json", "config.json", "sealed.json", "audits.json"):
    add(Path(config["prepared"]) / name)
for path in config["source_code_sha256"]:
    add(path)
man = json.loads((ROOT / "outputs/claude_official/batch0.json").read_text())
for row in selected:
    add(row["feature_export"])
    add(row["packet_export"])
    add(Path(row["recheck_run"]) / "fields" / (row["key"] + ".npz"))
    add(Path(row["recheck_run"]) / "predictions" / (row["key"] + ".npz"))
    for role in ("support", "query"):
        add(Path(man["data_root"]) / row[role])
    add(Path(man["annotation_root"]) / Path(row["support"]).with_suffix(".png"))
sys.path[:0] = [
    "/root/autodl-tmp/demo8_local_verification/foris_source",
    "/root/autodl-tmp/demo9_extent",
    "/root/autodl-tmp/demo4",
    "/root/demo4_cache/env",
    "/root/autodl-tmp/demo8_local_verification/crf_source/src",
    "/root/autodl-tmp/demo8_local_verification/runtime/extensions",
]
imports = {}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    for name in (
        "numpy", "torch", "scipy", "PIL", "timm", "safetensors", "CRF",
        "PermutohedralFiltering", "Permutohedral", "Permutohedral_gpu",
        "models.foris", "tics.native_assets", "icx.common",
    ):
        try:
            module = importlib.import_module(name)
            imports[name] = dict(ok=True, path=getattr(module, "__file__", None),
                                 version=str(getattr(module, "__version__", "not supplied")))
        except Exception as error:
            imports[name] = dict(ok=False, error=repr(error))
result = dict(state="READ_ONLY_REMOTE_ASSET_AND_IMPORT_CHECK", root=str(ROOT),
              checked_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              assets=assets, module_imports=imports, selected_pairs=selected,
              basis_expected_sha256="9b9b20755a796cbda11bb7220d246ee540106e40cb024e249a5f63f884b6a116",
              gpu_execution=False, cuda_visible_devices="", import_stdout=captured.getvalue(),
              existing_output=(ROOT / "outputs/part1_producer_two_pair_v1").exists())
print("REMOTE_RECEIPT_JSON_BEGIN")
print(json.dumps(result, indent=2))
