#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/demo1_sam
while [ ! -f assets/datasets/archives/DAVIS-2017-trainval-480p.zip ]; do sleep 15; done
/root/miniconda3/bin/python tools/prepare_davis_assets.py --archive assets/datasets/archives/DAVIS-2017-trainval-480p.zip --output-parent assets/datasets --extract --receipt assets/manifests/davis_archive_receipt.json
/root/miniconda3/bin/python - <<'PY'
import pathlib,zipfile,shutil
r=pathlib.Path('assets/checkpoints');p=r/'efficient_sam_vits.pt.zip';q=r/'efficient_sam_vits.pt'
if p.exists() and not q.exists():
 with zipfile.ZipFile(p) as z, z.open('efficient_sam_vits.pt') as src, q.with_suffix('.pt.extracting').open('wb') as out:
  shutil.copyfileobj(src,out,1024*1024)
 q.with_suffix('.pt.extracting').replace(q)
PY
/root/miniconda3/bin/python tools/audit_prepared_assets.py --verify-hashes --output assets/manifests/asset_preparation_remote_receipt.json
