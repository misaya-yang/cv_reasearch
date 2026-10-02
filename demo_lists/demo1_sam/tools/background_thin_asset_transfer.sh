#!/usr/bin/env bash
set -euo pipefail
cd /Users/yang/projects/CVPR2027/demo_lists/demo1_sam
python3 tools/prepare_thinobject_subset.py --workers 8 >> assets/datasets/thinobject5k_core_v1/download.log 2>&1
python3 -c 'import json; assert json.load(open("assets/datasets/thinobject5k_core_v1/receipt.json"))["full_subset_ready"]'
rsync -az --partial --exclude '.partial/' -e 'ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -p 57510' assets/datasets/thinobject5k_core_v1/ root@connect.westc.seetacloud.com:/root/autodl-tmp/demo1_sam/assets/datasets/thinobject5k_core_v1/
rsync -az -e 'ssh -o BatchMode=yes -o ConnectTimeout=15 -p 57510' tools/prepare_thinobject_subset.py root@connect.westc.seetacloud.com:/root/autodl-tmp/demo1_sam/tools/
rsync -az -e 'ssh -o BatchMode=yes -o ConnectTimeout=15 -p 57510' assets/manifests/thinobject5k_subset_ranges.json root@connect.westc.seetacloud.com:/root/autodl-tmp/demo1_sam/assets/manifests/
ssh -o BatchMode=yes -o ConnectTimeout=15 -p 57510 root@connect.westc.seetacloud.com '/root/miniconda3/bin/python /root/autodl-tmp/demo1_sam/tools/prepare_thinobject_subset.py --workers 8 --retries 1 --timeout 10 > /root/autodl-tmp/demo1_sam/assets/datasets/thinobject5k_core_v1/remote_copy_verify.log 2>&1'
date -u +%FT%TZ > assets/datasets/thinobject5k_core_v1/REMOTE_COPY_COMPLETE
