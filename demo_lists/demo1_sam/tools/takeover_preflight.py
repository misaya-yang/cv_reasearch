"""CPU-only readiness check of existing real assets; never downloads or starts CUDA."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
from takeover_common import *


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    configure()
    sys.path.insert(0, str(ROOT/"assets/source/segment-anything"))
    from segment_anything import sam_model_registry
    checkpoint = ROOT/"assets/checkpoints/sam_vit_b_01ec64.pth"
    subset = ROOT/"assets/coco_quality_seed2027_v1"
    manifest = json.loads((subset/"manifest.json").read_text())
    assert manifest["model_outputs_used_for_selection"] is False
    assert len(manifest["images"]) == 24
    objects = 0
    estimated_bytes = 0
    for item in manifest["images"]:
        with Image.open(subset/item["image_file"]) as image:
            assert image.size == (item["width"], item["height"])
        with np.load(subset/item["gt_file"], allow_pickle=False) as gt:
            assert gt["gt"].shape == (len(item["objects"]), item["height"], item["width"])
            assert gt["annotation_ids"].tolist() == [o["annotation_id"] for o in item["objects"]]
            for mask, obj in zip(gt["gt"], item["objects"]):
                for key in ("central_xy", "near_boundary_xy"):
                    x,y = obj[key]
                    assert mask[int(y),int(x)]
        count = len(item["objects"])
        objects += count
        # 12 arms x 3 regimes: full bool masks, GT and 256^2 FP32 logits.
        estimated_bytes += 12*3*count*(5*item["height"]*item["width"]+4*256*256*4)
    free = shutil.disk_usage(ROOT).free
    assert free > estimated_bytes*1.25, (free, estimated_bytes)
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).eval()
    assert sam.mask_decoder.mask_tokens.weight.shape == (4,256)
    assert sam.mask_decoder.iou_prediction_head.layers[-1].out_features == 4
    assert all(t.device.type == "cpu" for t in sam.parameters())
    report = {"status": "READY_CPU_ASSETS", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "torch": torch.__version__, "python": sys.version, "checkpoint_bytes": checkpoint.stat().st_size,
              "checkpoint_load_cpu": "PASS", "images": len(manifest["images"]), "instances": objects,
              "image_GT_prompt_contract": "PASS", "remaining_disk_bytes": free,
              "estimated_output_bytes": estimated_bytes, "cuda_executed": False,
              "arms": ARMS, "imports": "SAM/PyTorch/Pillow/NumPy/experimental adapters available"}
    atomic_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
