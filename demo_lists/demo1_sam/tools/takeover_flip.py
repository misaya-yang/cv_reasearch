"""One horizontal-flip view with fixed original prompts and official SAM outputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"research/quality_mechanisms"))
import run_real_sam


def mirror_subset(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"images").mkdir(); (destination/"ground_truth").mkdir()
    payload = (source/"manifest.json").read_bytes()
    manifest = copy.deepcopy(json.loads(payload))
    manifest["view"] = "horizontal_flip_RGB_pixels_no_JPEG_reencoding"
    manifest["source_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["point_mapping"] = "x'=width-1-x, y'=y; pixel centers preserved"
    manifest["box_mapping"] = "exclusive tight XYXY: [width-xmax,ymin,width-xmin,ymax]"
    manifest["model_outputs_used_for_selection"] = False
    for item in manifest["images"]:
        original = next(x for x in json.loads(payload)["images"] if x["image_id"]==item["image_id"])
        rgb = np.asarray(Image.open(source/original["image_file"]).convert("RGB"))
        filename = f"images/{item['image_id']:012d}.png"
        Image.fromarray(rgb[:,::-1].copy()).save(destination/filename, compress_level=0)
        item["image_file"] = filename
        with np.load(source/original["gt_file"], allow_pickle=False) as arrays:
            gt, ids = arrays["gt"][:, :, ::-1].copy(), arrays["annotation_ids"].copy()
        np.savez(destination/item["gt_file"], gt=gt, annotation_ids=ids)
        width = item["width"]
        for obj, mask in zip(item["objects"], gt):
            for key in ("central_xy", "near_boundary_xy"):
                x,y = obj[key]; obj[key] = [width-1-x,y]
                assert mask[int(y), int(width-1-x)]
            x0,y0,x1,y1 = obj["tight_box_xyxy"]
            obj["tight_box_xyxy"] = [width-x1,y0,width-x0,y1]
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--subset-dir",type=Path,default=ROOT/"assets/coco_quality_seed2027_v1")
    p.add_argument("--device",choices=("cpu","cuda:0"),required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    mirror_subset(args.subset_dir,args.output_dir/"mirror_subset")
    # An official-only view; no redundant cache-arm matrix or GT-based ranker.
    run_real_sam.METHODS=("official",)
    sys.argv=["run_real_sam.py","--subset-dir",str(args.output_dir/"mirror_subset"),
              "--checkpoint",str(ROOT/"assets/checkpoints/sam_vit_b_01ec64.pth"),
              "--output-dir",str(args.output_dir/"official_view"),"--device",args.device,
              "--limit-images",str(len(json.loads((args.output_dir/"mirror_subset/manifest.json").read_text())["images"])),"--microbatch","4","--defer-quality","--npz-compression","none"]
    raise SystemExit(run_real_sam.main())
