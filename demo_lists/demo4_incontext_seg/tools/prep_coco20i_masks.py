"""Rebuild the COCO-20i semantic masks (HSNet/PFENet format) from the official instance annotations.

The masks the benchmark normally uses are distributed on Google Drive, which this server cannot reach.
Format: one PNG per image, pixel value = class index + 1 (classes = the 80 COCO category ids in sorted order), 0 = background.
Annotations are painted in annotation-id order, so a later instance overwrites an earlier one where they overlap.

  python prep_coco20i_masks.py <annotations_trainval2017.zip> <out_dir>      # writes <out_dir>/val2014/COCO_val2014_*.png
"""
import json, os, sys, zipfile, pickle, glob
import numpy as np
from PIL import Image
import pycocotools.mask as mu

src, out = sys.argv[1], sys.argv[2]; os.makedirs(f"{out}/val2014", exist_ok=True)
need = set()
for f in glob.glob(os.path.join(os.path.dirname(out.rstrip("/")), "splits/val/fold*.pkl")):
    d = pickle.load(open(f, "rb"))
    for k in d: need.update(os.path.basename(n) for n in d[k])
print("images needed:", len(need), flush=True)
ids = {int(n.split("_")[-1].split(".")[0]): n for n in need}
z = zipfile.ZipFile(src); anns, sizes, cats = {}, {}, None
for name in ("annotations/instances_train2017.json", "annotations/instances_val2017.json"):
    d = json.load(z.open(name)); cats = sorted(c["id"] for c in d["categories"])
    for im in d["images"]:
        if im["id"] in ids: sizes[im["id"]] = (im["height"], im["width"])
    for a in d["annotations"]:
        if a["image_id"] in ids: anns.setdefault(a["image_id"], []).append(a)
    print(name, len(sizes), flush=True)
cid = {c: i for i, c in enumerate(cats)}; n = 0
for i, name in ids.items():
    if i not in sizes: continue
    h, w = sizes[i]; m = np.zeros((h, w), np.uint8)
    for a in sorted(anns.get(i, []), key=lambda a: a["id"]):
        s = a["segmentation"]
        rle = mu.merge(mu.frPyObjects(s, h, w)) if isinstance(s, list) else (mu.frPyObjects(s, h, w) if isinstance(s["counts"], list) else s)
        m[mu.decode(rle) == 1] = cid[a["category_id"]] + 1
    Image.fromarray(m).save(f"{out}/val2014/{name[:-4]}.png"); n += 1
print("written", n, "missing", len(ids) - n)
