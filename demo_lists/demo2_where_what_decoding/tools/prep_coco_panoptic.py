"""COCO panoptic -> 133-class semantic label PNGs (255 = unlabeled), for val2017 and a fixed subset of train2017.
Class index = rank of the COCO category id among the 133 panoptic categories (the detectron2 / Mask2Former order).

  python tools/prep_coco_panoptic.py /root/demo2_cache/coco /root/autodl-pub/COCO2017 10000
Layout written: <out>/images/{val,train}/*.jpg, <out>/semantic/{val,train}/*.png, <out>/categories.json
"""
import io, json, os, sys, zipfile
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

out, pub, n_train = sys.argv[1], sys.argv[2], int(sys.argv[3])
ann = zipfile.ZipFile(f"{out}/panoptic_annotations_trainval2017.zip")


def convert(args):
    split, start, stop = args
    z = zipfile.ZipFile(io.BytesIO(zipfile.ZipFile(f"{out}/panoptic_annotations_trainval2017.zip").read(f"annotations/panoptic_{split}2017.zip")))
    meta = json.load(open(f"{out}/panoptic_{split}.json")); cat = {c: i for i, c in enumerate(meta["cat_ids"])}
    for a in meta["annotations"][start:stop]:
        png = np.asarray(Image.open(io.BytesIO(z.read(f"panoptic_{split}2017/{a['file_name']}"))).convert("RGB")).astype(np.int64)
        seg = png[..., 0] + 256 * png[..., 1] + 65536 * png[..., 2]; sem = np.full(seg.shape, 255, np.uint8)
        for s in a["segments_info"]: sem[seg == s["id"]] = cat[s["category_id"]]
        Image.fromarray(sem).save(f"{out}/semantic/{split}/{a['file_name']}")
    return stop - start


if __name__ == "__main__":
    for split in ("val", "train"):
        d = json.loads(ann.read(f"annotations/panoptic_{split}2017.json")); cats = sorted(c["id"] for c in d["categories"])
        anns = sorted(d["annotations"], key=lambda a: a["file_name"])
        if split == "train": anns = anns[:: len(anns) // n_train][:n_train]       # evenly spaced, deterministic subset
        json.dump(dict(cat_ids=cats, annotations=anns), open(f"{out}/panoptic_{split}.json", "w"))
        if split == "val": json.dump([dict(index=i, id=c["id"], name=c["name"], isthing=c["isthing"]) for i, c in enumerate(sorted(d["categories"], key=lambda c: c["id"]))],
                                     open(f"{out}/categories.json", "w"), indent=1)
        os.makedirs(f"{out}/semantic/{split}", exist_ok=True); os.makedirs(f"{out}/images/{split}", exist_ok=True)
        zi = zipfile.ZipFile(f"{pub}/{split}2017.zip")
        for a in anns:
            name = a["file_name"].replace(".png", ".jpg"); dst = f"{out}/images/{split}/{name}"
            if not os.path.exists(dst): open(dst, "wb").write(zi.read(f"{split}2017/{name}"))
        chunks = [(split, i, min(i + 250, len(anns))) for i in range(0, len(anns), 250)]
        with ProcessPoolExecutor(8) as ex: print(split, sum(ex.map(convert, chunks)), "label maps", flush=True)
    print("PREP_COCO_DONE")
