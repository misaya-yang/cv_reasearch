#!/usr/bin/env python3
"""Closed-ontology reference/task aliases, then saved-performance subsets.

This is an oracle audit, not algorithm inference. Catalog pickles contain both
roles' annotations; the frozen membership rule only uses each row's reference.
No encoder, raw feature payload, fitting, or new algorithm masks are involved.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, OrderedDict
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / "cv_data"
OFFICIAL = ASSETS / "third_party/foris_official"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(OFFICIAL)]
import numpy as np
from PIL import Image
import pycocotools.mask as coco_mask
import torch
import torch.nn.functional as F
import opus55_reference_mask_trace as base
from ics.official_data import array_hash, decoded_rgb
from utils.data import polygons_to_bitmask, downsample_mask

ATLAS = ROOT / "evidence/local/five_dataset_case_mechanisms_20261011/integer_episodes.jsonl"
BINDINGS = ASSETS / "a/consensus_reference_calibration6380_v1_20261011/source_bindings.json"
OUT = ROOT / "evidence/local/task_identifiability6200_20261012"
SPEC = ROOT / "docs/research/IDENTIFIABILITY_AND_METHOD_CONSTRAINTS_20261012.md"
DOMAINS = ("coco", "lvis", "paco_part", "pascal_part", "suim")
COUNTS = dict(zip(DOMAINS, (200, 1400, 600, 2000, 2000)))
torch.set_num_threads(2)


def read(path):
    return json.loads(Path(path).read_text())


def catalog_paths():
    return [ASSETS / "datasets/ics/LVIS/lvis_val.pkl",
        ASSETS / "datasets/ics/PACO-Part/paco/paco_part_train.pkl",
        ASSETS / "datasets/ics/PACO-Part/paco/paco_part_val.pkl",
        ASSETS / "datasets/ics/Pascal-Part/VOCdevkit/VOC2010/all_obj_part_to_image.json"]


class Catalog:
    def __init__(self):
        paths = catalog_paths()
        with paths[0].open("rb") as stream: self.lvis = pickle.load(stream)
        eligible = [c for c in self.lvis if len(self.lvis[c]) > 1]
        self.lvis_allowed = set(eligible[:(len(eligible)//10)*10])
        self.lvis_images = defaultdict(dict)
        for cid in self.lvis_allowed:
            for name, annotation in self.lvis[cid].items():
                self.lvis_images[name][cid] = annotation
        with paths[1].open("rb") as stream: train = pickle.load(stream)
        with paths[2].open("rb") as stream: self.paco = pickle.load(stream)
        train_ids = list(train["cid2img"])
        assert len(train_ids) == 448
        self.paco_allowed = {c for c in train_ids if c in self.paco["cid2img"] and
            len({next(iter(x)) for x in self.paco["cid2img"][c]}) > 1}
        del train
        pascal = read(paths[3])
        self.pascal_allowed = set()
        for family in ("animals", "indoor", "person", "vehicles"):
            for obj, info in pascal[family]["object"].items():
                for part, entries in info["part"].items():
                    if entries["train"] and entries["val"] and obj+"+"+part != "aeroplane+TAIL":
                        self.pascal_allowed.add(obj+"+"+part)
        self.cache = OrderedDict()

    @staticmethod
    def decode(segmentation, size):
        if isinstance(segmentation, list):
            return polygons_to_bitmask(segmentation, size[1], size[0]).astype(bool)
        if isinstance(segmentation, dict):
            seg = dict(segmentation)
            if isinstance(seg.get("counts"), str): seg["counts"] = seg["counts"].encode("ascii")
            return coco_mask.decode(seg).astype(bool)
        assert isinstance(segmentation, np.ndarray) and segmentation.ndim == 2
        return segmentation.astype(bool)

    @staticmethod
    def crop(mask, box):
        return mask if box is None else mask[box[1]:box[3], box[0]:box[2]]

    def masks(self, row, role):
        domain, image_path, box = row["dataset"], ASSETS / row[role+"_path"], row.get(role+"_crop")
        key = domain, str(image_path), tuple(box) if box is not None else None
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        with Image.open(image_path) as image: size = image.size
        masks, provenance, selectors = {}, {}, []
        if domain == "coco":
            name = row[role+"_photo_id"]
            path = ASSETS / "demo4_cache/data/COCO2014/annotations" / Path(name).with_suffix(".png")
            with Image.open(path) as image: labels = np.asarray(image)
            for value in np.unique(labels):
                if 1 <= int(value) <= 80:
                    masks[str(int(value)-1)] = self.crop(labels == value, box)
            provenance[str(path)] = base.sha(path)
        elif domain == "lvis":
            name = str(image_path.relative_to(ASSETS / "datasets/ics/LVIS/coco"))
            for cid, info in self.lvis_images[name].items():
                mask = np.zeros(size[::-1], bool)
                for annotation in info["annotations"]:
                    mask |= self.decode(annotation["segmentation"], size)
                masks[str(cid)] = self.crop(mask, box)
            provenance["lvis_catalog"] = str(catalog_paths()[0])
        elif domain == "paco_part":
            image_id = int(image_path.stem)
            annos = self.paco["img2anno"][image_id]
            for annotation in annos:
                b = annotation["obj_bbox"]
                proposed = [int(b[0]), int(b[1]), int(b[0]+b[2]), int(b[1]+b[3])]
                if proposed != box: continue
                selector = int(annotation["obj_ann_id"])
                if selector not in selectors: selectors.append(selector)
                cid = int(annotation["category_id"])
                if cid not in self.paco_allowed: continue
                mask = self.crop(self.decode(annotation["segmentation"], size), box)
                task = str(cid)+"@parent="+str(selector)
                masks[task] = masks.get(task, np.zeros_like(mask)) | mask
            provenance["paco_catalog"] = str(catalog_paths()[2])
        elif domain == "pascal_part":
            path = ASSETS / "datasets/ics/Pascal-Part/VOCdevkit/VOC2010/Annotations_Part_json_merged_part_classes" / (image_path.stem+".json")
            annotation = read(path)
            for selector, obj in enumerate(annotation["object"]):
                bbox = [int(obj["bndbox"][k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                if bbox != box: continue
                selectors.append(selector)
                by_part = defaultdict(list)
                for part in obj["parts"]: by_part[part["name"]].extend(part["mask"])
                for name, segments in by_part.items():
                    category = obj["name"]+"+"+name
                    if category not in self.pascal_allowed: continue
                    mask = np.zeros(size[::-1], bool)
                    for seg in segments: mask |= self.decode(seg, size)
                    masks[category+"@parent="+str(selector)] = self.crop(mask, box)
            provenance[str(path)] = base.sha(path)
        else:
            assert domain == "suim"
            for category in ("FV", "HD", "PF", "RI", "RO", "SR", "WR"):
                directory = ASSETS / "datasets/ics/SUIM/masks" / category
                paths = sorted(directory.glob(image_path.stem+".*"))
                assert len(paths) <= 1
                if not paths: continue
                with Image.open(paths[0]) as image: mask = np.asarray(image.convert("L")) >= 128
                masks[category] = self.crop(mask, box)
                provenance[str(paths[0])] = base.sha(paths[0])
        # Retain empty Q masks to expose the role of query-positive sampling.
        result = masks, provenance, selectors
        self.cache[key] = result
        while len(self.cache) > 8:
            self.cache.popitem(last=False)
        return result


def signatures(mask):
    h, w = mask.shape
    y, x = np.arange(1024)*h//1024, np.arange(1024)*w//1024
    dense = mask[y[:, None], x[None, :]]
    n = dense[::16, ::16]
    v = sum(dense[yy::16, xx::16].astype(np.uint8) for yy in (7, 8) for xx in (7, 8))
    d = v > 2
    if not d.any():
        if n.any(): d = n.copy()
        elif dense.any():
            d = np.zeros((64, 64), bool)
            center = torch.argwhere(torch.from_numpy(dense)).float().mean(0)/16
            cy, cx = center.int().tolist()
            d[cy, cx] = True
    counts = dense.reshape(64, 16, 64, 16).sum((1, 3), dtype=np.uint16)
    hard = hashlib.sha256(np.packbits(n).tobytes()+np.packbits(d).tobytes()).hexdigest()
    soft = hashlib.sha256(np.packbits(n).tobytes()+counts.tobytes()).hexdigest()
    return {"full": array_hash(mask.astype(np.uint8)), "hard_joint": hard, "soft_joint": soft}


def concept(task):
    return task.split("@parent=")[0]


def prepare(out):
    if (out / "protocol.json").exists():
        assert base.sha(__file__) == read(out / "protocol.json")["source_sha256"]
        return
    bindings = {b["index"]: b for b in read(BINDINGS)}
    atlas = [json.loads(line) for line in ATLAS.open()]
    manifest = [{"id": r["binding_index"], "row": bindings[r["binding_index"]]["row"]} for r in atlas]
    assert Counter(m["row"]["dataset"] for m in manifest) == COUNTS
    assert all((r["row"].get("input_origin") != "legacy_transfer_pack") for r in manifest)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, out / "source.py")
    shutil.copyfile(SPEC, out / "spec.frozen.md")
    base.write(out / "manifest.json", manifest)
    protocol = {"state": "FROZEN_NATIVE_ONTOLOGY_ORACLE_AUDIT", "n": 6200, "counts": COUNTS,
        "source_sha256": base.sha(__file__), "spec_sha256": base.sha(SPEC),
        "manifest_sha256": base.sha(out / "manifest.json"), "atlas_sha256": base.sha(ATLAS),
        "bindings_sha256": base.sha(BINDINGS),
        "catalog_sha256": {str(p): base.sha(p) for p in catalog_paths()},
        "loader_sha256": {str(OFFICIAL / "datasets" / (d+".py")): base.sha(OFFICIAL / "datasets" / (d+".py")) for d in DOMAINS},
        "primary": "exact complete reference Mask; all legal native categories and same-bbox parent choices",
        "signatures": ["full pixel Mask", "joint N64/D64", "joint N64/soft64 area counts"],
        "membership": "reference oracle annotation only; no prediction/error/score selection",
        "access_caveat": "Catalog pickle includes annotations of all photos; functional membership only examines current R. Do not call the catalog physically query-label-free.",
        "query_phase": "only candidates aliased under full/hard/soft signatures; all native Q masks and parent choices at exact fixed crop",
        "performance": "saved6200 FoRIS/MEAN/R0/N integers; per-domain/frame/class coverage",
        "new_encoder_calls": 0, "raw_payload_reads": 0, "new_algorithm_predictions": 0}
    base.write(out / "protocol.json", protocol)


def ref_audit(out):
    prepare(out)
    if (out / "reference_membership_seal.json").exists(): return
    start = time.perf_counter()
    catalog = Catalog()
    manifest = read(out / "manifest.json")
    records, counts, checks = [], defaultdict(Counter), Counter()
    seen = {}
    for item in manifest:
        row = item["row"]
        path = Path(row["reference_mask_path"])
        observed = base.read_mask(path)
        assert array_hash(observed.astype(np.uint8)) == row["reference_mask_hash"]
        key = row["dataset"], row["reference_rgb_hash"], row["reference_mask_hash"], json.dumps(row.get("reference_crop"))
        if key in seen:
            prior = seen[key]
            record = {**prior, "id": item["id"], "episode_id": row["episode_id"]}
        else:
            masks, provenance, selectors = catalog.masks(row, "reference")
            image = decoded_rgb(ASSETS / row["reference_path"], row.get("reference_crop"))
            assert array_hash(np.asarray(image)) == row["reference_rgb_hash"]
            target_signatures = signatures(observed)
            matches = {name: [] for name in target_signatures}
            target_candidates = []
            for task, mask in masks.items():
                if not mask.any(): continue
                if mask.shape != observed.shape: continue
                sig = signatures(mask)
                if concept(task) == str(row["global_class_id"]): target_candidates.append(task)
                for name in matches:
                    if sig[name] == target_signatures[name]: matches[name].append(task)
            intended = [task for task in matches["full"] if concept(task) == str(row["global_class_id"])]
            status = "UNRESOLVED_REPLAY" if not intended else "REFERENCE_TASK_UNIQUE" if len(matches["full"]) == 1 else "REFERENCE_TASK_ALIAS"
            record = {"id": item["id"], "dataset": row["dataset"], "episode_id": row["episode_id"],
                "status": status, "matches": matches, "n_native_nonempty_tasks": int(sum(m.any() for m in masks.values())),
                "reference_signatures": target_signatures, "reference_mask_sha256": base.sha(path),
                "provenance": provenance, "matching_bbox_parent_selectors": selectors,
                "target_category": str(row["global_class_id"]), "intended_replayed_tasks": intended,
                "target_candidates": target_candidates}
            seen[key] = record
        counts[row["dataset"]][record["status"]] += 1
        counts[row["dataset"]]["hard_joint_reference_alias"] += len(record["matches"]["hard_joint"]) > 1
        counts[row["dataset"]]["soft_joint_reference_alias"] += len(record["matches"]["soft_joint"]) > 1
        records.append(record)
        checks["reference_mask_identity"] += 1
        if len(records) % 100 == 0:
            base.write(out / "activity.json", {"state": "REFERENCE_ORACLE", "completed": len(records), "total": 6200,
                "seconds": round(time.perf_counter()-start, 3)})
            print(json.dumps({"reference_members": len(records), "seconds": round(time.perf_counter()-start, 3)}), flush=True)
    base.write(out / "reference_membership.json", records)
    base.write(out / "reference_membership_seal.json", {"state": "REFERENCE_MEMBERSHIP_FIXED_BEFORE_QUERY_DIFFERENCE_OR_PERFORMANCE",
        "n": 6200, "unique_reference_tasks": len(seen), "counts": {d: dict(c) for d, c in counts.items()},
        "membership_sha256": base.sha(out / "reference_membership.json"),
        "protocol_sha256": base.sha(out / "protocol.json"), "seconds": time.perf_counter()-start,
        "checks": dict(checks), "new_encoder_calls": 0, "raw_payload_reads": 0})
    print(json.dumps({d: dict(c) for d, c in counts.items()}), flush=True)


def query_audit(out):
    seal = read(out / "reference_membership_seal.json")
    assert base.sha(out / "reference_membership.json") == seal["membership_sha256"]
    start = time.perf_counter()
    manifest = {r["id"]: r for r in read(out / "manifest.json")}
    membership = read(out / "reference_membership.json")
    catalog = Catalog()
    records, counts = [], defaultdict(Counter)
    for member in membership:
        item = manifest[member["id"]]; row = item["row"]
        result = {"id": member["id"], "dataset": row["dataset"], "status": member["status"], "certificates": {}}
        need = member["status"] != "UNRESOLVED_REPLAY" and any(len(v) > 1 for v in member["matches"].values())
        # Check hidden parent selection even when the reference category is unique.
        if row["dataset"] in ("pascal_part", "paco_part"): need = True
        if need:
            masks, provenance, selectors = catalog.masks(row, "query")
            image = decoded_rgb(ASSETS / row["query_path"], row.get("query_crop"))
            assert array_hash(np.asarray(image)) == row["query_rgb_hash"]
            expected = base.read_mask(row["query_mask_path"])
            assert array_hash(expected.astype(np.uint8)) == row["query_mask_hash"]
            intended = [mask for task, mask in masks.items() if concept(task) == str(row["global_class_id"]) and mask.shape == expected.shape and np.array_equal(mask, expected)]
            if not intended:
                result["status"] = "UNRESOLVED_QUERY_REPLAY"
            result["query_provenance"] = provenance
            result["query_matching_bbox_selectors"] = selectors
            for signature, candidates in member["matches"].items():
                wanted = {concept(t) for t in candidates}
                alternatives = [(task, mask) for task, mask in masks.items() if concept(task) in wanted and mask.shape == expected.shape]
                if not alternatives:
                    result["certificates"][signature] = {"status": "UNRESOLVED_NO_QUERY_ALTERNATIVE"}
                    continue
                masks_by_hash = defaultdict(list)
                payloads = {}
                for task, mask in alternatives:
                    digest = array_hash(mask.astype(np.uint8))
                    masks_by_hash[digest].append(task); payloads[digest] = mask
                nonempty = [h for h, m in payloads.items() if m.any()]
                certificate = {"status": "QUERY_CONSENSUS" if len(payloads) == 1 else "QUERY_LABEL_ALIAS",
                    "n_answers_open_query": len(payloads), "n_answers_query_positive": len(nonempty),
                    "tasks_by_mask_hash": dict(masks_by_hash),
                    "areas_by_hash": {h: int(m.sum()) for h, m in payloads.items()},
                    "same_observed_rgb": row["query_rgb_hash"], "same_crop": row.get("query_crop")}
                if len(payloads) > 1:
                    hashes = sorted(payloads)
                    first, second = hashes[0], hashes[1]
                    certificate["witness"] = {"first": first, "second": second,
                        "xor_pixels": int(np.count_nonzero(payloads[first]^payloads[second])),
                        "query_frame_pixels": int(expected.size)}
                result["certificates"][signature] = certificate
        else:
            result["certificates"] = {s: {"status": "REFERENCE_UNIQUE_IMPLIES_QUERY_TASK_UNIQUE"} for s in member["matches"]}
        for signature, certificate in result["certificates"].items():
            counts[row["dataset"]][signature+"/"+certificate["status"]] += 1
            if certificate.get("n_answers_query_positive", 0) > 1:
                counts[row["dataset"]][signature+"/positive_query_alias"] += 1
        records.append(result)
    base.write(out / "query_certificates.json", records)
    base.write(out / "oracle_membership_seal.json", {"state": "QUERY_ALIAS_CERTIFICATES_FIXED_BEFORE_SAVED_PERFORMANCE",
        "n": 6200, "reference_seal_sha256": base.sha(out / "reference_membership_seal.json"),
        "query_certificates_sha256": base.sha(out / "query_certificates.json"),
        "counts": {d: dict(c) for d, c in counts.items()}, "seconds": time.perf_counter()-start,
        "new_encoder_calls": 0, "raw_payload_reads": 0, "new_algorithm_predictions": 0})
    print(json.dumps({d: dict(c) for d, c in counts.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "ref", "query", "all"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args.out)
    elif args.mode == "ref": ref_audit(args.out)
    elif args.mode == "query": query_audit(args.out)
    else: ref_audit(args.out); query_audit(args.out)


if __name__ == "__main__": main()
