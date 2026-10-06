#!/usr/bin/env python3
"""One fixed GT-privileged object-crop CLS diagnostic; no legal proposals.

prepare: CPU only, freeze the existing197 region IDs and crop geometry.
infer: one frozen FP32 encoder, actual final normalized CLS,367 forwards.
score: only after every compact descriptor has been saved and sealed.
The primary comparison retains v2's18 paired episodes and71 regions. Other
existing regions are retained and reported only in a labeled supplement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
MEAN_RGB = tuple(int(round(v * 255)) for v in (0.485, 0.456, 0.406))
CONTROL_CUES = ("raw_mean_prototype", "projected_mean_prototype")
CUES = ("object_cls", *CONTROL_CUES)
CONFIG = dict(
    question="Can actual object-focused global CLS separate hard semantic errors where patch-region means fail?",
    cohort="full197 fixed eligible regions from exposed DEV241 region_prototypes_raw241_v2, no additional GT selection",
    primary="same18 paired episodes,40 missed GT plus31 stray fine64 regions",
    supplement="remaining fixed126 regions retained; all197-region pooled statistics are secondary only",
    dataset="COCO-20i, seed0, exposed development; not independent confirmation",
    resolution=1024, fixed_inference_batch=4,
    precision="FP32 weights/input/activations, no autocast, TF32 disabled",
    descriptor="actual final normalized prefix CLS token index0: encoder.m.forward_features(x)[:,0,:]",
    interface="installed timm.models.eva.Eva, CLS followed by4 register tokens; bypass default avg pooling",
    reference_positive="all supplied reference FG bounding box; square crop plus10percent context per edge; erase outside FG to ImageNet mean",
    reference_negative="original supplied reference image, erase supplied FG to ImageNet mean; square padding, no additional context crop",
    query="unmasked original RGB square crop around fixed GT/prediction region bbox plus10percent context per edge",
    aspect="map working1024 bbox back to original RGB coordinates before forming square; no clipping of square extent, pad out-of-image area",
    context_per_edge=0.1, neutral_RGB_uint8=MEAN_RGB,
    neutral_precision="ImageNet RGB mean rounded to nearest uint8 (124,116,104) for the unchanged PIL host transform",
    transform="existing utils.data.build_transform(1024): PIL Resize(1024,1024), ToTensor, ImageNet Normalize",
    margin="cosine(qCLS,rFG_CLS)-cosine(qCLS,rBG_CLS), higher is target FG",
    fixed_threshold=0.0, fitting=False, sign_flip=False, sweeps=False, other_augmentations=False,
    bootstrap=dict(draws=2000, rng="RandomState(0)", unit="connected reference/query photographs from source241", paired=True),
    privilege="query GT defines diagnostic region boxes before inference; this is not a deployable proposal generator or complete segmentation method",
    positive_consequence="only justifies seeking a legal proposal producer; does not establish SOTA or complete-mask improvement",
    negative_consequence="stop this fixed crop/global-CLS construction; not all global representations",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write(path, value):
    p = Path(path)
    temporary = p.with_suffix(p.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(p)


def unpack(value):
    if value.dtype != np.uint8 or value.shape != (131072,):
        raise ValueError("Require packed1024 source mask")
    return np.unpackbits(value).reshape(1024, 1024).astype(bool)


def bbox(mask):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        raise ValueError("Cannot crop empty supplied reference FG")
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


def square_box(box, original_size, working=False, context=0.1):
    """Exclusive integer box, preserving original aspect and fixed context."""
    width, height = original_size
    x0, y0, x1, y1 = box
    if working:
        x0, x1, y0, y1 = x0 * width / 1024, x1 * width / 1024, y0 * height / 1024, y1 * height / 1024
    side = max(1, int(math.ceil(max(x1 - x0, y1 - y0) * (1 + 2 * context))))
    left, top = math.floor((x0 + x1 - side) / 2), math.floor((y0 + y1 - side) / 2)
    return [left, top, left + side, top + side]


def square_crop(image, box):
    left, top, right, bottom = map(int, box)
    if right - left != bottom - top or right <= left:
        raise ValueError("Require fixed square crop")
    canvas = Image.new("RGB", (right - left, bottom - top), MEAN_RGB)
    x0, y0, x1, y1 = max(left, 0), max(top, 0), min(right, image.width), min(bottom, image.height)
    if x1 > x0 and y1 > y0:
        canvas.paste(image.crop((x0, y0, x1, y1)), (x0 - left, y0 - top))
    return canvas


def erase(image, foreground, erase_foreground):
    values = np.array(image, np.uint8, copy=True)
    if foreground.shape != (image.height, image.width):
        raise ValueError("Reference image/mask geometry differs")
    values[foreground if erase_foreground else ~foreground] = MEAN_RGB
    return Image.fromarray(values)


def prepare(a):
    if a.out.exists():
        raise FileExistsError("Fresh owned output required; never overwrite a prepared or failed run")
    source = a.source.resolve()
    state = json.loads((source / "state.json").read_text())
    prior = json.loads((source / "report.json").read_text())
    receipt = json.loads((source / "receipt.json").read_text())
    rows = [json.loads(l) for l in (source / "episodes.jsonl").read_text().splitlines() if l.strip()]
    if len(rows) != 241 or state["state"] != "CPU_GT_DIAGNOSTIC_COMPLETE" or sha(source / "report.json") != state["report_sha256"]:
        raise ValueError("Require unchanged full completed region-prototype v2 evidence")
    if prior["cues"][CONTROL_CUES[0]]["episode_auc"]["eligible_episodes"] != 18:
        raise ValueError("Require the fixed original18 paired diagnostic episodes")
    if any(r["old"] != j["old"] or r["new"] != j["new"] for r, j in zip(rows, receipt["inputs"])):
        raise ValueError("Source diagnostic and receipt identity mismatch")
    if len(receipt["inputs"]) != 241:
        raise ValueError("Missing source receipts")
    host = json.loads(a.host_manifest.read_text())
    data, annotations = Path(host["data_root"]), Path(host["annotation_root"])
    fine = Path(a.fine)
    fine_seal = json.loads((fine / "sealed.json").read_text())
    if sha(fine / "sealed.json") != receipt["source_metadata_sha256"][str(fine.resolve() / "sealed.json")]:
        raise ValueError("Changed fine source seal")
    cases, labels = [], []
    for row, source_input in zip(rows, receipt["inputs"]):
        rr = [r for r in row["diagnostic"]["regions"] if r["eligible"]]
        if not rr:
            continue
        packet, pred = source_input["paths"]["packet"], source_input["paths"]["prediction"]
        if sha(packet) != source_input["sha256"]["packet"] or sha(pred) != source_input["sha256"]["prediction"]:
            raise ValueError("Changed source mask/packet " + row["old"]["key"])
        with np.load(packet, allow_pickle=False) as z:
            truth_packed, cov = z["truth"].copy(), z["cov"].copy()
            truth = unpack(truth_packed)
        if array_sha(truth_packed) != source_input["truth_packed_sha256"]:
            raise ValueError("Source truth hash differs")
        with np.load(pred, allow_pickle=False) as z:
            prediction = unpack(z["fine.rcg64"])
        gl, _ = ndimage.label(truth, np.ones((3, 3), bool))
        pl, _ = ndimage.label(prediction, np.ones((3, 3), bool))
        objects = {"whole_missed_GT": ndimage.find_objects(gl), "stray_fine64": ndimage.find_objects(pl)}
        maps = {"whole_missed_GT": gl, "stray_fine64": pl}
        query, reference = data / row["old"]["query"], data / row["old"]["support"]
        reference_mask = annotations / Path(row["old"]["support"]).with_suffix(".png")
        with Image.open(query) as im:
            query_size = im.size
        with Image.open(reference) as im:
            reference_size = im.size
        with Image.open(reference_mask) as im:
            fg = np.array(im) == row["old"]["c"] + 1
        if fg.shape != reference_size[::-1]:
            raise ValueError("Supplied reference RGB/mask geometry differs")
        # Exact equivalent of host reference-mask nearest resize, no torch/model import.
        yy = (np.arange(1024) * fg.shape[0] / 1024).astype(int)
        xx = (np.arange(1024) * fg.shape[1] / 1024).astype(int)
        work_fg = fg[np.ix_(yy, xx)]
        if not np.array_equal(work_fg.reshape(64, 16, 64, 16).mean((1, 3)).astype(np.float32), cov):
            raise ValueError("Supplied reference mask does not reproduce sealed coverage")
        image_hashes = dict(query=sha(query), reference=sha(reference), reference_mask=sha(reference_mask))
        if image_hashes["query"] != fine_seal["inputs"][row["new"]["key"]]["query_image_sha256"]:
            raise ValueError("Query RGB differs from sealed source")
        case = dict(key=row["old"]["key"], c=row["old"]["c"], fold=row["old"]["fold"],
                    query=str(query), reference=str(reference), reference_mask=str(reference_mask),
                    query_size=query_size, reference_size=reference_size, image_sha256=image_hashes,
                    reference_FG_array_sha256=array_sha(fg),
                    reference_fg_crop=square_box(bbox(fg), reference_size),
                    reference_bg_crop=square_box([0, 0, *reference_size], reference_size, context=0), regions=[])
        paired = row["diagnostic"]["cues"][CONTROL_CUES[0]]["auc"] is not None
        for slot, r in enumerate(rr):
            y, x = objects[r["role"]][r["region"] - 1]
            work_box = [x.start, y.start, x.stop, y.stop]
            region_mask = maps[r["role"]] == r["region"]
            if int(region_mask.sum()) != r["area"]:
                raise ValueError("Source connected region ID/area mismatch")
            if r["role"] == "whole_missed_GT" and bool((region_mask & prediction).any()):
                raise ValueError("Previously missed GT region now overlaps prediction")
            if r["role"] == "stray_fine64" and bool((region_mask & truth).any()):
                raise ValueError("Previously stray region now overlaps GT")
            crop_id = f"{case['key']}--{slot}"
            case["regions"].append(dict(crop_id=crop_id, working_bbox=work_box,
                                       crop=square_box(work_box, query_size, working=True)))
            labels.append(dict(crop_id=crop_id, episode=case["key"], original_region_id=r["region"],
                               role=r["role"], area=r["area"], pure_tokens=r["pure_tokens"],
                               primary_paired_episode=paired, controls={cue: r["scores"][cue] for cue in CONTROL_CUES}))
        cases.append(case)
    counts = {role: sum(r["role"] == role for r in labels) for role in ("whole_missed_GT", "stray_fine64")}
    primary = [r for r in labels if r["primary_paired_episode"]]
    if len(cases) != 85 or counts != dict(whole_missed_GT=114, stray_fine64=83):
        raise ValueError("Changed197-region diagnostic cohort")
    if len({r["episode"] for r in primary}) != 18 or sum(r["role"] == "whole_missed_GT" for r in primary) != 40 or len(primary) != 71:
        raise ValueError("Changed18-episode primary71-region cohort")
    dependencies = dict(wrapper=REPO / "src/ics/data.py", model_config=a.model_dir / "config.json",
                        model_weights=a.model_dir / "model.safetensors", timm_eva=a.timm_eva,
                        host_transform=Path(host["foris_root"]) / "utils/data.py")
    config = dict(CONFIG, model_dir=str(a.model_dir.resolve()), host_root=host["foris_root"],
                  dependency_sha256={name: dict(path=str(p.resolve()), sha256=sha(p)) for name, p in dependencies.items()},
                  source_metadata_sha256={str(source / n): sha(source / n) for n in ("report.json", "receipt.json", "episodes.jsonl", "state.json")},
                  source_code_sha256=sha(__file__), source_episodes=241, encoded_episodes=85, query_regions=197,
                  expected_descriptor_forwards=367, primary_paired_episodes=18, primary_regions=71,
                  descriptor_storage_estimate_bytes=367 * 1024 * 4, no_encoder_or_GPU_execution_in_prepare=True)
    a.out.mkdir(parents=True)
    write(a.out / "config.json", config)
    write(a.out / "crops.json", cases)
    write(a.out / "labels.json", dict(regions=labels, source_rows=[r["old"] for r in rows],
                                     original_control_episode_auc=prior["cues"], original_paired_keys=sorted({r["episode"] for r in primary})))
    write(a.out / "prepared.json", dict(state="CPU_CROPS_PREPARED_NO_ENCODER_RUN", counts=counts,
        encoded_episodes=85, query_regions=197, expected_descriptor_forwards=367, primary_paired_episodes=18,
        config_sha256=sha(a.out / "config.json"), crops_sha256=sha(a.out / "crops.json"), labels_sha256=sha(a.out / "labels.json"),
        query_GT_used_for_geometry=True, source_code_sha256=sha(__file__)))
    print(json.dumps(json.loads((a.out / "prepared.json").read_text())), flush=True)


def prepared(a):
    seal = json.loads((a.out / "prepared.json").read_text())
    if seal["state"] != "CPU_CROPS_PREPARED_NO_ENCODER_RUN" or sha(a.out / "config.json") != seal["config_sha256"] or sha(a.out / "crops.json") != seal["crops_sha256"]:
        raise ValueError("Changed CPU preparation")
    return seal, json.loads((a.out / "config.json").read_text()), json.loads((a.out / "crops.json").read_text())


def infer(a):
    seal, config, cases = prepared(a)
    if (a.out / "descriptors").exists() or (a.out / "sealed.json").exists():
        raise FileExistsError("Never replace complete or partial descriptor output")
    if sha(__file__) != config["source_code_sha256"]:
        raise ValueError("Prepared implementation changed")
    for name, dep in config["dependency_sha256"].items():
        if sha(dep["path"]) != dep["sha256"]:
            raise ValueError("Changed prepared dependency " + name)
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Root must launch only on the existing free GPU")
    sys.path.insert(0, str(REPO / "src"))
    from ics.data import TimmDINOv3
    sys.path.insert(0, config["host_root"])
    from utils.data import build_transform
    transform = build_transform(1024)
    encoder = TimmDINOv3(config["model_dir"]).to("cuda").float().eval().requires_grad_(False)
    model = encoder.m
    if type(model).__module__ != "timm.models.eva" or type(model).__name__ != "Eva" or model.cls_token is None or model.num_prefix_tokens != 5:
        raise ValueError("Prepared actual normalized-CLS interface is unavailable")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Require FP32 model parameters")
    (a.out / "descriptors").mkdir()
    begin, forwards, encoder_calls, reused_images, hashes = time.monotonic(), 0, 0, 0, {}
    expected_forwards = int(config["expected_descriptor_forwards"])
    expected_total = sum(2 + len(case["regions"]) for case in cases)
    state = dict(state="GPU_CLS_DESCRIPTOR_INFERENCE", completed=0, n=len(cases), pid=os.getpid(),
                 descriptor_forwards=0, GT_boxes_privileged=True, labels_file_opened=False)
    write(a.out / "state.json", state)
    def encode(images):
        nonlocal forwards, encoder_calls
        x = torch.stack([transform(image) for image in images]).to(device="cuda", dtype=torch.float32)
        if x.shape != (len(images), 3, 1024, 1024) or not 1 <= len(images) <= 4:
            raise ValueError("Wrong host transform geometry")
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            tokens = model.forward_features(x)
            if tokens.shape != (len(images), 4101, 1024) or tokens.dtype != torch.float32:
                raise ValueError("Actual final normalized token interface does not match")
            cls_tensor = tokens[:, 0]
            norm_error = float((F.normalize(cls_tensor, dim=-1).norm(dim=-1)-1).abs().max())
            cls = cls_tensor.detach().cpu().numpy().copy()
        if not np.isfinite(cls).all() or np.any(np.linalg.norm(cls, axis=-1) <= 1e-12) or norm_error > 1e-6:
            raise ValueError("Invalid actual CLS descriptor")
        forwards += len(images)
        encoder_calls += 1
        if encoder_calls == 1:
            write(a.out / "model_setup_smoke.json", dict(state="ACTUAL_CLS_INTERFACE_SMOKE_OK",
                tokens_shape=[len(images),4101,1024], normalized_CLS_index=0, prefix_tokens=5,
                default_global_pool=model.global_pool, bypassed_pooling=True,
                dtype="float32", no_autocast=True, L2_normalization_max_error=norm_error,
                cuda_peak_bytes=int(torch.cuda.max_memory_allocated()), first_batch_descriptors_retained=True))
        return cls
    for n, case in enumerate(cases, 1):
        for role in ("query", "reference", "reference_mask"):
            if sha(case[role]) != case["image_sha256"][role]:
                raise ValueError("Changed source image/mask " + case["key"])
        if case.get("reuse_descriptor"):
            if sha(case["reuse_descriptor"]) != case["reuse_descriptor_sha256"]:
                raise ValueError("Changed source reusable CLS descriptor")
            with np.load(case["reuse_descriptor"], allow_pickle=False) as z:
                if z["crop_ids"].tolist() != case["reuse_source_crop_ids"]:
                    raise ValueError("Reusable source crop identities differ")
                rfg, rbg = z["r_fg"].copy(), z["r_bg"].copy()
                q = z["q"][case["reuse_query_indices"]].copy()
            if any(v.dtype != np.float32 for v in (q, rfg, rbg)) or q.shape != (len(case["regions"]),1024):
                raise ValueError("Reusable CLS precision/geometry differs")
            reused_images += 2 + len(case["regions"])
            path = a.out / "descriptors" / (case["key"] + ".npz")
            np.savez_compressed(path, r_fg=rfg, r_bg=rbg, q=q,
                                crop_ids=np.array([r["crop_id"] for r in case["regions"]]))
            hashes[case["key"]] = sha(path)
            state.update(completed=n, descriptor_forwards=forwards, encoder_calls=encoder_calls,
                         reused_descriptor_images=reused_images, seconds=time.monotonic()-begin)
            write(a.out / "state.json", state)
            continue
        with Image.open(case["reference"]) as im:
            reference = im.convert("RGB")
        with Image.open(case["reference_mask"]) as im:
            fg = np.array(im) == case["c"] + 1
        if array_sha(fg) != case["reference_FG_array_sha256"]:
            raise ValueError("Changed supplied reference mask")
        with Image.open(case["query"]) as im:
            query = im.convert("RGB")
        # One episode's views at a time, with fixed identity order; no pixel archive.
        views = [square_crop(erase(reference, fg, False), case["reference_fg_crop"]),
                 square_crop(erase(reference, fg, True), case["reference_bg_crop"])]
        views.extend(square_crop(query, r["crop"]) for r in case["regions"])
        descriptors = np.concatenate([encode(views[start:start+4]) for start in range(0,len(views),4)])
        rfg, rbg, q = descriptors[0], descriptors[1], descriptors[2:]
        path = a.out / "descriptors" / (case["key"] + ".npz")
        np.savez_compressed(path, r_fg=rfg, r_bg=rbg, q=q,
                            crop_ids=np.array([r["crop_id"] for r in case["regions"]]))
        hashes[case["key"]] = sha(path)
        state.update(completed=n, descriptor_forwards=forwards, encoder_calls=encoder_calls,
                     seconds=time.monotonic() - begin)
        write(a.out / "state.json", state)
        if n % 10 == 0 or n == len(cases):
            print(json.dumps(state), flush=True)
    if forwards != expected_forwards or forwards + reused_images != expected_total:
        raise ValueError("Wrong fixed descriptor-forward count")
    torch.cuda.synchronize()
    write(a.out / "sealed.json", dict(state="ALL_CLS_DESCRIPTORS_SEALED", n=len(cases),
        query_regions=sum(len(case["regions"]) for case in cases),
        descriptor_forwards=forwards, encoder_calls=encoder_calls, fixed_batch=4,
        reused_descriptor_images=reused_images, total_descriptor_images=expected_total,
        descriptors=hashes, config_sha256=seal["config_sha256"],
        crops_sha256=seal["crops_sha256"], prepared_sha256=sha(a.out / "prepared.json"),
        labels_opened_in_infer=False, GT_boxes_privileged=True, seconds=time.monotonic() - begin,
        torch_version=str(torch.__version__), cuda_peak_bytes=int(torch.cuda.max_memory_allocated()),
        model_device=str(torch.cuda.get_device_name(0))))
    state.update(state="ALL_CLS_DESCRIPTORS_SEALED")
    write(a.out / "state.json", state)


def unit(value):
    value = np.asarray(value, np.float32)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm <= 1e-12):
        raise ValueError("Invalid sealed CLS vector")
    return value / norm


def auc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    ranks = rankdata(np.r_[positive, negative], method="average")
    p, n = len(positive), len(negative)
    return float((ranks[:p].sum() - p * (p + 1) / 2) / (p * n))


def photo_groups(rows):
    parent, seen = list(range(len(rows))), {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, r in enumerate(rows):
        for role in ("support", "query"):
            photo = Path(r[role]).name
            if photo in seen:
                a, b = find(i), find(seen[photo])
                parent[max(a, b)] = min(a, b)
            seen[photo] = i
    mapping, groups = {}, []
    for i in range(len(rows)):
        root = find(i)
        mapping.setdefault(root, len(mapping))
        groups.append(mapping[root])
    return np.array(groups)


def statistics(rows, labels):
    groups = photo_groups(rows)
    g = int(groups.max()) + 1
    draws = np.random.RandomState(0).randint(g, size=(2000, g))
    weights = np.stack([np.bincount(d, minlength=g) for d in draws])
    def mean(values):
        values = np.array([np.nan if v is None else v for v in values], float)
        valid = np.isfinite(values)
        counts = np.bincount(groups[valid], minlength=g)
        totals = np.bincount(groups[valid], weights=values[valid], minlength=g)
        den = np.sum(weights * counts[None], axis=1)
        samples = np.divide(np.sum(weights * totals[None], axis=1), den, out=np.full(2000,np.nan), where=den>0)
        samples = samples[np.isfinite(samples)]
        return dict(mean=float(values[valid].mean()) if valid.any() else None,
                    ci95=np.percentile(samples,[2.5,97.5]).tolist() if len(samples) else None,
                    eligible_episodes=int(valid.sum()), finite_bootstrap_draws=int(len(samples)))
    by_episode = {r["key"]: [] for r in rows}
    for r in labels:
        by_episode[r["episode"]].append(r)
    result, values = dict(primary={}, paired_contrasts={}, folds={}, supplement={}), {}
    for cue in CUES:
        values[cue] = []
        for row in rows:
            rr = by_episode[row["key"]]
            p = [r["scores"][cue] for r in rr if r["role"] == "whole_missed_GT"]
            n = [r["scores"][cue] for r in rr if r["role"] == "stray_fine64"]
            values[cue].append(auc(p,n))
        result["primary"][cue] = mean(values[cue])
    for control in CONTROL_CUES:
        result["paired_contrasts"]["object_cls-minus-"+control] = mean(
            [x-y if x is not None and y is not None else None for x,y in zip(values["object_cls"],values[control])])
    result["primary"]["object_cls_minus_chance"] = mean([v-0.5 if v is not None else None for v in values["object_cls"]])
    for fold in range(4):
        ix = [i for i,r in enumerate(rows) if r["fold"] == fold and values["object_cls"][i] is not None]
        result["folds"][str(fold)] = dict(eligible_episodes=len(ix),
            episode_auc={c:float(np.mean([values[c][i] for i in ix])) if ix else None for c in CUES},
            CLS_minus_raw=float(np.mean([values["object_cls"][i]-values[CONTROL_CUES[0]][i] for i in ix])) if ix else None)
    # Only a supplement uses the197-region mixed-episode population.
    p = [r for r in labels if r["role"] == "whole_missed_GT"]
    n = [r for r in labels if r["role"] == "stray_fine64"]
    for cue in CUES:
        ps, ns = np.array([r["scores"][cue] for r in p]), np.array([r["scores"][cue] for r in n])
        wins = (ps[:,None]>ns[None,:])+0.5*(ps[:,None]==ns[None,:])
        mass = np.array([r["area"] for r in p],float)[:,None]*np.array([r["area"] for r in n],float)[None,:]
        result["supplement"][cue] = dict(region_pooled_auc=float(wins.mean()), pixel_mass_weighted_auc=float((wins*mass).sum()/mass.sum()),
            fixed_zero_margin_TP=int((ps>0).sum()), fixed_zero_margin_FP=int((ns>0).sum()), positive_regions=len(p), negative_regions=len(n))
    result.update(source_n=len(rows), source_photo_groups=g, encoded_episodes=len({r["episode"] for r in labels}),
                  supplemental_regions=len(labels), primary_paired_regions=sum(r["primary_paired_episode"] for r in labels))
    return result, draws, values


def score(a):
    prep, config, cases = prepared(a)
    seal = json.loads((a.out / "sealed.json").read_text())
    if seal["state"] != "ALL_CLS_DESCRIPTORS_SEALED" or seal["descriptor_forwards"] != 367 or sha(a.out / "prepared.json") != seal["prepared_sha256"]:
        raise ValueError("Require complete sealed197-region CLS inference")
    if sha(a.out / "labels.json") != prep["labels_sha256"]:
        raise ValueError("Changed fixed diagnostic labels")
    if (a.out / "report.json").exists():
        raise FileExistsError("Never overwrite scored evidence")
    labels = json.loads((a.out / "labels.json").read_text())
    margin = {}
    for case in cases:
        path = a.out / "descriptors" / (case["key"] + ".npz")
        if sha(path) != seal["descriptors"][case["key"]]:
            raise ValueError("Changed sealed descriptor")
        with np.load(path,allow_pickle=False) as z:
            if z["crop_ids"].tolist() != [r["crop_id"] for r in case["regions"]] or z["q"].shape != (len(case["regions"]),1024):
                raise ValueError("Descriptor IDs/geometry differ")
            if any(z[k].dtype != np.float32 for k in ("q","r_fg","r_bg")):
                raise ValueError("Stored CLS descriptor precision differs")
            scores = np.sum(unit(z["q"])*unit(z["r_fg"])[None],axis=1)-np.sum(unit(z["q"])*unit(z["r_bg"])[None],axis=1)
            margin.update(zip(z["crop_ids"].tolist(),map(float,scores)))
    records = labels["regions"]
    if set(margin) != {r["crop_id"] for r in records}:
        raise ValueError("Changed fixed197 region IDs")
    for r in records:
        r["scores"] = dict(r["controls"], object_cls=margin[r["crop_id"]])
    report, draws, values = statistics(labels["source_rows"], records)
    if report["primary"]["object_cls"]["eligible_episodes"] != 18:
        raise ValueError("Primary paired cohort changed")
    for cue in CONTROL_CUES:
        old = labels["original_control_episode_auc"][cue]["episode_auc"]
        new = report["primary"][cue]
        if abs(new["mean"]-old["mean"])>1e-14 or np.max(np.abs(np.array(new["ci95"])-old["ci95"]))>1e-14:
            raise ValueError("Exact fixed v2 region-mean controls do not reproduce")
    report.update(config=config,state="CPU_GT_CLS_DIAGNOSTIC_COMPLETE",descriptor_seal_sha256=sha(a.out / "sealed.json"),
                  source_code_sha256=sha(__file__),complete_method_result=False,positive_semantic_signal_unverified_as_legal_proposals=True)
    write(a.out / "report.json",report)
    write(a.out / "scored_regions.json",records)
    np.save(a.out / "bootstrap_photo_draws.npy",draws)
    np.savez_compressed(a.out / "episode_auc.npz",**{c:np.array([np.nan if x is None else x for x in v]) for c,v in values.items()})
    lines=["# Fixed object-crop CLS diagnostic: exposed DEV241", "", CONFIG["privilege"], "",
           "All197 fixed region descriptors retained; primary comparison is18 paired episodes/71 regions. No fitted weights, sign reversal or sweep.", "",
           "| cue | paired episodes | episode region AUROC [95% CI] |", "|---|---:|---:|"]
    for cue in CUES:
        s=report["primary"][cue]
        lines.append("| %s | %d | %.4f [%.4f,%.4f] |" % (cue,s["eligible_episodes"],s["mean"],*s["ci95"]))
    lines += ["", "Paired contrasts:", "",json.dumps(report["paired_contrasts"],indent=2),
              "", "Folds:", "",json.dumps(report["folds"],indent=2),
              "", "Supplement only (all197 regions across85 episodes):", "",json.dumps(report["supplement"],indent=2),
              "", CONFIG["positive_consequence"], "",CONFIG["negative_consequence"]]
    (a.out / "report.md").write_text("\n".join(lines)+"\n")
    write(a.out / "score_state.json",dict(state="CPU_GT_CLS_DIAGNOSTIC_COMPLETE",report_sha256=sha(a.out / "report.json")))
    print(json.dumps(dict(state=report["state"],primary=report["primary"],paired_contrasts=report["paired_contrasts"])),flush=True)


def self_test():
    assert MEAN_RGB==(124,116,104)
    assert square_box([0,0,1024,1024],(640,320),working=True,context=0)==[0,-160,640,480]
    assert square_box([10,20,30,40],(100,100))==[8,18,32,42]
    image=Image.new("RGB",(2,1),(255,0,0))
    crop=np.array(square_crop(image,[-1,-1,3,3]))
    assert crop.shape==(4,4,3) and tuple(crop[0,0])==MEAN_RGB and tuple(crop[1,1])==(255,0,0)
    foreground=np.array([[True,False]])
    assert tuple(np.array(erase(image,foreground,False))[0,0])==(255,0,0)
    assert tuple(np.array(erase(image,foreground,False))[0,1])==MEAN_RGB
    assert tuple(np.array(erase(image,foreground,True))[0,0])==MEAN_RGB
    assert tuple(np.array(erase(image,foreground,True))[0,1])==(255,0,0)
    assert auc([1],[0])==1 and auc([0],[1])==0 and auc([1],[1])==0.5
    rows=[dict(key="a",fold=0,c=0,support="p",query="q")]
    labels=[dict(episode="a",role=role,area=16,primary_paired_episode=True,
                 scores={cue:value for cue in CUES}) for role,value in (("whole_missed_GT",1),("stray_fine64",-1))]
    report,_,_=statistics(rows,labels)
    assert report["primary"]["object_cls"]["mean"]==1 and report["paired_contrasts"]["object_cls-minus-raw_mean_prototype"]["mean"]==0
    print("SELF_TEST_OK_NO_MODEL_OR_GPU",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","infer","score","self-test"))
    p.add_argument("--out",type=Path)
    p.add_argument("--source",type=Path)
    p.add_argument("--fine",type=Path)
    p.add_argument("--host-manifest",type=Path)
    p.add_argument("--model-dir",type=Path,default=Path("/root/demo4_cache/models/dinov3-vitl16-timm"))
    p.add_argument("--timm-eva",type=Path,default=Path("/root/demo4_cache/env/timm/models/eva.py"))
    a=p.parse_args()
    if a.stage=="self-test":
        self_test();return
    if not a.out:
        p.error("Require --out")
    if a.stage=="prepare" and not all((a.source,a.fine,a.host_manifest)):
        p.error("Prepare requires source/fine/host-manifest")
    {"prepare":prepare,"infer":infer,"score":score}[a.stage](a)


if __name__=="__main__":
    main()
