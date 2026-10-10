"""Independent post-seal binary, physical-measure and signed-field audit.

No encoder, raw feature fitting, new CRF, parameter selection or output mask.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
RUN = REPO.parent / "cv_data/a/reference_focus_head200_20261010"
sys.path.insert(0, str(OUT.parent / "evidence_validation"))
from audit_saved_outputs import point_and_intervals
from audit_scene600 import weighted_rank

FIELDS = ("actual.global", "actual.local4", "actual.equal", "derived.global", "derived.local4", "derived.equal")
ARMS = tuple(a + "." + s for a in FIELDS for s in ("direct", "crf"))
CONTROLS = ("foris.crf", "mean", "region.fast")
HASHES, CHECKS = {}, Counter()


def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    path = Path(path).resolve()
    if str(path) not in HASHES:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
        HASHES[str(path)] = h.hexdigest()
    return HASHES[str(path)]


def check_sha(path, digest):
    assert sha(path) == digest, (str(path), "SHA mismatch")
    CHECKS["file_SHA"] += 1


def array_sha(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(json.dumps([list(x.shape), x.dtype.str]).encode()); h.update(x.tobytes())
    return h.hexdigest()


def canonical(x):
    ys = np.arange(1024, dtype=np.int64)*x.shape[0]//1024
    xs = np.arange(1024, dtype=np.int64)*x.shape[1]//1024
    return x[ys[:, None], xs[None, :]]


def mask(row, role):
    with Image.open(row[role+"_mask_path"]) as im:
        x = (np.asarray(im.convert("L")) > 0).astype(np.uint8)
    assert array_sha(x) == row[role+"_mask_hash"]
    CHECKS[role+"_GT_identity"] += 1
    return x.astype(bool)


def mass128(x): return x.reshape(128, 8, 128, 8).sum((1, 3), dtype=np.float64).reshape(-1)
def iu(p, g): return [int((p & g).sum()), int((p | g).sum())]
def edits(p, b, g): return [int((p & ~b & g).sum()), int((p & ~b & ~g).sum()), int((~p & b & g).sum()), int((~p & b & ~g).sum())]


def unpack(z, key, hw):
    value = z[key]
    assert value.dtype == np.uint8 and value.ndim == 1 and len(value) == (int(np.prod(hw))+7)//8
    CHECKS["packed_geometry"] += 1
    return np.unpackbits(value, count=int(np.prod(hw))).reshape(hw).astype(bool)


def identities():
    assert (RUN/"sealed.json").exists(), "No query GT/result access before full seal"
    cfg, seal = read(RUN/"config.json"), read(RUN/"sealed.json")
    assert seal["state"] == "ALL_PREDICTIONS_SEALED" and cfg["n"] == seal["n"] == 200
    for name, key in (("config.json", "config_sha256"), ("manifest.json", "manifest_sha256"),
                      ("tasks.json", "tasks_sha256"), ("inference.jsonl", "inference_sha256")):
        check_sha(RUN/name, seal[key])
    for name, key in (("producer_COMPLETE.json", "producer_complete_sha256"),
                      ("producer_input_bindings.json", "producer_bindings_sha256"),
                      ("selection_audit.json", "selection_audit_sha256"),
                      ("profile.json", "profile_sha256"), ("positional_basis.pt", "basis_sha256")):
        check_sha(RUN/name, cfg[key])
    for relative, digest in cfg["source_sha256"].items(): check_sha(RUN/"frozen"/relative, digest)
    for path, digest in cfg["external_source_sha256"].items(): check_sha(path, digest)
    check_sha(Path(cfg["source_pilot_root"])/"sealed.json", cfg["source_pilot_seal_sha256"])
    manifest, tasks, recs = read(RUN/"manifest.json"), read(RUN/"tasks.json"), lines(RUN/"inference.jsonl")
    assert len(manifest) == len(tasks) == len(recs) == 200
    for row, task, rec in zip(manifest, tasks, recs):
        assert row["episode_id"] == task["episode_id"] == rec["episode_id"]
        assert all(rec[k] == 0 for k in ("encoder_constructions", "encoder_forward", "raw_cache_writes",
                   "query_GT_reads", "baseline_mask_reads", "denied_file_attempts"))
        check_sha(RUN/"predictions"/rec["filename"], rec["prediction_sha256"])
        check_sha(RUN/"fields"/rec["filename"], rec["fields_sha256"])
        check_sha(task["baseline"]["path"], task["baseline"]["sha256"])
        assert len(rec["raw_requests"]) == 7
        for request in rec["raw_requests"]:
            check_sha(request["entry_path"], request["entry_sha256"])
            entry = read(request["entry_path"])
            assert entry["file_sha256"] == request["payload_sha256"]
            assert entry["features"]["O/24"]["tensor_sha256"] == request["tensor_sha256"]
            CHECKS["raw_request_metadata_identity"] += 1
    producer = read(RUN/"producer_COMPLETE.json")
    assert producer["new_raw_inputs"] == 192 and producer["cache_hit_unique"] == 4
    assert producer["unique_focus_inputs"] == 196 and producer["duplicate_episode_views"] == 4
    assert producer["query_GT_reads"] == producer["query_RGB_actors_called"] == producer["scoring_calls"] == 0
    # Freeze the analysis implementation and helper identities before labels open.
    (OUT/"frozen").mkdir(exist_ok=True)
    shutil.copyfile(__file__, OUT/"frozen/audit.py")
    write(OUT/"ANALYSIS_FROZEN.json", dict(seal_sha256=sha(RUN/"sealed.json"),
        source_sha256=sha(__file__), helpers={str(OUT.parent/"evidence_validation"/f):sha(OUT.parent/"evidence_validation"/f)
        for f in ("audit_saved_outputs.py", "audit_scene600.py")}, query_GT_reads=0,
        rank_recipe="signed128 fields; exact1024 FG/BG masses whole/FoRIS_FG/FoRIS_BG plus actual.equal.crf added/deleted versusFoRIS; sameROI all6fields; exactties; no selector",
        physical_recipe="full Rmasknearest1024; whole area256 outsidecrop128inside; focus area32;128mass quantiles perrole; compare frozen diagnostics and savedcoverage"))
    return cfg, seal, manifest, tasks, recs, producer


def physical(row, task, rec):
    r = canonical(mask(row, "reference")).astype(np.uint8)
    focus = task["focus"]; x0, y0, x1, y1 = focus["box_xyxy"]
    assert array_sha(r) == focus["canonical_mask_uint8_array_sha256"]
    crop = r[y0:y1, x0:x1]
    assert array_sha(crop) == focus["focus_mask512_hash"]
    whole_c = r.reshape(64, 16, 64, 16).mean((1, 3)); focus_c = crop.reshape(64, 8, 64, 8).mean((1, 3))
    area = np.full((64, 64), 256.); area[y0//16:y1//16, x0//16:x1//16] = 128.
    c, a = np.r_[whole_c.reshape(-1), focus_c.reshape(-1)], np.r_[area.reshape(-1), np.full(4096, 32.)]
    fg, bg = a*c, a*(1-c)
    assert fg.sum() == r.sum() and bg.sum() == r.size-r.sum() and a.sum() == 1024**2
    diag = rec["diagnostics"]["physical_measure"]
    assert diag["pooled_foreground_pixel_mass"] == float(fg.sum()) and diag["pooled_background_pixel_mass"] == float(bg.sum())
    check_sha(focus["coverage64_path"], focus["coverage64_file_sha256"])
    saved = np.load(focus["coverage64_path"], allow_pickle=False)
    assert array_sha(saved) == focus["coverage64_tensor_sha256"] and np.array_equal(saved, focus_c)
    occurrence = []
    for m in (fg, bg):
        cum = np.cumsum(m, dtype=np.float64)
        occurrence.append(np.searchsorted(cum, (np.arange(128)+.5)*cum[-1]/128, side="left"))
    q = rec["diagnostics"]["quadrature"]
    assert occurrence[0].tolist() == q["foreground_occurrence_ids"] and occurrence[1].tolist() == q["background_occurrence_ids"]
    ids = np.unique(np.r_[occurrence[0], occurrence[1]])
    assert ids.tolist() == q["sample_ids"]
    for k, occ in (("foreground_occurrence_counts", occurrence[0]), ("background_occurrence_counts", occurrence[1])):
        assert np.bincount(occ, minlength=8192)[ids].tolist() == q[k]
    CHECKS["physical_mass_and_quadrature"] += 1
    return dict(foreground_mass=int(r.sum()), background_mass=int(r.size-r.sum()), selected_ids=len(ids),
                full_physical_fit_loss={k:rec["diagnostics"]["fits"][k]["full_physical_binary_role_loss"] for k in ("actual", "derived")})


def main():
    started = time.monotonic(); cfg, seal, manifest, tasks, recs, producer = identities()
    print("All200 seal/source/prediction/field/control identities passed; GT scoring starts", flush=True)
    records, mechanisms, groups = [], [], defaultdict(list)
    for row, task, rec in zip(manifest, tasks, recs):
        pdiag = physical(row, task, rec)
        original = mask(row, "query"); cli = canonical(original)
        with np.load(RUN/"fields"/rec["filename"], allow_pickle=False) as z:
            assert set(z.files) == set(FIELDS)
            fields = {k:z[k].copy() for k in FIELDS}
            for v in fields.values(): assert v.dtype == np.float32 and v.shape == (128, 128) and np.isfinite(v).all()
            CHECKS["field_shape_finite_FP32"] += 6
        with np.load(RUN/"predictions"/rec["filename"], allow_pickle=False) as new, np.load(task["baseline"]["path"], allow_pickle=False) as old:
            assert tuple(new["original_hw"]) == original.shape
            assert set(new.files) == {"original_hw"} | {f+"/"+a for f in ("cli1024", "original") for a in ARMS}
            for frame, gt in (("cli1024", cli), ("original", original)):
                masks = {a:unpack(old, frame+"/"+a, gt.shape) for a in CONTROLS}
                masks.update({a:unpack(new, frame+"/"+a, gt.shape) for a in ARMS})
                entry = dict(episode_id=row["episode_id"], dataset=row["dataset"], fold=row["fold"], class_id=row["loader_class_id"],
                    query_photo_id=row["query_photo_id"], frame=frame, truth_pixels=int(gt.sum()),
                    iu={a:iu(p, gt) for a,p in masks.items()}, edits={})
                for a in ARMS:
                    entry["edits"][a] = {}
                    for b in ("foris.crf", "derived.equal.crf"):
                        e = edits(masks[a], masks[b], gt); entry["edits"][a][b] = e
                        i,u = entry["iu"][b]; assert entry["iu"][a] == [i+e[0]-e[2],u+e[1]-e[3]]
                        CHECKS["IU_edit_identity"] += 1
                CHECKS["candidate_IU"] += 12; CHECKS["baseline_IU"] += 3
                groups[row["dataset"],frame].append(entry); records.append(entry)
                if frame == "cli1024":
                    base, pred = masks["foris.crf"], masks["actual.equal.crf"]
                    rois = dict(whole=np.ones_like(gt), FoRIS_FG=base, FoRIS_BG=~base,
                                actual_added=pred & ~base, actual_deleted=~pred & base)
                    rank = {}
                    for name, roi in rois.items():
                        pos, neg = mass128(roi & gt), mass128(roi & ~gt)
                        rank[name] = {a:weighted_rank(v, pos, neg) for a,v in fields.items()}
                    confusion = {a:dict(TP=int((p & gt).sum()), FP=int((p & ~gt).sum()), FN=int((~p & gt).sum())) for a,p in masks.items()}
                    mechanisms.append(dict(episode_id=row["episode_id"], dataset=row["dataset"], rank=rank, confusion=confusion, physical=pdiag))
        if len(mechanisms)%50 == 0: print(json.dumps(dict(completed=len(mechanisms),n=200,seconds=time.monotonic()-started)),flush=True)
    results={}
    pairs=tuple(("actual.equal.crf",b) for b in CONTROLS+("derived.equal.crf",))
    for (ds,frame), rr in groups.items():
        results[ds+"/"+frame]=point_and_intervals(rr,cfg["paco_observed_classes"] if ds=="paco_part" else None,pairs)
        if ds=="paco_part": results[ds+"/"+frame+"/fixed303"]=point_and_intervals(rr,cfg["paco_expected_classes"],pairs)
    print(json.dumps({k:v["miou"] for k,v in results.items()}),flush=True)
    if (RUN/"scored_episodes.jsonl").exists():
        prior={(r["episode_id"],r["frame"]):r for r in lines(RUN/"scored_episodes.jsonl")}
        assert len(prior)==len(records)==400
        for r in records:
            assert r==prior[r["episode_id"],r["frame"]]
            CHECKS["parent_episode_ledger_identity"]+=1
    if (RUN/"report.json").exists():
        prior=read(RUN/"report.json")["results"]
        for key,value in results.items():
            parts=key.split('/'); old=prior[parts[0]][parts[1]]
            if len(parts)==3: old=old["fixed303_slots"]
            assert max(abs(value["miou"][a]-old["miou"][a]) for a in value["miou"])<1e-10
            for b in CONTROLS+("derived.equal.crf",):
                own=value["paired"]["actual.equal.crf - "+b]; recorded=old["paired"][b]["actual.equal.crf"]
                assert np.max(np.abs(np.array(own["ci95_pp"])-recorded["ci95_pp"]))<1e-10
                CHECKS["parent_point_and_paired_CI"]+=1
    ranking,confusion,fit={}, {}, {}
    for ds in ("deepglobe_road","paco_part"):
        mm=[r for r in mechanisms if r["dataset"]==ds];ranking[ds]={};confusion[ds]={};fit[ds]={}
        for roi in mm[0]["rank"]:
            ranking[ds][roi]={}
            for a in FIELDS:
                rr=[r["rank"][roi][a] for r in mm if r["rank"][roi][a]["auc"] is not None]
                ranking[ds][roi][a]=dict(n=len(rr),auc=float(np.mean([r["auc"] for r in rr])) if rr else None,
                    ap=float(np.mean([r["ap"] for r in rr])) if rr else None)
        for a in mm[0]["confusion"]:confusion[ds][a]={k:sum(r["confusion"][a][k] for r in mm) for k in ("TP","FP","FN")}
        for a in ("actual","derived"):fit[ds][a]=float(np.mean([r["physical"]["full_physical_fit_loss"][a] for r in mm]))
    timing={k:dict(mean=float(np.mean([r[k] for r in recs])),sum=float(np.sum([r[k] for r in recs]))) for k in
        ("io_seconds","gate_seconds","head_seconds","rendering_seconds","serialization_seconds","total_seconds")}
    timing["six_CRFs"]={"mean":float(np.mean([sum(r["crf_seconds"].values()) for r in recs]))}
    output=dict(state="VERIFIED",n=200,checks=dict(CHECKS),results=results,ranking=ranking,CLI_confusion=confusion,
        mean_full_physical_source_fit_loss=fit,paired_six_head_CPU_timing=timing,producer_cost=producer,
        inference_wall_seconds=seal["inference_wall_seconds"],audit_seconds=time.monotonic()-started,
        input_SHA=HASHES,encoder_calls=0,new_prediction_masks=0,raw_feature_arrays_loaded=0,
        raw_source_scope="Entry/file/tensor identities compared to producer+runner receipts; raw feature payloads not reread or recomputed by this audit",
        limits="Exposed Deep100/PACO100; same correlated reference whole/focus annotation;128 rankings with exact1024 masses are not finalmIoU; image-group CIs conditional; sixhead cost not isolatedmethod/cold cost")
    write(OUT/"verification.json",output)
    (OUT/"scored_episodes.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in records))
    (OUT/"mechanism_episodes.jsonl").write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in mechanisms))
    print(json.dumps(dict(state="VERIFIED",checks=dict(CHECKS),seconds=output["audit_seconds"])),flush=True)


if __name__=="__main__":main()
