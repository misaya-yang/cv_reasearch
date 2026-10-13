"""Execute Astra's frozen 900-episode composition, without new encoders/fits."""
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import composition_core_interval_audit as core

DESIGN = REPO / "evidence/local/insid3_foris_generalization_20261010"
PROTOCOL = DESIGN / "composition_bg_admission_protocol.json"
INPUTS = DESIGN / "composition_inputs900.json"
DESCRIPTORS = DESIGN / "composition_descriptors900.json"
OUT = REPO.parent / "cv_data/a/bg_admission_composition900_v1_20261010"
ARMS = ("full_composition", "same_added_area_source_order")


@contextmanager
def deny_query_masks(rows, attempts):
    denied = {Path(s["row"]["query_mask_path"]).resolve() for s in rows}
    previous = builtins.open, io.open, Image.open
    def wrapped(fn):
        def opening(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve() in denied:
                attempts.append(str(path))
                raise RuntimeError("Query ground truth is forbidden during composition inference")
            return fn(path, *args, **kwargs)
        return opening
    builtins.open, io.open, Image.open = map(wrapped, previous)
    try:
        yield
    finally:
        builtins.open, io.open, Image.open = previous


def signed_readout(host, secondary, baseline):
    score = host + .5 * secondary
    return (score > 0) | ((score == 0) & baseline)


def same_k_control(old, baseline, score, k):
    possible = np.flatnonzero((old & ~baseline).ravel())
    assert 0 <= k <= len(possible)
    selected = np.zeros(old.size, dtype=bool)
    tie = {"cut_score": None, "cut_tie_pixels": 0, "cut_tie_selected": 0, "selection": "none"}
    if k == len(possible):
        selected[possible] = True
        tie["selection"] = "all_old_additions"
    elif k:
        values = score.ravel()[possible]
        order = np.argsort(-values, kind="stable")  # possible is already row-major
        selected[possible[order[:k]]] = True
        cutoff = float(values[order[k - 1]])
        greater, equal = int((values > cutoff).sum()), int((values == cutoff).sum())
        tie = {"cut_score": cutoff, "cut_tie_pixels": equal, "cut_tie_selected": k - greater,
               "selection": "descending_source_score_then_row_major"}
    result = (old & baseline) | selected.reshape(old.shape)
    assert int((result & ~baseline).sum()) == k
    assert np.array_equal(result & baseline, old & baseline)
    return result, tie


def infer():
    core.torch.set_num_threads(2)
    protocol, inputs, desc = [core.read(p) for p in (PROTOCOL, INPUTS, DESCRIPTORS)]
    assert protocol["root_role"].endswith("01a12763-7050-7311-9cca-8ad5754b1cf5.")
    assert inputs["protocol_sha256"] == desc["protocol_sha256"] == core.sha(PROTOCOL)
    assert desc["input_binding_sha256"] == core.sha(INPUTS)
    rows = inputs["rows"]
    assert len(rows) == len(desc["rows"]) == 900
    assert [r["row"]["episode_id"] for r in rows] == protocol["cohort"]["episode_order"]
    parent = core.read(DESIGN / "composition_parent900_verification.json")
    assert parent["state"] == "ALL900_PARENT_MASKS_REPLAYED_EXACTLY" and parent["manifest_sha256"] == core.sha(INPUTS)
    cfg = {"protocol_sha256": core.sha(PROTOCOL), "inputs_sha256": core.sha(INPUTS),
           "descriptors_sha256": core.sha(DESCRIPTORS), "parent_verification_sha256": core.sha(DESIGN / "composition_parent900_verification.json"),
           "source_sha256": {str(Path(__file__)): core.sha(__file__), str(Path(core.__file__)): core.sha(core.__file__)},
           "n": 900, "arms": list(ARMS), "primary_frame": "original",
           "recipe": protocol["complete_rule"], "control": protocol["new_arms"][ARMS[1]],
           "background_core": protocol["query_core"], "background_floor": protocol["background_floor"],
           "threshold": protocol["admission_threshold"], "no_additional_crf": True,
           "new_encoder_calls": 0, "new_raw_reads": 0, "new_reference_fits": 0,
           "exposure": protocol["cohort"]["exposure"],
           "source_foreground_median_or_condition_used_in_inference": False}
    if (OUT / "config.json").exists():
        assert core.read(OUT / "config.json") == cfg
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        for sub in ("predictions", "records", "frozen"):
            (OUT / sub).mkdir()
        core.write(OUT / "config.json", cfg)
        core.write(OUT / "manifest.json", rows)
        for path in cfg["source_sha256"]:
            (OUT / "frozen" / Path(path).name).write_bytes(Path(path).read_bytes())
    if (OUT / "sealed.json").exists():
        print("Existing all900 prediction seal; no repeated inference")
        return
    records, attempts, start = [], [], time.monotonic()
    with deny_query_masks(rows, attempts):
        for index, (source, expected) in enumerate(zip(rows, desc["rows"])):
            row = source["row"]
            eid, shape = row["episode_id"], tuple(row["query_size_hw"])
            rp = OUT / "records" / f"{index:06}.json"
            if rp.exists():
                r = core.read(rp)
                assert r["episode_id"] == eid and core.sha(OUT / "predictions" / r["file"]) == r["sha256"]
                records.append(r)
                continue
            tick = time.monotonic()
            for key in ("field", "old", "baseline", "source_score"):
                assert core.sha(source[key]) == source[key + "_sha256"]
            with np.load(source["field"], allow_pickle=False) as z:
                ridge = z[source["field_key"]].copy()
            with np.load(source["source_score"], allow_pickle=False) as z:
                s = z[source["score_key"]].reshape(64, 64).astype(float)
            with np.load(source["baseline"], allow_pickle=False) as z:
                b = core.unpack(z["original/" + source["baseline_key"]], shape)
                b_cli = core.unpack(z["cli1024/" + source["baseline_key"]], (1024, 1024))
            with np.load(source["old"], allow_pickle=False) as z:
                old = core.unpack(z["anchor.global"], shape)
            assert ridge.shape == (128, 128) and np.isfinite(ridge).all()
            span = float(s.max() - s.min())
            normalized = (s - s.min()) / max(span, 1e-6)
            confidence_cli = np.abs(2 * core.resize(normalized, (1024, 1024)) - 1) if span else np.zeros((1024, 1024))
            bg_core = (~b_cli) & (confidence_cli > .5)
            weights = bg_core.reshape(128, 8, 128, 8).sum((1, 3))
            floor = core.median(ridge, weights)
            assert floor == expected["core_median"]["background"] and int(weights.sum()) == expected["core_pixels"]["background"]
            threshold, scale = float(source["calibration"]["threshold"]), float(source["calibration"]["separation"])
            admitted = threshold if floor is None else max(threshold, floor)
            valid_scale = np.isfinite(scale) and scale > np.finfo(float).eps
            h_r = np.clip((ridge - threshold) / scale, -1, 1) if valid_scale else np.zeros_like(ridge)
            h_a = np.clip((ridge - admitted) / scale, -1, 1) if valid_scale else np.zeros_like(ridge)
            assert np.all(h_a <= h_r)
            confidence = np.abs(2 * core.resize(normalized, shape) - 1) if span else np.zeros(shape)
            host = (2 * b.astype(float) - 1) * confidence
            replay = signed_readout(host, core.resize(h_r, shape), b)
            assert np.array_equal(replay, old), (eid, "parent replay")
            assert np.array_equal(signed_readout(host, np.zeros(shape), b), b)
            m_a = signed_readout(host, core.resize(h_a, shape), b)
            full = replay & (b | m_a)
            assert not np.any(m_a & ~replay)
            assert np.array_equal(full & b, old & b)
            assert not np.any(full & ~old)
            trigger = floor is not None and threshold < floor
            if not trigger:
                assert np.array_equal(full, old)
            if not valid_scale:
                assert np.array_equal(full, b)
            k = int((full & ~b).sum())
            control, ties = same_k_control(old, b, core.resize(s, shape), k)
            file = f"{index:06}.npz"
            packet = OUT / "predictions" / file
            np.savez_compressed(packet, original_hw=np.asarray(shape),
                                full_composition=np.packbits(full),
                                same_added_area_source_order=np.packbits(control))
            r = {"index": index, "episode_id": eid, "dataset": row["dataset"], "file": file,
                 "sha256": core.sha(packet), "background_core_pixels": int(weights.sum()),
                 "background_floor": floor, "source_threshold": threshold, "admission_threshold": admitted,
                 "source_separation": scale, "threshold_constraint_triggered": trigger,
                 "zero_secondary_fallback": not valid_scale,
                 "old_added_pixels": int((old & ~b).sum()), "accepted_added_pixels": k,
                 "rejected_old_additions": int((old & ~full).sum()),
                 "old_deleted_baseline_fg": int((b & ~old).sum()), "same_k_control": ties,
                 "parent_replay_bit_exact": True, "zero_evidence_identity": True,
                 "baseline_fg_deletions_unchanged": True, "seconds": time.monotonic() - tick}
            core.write(rp, r)
            records.append(r)
            if index % 50 == 0 or index == 899:
                core.write(OUT / "activity.json", {"state": "INFERENCE", "pid": os.getpid(), "completed": len(records), "total": 900})
                print(json.dumps({"completed": len(records), "total": 900, "seconds": round(time.monotonic() - start, 3)}), flush=True)
    assert len(records) == 900 and not attempts
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"state": "ALL900_COMPOSITION_PREDICTIONS_SEALED", "n": 900, "records": records,
              "config_sha256": core.sha(OUT / "config.json"), "manifest_sha256": core.sha(OUT / "manifest.json"),
              "protocol_sha256": core.sha(PROTOCOL), "seconds": time.monotonic() - start,
              "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
              "new_encoder_calls": 0, "new_raw_reads": 0, "new_reference_fits": 0, "query_gt_access_attempts": attempts}
    core.write(OUT / "sealed.json", result)
    core.write(OUT / "activity.json", {"state": "COMPLETE_SEALED", "completed": 900, "total": 900})
    print(json.dumps({k: result[k] for k in ("state", "seconds", "peak_rss_bytes", "query_gt_access_attempts")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("infer",))
    args = parser.parse_args()
    infer()
