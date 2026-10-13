"""Independent bitmap, edit-budget and official-slot checks of consensus900."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent / "cv_data/a/consensus_composition900_v1_20261011"


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(a, s): return np.unpackbits(a, count=int(np.prod(s))).reshape(s).astype(bool)


def bilinear(a, shape):
    a = np.asarray(a, np.float64)
    y = (np.arange(shape[0]) + .5) * a.shape[0] / shape[0] - .5
    x = (np.arange(shape[1]) + .5) * a.shape[1] / shape[1] - .5
    yi, xi = np.floor(y).astype(int), np.floor(x).astype(int)
    dy, dx = y - yi, x - xi
    y1, x1 = np.clip(yi + 1, 0, a.shape[0] - 1), np.clip(xi + 1, 0, a.shape[1] - 1)
    yi, xi = np.clip(yi, 0, a.shape[0] - 1), np.clip(xi, 0, a.shape[1] - 1)
    horizontal = (1 - dx) * a[:, xi] + dx * a[:, x1]
    return (1 - dy[:, None]) * horizontal[yi] + dy[:, None] * horizontal[y1]


def main():
    tick = time.monotonic()
    seal, report = read(ROOT / "sealed.json"), read(ROOT / "report.json")
    assert seal["state"] == "ALL900_CONSENSUS_PREDICTIONS_SEALED" and seal["n"] == 900
    assert report["seal_sha256"] == sha(ROOT / "sealed.json")
    assert seal["manifest_sha256"] == sha(ROOT / "manifest.json")
    scored = read(ROOT / "scored_episodes.json")
    assert report["scored_records_sha256"] == sha(ROOT / "scored_episodes.json")
    counts, checks = defaultdict(lambda: np.zeros(2, np.int64)), 0
    for source, rec, row in zip(read(ROOT / "manifest.json"), seal["records"], scored):
        meta, shape = source["row"], tuple(source["row"]["query_size_hw"])
        assert meta["episode_id"] == rec["episode_id"] == row["episode_id"]
        packet = ROOT / "predictions" / rec["file"]
        assert sha(packet) == rec["sha256"]
        for key in ("baseline", "old", "mean", "source_score"):
            assert sha(source[key]) == source[key + "_sha256"]
            checks += 1
        with np.load(source["baseline"], allow_pickle=False) as z: b = unpack(z["original/" + source["baseline_key"]], shape)
        with np.load(source["old"], allow_pickle=False) as z: r = unpack(z["anchor.global"], shape)
        with np.load(source["mean"], allow_pickle=False) as z: m = unpack(z[source["mean_key"]], shape)
        with np.load(source["source_score"], allow_pickle=False) as z: s = z[source["score_key"]].reshape(64, 64).astype(float)
        with np.load(packet, allow_pickle=False) as z:
            shared, control = [unpack(z[k], shape) for k in ("shared_edits_complete", "same_edit_budget_source_order")]
        disagreement = r != m
        expected = np.where(disagreement, b, r)
        assert np.array_equal(shared, expected)
        add, delete = shared & ~b, b & ~shared
        assert int(add.sum()) == rec["added_pixels"] and int(delete.sum()) == rec["deleted_pixels"]
        raw_score = bilinear(s, shape)
        norm = (s - s.min()) / max(float(np.ptp(s)), 1e-6)
        confidence = np.abs(2 * bilinear(norm, shape) - 1) if np.ptp(s) else np.zeros(shape)
        mutable = confidence <= .5
        assert not np.any((shared != b) & ~mutable)
        chosen = []
        for eligible, k, descending in (((~b) & (r | m) & mutable, int(add.sum()), True),
                                         (b & ((~r) | (~m)) & mutable, int(delete.sum()), False)):
            ix = np.flatnonzero(eligible.ravel())
            values = raw_score.ravel()[ix]
            order = np.lexsort((ix, -values if descending else values))
            selected = np.zeros(b.size, bool)
            selected[ix[order[:k]]] = True
            chosen.append(selected.reshape(shape))
        assert np.array_equal((b | chosen[0]) & ~chosen[1], control)
        checks += 5
        with Image.open(meta["query_mask_path"]) as image: original = np.asarray(image.convert("L")) > 0
        y, x = np.arange(shape[0]) * original.shape[0] // shape[0], np.arange(shape[1]) * original.shape[1] // shape[1]
        gt = original[y[:, None], x[None]]
        for method, mask in (("shared_edits_complete", shared), ("same_edit_budget_source_order", control),
                             ("foris.crf", b), ("anchor.global", r), ("mean", m)):
            bins = np.bincount((2 * gt.astype(np.uint8) + mask.astype(np.uint8)).ravel(), minlength=4)
            i, u, fp, fn = int(bins[3]), int(bins[1:4].sum()), int(bins[1]), int(bins[2])
            assert [i, u, fp, fn] == [row["metrics"][method][k] for k in ("I", "U", "FP", "FN")]
            counts[(meta["dataset"], str(meta["fold"]), str(meta["loader_class_id"]), method)] += [i, u]
            checks += 1
    official = {}
    for dataset, value in report["datasets"].items():
        groups = defaultdict(list)
        for fc in value["summary"]["per_class"]:
            f, c = fc.split(":", 1); groups[f].append(c)
        official[dataset] = {}
        for method in ("shared_edits_complete", "same_edit_budget_source_order", "foris.crf", "anchor.global", "mean"):
            score = float(np.mean([np.mean([100 * counts[dataset, f, c, method][0] / max(counts[dataset, f, c, method][1], 1)
                                           for c in cs]) for f, cs in groups.items()]))
            assert abs(score - value["summary"]["miou"][method]) < 1e-10
            official[dataset][method] = score
            checks += 1
    result = {"state": "INDEPENDENT900_CONSENSUS_VERIFIED", "checks": checks,
              "seal_sha256": sha(ROOT / "sealed.json"), "report_sha256": sha(ROOT / "report.json"),
              "verifier_sha256": sha(__file__), "seconds": time.monotonic() - tick, "official_scores": official,
              "control_source_order_mismatch_pixels": 0, "new_encoder_calls": 0, "new_predictions": 0,
              "scope": "900 shared/disagreement/protected identities, both edit budgets and NumPy source-order controls, exact bitmap metrics and independent official class pooling"}
    (ROOT / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
