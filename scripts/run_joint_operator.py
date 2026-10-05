#!/usr/bin/env python3
"""Exact bounded two-operation DEV search on sealed masks, with CUDA bit counts.

This tests the optimization gap of one-step greedy. It is DEV-label-based rule
selection, not query-GT inference or a global theorem for arbitrary-length paths.
The CPU score phase reads sealed final masks separately. No encoder is loaded.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

for env_name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[env_name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_hash(path, expected):
    from ics.experiment import sha
    if sha(path) != expected:
        raise ValueError("Changed sealed input: " + str(path))


def source(path, rows, names=None, seal_name="sealed.json", directory="predictions"):
    from ics.experiment import load_rows, sha
    seal = json.loads((path / seal_name).read_text())
    if seal.get("state", "ALL_PREDICTIONS_SEALED") != "ALL_PREDICTIONS_SEALED":
        raise ValueError("Incomplete source: " + str(path))
    check_hash(path / "manifest.json", seal["manifest_sha256"])
    other = {r["key"]: r for r in load_rows(path / "manifest.json")}
    if set(seal["predictions"]) != set(other) or seal.get("n", len(other)) != len(other):
        raise ValueError("Incomplete source prediction key set: " + str(path))
    for row in rows:
        if any(row[k] != other[row["key"]][k] for k in ("fold", "e", "c", "support", "query")):
            raise ValueError("Episode/photo identity differs: " + row["key"])
    names = names or [k for k in seal["arms"] if "pre-" not in k and "@never" not in k]
    for field, filename in (("protocol_sha256", "protocol.json"), ("config_sha256", "config.json")):
        if field in seal:
            check_hash(path / filename, seal[field])
    return dict(path=str(path), seal_sha256=sha(path / seal_name), seal_name=seal_name,
                directory=directory, arms=names, hashes=seal["predictions"],
                fields=seal.get("fields", {}))


def popcount(value):
    import torch
    value = value - ((value >> 1) & 85)
    value = (value & 51) + ((value >> 2) & 51)
    value = (value + (value >> 4)) & 15
    return value.sum(dim=1, dtype=torch.float32)


def infer(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.experiment import load_rows, packet, photo_groups, sha
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA required")
    if a.out.exists():
        raise FileExistsError("Fresh output required")
    all_rows = load_rows(a.stage_run / "manifest.json")
    if len(all_rows) != 241:
        raise ValueError("Require full DEV241 sources")
    rows = ([all_rows[0], next(r for r in all_rows if r["fold"] != all_rows[0]["fold"])]
            if a.smoke else all_rows)
    sources = {
        "stage": source(a.stage_run, rows),
        "recheck": source(a.recheck_run, rows, ["RCG", "RCG_count_matched_delete", "conservative_delete",
                         "external_mean__delete", "external_mean_delete_sameK_RCG", "MEAN_CONTROL"]),
        "proposal": source(a.proposal_run, rows, ["latent_native", "latent_native.control",
                          "latent_native.two_slot_em.control", "structure_conditioned.kernel.control"],
                          seal_name="proposal_sealed.json", directory="proposals"),
        "multilayer": source(a.d_run, rows, ["multilayer", "multilayer.control", "multilayer.delta.control"]),
        "mean": source(a.mean_run, rows, ["mean_graph", "mean_graph.control", "mean_graph.unary.control"]),
    }
    library, truth, receipts = {}, [], {}
    n, packed_size = len(rows), 131072

    def put(name, index, mask):
        if mask.dtype != np.uint8 or mask.shape != (packed_size,):
            raise ValueError("Noncanonical packed1024 mask: " + name)
        library.setdefault(name, np.empty((n, packed_size), np.uint8))[index] = mask

    begin = time.monotonic()
    for i, row in enumerate(rows):
        key = row["key"]
        for label, spec in sources.items():
            file = Path(spec["path"]) / spec["directory"] / (key + ".npz")
            check_hash(file, spec["hashes"][key])
            with np.load(file, allow_pickle=False) as z:
                for name in spec["arms"]:
                    put(label + ":" + name, i, z[name])
        field_path = a.recheck_run / "fields" / (key + ".npz")
        check_hash(field_path, sources["recheck"]["fields"][key])
        packet_path = packet(a.root, row)
        receipts[key] = dict(packet=str(packet_path), packet_sha256=sha(packet_path))
        with np.load(packet_path, allow_pickle=False) as z:
            truth.append(z["truth"].copy())  # DEV fitting/count phase, explicitly label-based.
            put("native", i, z["native"])
            score = np.asarray(z["score"], np.float32)
            score = (score - score.min()) / max(float(score.max() - score.min()), 1e-6)
        with np.load(field_path, allow_pickle=False) as z:
            field = np.asarray(z["rcg"], np.float32)
        for label, values in (("foris", score), ("rcg", field)):
            up = F.interpolate(torch.from_numpy(values)[None, None], (1024, 1024),
                               mode="bilinear", align_corners=False)[0, 0].numpy()
            for threshold in (.3, .4, .6, .7):
                put(f"field:{label}>{threshold:g}", i, np.packbits(up > threshold))
    names = sorted(library)
    lib = {k: torch.from_numpy(v).cuda() for k, v in library.items()}
    truth = torch.from_numpy(np.stack(truth)).cuda()
    origin_name = "stage:model.raw_nn"
    origin = lib[origin_name]
    ops = [("identity", None)] + [(op, k) for k in names for op in ("add", "delete")]
    recipes = [(0, 0)] + [(i, 0) for i in range(1, len(ops))]
    recipes += [(i, j) for i in range(1, len(ops)) for j in range(1, len(ops))]
    a.out.mkdir(); (a.out / "predictions").mkdir()
    code = {str(Path(__file__).resolve()): sha(__file__),
            str(REPO / "src/ics/experiment.py"): sha(REPO / "src/ics/experiment.py")}
    protocol = dict(n=n, sources=sources, source_code=code, library=names, ops=ops, recipes=recipes,
                    origin=origin_name, input_receipts=receipts, resolution=1024, seed=0,
                    additional_encoder_forwards=0, smoke=a.smoke,
                    selection="DEV GT on three folds; photo-connected groups of read fold excluded; no per-query routing",
                    comparison="same precomputed library: direct complete-mask selection, one step, exact up to two steps",
                    resources="cached rawl24/APD and multilayer ingredients; deployment cost depends on selected producers",
                    theorem_scope="global best inside the enumerated <=2-operation family on fitting data only")
    write_json(a.out / "manifest.json", rows); write_json(a.out / "protocol.json", protocol)
    write_json(a.out / "candidate_library_seal.json", dict(protocol_sha256=sha(a.out / "protocol.json"),
               stage="FIXED_LIBRARY_BEFORE_CANDIDATE_SCORING", candidate_count=len(recipes)))

    def apply(mask, index):
        op, name = ops[index]
        if op == "identity":
            return mask
        return mask | lib[name] if op == "add" else mask & lib[name]

    def iu(mask):
        return torch.stack((popcount(mask & truth), popcount(mask | truth)), 1).cpu().numpy().astype(np.int64)

    # Check the packed-byte kernel against all possible byte values, not its own formula.
    test = torch.arange(256, device="cuda", dtype=torch.int64).to(torch.uint8).reshape(256, 1)
    expected = np.array([int(v).bit_count() for v in range(256)])
    if not np.array_equal(popcount(test).cpu().numpy(), expected):
        raise ValueError("CUDA popcount parity failed")
    counts = np.empty((len(recipes), n, 2), np.int64)
    for index, (first, second) in enumerate(recipes):
        counts[index] = iu(apply(apply(origin, first), second))
        if index % 250 == 0:
            print(json.dumps(dict(evaluated=index, total=len(recipes), seconds=round(time.monotonic()-begin, 2))), flush=True)
    complete = np.stack([iu(lib[k]) for k in names])
    classes = np.array([r["c"] for r in rows]); folds = np.array([r["fold"] for r in rows])
    groups = photo_groups(rows)
    selected = {k: np.empty((n, packed_size), np.uint8) for k in ("select.complete", "select.greedy1", "select.joint2")}
    choices = {}

    def values(arrays, fit):
        cls = sorted(set(classes[fit]))
        weight = np.stack([fit & (classes == c) for c in cls], 1).astype(np.float64)
        intersections = arrays[:, :, 0] @ weight; unions = arrays[:, :, 1] @ weight
        return 100 * (intersections / np.maximum(unions, 1)).mean(1)

    for fold in sorted(set(folds)):
        read = folds == fold; fit = ~read & ~np.isin(groups, groups[read])
        if not fit.any():
            raise ValueError("No photo-isolated fitting episodes")
        score = values(counts, fit); direct_score = values(complete, fit)
        direct = int(direct_score.argmax()); greedy = int(score[:len(ops)].argmax()); joint = int(score.argmax())
        if score[joint] + 1e-9 < score[greedy]:
            raise ValueError("Exact-family containment failed")
        choices[str(fold)] = dict(fit_episodes=int(fit.sum()), read_episodes=int(read.sum()),
            complete=names[direct], greedy=recipes[greedy], joint=recipes[joint],
            fit_scores=dict(complete=float(direct_score[direct]), greedy1=float(score[greedy]), joint2=float(score[joint])))
        for label, mask in (("select.complete", lib[names[direct]]),
                            ("select.greedy1", apply(apply(origin, recipes[greedy][0]), recipes[greedy][1])),
                            ("select.joint2", apply(apply(origin, recipes[joint][0]), recipes[joint][1]))):
            selected[label][read] = mask.cpu().numpy()[read]
    comparisons = {"native": "native", "origin": origin_name, "rcg": "recheck:RCG",
                   "astra.control": "recheck:external_mean__delete", "mean.control": "recheck:MEAN_CONTROL",
                   "insid3.control": "stage:insid3.final"}
    seals = {}
    for i, row in enumerate(rows):
        masks = {k: library[v][i] for k, v in comparisons.items()}
        masks.update({k: v[i] for k, v in selected.items()})
        path = a.out / "predictions" / (row["key"] + ".npz")
        np.savez_compressed(path, **masks); seals[row["key"]] = sha(path)
    write_json(a.out / "choices.json", choices)
    write_json(a.out / "sealed.json", dict(state="ALL_PREDICTIONS_SEALED", n=n,
               manifest_sha256=sha(a.out / "manifest.json"), protocol_sha256=sha(a.out / "protocol.json"),
               choices_sha256=sha(a.out / "choices.json"), predictions=seals,
               query_gt_used_for_DEV_fold_fitting=True, per_query_gt_routing=False,
               seconds=time.monotonic()-begin, cuda_peak_bytes=torch.cuda.max_memory_allocated()))
    print(json.dumps(dict(state="ALL_PREDICTIONS_SEALED", n=n, candidates=len(recipes),
                          seconds=time.monotonic()-begin)), flush=True)


def score(a):
    import numpy as np
    from ics.experiment import sha, packet, summarize, unpack
    seal = json.loads((a.out / "sealed.json").read_text())
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or (a.out / "report.json").exists():
        raise ValueError("Incomplete/already scored run")
    for field, name in (("manifest_sha256", "manifest.json"), ("protocol_sha256", "protocol.json"),
                        ("choices_sha256", "choices.json")):
        check_hash(a.out / name, seal[field])
    protocol = json.loads((a.out / "protocol.json").read_text())
    for path, digest in protocol["source_code"].items():
        check_hash(Path(path), digest)
    rows = json.loads((a.out / "manifest.json").read_text()); arrays, corrections, details = {}, {}, []
    for row in rows:
        key = row["key"]; receipt = protocol["input_receipts"][key]
        check_hash(Path(receipt["packet"]), receipt["packet_sha256"])
        with np.load(receipt["packet"], allow_pickle=False) as z:
            truth = unpack(z["truth"])
        path = a.out / "predictions" / (key + ".npz"); check_hash(path, seal["predictions"][key])
        with np.load(path, allow_pickle=False) as z:
            masks = {k: unpack(z[k]) for k in z.files}
        origin = masks["origin"]; episode = dict(row, iu={})
        for name, mask in masks.items():
            iu = [int((mask & truth).sum()), int((mask | truth).sum())]
            arrays.setdefault(name, []).append(iu); episode["iu"][name] = iu
            add, delete = mask & ~origin, origin & ~mask
            counts = dict(add_TP=int((add & truth).sum()), add_FP=int((add & ~truth).sum()),
                          delete_TP=int((delete & truth).sum()), delete_FP=int((delete & ~truth).sum()))
            corrections.setdefault(name, []).append(dict(key=key, c=row["c"], fold=row["fold"],
                batch=row.get("batch", "unspecified"), **counts))
        details.append(episode)
    report, _ = summarize(rows, {k: np.array(v) for k, v in arrays.items()}, corrections)
    report["corrections_vs_origin"] = report.pop("corrections_vs_native")
    report["corrections_vs_origin_by_class"] = report.pop("corrections_by_class")
    report["corrections_vs_origin_by_batch"] = report.pop("corrections_by_batch")
    report.update(selection=protocol["selection"], candidate_count=len(protocol["recipes"]),
                  choices=json.loads((a.out / "choices.json").read_text()), prediction_seal_sha256=sha(a.out / "sealed.json"))
    write_json(a.out / "report.json", report)
    (a.out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in details))
    print(json.dumps(dict(n=report["n"], scores=report["scores"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("infer", "score")); parser.add_argument("--root", type=Path)
    for name in ("stage-run", "recheck-run", "proposal-run", "d-run", "mean-run"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "infer" and any(getattr(args, k) is None for k in
          ("root", "stage_run", "recheck_run", "proposal_run", "d_run", "mean_run")):
        parser.error("Inference requires every explicit source")
    {"infer": infer, "score": score}[args.phase](args)


if __name__ == "__main__":
    main()
