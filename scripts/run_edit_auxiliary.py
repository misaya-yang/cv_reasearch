#!/usr/bin/env python3
"""Prepare, infer, or independently evaluate one fixed cached DEV241 edit batch.

Default prepare is CPU-only dependency validation. No stage loads a model,
opens an image, downloads assets, or dispatches remote work. Inference completes
and seals all proposals before applying the existing witness helper to edits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
BATCH_NAME = "fixed_edit_region_witness_auxiliary_baseline_v1"
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from ics import edit_auxiliary as edit
from ics.experiment import load_inputs, load_rows, metric, packet, photo_groups, render, sha, summarize, unpack


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_npz(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def check_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Changed sealed file: {path}")


def source_run(path, rows, arm):
    seal = json.loads((path / "sealed.json").read_text())
    check_hash(path / "manifest.json", seal["manifest_sha256"])
    if seal.get("state", "ALL_PREDICTIONS_SEALED") != "ALL_PREDICTIONS_SEALED":
        raise ValueError(f"Source is not fully sealed: {path}")
    other = {r["key"]: r for r in load_rows(path / "manifest.json")}
    keys = {r["key"] for r in rows}
    if set(other) != keys or set(seal["predictions"]) != keys:
        raise ValueError(f"Source must have the exact DEV241 key set: {path}")
    for row in rows:
        if any(row[k] != other[row["key"]][k] for k in ("fold", "e", "c", "support", "query", "batch")):
            raise ValueError(f"Source episode identity mismatch: {row['key']}")
    for tag, filename in (("config_sha256", "config.json"), ("protocol_sha256", "protocol.json"),
                          ("arm_bases_sha256", "arm_bases.pt")):
        if tag in seal:
            check_hash(path / filename, seal[tag])
    return dict(path=str(path.resolve()), seal_sha256=sha(path / "sealed.json"), arm=arm,
                manifest_sha256=seal["manifest_sha256"]), seal


def code_receipts():
    modules = {name: importlib.import_module("ics.methods." + name) for name in
               ("latent_native", "latent", "structure_conditioned", "structure", "transport_query_witness", "transport")}
    paths = [Path(m.__file__) for m in modules.values()] + [Path(__file__), REPO / "src/ics/edit_auxiliary.py",
                                                           REPO / "src/ics/experiment.py"]
    return {str(p.relative_to(REPO)): sha(p) for p in paths}, {
        name: dict(module.CONFIG) for name, module in modules.items()}


def validate_input(inputs):
    q, r, cov, score = (np.asarray(x) for x in inputs)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError("Expected matching N x D query/reference features")
    if score.shape != (64, 64) or cov.shape != (64, 64) or len(q) != 4096 or len(r) != 4096:
        raise ValueError("Require existing 64x64 token grids and 1024 masks")
    if not all(np.isfinite(x).all() for x in (q, r, cov, score)):
        raise ValueError("Nonfinite cached input")
    if np.any(np.linalg.norm(q, axis=1) < 1e-8) or np.any(np.linalg.norm(r, axis=1) < 1e-8):
        raise ValueError("Zero-norm cached feature")
    if cov.min() < 0 or cov.max() > 1 or not np.any(cov >= .5) or not np.any(cov < .5):
        raise ValueError("Reference coverage must supply both FG and BG roles")


def prepare(args):
    """Check every dependency without matrix inference or query-truth access."""
    root = args.root.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        marker = args.out / ("protocol.json" if (args.out / "protocol.json").exists() else "dependency_check.json")
        if not marker.exists() or json.loads(marker.read_text()).get("name") != BATCH_NAME:
            raise ValueError("Output is owned by another run; choose a fresh output directory")
    rows = load_rows(args.manifest)
    if len(rows) != 241 or len(set(r["c"] for r in rows)) != 79 or len(set(photo_groups(rows))) != 239:
        raise ValueError("This batch requires the full DEV241 / 79 classes / 239 photo groups")
    document = json.loads(args.manifest.read_text())
    if isinstance(document, dict) and document.get("seed") != 0:
        raise ValueError("Require the fixed seed-0 DEV241 manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    failures, sources, seals, receipts = [], {}, {}, {}
    for name, path, arm in (("delta", args.d_run, "multilayer.delta.control"), ("rcg", args.rcg_run, "rcg")):
        try:
            sources[name], seals[name] = source_run(path, rows, arm)
        except (OSError, ValueError, KeyError) as error:
            failures.append(dict(dependency=name, error=str(error)))
    try:
        hashes, configs = code_receipts()
        packages = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "scipy")}
    except (ImportError, OSError, ValueError) as error:
        failures.append(dict(dependency="runtime", error=str(error)))
        hashes, configs, packages = {}, {}, {}
    for row in rows:
        key = row["key"]
        try:
            inputs, receipt = load_inputs(root, row)
            validate_input(inputs)
            with np.load(packet(root, row), allow_pickle=False) as p:
                unpack(p["native"])  # truth is deliberately never indexed here
            receipts[key] = receipt
            for name, seal in seals.items():
                path = Path(sources[name]["path"]) / "predictions" / f"{key}.npz"
                check_hash(path, seal["predictions"][key])
                with np.load(path, allow_pickle=False) as p:
                    unpack(p[sources[name]["arm"]])
                previous = seal.get("inputs", {}).get(key)
                if previous and previous.get("packet_sha256") != receipt["packet_sha256"]:
                    raise ValueError(f"{name} used a different cached packet")
                if previous and previous.get("feature_sha256", receipt["feature_sha256"]) != receipt["feature_sha256"]:
                    raise ValueError(f"{name} used different cached q/r features")
                if name == "rcg" and not previous:
                    raise ValueError("RCG source lacks original-packet identity receipt")
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            failures.append(dict(dependency=key, error=str(error)))
    check = dict(name=BATCH_NAME, state="DEPENDENCIES_MISSING" if failures else "PREPARED", episodes=len(rows),
                 inputs_checked=len(receipts), failures=failures, query_truth_opened=False,
                 encoder_instances=0, encoder_forwards=0, packages=packages)
    write_json(args.out / "dependency_check.json", check)
    if failures:
        raise RuntimeError(f"{len(failures)} dependency failures; see {args.out / 'dependency_check.json'}")
    protocol = dict(name=BATCH_NAME, rows=rows, packages=packages,
                    root=str(root), sources=sources, input_receipts=receipts,
                    source_manifest_sha256=sha(args.manifest), code_sha256=hashes, configs=configs,
                    proposal_arms=edit.PROPOSALS, recipes=edit.recipes(), device=args.device, threads=args.threads,
                    helper="transport_query_witness.predict, unchanged continuous FG/BG contrast",
                    helper_threshold=.5, helper_resize="bilinear 1024, align_corners=False, then strict >.5",
                    native="original shared cached packet.native; never replayed native",
                    native_score_control="raw score bilinear to1024; A descending, B ascending; flat-index tie break; exact auxiliary edit count per episode",
                    proposal_location_control="raw proposal edit count; highest native scores over all native background for A, lowest over all native foreground for B; no proposal-domain restriction",
                    stage_order="all241 proposals sealed, then all241 auxiliary/edit masks sealed, then separate evaluation",
                    seed=0, resolution=1024, dataset="COCO-20i 1-shot DEV241", development_only=True,
                    query_gt_usage="independent evaluation only", query_gt_routing=False,
                    encoder_instances=0, encoder_forwards=0, resource_setting="one labeled reference, cached frozen DINOv3",
                    novelty="thresholded edit-region agreement baseline; not an independent new method",
                    parameter_selection="fixed before outcomes; no proposal dropped for net score")
    destination = args.out / "protocol.json"
    if destination.exists():
        # JSON turns tuples into lists; compare canonical serialization.
        if json.loads(destination.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError("Prepared protocol changed; choose a new output directory")
    else:
        write_json(destination, protocol)
        write_json(args.out / "manifest.json", rows)
    return protocol


def verify_protocol(out):
    protocol = json.loads((out / "protocol.json").read_text())
    hashes, configs = code_receipts()
    if hashes != protocol["code_sha256"] or configs != protocol["configs"]:
        raise ValueError("Method, helper, runner or scorer changed since preparation")
    if json.loads((out / "manifest.json").read_text()) != protocol["rows"]:
        raise ValueError("Prepared manifest changed")
    for source in protocol["sources"].values():
        check_hash(Path(source["path"]) / "sealed.json", source["seal_sha256"])
    return protocol


def verify_receipt(out, receipt, protocol_hash):
    if receipt["protocol_sha256"] != protocol_hash:
        raise ValueError("Episode belongs to a different prepared protocol")
    for relative, expected in receipt["files"].items():
        check_hash(out / relative, expected)


def initialize_worker(threads):
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def inference_episode(task):
    """Separate proposal and helper passes; class/GT never passed to methods."""
    out, protocol, row, phase = task
    out = Path(out)
    key, started = row["key"], time.monotonic()
    protocol_hash = sha(out / "protocol.json")
    receipt_path = out / f"{phase}_receipts" / f"{key}.json"
    inputs, receipt = load_inputs(protocol["root"], row)
    if receipt != protocol["input_receipts"][key]:
        raise ValueError(f"Prepared inputs changed: {key}")
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        verify_receipt(out, previous, protocol_hash)
        return dict(key=key, phase=phase, resumed=True)
    if phase == "proposal":
        fields, masks, audits = {}, {}, {}
        with np.load(packet(protocol["root"], row), allow_pickle=False) as p:
            masks["native"] = p["native"].copy()
        for alias, (module, function, arm) in edit.PROPOSALS.items():
            if module:
                fn = getattr(importlib.import_module("ics.methods." + module), function)
                field, audits[arm] = fn(*inputs, device=protocol["device"])
                fields[arm] = field
                masks[arm] = np.packbits(render(field))
            else:
                source = protocol["sources"][alias]
                path = Path(source["path"]) / "predictions" / f"{key}.npz"
                seal = json.loads((Path(source["path"]) / "sealed.json").read_text())
                check_hash(path, seal["predictions"][key])
                with np.load(path, allow_pickle=False) as p:
                    masks[arm] = p[arm].copy()
                audits[arm] = dict(source_prediction_sha256=sha(path), source_seal_sha256=source["seal_sha256"])
        products = {f"proposals/{key}.npz": masks, f"fields/{key}.npz": fields}
    else:
        first = json.loads((out / "proposal_receipts" / f"{key}.json").read_text())
        verify_receipt(out, first, protocol_hash)
        with np.load(out / "proposals" / f"{key}.npz", allow_pickle=False) as p:
            native = unpack(p["native"])
            proposals = {a: unpack(p[v[2]]) for a, v in edit.PROPOSALS.items()}
            masks = {v[2]: p[v[2]].copy() for v in edit.PROPOSALS.values()}
        helper = importlib.import_module("ics.methods.transport_query_witness")
        h, info = helper.predict(*inputs, device=protocol["device"])
        helper_fg = render(h)
        score = edit.resize_continuous(inputs[3])
        edits = edit.accepted_edits(native, proposals, helper_fg, score)
        for arm, recipe in protocol["recipes"].items():
            masks[arm] = np.packbits(edit.compose(native, edits, recipe))
        for alias, proposal in proposals.items():
            if not np.array_equal(edit.compose(native, edits, {"A": alias, "B": alias, "mode": "noaux"}), proposal):
                raise ValueError("Unassisted combination does not equal its proposal")
        products = {f"predictions/{key}.npz": masks,
                    f"edits/{key}.npz": {k: np.packbits(v) for k, v in edits.items()},
                    f"auxiliary/{key}.npz": {"h": h, "helper_fg": np.packbits(helper_fg)}}
        audits = dict(helper=info, accepted_counts={k: int(v.sum()) for k, v in edits.items()})
    files = {}
    for relative, values in products.items():
        write_npz(out / relative, values)
        files[relative] = sha(out / relative)
    write_json(receipt_path, dict(key=key, protocol_sha256=protocol_hash, inputs=receipt, files=files,
                                audits=audits, seconds=time.monotonic() - started, query_truth_opened=False))
    return dict(key=key, phase=phase, resumed=False, seconds=time.monotonic() - started)


def pool_map(function, tasks, workers, threads=1):
    # Explicit spawn: model-free workers, no inherited CUDA context or encoder.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker, initargs=(threads,)) as pool:
        yield from pool.map(function, tasks, chunksize=1)


def seal_phase(out, protocol, phase):
    files, receipts = {}, {}
    for row in protocol["rows"]:
        path = out / f"{phase}_receipts" / f"{row['key']}.json"
        receipt = json.loads(path.read_text())
        verify_receipt(out, receipt, sha(out / "protocol.json"))
        receipts[str(path.relative_to(out))] = sha(path)
        files.update(receipt["files"])
    seal = dict(state="ALL_PREDICTIONS_SEALED", phase=phase, n=241,
                manifest_sha256=sha(out / "manifest.json"), protocol_sha256=sha(out / "protocol.json"),
                files=files, receipts=receipts, inputs=protocol["input_receipts"], query_labels_opened=False)
    if phase == "auxiliary":
        seal["proposal_seal_sha256"] = sha(out / "proposal_sealed.json")
        seal["predictions"] = {r["key"]: files[f"predictions/{r['key']}.npz"] for r in protocol["rows"]}
    path = out / ("proposal_sealed.json" if phase == "proposal" else "sealed.json")
    if path.exists() and json.loads(path.read_text()) != seal:
        raise ValueError("Existing cohort seal changed; refusing to replace it")
    write_json(path, seal)


def infer(args):
    protocol = prepare(args)
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Explicit CUDA inference requested but no visible GPU")
    for directory in ("proposals", "fields", "proposal_receipts", "predictions", "edits", "auxiliary", "auxiliary_receipts"):
        (args.out / directory).mkdir(exist_ok=True)
    phases = ("proposal",) if args.proposals_only else ("proposal", "auxiliary")
    for phase in phases:
        tasks = [(str(args.out), protocol, row, phase) for row in protocol["rows"]]
        for result in pool_map(inference_episode, tasks, args.workers, args.threads):
            print(json.dumps(result), flush=True)
        seal_phase(args.out, protocol, phase)
    state = "PROPOSALS_SEALED_UNSCORED" if args.proposals_only else "INFERRED_SEALED_UNSCORED"
    completion_name = "proposal_completion.json" if args.proposals_only else "completion.json"
    write_json(args.out / completion_name, dict(state=state, episodes=241,
                                                query_truth_opened=False, encoder_forwards=0))


def evaluate_episode(task):
    """Only this phase indexes truth; verify masks and exact I/U edit algebra."""
    out, protocol, row = task
    out = Path(out)
    with np.load(packet(protocol["root"], row), allow_pickle=False) as p:
        truth, native = unpack(p["truth"]), unpack(p["native"])
    buckets = edit.error_buckets(native, truth)
    with np.load(out / "edits" / f"{row['key']}.npz", allow_pickle=False) as p:
        edits = {a: unpack(p[a]) for a in p.files}
    masks = {"native": native}
    with np.load(out / "predictions" / f"{row['key']}.npz", allow_pickle=False) as p:
        masks.update({a: unpack(p[a]) for a in p.files})
    i0, u0 = int((native & truth).sum()), int((native | truth).sum())
    arrays, corrections, reach = {}, {}, {}
    for arm, mask in masks.items():
        addition, deletion = mask & ~native, native & ~mask
        counts = edit.action_counts(addition, deletion, truth)
        iu = [int((mask & truth).sum()), int((mask | truth).sum())]
        if iu != [i0 + counts["add_TP"] - counts["delete_TP"], u0 + counts["add_FP"] - counts["delete_FP"]]:
            raise ValueError("Exact edit I/U algebra failed")
        if arm in protocol["recipes"] and not np.array_equal(mask, edit.compose(native, edits, protocol["recipes"][arm])):
            raise ValueError(f"Saved combination mask disagrees with accepted edits: {arm}")
        arrays[arm] = iu
        corrections[arm] = dict(key=row["key"], c=row["c"], fold=row["fold"], batch=str(row.get("batch", "unspecified")), **counts)
        reach[arm] = {name: int(((deletion if name in edit.BUCKETS[:2] else addition) & b).sum())
                      for name, b in buckets.items()}
    losses = {}
    for alias in edit.PROPOSALS:
        raw = edit.action_counts(edits[f"{alias}.noaux.A"], edits[f"{alias}.noaux.B"], truth)
        for mode in ("aux", "native_score"):
            kept = edit.action_counts(edits[f"{alias}.{mode}.A"], edits[f"{alias}.{mode}.B"], truth)
            losses[f"{alias}.{mode}"] = dict(raw=raw, kept=kept, rates=edit.retention(raw, kept))
    return dict(key=row["key"], arrays=arrays, corrections=corrections, reach=reach, losses=losses,
                error_mass={name: int(b.sum()) for name, b in buckets.items()},
                native_FN=int((truth & ~native).sum()), native_FP=int((native & ~truth).sum()))


def evaluate(args):
    protocol = verify_protocol(args.out)
    seal = json.loads((args.out / "sealed.json").read_text())
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 241:
        raise ValueError("Evaluation requires complete DEV241 seal")
    check_hash(args.out / "manifest.json", seal["manifest_sha256"])
    check_hash(args.out / "protocol.json", seal["protocol_sha256"])
    check_hash(args.out / "proposal_sealed.json", seal["proposal_seal_sha256"])
    proposal_seal = json.loads((args.out / "proposal_sealed.json").read_text())
    for verified in (proposal_seal, seal):
        for relative, expected in {**verified["files"], **verified["receipts"]}.items():
            check_hash(args.out / relative, expected)
    for row in protocol["rows"]:
        check_hash(packet(protocol["root"], row), seal["inputs"][row["key"]]["packet_sha256"])
    details = list(pool_map(evaluate_episode, [(str(args.out), protocol, r) for r in protocol["rows"]], args.score_workers))
    arms = list(details[0]["arrays"])
    if any(set(d["arrays"]) != set(arms) for d in details):
        raise ValueError("Unpaired prediction arms")
    arrays = {a: np.array([d["arrays"][a] for d in details], np.int64) for a in arms}
    corrections = {a: [d["corrections"][a] for d in details] for a in arms}
    report, draws = summarize(protocol["rows"], arrays, corrections)
    classes = np.array([r["c"] for r in protocol["rows"]])
    report["per_class_IU"] = {a: {str(c): dict(intersection=int(v[classes == c, 0].sum()), union=int(v[classes == c, 1].sum()))
                                  for c in sorted(set(classes))} for a, v in arrays.items()}
    report["error_buckets_GT_diagnostic"] = dict(definition="8-connected; <10% overlap; FP/FN pixels partitioned",
        mass={b: sum(d["error_mass"][b] for d in details) for b in edit.BUCKETS},
        repaired={a: {b: sum(d["reach"][a][b] for d in details) for b in edit.BUCKETS} for a in arms})
    report["auxiliary_retention"] = {}
    for name in details[0]["losses"]:
        raw = {k: sum(d["losses"][name]["raw"][k] for d in details) for k in edit.ACTIONS}
        kept = {k: sum(d["losses"][name]["kept"][k] for d in details) for k in edit.ACTIONS}
        report["auxiliary_retention"][name] = dict(raw=raw, kept=kept, rates=edit.retention(raw, kept))
    fn, fp = sum(d["native_FN"] for d in details), sum(d["native_FP"] for d in details)
    report["proposal_edit_coverage"] = {}
    for alias, (_, _, arm) in edit.PROPOSALS.items():
        counts = report["corrections_vs_native"][arm]
        report["proposal_edit_coverage"][arm] = dict(
            as_A=dict(recall=edit.ratio(counts["add_TP"], fn), purity=edit.ratio(counts["add_TP"], counts["add_TP"] + counts["add_FP"])),
            as_B=dict(recall=edit.ratio(counts["delete_FP"], fp), purity=edit.ratio(counts["delete_FP"], counts["delete_FP"] + counts["delete_TP"])))
    # Required proposal-matched comparisons use the very same photograph draws.
    groups = photo_groups(protocol["rows"])
    weights = np.stack([np.bincount(d, minlength=int(groups.max()) + 1) for d in draws])[:, groups]
    samples = {a: np.array([metric(v, classes, w) for w in weights]) for a, v in arrays.items() if a.startswith("edit.")}
    report["matched_proposal_contrasts"] = {}
    report["matched_proposal_location_contrasts"] = {}
    for arm, recipe in protocol["recipes"].items():
        if recipe["mode"] == "aux":
            targets = (arm.replace(".aux.", ".noaux."), arm.replace(".aux.", ".native_score.") + ".control")
            destination = report["matched_proposal_contrasts"]
        elif recipe["mode"] == "noaux" and bool(recipe["A"]) != bool(recipe["B"]):
            targets = (arm.replace(".noaux.", ".global_native_count.") + ".control",)
            destination = report["matched_proposal_location_contrasts"]
        else:
            continue
        destination[arm] = {}
        for base in targets:
            delta = arrays[arm][:, 0] / np.maximum(arrays[arm][:, 1], 1) - arrays[base][:, 0] / np.maximum(arrays[base][:, 1], 1)
            destination[arm][base] = dict(gain=report["scores"][arm] - report["scores"][base],
                ci95=np.percentile(samples[arm] - samples[base], [2.5, 97.5]).tolist(),
                per_fold_gain={str(f): metric(arrays[arm][keep], classes[keep]) - metric(arrays[base][keep], classes[keep])
                               for f in sorted({int(r["fold"]) for r in protocol["rows"]})
                               for keep in [np.array([int(r["fold"]) == f for r in protocol["rows"]])]},
                up=int((delta > 1e-12).sum()), down=int((delta < -1e-12).sum()), tie=int((abs(delta) <= 1e-12).sum()))
    report["prediction_seal_sha256"] = sha(args.out / "sealed.json")
    report["protocol"] = {k: protocol[k] for k in ("name", "proposal_arms", "recipes", "configs", "code_sha256", "sources", "seed", "resolution", "dataset", "resource_setting", "novelty")}
    write_json(args.out / "report.json", report)
    write_json(args.out / "episode_metrics.json", details)
    np.save(args.out / "bootstrap_photo_draws.npy", draws)
    write_json(args.out / "completion.json", dict(state="EVALUATED", episodes=241, prediction_seal_sha256=report["prediction_seal_sha256"]))
    print(json.dumps(dict(n=report["n"], native=report["scores"]["native"], rcg=report["scores"]["rcg"], arms=len(arms))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "infer", "evaluate"), default="prepare")
    parser.add_argument("--root", type=Path, help="Shared q/r and original native packet cache root")
    parser.add_argument("--manifest", type=Path, default=REPO / "evidence/local/research_20261005/dev241.json")
    parser.add_argument("--d-run", type=Path, help="Sealed gpu_multilayer_dev241_v1 source")
    parser.add_argument("--rcg-run", type=Path, help="Sealed rcg241 source")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="prepare never initializes CUDA")
    parser.add_argument("--workers", type=int, default=1, help="Model-free inference episode workers, 1..3")
    parser.add_argument("--score-workers", type=int, default=2, help="Independent CPU scoring workers, 1..2")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--proposals-only", action="store_true", help="infer seals the six proposals and stops before the witness baseline; later infer reuses those receipts")
    args = parser.parse_args()
    if args.proposals_only and args.stage != "infer":
        parser.error("--proposals-only applies only to --stage infer")
    if args.workers not in (1, 2, 3) or args.score_workers not in (1, 2) or args.threads < 1:
        parser.error("Require inference workers1..3, scoring workers1..2, positive threads")
    if args.stage != "evaluate" and any(getattr(args, k) is None for k in ("root", "d_run", "rcg_run")):
        parser.error("--root, --d-run and --rcg-run are required for prepare/infer")
    if args.stage == "prepare":
        protocol = prepare(args)
        print(json.dumps(dict(state="PREPARED", n=len(protocol["rows"]), query_truth_opened=False)))
    elif args.stage == "infer":
        infer(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
