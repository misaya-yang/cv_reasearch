#!/usr/bin/env python3
"""Compose sealed D scores with existing feature caches; never load an encoder.

Inference opens only q/r, source coverage, raw scores and sealed prediction masks.
The complete selected cohort is sealed before the separate CPU evaluation stage.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def sealed_source(path):
    from ics.experiment import load_rows, sha
    path = Path(path)
    seal = json.loads((path / "sealed.json").read_text())
    if sha(path / "manifest.json") != seal["manifest_sha256"]:
        raise ValueError(f"Changed source manifest: {path}")
    return path, seal, {r["key"]: r for r in load_rows(path / "manifest.json")}


def check_row(row, source_rows):
    other = source_rows[row["key"]]
    if any(row[k] != other[k] for k in ("fold", "e", "c", "support", "query")):
        raise ValueError(f"Source cohort identity mismatch: {row['key']}")


def read_d_scores(path):
    """Old combined packets and new score-only packets are distinct schemas."""
    import numpy as np
    with np.load(path, allow_pickle=False) as packet:
        if "multilayer_score" in packet:
            if "multilayer" in packet and packet["multilayer"].ndim != 1:
                raise ValueError("Ambiguous D score schema")
            schema, score_key = "first20_combined_packed_mask_and_raw_score", "multilayer_score"
            fields = {k[:-6]: packet[k].copy() for k in packet.files if k.endswith("_score")
                      and (k.startswith("multilayer") or k == "native_score")}
        elif "multilayer" in packet and packet["multilayer"].ndim == 2:
            schema, score_key = "D241_raw_score_only", "multilayer"
            fields = {k: packet[k].copy() for k in packet.files
                      if (k.startswith("multilayer") or k == "native") and packet[k].ndim == 2}
        else:
            raise ValueError(f"No declared raw D part-4 score in {path}")
        raw = packet[score_key].copy()
    if raw.dtype != np.float32 or raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("Raw D scores must be finite float32 HxW, never packed masks")
    return raw, fields, dict(schema=schema, score_key=score_key,
                             score_origin="multilayer.transition.part4.raw")


class NativeFinalizer:
    """The public FoRIS CRF call at 1024, without a host or encoder constructor."""

    def __init__(self, foris_root, device):
        import torch
        from ics.experiment import sha
        sys.path.insert(0, str(foris_root))
        from utils.data import build_transform
        from utils.refinement import init_crf, crf_refine
        self.device = torch.device(device)
        self.mask_refiner = "crf"
        self.resize_to_orig_size = False
        self.image_size = 1024
        self.transform = build_transform(self.image_size)
        self._crf, self._crf_band_px, self._crf_p_core = init_crf(1024, str(self.device))
        self._crf_refine = crf_refine
        self.receipt = dict(encoder_instances=0, encoder_forwards=0, crf_initializations=1,
                            init_crf_source_call="init_crf(1024, str(device))",
                            finalizer_source_sha256=sha(Path(foris_root) / "models/foris.py"),
                            source_sha256={str(Path(inspect.getfile(fn))): sha(inspect.getfile(fn))
                                           for fn in (build_transform, init_crf, crf_refine)})

    def _finalize_mask(self, mask, query):
        # Exact source _finalize_mask call: input already has target H/W and
        # resize_to_orig_size=False, so both optional resize branches are idle.
        return self._crf_refine(self._crf, self._crf_band_px, self._crf_p_core, query, mask)

    def query(self, path):
        from PIL import Image
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")).to(self.device)


def render(field, device):
    import torch
    import torch.nn.functional as F
    value = torch.as_tensor(field, dtype=torch.float32, device=device)
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError("Expected a finite probability field")
    return (F.interpolate(value[None, None], (1024, 1024), mode="bilinear",
                          align_corners=False)[0, 0] > .5).cpu().numpy()


def normalized(raw):
    return (raw - raw.min()) / max(float(raw.max() - raw.min()), 1e-6)


def infer(args):
    import numpy as np
    import torch
    from ics.experiment import load_rows, load_inputs, sha, unpack
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA matrix algebra requires a visible GPU")
    if args.out.exists():
        raise FileExistsError("Choose a fresh output directory")
    document = json.loads(args.manifest.read_text())
    allowed = ("fold", "e", "c", "key", "support", "query", "batch", "split", "seed",
               "feature_export", "packet_export")
    rows = [{k: r[k] for k in allowed if k in r} for r in load_rows(args.manifest)]
    d_path, d_seal, d_rows = sealed_source(args.score_run or args.baseline_run)
    rcg_path, rcg_seal, rcg_rows = sealed_source(args.rcg_run)
    score_folder = args.score_packets or d_path / "packets"
    methods = list(args.methods)
    if args.include_latent_native and "latent_native" not in methods:
        methods.append("latent_native")
    modules = {name: importlib.import_module("ics.methods." + name) for name in methods}
    score_origin = {"native": "foris.part4.raw", "multilayer": "multilayer.transition.part4.raw",
                    "multilayer.delta.control": "multilayer.delta.part4.raw"}[args.source_arm]
    if "composition" in modules:
        modules["composition"].CONFIG = dict(modules["composition"].CONFIG, score_origin=score_origin)
    finalizer = NativeFinalizer(args.foris_root or document["foris_root"], args.device)
    start = time.monotonic()
    args.out.mkdir(parents=True)
    for directory in ("predictions", "pre", "fields"):
        (args.out / directory).mkdir()
    write_json(args.out / "manifest.json", rows)
    protocol = dict(state="INFERENCE_ONLY", methods={k: dict(config=v.CONFIG, source_sha256=sha(v.__file__))
                    for k, v in modules.items()}, finalizer=finalizer.receipt, device=args.device,
                    threads=args.threads, source_manifest_sha256=sha(args.manifest),
                    cache_root=str(args.root.resolve()), data_root=document["data_root"],
                    d_run=str(d_path.resolve()), rcg_run=str(rcg_path.resolve()),
                    score_packets=str(score_folder.resolve()), score_origin=score_origin, source_arm=args.source_arm,
                    encoder_instances=0, new_encoder_forwards=0, query_gt_usage="evaluation only after complete cohort seal",
                    development_only=True, parameter_selection="fixed before outcomes",
                    matrix_device=args.device, evaluation_device="cpu",
                    note="Composition wiring; not an additional independent research mechanism")
    write_json(args.out / "protocol.json", protocol)
    seal = dict(state="ALL_PREDICTIONS_SEALED", manifest_sha256=sha(args.out / "manifest.json"),
                protocol_sha256=sha(args.out / "protocol.json"), source_seals={str(p / "sealed.json"): sha(p / "sealed.json")
                for p in (d_path, rcg_path)}, predictions={}, pre={}, fields={}, inputs={}, query_labels_opened=False)
    with torch.inference_mode(), (args.out / "episodes.jsonl").open("w") as stream:
        for index, row in enumerate(rows):
            started = time.monotonic()
            key = row["key"]
            check_row(row, d_rows)
            check_row(row, rcg_rows)
            (q, r, cov, native_score), receipt = load_inputs(args.root, row)
            score_path = score_folder / f"{key}.npz"
            d_score, source_fields, score_info = read_d_scores(score_path)
            if args.source_arm == "native":
                d_score = native_score
            elif args.source_arm != "multilayer":
                d_score = source_fields[args.source_arm]
            score_info.update(score_key=args.source_arm, score_origin=score_origin)
            if np.shape(d_score) != np.shape(native_score):
                raise ValueError("Cached D/native score token grids differ")
            d_prediction = d_path / "predictions" / f"{key}.npz"
            rcg_prediction = rcg_path / "predictions" / f"{key}.npz"
            for path, source in ((d_prediction, d_seal), (rcg_prediction, rcg_seal)):
                if sha(path) != source["predictions"][key]:
                    raise ValueError(f"Changed sealed source prediction: {path}")
            with np.load(d_prediction, allow_pickle=False) as packet:
                masks = {k: unpack(packet[k]) for k in packet.files if k == "native" or k.startswith("multilayer")}
            if "native" not in masks or "multilayer" not in masks or "multilayer.control" not in masks:
                raise ValueError("Require sealed complete native, D and D concat-control masks")
            with np.load(rcg_prediction, allow_pickle=False) as packet:
                masks["rcg"] = unpack(packet["rcg"])
            source_packet_sha = rcg_seal.get("inputs", {}).get(key, {}).get("packet_sha256")
            if source_packet_sha is None or source_packet_sha != receipt["packet_sha256"]:
                raise ValueError("RCG sealed cache packet differs from current input")
            query_path = Path(document["data_root"]) / row["query"]
            query_tensor = finalizer.query(query_path)
            receipt.update(score_packet=str(score_path.resolve()), score_packet_sha256=sha(score_path),
                           d_prediction=str(d_prediction.resolve()), d_prediction_sha256=sha(d_prediction),
                           rcg_prediction=str(rcg_prediction.resolve()), rcg_prediction_sha256=sha(rcg_prediction),
                           query_image=str(query_path.resolve()), query_image_sha256=sha(query_path), **score_info)
            seal["inputs"][key] = receipt
            fields, timings, audits = {}, {}, {}
            pre = {k: render(normalized(v), args.device) for k, v in source_fields.items()}
            pre["native"] = render(normalized(native_score), args.device)
            pre["rcg"] = masks["rcg"]
            if "multilayer.delete_only" in masks:
                pre["multilayer.delete_only"] = pre["native"] & pre["multilayer"]
            if "multilayer.add_only" in masks:
                pre["multilayer.add_only"] = pre["native"] | pre["multilayer"]
            extras = dict(host=finalizer, query_tensor=query_tensor, **score_info)
            for name, module in modules.items():
                functions = [(name, module.predict), (name + ".control", module.control)]
                functions += [(name + "." + k + ".control", fn)
                              for k, fn in getattr(module, "additional_controls", {}).items()]
                for arm, function in functions:
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    begin = time.monotonic()
                    field, info = function(q, r, cov, d_score, device=args.device, extras=extras)
                    fields[arm] = field
                    pre[arm] = render(field, args.device)
                    # Every generated arm has the same probability renderer and native CRF.
                    masks[arm] = finalizer._finalize_mask(torch.as_tensor(pre[arm], device=finalizer.device),
                                                         query_tensor[None]).bool().cpu().numpy()
                    if masks[arm].shape != (1024, 1024):
                        raise ValueError("Native CRF must return a complete 1024 mask")
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    info["native_finalization_pending"] = False
                    audits[arm] = info
                    timings[arm] = dict(seconds=time.monotonic() - begin,
                                       peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                                       peak_rss_bytes=peak_rss_bytes())
            replay = masks.get("composition.unary.control")
            if replay is not None and not np.array_equal(replay, masks[args.source_arm]):
                write_json(args.out / "audit_failure.json", dict(key=key, kind="D_native_CRF_replay_mismatch",
                           mismatched_pixels=int(np.count_nonzero(replay != masks[args.source_arm]))))
                raise RuntimeError("Cached D unary did not reproduce its sealed complete mask")
            for directory, values in (("predictions", masks), ("pre", pre), ("fields", fields)):
                destination = args.out / directory / f"{key}.npz"
                arrays = values if directory == "fields" else {arm: np.packbits(mask) for arm, mask in values.items()}
                np.savez_compressed(destination, **arrays)
                seal[directory][key] = sha(destination)
            record = dict(key=key, score_source=score_info, seconds=time.monotonic() - started,
                          timings=timings, receipts=audits, peak_rss_bytes=peak_rss_bytes())
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(json.dumps(dict(completed=index + 1, total=len(rows), key=key,
                                  seconds=record["seconds"], source_arm=args.source_arm)), flush=True)
            del q, r, query_tensor, masks, pre, fields
    write_json(args.out / "sealed.json", seal)
    write_json(args.out / "completion.json", dict(state="COMPLETED", episodes=len(rows),
               elapsed_seconds=time.monotonic() - start, peak_rss_bytes=peak_rss_bytes()))


def evaluate(args):
    import numpy as np
    from ics.experiment import sha, unpack, packet, summarize, metric, photo_groups
    started = time.monotonic()
    seal = json.loads((args.out / "sealed.json").read_text())
    for name in ("manifest", "protocol"):
        if sha(args.out / f"{name}.json") != seal[f"{name}_sha256"]:
            raise ValueError(f"Changed sealed {name}")
    for path, digest in seal["source_seals"].items():
        if sha(path) != digest:
            raise ValueError(f"Changed source seal: {path}")
    rows = json.loads((args.out / "manifest.json").read_text())
    stage_arrays, stage_corrections, details = {k: {} for k in ("final", "pre")}, {k: {} for k in ("final", "pre")}, []
    for row in rows:
        key = row["key"]
        receipt = seal["inputs"][key]
        if sha(packet(args.root, row)) != receipt["packet_sha256"]:
            raise ValueError("Changed evaluation cache packet")
        for role in ("score_packet", "d_prediction", "rcg_prediction", "query_image"):
            if sha(receipt[role]) != receipt[role + "_sha256"]:
                raise ValueError(f"Changed inference input: {role}")
        for folder in ("predictions", "pre", "fields"):
            if sha(args.out / folder / f"{key}.npz") != seal[folder][key]:
                raise ValueError(f"Changed sealed {folder}: {key}")
        with np.load(packet(args.root, row), allow_pickle=False) as source:
            truth, cached_native, cached_pre = (unpack(source[k]) for k in ("truth", "native", "pre"))
        for stage, folder, baseline in (("final", "predictions", cached_native), ("pre", "pre", cached_pre)):
            with np.load(args.out / folder / f"{key}.npz", allow_pickle=False) as source:
                masks = {k: unpack(source[k]) for k in source.files}
            if not np.array_equal(masks["native"], baseline):
                raise ValueError(f"Native {stage} replay mismatch: {key}")
            for arm, mask in masks.items():
                iu = [int((mask & truth).sum()), int((mask | truth).sum())]
                stage_arrays[stage].setdefault(arm, []).append(iu)
                add, delete = mask & ~baseline, baseline & ~mask
                corr = dict(key=key, c=row["c"], fold=row["fold"], batch=str(row.get("batch", "unspecified")),
                            add_TP=int((add & truth).sum()), delete_FP=int((delete & ~truth).sum()),
                            delete_TP=int((delete & truth).sum()), add_FP=int((add & ~truth).sum()))
                stage_corrections[stage].setdefault(arm, []).append(corr)
                details.append(dict(stage=stage, arm=arm, intersection=iu[0], union=iu[1], **corr))
    for stage in ("final", "pre"):
        arrays = {k: np.asarray(v, dtype=np.int64) for k, v in stage_arrays[stage].items()}
        if any(len(v) != len(rows) for v in arrays.values()):
            raise ValueError(f"Unpaired {stage} cohort")
        report, draws = summarize(rows, arrays, stage_corrections[stage])
        # Existing summary includes native/RCG and every *.control. Also pair
        # every arm explicitly against frozen D itself, without changing it.
        classes = np.array([r["c"] for r in rows])
        groups = photo_groups(rows)
        count = int(groups.max()) + 1
        weights = np.stack([np.bincount(d, minlength=count) for d in draws])[:, groups]
        d_samples = np.array([metric(arrays["multilayer"], classes, w) for w in weights])
        for arm, values in arrays.items():
            if arm == "multilayer":
                continue
            samples = np.array([metric(values, classes, w) for w in weights])
            delta = values[:, 0] / np.maximum(values[:, 1], 1) - arrays["multilayer"][:, 0] / np.maximum(arrays["multilayer"][:, 1], 1)
            report["contrasts"][arm]["multilayer"] = dict(gain=report["scores"][arm] - report["scores"]["multilayer"],
                ci95=np.percentile(samples - d_samples, [2.5, 97.5]).tolist(), up=int((delta > 1e-12).sum()),
                down=int((delta < -1e-12).sum()), tie=int((np.abs(delta) <= 1e-12).sum()))
        for field in ("folds", "batchs"):
            for entry in report[field].values():
                entry["gains_vs_baselines"] = {base: {arm: value - entry["scores"][base]
                    for arm, value in entry["scores"].items()} for base in entry["scores"]
                    if base in ("native", "rcg", "multilayer") or base.endswith(".control")}
        report.update(stage=stage, prediction_seal_sha256=sha(args.out / "sealed.json"),
                      native_replay_mismatched_pixels=0, evaluation_device="cpu")
        write_json(args.out / ("report.json" if stage == "final" else "pre_report.json"), report)
        if stage == "final":
            np.save(args.out / "bootstrap_photo_draws.npy", draws)
            print(json.dumps(report["scores"], indent=2), flush=True)
    write_json(args.out / "episode_metrics.json", details)
    write_json(args.out / "evaluation_completion.json", dict(state="COMPLETED", elapsed_seconds=time.monotonic() - started,
                                                             peak_rss_bytes=peak_rss_bytes(), device="cpu"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer", "evaluate", "all"))
    parser.add_argument("--root", "--cache-root", dest="root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--score-run", type=Path)
    source.add_argument("--score-packets", type=Path)
    parser.add_argument("--baseline-run", type=Path, help="Sealed complete D run when --score-packets is used")
    parser.add_argument("--rcg-run", type=Path)
    parser.add_argument("--foris-root", type=Path)
    parser.add_argument("--source-arm", choices=("native", "multilayer", "multilayer.delta.control"),
                        default="multilayer", help="Frozen unary-source comparison; inference equations unchanged")
    parser.add_argument("--methods", nargs="+", choices=("composition", "latent_native"), default=["composition"])
    parser.add_argument("--include-latent-native", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.stage in ("infer", "all"):
        if args.manifest is None or args.rcg_run is None or (args.score_run is None and (args.score_packets is None or args.baseline_run is None)):
            parser.error("Inference requires --manifest, --rcg-run and --score-run (or --score-packets plus --baseline-run)")
        infer(args)
    if args.stage in ("evaluate", "all"):
        evaluate(args)


if __name__ == "__main__":
    main()
