#!/usr/bin/env python3
"""Future two-pair producer benchmark/parity check; never starts the4000 job.

The optimization stops at the ACTUAL FoRIS Part1 prefix. The source transform,
reference mask, paired FP32 forward, channel normalization, source gate and
explicit projection remain unchanged. Real full FoRIS/native is a separate
benchmark/control. No expected native is returned by the optimized prefix.

Run only after root releases the current GPU job:
  python scripts/benchmark_part1_producer.py --out outputs/part1_producer_two_pair_v1

Requires sibling run_frozen_subtoken4000.py/run_frozen_subtoken1200.py,
src/ics, and the same existing host/CRF environment. No query GT is opened.
Default: two fixed cached1200 pairs from different folds, one warmup per
producer, three alternating timing repeats, then one separately instrumented
stage pass. All timings/parity evidence are retained; this is not a method run.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ics.experiment import render, sha, unpack
from run_frozen_subtoken1200 import parameters, write
from run_frozen_subtoken4000 import BASIS_SHA, make_host, read_source, reuse_metadata, source_rows


def part1_only(host, support, reference_mask, query):
    """Source foris.py:178-188 prefix, returning q/r/target and NO native mask.

    Source _part1_positional_debias signature is (fmaps_norm, ref_masks, n_refs).
    Keep host.set_reference/set_target and its original P_perp matrix order.
    This function is the future encoder API; it neither runs nor replaces CRF.
    """
    import torch
    import torch.nn.functional as F
    try:
        host.set_reference(support, reference_mask)
        host.set_target(query)
        n_refs = host._ref_images.shape[0]
        images = torch.cat([host._ref_images, host._tgt_image.unsqueeze(0)], dim=0).unsqueeze(0)
        raw = host._extract_features(images)
        normalized = F.normalize(raw, p=2, dim=2)
        deb = host._part1_positional_debias(normalized, host._ref_masks.unsqueeze(1), n_refs)
        if raw.shape != (1, 2, 1024, 64, 64) or raw.dtype != torch.float32 or deb.dtype != torch.float32:
            raise ValueError("Require source one-reference/query paired FP32 features")
        q16 = deb[0, -1].flatten(1).T.half().cpu().numpy()
        r16 = deb[0, 0].flatten(1).T.half().cpu().numpy()
        # Reproduce exporter evidence, not potentially stale host.should_debiass.
        debiased = bool((F.normalize(raw[0], dim=1) - deb[0]).abs().max() > 1e-4)
        cov = F.interpolate(host._ref_masks[0][None, None].float(), (64, 64), mode="area")[0, 0].cpu().numpy()
        return dict(q=q16, r=r16, debiased=debiased, cov=cov, target=host._tgt_image.clone())
    finally:
        host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None


def full_producer(host, producer, support, reference_mask, query):
    """Call the real full source; keep its actual native, score, q/r and gate."""
    import torch.nn.functional as F
    native, got, mask, target = producer.run_foris(host, support, reference_mask, query)
    raw, deb = F.normalize(got["raw"][0], dim=1), got["deb"][0]
    return dict(q=deb[-1].flatten(1).T.half().cpu().numpy(),
                r=deb[0].flatten(1).T.half().cpu().numpy(),
                debiased=bool((raw - deb).abs().max() > 1e-4),
                cov=F.interpolate(mask[None, None].float(), (64, 64), mode="area")[0, 0].cpu().numpy(),
                score=got["score"].float().cpu().numpy(),
                native=np.packbits(native.cpu().numpy()), target=target)


def measure(call):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


@contextmanager
def stage_timers(host):
    """A separate synchronized profiling pass; outputs and source calls unchanged.

    Timings are inclusive when a source stage invokes another timed stage.
    They must not be added together or used as uninstrumented end-to-end time.
    """
    import torch
    names = {"set_reference", "set_target", "_transform", "_extract_features",
             "_debias_features", "_binarize_response", "_finalize_mask"}
    names.update(name for name in dir(host) if name.startswith("_part"))
    names = sorted(name for name in names if callable(getattr(host, name, None)))
    saved, observations = {}, {}
    def wrapped(name, function):
        def invoke(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = function(*args, **kwargs)
            torch.cuda.synchronize()
            observations.setdefault(name, []).append(time.perf_counter() - start)
            return result
        return invoke
    try:
        for name in names:
            saved[name] = (name in host.__dict__, host.__dict__.get(name))
            setattr(host, name, wrapped(name, getattr(host, name)))
        yield observations
    finally:
        for name, (had, old) in saved.items():
            setattr(host, name, old) if had else delattr(host, name)


def tensor_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def compare(prefix, full, cached, source):
    cov, score, _, masks, _ = source
    result = {}
    for role in ("q", "r"):
        expected = cached[role].numpy()
        for label, other in (("full", full[role]), ("cached", expected)):
            result["part1_%s_vs_%s_mismatches" % (role, label)] = int(np.count_nonzero(prefix[role] != other))
        result["full_%s_vs_cached_mismatches" % role] = int(np.count_nonzero(full[role] != expected))
    result.update(gate_part1=prefix["debiased"], gate_full=full["debiased"], gate_cached=bool(cached["debiased"]),
                  gate_all_equal=prefix["debiased"] == full["debiased"] == bool(cached["debiased"]),
                  part1_cov_vs_full_maxdiff=float(np.abs(prefix["cov"] - full["cov"]).max()),
                  part1_cov_vs_source_maxdiff=float(np.abs(prefix["cov"] - cov).max()),
                  full_score_vs_source_maxdiff=float(np.abs(full["score"] - score).max()),
                  actual_full_native_vs_source_mismatched_pixels=int(np.unpackbits(full["native"] ^ masks["native"]).sum()))
    failures = [name for name, value in result.items() if
                ((name.endswith("mismatches") or name.endswith("maxdiff") or name.endswith("mismatched_pixels")) and value != 0)
                or (name == "gate_all_equal" and not value)]
    result["passed"] = not failures
    result["failures"] = failures
    return result


def rcg16_parity(prefix, full, source):
    import torch
    from ics.methods import rcg
    if rcg.CONFIG["lambda"] != 16.0:
        raise ValueError("Require unchanged locked lambda16; no CONFIG mutation")
    cov, score, expected_field, masks, _ = source
    previous_threads = torch.get_num_threads()
    result = {}
    try:
        torch.set_num_threads(1)
        for name, values in (("part1", prefix), ("actual_full", full)):
            start = time.perf_counter()
            field, info = rcg.predict(values["q"], values["r"], cov, score, device="cpu")
            result[name] = dict(field_mismatches=int(np.count_nonzero(field != expected_field)),
                               field_maxdiff=float(np.abs(field - expected_field).max()),
                               stored_mask_mismatched_pixels=int(np.count_nonzero(render(field) != unpack(masks["rcg"]))),
                               seconds=time.perf_counter() - start, solver=info,
                               field_sha256=tensor_hash(field),
                               inputs="producer q/r FP16; unchanged independently sealed cov/score")
    finally:
        torch.set_num_threads(previous_threads)
    result["passed"] = all(not v["field_mismatches"] and not v["stored_mask_mismatched_pixels"]
                           for name, v in result.items() if name != "passed")
    return result


def timing_summary(values):
    return dict(n=len(values), mean_seconds=float(np.mean(values)), median_seconds=float(np.median(values)),
                min_seconds=float(np.min(values)), max_seconds=float(np.max(values)), samples_seconds=values)


def benchmark(a):
    import torch
    from PIL import Image
    frozen = parameters(a.parameter_report)
    mapping, _, _, _, _ = reuse_metadata(a, frozen)
    public, providers = source_rows(a)
    lookup = {r["key"]: r for r in public}
    protected = list(mapping.values())
    protected = [protected[0], next(row for row in protected if row["fold"] != protected[0]["fold"])]
    a.out.mkdir(parents=True, exist_ok=False)
    host, producer, man = make_host(a)
    import inspect
    part1_source = Path(inspect.getsourcefile(host._part1_positional_debias)).resolve()
    config = dict(purpose="two-pair Part1-only producer optimization parity/timing; no method experiment",
                  query_truth_opened=False, pairs=2, sampling="first cached1200 pair and first pair of another fold",
                  basis_sha256=BASIS_SHA, repeats=a.repeats, warmup_per_producer=a.warmup,
                  dtype="source FP32 paired forward, channel normalize, original Part1, q/r half; no autocast",
                  native_control="actual full source producer and independently stored native; prefix returns no native",
                  timing="uninstrumented CUDA-synchronized end-to-end producer times; separate synchronized inclusive stage pass",
                  io="RGB/reference mask/source arrays loaded once outside producer timers; input I/O time recorded separately",
                  fixed_parameters=frozen, runtime_threads=torch.get_num_threads(), providers=providers,
                  source_code_sha256={str(Path(__file__).resolve()): sha(Path(__file__)),
                      str(REPO / "scripts/run_frozen_subtoken4000.py"): sha(REPO / "scripts/run_frozen_subtoken4000.py"),
                      str(REPO / "scripts/run_frozen_subtoken1200.py"): sha(REPO / "scripts/run_frozen_subtoken1200.py"),
                      str(REPO / "src/ics/methods/rcg.py"): sha(REPO / "src/ics/methods/rcg.py"),
                      str(part1_source): sha(part1_source)},
                  host_builder_sha256=sha(producer.__file__), host_manifest_sha256=sha(a.host_manifest),
                  torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0))
    write(a.out / "config.json", config)
    begin, evidence = time.perf_counter(), []
    write(a.out / "state.json", dict(state="RUNNING_TWO_PAIR_BENCHMARK", query_truth_opened=False))
    with torch.inference_mode():
        for pair_index, original in enumerate(protected):
            key = "public%d:%s" % (original["public_batch"], original.get("source_key", original["key"]))
            row = lookup[key]
            start_io = time.perf_counter()
            source = read_source(row)
            cached = torch.load(original["feature_export"], map_location="cpu", weights_only=True)
            with Image.open(Path(man["data_root"]) / row["support"]) as im:
                support = im.convert("RGB")
            with Image.open(Path(man["data_root"]) / row["query"]) as im:
                query = im.convert("RGB")
            reference_path = Path(man["annotation_root"]) / Path(row["support"]).with_suffix(".png")
            with Image.open(reference_path) as im:
                reference_mask = torch.from_numpy((np.asarray(im) == row["c"] + 1).copy())
            io_seconds = time.perf_counter() - start_io
            calls = {"actual_full": lambda: full_producer(host, producer, support, reference_mask, query),
                     "part1_only": lambda: part1_only(host, support, reference_mask, query)}
            for _ in range(a.warmup):
                for call in calls.values():
                    value, _ = measure(call)
                    del value
            times, checks = {name: [] for name in calls}, []
            final = {}
            for repeat in range(a.repeats):
                order = ("actual_full", "part1_only") if (repeat + pair_index) % 2 == 0 else ("part1_only", "actual_full")
                values = {}
                for name in order:
                    values[name], seconds = measure(calls[name])
                    times[name].append(seconds)
                check = compare(values["part1_only"], values["actual_full"], cached, source)
                checks.append(dict(repeat=repeat, order=list(order), **check))
                final = values
                if not check["passed"]:
                    write(a.out / "failure.json", dict(key=key, repeat=repeat, parity=check, query_truth_opened=False))
                    raise RuntimeError("Real Part1/full/cache/native producer parity failed: " + key)
            stages, stage_parity = {}, []
            profiled = {}
            for name in calls:
                with stage_timers(host) as records:
                    profiled[name], elapsed = measure(calls[name])
                stages[name] = dict(instrumented_total_seconds=elapsed,
                                    stages={stage: timing_summary(v) for stage, v in records.items()})
            stage_check = compare(profiled["part1_only"], profiled["actual_full"], cached, source)
            if not stage_check["passed"]:
                write(a.out / "failure.json", dict(key=key, instrumented_parity=stage_check, query_truth_opened=False))
                raise RuntimeError("Instrumented calls altered source parity")
            fixed_rcg = rcg16_parity(final["part1_only"], final["actual_full"], source)
            record = dict(key=key, fold=row["fold"], c=row["c"], source_receipt=source[4],
                          cached_feature_sha256=sha(original["feature_export"]), reference_mask_sha256=sha(reference_path),
                          input_io_seconds=io_seconds, repeats=checks, instrumented_parity=stage_check,
                          timings={name: timing_summary(v) for name, v in times.items()}, stages=stages,
                          fixed_lambda16=fixed_rcg,
                          part1_outputs={role + "_FP16_sha256": tensor_hash(final["part1_only"][role]) for role in ("q", "r")})
            full_times, prefix_times = np.asarray(times["actual_full"]), np.asarray(times["part1_only"])
            record["producer_savings"] = dict(mean_saved_seconds=float(np.mean(full_times - prefix_times)),
                                               mean_full_over_part1=float(np.mean(full_times) / np.mean(prefix_times)),
                                               saved_fraction=float(1 - np.mean(prefix_times) / np.mean(full_times)))
            evidence.append(record)
            write(a.out / "evidence.json", evidence)
            if not fixed_rcg["passed"]:
                raise RuntimeError("Part1/full lambda16 field/mask parity failed; benchmark evidence retained")
            del final, profiled, values, cached
    result = dict(state="TWO_PAIR_PART1_PRODUCER_PARITY_AND_TIMING_COMPLETE", n=2, query_truth_opened=False,
                  evidence=evidence, config_sha256=sha(a.out / "config.json"), evidence_sha256=sha(a.out / "evidence.json"),
                  seconds=time.perf_counter() - begin,
                  limits="two fixed pairs, warmed producer latency; not measured end-to-end4000 throughput or method efficacy",
                  current4000_modified=False, current4000_interrupted=False, native_control_replaced=False)
    write(a.out / "report.json", result)
    write(a.out / "state.json", dict(state=result["state"], n=2, query_truth_opened=False))
    print(json.dumps(dict(state=result["state"], producer_savings={r["key"]: r["producer_savings"] for r in evidence})), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--public", type=Path, default=Path("outputs/claude_official"))
    p.add_argument("--reuse1200", type=Path, default=Path("outputs/frozen_subtoken1200_v1"))
    p.add_argument("--parameter-report", type=Path, default=Path("outputs/claude_subtoken_fresh600/report.json"))
    p.add_argument("--host-manifest", type=Path, default=Path("outputs/claude_official/batch0.json"))
    p.add_argument("--host-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent"))
    p.add_argument("--demo4-root", default="/root/autodl-tmp/demo4")
    p.add_argument("--basis", type=Path, default=Path("/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt"))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    a = p.parse_args()
    if a.out.exists():
        p.error("Use a fresh benchmark directory; no current output/snapshot is modified")
    if a.repeats < 1 or a.warmup < 0:
        p.error("repeats must be positive and warmup nonnegative")
    try:
        benchmark(a)
    except Exception as error:
        if a.out.exists():
            write(a.out / "state.json", dict(state="FAILED", error=repr(error), query_truth_opened=False))
        raise


if __name__ == "__main__":
    main()
