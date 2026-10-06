#!/usr/bin/env python3
"""Stream frozen covariance evidence into the existing graph-only objective.

CPU inference reads no query annotations or native/truth packet member. Parent
atomic fields are provisional until the parent's complete prefinal seal binds
every consumed field and mathematical input. Native CUDA CRF is a separate phase.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ics.experiment import load_rows, packet, sha, unpack
import run_part2_evidence as shared

ARMS = ("native.cache.control", "query_covariance.graph", "query_isotropic.graph.control",
        "query_covariance.unary.control", "query_isotropic.unary.control")
SOURCE_ARMS = ("native.cache.control", "query_covariance", "query_isotropic.control")
INPUT_NAMES = ("feature_sha256", "packet_sha256", "query_rgb_sha256",
               "reference_rgb_sha256", "reference_mask_sha256")
MEMORY_BUDGET = int(1.5 * 1024**3)
GRAPH_COMPUTE_SLOTS = 2


def start_worker(config, graph_gate):
    shared.start_worker(config)
    shared.WORKER["graph_gate"] = graph_gate
    if sys.platform == "linux":
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
        shared.WORKER["malloc_trim"] = trim


def render(field):
    """Execute the unmodified source binarizer without loading unrelated stages.

    A full models.foris import also loads sklearn/torchvision in every worker.
    Extracting this self-independent method from the pinned class source keeps
    its exact function body and reduces the three-worker resident memory.
    """
    import torch
    import torch.nn.functional as F
    if "binarizer" not in shared.WORKER:
        config = shared.WORKER["config"]
        path = Path(config["foris_root"]) / "models/foris.py"
        if shared.native_sources(config["foris_root"]) != config["native_source"]:
            raise ValueError("Native source changed before binarization")
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FoRIS")
        function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_binarize_response")
        if [arg.arg for arg in function.args.args] != ["self", "score_hw"] or [arg.arg for arg in function.args.kwonlyargs] != ["target_hw"]:
            raise ValueError("Source binarizer signature changed")
        if any(isinstance(node, ast.Name) and node.id == "self"
               for statement in function.body for node in ast.walk(statement)):
            raise ValueError("Source binarizer requires a host; constructor-free extraction is invalid")
        namespace = dict(torch=torch, F=F)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        shared.WORKER["binarizer"] = namespace["_binarize_response"]
    value = torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))
    return shared.WORKER["binarizer"](None, value, target_hw=(1024, 1024)).numpy()


def literal_config(path):
    """Read the locked literal CONFIG without importing torch in the controller."""
    tree = ast.parse(Path(path).read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets))
    value = assignment.value
    if isinstance(value, ast.Dict):
        return ast.literal_eval(value)
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != "dict" or value.args:
        raise ValueError("Expected locked literal CONFIG")
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in value.keywords}


def process_start(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        return int(text[text.rfind(")") + 2:].split()[19])
    except FileNotFoundError:
        return None


def pss_bytes(pids):
    total = 0
    for pid in pids:
        try:
            lines = Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()
        except FileNotFoundError:
            continue
        total += next(int(line.split()[1]) * 1024 for line in lines if line.startswith("Pss:"))
    return total


def parent_status(config):
    source = Path(config["parent_run"])
    if sha(source / "config.json") != config["parent_config_sha256"]:
        raise ValueError("Parent config changed; no automatic new variant is allowed")
    if sha(source / "manifest.json") != config["parent_manifest_sha256"]:
        raise ValueError("Parent episode manifest changed")
    frozen = source / "prefinal_sealed.json"
    seal_path = frozen if frozen.exists() else source / "sealed.json"
    seal = json.loads(seal_path.read_text())
    if seal["config_sha256"] != config["parent_config_sha256"] or seal["manifest_sha256"] != config["parent_manifest_sha256"]:
        raise ValueError("Parent seal/config/manifest binding failed")
    state = seal.get("state", "")
    if "FAILED" in state or "HELD" in state:
        raise RuntimeError(f"Parent producer {state}; preserve consumer partial artifacts, do not substitute a variant")
    complete = state == "ALL_PREFINALS_SEALED"
    if not complete and process_start(config["parent_pid"]) != config["parent_startticks"]:
        raise RuntimeError("Bound parent PID/startticks is no longer live and its prefinal run is unsealed")
    return complete, seal, seal_path


def verify_parent_sources(source, parent):
    if shared.native_sources(source / "native_source") != parent["native_source"]:
        raise ValueError("Parent native source snapshot changed")
    for name, digest in parent["method_sources"].items():
        if sha(source / "method_source" / f"{name}.py") != digest:
            raise ValueError(f"Parent mathematical/runner source changed: {name}")
    if shared.native_sources(parent["foris_root"]) != parent["native_source"]:
        raise ValueError("Current native source differs from parent snapshot")
    if sha(parent["projection_basis"]) != parent["projection_basis_sha256"]:
        raise ValueError("Parent positional basis changed")


def input_paths(config, row):
    root, key = Path(config["root"]), row["key"]
    return dict(feature_sha256=root / row.get("feature_export", f"cache/evidence_v1/feat/{key}.pt"),
                packet_sha256=packet(root, row),
                query_rgb_sha256=Path(config["data_root"]) / row["query"],
                reference_rgb_sha256=Path(config["data_root"]) / row["support"],
                reference_mask_sha256=Path(config["annotation_root"]) / Path(row["support"]).with_suffix(".png"))


def infer_episode(row):
    import torch
    from ics.methods import mean_graph
    config, key = shared.WORKER["config"], row["key"]
    source, out = Path(config["parent_run"]), Path(config["out"])
    begin = time.monotonic()
    paths = input_paths(config, row)
    inputs = {name: sha(path) for name, path in paths.items()}
    source_fields, source_pre = source / "fields" / f"{key}.npz", source / "prefinal" / f"{key}.npz"
    inputs.update(parent_fields_sha256=sha(source_fields), parent_prefinal_sha256=sha(source_pre),
                  parent_config_sha256=config["parent_config_sha256"])
    with np.load(source_fields, allow_pickle=False) as saved:
        values = {f"{arm}.{name}": saved[f"{arm}.{name}"].astype(np.float32)
                  for arm in SOURCE_ARMS for name in ("s2", "score")}
    with np.load(source_pre, allow_pickle=False) as saved:
        native_pre = saved[ARMS[0]].copy()
    if sha(source_fields) != inputs["parent_fields_sha256"] or sha(source_pre) != inputs["parent_prefinal_sha256"]:
        raise ValueError("Parent atomic artifact changed during read")
    if any(field.shape != (64, 64) or not np.isfinite(field).all() for field in values.values()):
        raise ValueError("Parent evidence must contain finite 64x64 native-tail and unary fields")
    # Parity is with parent native replay, never the official native mask/truth.
    native_parity = int((render(values[f"{ARMS[0]}.score"]) != unpack(native_pre)).sum())
    if native_parity:
        raise ValueError(f"Parent native score/pre binarizer parity failed: {key}, {native_parity} pixels")
    cache = torch.load(paths["feature_sha256"], map_location="cpu", weights_only=True)
    q, r = cache["q"], cache["r"]
    if any(tuple(x.shape) != (4096, 1024) or not torch.isfinite(x).all() for x in (q, r)):
        raise ValueError("Require finite retained [4096,1024] q/r")
    with np.load(paths["packet_sha256"], allow_pickle=False) as saved:
        cov = saved["cov"].astype(np.float32)  # Only approved reference coverage member.
    if cov.shape != (64, 64) or not np.isfinite(cov).all():
        raise ValueError("Invalid reference coverage")
    prefinal, fields = {ARMS[0]: native_pre}, {f"{ARMS[0]}.score": values[f"{ARMS[0]}.score"]}
    timings, solvers = {}, {}
    for arm, parent_arm in zip(ARMS[1:3], SOURCE_ARMS[1:]):
        started = time.monotonic()
        with shared.WORKER["graph_gate"]:
            acquired = time.monotonic()
            z, solvers[arm] = mean_graph.control(q, r, cov, values[f"{parent_arm}.score"], device="cpu")
            solved = time.monotonic()
            if "malloc_trim" in shared.WORKER:
                shared.WORKER["malloc_trim"](0)
        timings[f"{arm}.resource_wait_seconds"] = acquired - started
        timings[f"{arm}.graph_solve_seconds"] = solved - acquired
        if solvers[arm]["alpha"] != 0 or solvers[arm]["graph_lambda"] != 16:
            raise ValueError("Graph-only locked alpha/lambda contract changed")
        fields[f"{arm}.score"] = z
        fields[f"{arm}.source_score"] = values[f"{parent_arm}.score"]
        prefinal[arm] = np.packbits(render(z))
        timings[arm] = time.monotonic() - started
    for arm, parent_arm in zip(ARMS[3:], SOURCE_ARMS[1:]):
        started = time.monotonic()
        unary = values[f"{parent_arm}.s2"]
        fields[f"{arm}.score"] = unary
        prefinal[arm] = np.packbits(render(unary))
        timings[arm] = time.monotonic() - started
    for folder, data in (("prefinal", prefinal), ("fields", fields)):
        shared.write_npz(out / folder / f"{key}.npz", data)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(key=key, inputs=inputs, graph_metadata=solvers, timings=timings,
                parent_native_pre_changed_pixels=native_parity, elapsed_seconds=time.monotonic() - begin,
                worker_pid=os.getpid(), peak_worker_rss_bytes=int(rss * (1024 if sys.platform == "linux" else 1)),
                **{folder: sha(out / folder / f"{key}.npz") for folder in ("prefinal", "fields")})


def final_parent_binding(config, rows, audit):
    complete, seal, seal_path = parent_status(config)
    if not complete:
        raise ValueError("Parent complete prefinal seal is required before consumer seal")
    source = Path(config["parent_run"])
    verify_parent_sources(source, config["parent_config"])
    shared.verify_prefinal(source, seal)
    for row in rows:
        key, receipt = row["key"], audit[row["key"]]
        for group, input_name in (("fields", "parent_fields_sha256"), ("prefinal", "parent_prefinal_sha256")):
            if receipt["inputs"][input_name] != seal[group][key] or sha(source / group / f"{key}.npz") != seal[group][key]:
                raise ValueError(f"Consumed parent {group} differs from its final seal: {key}")
        for name in INPUT_NAMES:
            if receipt["inputs"][name] != seal["inputs"][key][name]:
                raise ValueError(f"Consumer/parent mathematical input differs: {key}.{name}")
    return dict(prefinal_seal_sha256=sha(seal_path), audit_sha256=seal["audit_sha256"],
                config_sha256=config["parent_config_sha256"], manifest_sha256=config["parent_manifest_sha256"],
                math_sources=config["parent_config"]["method_sources"], input_hashes_bound=True,
                consumed_fields_bound=True, episodes=len(rows))


def infer(args):
    source = args.parent.resolve()
    parent = json.loads((source / "config.json").read_text())
    rows = load_rows(source / "manifest.json")
    if args.keys:
        keys = args.keys.split(",")
        lookup = {row["key"]: row for row in rows}
        if len(set(keys)) != len(keys) or any(key not in lookup for key in keys):
            raise ValueError("Smoke keys must be unique parent manifest members")
        rows = [lookup[key] for key in keys]
    if len(rows) != args.expected or not rows:
        raise ValueError("Unexpected parent episode count")
    if args.out.exists():
        raise FileExistsError("Use a fresh output directory; partial outputs are never replaced")
    if tuple(parent["arms"]) != SOURCE_ARMS or not parent.get("native_host_class"):
        raise ValueError("Require the fixed covariance/isotropic full native-tail producer")
    verify_parent_sources(source, parent)
    graph_path = PROJECT / "src/ics/methods/mean_graph.py"
    graph = literal_config(graph_path)
    if graph["query_k"] != 20 or graph["graph_lambda"] != 16 or graph["fidelity_floor"] != .1:
        raise ValueError("Locked graph constants changed")
    method_files = dict(runner=Path(__file__).resolve(), shared_runner=Path(shared.__file__).resolve(),
                        native_binarizer_runner=PROJECT / "scripts/run_middle_rcg.py", mean_graph=graph_path,
                        rcg=PROJECT / "src/ics/methods/rcg.py", experiment=PROJECT / "src/ics/experiment.py",
                        native_basis=PROJECT / "src/ics/native_basis.py")
    hashes = {name: sha(path) for name, path in method_files.items()}
    config = dict(root=parent["root"], out=str(args.out.resolve()), parent_run=str(source),
                  parent_config=parent, parent_config_sha256=sha(source / "config.json"),
                  parent_manifest_sha256=sha(source / "manifest.json"), parent_pid=args.parent_pid,
                  parent_startticks=args.parent_startticks,
                  **{name: parent[name] for name in ("foris_root", "data_root", "annotation_root", "projection_basis",
                                                    "projection_basis_sha256", "native_source")},
                  workers=args.workers, threads=1, arms=list(ARMS), native_arm=ARMS[0],
                  primary=ARMS[1], primary_baseline=ARMS[0], official_baseline="native",
                  cpu_phase="query_covariance_graph", native_host_class=True, decoder_mode="native",
                  gpu_phase="native_CUDA_CRF_only", method_sources=hashes, graph_config=graph,
                  graph_alpha=0, graph_lambda=16, graph_query_k=20, fidelity_floor=.1,
                  objective="min_s (s-y)^T H (s-y) + 16 s^T L s; y=parent native-tail score minmax, H=normalized(.1+abs(2y-1))",
                  source_prior="new upstream covariance/isotropic ordering plus existing graph-only prior; no reference reranking",
                  unary_controls="parent covariance/isotropic Part2 s2 alone, same native binarizer/CRF; no new covariance or graph compute",
                  finalizer="source native minmax/bilinear/threshold binarizer then shared native CUDA CRF",
                  query_labels_in_inference=False, approved_packet_members=["cov"], encoder_forwards=0,
                  memory_budget_bytes=MEMORY_BUDGET,
                  graph_compute_slots=GRAPH_COMPUTE_SLOTS,
                  resource_policy="3 worker processes, one thread each; at most two simultaneous canonical dense graph calls; reclaim unused malloc pages",
                  memory_measure="aggregate proportional resident set size (PSS); shared libraries counted proportionally",
                  failure_policy="parent failure or changed provenance holds partial output; no new variant or automatic retry")
    parent_status(config)
    for folder in ("predictions", "prefinal", "fields", "native_source/models", "native_source/utils", "method_source"):
        (args.out / folder).mkdir(parents=True, exist_ok=True)
    for name in shared.NATIVE_FILES:
        shutil.copyfile(source / "native_source" / name, args.out / "native_source" / name)
    for name, path in method_files.items():
        shutil.copyfile(path, args.out / "method_source" / f"{name}.py")
        if sha(args.out / "method_source" / f"{name}.py") != hashes[name]:
            raise ValueError("Consumer source changed while snapshotting")
    shared.write_json(args.out / "native_source_digest.json", parent["native_source"])
    shared.write_json(args.out / "manifest.json", rows)
    shared.write_json(args.out / "config.json", config)
    seal = dict(state="STREAMING_PARENT_FIELDS", manifest_sha256=sha(args.out / "manifest.json"),
                config_sha256=sha(args.out / "config.json"),
                native_source_snapshot_sha256=parent["native_source"]["sha256"],
                predictions={}, prefinal={}, fields={}, inputs={}, query_labels_opened=False, arms=list(ARMS))
    shared.write_json(args.out / "sealed.json", seal)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    audit, remaining, running = {}, list(rows), {}
    started, last_status, peak_pss = time.monotonic(), 0., 0
    try:
        context = mp.get_context("spawn")
        graph_gate = context.Semaphore(min(args.workers, GRAPH_COMPUTE_SLOTS))
        with context.Pool(args.workers, initializer=start_worker, initargs=(config, graph_gate)) as pool:
            while remaining or running:
                if time.monotonic() - last_status >= 3:
                    parent_complete, _, _ = parent_status(config)
                    live_pids = [os.getpid()] + [worker.pid for worker in pool._pool]
                    peak_pss = max(peak_pss, pss_bytes(live_pids))
                    if peak_pss > MEMORY_BUDGET:
                        raise MemoryError(f"Consumer aggregate PSS {peak_pss} exceeds {MEMORY_BUDGET}; preserve partial outputs")
                    last_status = time.monotonic()
                for key, result in list(running.items()):
                    if not result.ready():
                        continue
                    receipt = result.get()
                    audit[key] = receipt
                    for group in ("prefinal", "fields", "inputs"):
                        seal[group][key] = receipt[group]
                    del running[key]
                    shared.write_json(args.out / "progress.json", dict(completed=len(audit), total=len(rows),
                                      running=len(running), remaining=len(remaining), peak_job_pss_bytes=peak_pss))
                    print(json.dumps(dict(completed=len(audit), total=len(rows), key=key,
                                          seconds=receipt["elapsed_seconds"], peak_job_pss_bytes=peak_pss)), flush=True)
                for row in list(remaining):
                    if len(running) >= args.workers:
                        break
                    key = row["key"]
                    if (source / "fields" / f"{key}.npz").exists() and (source / "prefinal" / f"{key}.npz").exists():
                        running[key] = pool.apply_async(infer_episode, (row,))
                        remaining.remove(row)
                if parent_complete and remaining and not running:
                    raise ValueError("Sealed parent lacks a required atomic case file")
                time.sleep(.2 if running else 2)
        while not parent_status(config)[0]:
            if seal["state"] != "WAITING_PARENT_SEAL":
                seal["state"] = "WAITING_PARENT_SEAL"
                shared.write_json(args.out / "sealed.json", seal)
                print("WAITING_PARENT_SEAL", flush=True)
            time.sleep(3)
        binding = final_parent_binding(config, rows, audit)
        if {name: sha(path) for name, path in method_files.items()} != hashes:
            raise ValueError("Consumer source changed during inference")
        shared.write_json(args.out / "audit.json", dict(episodes=audit, elapsed_seconds=time.monotonic() - started,
                          parent_binding=binding, peak_job_pss_bytes=peak_pss, memory_budget_bytes=MEMORY_BUDGET,
                          encoder_forwards=0, query_labels_opened=False, complete_method_result=False))
        seal.update(state="ALL_PREFINALS_SEALED", audit_sha256=sha(args.out / "audit.json"), parent_binding=binding)
    except BaseException as error:
        seal.update(state="INFERENCE_FAILED", error=f"{type(error).__name__}: {error}", parent_run=str(source))
        shared.write_json(args.out / "sealed.json", seal)
        raise
    shared.write_json(args.out / "sealed.json", seal)
    shared.write_json(args.out / "prefinal_sealed.json", seal)
    print(json.dumps(dict(state=seal["state"], episodes=len(rows), arms=list(ARMS), peak_job_pss_bytes=peak_pss)), flush=True)


def check(args):
    """Read-only constructor-free native-mask and provenance preflight."""
    source = args.parent.resolve()
    parent = json.loads((source / "config.json").read_text())
    verify_parent_sources(source, parent)
    rows = load_rows(source / "manifest.json")
    row = next(row for row in rows if row["key"] == args.key)
    config = dict(parent, parent_run=str(source))
    shared.start_worker(config)
    with np.load(source / "fields" / f"{args.key}.npz", allow_pickle=False) as saved:
        g = saved[f"{ARMS[0]}.score"].astype(np.float32)
    with np.load(source / "prefinal" / f"{args.key}.npz", allow_pickle=False) as saved:
        pre = unpack(saved[ARMS[0]])
    changed = int((render(g) != pre).sum())
    if changed:
        raise ValueError("Parent native source binarizer parity failed")
    print(json.dumps(dict(key=row["key"], native_pre_changed_pixels=changed,
                         parent_config_sha256=sha(source / "config.json"),
                         native_source_snapshot_sha256=parent["native_source"]["sha256"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser("infer")
    replay.add_argument("--parent", type=Path, required=True)
    replay.add_argument("--out", type=Path, required=True)
    replay.add_argument("--parent-pid", type=int, required=True)
    replay.add_argument("--parent-startticks", type=int, required=True)
    replay.add_argument("--expected", type=int, required=True)
    replay.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    replay.add_argument("--keys", help="Explicit parent keys for execution checks")
    preflight = commands.add_parser("check")
    preflight.add_argument("--parent", type=Path, required=True)
    preflight.add_argument("--key", required=True)
    finalizer = commands.add_parser("finalize")
    finalizer.add_argument("--out", type=Path, required=True)
    finalizer.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    {"infer": infer, "check": check, "finalize": shared.finalize}[args.command](args)


if __name__ == "__main__":
    main()
