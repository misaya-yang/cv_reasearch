"""One-process FP32 comparison; CPU execution is only a harness smoke test."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import time
import traceback

from takeover_common import *


def run(args):
    configure(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable")
    # All arms are compiled through the same adapter code. Raise only its
    # specialization capacity; never silently fall back from fullgraph.
    torch._dynamo.config.recompile_limit = 64
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "options": vars(args) | {"output_dir": str(args.output_dir)},
              "torch": torch.__version__, "precision": "float32", "tf32": False,
              "tolerance": 5e-5, "compiler": "inductor/fullgraph" if args.mode == "compile" else "eager",
              "scope": "Random fixed weights and synthetic encoded image, complete four-mask/four-IoU decoder; encoder, prompt encoder, final resize, persistence and cross-microbatch concatenation excluded equally",
              "round_contract": "Same fixture; reverse arm order in round 2. These are measurement rounds, not independent seeds.",
              "arms": ARMS, "records": []}
    path = args.output_dir / "report.json"
    atomic_json(path, report)
    model, image, pe, dense, sparse, fixture = make_fixture(
        seed=args.seed, grid=args.grid, tokens=7, batch=args.prompts,
        device=device, dtype=torch.float32)
    report["fixture"] = fixture
    report["hardware"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU smoke only"
    compiled = {}
    try:
        with torch.inference_mode():
            for mb in args.microbatches:
                chunks = tuple(sparse[lo:lo+mb] for lo in range(0, args.prompts, mb))
                for round_index in range(args.rounds):
                    order = ARMS if round_index % 2 == 0 else tuple(reversed(ARMS))
                    for arm in order:
                        started = time.monotonic()
                        label = arm[0]
                        print(f"START mb={mb} round={round_index+1} arm={label}", flush=True)
                        build, eager, plans = arm_functions(arm, model, image, pe, dense, 7)
                        key = (mb, label)
                        if key not in compiled:
                            compiled[key] = torch.compile(eager, backend="inductor", fullgraph=True,
                                                          dynamic=False) if args.mode == "compile" else eager
                        forward = compiled[key]
                        cache = build()

                        def batch(ca):
                            value = None
                            for chunk in chunks:
                                value = None
                                value = forward(chunk, ca)
                            return value

                        synchronize(device)
                        start = time.perf_counter()
                        value = batch(cache)
                        synchronize(device)
                        first_ms = (time.perf_counter()-start)*1000
                        del value
                        checks = []
                        for chunk in chunks:
                            reference = official_predict(model, image, pe, dense, chunk)
                            candidate = forward(chunk, cache)
                            checks.append(compare(reference, candidate, 5e-5))
                            del reference, candidate
                        passed = all(c["passed"] for c in checks)
                        record = {"arm": label, "method": arm[1], "attention": arm[2],
                                  "plans": plans, "microbatch": mb, "round": round_index+1,
                                  "numeric_status": "PASSED_RANDOM_FIXTURE_ONLY" if passed else "FAILED",
                                  "verification": checks, "first_call_ms_excluded": first_ms,
                                  "logical_persistent_cache_bytes": tensor_bytes(cache)}
                        if passed:
                            for _ in range(args.warmup):
                                value = batch(cache)
                                del value
                            record["warm_complete_prompt_batch"] = measure(lambda: batch(cache), device, args.repetitions)
                            cache = None
                            gc.collect()
                            value = batch(build())  # Fresh-cache specialization excluded.
                            del value
                            record["cold_image_including_cache_and_prompt_batch"] = measure(
                                lambda: batch(build()), device, args.repetitions)
                            record["cold_cache_build_including_fixed_PE"] = measure(build, device, args.repetitions)
                        cache = None
                        record["wall_seconds"] = time.monotonic()-started
                        report["records"].append(record)
                        atomic_json(path, report)
                        print(f"DONE mb={mb} round={round_index+1} arm={label} numeric={record['numeric_status']} warm_ms={record.get('warm_complete_prompt_batch',{}).get('median_ms')}", flush=True)
                        del build, eager, forward, batch
                        gc.collect()
        report["status"] = "COMPLETED" if all(r["numeric_status"] == "PASSED_RANDOM_FIXTURE_ONLY" for r in report["records"]) else "COMPLETED_WITH_NUMERIC_FAILURE"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [Path(__file__), ROOT/"tools/takeover_common.py",
                                              ROOT/"sam_shared_decoder/execution_baselines/experimental_methods.py",
                                              ROOT/"research/projection_merge/merged_phase.py"]}
        atomic_json(path, report)
        return 0 if report["status"] == "COMPLETED" else 2
    except BaseException:
        report["status"] = "ERROR"
        report["error"] = traceback.format_exc()
        atomic_json(path, report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    parser.add_argument("--mode", choices=("eager", "compile"), default="compile")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--microbatches", type=lambda s: [int(x) for x in s.split(",")], default=[8,128])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.prompts, args.grid, args.rounds, args.warmup, args.repetitions, args.threads, *args.microbatches) < 1 or max(args.microbatches)>args.prompts:
        parser.error("invalid positive counts/microbatch")
    if args.device == "cpu" and args.mode == "compile":
        parser.error("CPU smoke uses eager; optimizing compilation is a CUDA experiment")
    raise SystemExit(run(args))
