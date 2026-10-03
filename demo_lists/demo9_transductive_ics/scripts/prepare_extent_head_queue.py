#!/usr/bin/env python3
"""Write the finite plan for scripts/experiment_resource_guard.py: build the inputs of the extent head, replay the
same inputs for the test episodes, fit and score the heads.

CPU only. Run on the server after scripts/cpu_checks_extent_head.sh has passed, then
  python scripts/experiment_resource_guard.py --plan results/extent_head_v0/plan.json \
      --state-file results/extent_head_v0/guard.json --preflight-only
and, to run, the same command with `--run` (add `--allow-shutdown` only if the instance should power off at the end).
"""
import argparse
import hashlib
import json
from pathlib import Path

ENV_PATH = ("/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:"
            "/root/autodl-tmp/demo8_local_verification/runtime/extensions")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--python", default="/root/miniconda3/bin/python")
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--test-manifest", type=Path, required=True)
    p.add_argument("--extent-run", type=Path, required=True, help="finished extent run: episodes.jsonl and packets")
    p.add_argument("--evidence-cache", type=Path, required=True, help="cache of evidence_cache.py: feat/ and pool.pt")
    p.add_argument("--selfcheck", type=Path, required=True, help="output of extent_head.py --selfcheck, state PASSED")
    p.add_argument("--fixture-report", type=Path, required=True, help="report of the CPU fixture chain, state COMPLETED")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    root, run, ev = a.root.resolve(), a.extent_run.resolve(), a.evidence_cache.resolve()
    train, test = json.loads(a.train_manifest.read_text()), json.loads(a.test_manifest.read_text())
    for path, state in ((a.selfcheck, "PASSED"), (a.fixture_report, "COMPLETED"), (ev / "report.json", "COMPLETED")):
        if json.loads(path.read_text()).get("state") != state:
            raise SystemExit("not ready: %s" % path)
    name = lambda r: "%d_%d_%d" % (r["fold"], r["e"], r["c"])
    if any(not (ev / "feat" / (name(r) + ".pt")).is_file() or not (run / "packets" / (name(r) + ".npz")).is_file() for r in test["episodes"]):
        raise SystemExit("cached features or packets missing for a test episode")
    cache, out = a.cache.resolve(), a.out.resolve()
    if cache.exists() or (out / "report.json").exists():
        raise SystemExit("fresh cache directory and report required")
    foris = Path(train["foris_root"])
    frozen = [foris / "models/foris.py", foris / "utils/clustering.py", foris / "utils/refinement.py",
              foris / "utils/data.py", Path(train["projection_basis"])]
    code = [root / x for x in ("scripts/extent_train_cache.py", "scripts/extent_head.py", "scripts/extent_experiment.py",
                               "scripts/analyze_extent.py", "tics/relations.py", "tics/extent_cut.py", "tics/native_assets.py", "tics/__init__.py")]
    env = dict(PYTHONPATH=ENV_PATH, DEMO9_CUDA_GUARD="1", DEMO4_GPU_FRAC=".4", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    done = lambda path, state="COMPLETED": dict(path=str(path), json_equals=dict(state=state))
    built, replayed, scored = done(cache / "report.json"), done(cache / "replay_report.json"), done(out / "report.json")
    py, tm, sm = a.python, str(a.train_manifest.resolve()), str(a.test_manifest.resolve())
    plan = dict(platform="autodl", cuda_python=py, stages=[
        dict(name="extent_head_inputs", kind="gpu", cwd=str(root), timeout_seconds=5400,
             argv=[py, str(root / "scripts/extent_train_cache.py"), "--manifest", tm, "--test-manifest", sm,
                   "--pool", str(ev / "pool.pt"), "--out", str(cache)],
             requires=[dict(path=tm, json_equals=dict(state="PREPARED"), sha256=sha(a.train_manifest)),
                       dict(path=sm, json_equals=dict(state="PREPARED"), sha256=sha(a.test_manifest)),
                       done(a.selfcheck.resolve(), "PASSED"), done(a.fixture_report.resolve()), done(ev / "report.json")],
             cpu_artifacts=[dict(path=str(f), sha256=sha(f)) for f in frozen],
             code_files=[str(f) for f in code], env=env, produces=[built], success_checks=[built]),
        dict(name="extent_head_test_inputs", kind="gpu", cwd=str(root), timeout_seconds=900,
             argv=[py, str(root / "scripts/extent_train_cache.py"), "--replay", "--test-manifest", sm, "--features", str(ev / "feat"),
                   "--packets", str(run / "packets"), "--out", str(cache)],
             requires=[built], code_files=[str(f) for f in code], env=env, produces=[replayed], success_checks=[replayed]),
        dict(name="extent_head_fit", kind="gpu", cwd=str(root), timeout_seconds=3600,
             argv=[py, str(root / "scripts/extent_head.py"), "--cache", str(cache), "--run", str(run), "--out", str(out / "report.json")],
             requires=[replayed], code_files=[str(f) for f in code], env=env, produces=[scored], success_checks=[scored])],
        protocol="one pass; no retries, no sweeps; prediction, gate and what a mismatch means are CARD and GATE in "
                 "scripts/extent_head.py; the replay stops if its inputs differ from the live pass",
        train_episodes=len(train["episodes"]), test_episodes=len(test["episodes"]))
    a.plan.parent.mkdir(parents=True, exist_ok=True)
    a.plan.write_text(json.dumps(plan, indent=1))
    print(json.dumps(dict(state="PLAN_WRITTEN", stages=3, train_episodes=len(train["episodes"]), test_episodes=len(test["episodes"]))))


if __name__ == "__main__":
    main()
