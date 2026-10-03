#!/usr/bin/env python3
"""Write the finite two-stage plan for scripts/experiment_resource_guard.py: the extent run, then its analysis.

CPU only. Run on the server before any GPU is booked, then
  python scripts/experiment_resource_guard.py --plan results/extent_v1/plan.json \
      --state-file results/extent_v1/guard.json --preflight-only
and, on a GPU instance, the same command with `--run --allow-shutdown` instead of `--preflight-only`.
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
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--selfcheck", type=Path, required=True, help="output of extent_selfcheck_cpu.py, state PASSED")
    p.add_argument("--fixture-report", type=Path, required=True, help="report.json of a --fixture run, state COMPLETED")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=4500)
    a = p.parse_args()
    root = a.root.resolve()
    man = json.loads(a.manifest.read_text())
    if man.get("state") != "PREPARED" or not man.get("episodes"):
        raise SystemExit("prepared manifest required")
    for path, state in ((a.selfcheck, "PASSED"), (a.fixture_report, "COMPLETED")):
        if json.loads(path.read_text()).get("state") != state:
            raise SystemExit("CPU check not passed: %s" % path)
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("fresh output directory required")
    foris = Path(man["foris_root"])
    frozen = [foris / "models/foris.py", foris / "utils/clustering.py", foris / "utils/refinement.py",
              foris / "utils/data.py", Path(man["projection_basis"])]
    code = [root / "tics/extent_cut.py", root / "tics/native_assets.py", root / "tics/__init__.py"]
    env = dict(PYTHONPATH=ENV_PATH, DEMO9_CUDA_GUARD="1", DEMO4_GPU_FRAC=".4", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    report, analysis = out / "report.json", out / "analysis.json"
    done = dict(path=str(report), json_equals=dict(state="COMPLETED"))
    plan = dict(platform="autodl", cuda_python=a.python, stages=[
        dict(name="extent_v1", kind="gpu", cwd=str(root), timeout_seconds=a.timeout,
             argv=[a.python, str(root / "scripts/extent_experiment.py"), "--manifest", str(a.manifest.resolve()),
                   "--out", str(out)],
             requires=[dict(path=str(a.manifest.resolve()), json_equals=dict(state="PREPARED"), sha256=sha(a.manifest)),
                       dict(path=str(a.selfcheck.resolve()), json_equals=dict(state="PASSED")),
                       dict(path=str(a.fixture_report.resolve()), json_equals=dict(state="COMPLETED"))],
             cpu_artifacts=[dict(path=str(f), sha256=sha(f)) for f in frozen],
             code_files=[str(f) for f in code], env=env, produces=[done], success_checks=[done]),
        dict(name="extent_v1_analysis", kind="cpu", role="handoff", cwd=str(root), timeout_seconds=60,
             argv=[a.python, str(root / "scripts/analyze_extent.py"), "--run", str(out), "--out", str(analysis)],
             requires=[done], env=dict(env, CUDA_VISIBLE_DEVICES=""),
             produces=[dict(path=str(analysis), json_equals=dict(state="ANALYSED"))],
             success_checks=[dict(path=str(analysis), json_equals=dict(state="ANALYSED"))])],
        protocol="one pass; no retries, no sweeps; the run stops itself after 80 fresh episodes if no arm gains 0.5 mIoU",
        episodes=len(man["episodes"]))
    a.plan.parent.mkdir(parents=True, exist_ok=True)
    a.plan.write_text(json.dumps(plan, indent=1))
    print(json.dumps(dict(state="PLAN_WRITTEN", stages=2, episodes=len(man["episodes"]), timeout_seconds=a.timeout)))


if __name__ == "__main__":
    main()
