#!/usr/bin/env python3
"""Write the finite plan for scripts/experiment_resource_guard.py: the scale-alignment run, then its analysis.

CPU only. Run on the server after scripts/cpu_checks_scale_align.sh has passed, then
  python scripts/experiment_resource_guard.py --plan results/scale_align_v0/plan.json \
      --state-file results/scale_align_v0/guard.json --preflight-only
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
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--packets", type=Path, required=True, help="packets of the finished extent run")
    p.add_argument("--fixture-report", type=Path, required=True, help="report.json of a --fixture run, state COMPLETED")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    man = json.loads(a.manifest.read_text())
    if man.get("state") != "PREPARED" or json.loads(a.fixture_report.read_text()).get("state") != "COMPLETED":
        raise SystemExit("prepared manifest and a completed fixture run required")
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("fresh output directory required")
    foris = Path(man["foris_root"])
    frozen = [foris / "models/foris.py", foris / "utils/clustering.py", foris / "utils/refinement.py",
              foris / "utils/data.py", Path(man["projection_basis"])]
    code = [root / x for x in ("scripts/scale_align_experiment.py", "scripts/analyze_scale_align.py", "scripts/extent_experiment.py",
                               "scripts/analyze_extent.py", "tics/extent_cut.py", "tics/native_assets.py", "tics/__init__.py")]
    env = dict(PYTHONPATH=ENV_PATH, DEMO9_CUDA_GUARD="1", DEMO4_GPU_FRAC=".4", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    done = dict(path=str(out / "report.json"), json_equals=dict(state="COMPLETED"))
    analysed = dict(path=str(out / "analysis.json"), json_equals=dict(state="ANALYSED"))
    py = a.python
    plan = dict(platform="autodl", cuda_python=py, stages=[
        dict(name="scale_align_v0", kind="gpu", cwd=str(root), timeout_seconds=4500,
             argv=[py, str(root / "scripts/scale_align_experiment.py"), "--manifest", str(a.manifest.resolve()),
                   "--packets", str(a.packets.resolve()), "--out", str(out)],
             requires=[dict(path=str(a.manifest.resolve()), json_equals=dict(state="PREPARED"), sha256=sha(a.manifest)),
                       dict(path=str(a.fixture_report.resolve()), json_equals=dict(state="COMPLETED"))],
             cpu_artifacts=[dict(path=str(f), sha256=sha(f)) for f in frozen],
             code_files=[str(f) for f in code], env=env, produces=[done], success_checks=[done]),
        dict(name="scale_align_v0_analysis", kind="cpu", role="handoff", cwd=str(root), timeout_seconds=60,
             argv=[py, str(root / "scripts/analyze_scale_align.py"), "--run", str(out), "--out", str(out / "analysis.json")],
             requires=[done], env=dict(env, CUDA_VISIBLE_DEVICES=""), produces=[analysed], success_checks=[analysed])],
        protocol="one pass; no retries, no sweeps; stops after 80 fresh episodes if the true-scale arm gains under 0.5; "
                 "prediction and gate are CARD and GATE in scripts/analyze_scale_align.py",
        episodes=len(man["episodes"]))
    a.plan.parent.mkdir(parents=True, exist_ok=True)
    a.plan.write_text(json.dumps(plan, indent=1))
    print(json.dumps(dict(state="PLAN_WRITTEN", stages=2, episodes=len(man["episodes"]))))


if __name__ == "__main__":
    main()
