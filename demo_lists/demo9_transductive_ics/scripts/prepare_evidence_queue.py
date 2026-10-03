#!/usr/bin/env python3
"""Write the finite plan for scripts/experiment_resource_guard.py: cache the features, replay the evidence, analyse.

CPU only. Run on the server after scripts/cpu_checks_evidence.sh has passed, then
  python scripts/experiment_resource_guard.py --plan results/evidence_v1/plan.json \
      --state-file results/evidence_v1/guard.json --preflight-only
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
    p.add_argument("--extent-run", type=Path, required=True, help="finished extent run (report.json COMPLETED, packets)")
    p.add_argument("--selfcheck", type=Path, required=True, help="output of evidence_audit.py --selfcheck, state PASSED")
    p.add_argument("--fixture-report", type=Path, required=True, help="maps_report.json of the CPU fixture chain")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    man = json.loads(a.manifest.read_text())
    run = a.extent_run.resolve()
    for path, state in ((a.selfcheck, "PASSED"), (a.fixture_report, "COMPLETED"), (run / "report.json", "COMPLETED")):
        if json.loads(path.read_text()).get("state") != state:
            raise SystemExit("not ready: %s" % path)
    missing = [r for r in man["episodes"] if not (run / "packets" / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"]))).is_file()]
    if missing:
        raise SystemExit("%d packets missing" % len(missing))
    cache, out = a.cache.resolve(), a.out.resolve()
    if cache.exists() or (out / "maps").exists():
        raise SystemExit("fresh cache and maps directories required")
    foris = Path(man["foris_root"])
    frozen = [foris / "models/foris.py", foris / "utils/clustering.py", foris / "utils/refinement.py",
              foris / "utils/data.py", Path(man["projection_basis"])]
    code = [root / x for x in ("scripts/evidence_cache.py", "scripts/evidence_audit.py", "scripts/extent_experiment.py",
                               "scripts/analyze_extent.py", "tics/extent_cut.py", "tics/native_assets.py", "tics/__init__.py")]
    env = dict(PYTHONPATH=ENV_PATH, DEMO9_CUDA_GUARD="1", DEMO4_GPU_FRAC=".4", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    done = lambda path, state: dict(path=str(path), json_equals=dict(state=state))
    cached, mapped, audited = done(cache / "report.json", "COMPLETED"), done(out / "maps_report.json", "COMPLETED"), done(out / "audit.json", "ANALYSED")
    py = a.python
    plan = dict(platform="autodl", cuda_python=py, stages=[
        dict(name="evidence_cache", kind="gpu", cwd=str(root), timeout_seconds=1800,
             argv=[py, str(root / "scripts/evidence_cache.py"), "--manifest", str(a.manifest.resolve()), "--packets",
                   str(run / "packets"), "--out", str(cache)],
             requires=[dict(path=str(a.manifest.resolve()), json_equals=dict(state="PREPARED"), sha256=sha(a.manifest)),
                       done(a.selfcheck.resolve(), "PASSED"), done(a.fixture_report.resolve(), "COMPLETED"),
                       done(run / "report.json", "COMPLETED")],
             cpu_artifacts=[dict(path=str(f), sha256=sha(f)) for f in frozen],
             code_files=[str(f) for f in code], env=env, produces=[cached], success_checks=[cached]),
        dict(name="evidence_maps", kind="gpu", cwd=str(root), timeout_seconds=1800,
             argv=[py, str(root / "scripts/evidence_audit.py"), "--cache", str(cache), "--run", str(run), "--out", str(out)],
             requires=[cached], code_files=[str(f) for f in code], env=env, produces=[mapped], success_checks=[mapped]),
        dict(name="evidence_analysis", kind="cpu", role="handoff", cwd=str(root), timeout_seconds=60,
             argv=[py, str(root / "scripts/evidence_audit.py"), "--analyse", "--run", str(run), "--out", str(out)],
             requires=[mapped], env=dict(env, CUDA_VISIBLE_DEVICES=""), produces=[audited], success_checks=[audited])],
        protocol="one pass; no retries, no sweeps; the cache step stops at the first episode whose stored mask is not "
                 "reproduced; gates and predictions are the CARDS and GATE of scripts/evidence_audit.py",
        episodes=len(man["episodes"]))
    a.plan.parent.mkdir(parents=True, exist_ok=True)
    a.plan.write_text(json.dumps(plan, indent=1))
    print(json.dumps(dict(state="PLAN_WRITTEN", stages=3, episodes=len(man["episodes"]))))


if __name__ == "__main__":
    main()
