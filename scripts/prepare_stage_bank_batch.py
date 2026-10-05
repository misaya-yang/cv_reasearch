#!/usr/bin/env python3
"""Write the finite plan for the stage bank on the remote workspace; never start it.

Two stages (validate with `experiment_pipeline.py --plan PLAN --state-file S --log-file L`; add --run to execute):
  stage_infer  label-free, CPU only: the raw-model origin, every stage of INSID3 and FoRIS and their terms, sealed.
               It reads the saved final-layer tokens of the D run and needs no encoder and no GPU.
  stage_score  the ledger of every stage as additions and deletions of the origin, the family search and the
               value-by-level curves. It uses the GPU when one is visible and falls back to CPU threads.
The sealed output is also the explicit origin of the other runners: `--origin OUT:model.raw_nn`, `--host OUT/predictions:model.raw_nn`.
"""
import argparse
import json
from pathlib import Path

D241 = "outputs/gpu_multilayer_dev241_v1"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", type=Path, required=True); p.add_argument("--run-name", required=True)
    p.add_argument("--cache-root", default="/root/autodl-tmp/demo9_extent"); p.add_argument("--python", default="/root/miniconda3/bin/python")
    p.add_argument("--manifest", default="evidence/local/research_20261005/dev241.json"); p.add_argument("--expected", type=int, default=241)
    p.add_argument("--workers", type=int, default=10); p.add_argument("--random", type=int, default=2000)
    a = p.parse_args(); root = a.workspace.resolve()
    if not a.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in a.run_name):
        p.error("run-name must contain only letters, digits, underscore or hyphen")
    out, plan_dir = root / "outputs" / a.run_name, root / "launch" / a.run_name
    if out.exists(): raise FileExistsError("Use a new run name")
    env = {"PYTHONPATH": "/root/demo4_cache/env"}; manifest = str(root / a.manifest)
    sealed = {"path": str(out / "sealed.json"), "json_equals": {"state": "ALL_PREDICTIONS_SEALED"}}
    stages = [dict(name="stage_infer", kind="cpu", cwd=str(root), env=env,
                   argv=[a.python, "scripts/run_stage_bank.py", "infer", "--manifest", manifest, "--layers", str(root / D241 / "layers"),
                         "--root", a.cache_root, "--out", str(out), "--workers", str(a.workers), "--expected", str(a.expected)],
                   requires=[{"path": manifest}, {"path": str(root / D241 / "sealed.json")}], produces=[sealed]),
              dict(name="stage_score", kind="gpu", cwd=str(root), env=env,
                   argv=[a.python, "scripts/run_stage_bank.py", "score", "--root", a.cache_root, "--out", str(out), "--random", str(a.random)],
                   requires=[sealed], produces=[{"path": str(out / "report.json"), "json_equals": {"n": a.expected}}])]
    plan = dict(stages=stages, time_limits=None, cohort=a.manifest,
                authorization="Prepared only. No rental, download, shutdown, commit or push is implied.",
                note="All DEV241 episodes are development data. stage_infer needs the 16 GiB of saved layers of the D run and about ten CPU processes.")
    plan_dir.mkdir(parents=True, exist_ok=False)
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(plan_dir / "plan.json")


if __name__ == "__main__":
    main()
