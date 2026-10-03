#!/usr/bin/env python3
"""Write finite plans for scripts/experiment_resource_guard.py: the decision cache, or a fit on a finished cache.

  python scripts/prepare_decision_queue.py cache --suite DIR --extent-run results/extent_v1/run \
      --fixture-report results/decision_v1/fixture/decision/report.json --cache cache/decision_v1 --plan results/decision_v1/plan_cache.json
  python scripts/prepare_decision_queue.py fit --cache cache/decision_v1 --out results/decision_v1/fit --plan results/decision_v1/plan_fit.json -- <arguments of decision_fit.py>
Then `experiment_resource_guard.py --plan P --state-file S --preflight-only`, and the same with `--run`.
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("cache", "fit", "infer", "zoom"))
    p.add_argument("--models", type=Path, help="infer: the saved read-outs of one arm")
    p.add_argument("--episodes", type=Path, help="infer: the manifest to score")
    p.add_argument("--packets", type=Path, help="infer: stored FoRIS packets for the bit-exactness check (development only)")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--python", default="/root/miniconda3/bin/python")
    p.add_argument("--suite", type=Path)
    p.add_argument("--manifest", type=Path, help="another training manifest to add to a finished cache")
    p.add_argument("--extent-run", type=Path)
    p.add_argument("--fixture-report", type=Path)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--lean", action="store_true", help="cache: leave out feature components and reference neighbours")
    p.add_argument("--timeout", type=int, default=3600)
    a, rest = p.parse_known_args()  # what is not known here goes to decision_fit.py
    root, py, cache = a.root.resolve(), a.python, a.cache.resolve()
    done = lambda path, state: dict(path=str(path), json_equals=dict(state=state))
    env = dict(PYTHONPATH=ENV_PATH, DEMO9_CUDA_GUARD="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    if a.what == "cache":
        suite, run = a.suite.resolve(), a.extent_run.resolve()
        dev = json.loads((suite / "dev_episodes.json").read_text())
        for path, state in ((a.fixture_report, "COMPLETED"), (run / "report.json", "COMPLETED")):
            if json.loads(path.read_text()).get("state") != state:
                raise SystemExit("not ready: %s" % path)
        tag = "_" + a.manifest.stem if a.manifest else ""
        if (cache / ("report%s.json" % tag)).exists():
            raise SystemExit("this cache step has already been run")
        foris = Path(dev["foris_root"])
        frozen = [foris / "models/foris.py", foris / "utils/clustering.py", foris / "utils/refinement.py", foris / "utils/data.py",
                  Path(dev["projection_basis"])]
        code = [root / x for x in ("scripts/decision_cache_parallel.py", "scripts/decision_cache.py", "scripts/extent_train_cache.py",
                                   "scripts/extent_experiment.py", "tics/relations.py", "tics/extent_cut.py", "tics/native_assets.py")]
        names = ("train", "confirm", "dev") if not a.manifest else ("dev",)
        needs = [dict(path=str(suite / (k + "_episodes.json")), json_equals=dict(state="PREPARED"), sha256=sha(suite / (k + "_episodes.json"))) for k in names]
        argv = [py, str(root / "scripts/decision_cache_parallel.py"), "--workers", str(a.workers), "--", "--suite", str(suite),
                "--packets", str(run / "packets"), "--out", str(cache)]
        if a.manifest:
            needs.append(dict(path=str(a.manifest.resolve()), json_equals=dict(state="PREPARED"), sha256=sha(a.manifest)))
            needs.append(done(cache / "report.json", "COMPLETED"))
            argv += ["--manifest", str(a.manifest.resolve()), "--kinds", "train"] + (["--lean"] if a.lean else [])
        made = done(cache / ("report%s.json" % tag), "COMPLETED")
        stages = [dict(name="decision_cache" + tag, kind="gpu", cwd=str(root), timeout_seconds=a.timeout, argv=argv,
                       requires=needs + [done(a.fixture_report.resolve(), "COMPLETED"), done(run / "report.json", "COMPLETED")],
                       cpu_artifacts=[dict(path=str(f), sha256=sha(f)) for f in frozen], code_files=[str(f) for f in code],
                       env=dict(env, DEMO4_GPU_FRAC="%.2f" % min(0.4, 0.75 / a.workers)), produces=[made], success_checks=[made])]
    elif a.what in ("infer", "zoom"):
        out = a.out.resolve()
        if out.exists() and any(out.iterdir()):
            raise SystemExit("fresh output directory required")
        script = "scripts/decision_%s.py" % a.what
        code = [root / x for x in (script, "scripts/decision_infer.py", "scripts/decision_cache.py", "scripts/extent_experiment.py",
                                   "scripts/scale_align_experiment.py", "scripts/analyze_extent.py", "tics/decision_heads.py", "tics/relations.py")]
        made = done(out / "report.json", "COMPLETED")
        argv = [py, str(root / script), "--models", str(a.models.resolve()), "--cache", str(cache), "--manifest",
                str(a.episodes.resolve()), "--out", str(out), "--parallel", str(a.workers)] + (["--packets", str(a.packets.resolve())] if a.packets and a.what == "infer" else [])
        stages = [dict(name="decision_%s_%s" % (a.what, out.name), kind="gpu", cwd=str(root), timeout_seconds=a.timeout, argv=argv,
                       requires=[done(cache / "report.json", "COMPLETED"), dict(path=str(a.models.resolve())),
                                 dict(path=str(a.episodes.resolve()), json_equals=dict(state="PREPARED"))],
                       code_files=[str(f) for f in code], env=dict(env, DEMO4_GPU_FRAC="%.2f" % min(0.4, 0.75 / a.workers)),
                       produces=[made], success_checks=[made])]
    else:
        out = a.out.resolve()
        if (out / "report.json").exists():
            raise SystemExit("fresh output directory required")
        code = [root / x for x in ("scripts/decision_fit_parallel.py", "scripts/decision_fit.py", "tics/decision_heads.py", "scripts/analyze_extent.py")]
        made = done(out / "report.json", "COMPLETED")
        stages = [dict(name="decision_fit", kind="gpu", cwd=str(root), timeout_seconds=a.timeout,
                       argv=[py, str(root / "scripts/decision_fit_parallel.py"), "--cache", str(cache), "--out", str(out), "--workers", str(a.workers)]
                       + [x for x in rest if x != "--"],
                       requires=[done(cache / "report.json", "COMPLETED")], code_files=[str(f) for f in code],
                       env=dict(env, DEMO4_GPU_FRAC=".5"), produces=[made], success_checks=[made])]
    a.plan.parent.mkdir(parents=True, exist_ok=True)
    a.plan.write_text(json.dumps(dict(platform="autodl", cuda_python=py, stages=stages,
                                      protocol="one pass, no retries; cards and gates are in the scripts named in code_files"), indent=1))
    print(json.dumps(dict(state="PLAN_WRITTEN", stages=[s["name"] for s in stages])))


if __name__ == "__main__":
    main()
