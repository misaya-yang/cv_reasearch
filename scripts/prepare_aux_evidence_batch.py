#!/usr/bin/env python3
"""Write the finite plan for the evidence sweep on the remote workspace; never start it.

Two stages (validate with `experiment_pipeline.py --plan PLAN --state-file S --log-file L`; add --run to execute):
  aux_infer   label-free: the whole evidence library for every episode, sealed. One GPU process, or CPU workers.
  aux_score   CPU only: the sweep over every proposal and every map, placebo calibration, nested composition,
              sealed masks and the common report. It needs no GPU and can run beside other GPU work.

Proposals are read, never recomputed: the four arms sealed on DEV241, the proposals of the fixed edit-auxiliary
suite (--suite: E, its fixed-slot and two-slot controls, the exact B kernel control) and the Astra replay (--astra).
Layer spaces: the saved layers 16 and 24 of the D run by default; --forward-layers taps more blocks in one extra
paired forward per episode and stores nothing but the maps; --no-layers keeps to the cached last layer.
"""
import argparse
import json
from pathlib import Path

D241, RCG = "outputs/gpu_multilayer_dev241_v1", "outputs/rcg241"
SEALED = [("rcg", RCG, "rcg", "fields"), ("delta", D241, "multilayer.delta.control", "packets"),
          ("D", D241, "multilayer", "packets"), ("concat", D241, "multilayer.control", "packets")]
SUITE = [("E", "latent_native"), ("Efixed", "latent_native.control"), ("Etwo", "latent_native.two_slot_em.control"),
         ("Bkernel", "structure_conditioned.kernel.control")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", type=Path, required=True); p.add_argument("--run-name", required=True)
    p.add_argument("--cache-root", default="/root/autodl-tmp/demo9_extent"); p.add_argument("--python", default="/root/miniconda3/bin/python")
    p.add_argument("--manifest", default="evidence/local/research_20261005/dev241.json"); p.add_argument("--expected", type=int, default=241)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda"); p.add_argument("--threads", type=int, default=2)
    p.add_argument("--suite", type=Path, help="output directory of run_edit_auxiliary.py (its proposals/ and fields/)")
    p.add_argument("--astra", type=Path, help="output directory of the Astra replay (its frozen/)")
    p.add_argument("--host", help="MASK_DIR:ARM; default: the suite's copy of the cached native, else the replayed native of the D run")
    p.add_argument("--no-layers", action="store_true", help="last-layer evidence only")
    p.add_argument("--forward-layers", help="e.g. 4,8,12,16,20,22,24: one extra paired forward per episode (GPU only)")
    p.add_argument("--workers", type=int, default=6, help="CPU processes for the counting pass, and for inference on CPU")
    p.add_argument("--proposer", action="append", default=[], help="further NAME=MASK_DIR:ARM[:FIELD_DIR:FIELD_KEY]")
    a = p.parse_args(); root = a.workspace.resolve()
    if not a.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in a.run_name):
        p.error("run-name must contain only letters, digits, underscore or hyphen")
    out, plan_dir = root / "outputs" / a.run_name, root / "launch" / a.run_name
    if out.exists(): raise FileExistsError("Use a new run name")
    env = {"PYTHONPATH": "/root/demo4_cache/env"}; manifest = str(root / a.manifest)
    if a.forward_layers and a.device != "cuda": p.error("--forward-layers needs --device cuda")
    genv = dict(env, DEMO4_GPU_FRAC="0.85", PYTHONPATH=":".join(["/root/demo4_cache/env", "/root/autodl-tmp/demo8_local_verification/crf_source/src",
                                                              "/root/autodl-tmp/demo8_local_verification/runtime/extensions"])) if a.forward_layers else env
    proposers = [f"{n}={root / run / 'predictions'}:{arm}:{root / run / sub}:{arm}" for n, run, arm, sub in SEALED]
    requires = [{"path": manifest}, {"path": str(root / D241 / "sealed.json")}, {"path": str(root / RCG / "sealed.json")}]
    host = a.host or f"{root / D241 / 'predictions'}:native"
    if a.suite:
        suite = a.suite.resolve(); requires.append({"path": str(suite / "proposal_sealed.json"), "json_equals": {"state": "ALL_PREDICTIONS_SEALED"}})
        proposers += [f"{n}={suite / 'proposals'}:{arm}:{suite / 'fields'}:{arm}" for n, arm in SUITE]
        host = a.host or f"{suite / 'proposals'}:native"
    if a.astra:
        astra = a.astra.resolve(); requires.append({"path": str(astra / "freeze.json")})
        proposers.append(f"astra={astra / 'frozen'}:RCG_count_matched_delete")
    proposers += a.proposer
    argv = [a.python, "scripts/run_aux_evidence.py", "infer", "--root", a.cache_root, "--manifest", manifest, "--out", str(out),
            "--device", a.device, "--threads", str(a.threads), "--workers", "1" if a.device == "cuda" else str(a.workers), "--host", host]
    for s in proposers: argv += ["--proposer", s]
    if a.forward_layers:
        argv += ["--forward-layers", a.forward_layers]
    elif not a.no_layers:
        argv += ["--layers", str(root / D241 / "layers"), "--bases", str(root / D241 / "arm_bases.pt")]
        requires.append({"path": str(root / D241 / "arm_bases.pt")})
    stages = [dict(name="aux_infer", kind="gpu" if a.device == "cuda" else "cpu", cwd=str(root), env=genv, argv=argv, requires=requires,
                   produces=[{"path": str(out / "sealed.json"), "json_equals": {"state": "ALL_EVIDENCE_SEALED"}}]),
              dict(name="aux_score", kind="cpu", cwd=str(root), env=env,
                   argv=[a.python, "scripts/run_aux_evidence.py", "score", "--root", a.cache_root, "--out", str(out), "--workers", str(a.workers)],
                   requires=[{"path": str(out / "sealed.json"), "json_equals": {"state": "ALL_EVIDENCE_SEALED"}}],
                   produces=[{"path": str(out / "report.json"), "json_equals": {"n": a.expected}}])]
    plan = dict(stages=stages, time_limits=None, cohort=a.manifest,
                authorization="Prepared only. No rental, download, shutdown, commit or push is implied.",
                note="All DEV241 episodes are development data. Without --forward-layers the layer spaces need the 16 GiB of saved layers of the D run.")
    plan_dir.mkdir(parents=True, exist_ok=False)
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(plan_dir / "plan.json")


if __name__ == "__main__":
    main()
