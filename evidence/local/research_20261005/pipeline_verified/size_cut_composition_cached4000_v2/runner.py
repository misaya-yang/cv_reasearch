"""File-transferred launcher; validation never executes CUDA or reads GT."""
import argparse
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--validate", action="store_true")
a = p.parse_args()
plan = json.loads((Path(__file__).parent / "plan.json").read_text())
py_compile.compile(plan["entry"], doraise=True)
for stage in (("validate",) if a.validate else ("infer", "score")):
    cmd = ["/root/miniconda3/bin/python", "-u", plan["entry"], stage]
    for key in ("source", "frozen", "out", "prior"):
        cmd += ["--" + key, plan[key]]
    env = dict(os.environ)
    if stage in ("validate", "score"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(cmd, env=env)
    if result.returncode:
        sys.exit(result.returncode)
print("VALIDATED" if a.validate else "CACHED_SIZECUT_INFER_AND_SCORE_COMPLETE", flush=True)
