"""CPU-only stage supervisor. No waiting for unrelated CUDA jobs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
previous = root / "results/dev128_v1/status.json"
state_path = root / "results/followup_status.json"
state_path.parent.mkdir(exist_ok=True)


def save(state, **extra):
    state_path.write_text(json.dumps(dict(state=state, pid=os.getpid(), **extra), indent=2))


save("CPU_WAITING_PREVIOUS_STAGE")
while True:
    if previous.exists():
        try:
            state = json.loads(previous.read_text())
        except json.JSONDecodeError:
            time.sleep(2)
            continue
        if state["state"] == "ERROR":
            save("ERROR_PREVIOUS_STAGE", previous_error=state.get("error"))
            sys.exit(1)
        if state["state"] == "COMPLETED":
            # Previous scientific job releases its CUDA context before loading ours.
            try:
                os.kill(state["pid"], 0)
            except ProcessLookupError:
                break
    time.sleep(2)
save("RUNNING_LOW_RESOLUTION_STAGE")
result = subprocess.run([sys.executable, "-u", str(root / "low_resolution.py"),
                         "--prior", str(root / "results/dev128_v1"),
                         "--out", str(root / "results/low_dev128_v1")], cwd=root)
save("COMPLETED" if result.returncode == 0 else "ERROR_LOW_RESOLUTION_STAGE",
     child_exit_code=result.returncode)
sys.exit(result.returncode)
