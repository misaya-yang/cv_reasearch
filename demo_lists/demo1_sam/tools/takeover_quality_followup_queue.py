"""Serialized quality follow-ups; waits for other GPU owners without stopping them."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/takeover_20261001_v1"
OUT = R / "quality_followup_v1"


def main():
    OUT.mkdir(exist_ok=False)
    state = {"status": "PREPARING_CPU", "pid": os.getpid(), "stages": []}
    def save():
        tmp = OUT / "queue_status.tmp"
        tmp.write_text(json.dumps(state, indent=2)+"\n")
        tmp.replace(OUT / "queue_status.json")
    save()
    try:
        # Validate intervention contracts before acquiring a GPU slot.
        from takeover_prompt_stability import perturb_objects, select
        import numpy as np
        obj = [{"central_xy": [0, 0], "near_boundary_xy": [9, 9], "tight_box_xyxy": [1,1,9,9]}]
        a = perturb_objects(obj, "central", (10,10), (100,100), (-8,0))
        assert a[0]["central_xy"] == [0,0] and obj[0]["central_xy"] == [0,0]
        b = perturb_objects(obj, "box", (10,10), (100,100), (8,0))
        assert np.allclose(b[0]["tight_box_xyxy"], [1.8,1,9.8,9])
        masks = np.zeros((4,4,4),dtype=bool)
        masks[1,:1,:1] = True; masks[2,:2,:2] = True; masks[3,:3,:3] = True
        scores = np.array([0,.2,.8,.1])
        low = np.where(masks, 2., -2.)
        choices, consistency, _ = select(masks, scores, low, np.stack([masks]*4), np.stack([scores]*4))
        assert choices["perturb_consistency"] == 2 and consistency == [1.,1.,1.]
        assert choices["largest_original_mask"] == 3 and choices["perturb_mean_iou_head"] == 2
        state["CPU_contracts"] = "PASS: clipping, box size preservation, immutable metadata, matching tie fallback, simple controls"
        state["status"] = "WAITING_FOR_OTHER_GPU_PROCESSES"
        save()
        while True:
            pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
            state["other_GPU_PIDs"] = pids.splitlines() if pids else []
            save()
            if not pids: break
            time.sleep(15)
        lock = (ROOT/"runtime/takeover_gpu.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state["status"] = "RUNNING"
        save()
        source = R / "real_original"
        checkpoint = ROOT / "assets/checkpoints/sam_vit_b_01ec64.pth"
        cached = OUT / "cached_128"
        jobs = [
            ("virtual_flip_24", ["tools/takeover_virtual_flip.py", "--source-run", str(source), "--output-dir", str(OUT/"virtual_flip_24"), "--device", "cuda:0"]),
            ("virtual_flip_score_24", ["tools/takeover_flip_score.py", "--original-run", str(source), "--flipped-run", str(OUT/"virtual_flip_24"), "--output", str(OUT/"virtual_flip_score_24.json")]),
            ("prompt_stability_24", ["tools/takeover_prompt_stability.py", "--source-run", str(source), "--output", str(OUT/"prompt_stability_24.json"), "--checkpoint", str(checkpoint), "--cohort", "24 image development"]),
            ("cache_original_128", ["tools/takeover_official_export.py", "--subset-dir", str(ROOT/"assets/coco_flip_validation_seed2028_v1"), "--checkpoint", str(checkpoint), "--output-dir", str(cached), "--device", "cuda:0", "--limit-images", "128", "--microbatch", "4", "--defer-quality", "--npz-compression", "none", "--save-encoded-inputs"]),
            ("prompt_stability_128", ["tools/takeover_prompt_stability.py", "--source-run", str(cached), "--output", str(OUT/"prompt_stability_128.json"), "--checkpoint", str(checkpoint), "--cohort", "Fixed follow-up on 128 images disjoint from development24; same images previously used for RGB flip; new perturbation outputs with unchanged fixed rules"]),
        ]
        (OUT/"protocol.json").write_text(json.dumps({"fixed_before_outputs": True, "offsets_resized_pixels": [[-8,0],[8,0],[0,-8],[0,8]], "selection_GT": False, "primary_endpoints": ["central mean IoU", "near boundary mean IoU"], "controls": ["original IoU head", "largest original mask", "logit stability", "mean matched IoU head"], "outputs": "original four masks retained; only tokens1..3 selectable", "new_heldout_dataset": False}, indent=2)+"\n")
        for name, argv in jobs:
            command = [sys.executable, *argv]
            row = {"name": name, "status": "RUNNING", "command": command}
            state["stages"].append(row);save()
            print("START "+name, flush=True)
            started = time.monotonic()
            with (OUT/(name+".log")).open("x") as log:
                p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                row["pid"] = p.pid;save();code = p.wait()
            row.update(status="DONE" if code == 0 else "FAILED", returncode=code, wall_seconds=time.monotonic()-started);save()
            if code: raise RuntimeError(name+" failed")
            print("DONE "+name, flush=True)
        state["status"] = "COMPLETED"
    except BaseException as error:
        state.update(status="ERROR", error=str(error));raise
    finally: save()


if __name__ == "__main__": main()
