"""New disjoint-image test of fixed prompt rules with inference budget ablations."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/takeover_20261001_v1/quality_followup_v1"
OUT = R / "holdout_256"
SUBSET = ROOT / "assets/coco_prompt_holdout_seed2029_v1"


def main():
    OUT.mkdir(exist_ok=False)
    manifests = [json.loads((ROOT/"assets"/p/"manifest.json").read_text()) for p in ("coco_quality_seed2027_v1", "coco_flip_validation_seed2028_v1")]
    exclusions = {i["image_id"] for m in manifests for i in m["images"]}
    assert len(exclusions) == 152
    exclusion_file = OUT / "excluded_manifest.json"
    exclusion_file.write_text(json.dumps({"images":[{"image_id":i} for i in sorted(exclusions)]})+"\n")
    protocol = {"fixed_before_new_outputs": True, "new_images":256, "seed":2029,
                "excluded_images":sorted(exclusions), "point_offsets_resized_pixels":[[-8,0],[8,0],[0,-8],[0,8]],
                "primary_policy":"perturb_consistency (four views)", "primary_endpoint":"central mean mask IoU",
                "secondary_endpoints":["near boundary IoU", "BoundaryIoU", "box control"],
                "strong_controls":["largest original mask", "original IoU head", "original logit stability"],
                "ablation_policies":["one left view", "two horizontal views"],
                "candidate_set":"original tokens1..3, no new masks", "training":False,
                "GT_used_for_selection":False, "threshold_tuning":False,
                "scope":"same COCO distribution; disjoint-image test, not cross-dataset or SOTA"}
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    report = {"status":"RUNNING", "stages":[]}
    def save():
        t=OUT/"queue_status.tmp";t.write_text(json.dumps(report,indent=2)+"\n");t.replace(OUT/"queue_status.json")
    jobs = [
        ("prepare_CPU", ["research/quality_mechanisms/prepare_coco_subset.py", "--annotations-zip", "/root/autodl-pub/COCO2017/annotations_trainval2017.zip", "--images-zip", "/root/autodl-pub/COCO2017/val2017.zip", "--output-dir", str(SUBSET), "--seed", "2029", "--num-images", "256", "--exclude-manifest", str(exclusion_file)]),
        ("official_cached_outputs", ["tools/takeover_official_export.py", "--subset-dir",str(SUBSET),"--checkpoint",str(ROOT/"assets/checkpoints/sam_vit_b_01ec64.pth"),"--output-dir",str(OUT/"original"),"--device","cuda:0","--limit-images","256","--microbatch","4","--defer-quality","--npz-compression","none","--save-encoded-inputs"]),
        ("fixed_prompt_rules", ["tools/takeover_prompt_stability.py","--source-run",str(OUT/"original"),"--output",str(OUT/"prompt_stability_256.json"),"--checkpoint",str(ROOT/"assets/checkpoints/sam_vit_b_01ec64.pth"),"--cohort","256 newly sampled COCO images excluding all prior152; fixed rules before outputs; image-cluster intervals"])
    ]
    save()
    try:
        for name,argv in jobs:
            if name != "prepare_CPU":
                while True:
                    pids = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
                    if not pids:break
                    report["status"]="WAITING_OTHER_GPU_OWNER";report["other_GPU_PIDs"]=pids.splitlines();save();time.sleep(15)
            report["status"]="RUNNING"
            row={"name":name,"status":"RUNNING","command":[sys.executable,*argv]};report["stages"].append(row);save()
            started=time.monotonic();print("START "+name,flush=True)
            with (OUT/(name+".log")).open("x") as log:
                p=subprocess.Popen(row["command"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);row["pid"]=p.pid;save();code=p.wait()
            row.update(status="DONE" if not code else "FAILED",returncode=code,wall_seconds=time.monotonic()-started);save()
            if code:raise RuntimeError(name+" failed")
            if name == "prepare_CPU":
                m=json.loads((SUBSET/"manifest.json").read_text())
                assert len(m["images"])==256 and not exclusions.intersection(i["image_id"] for i in m["images"])
                report["instances"]=m["num_instances"]
            print("DONE "+name,flush=True)
        report["status"]="COMPLETED"
    except BaseException as e:report.update(status="ERROR",error=str(e));raise
    finally:save()


if __name__ == "__main__":main()
