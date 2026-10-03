#!/usr/bin/env python3
"""Metadata-only registered-photo confirmation adapter for the frozen F4 runner.

No all-history-unseen claim is made. The cohort is disjoint from the REGISTERED
DEV/confirmation/base-training/prior-pool sources named by the exposure audit.
This helper never reads PNG pixels, fits parameters, imports torch, downloads,
or leases a GPU. The run wrapper requires one root-frozen primary and delegates
unchanged inference to scripts/f4_experiment.py; other arms stay fixed controls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ARMS=("native","anchor_only","full001","perFG002","matchedGlobal","Part4cal","querycore")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError("Preserve frozen metadata; output already exists: "+str(path))
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w") as stream:
        json.dump(value,stream,indent=1,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)


def uid(path):
    found=re.search(r"_(\d{12})\.[^.]+$",Path(path).name)
    if not found:raise ValueError("Unrecognized existing COCO image UID: "+str(path))
    return int(found.group(1))


def prepare(a):
    from PIL import Image  # Header-only Image.open/size/format; never load/convert.
    source=json.loads(a.source_manifest.read_text())
    audit=json.loads(a.exposure_audit.read_text())
    template=json.loads(a.template.read_text())
    proof=json.loads(a.cpu_receipt.read_text())
    if (audit.get("state")!="FRESH160_REGISTERED_SCOPE_METADATA_PREPARED" or
        audit.get("manifest_sha256")!=sha(a.source_manifest) or
        any(v!=0 for v in audit.get("registered_overlap_counts",{}).values()) or
        audit.get("current_preparation_queryPNG_pixels_opened") is not False):
        raise ValueError("Exact registered-scope exposure audit required")
    if (proof.get("state")!="CPU_F4_SEVEN_SOURCE_PASSED" or proof.get("cases")!=10 or
        proof.get("CUDA_initialized") is not False or proof.get("no_query_GT_in_decisions") is not True):
        raise ValueError("Existing frozen ten-case public-source CPU proof required")
    if template.get("schema")!="f4_seven_arms_v1" or template.get("recipe",{}).get("arms")!=list(ARMS):
        raise ValueError("Frozen complete-seven-arm development template required")
    for path,digest in template["source_hashes"].items():
        if sha(path)!=digest or proof["source_hashes"].get(path)!=digest:
            raise ValueError("Frozen recipe/runtime differs from accepted source proof: "+path)
    rows=source.get("episodes",[])
    if len(rows)!=audit["episodes"] or len(rows)!=160:raise ValueError("Exact registered160 cohort required")
    kept=[];fresh_ids=set();fold_counts={str(f):0 for f in range(4)};assets=[];headers=[]
    data=Path(source.get("data_root",template["data_root"]))
    ann=Path(source.get("annotation_root",template["annotation_root"]))
    for original in rows:
        row={k:original[k] for k in ("fold","e","c","support","query")}
        if row["fold"] not in range(4) or row["c"]%4!=row["fold"]:raise ValueError("Official collection-fold class contract")
        fold_counts[str(row["fold"])]+=1
        for role in ("support","query"):
            image=data/row[role]
            mask=ann/Path(row[role]).with_suffix(".png")
            role_uid=uid(row[role])
            if role_uid in fresh_ids:raise ValueError("Repeated photograph in registered confirmation cohort")
            fresh_ids.add(role_uid)
            declared=original.get(role+"_image_id")
            if declared is not None and int(declared)!=role_uid:raise ValueError("Manifest UID/path disagreement")
            with Image.open(image) as im:
                image_hw=[im.height,im.width];image_format=im.format
            with Image.open(mask) as im:
                mask_hw=[im.height,im.width];mask_format=im.format
            if image_hw!=mask_hw or mask_format!="PNG":raise ValueError("Image/label header geometry mismatch")
            for path in (image,mask):
                st=path.stat();assets.append(dict(path=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns))
            headers.append(dict(fold=row["fold"],e=row["e"],role=role,UID=role_uid,
                                original_hw=image_hw,image_format=image_format,mask_format=mask_format,
                                pixel_data_opened=False))
        kept.append(row)
    if len(fresh_ids)!=audit["unique_all_role_UIDs"] or any(n!=40 for n in fold_counts.values()):
        raise ValueError("Registered UID/fold counts disagree")
    # Recheck only the small registered metadata sources, never images/tensors.
    registered_sets={}
    for name in ("dev","confirm","train"):
        item=audit["registered_sources"][name];path=Path(item["path"])
        if sha(path)!=item["sha256"]:raise ValueError("Registered exposure source changed: "+name)
        doc=json.loads(path.read_text())
        registered_sets[name]={uid(r[role]) for r in doc["episodes"] for role in ("support","query")}
    pool=audit["registered_sources"]["pool"]
    pool_path=Path(pool["metadata_source"]["path"])
    if sha(pool_path)!=pool["metadata_source"]["sha256"]:raise ValueError("Registered pool metadata changed")
    pool_doc=json.loads(pool_path.read_text())
    registered_sets["pool"]={int(v) for v in pool_doc["protocol"]["exposed_pool_UIDs"]}
    overlap={name:len(fresh_ids&ids) for name,ids in registered_sets.items()}
    if any(overlap.values()) or overlap!=audit["registered_overlap_counts"]:raise ValueError("Registered all-role photograph overlap")
    # Retain non-cohort assets such as the existing native SVD basis, not DEV images.
    for item in template["assets"]:
        path=Path(item["path"])
        if path.is_relative_to(Path(template["data_root"])) or path.is_relative_to(Path(template["annotation_root"])):continue
        st=path.stat()
        if (st.st_size,st.st_mtime_ns)!=(item["size"],item["mtime_ns"]):raise ValueError("Frozen non-cohort asset changed")
        assets.append(item)
    manifest={**template,"state":"REGISTERED_CONFIRMATION_PREPARED_PRIMARY_SELECTION_REQUIRED",
        "episodes":kept,"scope":"registered_photo_disjoint_confirmation","design_DEV":False,
        "parent_manifest":str(a.source_manifest),"parent_manifest_sha256":sha(a.source_manifest),
        "data_root":str(data),"annotation_root":str(ann),"seed":source.get("seed",2044),
        "baseline_rows":{},"first8_native_pixel_and_original_IU_gate":False,
        "native_public_uncached_cached_identity_required":True,"assets":assets,
        "unseen_receipt":None,"unseen_receipt_sha256":None,
        "selection_path":str(a.selection.resolve()),"primary_selection_required_before_query_inference":True,
        "template_DEV_manifest":str(a.template),"template_DEV_manifest_sha256":sha(a.template),
        "exposure_audit":str(a.exposure_audit),"exposure_audit_sha256":sha(a.exposure_audit),
        "CPU_receipt":str(a.cpu_receipt),"CPU_receipt_sha256":sha(a.cpu_receipt),
        "confirmation_helper_sha256":sha(__file__),
        "registered_overlap_counts":overlap,"registered_sources":audit["registered_sources"],
        "registered_all_role_UIDs":sorted(fresh_ids),"classes_observed":len({r["c"] for r in kept}),
        "all_historical_unseen_claim":False,
        "history_limit":audit["history_limit"],
        "confirmation_claim":"New inference, photo-disjoint from the explicitly registered sources; prior unregistered historical GT exposure is not certified",
        "controls_policy":"All seven fixed recipe arms may run, but only the unique pre-frozen primary is evaluated confirmatorily; no fresh-arm reselection",
        "query_GT_opened_after_all_seven_predictions":True}
    save(a.manifest,manifest)
    receipt=dict(state="REGISTERED_CONFIRMATION_METADATA_PASSED_SELECTION_PENDING",
        manifest=str(a.manifest),manifest_sha256=sha(a.manifest),episodes=len(kept),
        per_fold=fold_counts,unique_all_role_UIDs=len(fresh_ids),registered_overlap_counts=overlap,
        class_count=manifest["classes_observed"],headers_checked=len(headers),
        RGB_pixels_decoded=False,query_GT_pixels_opened=False,torch_imported="torch" in sys.modules,
        CUDA_started=False,GPU_started=False,source_files_unchanged=True,
        selection_path=str(a.selection),selection_exists=a.selection.exists(),
        all_historical_unseen_claim=False,history_limit=manifest["history_limit"],
        headers=headers)
    save(a.manifest.with_suffix(".metadata_receipt.json"),receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!="headers"}),flush=True)


def validate_selection(a,manifest):
    selection=json.loads(a.selection.read_text())
    if selection.get("state")=="REGISTERED_NO_EXPANSION":return selection
    if selection.get("state")!="FROZEN_SINGLE_PRIMARY" or selection.get("primary") not in ARMS:
        raise ValueError("Exactly one frozen primary required")
    controls=selection.get("naive")
    if not isinstance(controls,list) or any(arm not in ARMS for arm in controls):raise ValueError("Fixed registered controls list required")
    if (selection.get("confirmation_manifest_sha256")!=sha(a.manifest) or
        selection.get("dev_manifest_sha256")!=manifest["template_DEV_manifest_sha256"] or
        selection.get("source_hashes")!=manifest["source_hashes"] or
        not selection.get("selection_rule_version")):
        raise ValueError("Selection is not bound to these frozen cohorts and runtime")
    path=a.dev_analysis or selection.get("dev_analysis_path") or selection.get("dev_analysis")
    if not path or sha(path)!=selection.get("dev_analysis_sha256"):raise ValueError("Frozen completed DEV analysis required")
    analysis=json.loads(Path(path).read_text())
    if analysis.get("state")!="CPU_F4_SEVEN_ANALYZED" or analysis.get("episodes")!=241:
        raise ValueError("Primary must be selected from completed DEV241, not fresh or smoke8")
    if sha(manifest["template_DEV_manifest"])!=manifest["template_DEV_manifest_sha256"]:
        raise ValueError("Frozen DEV manifest changed")
    return selection


def run(a):
    manifest=json.loads(a.manifest.read_text())
    if (manifest.get("scope")!="registered_photo_disjoint_confirmation" or
        manifest.get("design_DEV") is not False or manifest.get("all_historical_unseen_claim") is not False or
        str(a.selection.resolve())!=manifest["selection_path"] or
        sha(__file__)!=manifest["confirmation_helper_sha256"]):
        raise ValueError("Exact registered-scope confirmation adapter required")
    if sha(manifest["exposure_audit"])!=manifest["exposure_audit_sha256"]:raise ValueError("Exposure audit changed")
    if sha(a.cpu_receipt)!=manifest["CPU_receipt_sha256"]:raise ValueError("Accepted CPU receipt changed")
    selection=validate_selection(a,manifest)
    if selection["state"]=="REGISTERED_NO_EXPANSION":
        a.out.mkdir(parents=True,exist_ok=True)
        save(a.out/("not_started_%d.json"%time.time_ns()),dict(state="NOT_STARTED_NO_EXPANSION",
            selection_sha256=sha(a.selection),manifest_sha256=sha(a.manifest),GPU_started=False))
        print(json.dumps(dict(state="NOT_STARTED_NO_EXPANSION",GPU_started=False)),flush=True);return
    if not a.allow_gpu or os.environ.get("DEMO9_F4_ROOT_LAUNCH")!="1":raise ValueError("Root-only explicit GPU authorization required")
    a.out.mkdir(parents=True,exist_ok=True)
    authorization=dict(state="PRIMARY_AND_CONTROLS_FROZEN_BEFORE_CONFIRMATION",
        primary=selection["primary"],controls=selection["naive"],manifest_sha256=sha(a.manifest),
        selection_sha256=sha(a.selection),DEV_analysis_sha256=selection["dev_analysis_sha256"],
        fresh_arm_reselection_allowed=False,all_historical_unseen_claim=False)
    gate=a.out/"confirmation_authorization.json"
    if gate.exists():
        if json.loads(gate.read_text())!=authorization:raise ValueError("Resume changed frozen primary/controls")
    else:save(gate,authorization)
    argv=[sys.executable,str(HERE/"f4_experiment.py"),"--run","--allow-gpu",
          "--manifest",str(a.manifest),"--cpu-receipt",str(a.cpu_receipt),"--out",str(a.out),
          "--budget-seconds",str(a.budget_seconds),"--memory-fraction",str(a.memory_fraction)]
    # The authorization receipt makes the output nonempty; resume is therefore
    # explicit even on the first inference launch. No finished cases are rescored.
    argv.append("--resume")
    print(json.dumps(dict(state="DELEGATING_FROZEN_CONFIRMATION",primary=selection["primary"],argv=argv)),flush=True)
    completed=subprocess.run(argv,check=False,env=os.environ.copy())
    if sha(a.selection)!=authorization["selection_sha256"]:raise RuntimeError("Selection changed during confirmation")
    raise SystemExit(completed.returncode)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument("--prepare",action="store_true");mode.add_argument("--run",action="store_true")
    p.add_argument("--source-manifest",type=Path);p.add_argument("--exposure-audit",type=Path);p.add_argument("--template",type=Path)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--cpu-receipt",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--dev-analysis",type=Path);p.add_argument("--allow-gpu",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--budget-seconds",type=int,default=1800);p.add_argument("--memory-fraction",type=float,default=.35)
    a=p.parse_args()
    prepare(a) if a.prepare else run(a)


if __name__=="__main__":main()
