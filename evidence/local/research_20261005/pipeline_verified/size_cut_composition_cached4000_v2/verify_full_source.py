"""CPU-only physical mask/source verification after the supplied4000 seal."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--candidate",type=Path,required=True)
p.add_argument("--source",type=Path,required=True)
p.add_argument("--cached",type=Path,required=True)
p.add_argument("--prior",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args()
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
load=lambda d,n:json.loads((d/n).read_text())
base=("native","rcg","mean.control","rcg64.control","fine.rcg16.control","fine.rcg64")
primary="provided.rcg_tau15_sizecut"
half="provided.tau15.control"
cache_primary="provided_uniform_t15_size_cut.primary"
control="fixed_size_cut_on_foldtemp_fine16.control"
cs,ss,ks=load(a.candidate,"sealed.json"),load(a.source,"sealed.json"),load(a.cached,"sealed.json")
if cs["state"]!="ALL_PREDICTIONS_SEALED" or cs["n"]!=4000:raise ValueError("Full supplied4000 seal is required")
rows=load(a.candidate,"manifest.json");original=load(a.source,"manifest.json")
if len(rows)!=4000 or [r["key"] for r in rows]!=[r["key"] for r in original]:raise ValueError("Preserve all4000 draw identities/order")
for r,s in zip(rows,original):
    if any(r[k]!=s[k] for k in ("c","fold")) or any(Path(r[k]).name!=Path(s[k]).name for k in ("support","query")):
        raise ValueError("Class/photo identity differs")
prior={r["key"]:r for r in [json.loads(l) for l in (a.prior/"episodes.jsonl").read_text().splitlines()]}
lookup=np.array([i.bit_count() for i in range(256)],np.uint8)
arrays={arm:[] for arm in (*base,half,primary,control)}
matching=0
for n,row in enumerate(rows,1):
    key=row["key"];cp=a.candidate/"predictions"/(key+".npz");sp=a.source/"predictions"/(key+".npz");kp=a.cached/"predictions"/(key+".npz")
    cf=a.candidate/"fields"/(key+".npz")
    if sha(cp)!=cs["predictions"][key] or sha(sp)!=ss["predictions"][key] or sha(kp)!=ks["predictions"][key] or sha(cf)!=cs["fields"][key]:
        raise ValueError("Changed sealed mask/field file")
    with np.load(cp,allow_pickle=False) as z:masks={arm:z[arm].copy() for arm in (*base,half,primary)}
    with np.load(sp,allow_pickle=False) as z:
        if any(not np.array_equal(masks[arm],z[arm]) for arm in base):raise ValueError("Original six masks differ pixelwise")
    with np.load(kp,allow_pickle=False) as z:
        masks[control]=z[control].copy()
        if row["fold"] in (1,2):
            if not np.array_equal(masks[primary],z[cache_primary]):raise ValueError("Original matching2000 primary differs bytewise")
            matching+=1
    with np.load(cf,allow_pickle=False) as z:
        field=z[half].copy()
        if field.dtype!=np.float32 or field.shape!=(128,128) or not np.isfinite(field).all():raise ValueError("Unexpected uniformtau15 field")
        if row["fold"] in (1,2):
            with np.load(a.source/"fields"/(key+".npz"),allow_pickle=False) as old:
                if not np.array_equal(field,old["fine.rcg16.control"]):raise ValueError("Matching2000 field changed")
    packet=Path(original[n-1]["packet_export"])
    if sha(packet)!=ks["inputs"][key]["packet_sha256"]:raise ValueError("Changed canonical GT packet")
    with np.load(packet,allow_pickle=False) as z:truth=z["truth"].copy()
    if truth.shape!=(131072,) or truth.dtype!=np.uint8:raise ValueError("Unexpected packed GT")
    for arm,mask in masks.items():
        if mask.shape!=truth.shape or mask.dtype!=truth.dtype:raise ValueError("Unexpected packed mask")
        iu=[int(lookup[mask&truth].sum()),int(lookup[mask|truth].sum())]
        if arm in base and iu!=prior[key]["iu"][arm]:raise ValueError("Prior original4000 I/U mismatch")
        arrays[arm].append(iu)
    if n%800==0:print(json.dumps(dict(verified=n,n=4000)),flush=True)
if matching!=2000:raise ValueError("Allmatching2000 predictions must be reused")
a.out.mkdir(parents=True,exist_ok=False)
np.savez_compressed(a.out/"independent_counts.npz",**{"iu:"+k:np.array(v,np.int64) for k,v in arrays.items()})
(a.out/"manifest.json").write_text(json.dumps(original)+"\n")
result=dict(state="COMPLETE_SUPPLIED4000_MASKS_PHYSICALLY_VERIFIED",n=4000,all_draw_identities_preserved=True,
    all_candidate_mask_field_and_canonical_packet_hashes_verified=True,all4000_six_original_masks_pixelwise_and_IU_exact=True,
    all2000_cached_primary_masks_bytewise_exact=True,all2000_cached_uniform_fields_FP32_exact=True,
    no_GPU_encoder_images_or_new_parameters=True,GT_opened_only_after_complete_candidate_seal=True,
    primary=primary,candidate_seal_sha256=sha(a.candidate/"sealed.json"),cache_seal_sha256=sha(a.cached/"sealed.json"))
(a.out/"physical_verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
