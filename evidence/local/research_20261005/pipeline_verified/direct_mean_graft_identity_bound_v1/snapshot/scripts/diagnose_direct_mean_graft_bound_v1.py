#!/usr/bin/env python3
"""CPU-only identity/local-range change-envelope check; no new inference cue.

Validate once on sealed DirectMEAN1200: P(G)-[P(R)+I(G-R)] = (P-I)(G-R),
up to declared FP32 rounding. Generate label-free candidate-change bands from
existing4000 R/G/graft fields; after validation, GT is used only to score the
arbitrary-band-correction ceiling, explicitly not a method prediction/score.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from ics.experiment import metric,sha
EPS=np.finfo(np.float32).eps
GAMMA128=128*EPS/(1-128*EPS);GAMMA16=16*EPS/(1-16*EPS)
DIRECT="fine.mean16.control";GRAFT="mean_fine_residual_transfer_v1"
NEIGHBOR_CACHE={}


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+"\n")
def checked(p,h):
    if sha(p)!=h:raise ValueError("Changed sealed file: "+str(p))
def packed(v):
    if v.dtype!=np.uint8 or v.shape!=(131072,):raise ValueError("Invalid packed1024 mask")
    return v


def grids(job):
    row,base,graft,bs,gs=job
    key=row["key"];bp=Path(base)/"fields"/(key+".npz");gp=Path(graft)/"fields"/(key+".npz")
    checked(bp,bs["fields"][key]);checked(gp,gs["fields"][key])
    with np.load(bp,allow_pickle=False) as z:r,y=z["rcg"].copy(),z["fine.rcg16.control"].copy()
    with np.load(gp,allow_pickle=False) as z:g,c=z["mean.control"].copy(),z[GRAFT].copy()
    for v,size in ((r,64),(g,64),(y,128),(c,128)):
        if v.dtype!=np.float32 or v.shape!=(size,size) or not np.isfinite(v).all():raise ValueError("Invalid source field header")
    return r,g,y,c


def envelope(r,g,y,c):
    import torch
    import torch.nn.functional as F
    from scipy.ndimage import maximum_filter,minimum_filter
    torch.set_num_threads(1)
    delta=g.astype(np.float64)-r.astype(np.float64)
    local64=maximum_filter(delta,size=5,mode="nearest")-minimum_filter(delta,size=5,mode="nearest")
    local128=local64.repeat(2,0).repeat(2,1)
    scale=1.+max(float(np.abs(v).max()) for v in (r,g,y,c))*4
    tolerance=GAMMA128*scale
    render_tolerance=GAMMA16*(1.+float(np.abs(c).max())+float(np.abs(g).max()))
    up=lambda v:F.interpolate(torch.from_numpy(np.asarray(v,np.float64))[None,None],(1024,1024),mode="bilinear",align_corners=False)[0,0].numpy()
    cup=up(c);bound=up(local128+tolerance)+render_tolerance
    band=np.abs(cup-.5)<=bound
    return delta,local128,tolerance,band,dict(band_pixels=int(band.sum()),mean_local_range=float(local128.mean()),max_local_range=float(local128.max()),
        FP32_field_tolerance=tolerance,FP32_render_tolerance=render_tolerance)


def validate_one(job):
    fullrow,oldrow,base,graft,mean,bs,gs,ms,out=job
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    if mean not in NEIGHBOR_CACHE:
        NEIGHBOR_CACHE[mean]=(np.load(Path(mean)/"neighbors.npy",allow_pickle=False),np.load(Path(mean)/"valid_neighbors.npy",allow_pickle=False))
    neighbors,valid=NEIGHBOR_CACHE[mean]
    r,g,y,c=grids((fullrow,base,graft,bs,gs));delta,local,tol,band,stats=envelope(r,g,y,c)
    key=oldrow["key"];mp=Path(mean)/"fields"/(key+".npz");kp=Path(mean)/"kernels"/(key+".npz")
    checked(mp,ms["fields"][key]);checked(kp,ms["kernels"][key])
    with np.load(mp,allow_pickle=False) as z:pg,pr=z[DIRECT].copy(),z["fine.rcg16.control"].copy()
    if not np.array_equal(pr,y):raise ValueError("Same original P(R) field required")
    with np.load(kp,allow_pickle=False) as z:w,den=z["weights"].copy(),z["denominator"].copy()
    if w.shape!=(16384,25) or den.shape!=(16384,) or w.dtype!=np.float32 or den.dtype!=np.float32 or (w<0).any() or (den<=0).any() or np.any(w[~valid]!=0):raise ValueError("Nonnegative source local P kernel required")
    w64=w.astype(np.float64);den64=den.astype(np.float64);mass=w64.sum(1)/den64
    mass_error=float(np.abs(mass-1).max())
    if mass_error>8*EPS:raise ValueError("Source P normalization/clamp condition failed")
    pdelta=((w64*delta.ravel()[neighbors]).sum(1)/den64).reshape(128,128)
    idelta=F.interpolate(torch.from_numpy(delta)[None,None],(128,128),mode="bilinear",align_corners=False)[0,0].numpy()
    actual=pg.astype(np.float64)-c.astype(np.float64);predicted=pdelta-idelta
    identity_error=float(np.abs(actual-predicted).max())
    violations=int((np.abs(actual)>local+tol).sum())
    if identity_error>tol or violations:raise ValueError("Source identity/local-range field bound violated")
    pp=Path(mean)/"predictions"/(key+".npz");cp=Path(graft)/"predictions"/(fullrow["key"]+".npz")
    checked(pp,ms["predictions"][key]);checked(cp,gs["predictions"][fullrow["key"]])
    with np.load(pp,allow_pickle=False) as z:m=packed(z[DIRECT])
    with np.load(cp,allow_pickle=False) as z:cm=packed(z[GRAFT])
    xor=np.unpackbits(m^cm).reshape(1024,1024).astype(bool)
    uncovered=int((xor&~band).sum())
    if uncovered:raise ValueError("Actual DirectMEAN/graft mask XOR escaped label-free band")
    path=Path(out)/"bands"/(fullrow["key"]+".npz");np.savez_compressed(path,band=np.packbits(band))
    return dict(key=fullrow["key"],identity_max_error=identity_error,field_bound_violations=violations,source_P_mass_max_error=mass_error,
        actual_XOR_pixels=int(xor.sum()),XOR_outside_band=uncovered,band_sha256=sha(path),**stats)


def oracle_one(job):
    row,base,graft,bs,gs,out,existing_band=job
    key=row["key"];bp=Path(out)/"bands"/(key+".npz")
    if existing_band:
        checked(bp,existing_band["band_sha256"])
        with np.load(bp,allow_pickle=False) as z:band=np.unpackbits(packed(z["band"])).reshape(1024,1024).astype(bool)
        stats={k:existing_band[k] for k in ("band_pixels","mean_local_range","max_local_range","FP32_field_tolerance","FP32_render_tolerance")}
    else:
        r,g,y,c=grids((row,base,graft,bs,gs));_,_,_,band,stats=envelope(r,g,y,c)
        np.savez_compressed(bp,band=np.packbits(band))
    # Band construction above uses fields only. GT enters ONLY this ceiling phase.
    cp=Path(graft)/"predictions"/(key+".npz");checked(cp,gs["predictions"][key])
    packet=row["packet_export"];checked(packet,gs["inputs"][key]["packet_sha256"])
    with np.load(cp,allow_pickle=False) as z:mask=np.unpackbits(packed(z[GRAFT])).reshape(1024,1024).astype(bool)
    with np.load(packet,allow_pickle=False) as z:truth=np.unpackbits(packed(z["truth"])).reshape(1024,1024).astype(bool)
    i,u=int((mask&truth).sum()),int((mask|truth).sum());fn=int((band&truth&~mask).sum());fp=int((band&~truth&mask).sum())
    return dict(key=key,graft_IU=[i,u],GT_band_oracle_IU=[i+fn,u-fp],GT_FN_inside_band=fn,GT_FP_inside_band=fp,
        total_FN=int((truth&~mask).sum()),total_FP=int((mask&~truth).sum()),band_sha256=sha(bp),**stats)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",type=Path,default=Path("outputs/frozen_subtoken4000_v1"));p.add_argument("--graft",type=Path,default=Path("outputs/mean_fine_residual_transfer4000_v1"));p.add_argument("--mean",type=Path,default=Path("outputs/fine_mean1200_v1"));p.add_argument("--out",type=Path,default=Path("outputs/direct_mean_graft_identity_bound_v1"));p.add_argument("--workers",type=int,default=2);a=p.parse_args()
    if not 1<=a.workers<=2:raise ValueError("This diagnostic is limited to at most2 CPU workers")
    if a.out.exists():raise FileExistsError("Fresh bounded diagnostic output required")
    start=time.monotonic();bs,gs,ms=[read(p/"sealed.json") for p in (a.base,a.graft,a.mean)]
    if bs["n"]!=4000 or gs["n"]!=4000 or ms["n"]!=1200 or ms["state"]!="ALL_PREDICTIONS_AND_KERNELS_SEALED":raise ValueError("Complete sealed4000/1200 sources required")
    for path,seal in ((a.base,bs),(a.graft,gs),(a.mean,ms)):
        checked(path/"manifest.json",seal["manifest_sha256"]);checked(path/"config.json",seal["config_sha256"])
    rows=read(a.graft/"manifest.json");old=read(a.mean/"manifest.json");lookup={r["key"]:r for r in rows}
    neighbors=np.load(a.mean/"neighbors.npy",allow_pickle=False);valid=np.load(a.mean/"valid_neighbors.npy",allow_pickle=False)
    checked(a.mean/"neighbors.npy",ms["neighbors_sha256"]);checked(a.mean/"valid_neighbors.npy",ms["valid_neighbors_sha256"])
    fi=np.arange(128);base=fi//2;ti=base[:,None]+np.arange(-2,3)[None];ok=(ti>=0)&(ti<64);clipped=ti.clip(0,63)
    expected=(clipped[:,None,:,None]*64+clipped[None,:,None,:]).reshape(16384,25);ev=(ok[:,None,:,None]&ok[None,:,None,:]).reshape(16384,25)
    if not np.array_equal(neighbors,expected) or not np.array_equal(valid,ev):raise ValueError("Actual P/I local support differs")
    cfg=read(a.mean/"config.json");taus=[]
    for f in range(4):
        spec=cfg["parameters"][str(f)]
        if spec["sigma"]!=1.25 or spec["tau"]not in (.07,.15):raise ValueError("Locked readout policy differs")
        taus.append(spec["tau"])
    distance=((ti*16+8)-(fi*8+4)[:,None])/16
    spatial=(np.exp(-(distance[:,None,:,None]**2+distance[None,:,None,:]**2)/(2*1.25**2))*ev.reshape(128,128,5,5)).reshape(16384,25).sum(1)
    cosine_guard=4096*EPS/(1-4096*EPS)
    lower=float(spatial.min()*np.exp((-2-cosine_guard)/min(taus)))
    if lower<=1e-12:raise ValueError("Cannot certify non-clamped stochastic P for all4000")
    a.out.mkdir(parents=True);(a.out/"bands").mkdir()
    jobs=[]
    for r in old:
        public="public%d:%s"%(r["public_batch"],r.get("source_key",r["key"]));fr=lookup[public]
        if any(r[k]!=fr[k] for k in ("c","fold","e","support","query")):raise ValueError("Original1200 draw identity differs")
        perbs={"fields":{public:bs["fields"][public]}};pergs={"fields":{public:gs["fields"][public]},"predictions":{public:gs["predictions"][public]}}
        perms={k:{r["key"]:ms[k][r["key"]]} for k in ("fields","kernels","predictions")}
        jobs.append((fr,r,str(a.base),str(a.graft),str(a.mean),perbs,pergs,perms,str(a.out)))
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn")) as pool:validation=list(pool.map(validate_one,jobs,chunksize=1))
    val={r["key"]:r for r in validation};write(a.out/"validation1200.json",dict(state="ALL1200_IDENTITY_FIELD_BOUND_AND_MASK_XOR_VALIDATED",rows=validation,
        local_constant_delta="Pdelta=Idelta when delta is constant on shared support, modulo declared FP32 rounding",static_min_P_denominator_with_F32_cosine_guard=lower,
        source_P_normalization_max_error=max(r["source_P_mass_max_error"] for r in validation),identity_max_error=max(r["identity_max_error"] for r in validation),
        actual_XOR_pixels=sum(r["actual_XOR_pixels"] for r in validation),XOR_outside_band=sum(r["XOR_outside_band"] for r in validation),query_GT_opened=False))
    # This proceeds only after the entire real1200 identity/XOR validation passed.
    ojobs=[]
    for r in rows:
        key=r["key"];perbs={"fields":{key:bs["fields"][key]}};pergs={"fields":{key:gs["fields"][key]},"predictions":{key:gs["predictions"][key]},"inputs":{key:gs["inputs"][key]}}
        ojobs.append((r,str(a.base),str(a.graft),perbs,pergs,str(a.out),val.get(key)))
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn")) as pool:oracle=list(pool.map(oracle_one,ojobs,chunksize=4))
    cls=np.array([r["c"] for r in rows]);iu=np.array([r["graft_IU"] for r in oracle],np.int64);upper=np.array([r["GT_band_oracle_IU"] for r in oracle],np.int64)
    with np.load(a.graft/"counts.npz",allow_pickle=False) as z:
        if not np.array_equal(iu,z["iu:"+GRAFT]):raise ValueError("Existing graft4000 I/U parity failed")
    bands=np.array([r["band_pixels"] for r in oracle]);report=dict(state="DIRECT_MEAN_GRAFT_IDENTITY_AND_CHANGE_ENVELOPE_COMPLETE",n=4000,real_direct_mean_validation_n=1200,
        identity="P(G)-[P(R)+I(G-R)] = (P-I)(G-R), evaluated against actual ordered FP32 graft with conservative operation-count rounding",
        bound="abs(field_difference128)<=valid5x5_localrange(G-R)+FP32_tolerance; mask changes lie within abs(graft_up-.5)<=I128to1024(localrange+tol)+render_tol",
        rounding=dict(FP32_epsilon=float(EPS),gamma128=float(GAMMA128),gamma16=float(GAMMA16),field_scale="1+4*maxabs(R,G,P(R),graft)",renderer="CPU float64 evaluation of unchanged bilinear coordinates; declared FP32 renderer allowance"),
        validation1200=dict(identity_max_error=max(r["identity_max_error"] for r in validation),field_bound_violations=0,actual_XOR_pixels=sum(r["actual_XOR_pixels"] for r in validation),XOR_outside_band=0,
            total_band_pixels=sum(r["band_pixels"] for r in validation),stochastic_P_mass_max_error=max(r["source_P_mass_max_error"] for r in validation),static_min_denominator=lower,cosine_rounding_guard=float(cosine_guard)),
        unlabelled_band4000=dict(total_pixels=int(bands.sum()),fraction_of_all_pixels=float(bands.sum()/(4000*1024**2)),episode_fraction_percentiles=np.percentile(bands/(1024**2),[0,25,50,75,90,100]).tolist()),
        GT_informed_ceiling4000=dict(graft_class_miou=metric(iu,cls),arbitrary_band_correction_upper_miou=metric(upper,cls),upper_gain_pp=metric(upper,cls)-metric(iu,cls),
            GT_FN_inside_band=sum(r["GT_FN_inside_band"] for r in oracle),GT_FP_inside_band=sum(r["GT_FP_inside_band"] for r in oracle),
            all_FN=sum(r["total_FN"] for r in oracle),all_FP=sum(r["total_FP"] for r in oracle),role="GT-informed arbitrary-correction ceiling ONLY, not a method score or expected DirectMEAN gain"),
        source=dict(base=str(a.base.resolve()),graft=str(a.graft.resolve()),mean=str(a.mean.resolve()),base_seal_sha256=sha(a.base/"sealed.json"),graft_seal_sha256=sha(a.graft/"sealed.json"),mean_seal_sha256=sha(a.mean/"sealed.json")),
        interpretation="Local-constant delta predicts equality. The envelope predicts where flips are possible, not their direction or probability. A high ceiling cannot predict superiority; actual full4000 DirectMEAN remains necessary.",
        workers=a.workers,new_encoder=False,new_inference_cue=False,script_sha256=sha(Path(__file__)),seconds=time.monotonic()-start)
    write(a.out/"report.json",report);np.savez_compressed(a.out/"oracle_counts.npz",graft_IU=iu,GT_band_oracle_IU=upper)
    write(a.out/"band_manifest.json",dict(rows=rows,bands={r["key"]:r["band_sha256"] for r in oracle},construction_query_GT_free=True,GT_used_only_for_ceiling_after_validation=True))
    (a.out/"report.md").write_text("# DirectMEAN/graft identity and change envelope\n\n"+f"Real1200: max identity error {report['validation1200']['identity_max_error']:.9g}; all field bounds pass and all {report['validation1200']['actual_XOR_pixels']:,} changed pixels lie inside the label-free envelope.\n\n"+f"The4000 envelope occupies {100*report['unlabelled_band4000']['fraction_of_all_pixels']:.3f}% of pixels. Graft mIoU {report['GT_informed_ceiling4000']['graft_class_miou']:.6f}; arbitrary GT-corrected band ceiling {report['GT_informed_ceiling4000']['arbitrary_band_correction_upper_miou']:.6f}. This ceiling is not a method result or expected gain.\n\nLocal-constant delta predicts equality. The field-derived envelope gives an a priori flip domain and capacity bound; it gives no probability or direction. Full4000 DirectMEAN comparison is still required. No new encoder, cue, parameter or source mutation.\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
