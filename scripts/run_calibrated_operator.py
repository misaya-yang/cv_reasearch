#!/usr/bin/env python3
"""Reference-calibrated, query-label-free A/B selection on a sealed mask family.

A spatially excluded reference replay supplies class-conditional margin histograms.
Query mixture EM estimates its foreground proportion. Exact bounded operators then
maximize a ratio of expected counts. The probability transport assumption is tested,
not asserted. Direct-mask and unrestricted posterior controls use the same evidence.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
CONFIG = dict(reference_exclusion_radius=2, histogram_bins=32, pseudocount=.5,
              em_steps=100, em_tolerance=1e-8, initial_prior=.5,
              query_gt_in_inference=False, calibration_labels="single supplied reference coverage only",
              objective="ratio of expected pixel counts with tokenwise posterior; not expected realized IoU",
              optimizer="exact enumeration inside the sealed <=2-operation family",
              fallback="complete FoRIS if excluded reference calibration lacks either role")


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def check(path, digest):
    from ics.experiment import sha
    if sha(path) != digest: raise ValueError("Changed sealed input: "+str(path))


def bytes_count(mask):
    mask = mask - ((mask >> 1) & 85)
    mask = (mask & 51) + ((mask >> 2) & 51)
    return ((mask + (mask >> 4)) & 15).float()


def posterior(q, r, cov, exclusion, query_fg=None, query_bg=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    q, r = F.normalize(q.float(), dim=1), F.normalize(r.float(), dim=1)
    cover = np.asarray(cov, np.float64).reshape(-1)
    fg = torch.as_tensor(cover >= .5, device=q.device)
    if not bool(fg.any()) or not bool((~fg).any()): return None, {"fallback": "missing_reference_role"}
    def margins(sim, positive_role, negative_role, top=1):
        positive = sim[:, positive_role].topk(min(top, int(positive_role.sum())), dim=1).values.mean(1)
        negative = sim[:, negative_role].topk(min(top, int(negative_role.sum())), dim=1).values.mean(1)
        return positive-negative, torch.isfinite(positive)&torch.isfinite(negative)
    if query_fg is None:
        rr = r @ r.T; rr.masked_fill_(exclusion, -torch.inf)
        reference, valid = margins(rr, fg, ~fg); del rr
        query, good = margins(q @ r.T, fg, ~fg)
    else:
        if not bool(query_fg.any()) or not bool(query_bg.any()):
            return None, {"fallback": "missing_query_anchor_role"}
        pair = q @ r.T
        reference, valid = margins(pair.T, query_fg, query_bg, top=10)
        query, good = margins(pair, fg, ~fg, top=10)
    values, valid = reference.cpu().numpy(), valid.cpu().numpy()
    if cover[valid].sum() <= 1e-6 or (1-cover[valid]).sum() <= 1e-6:
        return None, {"fallback": "missing_excluded_reference_role"}
    edges = np.quantile(values[valid], np.linspace(0, 1, CONFIG["histogram_bins"]+1))[1:-1]
    reference_bin = np.searchsorted(edges, values[valid], side="right")
    positive = np.bincount(reference_bin, weights=cover[valid], minlength=CONFIG["histogram_bins"])+CONFIG["pseudocount"]
    negative = np.bincount(reference_bin, weights=1-cover[valid], minlength=CONFIG["histogram_bins"])+CONFIG["pseudocount"]
    positive /= positive.sum(); negative /= negative.sum()
    if not bool(good.all()): raise ValueError("Nonfinite query margin")
    qb = np.searchsorted(edges, query.cpu().numpy(), side="right")
    a, b = positive[qb], negative[qb]
    prior = CONFIG["initial_prior"]
    for step in range(CONFIG["em_steps"]):
        prob = prior*a / np.maximum(prior*a+(1-prior)*b, 1e-12)
        updated = float(np.clip(prob.mean(), 1e-6, 1-1e-6))
        if abs(updated-prior) < CONFIG["em_tolerance"]:
            prior=updated; break
        prior=updated
    prob = prior*a / np.maximum(prior*a+(1-prior)*b, 1e-12)
    return torch.as_tensor(prob, device=q.device, dtype=torch.float32), dict(fallback=None,
        prior=prior, em_steps=step+1, valid_reference_tokens=int(valid.sum()),
        reference_foreground_mass=float(cover[valid].sum()), reference_background_mass=float((1-cover[valid]).sum()),
        posterior_min=float(prob.min()), posterior_max=float(prob.max()), posterior_mean=float(prob.mean()))


def infer(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.experiment import load_rows, load_inputs, sha
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if a.out.exists(): raise FileExistsError("Fresh output required")
    parent_seal=json.loads((a.family/"sealed.json").read_text())
    check(a.family/"manifest.json",parent_seal["manifest_sha256"])
    check(a.family/"protocol.json",parent_seal["protocol_sha256"])
    family=json.loads((a.family/"protocol.json").read_text())
    all_rows=load_rows(a.family/"manifest.json")
    if len(all_rows)!=241: raise ValueError("Full DEV241 source required")
    rows=([all_rows[0],next(r for r in all_rows if r["fold"]!=all_rows[0]["fold"])] if a.smoke else all_rows)
    sources=family["sources"]
    for spec in sources.values():
        check(Path(spec["path"])/spec["seal_name"],spec["seal_sha256"])
    names, ops, recipes=family["library"], family["ops"], family["recipes"]
    a.out.mkdir();(a.out/"predictions").mkdir();(a.out/"posterior").mkdir()
    write(a.out/"manifest.json",rows)
    protocol=dict(config=CONFIG,family_protocol_sha256=sha(a.family/"protocol.json"),
        family_seal_sha256=sha(a.family/"sealed.json"),family_path=str(a.family),n=len(rows),
        input_receipts={},source_code={str(Path(__file__).resolve()):sha(__file__),
            str(REPO / "src/ics/experiment.py"):sha(REPO / "src/ics/experiment.py")},
        query_gt_in_inference=False,extra_encoder_forwards=0,origin=family["origin"],
        library=names,ops=ops,recipes=recipes,selection="per-query reference-only posterior; no DEV/query labels",
        resolution=1024,seed=0,smoke=a.smoke)
    write(a.out/"protocol.json",protocol)
    coordinates=torch.stack(torch.meshgrid(torch.arange(64,device="cuda"),torch.arange(64,device="cuda"),indexing="ij"),-1).reshape(-1,2).float()
    exclusion=torch.cdist(coordinates,coordinates,p=float("inf"))<=CONFIG["reference_exclusion_radius"]
    add_op=torch.tensor([op=="add" for op,_ in ops],device="cuda")
    first=torch.tensor([x for x,_ in recipes],device="cuda");second=torch.tensor([y for _,y in recipes],device="cuda")
    recipe_index={tuple(recipe):i for i,recipe in enumerate(recipes)}
    seals,posterior_seals,audit={},{},{};begin=time.monotonic()
    byte_test=torch.arange(256,device="cuda",dtype=torch.int64).to(torch.uint8)
    if not np.array_equal(bytes_count(byte_test).cpu().numpy(),[v.bit_count() for v in range(256)]):
        raise ValueError("Byte-count kernel parity failed")
    for num,row in enumerate(rows,1):
        key=row["key"];started=time.monotonic()
        (q,r,cov,score),receipt=load_inputs(a.root,row)
        protocol["input_receipts"][key]=receipt
        library={}
        for label,spec in sources.items():
            path=Path(spec["path"])/spec["directory"]/(key+".npz");check(path,spec["hashes"][key])
            with np.load(path,allow_pickle=False) as z:
                for name in spec["arms"]:library[label+":"+name]=z[name].copy()
        path=a.family/"predictions"/(key+".npz");check(path,parent_seal["predictions"][key])
        with np.load(path,allow_pickle=False) as z:
            comparisons={name:z[name].copy() for name in ("native","origin","rcg","astra.control","mean.control","insid3.control")}
        library["native"]=comparisons["native"]
        field_path=Path(sources["recheck"]["path"])/"fields"/(key+".npz")
        check(field_path,sources["recheck"]["fields"][key])
        with np.load(field_path,allow_pickle=False) as z:rcg=z["rcg"].copy()
        normalized=(score-score.min())/max(float(score.max()-score.min()),1e-6)
        for label,values in (("foris",normalized),("rcg",rcg)):
            up=F.interpolate(torch.as_tensor(values)[None,None],(1024,1024),mode="bilinear",align_corners=False)[0,0].numpy()
            for threshold in (.3,.4,.6,.7):library[f"field:{label}>{threshold:g}"]=np.packbits(up>threshold)
        if sorted(library)!=names:raise ValueError("Family membership changed")
        query_fg=query_bg=None
        if a.calibration=="cross":
            from ics.experiment import unpack
            cover=lambda mask:unpack(mask).reshape(64,16,64,16).mean((1,3)).reshape(-1)
            candidates=np.flatnonzero(cover(comparisons["astra.control"])>.5)
            anchors=candidates[np.argsort(-rcg.reshape(-1)[candidates],kind="stable")[:max(1,len(candidates)//2)]]
            query_fg=torch.zeros(4096,dtype=torch.bool,device="cuda");query_fg[anchors]=True
            outside=(cover(comparisons["native"])==0)&(cover(comparisons["rcg"])==0)&(cover(comparisons["mean.control"])==0)
            candidates=np.flatnonzero(outside)
            anchors=candidates[np.argsort(rcg.reshape(-1)[candidates],kind="stable")[:max(1,len(candidates)//2)]]
            query_bg=torch.zeros(4096,dtype=torch.bool,device="cuda");query_bg[anchors]=True
        probability,info=posterior(torch.as_tensor(q,device="cuda"),torch.as_tensor(r,device="cuda"),cov,exclusion,query_fg,query_bg)
        if query_fg is not None:info.update(query_foreground_anchors=int(query_fg.sum()),query_background_anchors=int(query_bg.sum()))
        masks=torch.as_tensor(np.stack([library[name] for name in names]),device="cuda")
        origin=torch.as_tensor(comparisons["origin"],device="cuda")
        if probability is None:
            final={name:comparisons["native"] for name in ("posterior.joint2","posterior.greedy2","posterior.direct.control","posterior.pixel.control","posterior.half.control")}
            probability=torch.full((4096,),.5,device="cuda")
        else:
            weights=probability.reshape(64,64).repeat_interleave(16,0).repeat_interleave(2,1).reshape(-1)
            area=probability.sum()*256
            def expected(mask):
                bits=bytes_count(mask);inter=(bits*weights).sum(-1);union=area+bits.sum(-1)-inter
                return inter/union.clamp_min(1e-6)
            op_masks=torch.stack([torch.full_like(origin,255) if name is None else masks[names.index(name)] for _,name in ops])
            results=torch.empty(len(recipes),device="cuda")
            for start in range(0,len(recipes),128):
                end=min(start+128,len(recipes));fi,se=first[start:end],second[start:end]
                intermediate=torch.where(add_op[fi,None],origin|op_masks[fi],origin&op_masks[fi])
                composed=torch.where(add_op[se,None],intermediate|op_masks[se],intermediate&op_masks[se])
                results[start:end]=expected(composed)
            choice=int(results.argmax());direct=int(expected(masks).argmax())
            greedy=int(results[:len(ops)].argmax());first_choice=recipes[greedy][0]
            pool=[recipe_index[(first_choice,j)] if first_choice and j else recipe_index[(first_choice or j,0)] for j in range(len(ops))]
            greedy2=pool[int(results[pool].argmax())]
            def apply(mask,index):
                return mask|op_masks[index] if add_op[index] else mask&op_masks[index]
            joint=apply(apply(origin,recipes[choice][0]),recipes[choice][1])
            gm=apply(apply(origin,recipes[greedy2][0]),recipes[greedy2][1])
            ordered,indices=torch.sort(probability,descending=True,stable=True)
            intersection=ordered.cumsum(0);unions=probability.sum()+torch.arange(1,4097,device="cuda")-intersection
            k=int((intersection/unions.clamp_min(1e-6)).argmax())+1
            pixels=torch.zeros(4096,dtype=torch.bool,device="cuda");pixels[indices[:k]]=True
            render=lambda x:np.packbits(x.reshape(64,64).repeat_interleave(16,0).repeat_interleave(16,1).cpu().numpy())
            final={"posterior.joint2":joint.cpu().numpy(),"posterior.greedy2":gm.cpu().numpy(),
                "posterior.direct.control":masks[direct].cpu().numpy(),"posterior.pixel.control":render(pixels),
                "posterior.half.control":render(probability>.5)}
            info.update(joint_recipe=recipes[choice],greedy2_recipe=recipes[greedy2],direct=names[direct],
                        expected_joint=float(results[choice]),expected_direct=float(expected(masks[direct:direct+1])[0]),
                        pixel_tokens=k,posterior_objective_certified_within_family=True)
            if info["expected_joint"]+1e-6<info["expected_direct"]:raise ValueError("Complete-mask family containment failed")
            if float((intersection/unions)[k-1])+1e-5<info["expected_joint"]:
                raise ValueError("Unrestricted posterior containment failed")
        path=a.out/"predictions"/(key+".npz");np.savez_compressed(path,**comparisons,**final);seals[key]=sha(path)
        np.savez_compressed(a.out/"posterior"/(key+".npz"),probability=probability.cpu().numpy())
        posterior_seals[key]=sha(a.out/"posterior"/(key+".npz"))
        audit[key]=dict(info,seconds=time.monotonic()-started)
        print(json.dumps(dict(n=num,total=len(rows),seconds=round(time.monotonic()-begin,1),fallback=info["fallback"])),flush=True)
    write(a.out/"protocol.json",protocol);write(a.out/"choices.json",audit)
    write(a.out/"sealed.json",dict(state="ALL_PREDICTIONS_SEALED",n=len(rows),
        manifest_sha256=sha(a.out/"manifest.json"),protocol_sha256=sha(a.out/"protocol.json"),
        choices_sha256=sha(a.out/"choices.json"),predictions=seals,
        posterior=posterior_seals,
        query_gt_used_for_DEV_fold_fitting=False,per_query_gt_routing=False,
        seconds=time.monotonic()-begin,cuda_peak_bytes=torch.cuda.max_memory_allocated()))


def score(a):
    import numpy as np
    from ics.experiment import packet,sha,summarize,unpack
    seal=json.loads((a.out/"sealed.json").read_text())
    if seal["state"]!="ALL_PREDICTIONS_SEALED" or (a.out/"report.json").exists():raise ValueError("Incomplete/already scored")
    for field,name in (("manifest_sha256","manifest.json"),("protocol_sha256","protocol.json"),("choices_sha256","choices.json")):check(a.out/name,seal[field])
    protocol=json.loads((a.out/"protocol.json").read_text())
    for path,digest in protocol["source_code"].items():check(path,digest)
    rows=json.loads((a.out/"manifest.json").read_text());arrays,corrections,details={},{},[]
    calibration=[]
    for row in rows:
        key=row["key"];receipt=protocol["input_receipts"][key]
        check(packet(a.root,row),receipt["packet_sha256"])
        with np.load(packet(a.root,row),allow_pickle=False) as z:truth=unpack(z["truth"])
        posterior_path=a.out/"posterior"/(key+".npz");check(posterior_path,seal["posterior"][key])
        with np.load(posterior_path,allow_pickle=False) as z:probability=z["probability"].copy()
        coverage=truth.reshape(64,16,64,16).mean((1,3)).reshape(-1)
        calibration.append(dict(key=key,brier=float(np.mean(probability**2-2*probability*coverage+coverage)),
            expected_area=float(probability.sum()*256),true_area=int(truth.sum()),
            posterior_mean=float(probability.mean()),true_fraction=float(coverage.mean())))
        path=a.out/"predictions"/(key+".npz");check(path,seal["predictions"][key])
        with np.load(path,allow_pickle=False) as z:masks={name:unpack(z[name]) for name in z.files}
        origin=masks["origin"];episode=dict(row,iu={})
        for name,mask in masks.items():
            iu=[int((mask&truth).sum()),int((mask|truth).sum())];arrays.setdefault(name,[]).append(iu);episode["iu"][name]=iu
            add,delete=mask&~origin,origin&~mask
            corrections.setdefault(name,[]).append(dict(key=key,c=row["c"],fold=row["fold"],batch=row.get("batch","unspecified"),
                add_TP=int((add&truth).sum()),add_FP=int((add&~truth).sum()),delete_TP=int((delete&truth).sum()),delete_FP=int((delete&~truth).sum())))
        details.append(episode)
    report,_=summarize(rows,{name:np.array(values) for name,values in arrays.items()},corrections)
    report["corrections_vs_origin"]=report.pop("corrections_vs_native")
    report["corrections_vs_origin_by_class"]=report.pop("corrections_by_class")
    report["corrections_vs_origin_by_batch"]=report.pop("corrections_by_batch")
    report.update(config=CONFIG,selection=protocol["selection"],prediction_seal_sha256=sha(a.out/"sealed.json"))
    report["reference_probability_transport_diagnostic"]=dict(
        mean_brier=float(np.mean([row["brier"] for row in calibration])),
        mean_area_bias=float(np.mean([row["expected_area"]-row["true_area"] for row in calibration])),
        mean_absolute_area_error=float(np.mean([abs(row["expected_area"]-row["true_area"]) for row in calibration])),
        episodes=calibration,role="GT scoring only; not an inference input")
    write(a.out/"report.json",report)
    (a.out/"episodes.jsonl").write_text("".join(json.dumps(row)+"\n" for row in details))
    print(json.dumps(dict(n=len(rows),scores=report["scores"])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("phase",choices=("infer","score"));p.add_argument("--root",type=Path,required=True)
    p.add_argument("--family",type=Path);p.add_argument("--out",type=Path,required=True);p.add_argument("--smoke",action="store_true")
    p.add_argument("--calibration",choices=("self","cross"),default="self")
    a=p.parse_args()
    CONFIG["calibration"]=a.calibration
    CONFIG["cross_calibration"]="symmetric cross-image top10 margins; reference labels; query anchor roles from sealed complete masks" if a.calibration=="cross" else None
    if a.phase=="infer" and a.family is None:p.error("--family required")
    {"infer":infer,"score":score}[a.phase](a)


if __name__=="__main__":main()
