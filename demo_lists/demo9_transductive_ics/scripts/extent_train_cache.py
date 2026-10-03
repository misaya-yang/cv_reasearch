#!/usr/bin/env python3
"""Inputs for the extent head: one public FoRIS pass per episode (refinement skipped, it does not change the score),
then the relation maps of tics/relations.py, a few principal components of the query features, and the target.

  python scripts/extent_train_cache.py --prepare --test-manifest results/extent_v1/episodes.json --per-fold 600 \
      --manifest results/extent_head_v0/train_episodes.json                       # CPU: episodes 60..659 per fold
  python scripts/extent_train_cache.py --prepare --test-manifest results/extent_v1/episodes.json --skip 600 --per-fold 150 \
      --avoid results/extent_head_v0/train_episodes.json --manifest results/extent_head_v0/confirm_episodes.json
                                                                                  # CPU: episodes 660..809, never looked at
  python scripts/extent_train_cache.py --manifest results/extent_head_v0/train_episodes.json \
      --confirm-manifest results/extent_head_v0/confirm_episodes.json \
      --test-manifest results/extent_v1/episodes.json --pool cache/evidence_v1/pool.pt --out cache/extent_head_v0
  python scripts/extent_train_cache.py --replay --test-manifest results/extent_v1/episodes.json \
      --features cache/evidence_v1/feat --packets results/extent_v1/run/packets --out cache/extent_head_v0
  python scripts/extent_train_cache.py --fixture DIR      # CPU; DIR holds finished extent and evidence fixtures

T1 --isolated-suite explicitly replaces colliding metadata draws; it is NOT the exact nominal ranges below.
Four PCA bases fit only other-three-collection TRAIN photos. Confirmation caches have no tf.

The live pass also redoes the first test episodes, and the replay compares its own maps with them: both routes must
give the same inputs. Training episodes are the standard seed-0 draws after the first 60 of each fold; an episode
that shares an image with a test episode is dropped, and a confirmation episode that shares an image with a
training episode is dropped as well.
"""
import argparse
import json
import os
import re
import struct
import zipfile
import pickletools
import hashlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from extent_experiment import build_host, coco_episodes, run_foris  # noqa: E402

CHECK = 8  # test episodes redone live, to compare with the replay


def prepare(a):
    test = json.loads(Path(a.test_manifest).read_text())
    seen = {x for r in test["episodes"] for x in (r["support"], r["query"])}
    if a.avoid:
        seen |= {x for r in json.loads(Path(a.avoid).read_text())["episodes"] for x in (r["support"], r["query"])}
    first = max(r["e"] for r in test["episodes"] if not r.get("dev40")) + 1 + a.skip
    rows, dropped = [], 0
    for f in range(4):
        draws = coco_episodes(test["data_root"], f, first + a.per_fold)
        for r in test["episodes"]:  # the standard list must be the one the test episodes came from
            if r["fold"] == f and r["e"] < len(draws) and (r["c"], r["query"], r["support"]) != draws[r["e"]]:
                raise SystemExit("episode list differs from the test manifest: %s" % r)
        for e in range(first, first + a.per_fold):
            c, t, s = draws[e]
            if s in seen or t in seen:
                dropped += 1
                continue
            for name in (s, t):
                if not (Path(test["data_root"]) / name).is_file() or not (Path(test["annotation_root"]) / Path(name).with_suffix(".png")).is_file():
                    raise SystemExit("missing file: " + name)
            rows.append(dict(fold=f, e=e, c=c, support=s, query=t))
    man = dict(state="PREPARED", episodes=rows, **{k: test[k] for k in ("data_root", "annotation_root", "foris_root", "projection_basis")})
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text(json.dumps(man))
    print(json.dumps(dict(state="PREPARED", episodes=len(rows), dropped_for_shared_images=dropped, first_index=first)))


def uid(name):
    m = re.search(r"(\d+)$", Path(name).stem)
    if m is None: raise ValueError("COCO UID absent: " + name)
    return int(m.group(1))


def role_uids(rows):
    return {uid(r[k]) for r in rows for k in ("support", "query")}


def pool_exposure(path):
    with zipfile.ZipFile(path) as z:
        key = next(x for x in z.namelist() if x.endswith("data.pkl"))
        names = [v for op,v,_ in pickletools.genops(z.read(key))
                 if op.name in ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE") and isinstance(v,str)]
    return {uid(x) for x in names if re.fullmatch(r"COCO_val2014_\d+",x)}


def png_shape(path):
    with open(path,"rb") as f: raw=f.read(24)
    if raw[:8]!=b"\x89PNG\r\n\x1a\n" or raw[12:16]!=b"IHDR": raise ValueError("invalid PNG header")
    w,h=struct.unpack(">II",raw[16:24]);return [h,w]


def prepare_isolated(a):
    from PIL import Image
    test=json.loads(Path(a.test_manifest).read_text());dev=role_uids(test["episodes"])
    exposed=pool_exposure(a.exposure_pool)
    if len(exposed)!=1200: raise SystemExit("prior exposure scope must be audited 1200 photos")
    root=Path(a.isolated_suite);root.mkdir(parents=True,exist_ok=True)
    if any((root/(n+"_episodes.json")).exists() for n in ("train","confirm","dev")): raise SystemExit("fresh manifests required")
    draws={f:coco_episodes(test["data_root"],f,20000) for f in range(4)}
    for r in test["episodes"]:
        if (r["c"],r["query"],r["support"])!=draws[r["fold"]][r["e"]]:raise SystemExit("frozen DEV seed draw mismatch")
    stats={}
    def sample(kind,first,count,blocked):
        rows=[];seen=set();fs={}
        for f in range(4):
            rejects=dict(blocked_UID=0,repeated_pair=0);picked=0
            for e in range(first,len(draws[f])):
                c,q,s=draws[f][e];key=(f,c,q,s)
                if uid(q) in blocked or uid(s) in blocked:rejects["blocked_UID"]+=1;continue
                if key in seen:rejects["repeated_pair"]+=1;continue
                if q==s:raise SystemExit("identical support/query")
                meta={}
                for role,name in (("support",s),("query",q)):
                    rgb=Path(test["data_root"])/name;label=Path(test["annotation_root"])/Path(name).with_suffix(".png")
                    if not rgb.is_file() or not label.is_file():raise SystemExit("missing existing asset: "+name)
                    with Image.open(rgb) as im:meta[role+"_original_shape"]=[im.height,im.width]
                    meta[role+"_mask_shape"]=png_shape(label)
                rows.append(dict(fold=f,collection_fold=f,e=e,c=c,support=s,query=q,role=kind,standard_draw_seed=0,
                                 query_mask_path=str(Path(test["annotation_root"])/Path(q).with_suffix(".png")),**meta))
                seen.add(key);picked+=1
                if picked==count:break
            if picked!=count:raise SystemExit("finite metadata sample exhausted")
            own=[r for r in rows if r["fold"]==f];classes=sorted({r["c"] for r in own})
            if classes!=list(range(f,80,4)):raise SystemExit("incomplete class coverage: do not drop classes")
            fs[str(f)]=dict(**rejects,first_kept=min(r["e"] for r in own),last_kept=max(r["e"] for r in own),
                           outside_nominal_range=sum(r["e"]>=first+count for r in own),classes=classes)
        stats[kind]=dict(episodes=len(rows),unique_UID=len(role_uids(rows)),
                         repeated_role_occurrences=2*len(rows)-len(role_uids(rows)),collection_folds=fs)
        return rows
    conf=sample("confirmation",660,150,dev|exposed)
    train=sample("training",60,600,dev|role_uids(conf))
    groups=[dev,role_uids(conf),role_uids(train)]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise SystemExit("UID isolation failed")
    protocol=dict(name="T1_EXPLICIT_IMAGE_ISOLATED_VARIANT",sampling_seed=0,not_exact_nominal_ranges=True,
                  not_claimed_never_seen=True,query_labels_are_not_inputs=True,confirmation_query_mask_pixels_read=False,
                  historical_exposure_scope="frozen DEV241 and prior 1200-photo pool; no claim of broader historical audit",
                  prior_pool_path=str(a.exposure_pool),exposed_pool_UIDs=sorted(exposed),
                  PCA_scope="per model f: only training collection folds !=f; excludes DEV/confirmation; NO OLD-POOL FEATURE reuse; training photos may have historical exposure")
    common={k:test[k] for k in ("data_root","annotation_root","foris_root","projection_basis")}
    for name,rows in (("train",train),("confirm",conf)):
        (root/(name+"_episodes.json")).write_text(json.dumps(dict(state="PREPARED",episodes=rows,protocol=protocol,**common)))
    (root/"dev_episodes.json").write_bytes(Path(a.test_manifest).read_bytes())
    receipt=dict(state="MANIFESTS_PREPARED_METADATA_ONLY",protocol=protocol,groups=stats,
                 DEV_episodes=len(test["episodes"]),DEV_unique_UID=len(dev),UID_intersections_all_zero=True,
                 prior_pool_UID_overlap=dict(train=len(role_uids(train)&exposed),confirmation=len(role_uids(conf)&exposed),DEV=len(dev&exposed)),
                 first3_train=train[:3],first3_confirmation=conf[:3],
                 manifests={n:dict(path=str(root/(n+"_episodes.json")),sha256=hashlib.sha256((root/(n+"_episodes.json")).read_bytes()).hexdigest())
                            for n in ("train","confirm","dev")})
    (root/"sampling_receipt.json").write_text(json.dumps(receipt,indent=1))
    print(json.dumps(dict(state=receipt["state"],groups=stats,receipt=str(root/"sampling_receipt.json"))),flush=True)


def fit_bases(prefix,k,device,fixture=False):
    import torch
    means,vs=[],[]
    for f in range(4):
        allowed=[x for cf,x in prefix if cf!=f or fixture]
        if not allowed:raise RuntimeError("no legitimate PCA prefix for model "+str(f))
        x=torch.cat(allowed).to(device).float();mean=x.mean(0)
        torch.manual_seed(31083+f)
        v=torch.pca_lowrank(x-mean,q=min(k,x.shape[1],x.shape[0]),center=False)[2][:,:k]
        if v.shape[1]<k:v=torch.cat([v,torch.zeros(v.shape[0],k-v.shape[1],device=v.device,dtype=v.dtype)],dim=1)
        means.append(mean.cpu());vs.append(v.cpu())
    return torch.stack(means),torch.stack(vs)


def pack(maps,comp,tf,side,ref_area,valid):
    import numpy as np
    result=dict(maps=maps.view(-1,side,side).cpu().numpy().astype(np.float16),
                pca=comp.permute(0,2,1).reshape(4,comp.shape[-1],side,side).cpu().numpy().astype(np.float16),
                pca_model_valid=np.asarray(valid,dtype=np.bool_),ref_area=np.float32(ref_area))
    if tf is not None:result["tf"]=tf.cpu().numpy().astype(np.float16)
    return result


def live(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tics.relations import near_mask, relation_maps
    if a.fixture:
        root = Path(a.fixture)
        a.manifest = a.test_manifest = a.confirm_manifest = str(root / "episodes.json")
        a.pool, a.out = str(root / "cache/pool.pt"), str(root / "head")
        a.packets = str(root / "run/packets")
    elif not a.unguarded and os.environ.get("DEMO9_CUDA_GUARD") != "1":
        raise SystemExit("run under scripts/experiment_resource_guard.py, or pass --unguarded")
    man, test = json.loads(Path(a.manifest).read_text()), json.loads(Path(a.test_manifest).read_text())
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("fresh output directory required")
    (out / "train").mkdir(parents=True)
    (out / "check").mkdir()
    (out / "confirm").mkdir()
    confirm = json.loads(Path(a.confirm_manifest).read_text())["episodes"] if a.confirm_manifest else []
    dev = "cpu" if a.fixture else "cuda"
    report = dict(state="RUNNING", episodes=0, skipped=0, checked=0, confirmation=0, native_checks=0, native_bit_identical=0)
    start = time.monotonic()

    def save():
        report["elapsed_s"] = time.monotonic() - start
        (out / "report.json").write_text(json.dumps(report, indent=1))
    save()
    try:
        data, ann = Path(man["data_root"]), Path(man["annotation_root"])
        with torch.inference_mode():
            host = build_host(a, man, dev)
            original_finalize = host._finalize_mask
            skip_finalize = lambda mask, image: mask
            host._finalize_mask = skip_finalize  # PCA/training/confirmation need score maps, not CRF
            if not a.packets:
                raise RuntimeError("readonly stored native packets are required for the full public-entry check")
            # An extra fixed prefix of 32 TRAIN pairs per collection fold; never read the old pool.
            prefix_rows=[r for f in range(4) for r in [x for x in man["episodes"] if x["fold"]==f][:a.pca_prefix_per_fold]]
            if not a.fixture and any(sum(r["fold"]==f for r in prefix_rows)!=a.pca_prefix_per_fold for f in range(4)):
                raise RuntimeError("incomplete balanced training PCA prefix")
            prefix=[]
            for row in prefix_rows:
                sp=Image.open(data/row["support"]).convert("RGB");qp=Image.open(data/row["query"]).convert("RGB")
                gold=torch.from_numpy((np.asarray(Image.open(ann/Path(row["support"]).with_suffix(".png")))==row["c"]+1).copy())
                _,got,_,_=run_foris(host,sp,gold,qp);states=got["deb"][0];chunks=[]
                for state in (states[0],states[-1]):
                    x=state.flatten(1).T.half().float()
                    ix=torch.linspace(0,len(x)-1,min(256,len(x)),device=dev).long();chunks.append(x[ix].cpu())
                prefix.append((row["fold"],torch.cat(chunks)))
                report["PCA_prefix_episodes"]=len(prefix);save()
            means,Vs=fit_bases(prefix,a.components,dev,bool(a.fixture))
            torch.save(dict(mean=means,V=Vs,model_axis=list(range(4)),components=a.components,
                            training_prefix_rows=prefix_rows,fitting_scope="collection_fold != model_holdfold",
                            no_prior_pool_used=True,fixture_synthetic_only=bool(a.fixture)),out/"pca.pt")
            report.update(pca_state="PCA_READY",fourfold_train_prefix_verified=bool(a.fixture) or all(sum(r["fold"]==f for r in prefix_rows)==a.pca_prefix_per_fold for f in range(4)),
                          PCA_prefix_extra_encoderpasses=len(prefix_rows),PCA_prior_pool_features_used=False,components=int(Vs.shape[-1]),model_axis=list(range(4)))
            save()
            means,Vs=means.to(dev),Vs.to(dev);del prefix
            (out/"confirm_manifest.json").write_text(json.dumps(json.loads(Path(a.confirm_manifest).read_text()) if a.confirm_manifest else dict(episodes=[])))
            near = None

            def one(row, kind="test"):
                nonlocal near
                c = row["c"]
                sp = Image.open(data / row["support"]).convert("RGB")
                qp = Image.open(data / row["query"]).convert("RGB")
                gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == c + 1).copy())
                if kind == "check":
                    host._finalize_mask = original_finalize
                    try:
                        native, got, ref_mask, _ = run_foris(host, sp, gold, qp)
                    finally:
                        host._finalize_mask = skip_finalize
                    packet = Path(a.packets) / ("%d_%d_%d.npz" % (row["fold"], row["e"], row["c"]))
                    with np.load(packet) as z:
                        stored = np.asarray(z["native"], dtype=np.uint8).reshape(-1)
                    actual = np.packbits(native.cpu().numpy()).reshape(-1)
                    report["native_checks"] += 1
                    if actual.shape != stored.shape:
                        raise RuntimeError("native mask shape differs from stored complete FoRIS: " + str(packet))
                    differing = int(np.unpackbits(actual ^ stored).sum())
                    if differing:
                        raise RuntimeError("complete public FoRIS mask is not bit exact on %s: %d pixels" % (packet, differing))
                    report["native_bit_identical"] += 1
                else:
                    native, got, ref_mask, _ = run_foris(host, sp, gold, qp)
                deb = got["deb"][0]
                q, r = deb[-1].flatten(1).T.half().float(), deb[0].flatten(1).T.half().float()  # as the feature cache stores them
                h, w = got["score"].shape
                if near is None or near.shape[0] != h * w:
                    near = near_mask(h * w, dev)
                cov = F.interpolate(ref_mask[None, None].float(), (h, w), mode="area")[0, 0]
                maps = relation_maps(q, r, cov, got["score"], got["s2"], got["s3"], near)
                tf=None  # Query confirmation label pixels are embargoed through cache construction.
                if kind != "confirm":
                    truth=torch.from_numpy((np.asarray(Image.open(ann/Path(row["query"]).with_suffix(".png")))==c+1).copy()).to(dev)
                    truth=F.interpolate(truth[None,None].float(),native.shape,mode="nearest")
                    tf=F.avg_pool2d(truth,native.shape[0]//h)[0,0]
                valid=[f != row["fold"] if kind=="train" else f==row["fold"] for f in range(4)]
                comp=torch.zeros(4,len(q),Vs.shape[-1],device=dev)
                for f,yes in enumerate(valid):
                    if yes:comp[f]=(q-means[f])@Vs[f]
                result=pack(maps,comp,tf,h,float(cov.mean()),valid)
                result["source_score"]=got["score"].float().cpu().numpy().astype(np.float16)
                result["model_shape"]=np.asarray(native.shape,dtype=np.int32)
                result["query_mask_path"]=np.asarray(str(ann/Path(row["query"]).with_suffix(".png")))
                result["query_original_shape"]=np.asarray([qp.height,qp.width],dtype=np.int32)
                return result
            for row in test["episodes"][:CHECK]:
                np.savez(out / "check" / ("%d_%d_%d.npz" % (row["fold"], row["e"], row["c"])), **one(row, "check"))
                report["checked"] += 1
            for kind, rows in (("train", man["episodes"][:a.limit]), ("confirm", confirm)):
                for row in rows:
                    try:
                        np.savez(out / kind / ("%d_%d_%d.npz" % (row["fold"], row["e"], row["c"])), **one(row, kind))
                        report["episodes" if kind == "train" else "confirmation"] += 1
                    except RuntimeError:
                        raise  # never silently reduce the contracted 2400/600 episodes
                    if (report["episodes"] + report["confirmation"] + report["skipped"]) % 50 == 0:
                        save()
                        print(json.dumps(dict(n=report["episodes"], confirmation=report["confirmation"], skipped=report["skipped"],
                                              s=round(time.monotonic() - start, 1))), flush=True)
        report["state"] = "COMPLETED"
        save()
        print(json.dumps(report), flush=True)
    except BaseException as err:
        report.update(state="ERROR", error=repr(err))
        save()
        raise


def replay(a):
    """The same inputs for the test episodes, from the cached features and the stored packets."""
    import numpy as np
    import torch
    from tics.relations import near_mask, relation_maps
    if a.fixture:
        root = Path(a.fixture)
        a.test_manifest, a.features, a.packets, a.out = (str(root / x) for x in ("episodes.json", "cache/feat", "run/packets", "head"))
    test = json.loads(Path(a.test_manifest).read_text())
    out = Path(a.out)
    (out / "test").mkdir()
    dev = "cuda" if torch.cuda.is_available() and not a.fixture else "cpu"
    basis = torch.load(out / "pca.pt", map_location=dev, weights_only=False)
    near, drift, start = None, 0.0, time.monotonic()
    with torch.no_grad():
        for row in test["episodes"]:
            name = "%d_%d_%d" % (row["fold"], row["e"], row["c"])
            feat = torch.load(Path(a.features) / (name + ".pt"), map_location="cpu", weights_only=False)
            z = np.load(Path(a.packets) / (name + ".npz"))
            q, r = feat["q"].to(dev).float(), feat["r"].to(dev).float()
            n = q.shape[0]
            side = int(round(n ** 0.5))
            if near is None or near.shape[0] != n:
                near = near_mask(n, dev)
            t = lambda k: torch.from_numpy(z[k].astype(np.float32)).to(dev)
            maps = relation_maps(q, r, t("cov"), t("score"), t("s2"), t("s3"), near)
            ps = int(round((len(z["truth"]) * 8) ** 0.5)) // side
            tf = torch.from_numpy(np.unpackbits(z["truth"])[:(side * ps) ** 2].reshape(side, ps, side, ps).mean((1, 3)).astype(np.float32))
            comp=torch.zeros(4,n,basis["V"].shape[-1],device=dev)
            comp[row["fold"]]=(q-basis["mean"][row["fold"]])@basis["V"][row["fold"]]
            packed=pack(maps,comp,tf,side,float(z["cov"].mean()),[f==row["fold"] for f in range(4)])
            np.savez(out / "test" / (name + ".npz"), **packed)
            ref = out / "check" / (name + ".npz")
            if ref.is_file():
                seen = np.load(ref)
                drift = max([drift] + [float(np.abs(seen[k].astype(np.float32) - packed[k].astype(np.float32)).max()) for k in ("maps", "pca", "tf")])
    rep = dict(state="COMPLETED" if drift <= 0.03 else "ERROR", episodes=len(test["episodes"]), live_against_replay_max_difference=drift,
               elapsed_s=time.monotonic() - start)
    (out / "replay_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    if rep["state"] != "COMPLETED":
        raise SystemExit("the live pass and the replay give different inputs")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--isolated-suite", help="metadata-only preparation of three explicit photo-isolated manifests")
    p.add_argument("--exposure-pool", help="prior pool: parse only pickle string opcodes")
    p.add_argument("--pca-prefix-per-fold", type=int, default=32)
    p.add_argument("--replay", action="store_true")
    p.add_argument("--manifest")
    p.add_argument("--test-manifest")
    p.add_argument("--per-fold", type=int, default=600)
    p.add_argument("--skip", type=int, default=0, help="with --prepare: start this many episodes later in the standard list")
    p.add_argument("--avoid", help="with --prepare: drop episodes that share an image with this manifest")
    p.add_argument("--confirm-manifest", help="untouched episodes for the one confirmation reading")
    p.add_argument("--pool")
    p.add_argument("--features")
    p.add_argument("--packets")
    p.add_argument("--out")
    p.add_argument("--components", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.add_argument("--fixture")
    p.add_argument("--unguarded", action="store_true")
    p.add_argument("--foris-root")
    p.add_argument("--demo4-root", default=os.environ.get("DEMO4_ROOT", "/root/autodl-tmp/demo4"))
    a = p.parse_args()
    prepare_isolated(a) if a.isolated_suite else prepare(a) if a.prepare else replay(a) if a.replay else live(a)


if __name__ == "__main__":
    main()
