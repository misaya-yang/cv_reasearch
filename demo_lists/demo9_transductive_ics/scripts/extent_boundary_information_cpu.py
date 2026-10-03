#!/usr/bin/env python3
"""Bounded edge-information diagnostic on frozen extent packets; never a mask method.

Truth labels and score-stratum balancing are evaluation instrumentation only.
Four-fold fitting excludes every support/query photo used by the held fold.
No parameter search, encoder, CUDA, or full feature cache is used.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
SOLVER_RESIDUALS = []


def auc(y, score):
    pos = y == 1
    n, m = int(pos.sum()), int((~pos).sum())
    if not n or not m:
        return None
    return float((rankdata(score)[pos].sum() - n * (n + 1) / 2) / (n * m))


def packet_rows(path, meta, rng):
    with np.load(path) as p:
        score = p["score"].astype(float)
        score = (score - score.min()) / max(float(np.ptp(score)), 1e-6)
        h, w = score.shape
        truth = np.unpackbits(p["truth"])[:1024 * 1024].reshape(1024, 1024)
        if 1024 % h or 1024 % w:
            raise ValueError("Unsupported saved truth/grid geometry")
        cov = truth.reshape(h, 1024 // h, w, 1024 // w).mean((1, 3))
        # Only nearly pure endpoints: mixed-patch geometry is not silently labelled.
        pure = (cov <= .05) | (cov >= .95)
        fg = cov >= .95
        idx = np.arange(h * w).reshape(h, w)
        ia = np.concatenate([idx[:, :-1].ravel(), idx[:-1].ravel()])
        ib = np.concatenate([idx[:, 1:].ravel(), idx[1:].ravel()])
        fa, fb = fg.ravel()[ia], fg.ravel()[ib]
        kind = np.where(fa != fb, 0, np.where(fa, 1, 2))
        good = pure.ravel()[ia] & pure.ravel()[ib]
        sa, sb = score.ravel()[ia], score.ravel()[ib]
        mean, jump = (sa + sb) / 2, np.abs(sa - sb)
        strata = np.minimum((mean * 10).astype(int), 9) * 20 + np.minimum((jump * 20).astype(int), 19)
        blocks = []
        for b in np.unique(strata[good]):
            ids = [np.flatnonzero(good & (strata == b) & (kind == k)) for k in range(3)]
            k = min(len(ids[0]) // 2, len(ids[1]), len(ids[2]), 8)
            if k:
                blocks.extend([np.array([a, aa, f, bg]) for a, aa, f, bg in zip(
                    rng.permutation(ids[0])[:2*k:2], rng.permutation(ids[0])[:2*k:2],
                    rng.permutation(ids[1])[:k], rng.permutation(ids[2])[:k])])
                # Replace the duplicate-prone positive drawing with one disjoint draw.
                positive = rng.permutation(ids[0])[:2*k].reshape(k, 2)
                for j in range(k):
                    blocks[-k+j][:2] = positive[j]
        if not blocks:
            return None, dict(total_edges=len(ia), pure_edges=int(good.sum()),
                              pure_cross=int((good & (kind == 0)).sum()), matched_edges=0)
        rng.shuffle(blocks)
        pick = np.concatenate(blocks[:32])  # <=128 edges/episode, exactly balanced.
        sa, sb, mean, jump = sa[pick], sb[pick], mean[pick], jump[pick]
        common = np.column_stack([sa, sb, mean, jump, np.minimum(sa,sb), np.maximum(sa,sb),
            sa**2, sb**2, mean**2, jump**2, mean*jump, (sa-.5)*(sb-.5)])
        poly = np.column_stack([sa**3, sb**3, mean**3, jump**3, sa*sb, mean**2*jump,
            mean*jump**2, sa**4, sb**4, jump**4, np.abs(sa-.5), np.abs(sb-.5)])
        aff = np.concatenate([p["aff_r"].ravel(),p["aff_d"].ravel()])[pick]
        sf = np.concatenate([p["sig_right"].ravel(),p["sig_down"].ravel()])[pick]
        sr = np.concatenate([p["sig_left"].ravel(),p["sig_up"].ravel()])[pick]
        fga, fgb = p["fg_max"][ia[pick]], p["fg_max"][ib[pick]]
        bga, bgb = p["bg_max"][ia[pick]], p["bg_max"][ib[pick]]
        ma, mb = fga-bga, fgb-bgb
        local = np.column_stack([aff, fga, fgb, bga, bgb, ma, mb, ma-mb, ma*mb, sf, sr, np.maximum(sf,sr)])
        fields = {"score_common": common, "score_capacity24": np.column_stack([common,poly]),
                  "score_plus_local24": np.column_stack([common,local])}
        for name, lo, hi in (("score_plus_affinity",0,1),("score_plus_fg_bg",1,9),("score_plus_signature",9,12)):
            ablated = np.zeros_like(local)
            ablated[:,lo:hi] = local[:,lo:hi]
            fields[name] = np.column_stack([common,ablated])
        if not all(np.isfinite(v).all() for v in fields.values()):
            raise ValueError("Nonfinite diagnostic fields, not an information negative")
        rows = dict(meta=meta, x=fields, y=(kind[pick] == 0).astype(float), kind=kind[pick],strata=strata[pick])
        return rows, dict(total_edges=len(ia), pure_edges=int(good.sum()),
            pure_cross=int((good & (kind == 0)).sum()), matched_edges=len(pick),
            bins_used=len(np.unique(strata[pick])))


def fit(x, y):
    mean, std = x.mean(0), x.std(0)
    std = np.maximum(std, 1e-6)
    z = np.column_stack([(x-mean)/std,np.ones(len(x))])
    reg = np.eye(z.shape[1])*.01
    reg[-1,-1] = 0
    # Explicit contractions avoid platform BLAS floating-point-status warnings.
    lhs = np.einsum("ni,nj->ij",z,z,optimize=False)/len(z)+reg
    rhs = np.einsum("ni,n->i",z,y,optimize=False)/len(z)
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ValueError("Nonfinite ridge input; not an information negative")
    w = np.linalg.solve(lhs,rhs)
    residual=float(np.linalg.norm(np.einsum("ij,j->i",lhs,w,optimize=False)-rhs))
    if not np.isfinite(w).all() or residual>1e-8:
        raise ValueError("Invalid ridge solve; not an information negative")
    SOLVER_RESIDUALS.append(residual)
    return mean,std,w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--negative-control-out",type=Path)
    args = p.parse_args()
    start = time.monotonic()
    manifest = json.loads((args.root/"episodes.json").read_text())
    rng = np.random.default_rng(31051)
    shuffle_rng = np.random.default_rng(31052)
    shuffled_edges, changed_edges = 0, 0
    data, audit = [], []
    for meta in manifest["episodes"]:
        path = args.root/"run/packets"/f'{meta["fold"]}_{meta["e"]}_{meta["c"]}.npz'
        if not path.is_file():
            audit.append(dict(meta=meta,state="MISSING_PACKET")); continue
        with np.load(path) as pk:
            missing = sorted(set(("sig_right","sig_left","sig_down","sig_up"))-set(pk.files))
        if missing:
            audit.append(dict(meta=meta,state="MISSING_SIGNATURE",missing=missing)); continue
        rows, counts = packet_rows(path,meta,rng)
        audit.append(dict(meta=meta,state="MATCHED" if rows else "NO_MATCHED_PURE_EDGE_STRATUM",**counts))
        if rows:
            if args.negative_control_out:
                x = rows["x"]["score_plus_local24"]
                shuffled = x.copy()
                for s in np.unique(rows["strata"]):
                    idx = np.flatnonzero(rows["strata"] == s)
                    donor = shuffle_rng.permutation(idx)
                    shuffled[idx,12:] = x[donor,12:]
                    shuffled_edges += len(idx)
                    changed_edges += int((idx != donor).sum())
                assert np.array_equal(shuffled[:,:12],x[:,:12])
                rows["x"]["score_plus_shuffled_local24"] = shuffled
            data.append(rows)
    arms = ("score_common","score_capacity24","score_plus_local24",
            "score_plus_affinity","score_plus_fg_bg","score_plus_signature")
    if args.negative_control_out:
        arms += ("score_plus_shuffled_local24",)
    fold_audit, episodes = [], []
    for fold in range(4):
        held_photos = {m[k] for m in manifest["episodes"] if m["fold"] == fold for k in ("support","query")}
        tr = [d for d in data if d["meta"]["fold"] != fold and not
              ({d["meta"]["support"],d["meta"]["query"]}&held_photos)]
        te = [d for d in data if d["meta"]["fold"] == fold]
        fa = dict(fold=fold,training_episodes=len(tr),held_episodes=len(te),
            photo_overlap_after_filter=0,training_edges=sum(len(d["y"]) for d in tr))
        fold_audit.append(fa)
        if not tr or not te:
            fa["state"]="INSUFFICIENT_DISJOINT_DATA"; continue
        models={a:fit(np.concatenate([d["x"][a] for d in tr]),np.concatenate([d["y"] for d in tr])) for a in arms}
        for d in te:
            row=dict(meta=d["meta"],edges=len(d["y"]),auc={})
            for a in arms:
                mu,sd,w=models[a]
                score=np.einsum("ij,j->i",np.column_stack([(d["x"][a]-mu)/sd,np.ones(len(d["y"]))]),w,optimize=False)
                if not np.isfinite(score).all():
                    raise ValueError("Nonfinite held score; not an information negative")
                row["auc"][a]={"both":auc(d["y"],score)}
                for kind,name in ((1,"vs_foreground_internal"),(2,"vs_background_internal")):
                    mask=(d["kind"]==0)|(d["kind"]==kind)
                    row["auc"][a][name]=auc(d["y"][mask],score[mask])
            episodes.append(row)
    parent=list(range(len(episodes)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    photos={}
    for i,r in enumerate(episodes):
        for k in ("support","query"):
            name=r["meta"][k]
            if name in photos: parent[find(i)]=find(photos[name])
            else: photos[name]=i
    groups={}
    for i in range(len(episodes)):groups.setdefault(find(i),[]).append(i)
    gs=list(groups.values())
    summary={}
    for target in ("both","vs_foreground_internal","vs_background_internal"):
        vals=np.array([[r["auc"][a][target] for a in arms] for r in episodes])
        delta=vals[:,2]-vals[:,1]
        boot=[]
        for _ in range(1000):
            ids=np.concatenate([gs[j] for j in rng.integers(0,len(gs),len(gs))])
            boot.append(float(delta[ids].mean()))
        summary[target]=dict(macro_episode_auc={a:float(vals[:,j].mean()) for j,a in enumerate(arms)},
            added_local_minus_capacity_control=float(delta.mean()),photo_group_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
            per_fold={str(f):float(delta[[i for i,r in enumerate(episodes) if r["meta"]["fold"]==f]].mean()) for f in range(4)})
        if args.negative_control_out:
            ndelta = vals[:,2]-vals[:,-1]
            nb=[]
            for _ in range(1000):
                ids=np.concatenate([gs[j] for j in rng.integers(0,len(gs),len(gs))])
                nb.append(float(ndelta[ids].mean()))
            summary[target]["real_minus_conditional_shuffle"] = dict(delta=float(ndelta.mean()),
                photo_group_bootstrap95=np.quantile(nb,[.025,.975]).tolist(),
                per_fold={str(f):float(ndelta[[i for i,r in enumerate(episodes) if r["meta"]["fold"]==f]].mean()) for f in range(4)})
    from collections import Counter
    out=dict(state="COMPLETED_CPU_INFORMATION_DIAGNOSTIC",seconds=time.monotonic()-start,
        manifest_episodes=len(manifest["episodes"]),usable_episodes=len(episodes),held_photo_groups=len(gs),
        sampling_states=dict(Counter(a["state"] for a in audit)),held_edges=sum(r["edges"] for r in episodes),
        arms_parameters={"score_common":13,"score_capacity24":25,"score_plus_local24":25},ridge_lambda=.01,seed=31051,
        ablation_scope="Additional affinity-only/FG-BG-only/signature-only arms zero omitted local columns; fixed identical fit, narrower effective feature counts, no model selection.",
        numerical_check=dict(all_inputs_weights_predictions_finite=True,max_normal_equation_residual=max(SOLVER_RESIDUALS),
            contraction="Explicit einsum; initial platform matmul emitted floating-point-status warnings, so final evidence uses finite-checked scalar contractions."),
        prediction_scope="Cross-class/photo-disjoint local boundary evidence AUC only; NOT mIoU, deployment method, posterior calibration or independent confirmatory test.",
        matching="Within each episode, score-mean .1 and absolute-score-jump .05 bins; equal FG-BG positives and negatives, negative FG-FG/BG-BG equal; max128 edges per episode; pure endpoints coverage<=.05 or>=.95.",
        limitation="Pure endpoints exclude mixed-patch boundaries and thin targets. GT-stratified matched sampling is diagnostic only. Ridge failure does not establish absence of information. Source fields are already seen development evidence.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),folds=fold_audit,summary=summary,
        episodes=episodes,sampling_audit=audit)
    destination=args.negative_control_out or args.out
    if args.negative_control_out:
        out["fixed_negative_control"]=dict(seed=31052,shuffled_edges=shuffled_edges,changed_edges=changed_edges,
            operation="Jointly permute all12 local fields within SAME episode and SAME score mean/jump stratum; no label access; score columns unchanged; train and held permutations use the fixed seed.",
            limitation="Bins are approximate conditioning, not exact score matching. One shuffle is a falsification control, not a permutation significance test. Sparse selected32 tasks do not establish deployment coverage.",
            preserved_original=str(args.out),preserved_original_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())
    destination.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps({k:out[k] for k in ("state","seconds","usable_episodes","held_photo_groups","summary")},indent=2))


if __name__=="__main__":
    main()
