#!/usr/bin/env python3
"""Can the extent be learned without explicit class-ID/text inputs? Heads fitted on base-class episodes of three folds, scored on the
fourth fold's novel classes.

  python scripts/extent_head.py --cache cache/extent_head_v0 --run results/extent_v1/run --out results/extent_head_v0/report.json

Three readings: a learning curve on the development episodes (100, 200, 400 training episodes per fold, then all),
every head on the development episodes (the head is chosen here), and one reading of the chosen head on the
confirmation episodes that nobody has looked at.
  python scripts/extent_head.py --selfcheck

Heads (inputs are the relation maps of tics/relations.py):
  dense       a small convolutional net gives the mask directly
  dense_area  the same net only says how large the target is; the mask is the FoRIS level set of that area
  size        a regressor of the log area; the mask is the FoRIS level set of that area
  quality     a net that sees a candidate cut of the FoRIS score and predicts its IoU; the best of 8 cuts is taken
Input sets: the score alone (control: what a fitted read-out gets from the same information), all relation maps,
and relation maps plus 16 principal components of the query features (not class-free, to see what raw features add).
GT-area intervention diagnostic: not an information or task-optimal upper bound. Patch level, no refinement; a premise test, not a method score.
"""
import argparse
import os
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from analyze_extent import load, miou  # noqa: E402
from tics.relations import NAMES as RELATION_NAMES

# Written 2026-10-03 before the run. Reference points (241 episodes, patch level unless stated): native 56.99; the
# level set of the true area +8.7 (model size); FoRIS's own log-area error 0.67; a read-out fitted on about 180
# episodes per fold loses 3.4 with relation inputs and gains 0.15 with the score alone.
CARD = dict(
    assumption="a fixed extent readout can transfer to held classes using reference-conditioned evidence without explicit class IDs, and can be "
               "learned from base-class episodes",
    prediction="with about 1800 training episodes per fold: relation inputs give the dense head +2 to +5 over native "
               "with the interval above 0 and at least +2 over the score-only control; log-area error of the size "
               "estimate 0.45 or less (FoRIS's own 0.67); the score-only control within 1 of native",
    match="extent is learnable across classes (the chosen head also passes on the confirmation episodes): fit on "
          "train2014 under the standard protocol, add refinement, read 1000 episodes per fold",
    mismatch="relation inputs not above the score-only control: this fixed readout did not recover the size, reconsider its inputs "
             "(other layers, raw features), not the task; principal components far above relations: the knowledge "
             "sits in raw features, build the head on them; training fit high and held-out low with a curve that "
             "still rises at 600 per fold: more base data; a flat curve under +1: stop this head")
RELATION_CHANNELS = len(RELATION_NAMES)
MODEL_CHANNELS = 32
assert MODEL_CHANNELS == RELATION_CHANNELS + 16, "source maps16 + PCA16 contract"
RELATION_SCHEMA_VERSION = 2
RELATION_NAMES_SHA256 = hashlib.sha256(json.dumps(list(RELATION_NAMES)).encode()).hexdigest()
GATE = dict(gain=4.0, folds=4)
LEVELS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # candidate cuts of the normalised FoRIS score for the quality head


def load_dir(folder, rows=None, labels=True):
    """Load features; confirmation labels are forbidden until selection is frozen."""
    files = sorted(folder.glob("*.npz")) if rows is None else [folder / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])) for r in rows]
    M, P, T = [], [], []
    for file in files:
        with np.load(file) as z:
            if not labels and "tf" in z.files:
                raise RuntimeError("confirmation cache must not contain tf")
            pc = z["pca"].astype(np.float32)
            if pc.ndim != 4 or pc.shape[0] != 4:
                raise RuntimeError("PCA must be [held_model, K, H, W], never global-PCA")
            if "pca_model_valid" in z.files:
                pc[~z["pca_model_valid"].astype(bool)] = np.nan
            maps=z["maps"].astype(np.float32)
            if maps.ndim!=3 or maps.shape[0]!=RELATION_CHANNELS:
                raise RuntimeError("maps channels must exactly match source NAMES ordering; no truncation")
            M.append(maps); P.append(pc)
            if labels: T.append(z["tf"].astype(np.float32))
    if not files: raise RuntimeError("empty cache folder")
    fold, cls = (np.array([int(f.stem.split("_")[k]) for f in files]) for k in (0, 2))
    if not np.all(cls % 4 == fold): raise RuntimeError("collection fold/class mismatch")
    return np.stack(M), np.stack(P), np.stack(T) if labels else None, cls, fold


def model_inputs(data, ids, f, arm):
    maps, pca = data[0][ids], data[1][ids]
    if maps.shape[1]!=RELATION_CHANNELS:raise RuntimeError("wrong source relation map schema")
    X = np.zeros((len(ids), MODEL_CHANNELS, *maps.shape[-2:]), np.float32)
    if arm == "score_only": X[:, :1] = maps[:, :1]
    else:
        if maps.shape[1]!=RELATION_CHANNELS:raise RuntimeError("wrong source relation map schema")
        X[:, :RELATION_CHANNELS] = maps
        if arm == "relations_pca":
            q = pca[:, f, :16]
            if not np.isfinite(q).all(): raise RuntimeError("invalid held-model PCA input")
            X[:, RELATION_CHANNELS:RELATION_CHANNELS+q.shape[1]] = q
    if not np.isfinite(X).all(): raise RuntimeError("nonfinite maps")
    return X


def make_net(d, head):
    import torch.nn as nn
    layers, c = [], d + (head == "quality")
    if head in ("size", "quality"):
        for _ in range(4):
            layers += [nn.Conv2d(c, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.ReLU(), nn.AvgPool2d(2)]
            c = 48
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 64), nn.ReLU(), nn.Linear(64, 1)]
    else:
        for dil in (1, 2, 4, 8, 1, 1):
            layers += [nn.Conv2d(c, 64, 3, padding=dil, dilation=dil), nn.GroupNorm(8, 64), nn.ReLU()]
            c = 64
        layers += [nn.Dropout2d(0.1), nn.Conv2d(64, 1, 1)]
    return nn.Sequential(*layers)


def save_model(folder, seed, net, mu, sd, head, a):
    if folder is None: return
    import torch
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("seed%d.pt" % seed)
    torch.save(dict(state="FROZEN_FULL_FIT", head=head, channels=MODEL_CHANNELS,
        weights={k:v.detach().cpu() for k,v in net.state_dict().items()},
        mu=mu.detach().cpu(), sd=sd.detach().cpu(), seed=seed, epochs=a.epochs,
        optimizer="AdamW lr=.002 wd=.01; OneCycleLR; no DEV/confirm gradients",
        relation_schema_version=RELATION_SCHEMA_VERSION,relation_names_sha256=RELATION_NAMES_SHA256), path)

def subset(train, per_fold, seed=0):
    """The first `per_fold` training episodes of every fold in a fixed random order (for the learning curve)."""
    fold = train[4]
    rng = np.random.default_rng(seed)
    keep = np.concatenate([rng.permutation(np.nonzero(fold == f)[0])[:per_fold] for f in np.unique(fold)])
    return tuple(x[np.sort(keep)] for x in train)


def level_of_area(sn, k):
    """The FoRIS level set with k patches: the k highest scores. sn: [n, P]; k: [n] ints."""
    order = np.argsort(-sn, 1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(sn.shape[1])[None].repeat(len(sn), 0), 1)
    return rank < np.clip(k, 1, sn.shape[1])[:, None]


def fit(a, Xtr, Ttr, Xte, d, size_head, dev, save_dir=None):
    """Returns held-out outputs averaged over seeds, and the fit on a sample of the training episodes."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    mu, sd = Xtr.mean((0, 2, 3), keepdim=True), Xtr.std((0, 2, 3), keepdim=True) + 1e-6
    target = torch.log(Ttr.mean((1, 2)).clamp_min(1e-5))
    sample = torch.arange(0, len(Xtr), max(1, len(Xtr) // 200), device=dev)
    held, seen = 0, 0
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        net = make_net(d, "size" if size_head else "dense").to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        steps = a.epochs * ((len(Xtr) + a.batch - 1) // a.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
        net.train()
        for _ in range(a.epochs):
            perm = torch.randperm(len(Xtr), device=dev)
            for s in range(0, len(perm), a.batch):
                idx = perm[s:s + a.batch]
                x, y = (Xtr[idx] - mu) / sd, Ttr[idx]
                if torch.rand(()) < 0.5:
                    x, y = x.flip(3), y.flip(2)
                if size_head:
                    loss = F.smooth_l1_loss(net(x)[:, 0], target[idx])
                else:
                    logit = net(x)[:, 0]
                    pr = logit.sigmoid()
                    soft = 1 - (pr * y).sum((1, 2)) / (pr + y - pr * y).sum((1, 2)).clamp_min(1e-6)
                    loss = F.binary_cross_entropy_with_logits(logit, (y > 0.5).float()) + soft.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        save_model(save_dir, seed, net, mu, sd, "size" if size_head else "dense", a)
        with torch.no_grad():
            run = lambda x: torch.cat([net((x[i:i + 128] - mu) / sd)[:, 0] for i in range(0, len(x), 128)])
            post = (lambda v: v) if size_head else torch.sigmoid
            held = held + post(run(Xte)).cpu().numpy() / a.seeds
            seen = seen + post(run(Xtr[sample])).cpu().numpy() / a.seeds
    return held, seen, sample.cpu().numpy()


def fit_quality(a, Xtr, Ttr, Xte, d, dev, save_dir=None):
    """Predicted IoU of each candidate cut for the held-out episodes: [n, len(LEVELS)]. Channel 0 is the score."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    mu, sd = Xtr.mean((0, 2, 3), keepdim=True), Xtr.std((0, 2, 3), keepdim=True) + 1e-6
    levels = torch.tensor(LEVELS, device=dev)
    held = 0
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        net = make_net(d, "quality").to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * ((len(Xtr) + a.batch - 1) // a.batch))
        net.train()
        for _ in range(a.epochs):
            perm = torch.randperm(len(Xtr), device=dev)
            for s in range(0, len(perm), a.batch):
                idx = perm[s:s + a.batch]
                x, y = Xtr[idx], Ttr[idx]
                cand = (x[:, 0] > levels[torch.randint(len(LEVELS), (len(idx),), device=dev)][:, None, None]).float()
                inter = (y * cand).sum((1, 2))
                iou = inter / (y.sum((1, 2)) + cand.sum((1, 2)) - inter).clamp_min(1e-6)
                loss = F.binary_cross_entropy_with_logits(net(torch.cat([(x - mu) / sd, cand[:, None]], 1))[:, 0], iou)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        save_model(save_dir, seed, net, mu, sd, "quality", a)
        with torch.no_grad():
            cols = [torch.cat([net(torch.cat([(Xte[i:i + 128] - mu) / sd, (Xte[i:i + 128, :1] > t).float()], 1))[:, 0]
                               for i in range(0, len(Xte), 128)]).sigmoid() for t in LEVELS]
            held = held + torch.stack(cols, 1).cpu().numpy() / a.seeds
    return held


def predict(a, train, test, dev, arms=("score_only", "relations", "relations_pca"), extra_heads=True, model_root=None):
    """Independent full fits on three collection folds; only DEV is read here."""
    import torch
    Ftr, Fte = train[4], test[4]
    n, side = len(test[0]), test[0].shape[-1]
    out = {}
    for arm in arms:
        prob, seen_gain = np.zeros((n, side, side), np.float32), []
        size, qual = np.zeros(n, np.float32), np.zeros((n, len(LEVELS)), np.float32)
        paths = {}
        for f in np.unique(Fte):
            tr, te = np.nonzero(Ftr != f)[0], np.nonzero(Fte == f)[0]
            if not len(tr): raise RuntimeError("no base training episodes")
            if np.any(train[3][tr] % 4 == f): raise RuntimeError("held classes entered fit")
            Xtr = torch.from_numpy(model_inputs(train, tr, int(f), arm)).to(dev)
            Xte = torch.from_numpy(model_inputs(test, te, int(f), arm)).to(dev)
            Ttr = torch.from_numpy(train[2][tr]).to(dev)
            folder = None if model_root is None else model_root / arm / ("fold%d" % f)
            dense_dir = None if folder is None else folder / "dense"
            held, seen, sample = fit(a, Xtr, Ttr, Xte, MODEL_CHANNELS, False, dev, dense_dir)
            prob[te] = held
            ttr = train[2][tr][sample].reshape(len(sample), -1)
            strn = train[0][tr][sample][:, 0].reshape(len(sample), -1)
            fit_iou = lambda q: float(np.mean((ttr*q).sum(1) / np.maximum(ttr.sum(1)+q.sum(1)-(ttr*q).sum(1),1e-6)))
            seen_gain.append(100*(fit_iou(seen.reshape(len(sample),-1)>.5)-fit_iou(strn>.5)))
            kinds = ["dense"]
            if arm != "relations_pca" and extra_heads:
                size_dir = None if folder is None else folder / "size"
                quality_dir = None if folder is None else folder / "quality"
                size[te] = fit(a, Xtr, Ttr, Xte, MODEL_CHANNELS, True, dev, size_dir)[0]
                qual[te] = fit_quality(a, Xtr, Ttr, Xte, MODEL_CHANNELS, dev, quality_dir)
                kinds += ["size", "quality"]
            if folder is not None:
                paths[str(int(f))] = {kind:[str(folder/kind/("seed%d.pt"%seed)) for seed in range(a.seeds)] for kind in kinds}
            print("fit "+json.dumps(dict(arm=arm, model_fold=int(f), train_episodes=len(tr), epochs=a.epochs, seeds=a.seeds, heads=kinds)),flush=True)
        out[arm] = dict(prob=prob, size=size if arm != "relations_pca" and extra_heads else None,
            qual=qual if arm != "relations_pca" and extra_heads else None,
            training_fit=float(np.mean(seen_gain)), inputs=MODEL_CHANNELS, model_paths=paths)
    return out


def prediction_mask(pr, sn, head):
    removal = head.endswith("_removal")
    base = head[:-8] if removal else head
    n, side = len(sn), sn.shape[-1]
    flat = sn.reshape(n,-1); P=flat.shape[1]
    if base == "dense": mask=pr["prob"]>.5
    elif base == "dense_area": mask=level_of_area(flat,np.round(pr["prob"].reshape(n,-1).sum(1)).astype(int)).reshape(sn.shape)
    elif base == "size":
        log_area = np.clip(pr["size"],np.log(1/P),0)
        mask=level_of_area(flat,np.round(np.exp(log_area)*P).astype(int)).reshape(sn.shape)
    elif base == "quality": mask=sn > np.array(LEVELS)[pr["qual"].argmax(1)][:,None,None]
    else: raise ValueError("unknown output head")
    return mask & (sn>.5) if removal else mask

def summarise(test, preds, sel=None, log=print):
    """The table for the selected test episodes, from outputs computed once."""
    sel = np.ones(len(test[0]), bool) if sel is None else sel
    Mte, _, Tte, cls, Fte = (x[sel] for x in test)
    n, side = len(Mte), Mte.shape[-1]
    P = side * side
    tf, sn = Tte.reshape(n, -1), Mte[:, 0].reshape(n, -1)

    def iu(pred):
        pred = pred.reshape(n, -1)
        i = (tf * pred).sum(1)
        return np.stack([i, tf.sum(1) + pred.sum(1) - i], 1)
    base = iu(sn > 0.5)
    w = np.random.default_rng(0).multinomial(n, np.ones(n) / n, size=2000).astype(float)
    score = lambda t, m=slice(None): float(miou(t[m, 0], t[m, 1], cls[m])[0])

    def row(pred, ref=base):
        t = iu(pred)
        d = miou(t[:, 0], t[:, 1], cls, w) - miou(ref[:, 0], ref[:, 1], cls, w)
        by = {int(f): score(t, Fte == f) - score(ref, Fte == f) for f in np.unique(Fte)}
        return dict(patch_miou=score(t), gain=score(t) - score(ref), ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])],
                    by_fold=by, folds_up=int(sum(v > 0 for v in by.values())))
    true_log = np.log(np.maximum(tf.mean(1), 1e-5))
    err = lambda est: dict(log_error_std=float(np.std(est - true_log)), log_error_bias=float(np.mean(est - true_log)),
                           log_error_median_abs=float(np.median(np.abs(est - true_log))))
    cuts = np.stack([sn > t for t in LEVELS], 1)  # [n, levels, P]
    cut_iou = (tf[:, None] * cuts).sum(2) / np.maximum(tf.sum(1)[:, None] + cuts.sum(2) - (tf[:, None] * cuts).sum(2), 1e-6)
    out = dict(episodes=n, native_patch_miou=score(base), true_area_level=row(level_of_area(sn, np.round(tf.sum(1)).astype(int))),
               best_of_candidate_cuts=row(cuts[np.arange(n), cut_iou.argmax(1)]),
               foris_own_area=err(np.log(np.maximum((sn > 0.5).mean(1), 1e-5))), arms={})
    for arm, pr in preds.items():
        prob = pr["prob"][sel]
        area = prob.reshape(n, -1).sum(1)
        res = dict(dense=row(prob > 0.5), dense_area=dict(row(level_of_area(sn, np.round(area).astype(int))), **err(np.log(np.maximum(area / P, 1e-5)))),
                   training_fit_gain_mean_iou=pr["training_fit"], inputs=pr["inputs"])
        if pr["size"] is not None:
            res["size"] = dict(row(level_of_area(sn, np.round(np.exp(pr["size"][sel]) * P).astype(int))), **err(pr["size"][sel]))
        if pr["qual"] is not None:
            qual = pr["qual"][sel]
            res["quality"] = dict(row(cuts[np.arange(n), qual.argmax(1)]),
                                  rank_correlation_with_true_iou=float(np.mean([np.corrcoef(qual[j], cut_iou[j])[0, 1] for j in range(n) if cut_iou[j].std() > 0])))
        for head in ("dense", "dense_area", "size", "quality"):
            if head in res:
                projected={k:(None if v is None else v[sel]) for k,v in pr.items() if k in ("prob","size","qual")}
                res[head+"_removal"]=row(prediction_mask(projected,Mte[:,0],head+"_removal"))
        if arm != "score_only" and "score_only" in preds:
            res["dense_over_score_only"] = row(prob > 0.5, iu(preds["score_only"]["prob"][sel] > 0.5))
        out["arms"][arm] = res
        log(arm + " " + json.dumps({k: ({x: (round(y, 2) if isinstance(y, float) else y) for x, y in v.items() if x != "by_fold"} if isinstance(v, dict) else v)
                                    for k, v in res.items()}))
    if "relations" in out["arms"]:
        rel = out["arms"]["relations"]
        name = max((k for k in ("dense", "dense_area", "size", "quality") if k in rel), key=lambda k: rel[k]["gain"])
        best = rel[name]
        out["verdict"] = dict(best_relation_head=name, best_relation_head_gain=best["gain"], ci95=best["ci95"], folds_up=best["folds_up"],
                              passes=bool(best["gain"] >= GATE["gain"] and best["ci95"][0] > 0 and best["folds_up"] >= GATE["folds"]))
    return out


def evaluate(a, train, test, dev, log=print, arms=("score_only", "relations", "relations_pca"), extra_heads=True):
    return summarise(test, predict(a, train, test, dev, arms, extra_heads), log=log)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def runtime_device(a):
    import torch
    if a.fixture:
        if a.device != "cpu":raise RuntimeError("tiny fixtures are CPU-only")
        return "cpu"
    if a.device != "cuda" or os.environ.get("DEMO9_CUDA_GUARD") != "1":
        raise RuntimeError("real T1 fitting/scoring requires root-owned CUDA guard; CPU only with --fixture")
    if not torch.cuda.is_available():raise RuntimeError("no CUDA device; prepare only")
    torch.cuda.set_per_process_memory_fraction(.4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return "cuda"


def development_phase(a):
    import torch
    dev=runtime_device(a)
    recs=load(a.run); cache=Path(a.cache); out=Path(a.out).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    train,test=load_dir(cache/"train"),load_dir(cache/"test",recs)
    if not a.train_manifest: raise RuntimeError("--train-manifest required for training UID freeze")
    tm=json.loads(Path(a.train_manifest).read_text())
    by_name={"%d_%d_%d"%(r["fold"],r["e"],r["c"]):r for r in tm["episodes"]}
    actual_rows=[by_name[f.stem] for f in sorted((cache/"train").glob("*.npz"))]
    recipe=dict(epochs=a.epochs,seeds=a.seeds,batch=a.batch,channels=MODEL_CHANNELS,
        relation_names=list(RELATION_NAMES),relation_names_sha256=RELATION_NAMES_SHA256,
        relation_schema_version=RELATION_SCHEMA_VERSION,
        lr=.002,wd=.01,scheduler="OneCycleLR",curve=a.curve,epoch_selection="fixed_recipe_no_DEV_epoch_selection",
        extra_heads="score/relations size+quality; PCA dense/area only",precision="FP32 tensor operations, TF32 disabled, root CUDA guard memory fraction .4")
    start=time.monotonic()
    res=dict(state="RUNNING",card=CARD,gate=GATE,train_episodes=len(train[0]),test_episodes=len(test[0]),
        train_per_fold={int(f):int((train[4]==f).sum()) for f in np.unique(train[4])},
        training_epochs=a.epochs,seeds=a.seeds,score_control_capacity_matched=True,channel_count=MODEL_CHANNELS,
        relation_names=list(RELATION_NAMES),relation_names_sha256=RELATION_NAMES_SHA256,relation_schema_version=RELATION_SCHEMA_VERSION,
        scope="T1 patch-level premise; no CRF/original-resolution method claim; DEV used for selection",
        recipe=recipe,learning_curve={})
    for size in [int(x) for x in a.curve.split(",") if x]:
        small=subset(train,size)
        part=evaluate(a,small,test,dev,log=lambda s:None,arms=("score_only","relations"),extra_heads=False)
        res["learning_curve"][str(size)]={k:dict(gain=v["dense"]["gain"],ci95=v["dense"]["ci95"],
            training_fit=v["training_fit_gain_mean_iou"]) for k,v in part["arms"].items()}
        res["learning_curve"][str(size)]["fit_counts"]={str(f):int((small[4]!=f).sum()) for f in range(4)}
        res["learning_curve"][str(size)]["independently_initialized"]=True
        print("curve "+str(size)+" "+json.dumps(res["learning_curve"][str(size)]),flush=True)
        out.write_text(json.dumps(res,indent=1))
    preds=predict(a,train,test,dev,model_root=out.parent/"models")
    res["development"]=summarise(test,preds)
    heads=("dense","dense_area","size","quality")
    choices=[(arm,head,metrics) for arm,table in res["development"]["arms"].items() if arm!="score_only"
        for head in heads+tuple(h+"_removal" for h in heads) if (metrics:=table.get(head)) is not None]
    arm,head,metric=max(choices,key=lambda v:v[2]["gain"])
    base_head=head[:-8] if head.endswith("_removal") else head
    model_head="dense" if base_head in ("dense","dense_area") else base_head
    checkpoints={}
    for input_arm in (arm,"score_only"):
        checkpoints[input_arm]={}
        for f,paths in preds[input_arm]["model_paths"].items():
            checkpoints[input_arm][f]=[dict(path=str(Path(x).resolve()),sha256=file_sha(x)) for x in paths[model_head]]
    pca_path=cache/"pca.pt"
    selection=dict(state="SELECTION_FROZEN",input_arm=arm,head=head,model_head=model_head,
        variant="removal" if head.endswith("_removal") else "normal",chosen_on="DEV_only",checkpoint=checkpoints,
        recipe=recipe,recipe_sha256=json_sha(recipe),script_sha256=file_sha(__file__),
        train_manifest=str(Path(a.train_manifest).resolve()),train_manifest_sha256=file_sha(a.train_manifest),
        training_model_provenance={str(f):dict(rows=sum(r["fold"]!=f for r in actual_rows),
            UID_sha256=json_sha(sorted({Path(r[k]).stem for r in actual_rows if r["fold"]!=f for k in ("support","query")}))) for f in range(4)},
        pca_identity=dict(path=str(pca_path.resolve()),sha256=file_sha(pca_path) if pca_path.is_file() else None,
            contract="per-held-model basis from legitimate training prefix only; no confirmation inputs"),
        development_report=str(out),development_metric=metric,
        confirmation_queryGT_opened=False,source_scope="T1 patch-level, no CRF",
        removal_scope="intersection with same T1 native patch mask sn>.5; not complete public CRF mask")
    selection_path=out.with_name("selection.json")
    selection_path.write_text(json.dumps(selection,indent=2))
    res.update(state="DEVELOPMENT_SELECTED_CONFIRM_SEALED",selection_path=str(selection_path),
        selection_sha256=file_sha(selection_path),selected_configuration=dict(input_arm=arm,head=head,variant=selection["variant"]),
        confirmation_queryGT_opened=False,elapsed_s=time.monotonic()-start,
        true_area_scope="GT area intervention, not an information or task-optimal upper bound")
    out.write_text(json.dumps(res,indent=1))
    print(json.dumps(dict(state=res["state"],selected=res["selected_configuration"],selection=str(selection_path))),flush=True)


def infer_saved(a,data,selection,input_arm,dev):
    import torch
    n,side=len(data[0]),data[0].shape[-1]
    head=selection["model_head"]
    prob=np.zeros((n,side,side),np.float32);size=np.zeros(n,np.float32);qual=np.zeros((n,len(LEVELS)),np.float32)
    for f in np.unique(data[4]):
        ids=np.nonzero(data[4]==f)[0]
        x=torch.from_numpy(model_inputs(data,ids,int(f),input_arm)).to(dev)
        states=selection["checkpoint"][input_arm][str(int(f))]
        total=0
        for entry in states:
            if file_sha(entry["path"])!=entry["sha256"]: raise RuntimeError("checkpoint changed after selection")
            ck=torch.load(entry["path"],map_location="cpu",weights_only=True)
            if ck["head"]!=head or ck["channels"]!=MODEL_CHANNELS: raise RuntimeError("checkpoint head mismatch")
            net=make_net(MODEL_CHANNELS,head).to(dev);net.load_state_dict(ck["weights"]);net.eval()
            mu,sd=ck["mu"].to(dev),ck["sd"].to(dev)
            with torch.inference_mode():
                if head=="quality":
                    cols=[]
                    for t in LEVELS:
                        cols.append(torch.cat([net(torch.cat([(x[i:i+128]-mu)/sd,
                            (x[i:i+128,:1]>t).float()],1))[:,0] for i in range(0,len(x),128)]).sigmoid())
                    val=torch.stack(cols,1).cpu().numpy()
                else:
                    val=torch.cat([net((x[i:i+128]-mu)/sd)[:,0] for i in range(0,len(x),128)])
                    val=(val.sigmoid() if head=="dense" else val).cpu().numpy()
            total=total+val/len(states)
        if not np.isfinite(total).all(): raise RuntimeError("nonfinite selected prediction")
        if head=="dense":prob[ids]=total
        elif head=="size":size[ids]=total
        else:qual[ids]=total
    return dict(prob=prob,size=size if head=="size" else None,qual=qual if head=="quality" else None)


def photo_bootstrap(rows, repetitions=2000, seed=0):
    """Paired bootstrap over connected S/Q photograph groups, with fixed models."""
    parent=list(range(len(rows)));seen={}
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(rows):
        for key in ("support","query"):
            uid=Path(r[key]).stem
            if uid in seen: parent[find(i)]=find(seen[uid])
            else: seen[uid]=i
    group_ids={};labels=[]
    for i in range(len(rows)):
        label=find(i)
        if label not in group_ids:group_ids[label]=len(group_ids)
        labels.append(group_ids[label])
    G=len(group_ids)
    counts=np.random.default_rng(seed).multinomial(G,np.ones(G)/G,size=repetitions).astype(float)
    return counts[:,np.array(labels)],G


def confirmation_phase(a):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    selection_path=Path(a.choice).resolve();sel=json.loads(selection_path.read_text())
    if sel["state"]!="SELECTION_FROZEN" or sel.get("confirmation_queryGT_opened"):
        raise RuntimeError("invalid selection/GT embargo")
    if file_sha(__file__)!=sel["script_sha256"]:raise RuntimeError("head source changed after selection")
    if file_sha(sel["train_manifest"])!=sel["train_manifest_sha256"]:raise RuntimeError("training manifest changed")
    if sel["pca_identity"]["sha256"] and file_sha(sel["pca_identity"]["path"])!=sel["pca_identity"]["sha256"]:
        raise RuntimeError("PCA basis changed")
    out=Path(a.out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    receipt_path=out.with_name("confirmation_prediction_freeze.json")
    consumed=selection_path.with_name("confirmation_consumed.json")
    if receipt_path.exists() or consumed.exists():raise RuntimeError("confirmation already attempted; no automatic repeated opening")
    cm=json.loads(Path(a.confirm_manifest).read_text());rows=cm["episodes"]
    data=load_dir(Path(a.cache)/"confirm",rows,labels=False)
    dev=runtime_device(a)
    start=time.monotonic()
    chosen=infer_saved(a,data,sel,sel["input_arm"],dev)
    control=infer_saved(a,data,sel,"score_only",dev)
    sn=data[0][:,0]
    masks=dict(chosen=prediction_mask(chosen,sn,sel["head"]),score_control=prediction_mask(control,sn,sel["head"]),native=sn>.5)
    frozen=out.with_name("confirmation_predictions.npz");np.savez_compressed(frozen,**masks)
    receipt=dict(state="PREDICTIONS_FROZEN_GT_SEALED",selection_sha256=file_sha(selection_path),
        predictions_sha256=file_sha(frozen),selected_configuration=dict(input_arm=sel["input_arm"],head=sel["head"],variant=sel["variant"]),
        confirm_manifest_sha256=file_sha(a.confirm_manifest),queryGT_opened=False,episodes=len(rows))
    receipt_path.write_text(json.dumps(receipt,indent=2))
    consumed.write_text(json.dumps(dict(state="GT_OPENING_STARTED",selection_sha256=receipt["selection_sha256"],
        receipt=str(receipt_path),predictions_sha256=receipt["predictions_sha256"]),indent=2))
    # The first query-label opening follows BOTH selection and prediction commits.
    labels_by_path={};targets=[]
    for r in rows:
        path=Path(r.get("query_mask_path",str(Path(cm["annotation_root"])/Path(r["query"]).with_suffix(".png"))))
        if str(path) not in labels_by_path:
            with Image.open(path) as image: labels_by_path[str(path)]=np.asarray(image).copy()
        target=torch.from_numpy((labels_by_path[str(path)]==r["c"]+1).copy()).float()[None,None]
        model=F.interpolate(target,(1024,1024),mode="nearest")
        targets.append(F.avg_pool2d(model,1024//sn.shape[-1])[0,0].numpy())
    tf=np.stack(targets);cls=data[3];Fte=data[4];flat=tf.reshape(len(tf),-1)
    def iu(pred):
        q=pred.reshape(len(tf),-1);i=(flat*q).sum(1)
        return np.stack([i,flat.sum(1)+q.sum(1)-i],1)
    baseline=iu(masks["native"])
    weights,photo_groups=photo_bootstrap(rows)
    episode_weights=np.random.default_rng(0).multinomial(len(tf),np.ones(len(tf))/len(tf),size=2000).astype(float)
    score=lambda t,ids=slice(None):float(miou(t[ids,0],t[ids,1],cls[ids])[0])
    def metric(mask,reference=baseline):
        t=iu(mask)
        delta=miou(t[:,0],t[:,1],cls,weights)-miou(reference[:,0],reference[:,1],cls,weights)
        episode_delta=miou(t[:,0],t[:,1],cls,episode_weights)-miou(reference[:,0],reference[:,1],cls,episode_weights)
        by={str(f):score(t,Fte==f)-score(reference,Fte==f) for f in np.unique(Fte)}
        return dict(patch_miou=score(t),gain=score(t)-score(reference),ci95=np.percentile(delta,[2.5,97.5]).tolist(),
            by_fold=by,folds_up=sum(v>0 for v in by.values()),photo_groups=photo_groups,
            ci_unit="support/query photo-connected groups; models fixed",
            diagnostic_episode_CI95=np.percentile(episode_delta,[2.5,97.5]).tolist())
    area=np.round(flat.sum(1)).astype(int)
    oracle=level_of_area(sn.reshape(len(tf),-1),area).reshape(sn.shape)
    report=dict(state="COMPLETED",scope="one selected T1 configuration and matched same-head/same-variant score control; patch-level without CRF",
        episodes=len(rows),training_epochs=sel["recipe"]["epochs"],seeds=sel["recipe"]["seeds"],
        score_control_capacity_matched=True,channel_count=MODEL_CHANNELS,
        relation_names=list(RELATION_NAMES),relation_names_sha256=RELATION_NAMES_SHA256,relation_schema_version=RELATION_SCHEMA_VERSION,
        selected_configuration=receipt["selected_configuration"],
        native_patch_miou=score(baseline),chosen=metric(masks["chosen"]),score_only_control=metric(masks["score_control"]),
        chosen_minus_score_control=metric(masks["chosen"],iu(masks["score_control"])),
        true_area_diagnostic=metric(oracle),true_area_scope="GT intervention, NOT task/information upper bound",
        selection_sha256=receipt["selection_sha256"],prediction_freeze=receipt,
        exposure_registry=dict(queryGT_opened_after_prediction_commit=True,queryPNG_open_count=len(labels_by_path),
            one_label_decode_per_unique_query=True,all_scoring_rows_share_labels=True,no_confirmation_fit=True),
        interval_scope="primary paired S/Q photo-connected-group bootstrap; episode bootstrap diagnostic only; selected models fixed",elapsed_s=time.monotonic()-start)
    report["passes"]=bool(report["chosen"]["gain"]>=GATE["gain"] and report["chosen"]["ci95"][0]>0 and report["chosen"]["folds_up"]>=GATE["folds"] and report["chosen_minus_score_control"]["ci95"][0]>0)
    receipt.update(state="SCORED_ONCE",queryGT_opened=True,queryPNG_open_count=len(labels_by_path))
    receipt_path.write_text(json.dumps(receipt,indent=2));out.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(state="COMPLETED",selected=report["selected_configuration"],gain=report["chosen"]["gain"])),flush=True)


def main_run(a):
    if a.phase=="development":development_phase(a)
    elif a.phase=="confirmation":confirmation_phase(a)
    else:
        development_phase(a)
        a.choice=str(Path(a.out).with_name("selection.json"));a.out=str(Path(a.out).parent/"confirmation"/"report.json")
        confirmation_phase(a)

def selfcheck():
    """A world where the score overshoots the target by an attached region that only a second map tells apart."""
    rng = np.random.default_rng(0)
    side, K = 16, RELATION_CHANNELS

    def make(m, fold):
        maps, tf = np.zeros((m, K, side, side), np.float32), np.zeros((m, side, side), np.float32)
        for j in range(m):
            x0, w0 = int(rng.integers(1, 5)), int(rng.integers(3, 7))
            extra = int(rng.integers(0, 5))  # width of the attached region; zero in some episodes
            tf[j, 4:12, x0:x0 + w0] = 1
            s = rng.uniform(0.0, 0.3, (side, side))
            s[4:12, x0:x0 + w0] = rng.uniform(0.6, 1.0, (8, w0))
            s[4:12, x0 + w0:x0 + w0 + extra] = rng.uniform(0.55, 0.95, (8, extra))  # overlaps the target's scores
            maps[j, 0] = (s - s.min()) / (s.max() - s.min())
            maps[j, 1] = tf[j] + 0.1 * rng.normal(size=s.shape)  # the map that tells the attached region apart
            maps[j, 2] = rng.normal(size=s.shape)
        return maps, np.zeros((m,4,16,side,side),np.float32), tf, np.array([fold+4*(j%20) for j in range(m)]), np.full(m,fold)
    cat = lambda parts: tuple(np.concatenate(x) for x in zip(*parts))
    train, test = cat([make(50, f) for f in range(4)]), cat([make(8, f) for f in range(4)])
    test = test[:3] + (np.array([int(f) + 4 * (j % 3) for j, f in enumerate(test[4])]), test[4])
    a = argparse.Namespace(epochs=25, seeds=1, batch=32)
    res = evaluate(a, train, test, "cpu", log=lambda s: None)
    few = evaluate(a, subset(train, 10), test, "cpu", log=lambda s: None, arms=("relations",), extra_heads=False)
    sn = np.array([[0.9, 0.2, 0.7, 0.1]])
    out = dict(level_of_area_takes_top_scores=bool((level_of_area(sn, np.array([2])) == np.array([[True, False, True, False]])).all()),
               quality_head_scores_every_cut=bool(np.isfinite(res["arms"]["relations"]["quality"]["patch_miou"])),
               relation_head_beats_native=bool(res["arms"]["relations"]["dense"]["gain"] > 5),
               relation_head_beats_score_only=bool(res["arms"]["relations"]["dense_over_score_only"]["gain"] > 3),
               size_head_reduces_area_error=bool(res["arms"]["relations"]["size"]["log_error_std"] < res["foris_own_area"]["log_error_std"]),
               verdict_reads_it=bool(res["verdict"]["passes"]),
               more_data_is_not_worse=bool(res["arms"]["relations"]["dense"]["gain"] >= few["arms"]["relations"]["dense"]["gain"] - 3))
    out.update(channel_count=MODEL_CHANNELS,relation_schema_version=RELATION_SCHEMA_VERSION,
               relation_names=list(RELATION_NAMES),relation_names_sha256=RELATION_NAMES_SHA256,
               checks=len(out), passed=int(sum(out.values())), state="PASSED" if all(out.values()) else "FAILED",
               gains={k: round(v["dense"]["gain"], 2) for k, v in res["arms"].items()})
    print(json.dumps(out, indent=1))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--fixture",action="store_true",help="tiny CPU contract checks only; never production training")
    p.add_argument("--phase",choices=("development","confirmation","all"),default="all")
    p.add_argument("--train-manifest")
    p.add_argument("--confirm-manifest")
    p.add_argument("--choice")
    p.add_argument("--cache")
    p.add_argument("--run")
    p.add_argument("--out")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--curve", default="100,200,400", help="training episodes per fold for the learning curve; the full set is always run")
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args()
    import torch
    torch.set_num_threads(2)
    if a.selfcheck:
        out = selfcheck()
        if a.out:
            Path(a.out).write_text(json.dumps(out, indent=1))
        raise SystemExit(0 if out["state"] == "PASSED" else 1)
    if a.phase in ("development","all") and (not a.run or not a.train_manifest): p.error("DEV requires --run/--train-manifest")
    if a.phase in ("confirmation","all") and not a.confirm_manifest: p.error("confirmation requires metadata --confirm-manifest")
    if a.phase=="confirmation" and not a.choice:p.error("--choice required; confirmation never fits")
    main_run(a)


if __name__ == "__main__":
    main()
