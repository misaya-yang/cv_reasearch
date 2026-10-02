"""Limited actual-evidence test: crop similarities -> purity -> shared masses.

Uses the existing 300-episode crop cache, not new encoder calls. Categories are
split four ways and overlapping reference/query images are excluded. These are
development validation data, not an untouched benchmark. No intervals or
statistical certificate are claimed for these fitted heads.
"""
import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import torch
from scipy.optimize import lsq_linear
from sklearn.ensemble import HistGradientBoostingRegressor
from cpu_region_readout import ids
from mass_decision import common_atoms, iou_from_masses

torch.set_num_threads(2)


def node_mask(children, n, node):
    mask = np.zeros(n, bool)
    stack = [int(node)]
    while stack:
        v = stack.pop()
        if v < n:
            mask[v] = True
        else:
            stack.extend(children[v-n])
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    crop = torch.load(args.cache / "look_f0_300.look.pt", map_location="cpu", weights_only=False)
    trees = torch.load(args.cache / "l3b_f0.l3.pt", map_location="cpu", weights_only=False)
    identity = ids(args.data, 0, len(crop))
    keys = ["F1", "plain_cls", "plain_pool", "plain_poold", "grey_cls", "grey_pool", "grey_poold"]
    for t, tree, (cls, query, reference) in zip(crop, trees, identity, strict=True):
        assert t["c"] == tree["c"] == cls and t["e"] == tree["e"]
        n = len(tree["g"])
        masks = np.stack([node_mask(tree["children"], n, v) for v in t["top"]])
        assert np.array_equal(masks.sum(axis=1), t["area"]), "Tree/crop cache geometry differs."
        membership, areas, labels = common_atoms(masks)
        prior = np.bincount(labels, weights=tree["ev"]["q30"].astype(float), minlength=len(areas)) / areas
        global_mass = float(prior @ areas)
        x = np.stack([t[k].astype(float) for k in keys] + [np.log(t["area"] / n), np.full(len(masks), global_mass/n)], axis=1)
        t.update(x=x, images=(query,reference), membership=membership, atoms=areas, prior=prior,
                 purity=t["fg"].astype(float)/t["area"], global_mass=global_mass)
    totals, errors, rows, train_sizes = {}, [], [], []
    for fold in range(4):
        test = [t for t in crop if (t["c"]//4)%4 == fold]
        excluded = {im for t in test for im in t["images"]}
        train = [t for t in crop if (t["c"]//4)%4 != fold and not excluded.intersection(t["images"])]
        train_sizes.append(len(train))
        x = np.concatenate([t["x"] for t in train])
        model_args = dict(max_iter=120,max_leaf_nodes=7,l2_regularization=5,
                          min_samples_leaf=25,early_stopping=False,random_state=0)
        quality_head = HistGradientBoostingRegressor(**model_args).fit(x,np.concatenate([t["iou"] for t in train]))
        purity_head = HistGradientBoostingRegressor(**model_args).fit(x,np.concatenate([t["purity"] for t in train]))
        for t in test:
            prediction = np.clip(purity_head.predict(t["x"]),0,1)
            errors.extend(np.abs(prediction-t["purity"]).tolist())
            b = t["membership"] * t["atoms"][None,:] / t["area"][:,None]
            # Fixed weak shrinkage; never selected using these query labels.
            design = np.vstack([b,.1*np.eye(len(t["atoms"]))])
            response = np.r_[prediction,.1*t["prior"]]
            fit = lsq_linear(design,response,bounds=(0,1),tol=1e-7).x
            coherent_scores = iou_from_masses(t["membership"],t["atoms"],fit*t["atoms"])
            unshared_mass = prediction*t["area"]
            global_estimate = max(t["global_mass"],float(unshared_mass.max()))
            raw_scores = unshared_mass/(t["area"]+global_estimate-unshared_mass+1e-9)
            choices = {"F1":0, "direct_iou_head":int(quality_head.predict(t["x"]).argmax()),
                       "unshared_mass":int(raw_scores.argmax()),"coherent_mass":int(coherent_scores.argmax())}
            result = {"fold":fold,"episode":int(t["e"]),"class":int(t["c"]),"counts":{}}
            # Query GT enters only here, after every choice has been made.
            for name,j in choices.items():
                inter=float(t["fg"][j]);union=float(t["area"][j]+t["G"]-inter)
                value=totals.setdefault(name,{}).setdefault(int(t["c"]),[0.,0.,[]]);value[0]+=inter;value[1]+=union;value[2].append(inter/max(union,1e-9))
                result["counts"][name]=[inter,union]
            rows.append(result)
    summary={k:{"class_miou":100*float(np.mean([v[0]/max(v[1],1e-9) for v in d.values()])),
                "mean_episode_iou":100*float(np.mean([x for v in d.values() for x in v[2]]))} for k,d in totals.items()}
    result={"status":"development only; full candidate crop scores; class and image exclusion; no calibrated certificate",
            "n":len(rows),"train_episodes":train_sizes,"purity_mae":float(np.mean(errors)),"summary":summary,"rows":rows,"gpu_used":False}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2),flush=True)


if __name__=="__main__":
    main()
