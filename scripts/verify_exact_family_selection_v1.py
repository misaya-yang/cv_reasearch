#!/usr/bin/env python3
"""One independent CPU check of a completed exact global A/B family result.

Prepare checks the Boolean contract on synthetic memberships only. Verify waits
for the producer's final recipe/report, reconstructs that one recipe from the
existing complete masks, and evaluates it against the existing query packets.
No search, model, parameter update, per-example selector or source mutation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ics.experiment import metric, photo_groups, sha

POPCOUNT = np.array([bin(x).count("1") for x in range(256)], np.uint8)


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"); tmp.replace(path)


def read(path): return json.loads(Path(path).read_text())
def checked(path, digest):
    if sha(path) != digest: raise ValueError("Changed sealed source: " + str(path))


def indices(k, recipe):
    p = recipe["origin"]
    if not isinstance(p, int) or not 0 <= p < k: raise ValueError("Invalid global origin")
    others = [j for j in range(k) if j != p]
    for name in ("a", "b"):
        if not isinstance(recipe[name], int) or not 0 <= recipe[name] < (1 << (k-1)):
            raise ValueError("Invalid global operator subset")
    return p, [j for i,j in enumerate(others) if recipe["a"] & (1 << i)], [j for i,j in enumerate(others) if recipe["b"] & (1 << i)]


def membership_lut(k, recipe):
    """Outside P0 use OR(add producers); inside P0 use AND(keep producers)."""
    p, adds, deletes = indices(k, recipe)
    states = np.arange(1 << k, dtype=np.uint32)
    outside = np.zeros(len(states), bool)
    inside = np.ones(len(states), bool)
    for j in adds: outside |= (states & (1 << j)) != 0
    for j in deletes: inside &= (states & (1 << j)) != 0
    return np.where((states & (1 << p)) != 0, inside, outside)


def packed_recipe(masks, recipe):
    p, adds, deletes = indices(len(masks), recipe)
    origin = masks[p]
    # Direct C=(P0 union union(Pi\P0))\union(P0\Pj).
    add = np.zeros_like(origin); delete = np.zeros_like(origin)
    for j in adds: add |= masks[j] & ~origin
    for j in deletes: delete |= origin & ~masks[j]
    direct = (origin | add) & ~delete
    # A separately written outside-OR / inside-AND representation.
    outside = np.zeros_like(origin); inside = np.full_like(origin, 255)
    for j in adds: outside |= masks[j]
    for j in deletes: inside &= masks[j]
    second = (~origin & outside) | (origin & inside)
    if not np.array_equal(direct, second): raise ValueError("Global recipe Boolean representations disagree")
    return direct


def synthetic_check():
    tested = 0
    for k in range(1, 5):
        states = np.arange(1 << k, dtype=np.uint32)
        masks = [np.packbits((states & (1 << j)) != 0) for j in range(k)]
        for p in range(k):
            for a in range(1 << (k-1)):
                for b in range(1 << (k-1)):
                    recipe = dict(origin=p, a=a, b=b)
                    actual = np.unpackbits(packed_recipe(masks, recipe))[:len(states)].astype(bool)
                    if not np.array_equal(actual, membership_lut(k, recipe)): raise ValueError("Independent packed/LUT contract failed")
                    tested += 1
    return dict(state="SYNTHETIC_GLOBAL_RECIPE_CONTRACT_PASSED",tested_recipes=tested,K_range=[1,4],query_GT_opened=False,
        enumeration="L*4^(L-1), excluding redundant own-origin operators",definition="C=(P0 union union(Pi\\P0))\\union(P0\\Pj)")


def packed(value):
    if value.dtype != np.uint8 or value.shape != (131072,): raise ValueError("Invalid packed complete1024 mask")
    return value


def replay_one(job):
    row, ids, recipe = job; masks = []; loaded = {}
    for arm in ids:
        record = row["masks"][arm]; path = record["path"]
        if path not in loaded:
            names = {row["masks"][x]["key"] for x in ids if row["masks"][x]["path"] == path}
            with np.load(path, allow_pickle=False) as z: loaded[path] = {name:z[name].copy() for name in names}
        value = packed(loaded[path][record["key"]])
        if hashlib.sha256(value.tobytes()).hexdigest() != record["packed_array_sha256"]: raise ValueError("Packed producer identity changed")
        masks.append(value)
    selected = packed_recipe(masks, recipe)
    # This is the first query truth access; main verified global recipe is fixed.
    with np.load(row["packet"]["path"], allow_pickle=False) as z: truth = packed(z["truth"].copy())
    count = lambda v:int(POPCOUNT[v].sum(dtype=np.int64))
    iu = {arm:[count(value & truth),count(value | truth)] for arm,value in zip(ids,masks)}
    iu["selected"] = [count(selected & truth),count(selected | truth)]
    return dict(key=row["key"],iu=iu,selected_packed_array_sha256=hashlib.sha256(selected.tobytes()).hexdigest())


def prepare(a):
    if a.out.exists(): raise FileExistsError("Fresh owned verification output required")
    contract = synthetic_check()
    a.out.mkdir(parents=True)
    write(a.out / "prepared.json", dict(contract, run=str(a.run.resolve()), verifier_sha256=sha(Path(__file__)),
        stats_sha256=sha(ROOT / "src/ics/experiment.py"), requires="producer optimization.json plus completed finalized main recipe/report before GT replay"))
    print(json.dumps(contract), flush=True)


def verify(a):
    """The exact finalized report/count key is supplied explicitly, never guessed."""
    started = time.monotonic(); synthetic_check()
    run = a.run; optimization, arms, library, complete, report = [read(run / name) for name in ("optimization.json", "arms.json", "manifest.json", "histogram_complete.json", "report.json")]
    if optimization["state"] != "EXACT_GLOBAL_DEV_OPTIMUM" or complete["state"] != "MEMBERSHIP_HISTOGRAM_COMPLETE": raise ValueError("Require completed optimization/histogram")
    if a.main_arm is None or a.main_count_key is None: raise ValueError("Explicit producer finalized main arm/count key required")
    rows, k = library["rows"], len(arms); ids = [arm["id"] for arm in arms]
    if len(rows) != 4000 or len({row["key"] for row in rows}) != 4000 or complete["n"] != 4000:
        raise ValueError("Require all4000 original draws")
    if optimization["distinct_producers"] != k or optimization["recipes_evaluated"] != k * (1 << (2*(k-1))): raise ValueError("Global enumeration dimension differs")
    recipe = optimization["selected_recipes"][0]; p, adds, deletes = indices(k, recipe)
    if recipe["origin_method"] != ids[p] or recipe["add_methods"] != [ids[i] for i in adds] or recipe["delete_methods"] != [ids[i] for i in deletes]: raise ValueError("Selected recipe index/name binding differs")
    source_files = {}
    for row in rows:
        for arm in ids:
            r = row["masks"][arm]
            if r["path"] in source_files and source_files[r["path"]] != r["sha256"]: raise ValueError("Conflicting source receipt")
            source_files[r["path"]] = r["sha256"]
        r = row["packet"]; source_files[r["path"]] = r["sha256"]
    for path,digest in source_files.items(): checked(path,digest)
    source_digests = {name:sha(run/name) for name in ("manifest.json","arms.json","optimization.json","histogram_complete.json","patterns.npy","class_patterns.npy","baseline_counts.npz","report.json","counts.npz")}
    # No new search: only the explicitly selected global recipe is evaluated.
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn")) as pool:
        replay = list(pool.map(replay_one,[(row,ids,recipe) for row in rows],chunksize=4))
    arrays = {arm:np.array([r["iu"][arm] for r in replay],np.int64) for arm in (*ids,"selected")}
    histogram = np.load(run / "patterns.npy",mmap_mode="r",allow_pickle=False)
    lut = membership_lut(k, recipe)
    selected_histogram_iu = np.stack((histogram[:,lut,1].sum(axis=1,dtype=np.int64),histogram[:,:,1].sum(axis=1,dtype=np.int64)+histogram[:,lut,0].sum(axis=1,dtype=np.int64)),axis=1)
    if not np.array_equal(arrays["selected"],selected_histogram_iu): raise ValueError("Full packed main recipe differs from membership I/U")
    with np.load(run / "baseline_counts.npz",allow_pickle=False) as z:
        if any(not np.array_equal(arrays[arm],z[arm]) for arm in ids): raise ValueError("Original single-producer I/U parity failed")
    with np.load(run / "counts.npz",allow_pickle=False) as z:
        if not np.array_equal(arrays["selected"],z[a.main_count_key]): raise ValueError("Finalized main count export differs")
    classes = np.array([row["c"] for row in rows]); groups = photo_groups(rows); g=int(groups.max())+1
    draws = np.random.RandomState(0).randint(g,size=(2000,g))
    weights = np.stack([np.bincount(draw,minlength=g) for draw in draws])[:,groups]
    scores = {arm:metric(value,classes) for arm,value in arrays.items()}
    error = abs(scores["selected"] - optimization["maximum_miou"])
    error = max(error,abs(scores["selected"]-report["scores"][a.main_arm]))
    for arm in ids:error=max(error,abs(scores[arm]-complete["scores"][arm]))
    samples = {arm:np.array([metric(value,classes,w) for w in weights]) for arm,value in arrays.items()}
    contrasts = {}
    for arm in ids:
        delta = arrays["selected"][:,0]/np.maximum(arrays["selected"][:,1],1)-arrays[arm][:,0]/np.maximum(arrays[arm][:,1],1)
        value=dict(gain=scores["selected"]-scores[arm],ci95=np.percentile(samples["selected"]-samples[arm],[2.5,97.5]).tolist(),up=int((delta>1e-12).sum()),down=int((delta< -1e-12).sum()),tie=int((np.abs(delta)<=1e-12).sum()))
        contrasts[arm]=value
        if arm in report["contrasts"][a.main_arm]:
            saved=report["contrasts"][a.main_arm][arm]
            error=max(error,abs(value["gain"]-saved["gain"]),float(np.abs(np.array(value["ci95"])-saved["ci95"]).max()))
            if any(value[key]!=saved[key] for key in ("up","down","tie")):raise ValueError("Final main paired signs differ")
    folds={}
    for fold in range(4):
        selected=np.array([row["fold"]==fold for row in rows])
        values={arm:metric(value[selected],classes[selected]) for arm,value in arrays.items()}
        folds[str(fold)]=dict(n=int(selected.sum()),scores=values,gain_vs_producers={arm:values["selected"]-values[arm] for arm in ids})
        error=max(error,abs(values["selected"]-report["folds"][str(fold)]["scores"][a.main_arm]))
    if error>1e-10:raise ValueError("Main score/CI/fold reconstruction differs: "+str(error))
    for path,digest in source_files.items():checked(path,digest)
    for name,digest in source_digests.items():checked(run/name,digest)
    result=dict(state="EXACT_FAMILY_MAIN_RECIPE_INDEPENDENTLY_VERIFIED",n=4000,K=k,recipes_evaluated=optimization["recipes_evaluated"],recipe=recipe,
        main_arm=a.main_arm,scores=scores,contrasts=contrasts,folds=folds,photo_groups=g,
        all_selected_full_masks_replayed=True,all_membership_selected_IU_exact=True,all_original_single_producer_IU_exact=True,
        main_score_CI_fold_max_error_pp=error,source_files_sha256=source_digests,verifier_sha256=sha(Path(__file__)),
        bootstrap=dict(draws=2000,rng="RandomState(0)",unit="connected support/query photographs"),
        selection_boundary="Fixed recipe selected with GT-labelled DEV; paired intervals condition on this selected recipe and are not selection-adjusted or fresh confirmation",
        query_GT_role="independent score of only the already selected main recipe and original complete producers",new_search=False,new_encoder=False,
        seconds=time.monotonic()-started)
    write(a.out / "verification.json",result)
    np.savez_compressed(a.out / "recomputed_counts.npz",**arrays)
    print(json.dumps(dict(state=result["state"],main_score=scores["selected"],error=error,seconds=result["seconds"])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","verify"))
    p.add_argument("--run",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--main-arm");p.add_argument("--main-count-key");p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if a.workers<1:p.error("Positive worker count required")
    prepare(a) if a.stage=="prepare" else verify(a)


if __name__=="__main__":main()
