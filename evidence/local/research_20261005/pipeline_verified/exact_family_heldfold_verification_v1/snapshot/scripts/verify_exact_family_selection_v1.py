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
    row, ids, recipe, final_record = job; masks = []; loaded = {}
    for arm in ids:
        record = row["masks"][arm]; path = record["path"]
        if path not in loaded:
            names = {row["masks"][x]["key"] for x in ids if row["masks"][x]["path"] == path}
            with np.load(path, allow_pickle=False) as z: loaded[path] = {name:z[name].copy() for name in names}
        value = packed(loaded[path][record["key"]])
        if hashlib.sha256(value.tobytes()).hexdigest() != record["packed_array_sha256"]: raise ValueError("Packed producer identity changed")
        masks.append(value)
    selected = packed_recipe(masks, recipe)
    with np.load(final_record["path"], allow_pickle=False) as z:
        finalized = packed(z[final_record["key"]])
        if not np.array_equal(selected,finalized): raise ValueError("Independent main complete mask differs from sealed finalist")
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
    if read(run / "state.json")["state"] != "FINAL_FIXED_RECIPES_RECOUNT_COMPLETE": raise ValueError("Wait for completed final recount")
    if optimization["state"] != "EXACT_GLOBAL_DEV_OPTIMUM" or complete["state"] != "MEMBERSHIP_HISTOGRAM_COMPLETE": raise ValueError("Require completed optimization/histogram")
    if a.main_arm is None or a.main_count_key is None: raise ValueError("Explicit producer finalized main arm/count key required")
    rows, k = library["rows"], len(arms); ids = [arm["id"] for arm in arms]
    if len(rows) != 4000 or len({row["key"] for row in rows}) != 4000 or complete["n"] != 4000:
        raise ValueError("Require all4000 original draws")
    if optimization["distinct_producers"] != k or optimization["recipes_evaluated"] != k * (1 << (2*(k-1))): raise ValueError("Global enumeration dimension differs")
    recipe = optimization["selected_recipes"][0]; p, adds, deletes = indices(k, recipe)
    if recipe["origin_method"] != ids[p] or recipe["add_methods"] != [ids[i] for i in adds] or recipe["delete_methods"] != [ids[i] for i in deletes]: raise ValueError("Selected recipe index/name binding differs")
    if a.main_arm not in report["selected_global_DEV_recipes"] or not a.main_arm.startswith("recipe"): raise ValueError("Explicit main arm is not producer's selected global recipe")
    main_index = int(a.main_arm[len("recipe"):])
    if report["recipes"][main_index] != recipe: raise ValueError("Finalized main recipe differs from selected optimum")
    fs = read(run / "finalists_sealed.json")
    if fs["state"] != "ALL_FINALIST_MASKS_SEALED" or fs["n"] != 4000 or fs["query_GT_read_in_final_mask_construction"] is not False: raise ValueError("Full finalist mask seal required")
    checked(run / "manifest.json",fs["manifest_sha256"]); checked(run / "final_recipes.json",fs["recipes_sha256"])
    if set(fs["predictions"]) != {row["key"] for row in rows}: raise ValueError("Finalist mask manifest incomplete")
    source_files = {}
    for row in rows:
        for arm in ids:
            r = row["masks"][arm]
            if r["path"] in source_files and source_files[r["path"]] != r["sha256"]: raise ValueError("Conflicting source receipt")
            source_files[r["path"]] = r["sha256"]
        r = row["packet"]; source_files[r["path"]] = r["sha256"]
        source_files[str(run / "finalists" / (row["key"] + ".npz"))] = fs["predictions"][row["key"]]
    for path,digest in source_files.items(): checked(path,digest)
    source_digests = {name:sha(run/name) for name in ("manifest.json","arms.json","optimization.json","histogram_complete.json","patterns.npy","class_patterns.npy","baseline_counts.npz","report.json","final_counts.npz","final_recipes.json","finalists_sealed.json")}
    # No new search: only the explicitly selected global recipe is evaluated.
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn")) as pool:
        jobs = [(row,ids,recipe,dict(path=str(run / "finalists" / (row["key"] + ".npz")),key=a.main_arm)) for row in rows]
        replay = list(pool.map(replay_one,jobs,chunksize=4))
    arrays = {arm:np.array([r["iu"][arm] for r in replay],np.int64) for arm in (*ids,"selected")}
    histogram = np.load(run / "patterns.npy",mmap_mode="r",allow_pickle=False)
    lut = membership_lut(k, recipe)
    selected_histogram_iu = np.stack((histogram[:,lut,1].sum(axis=1,dtype=np.int64),histogram[:,:,1].sum(axis=1,dtype=np.int64)+histogram[:,lut,0].sum(axis=1,dtype=np.int64)),axis=1)
    if not np.array_equal(arrays["selected"],selected_histogram_iu): raise ValueError("Full packed main recipe differs from membership I/U")
    with np.load(run / "baseline_counts.npz",allow_pickle=False) as z:
        if any(not np.array_equal(arrays[arm],z[arm]) for arm in ids): raise ValueError("Original single-producer I/U parity failed")
    with np.load(run / "final_counts.npz",allow_pickle=False) as z:
        if not np.array_equal(arrays["selected"],z[a.main_count_key]): raise ValueError("Finalized main count export differs")
    classes = np.array([row["c"] for row in rows]); groups = photo_groups(rows); g=int(groups.max())+1
    class_patterns = np.load(run / "class_patterns.npy",mmap_mode="r",allow_pickle=False)
    class_iu = np.stack((class_patterns[:,lut,1].sum(axis=1,dtype=np.int64),class_patterns[:,:,1].sum(axis=1,dtype=np.int64)+class_patterns[:,lut,0].sum(axis=1,dtype=np.int64)),axis=1)
    expected_class_iu = np.array([arrays["selected"][classes==c].sum(axis=0,dtype=np.int64) for c in complete["class_ids"]])
    if not np.array_equal(class_iu,expected_class_iu): raise ValueError("Selected class I/U membership aggregation differs")
    draws = np.random.RandomState(0).randint(g,size=(2000,g))
    if report["bootstrap"]["draws"] != 2000 or report["bootstrap"]["rng"] != "RandomState(0)" or report["photo_groups"] != g:
        raise ValueError("Finalized bootstrap protocol differs")
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
        saved=report["contrasts"][a.main_arm][arm]
        error=max(error,abs(value["gain"]-saved["gain"]),float(np.abs(np.array(value["ci95"])-saved["ci95"]).max()))
        if any(value[key]!=saved[key] for key in ("up","down","tie")):raise ValueError("Final main paired signs differ")
    folds={}
    for fold in range(4):
        selected=np.array([row["fold"]==fold for row in rows])
        values={arm:metric(value[selected],classes[selected]) for arm,value in arrays.items()}
        folds[str(fold)]=dict(n=int(selected.sum()),scores=values,gain_vs_producers={arm:values["selected"]-values[arm] for arm in ids})
        error=max(error,abs(values["selected"]-report["folds"][str(fold)]["scores"][a.main_arm]))
        for arm in ids:error=max(error,abs(values[arm]-report["folds"][str(fold)]["scores"][arm]))
    if error>1e-10:raise ValueError("Main score/CI/fold reconstruction differs: "+str(error))
    for path,digest in source_files.items():checked(path,digest)
    for name,digest in source_digests.items():checked(run/name,digest)
    result=dict(state="EXACT_FAMILY_MAIN_RECIPE_INDEPENDENTLY_VERIFIED",n=4000,K=k,recipes_evaluated=optimization["recipes_evaluated"],recipe=recipe,
        outer_complete_mask_producer_count_proxy=recipe["cost"],cost_definition=optimization["cost_definition"],
        main_arm=a.main_arm,scores=scores,contrasts=contrasts,folds=folds,photo_groups=g,
        all_selected_full_masks_replayed_and_sealed_bit_exact=True,all_membership_selected_IU_exact=True,all_selected_class_IU_exact=True,all_original_single_producer_IU_exact=True,
        main_score_CI_fold_max_error_pp=error,source_files_sha256=source_digests,verifier_sha256=sha(Path(__file__)),
        bootstrap=dict(draws=2000,rng="RandomState(0)",unit="connected support/query photographs"),
        selection_boundary="Fixed recipe selected with GT-labelled DEV; paired intervals condition on this selected recipe and are not selection-adjusted or fresh confirmation",
        query_GT_role="independent score of only the already selected main recipe and original complete producers",new_search=False,new_encoder=False,
        seconds=time.monotonic()-started)
    write(a.out / "verification.json",result)
    np.savez_compressed(a.out / "recomputed_counts.npz",**arrays)
    print(json.dumps(dict(state=result["state"],main_score=scores["selected"],error=error,seconds=result["seconds"])),flush=True)


def verify_held(a):
    """Replay each held fold's own already selected recipe; never reuse one global LUT."""
    started=time.monotonic();run=a.run;source=a.source
    if source is None:raise ValueError("Explicit parent membership source required")
    if a.out.exists():raise FileExistsError("Fresh held-fold verification output required")
    state,report,selection,seal,library,arms=[read(run/name) for name in ("state.json","report.json","selection.json","sealed.json","manifest.json","arms.json")]
    if state["state"]!="HELD_FOLD_FIXED_RECIPES_RECOUNT_COMPLETE" or seal["state"]!="ALL_HELDFOLD_MASKS_SEALED" or seal["n"]!=4000:
        raise ValueError("Completed held-fold masks/recount required")
    checked(run/"manifest.json",seal["manifest_sha256"]);checked(run/"selection.json",seal["selection_sha256"])
    checked(source/"manifest.json",report["source_manifest_sha256"])
    if read(source/"manifest.json")!=library or read(source/"arms.json")!=arms:raise ValueError("Parent membership library identities differ")
    if seal["query_GT_read_in_mask_construction"] is not False or selection["folds"]!=report["selection"]:raise ValueError("Held-fold selection/seal contract differs")
    rows=library["rows"];k=len(arms);ids=[arm["id"] for arm in arms]
    folds=np.array([row["fold"] for row in rows]);classes=np.array([row["c"] for row in rows])
    if len(rows)!=4000 or len({row["key"] for row in rows})!=4000 or set(selection["folds"])!={"0","1","2","3"} or selection["library_sources"]!=k:
        raise ValueError("Full4000/fourfold source cohort required")
    if set(seal["predictions"])!={row["key"] for row in rows}:raise ValueError("Held-fold mask manifest incomplete")
    recipes={}
    for fold in range(4):
        item=selection["folds"][str(fold)];recipe=item["selected"];p,adds,deletes=indices(k,recipe)
        if recipe["origin_method"]!=ids[p] or recipe["add_methods"]!=[ids[i] for i in adds] or recipe["delete_methods"]!=[ids[i] for i in deletes]:raise ValueError("Held recipe index/name binding differs")
        held=folds==fold;training=~held
        if item["held_n"]!=int(held.sum()) or item["training_n"]!=int(training.sum()) or item["held_classes"]!=np.unique(classes[held]).tolist() or item["training_classes"]!=np.unique(classes[training]).tolist():
            raise ValueError("Actual train/held row or class identities differ")
        if set(item["held_classes"])&set(item["training_classes"]):raise ValueError("Held classes overlap training classes")
        recipes[str(fold)]=recipe
    source_files={};jobs=[]
    for row in rows:
        for arm in ids:
            r=row["masks"][arm]
            if r["path"] in source_files and source_files[r["path"]]!=r["sha256"]:raise ValueError("Conflicting producer receipt")
            source_files[r["path"]]=r["sha256"]
        r=row["packet"];source_files[r["path"]]=r["sha256"]
        path=run/"finalists"/(row["key"]+".npz");source_files[str(path)]=seal["predictions"][row["key"]]
        jobs.append((row,ids,recipes[str(row["fold"])],dict(path=str(path),key="recipe0")))
    # All complete sources/finalists/packets pass before independent GT evaluation.
    for path,digest in source_files.items():checked(path,digest)
    own_digests={name:sha(run/name) for name in ("manifest.json","arms.json","selection.json","sealed.json","report.json","counts.npz","state.json")}
    parent_digests={name:sha(source/name) for name in ("manifest.json","arms.json","patterns.npy","baseline_counts.npz","histogram_complete.json")}
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context("spawn")) as pool:replay=list(pool.map(replay_one,jobs,chunksize=4))
    arrays={arm:np.array([r["iu"][arm] for r in replay],np.int64) for arm in (*ids,"selected")}
    with np.load(run/"counts.npz",allow_pickle=False) as z:
        if not np.array_equal(arrays["selected"],z["heldfold_recipe"]):raise ValueError("Held main count export differs")
        if any(not np.array_equal(arrays[arm],z[arm]) for arm in ids):raise ValueError("Held baseline count export differs")
    with np.load(source/"baseline_counts.npz",allow_pickle=False) as z:
        if any(not np.array_equal(arrays[arm],z[arm]) for arm in ids):raise ValueError("Original single-producer I/U differs")
    histogram=np.load(source/"patterns.npy",mmap_mode="r",allow_pickle=False)
    if histogram.shape!=(4000,1<<k,2):raise ValueError("Parent membership shape differs")
    expected=np.empty((4000,2),np.int64);class_iu={}
    for fold in range(4):
        ix=folds==fold;h=histogram[ix];lut=membership_lut(k,recipes[str(fold)])
        expected[ix]=np.stack((h[:,lut,1].sum(axis=1,dtype=np.int64),h[:,:,1].sum(axis=1,dtype=np.int64)+h[:,lut,0].sum(axis=1,dtype=np.int64)),axis=1)
        for c in np.unique(classes[ix]):
            ch=histogram[classes==c].sum(axis=0,dtype=np.int64)
            value=np.array([ch[lut,1].sum(dtype=np.int64),ch[:,1].sum(dtype=np.int64)+ch[lut,0].sum(dtype=np.int64)],np.int64)
            if not np.array_equal(value,arrays["selected"][classes==c].sum(axis=0,dtype=np.int64)):raise ValueError("Held class I/U aggregation differs")
            class_iu[str(int(c))]=value.tolist()
    if not np.array_equal(expected,arrays["selected"]):raise ValueError("Per-fold selected membership I/U differs")
    groups=photo_groups(rows);g=int(groups.max())+1;draws=np.random.RandomState(0).randint(g,size=(2000,g))
    if report["bootstrap"]["draws"]!=2000 or report["bootstrap"]["rng"]!="RandomState(0)" or report["photo_groups"]!=g:raise ValueError("Held bootstrap protocol differs")
    weights=np.stack([np.bincount(draw,minlength=g) for draw in draws])[:,groups]
    scores={arm:metric(value,classes) for arm,value in arrays.items()};samples={arm:np.array([metric(value,classes,w) for w in weights]) for arm,value in arrays.items()}
    error=abs(scores["selected"]-report["scores"]["heldfold_recipe"]);contrasts={};fold_results={}
    for arm in ids:
        error=max(error,abs(scores[arm]-report["scores"][arm]));delta=arrays["selected"][:,0]/np.maximum(arrays["selected"][:,1],1)-arrays[arm][:,0]/np.maximum(arrays[arm][:,1],1)
        value=dict(gain=scores["selected"]-scores[arm],ci95=np.percentile(samples["selected"]-samples[arm],[2.5,97.5]).tolist(),up=int((delta>1e-12).sum()),down=int((delta< -1e-12).sum()),tie=int((np.abs(delta)<=1e-12).sum()))
        contrasts[arm]=value;saved=report["contrasts"]["heldfold_recipe"][arm]
        error=max(error,abs(value["gain"]-saved["gain"]),float(np.abs(np.array(value["ci95"])-saved["ci95"]).max()))
        if any(value[x]!=saved[x] for x in ("up","down","tie")):raise ValueError("Held paired sign counts differ")
    for fold in range(4):
        ix=folds==fold;point={arm:metric(value[ix],classes[ix]) for arm,value in arrays.items()}
        error=max(error,abs(point["selected"]-report["folds"][str(fold)]["scores"]["heldfold_recipe"]))
        for arm in ids:error=max(error,abs(point[arm]-report["folds"][str(fold)]["scores"][arm]))
        fold_results[str(fold)]=dict(n=int(ix.sum()),recipe=recipes[str(fold)],scores=point,gain_vs_producers={arm:point["selected"]-point[arm] for arm in ids})
    if error>1e-10:raise ValueError("Held score/CI/fold error "+str(error))
    for name,digest in own_digests.items():checked(run/name,digest)
    for name,digest in parent_digests.items():checked(source/name,digest)
    a.out.mkdir(parents=True)
    result=dict(state="STRICT_HELDFOLD_MAIN_INDEPENDENTLY_VERIFIED",n=4000,K=k,per_fold_recipes=recipes,
        distinct_recipe_tuples=len({(r["origin"],r["a"],r["b"]) for r in recipes.values()}),scores=scores,contrasts=contrasts,folds=fold_results,class_IU=class_iu,
        all4000_fold_specific_masks_replayed_and_sealed_bit_exact=True,all4000_selected_membership_IU_exact=True,all80_class_IU_exact=True,
        all_original12_producer_IU_exact=True,score_CI_fold_max_error_pp=error,photo_groups=g,
        source=dict(run=str(run.resolve()),membership_source=str(source.resolve()),held_files=own_digests,parent_files=parent_digests),
        bootstrap=dict(draws=2000,rng="RandomState(0)",unit="connected support/query photographs"),
        selection_boundary="Each fold's already selected fixed recipe uses three other folds' GT. Source-producer exposure is unchanged; paired CIs do not repeat selection; development reuse is not fresh confirmation.",
        resource_boundary="Outer producer count proxy3 per fold is not atomic upstream cost or measured seconds.",new_search=False,new_encoder=False,
        verifier_sha256=sha(Path(__file__)),seconds=time.monotonic()-started)
    write(a.out/"verification.json",result);np.savez_compressed(a.out/"recomputed_counts.npz",**arrays)
    print(json.dumps(dict(state=result["state"],main_score=scores["selected"],error=error,seconds=result["seconds"],recipe_tuples={f:[r["origin"],r["a"],r["b"]] for f,r in recipes.items()})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","verify","verify-held"))
    p.add_argument("--run",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source",type=Path)
    p.add_argument("--main-arm");p.add_argument("--main-count-key");p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if a.workers<1:p.error("Positive worker count required")
    {"prepare":prepare,"verify":verify,"verify-held":verify_held}[a.stage](a)


if __name__=="__main__":main()
