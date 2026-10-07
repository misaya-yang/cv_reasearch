#!/usr/bin/env python3
"""Necessary solver audits, paired counterexamples, native-four full outputs."""
from __future__ import annotations

from itertools import permutations
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment, nnls

from ics.cpu100.common import Episode, Result, load_episode, prototype_margin, render, sha, unit
from ics.cpu100.cross_image_matching_batch2 import (
    METHODS, CONTROLS, active_nnls, path_endpoints, opponent_dictionary,
    facility_greedy, _dictionary,
)
from check_cpu100_cross_image_matching import confusion, result_margin


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/local/cpu100_20261006/reports/cross_image_matching_batch2"
ARMS = METHODS | CONTROLS


def ep(q, r, wf):
    return Episode(q=unit(np.asarray(q, float)), r=unit(np.asarray(r, float)),
                   wf=np.asarray(wf, float), wvalid=np.ones(len(r)),
                   q_hw=(1, len(q)), r_hw=(1, len(r)), q_valid=np.ones(len(q)))


def audit():
    rng = np.random.default_rng(41)
    errors = []
    for k in range(1, 4):
        for example in range(8):
            a = rng.normal(size=(5, k)); q = rng.normal(size=5)
            if example == 7 and k > 1: a[:, -1] = a[:, 0]
            coeff, error = active_nnls(a.T@a, a.T@q, float(q@q))
            other, _ = nnls(a, q, maxiter=200)
            errors.append(abs(error - np.sum((q - a@other)**2)))
            assert np.all(coeff >= 0)
    assert max(errors) < 1e-8
    w = rng.uniform(.1, 1, size=(3, 4)); path_errors = []
    for no_q in (False, True):
        for no_r in (False, True):
            brute = np.zeros_like(w)
            for i in range(len(w)):
                for a in range(w.shape[1]):
                    for j in range(len(w)):
                        for b in range(w.shape[1]):
                            if (no_q and j == i) or (no_r and b == a): continue
                            brute[i, b] += w[i, a] * w[j, a] * w[j, b]
            path_errors.append(float(np.max(np.abs(path_endpoints(w, no_q, no_r) - brute))))
    assert max(path_errors) < 1e-10
    return dict(nnls_cost_max_error=max(errors), nnls_problems=len(errors),
                path_enumerator_max_error=max(path_errors), path_flag_settings=4)


def fixtures():
    record = {"audits": audit()}
    r = [[1,0,0],[0,1,0],[1,0,1],[0,1,1],[0,0,-1]]
    q = unit(np.array([[1,1,.3]], float))
    sparse = ep(q, r, [1,1,0,0,0])
    sparse_keys = ["cross_image_joint_sparse_role_removal", "cross_image_independent_sparse_control",
                   "cross_image_sparse_coefficient_vote_control", "cross_image_dense_class_cone_control",
                   "cross_image_convex_hull_control"]
    margins = {k: float(result_margin(ARMS[k], sparse)[0]) for k in sparse_keys}
    assert margins[sparse_keys[0]] > 0 > margins[sparse_keys[1]]
    assert margins[sparse_keys[4]] > 0
    negative = ep(q, np.vstack((unit(np.array(r, float)), q)), [1,1,0,0,0,0])
    negative_margin = float(result_margin(METHODS[sparse_keys[0]], negative)[0])
    assert negative_margin < 0
    record["joint_sparse"] = dict(
        positive_margins=margins, negative_target_margin=negative_margin,
        reflection="Joint OMP retains the generated dominant FG part where independent sparse/cone explanations choose BG. Full hull and simple coefficient vote also solve this witness: role-removal necessity is unproved. A wrong-role exact dictionary atom gives the best sparse reconstruction and deletes the same target; residual quality is not semantic quality.")

    vec = lambda a: np.column_stack((np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))))
    r = vec(np.array([-45.,0.,45.]))
    q = vec(np.r_[np.full(50,45.), np.full(8,-45.), 10.])
    facility = ep(q, r, [1,1,0]); truth = np.r_[np.zeros(50),np.ones(8),0].astype(bool)
    facility_keys = ["cross_image_exemplar_facility_cover", "cross_image_all_exemplars_control",
                     "cross_image_facility_unary_same_count_control", "cross_image_facility_random_same_count_control"]
    results = {k: confusion(result_margin(ARMS[k], facility), truth) for k in facility_keys}
    assert results[facility_keys[0]]["fp"] == 0 and results[facility_keys[0]]["fn"] == 0
    assert results[facility_keys[1]]["fp"] == 1 and results[facility_keys[2]]["fp"] == 1
    atoms, _, _ = _dictionary(facility)
    cost = 1 - facility.q @ atoms.T; weights = facility.q_valid / facility.q_valid.sum()
    opened, trace = facility_greedy(cost, weights)
    objective = lambda s: float(weights @ cost[:, s].min(1) + .05*len(s))
    best = min(objective([j for j in range(len(atoms)) if bit & (1<<j)])
               for bit in range(1, 1<<len(atoms)))
    assert abs(trace[-1]-objective(opened)) < 1e-12
    assert abs(trace[-1]-best) < 1e-12
    tiny = ep(vec(np.r_[np.full(50,45.),-45.]), r, [1,1,0])
    tiny_truth = np.r_[np.zeros(50),1].astype(bool)
    tiny_result = confusion(result_margin(METHODS[facility_keys[0]], tiny), tiny_truth)
    tiny_control = confusion(result_margin(CONTROLS[facility_keys[1]], tiny), tiny_truth)
    assert tiny_result["fn"] == 1 and tiny_control["fn"] == 0
    record["facility"] = dict(positive=results, tiny_target_negative=tiny_result,
                               tiny_all_atoms_control=tiny_control, small_fixture_optimum=best,
                               greedy_objective=trace[-1],
                               reflection="Sharing activation cost eliminates the isolated FG explanation beyond a same-count global-unary control. The same area-sensitive complexity prior deletes a genuine tiny target; no fixed target proportion or capacity is imposed, but this is a prior.")

    basis = np.eye(5)
    r = unit(np.array([basis[0], basis[1]+.1*basis[4], basis[1]-.1*basis[4],
                       .6*basis[0]+.8*basis[2], .6*basis[0]+.8*basis[2]+.1*basis[3]]))
    q = unit(np.array([basis[0], r[3], basis[1]+.05*basis[4], basis[1]-.05*basis[4]]))
    paths = ep(q, r, [1,1,1,0,0]); truth = np.array([False,False,True,True])
    path_keys = ["cross_image_nonreturn_path_consensus", "cross_image_ordinary_three_hop_control",
                 "cross_image_one_hop_path_control", "cross_image_no_query_return_control",
                 "cross_image_no_reference_return_control"]
    values = {k: result_margin(ARMS[k], paths) for k in path_keys}
    assert values[path_keys[0]][0] < 0 < values[path_keys[1]][0]
    assert values[path_keys[0]][1] > 0 > values[path_keys[1]][1]
    record["nonreturn_paths"] = dict(
        positive_and_negative_same_episode={k: confusion(v,truth) for k,v in values.items()},
        margins={k:v.tolist() for k,v in values.items()},
        reflection="Removing returns deletes the designated false FG hub but selects its true-BG bridge. Both ordinary and nonreturn have one FP. Different-anchor path support can exchange an error rather than improve the complete method; no path-independence or net benefit claim.")

    r = vec(np.array([-30.,0.,30.,-10.,20.,60.])); q = vec(np.array([0.,30.]))
    opponents = ep(q,r,[1,1,1,0,0,0]); truth = np.ones(2,bool)
    opponent_keys = ["cross_image_source_opponent_matching", "cross_image_nearest_opponent_median_control",
                     "cross_image_allpair_opponent_median_control", "cross_image_random_opponent_median_control",
                     "cross_image_raw_matched_mean_control"]
    values = {k:result_margin(ARMS[k],opponents) for k in opponent_keys}
    fg,bg = opponent_dictionary(opponents); cost=1-fg@bg.T
    indices=linear_sum_assignment(cost); actual=float(cost[indices].sum())
    brute=min(float(cost[np.arange(len(fg)),list(p)].sum()) for p in permutations(range(len(bg))))
    raw_error=float(np.max(np.abs((fg[indices[0]]-bg[indices[1]]).mean(0)-(fg.mean(0)-bg.mean(0)))))
    assert abs(actual-brute)<1e-12 and raw_error<1e-12
    assert values[opponent_keys[0]][0]>0>values[opponent_keys[1]][0]
    assert values[opponent_keys[0]][1]<0<values[opponent_keys[3]][1]
    record["opponent_matching"] = dict(
        confusion={k:confusion(v,truth) for k,v in values.items()},
        margins={k:v.tolist() for k,v in values.items()}, assignment_brute_error=abs(actual-brute),
        raw_mean_permutation_identity_error=raw_error,
        reflection="Source global assignment restores one FG token rejected by nearest-opponent median but deletes another FG token; random matching retains both, and all-pairs already restores the first. Matching optimum is not a semantic optimum. Fixed-version quality advantage is unsupported.")
    return record


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(schema="CPU100_CROSS_IMAGE_BATCH2_V1",synthetic=fixtures(),native_four=[])
    rows=json.loads((ROOT/"evidence/local/research_20261006/direct_dino_input_01a1100b/inputs4/manifest.json").read_text())["rows"]
    for row in rows:
        episode=load_episode(row); folder=OUT/"native_four"/episode.source_id;folder.mkdir(parents=True,exist_ok=True)
        with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
            baseline=render(episode,Result(prototype_margin(episode)))
        for key,fn in ARMS.items():
            started=time.perf_counter()
            with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
                result=fn(episode)
            elapsed=time.perf_counter()-started; outputs=render(episode,result)
            path=folder/f"{key}.npz"
            np.savez_compressed(path,signed_margin_native=result.margin,
                                signed_margin_work64=outputs["margin"],work_mask=outputs["work"],
                                original_mask=outputs["original"])
            report["native_four"].append(dict(episode=episode.source_id,method=key,
                inference_seconds=elapsed,original_fg_pixels=int(outputs["original"].sum()),
                changed_original_pixels_vs_prototype=int(np.count_nonzero(outputs["original"]!=baseline["original"])),
                complete_output=str(path),sha256=sha(path),info=result.info))
    report.update(query_GT_used=False,empirical_quality="native four activity only; no QGT/mIoU",
                  native_scope="four exposed native final-LN DINO RGB128 8x8x1024, not the server native1024 producer",
                  code_sha256=sha(ROOT/"src/ics/cpu100/cross_image_matching_batch2.py"))
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(methods=4,outputs=len(report["native_four"]),report=str(OUT/"report.json"))))


if __name__=="__main__":main()
