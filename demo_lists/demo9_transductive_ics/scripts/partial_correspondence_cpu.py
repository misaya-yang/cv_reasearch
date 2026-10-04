#!/usr/bin/env python3
"""SERVER-only tiny enumeration/contracts for the full-state PCF CPU reference.

No pretrained model, real QueryGT or scientific result. The shape estimate is
arithmetic metadata, not a measured full-grid runtime or a readiness claim.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def require_server():
    if not sys.platform.startswith("linux") or not str(ROOT).startswith("/root/"):
        raise RuntimeError("SERVER CPU execution only; no local numerical/model tests")


def shape_estimate(query_shape=(64,64),source_shape=(64,64)):
    n=query_shape[0]*query_shape[1];m=source_shape[0]*source_shape[1]
    nnz=min(m*(m-1),312*m);calls=2*max(n-1,0) if m>=4 else 0
    return dict(Query_shape=list(query_shape),Source_shape=list(source_shape),
        all_Source_FG_assumed_for_memory_bound=True,Query_nodes=n,Source_states=m,
        dense_pi_bytes=8*n*m,two_BP_dense_log_work_arrays_bytes=16*n*m,
        combined_pi_and_BP_workspace_bytes=24*n*m,
        optional_full_matched_state_output_bytes=8*n*m,
        source_sparse_directed_nnz_bound=nnz,source_CPU_CSR_storage_conservative_bytes=12*nnz+8*(m+1),
        R_matvec_calls_per_arm_upper_bound=calls,
        sparse_multiply_add_ops_per_arm_upper_bound=2*calls*nnz,
        four_fixed_structural_arms_sparse_ops_upper_bound=8*calls*nnz,
        CPU_work_is_not_a_GPU_load=True,actual_SERVER_full_shape_profile_required=True,
        measured_full_runtime_seconds=None,dense_edge_M_squared_allocations=0,
        downsampling_or_Source_topk_allowed=False)


def run():
    require_server()
    import numpy as np
    sys.path.insert(0,str(ROOT))
    from tics.partial_correspondence import (RelationKernel,build_relation,build_query_tree,
        semantic_correspondence,tree_sum_product,binary_potts,run_controls,stable_source_permutation)
    checks=[];started=time.monotonic()
    source=np.ones((2,2),bool);relation=build_relation(source)
    dense=np.column_stack([relation.matvec(np.eye(4)[i]) for i in range(4)])
    assert np.allclose(dense,dense.T,atol=1e-12,rtol=0)
    assert np.array_equal(np.diag(dense),np.ones(4))
    assert np.allclose(dense.sum(1),4,atol=1e-10,rtol=0)
    assert relation.metadata["kernelrowresid"]<=1e-12
    assert relation.sparseK.diagonal().sum()==0 and not relation.metadata["dense_relation_allocated"]
    checks.append("symmetric_sparse_plus_rank1_relation_offdiag_rowsum_Mminus1_diag1")
    # A detour in the Source FG graph must use the mask's geodesic, not RGB/Euclidean distance.
    detour=np.array([[1,0,1],[1,0,1],[1,1,1]],np.uint8);detour_relation=build_relation(detour)
    assert np.isclose(detour_relation.sparseK[0,1],np.exp(-36/32),atol=1e-15,rtol=0)
    checks.append("all_FG_retained_and_four_neighbour_mask_geodesic_not_coordinate_distance")
    features=np.ones((2,2,1),np.float64)
    tied=build_query_tree(features,(2,2))
    assert list(zip(tied.parent.tolist(),tied.edgew.tolist()))==[(-1,0.),(0,1.),(0,1.),(1,1.)]
    assert tied.metadata["nonlocal_edges"]==0 and tied.metadata["tree_edges"]==3
    assert all(child in (node-2,node-1,node+1,node+2) for node,items in enumerate(tied.children) for child in items)
    checks.append("maximum_four_neighbour_tree_exact_weight_index_ties")
    raw=np.array([[[1.,0.],[.6,.8],[0.,1.]]]);tree=build_query_tree(raw,(1,3))
    h=np.array([[.22,.78,.51]]);pi=np.array([[.7,.1,.1,.1],[.1,.6,.2,.1],[.1,.1,.2,.6]])
    actual=tree_sum_product(h,pi,relation,tree,return_state_probabilities=True)
    rho=.1;strength=1.;unary=np.concatenate([.9*h.reshape(-1,1)*pi,.1*h.reshape(-1,1),(1-h).reshape(-1,1)],axis=1)
    enumeration=np.zeros((3,6));partition=0.
    for assignment in itertools.product(range(6),repeat=3):
        mass=float(np.prod([unary[i,state] for i,state in enumerate(assignment)]))
        for child,parent in enumerate(tree.parent):
            if parent>=0 and assignment[child]<4 and assignment[parent]<4:
                mass*=1+strength*tree.edgew[child]*(dense[assignment[child],assignment[parent]]-1)
        partition+=mass
        for i,state in enumerate(assignment):enumeration[i,state]+=mass
    enumeration/=partition
    assembled=np.column_stack([actual["matched_states"],actual["unmatched_foreground"].reshape(-1),actual["background"].reshape(-1)])
    assert np.allclose(assembled,enumeration,atol=3e-12,rtol=0)
    checks.append("exact_two_pass_all_latent_node_marginals_match_full_small_tree_enumeration")
    # Closed-form two-node odds is a check of ONE edge, never the inference algorithm.
    two=build_query_tree(np.ones((1,2,1)),(1,2));h2=np.array([[.3,.7]])
    pi2=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
    result2=tree_sum_product(h2,pi2,relation,two)
    excess=float(pi2[0]@dense@pi2[1]-1);delta=.81*.3*.7*excess
    expected=(h2+delta)/(1+delta)
    assert np.allclose(result2["foreground"],expected,atol=2e-12,rtol=0)
    rescaled=tree_sum_product(h2,pi2,relation,two,rho=0,pair_strength=.81)
    assert np.allclose(result2["foreground"],rescaled["foreground"],atol=2e-12,rtol=0)
    checks.append("two_node_odds_formula_and_noUF_lambda081_pair_strength_control")
    neutral=tree_sum_product(h,pi,relation,tree,pair_strength=0)
    uniform=tree_sum_product(h,np.full((3,4),.25),relation,tree)
    ones=RelationKernel(relation.sparseK,relation.scale,.5,4,relation.coords,True,relation.metadata)
    ones_result=tree_sum_product(h,pi,ones,tree)
    assert np.array_equal(neutral["foreground"],h) and np.array_equal(ones_result["foreground"],h)
    assert np.allclose(uniform["foreground"],h,atol=1e-11,rtol=0)
    checks.append("lambda0_Rones_and_uniform_pi_are_foreground_mass_NoOps")
    for m in (1,2,3):
        small=build_relation(np.ones((1,m),bool))
        small_pi=np.full((3,m),1/m)
        degenerate=tree_sum_product(h,small_pi,small,tree)
        assert small.degenerate and np.array_equal(degenerate["foreground"],h)
        assert np.array_equal(small.matvec(np.arange(m,dtype=float)),np.full(m,np.arange(m).sum()))
    checks.append("Source_M1_M2_M3_exact_R_all_ones_return_h_degeneracy")
    empty=build_relation(np.zeros((2,2),bool));abstain=tree_sum_product(h,np.empty((3,0)),empty,tree)
    assert abstain["metadata"]["abstained"] and np.array_equal(abstain["foreground"],h)
    checks.append("Source_M0_explicit_structure_abstention_retains_every_Query_case")
    permutation=np.array([0,1,3,2])
    permuted=tree_sum_product(h,pi[:,permutation],relation,tree,relation_permutation=permutation,return_state_probabilities=True)
    assert np.allclose(permuted["foreground"],actual["foreground"],atol=2e-12,rtol=0)
    assert np.allclose(permuted["matched_states"],actual["matched_states"][:,permutation],atol=2e-12,rtol=0)
    assert np.allclose(permuted["unmatched_foreground"],actual["unmatched_foreground"],atol=2e-12,rtol=0)
    checks.append("joint_Source_state_and_pi_permutation_invariance_not_relation_only_shuffle")
    lower=.1*h/(.1*h+1-h)
    assert np.all(actual["foreground"]>=lower-1e-12)
    assert np.allclose(assembled.sum(1),1,atol=1e-12,rtol=0) and (assembled>=0).all()
    assert np.allclose(actual["unmatched_foreground"]/(actual["background"]),(.1*h)/(1-h),atol=1e-12,rtol=0)
    checks.append("UF_lower_bound_neutral_UF_to_BG_odds_probability_sum_and_nonnegativity")
    extremes=tree_sum_product(np.array([[0.,1.,0.]]),pi,relation,tree)
    assert np.array_equal(extremes["foreground"],np.array([[0.,1.,0.]]))
    no_uf=tree_sum_product(h,pi,relation,tree,rho=0)
    assert np.array_equal(no_uf["unmatched_foreground"],np.zeros_like(h))
    checks.append("exact_h0_h1_endpoint_mass_and_noUF_preserved_initial_foreground_unary")
    controls=run_controls(h,pi,source,raw,"Source:s/Query:q",relation=relation,tree=tree)
    assert set(controls["predictions"])=={"PCF","sourceRelationShuffled","noUF","noUF_lambda081","binaryPotts","unary"}
    assert np.array_equal(controls["predictions"]["unary"],h)
    seed1,sha1=stable_source_permutation("Source:s/Query:q",4);seed2,sha2=stable_source_permutation("Source:s/Query:q",4)
    assert sha1==sha2 and np.array_equal(seed1,seed2)
    assert np.array_equal(controls["predictions"]["binaryPotts"],binary_potts(h,tree))
    checks.append("all_six_registered_arms_same_inputs_and_stable_legal_identity_relation_shuffle")
    src_feat=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    qry_feat=np.array([[1.,0.],[0.,1.]])
    probability=semantic_correspondence(src_feat,qry_feat,source)
    assert probability.shape==(2,4) and np.allclose(probability.sum(1),1,atol=1e-14,rtol=0)
    explicit=qry_feat@src_feat.T/.07;explicit-=explicit.max(1,keepdims=True);explicit=np.exp(explicit);explicit/=explicit.sum(1,keepdims=True)
    assert np.array_equal(probability,explicit)
    checks.append("caller_unit_semantic_softmax_temperature007_all_Source_FG_states")
    try:run_controls(h,pi,source,raw,"Source:s/Query:q",relation=relation,tree=tree,max_sparse_operations=0)
    except RuntimeError:pass
    else:raise AssertionError("Budget failure did not stop full-state inference")
    checks.append("operation_budget_failure_holds_without_topk_downsampling_or_fake_case_padding")
    assert len(checks)==14
    return dict(state="SERVER_PARTIAL_CORRESPONDENCE_CPU_FIXTURES_PASSED",checks=len(checks),passed=checks,
        fixture_seconds=time.monotonic()-started,fixture_scope="Tiny artificial features/trees and exact enumeration; no pretrained model or real QueryGT",
        source_relation=relation.metadata,shape_operation_estimate=shape_estimate(),
        no_model=True,no_GPU=True,no_training=True,real_task_gain_verified=False,
        full_grid_runtime_profile_completed=False,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (ROOT/"tics/partial_correspondence.py",Path(__file__))})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();require_server()
    if args.out.exists():raise ValueError("Preserve previous CPU evidence; choose a fresh output path")
    result=run();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(state=result["state"],checks=result["checks"],full_grid_runtime_profile_completed=False)))


if __name__=="__main__":main()
