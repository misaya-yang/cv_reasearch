"""Unlabelled candidate/initialization checks; no graph solve or masks."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))

def main(root):
    from ics.methods.joint_reference_bp import build_grid_graph,_initialize
    root=Path(root);source=root/'legal_observation.npz'
    receipt=json.loads((root/'input_receipt.json').read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=receipt['observation_sha256']:raise ValueError('Prepared observation changed')
    started=time.monotonic()
    with np.load(source,allow_pickle=False) as z:
        p=z['b0'].astype(np.float64);valid=z['valid'].astype(bool);roles=z['role'];patch=z['refpatch']
        refxy=z['ref_xy'].astype(np.float64);queryxy=z['query_xy'].astype(np.float64);alpha=z['alpha']
    p=np.where(valid,p,0.);p/=p.sum(1,keepdims=True)
    graph=build_grid_graph(queryxy,(128,128))
    J,info=_initialize(p,refxy,graph,(.125,8.))
    scale=np.sqrt(np.linalg.det(J))
    common=0;pair_mass_sum=0.;seam_common=seam_count=0
    for start in range(0,len(graph.u),256):
        end=min(len(graph.u),start+256);u,v=graph.u[start:end],graph.v[start:end]
        same=(patch[u,:,None]==patch[v,None,:])&(roles[u,:,None]==roles[v,None,:])&valid[u,:,None]&valid[v,None,:]
        overlap=same.any((1,2));common+=int(overlap.sum())
        pair_mass_sum+=float((p[u,:,None]*p[v,None,:]*same).sum())
        # In the fine128 grid,64 is the boundary between native quadrants.
        ur,uc=np.divmod(u,128);vr,vc=np.divmod(v,128)
        seam=((ur<64)!=(vr<64))|((uc<64)!=(vc<64))
        seam_count+=int(seam.sum());seam_common+=int(overlap[seam].sum())
    variance=np.asarray(info['candidate_position_variance'])
    report=dict(state='CANDIDATE_POSITION_PREPARATION_ONLY',episode_id=receipt['episode_id'],
        observation_sha256=digest,solver_source_sha256=hashlib.sha256((REPO/'src/ics/methods/joint_reference_bp.py').read_bytes()).hexdigest(),
        nodes=len(p),edges=len(graph.u),
        min_scale_count=len(info['clipped_to_min_nodes']),min_scale_fraction=len(info['clipped_to_min_nodes'])/len(p),
        zero_or_numerical_covariance_count=len(info['zero_or_numerical_covariance_nodes']),
        candidate_position_variance_median=float(np.median(variance)),
        initial_scale_quantiles=dict(zip(('min','q25','median','q75','max'),np.quantile(scale,[0,.25,.5,.75,1]).tolist())),
        edge_has_common_reference_atom_fraction=common/len(graph.u),
        independent_unary_common_atom_pair_mass_mean=pair_mass_sum/len(graph.u),
        cross_window_edges=seam_count,cross_window_common_atom_fraction=seam_common/seam_count,
        median_position_mass_amplification={str(y):float(np.median(1/alpha[:,y])) for y in (0,1)},
        seconds=time.monotonic()-started,encoder_forward=0,raw_feature_reads=0,query_GT_reads=0,solver_calls=0,new_mask_predictions=0,
        limits='Initialization and feasible candidate overlap only. No posterior, actual collapse diagnosis, correctness, mIoU, new segmentation or cause attribution. Common atoms can be legitimate fine-to-coarse aliases.')
    (root/'candidate_position_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(root/'initial_similarity.npz',J=J)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: joint_reference_candidate_diagnostic.py PREPARATION_ROOT')
    main(sys.argv[1])
