"""Geometric discrimination witness, inverse-mask checks and synthetic CPU cost."""
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ics.methods.reference_constellation import Config,aligned_mask,evaluate,predict,propose,sample,select_prior


def main():
    cfg=Config()
    anchors=np.array([1+1j,5+1j,1+5j,5+5j])
    coherent=anchors+8+8j
    distorted=np.array([1+1j,5+1j,2+5j,7+4j])
    bank=np.zeros((4,16,16))
    for k in range(4):
        for location in (coherent[k],distorted[k]):
            bank[k,int(location.imag),int(location.real)]=3
    correct=evaluate(anchors,bank,[(1+0j,8+8j,False,0,1)],cfg)
    incorrect=evaluate(anchors,bank,[(1+0j,0+0j,False,0,1)],cfg)
    assert len(correct)==1 and len(incorrect)==0
    # Per-landmark maximum evidence is identical in both spatial arrangements.
    np.testing.assert_array_equal(bank[:,coherent.imag.astype(int),coherent.real.astype(int)].diagonal(),
                                  bank[:,distorted.imag.astype(int),distorted.real.astype(int)].diagonal())
    # A transform fitted to two strong peaks does not pass without held-out support.
    missing=bank.copy()
    missing[2,int(coherent[2].imag),int(coherent[2].real)]=0
    missing[3,int(coherent[3].imag),int(coherent[3].real)]=0
    assert not evaluate(anchors,missing,[(1+0j,8+8j,False,0,1)],cfg)
    proposals=propose(anchors,[[coherent[k],distorted[k]] for k in range(4)],cfg)
    scored=evaluate(anchors,bank,proposals,cfg)
    assert scored
    coverage=np.zeros((16,16));coverage[1:6,1:6]=1
    prior,selected=select_prior(coverage,(16,16),scored,cfg)
    assert len(selected)==1
    translated=np.zeros((16,16));translated[9:14,9:14]=1
    np.testing.assert_array_equal(prior,translated)
    duplicated=scored+scored
    duplicate_prior,duplicate_selection=select_prior(coverage,(16,16),duplicated,cfg)
    np.testing.assert_array_equal(duplicate_prior,prior)
    assert len(duplicate_selection)==len(selected)
    # Ninety-degree rotation: every inverse sample has integer coordinates.
    small=np.arange(64).reshape(8,8)/63
    pose=dict(a=1j,b=7+0j,mirror=False)
    rotated=aligned_mask(small,(8,8),pose)
    np.testing.assert_allclose(rotated,np.rot90(small,k=-1),atol=1e-14)
    yy,xx=np.indices((8,9));linear=2*xx+3*yy+1
    points=np.array([1.3+2.7j,4.1+5.4j])
    np.testing.assert_allclose(sample(linear,points),2*points.real+3*points.imag+1,atol=1e-14)
    # Sparse raw-feature fixture exercises foreground modes through final field.
    q=np.zeros((4096,16));r=np.zeros_like(q);q[:,4]=1;r[:,4]=1
    cov=np.zeros((64,64));base=np.full((64,64),.2)
    raw_anchors=np.array([10+10j,20+10j,10+20j,20+20j])
    for k,point in enumerate(raw_anchors):
        ref=int(point.imag)*64+int(point.real)
        query=point+28+28j;qry=int(query.imag)*64+int(query.real)
        r[ref]=0;r[ref,k]=1;cov[int(point.imag),int(point.real)]=1
        q[qry]=0;q[qry,k]=1
    raw=predict(q,r,cov,base)
    assert not raw['info']['abstention'] and raw['info']['retained_poses']==1
    assert int((raw['field']>.5).sum())==4
    # Single reference mode cannot validate a multi-part constellation.
    one=r.copy();one[:]=0;one[:,0]=1
    absent=predict(q,one,cov,base)
    np.testing.assert_array_equal(absent['field'],base)
    assert absent['info']['abstention']
    # Common equally strong responses to every part are not a distinct part arrangement.
    ambiguous=np.zeros_like(q);ambiguous[:,:4]=1
    common=predict(ambiguous,r,cov,base)
    np.testing.assert_array_equal(common['field'],base)
    assert common['info']['abstention']
    # Full feature dimensions and spatially separated eight reference modes.
    rng=np.random.RandomState(20261009)
    q=rng.normal(size=(4096,1024)).astype(np.float32)
    r=np.zeros_like(q);r[:,8]=1
    cov=np.zeros((64,64));cov[8:56,8:56]=1
    for row in range(8,56):
        for col in range(8,56):
            mode=((row-8)//24)*4+(col-8)//12
            idx=row*64+col;r[idx]=0;r[idx,mode]=1
    benchmark=predict(q,r,cov,rng.uniform(size=(64,64)))
    report=dict(kind='mathematical_and_synthetic_only',real_episodes=0,
                identical_landmark_similarity_geometry_witness=dict(correct_accepts=True,distorted_accepts=False),
                held_out_landmarks_required=True,duplicate_pose_neutral=True,
                rotation_and_bilinear_checked=True,raw_feature_fixture=raw['info'],
                fewer_than_four_modes_abstains=True,equal_all_mode_responses_abstain=True,
                synthetic_full_dimensions=benchmark['info'],
                real_gain='unmeasured',real_runtime='unmeasured',no_gpu=True,no_server=True)
    Path(__file__).with_name('geometry_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(geometric_witness=report['identical_landmark_similarity_geometry_witness'],
                          sparse_fixture_retained=raw['info']['retained_poses'],
                          synthetic_seconds=benchmark['info']['wall_seconds'],
                          proposed_poses=benchmark['info']['proposed_poses'],real_episodes=0)))


if __name__=='__main__':main()
