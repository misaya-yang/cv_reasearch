"""Shape invariance and physical-component counterexamples; synthetic CPU only."""
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ics.methods.reference_shape import Config,coordinate_features,evaluate,predict,raw_features,reference_bank,signatures


def main():
    rng=np.random.RandomState(20261010)
    points=rng.normal(size=37)+1j*rng.normal(size=37)
    original,_=signatures(raw_features(points).sum(0,keepdims=True),8)
    maximum_error=0.0
    for scale,angle,translation in ((2.3,.77,3+4j),(.35,-1.1,-2+.5j),(1,0,10-3j)):
        transformed,_=signatures(raw_features(scale*np.exp(1j*angle)*points+translation).sum(0,keepdims=True),8)
        error=float(np.max(np.abs(original-transformed)));maximum_error=max(maximum_error,error)
        np.testing.assert_allclose(original,transformed,atol=1e-10)
    mirror,_=signatures(raw_features(np.conj(points)).sum(0,keepdims=True),8)
    np.testing.assert_allclose(original,mirror,atol=1e-12)
    # Reference square and two query components: a complete-able square and an elongated distractor.
    shape=(32,32)
    cov=np.zeros(shape);cov[2:10,2:10]=1
    base=np.full(shape,.1);base[12:20,12:20]=.7;base[12:20,19]=.49
    base[27:29,:]=.9
    q=np.zeros((base.size,4));q[:,2]=1
    true=np.zeros(shape,dtype=bool);true[12:20,12:20]=True
    wrong=np.zeros(shape,dtype=bool);wrong[27:29,:]=True
    q[true.ravel()]=np.array([1,0,0,0]);q[wrong.ravel()]=np.array([0,1,0,0])
    cfg=Config()
    features=coordinate_features(shape);bank=reference_bank(cov,features,cfg)
    unary=(base-.5).ravel()
    _,_,wrong_energy,wrong_debt=evaluate(wrong,unary,features,bank,cfg)
    broken=wrong.copy();broken[27,0]=False
    _,_,broken_energy,broken_debt=evaluate(broken,unary,features,bank,cfg)
    assert wrong_energy[1]<0
    assert broken_debt[1]>.45 and broken_energy[1]<0
    # Removing one tree-root leaf cannot erase the shape cost of the actual connected object.
    assert abs(broken_debt[1]-wrong_debt[1])<.05
    result=predict(q,cov,base,reference_hw=shape,query_hw=shape)
    assert (result['token_mask'][12:20,12:20]).all()
    assert not result['token_mask'][27:29].any()
    history=result['info']['search']['moves']
    assert history and all(item['after']>=item['before']-1e-10 for item in history)
    assert result['info']['search']['final_energy']>=result['info']['search']['initial_energy']
    zero=predict(q,cov,base,Config(strength=0),reference_hw=shape,query_hw=shape)
    np.testing.assert_array_equal(zero['token_mask'],base>.5)
    missing=np.minimum(base,.49)
    none=predict(q,cov,missing,reference_hw=shape,query_hw=shape)
    assert not none['token_mask'].any()
    unknown=predict(q,cov,base)
    np.testing.assert_array_equal(unknown['token_mask'],base>.5)
    assert unknown['info']['original_geometry_missing']
    # Undoing canonical square resizing recovers the same physical filled square.
    square=np.zeros((64,64));square[16:32,16:32]=1
    compressed=np.zeros_like(square);compressed[16:32,16:24]=1
    one=reference_bank(square,coordinate_features(square.shape,(64,64)),cfg)
    restored=reference_bank(compressed,coordinate_features(compressed.shape,(64,128)),cfg)
    aspect_error=float(np.max(np.abs(one-restored)))
    np.testing.assert_allclose(one,restored,atol=1e-12)
    # Realized shape invariants are cheap; timing includes generic-control search.
    big_q=rng.normal(size=(4096,1024)).astype(np.float32)
    big_cov=np.zeros((64,64));big_cov[18:46,18:46]=1
    yy,xx=np.indices((64,64))
    big_base=.15+.6*np.exp(-((xx-33)**2+(yy-30)**2)/180)
    benchmark=predict(big_q,big_cov,big_base,reference_hw=(64,64),query_hw=(64,64))
    report=dict(kind='mathematical_and_synthetic_only',real_episodes=0,
                maximum_similarity_invariance_error=maximum_error,reflection_invariance_checked=True,
                wrong_component_debt=float(wrong_debt[1]),one_pixel_removed_debt=float(broken_debt[1]),
                connected_component_cost_not_evaded_by_tree_partition=True,
                illustrative_completion_and_distractor_deletion=True,
                illustrative_search=result['info']['search'],zero_strength_exact_control=True,
                all_subthreshold_cue_cannot_hallucinate=True,missing_original_geometry_abstains=True,
                maximum_aspect_restoration_error=aspect_error,
                synthetic_full_dimensions=benchmark['info'],real_gain='unmeasured',real_runtime='unmeasured',
                no_gpu=True,no_server=True)
    Path(__file__).with_name('shape_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(maximum_invariance_error=maximum_error,
                          wrong_component_debt=float(wrong_debt[1]),
                          one_pixel_removed_debt=float(broken_debt[1]),
                          synthetic_seconds=benchmark['info']['wall_seconds'],real_episodes=0)))


if __name__=='__main__':main()
