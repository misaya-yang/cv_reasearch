"""Independent min/max closure, color examples and synthetic 128-grid CPU timing."""
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ics.methods.color_bottleneck import adjacency,minimum_bottleneck,predict,resize_field


def closure(n,left,right,cost):
    distance=np.full((n,n),np.inf)
    np.fill_diagonal(distance,0)
    distance[left,right]=cost;distance[right,left]=cost
    for middle in range(n):
        distance=np.minimum(distance,np.maximum(distance[:,middle,None],distance[None,middle,:]))
    return distance


def main():
    rng=np.random.RandomState(20261008)
    left,right=np.triu_indices(7,1)
    max_error=0.0
    for _ in range(24):
        cost=rng.uniform(0,1,size=len(left))
        seeds=rng.uniform(size=7)>.5
        if not seeds.any():seeds[0]=True
        expected=closure(7,left,right,cost)[seeds].min(0)
        obtained=minimum_bottleneck(adjacency(7,left,right,cost),seeds)
        max_error=max(max_error,float(np.max(np.abs(obtained-expected))))
        np.testing.assert_array_equal(obtained,expected)
    # Long path of small local contrasts beats a shorter high-contrast edge.
    route=minimum_bottleneck(adjacency(3,np.array([0,1,0]),np.array([1,2,2]),np.array([.2,.2,.3])),
                             np.array([True,False,False]))
    np.testing.assert_array_equal(route,[0,.2,.2])
    # Uniform-color regions separated by a visible color edge.
    rgb=np.zeros((3,6,3));rgb[:,3:]=1
    base=np.full((3,6),.45);base[:,3:]=.55
    base[0,0]=.9;base[0,3]=.1
    example=predict(rgb,base)
    assert (example['field'][:,:3]>.5).all()
    assert (example['field'][:,3:]<.5).all()
    complementary=predict(rgb,1-base)
    np.testing.assert_allclose(complementary['field'],1-example['field'],atol=1e-14)
    # With flat RGB there is no photometric evidence; exact zero-barrier ties preserve base.
    flat=predict(np.zeros_like(rgb),base)
    np.testing.assert_array_equal(flat['field'],base)
    assert flat['info']['zero_barrier_ties']==base.size
    constant=predict(rgb,np.zeros_like(base))
    np.testing.assert_array_equal(constant['field'],np.zeros_like(base))
    uncertain=np.clip(base,.3,.7)
    no_seeds=predict(rgb,uncertain)
    np.testing.assert_array_equal(no_seeds['field'],uncertain)
    assert no_seeds['info']['missing_seed_abstention']
    # A gradual color ramp has equal positive bottlenecks to both ends in its interior.
    ramp=np.repeat(np.linspace(0,1,9)[None,:,None],3,axis=2)
    ramp_base=np.array([[.9,.6,.6,.6,.4,.4,.4,.4,.1]])
    gradient=predict(ramp,ramp_base)
    np.testing.assert_allclose(gradient['color_field'][0,1:-1],.5,atol=1e-14)
    # Float and uint8 RGB represent identical inputs after prescribed normalization.
    byte_rgb=rng.randint(0,256,size=(7,9,3)).astype(np.uint8)
    small_base=rng.uniform(size=(7,9))
    byte=predict(byte_rgb,small_base)
    unit=predict(byte_rgb.astype(float)/255,small_base)
    np.testing.assert_array_equal(byte['field'],unit['field'])
    # The NumPy half-pixel interpolator is checked against the established torch convention.
    import torch
    import torch.nn.functional as F
    source=rng.normal(size=(5,7))
    target=F.interpolate(torch.from_numpy(source)[None,None],(11,13),mode='bilinear',align_corners=False)[0,0].numpy()
    resize_error=float(np.max(np.abs(resize_field(source,(11,13))-target)))
    assert resize_error<1e-12
    synthetic=predict(rng.randint(0,256,size=(128,128,3)).astype(np.uint8),rng.uniform(size=(64,64)))
    report=dict(kind='mathematical_and_synthetic_only',real_episodes=0,
                independent_minmax_closure_cases=24,maximum_path_error=max_error,
                longer_low_barrier_path=route.tolist(),
                two_color_example=dict(info=example['info'],field=example['field'].tolist(),empirical_segmentation=False),
                complement_equivalence_checked=True,constant_rgb_preserves_base=True,
                constant_empty_base_preserved=True,missing_confident_seeds_preserves_base=True,
                rgb_ramp_ambiguity_checked=True,
                float_and_uint8_equivalence_checked=True,maximum_resize_error=resize_error,
                synthetic_128_grid=synthetic['info'],real_gain='unmeasured',real_runtime='unmeasured',
                no_gpu=True,no_server=True)
    Path(__file__).with_name('path_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(maximum_path_error=max_error,maximum_resize_error=resize_error,
                          synthetic_seconds=synthetic['info']['wall_seconds'],real_episodes=0)))


if __name__=='__main__':main()
