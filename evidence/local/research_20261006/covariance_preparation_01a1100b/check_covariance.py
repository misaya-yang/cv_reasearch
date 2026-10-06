"""Response-covariance discrimination and exact descriptor properties; synthetic only."""
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from ics.methods.reference_covariance import Config,covariance_debts,covariances,predict,response_features
from ics.methods.reference_occupancy import Config as TransportConfig,occupancy_debt


def covariance(values):
    return covariances(response_features(values).sum(0,keepdims=True),values.shape[1])[0]


def main():
    rng=np.random.RandomState(20261011)
    rows=rng.uniform(-.5,.8,size=(31,5))
    matrix=covariance(rows)
    np.testing.assert_allclose(matrix,np.cov(rows,rowvar=False,ddof=0),atol=1e-14)
    np.testing.assert_allclose(matrix,covariance(rows+np.array([.2,-.3,.1,.6,-.2])),atol=1e-14)
    np.testing.assert_allclose(matrix,covariance(np.repeat(rows,3,axis=0)),atol=1e-14)
    np.testing.assert_allclose(matrix,covariance(rows[rng.permutation(len(rows))]),atol=1e-14)
    # Every wrong point matches a reference mode more closely, but part variation is wrong.
    ref=np.tile(np.eye(2),(4,1))
    true=np.tile(np.array([[.8,0],[0,.8]]),(4,1))
    mixed=np.sqrt(1-.9**2)
    wrong=np.tile(np.array([[.9,mixed],[mixed,.9]]),(4,1))
    reference=covariance(ref)
    bank=dict(k=2,matrices=reference[None],norm2=np.array([np.sum(reference**2)]),
              trace=np.array([np.trace(reference)]),mode='full')
    debts=covariance_debts(np.stack([covariance(true),covariance(wrong)]),np.ones(2,dtype=bool),bank)
    assert wrong.max(1).mean()>true.max(1).mean() and debts[0]<debts[1]
    centered_wrong=wrong-wrong.mean(0)+true.mean(0)
    np.testing.assert_allclose(centered_wrong.mean(0),true.mean(0),atol=1e-14)
    np.testing.assert_array_equal(np.bincount(centered_wrong.argmax(1),minlength=2),
                                  np.bincount(true.argmax(1),minlength=2))
    np.testing.assert_allclose(covariance(centered_wrong),covariance(wrong),atol=1e-14)
    # Equal occupancy of both reference modes does not establish their response covariance.
    transport=[]
    for value in (true[:2],wrong[:2]):
        debt,_=occupancy_debt(np.array([[.5,.5]]),np.array([.5,.5]),1-value,TransportConfig())
        transport.append(float(debt[0]))
    assert max(abs(value) for value in transport)<1e-12
    # Equal total variance is insufficient when covariance directions differ.
    anisotropic=np.diag([.04,.01]);swapped=np.diag([.01,.04])
    second_bank=dict(k=2,matrices=anisotropic[None],norm2=np.array([np.sum(anisotropic**2)]),
                     trace=np.array([np.trace(anisotropic)]),mode='full')
    full=float(covariance_debts(swapped[None],np.array([True]),second_bank)[0])
    scalar=float(covariance_debts(swapped[None],np.array([True]),dict(second_bank,mode='trace'))[0])
    assert full>.5 and scalar==0
    # Fixed-affinity-bank statistic witness: same means, occupied modes, total
    # variance and distribution of best/second-best evidence, different covariance.
    common=np.array([.4,.4]);a=np.sqrt(.08);b=np.sqrt(.02)
    first=common+np.array([[a,0],[-a,0],[0,b],[0,-b]])
    second=common+np.array([[b,0],[-b,0],[0,a],[0,-a]])
    np.testing.assert_allclose(covariance(first),anisotropic,atol=1e-14)
    np.testing.assert_allclose(covariance(second),swapped,atol=1e-14)
    np.testing.assert_allclose(first.mean(0),second.mean(0),atol=1e-14)
    np.testing.assert_array_equal(np.bincount(first.argmax(1),minlength=2),np.bincount(second.argmax(1),minlength=2))
    np.testing.assert_allclose(np.sort(first.max(1)),np.sort(second.max(1)),atol=1e-14)
    np.testing.assert_allclose(np.sort(np.abs(first[:,0]-first[:,1])),np.sort(np.abs(second[:,0]-second[:,1])),atol=1e-14)
    perm=np.array([1,0])
    perm_bank=dict(second_bank,matrices=anisotropic[perm][:,perm][None])
    permutation=float(covariance_debts(swapped[perm][:,perm][None],np.array([True]),perm_bank)[0])
    np.testing.assert_allclose(full,permutation,atol=1e-14)
    # Complete controlled scene: same shapes/part locations, different part-response covariance.
    shape=(32,32);n=np.prod(shape)
    q=np.zeros((n,4));r=np.zeros_like(q);q[:,3]=1;r[:,3]=1
    cov=np.zeros(shape);cov[2:10,2:10]=1
    for row in range(2,10):
        for col in range(2,10):r[row*32+col]=np.eye(4)[0 if col<6 else 1]
    base=np.full(shape,.1);base[12:20,2:10]=.49;base[12:20,2:6]=.8;base[12:20,20:28]=.87
    for row in range(12,20):
        for col in range(2,10):
            value=np.array([.8,0,.6,0]) if col<6 else np.array([0,.8,.6,0])
            q[row*32+col]=value
        for col in range(20,28):
            q[row*32+col]=np.array([.9,mixed,0,0]) if col<24 else np.array([mixed,.9,0,0])
    result=predict(q,r,cov,base)
    assert result['token_mask'][12:20,2:10].all()
    assert not result['token_mask'][12:20,20:28].any()
    # This first contrast alone also favors a simple mode-selectivity margin;
    # it cannot be offered as unique evidence for the complex covariance model.
    assert (result['mode_margin_field'][12:20,2:10]>.5).all()
    assert (result['mode_margin_field'][12:20,20:28]<.5).all()
    for item in result['info']['search']['moves']:assert item['after']>=item['before']-1e-10
    zero=predict(q,r,cov,base,Config(strength=0))
    np.testing.assert_array_equal(zero['token_mask'],base>.5)
    # Two individually homogeneous reference instances cannot teach within-object variation.
    separate_cov=np.zeros(shape);separate_cov[2:6,2:6]=1;separate_cov[8:12,8:12]=1
    separate=r.copy();separate[:]=np.eye(4)[3]
    separate.reshape(32,32,4)[2:6,2:6]=np.eye(4)[0]
    separate.reshape(32,32,4)[8:12,8:12]=np.eye(4)[1]
    inactive=predict(q,separate,separate_cov,base)
    assert inactive['info']['abstention'] and inactive['info']['reference_templates']==0
    # Full cached-feature dimensions; timing includes the scalar-variance control search.
    big_q=rng.normal(size=(4096,1024)).astype(np.float32)
    big_r=np.zeros_like(big_q);big_r[:,8]=1
    big_cov=np.zeros((64,64));big_cov[8:56,8:56]=1
    for row in range(8,56):
        for col in range(8,56):
            mode=((row-8)//24)*4+(col-8)//12
            i=row*64+col;big_r[i]=0;big_r[i,mode]=1
    yy,xx=np.indices((64,64));big_base=.15+.6*np.exp(-((xx-32)**2+(yy-30)**2)/180)
    benchmark=predict(big_q,big_r,big_cov,big_base)
    report=dict(kind='analytical_and_synthetic_only',real_episodes=0,
                covariance_matches_numpy=True,common_affinity_offset_invariant=True,
                row_permutation_and_duplication_invariant=True,mode_index_permutation_invariant=True,
                pixel_similarity_order=dict(true=float(true.max(1).mean()),wrong=float(wrong.max(1).mean())),
                covariance_debts=dict(true=float(debts[0]),wrong=float(debts[1])),
                balanced_occupancy_debts=transport,equal_trace_witness=dict(full=full,trace_only=scalar),
                fixed_affinity_statistic_same_mean_occupancy_trace_max_margin_checked=True,
                controlled_scene=result['info'],zero_strength_exact=True,
                first_contrast_also_solved_by_mode_margin_control=True,
                homogeneous_reference_instances_abstain=True,synthetic_full_dimensions=benchmark['info'],
                real_gain='unmeasured',real_runtime='unmeasured',no_gpu=True,no_server=True)
    Path(__file__).with_name('covariance_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(true_debt=float(debts[0]),wrong_debt=float(debts[1]),
                          occupancy_debts=transport,equal_trace_witness=report['equal_trace_witness'],
                          synthetic_seconds=benchmark['info']['wall_seconds'],real_episodes=0)))


if __name__=='__main__':main()
