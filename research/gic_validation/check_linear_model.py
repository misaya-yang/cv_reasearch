"""Bounded independent checks. Output is numerical verification, not pose AUC."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from gic import (components,symmetric_root,inverse_product,logdet,gls_step,
                 innovation_step,profiled_reml,select_rho,covariance_from_precision)


def run(out):
    rng=np.random.default_rng(20261002);checks={};start=time.monotonic()
    n=96;p=5;xy=rng.random((n,2));a=rng.normal(size=(n,2,2))
    cov=a@a.swapaxes(-1,-2)+np.eye(2)[None]*.7
    normals=rng.normal(size=(n,2));normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    d,u,v=components(xy,cov,normals,.6,grid=5);s=np.diag(d)+u@u.T
    j=rng.normal(size=(n,p));r=rng.normal(size=n);x=rng.normal(size=(n,7))
    def check(name,error,tol=1e-9):
        value=float(error);checks[name]={'max_error':value,'tolerance':tol,'passed':value<tol}
        assert value<tol,(name,value,tol)
    check('fixed_point_marginal',np.max(np.abs(np.diag(s)-v)))
    check('woodbury_vs_dense',np.max(np.abs(inverse_product(d,u,x)-np.linalg.solve(s,x))))
    check('logdet_vs_dense',abs(logdet(d,u)-np.log(np.linalg.eigvalsh(s)).sum()))
    direct=-np.linalg.solve(j.T@np.linalg.solve(s,j),j.T@np.linalg.solve(s,r))
    check('gls_vs_dense',np.max(np.abs(gls_step(r,j,d,u)-direct)))
    check('innovation_vs_dense',np.max(np.abs(innovation_step(r,j,d,u)-direct)))
    shift=j@rng.normal(size=p)*20
    check('reml_geometry_shift_invariance',abs(profiled_reml(r,j,d,u)-profiled_reml(r+shift,j,d,u)),1e-8)
    d0,u0,v0=components(xy,cov,normals,0,grid=5)
    expected=-np.linalg.solve(j.T@(j/v0[:,None]),j.T@(r/v0))
    check('rho_zero_diagonal_fallback',np.max(np.abs(gls_step(r,j,d0,u0)-expected)))
    angle=.71;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    rotated=rot[None]@cov@rot.T[None]
    dr,ur,_=components(xy,rotated,normals@rot.T,.6,grid=5)
    check('target_coordinate_rotation',np.max(np.abs(np.diag(dr)+ur@ur.T-s)))
    # Both negative controls have CONSTANT row marginals: comparison to diag(S)
    # is then fair. Arbitrary U=J*A with changing row norms would confound this.
    t=2*np.pi*np.arange(128)/128
    h=np.column_stack([np.ones(128),np.cos(t),np.sin(t)])
    residual=rng.normal(size=128);dt=np.ones(128)
    for name,uu in [('pure_tangent',.8*h),('pure_normal',.8*np.column_stack([np.cos(2*t),np.sin(2*t)]))]:
        marginal=dt+(uu*uu).sum(1)
        diag=-np.linalg.solve(h.T@(h/marginal[:,None]),h.T@(residual/marginal))
        check(name+'_unchanged_point_estimate',np.max(np.abs(gls_step(residual,h,dt,uu)-diag)))
    c=np.array([[10.,9.],[9.,10.]])
    estimator=np.linalg.solve(np.array([[1.,0.]])@np.linalg.inv(c)@np.array([[1.],[0.]]),np.array([[1.,0.]])@np.linalg.inv(c))
    check('analytic_estimator',np.max(np.abs(estimator-np.array([[1.,-.9]]))))
    check('analytic_variance',abs(float((estimator@c@estimator.T)[0,0])-1.9))
    c1=np.array([[[2.,.4],[.4,1.]]]);precision=np.linalg.inv(c1)
    cout=covariance_from_precision(precision,800,600,1600,900)
    scale=np.diag([2.,1.5]);check('upstream_precision_coordinate_scaling',np.max(np.abs(cout-scale[None]@c1@scale.T[None])))
    # A coupled Gaussian counterexample checks the sign/direction against an
    # exact finite-dimensional expectation; it is explicitly not a benchmark.
    hj=np.array([[1.],[0.]]);diag_est=np.array([[1.,0.]])
    improvement=diag_est@c@diag_est.T-estimator@c@estimator.T
    check('positive_semidefinite_correction',max(0.,-np.linalg.eigvalsh(improvement).min()))
    # Independent full block conditioning verifies the report's covariance
    # reduction, not merely its two-variable illustrative counterexample.
    w=1/np.sqrt(v);qq,rr=np.linalg.qr(w[:,None]*j,mode='complete')
    qt,qn=qq[:,:p],qq[:,p:];rr=rr[:p]
    cw=w[:,None]*s*w[None,:]
    ctt=qt.T@cw@qt;ctn=qt.T@cw@qn;cnn=qn.T@cw@qn
    rinv=np.linalg.inv(rr)
    conditional=rinv@(ctt-ctn@np.linalg.solve(cnn,ctn.T))@rinv.T
    glscov=np.linalg.inv(j.T@np.linalg.solve(s,j))
    check('conditional_covariance_vs_gls',np.max(np.abs(conditional-glscov)))
    reduction=rinv@ctn@np.linalg.solve(cnn,ctn.T)@rinv.T
    check('full_covariance_reduction_psd',max(0.,-np.linalg.eigvalsh(reduction).min()))
    # The REML optimizer is exercised with an in-model spatial covariance,
    # without GT or model/parameter selection from observed pose quality.
    j2=np.column_stack([np.ones(n),xy[:,0],xy[:,1]])
    eps=np.sqrt(d)*rng.normal(size=n)+u@rng.normal(size=u.shape[1])
    rho,selection=select_rho(eps+j2@np.array([2.,-1.,.4]),j2,xy,cov,normals)
    result={'state':'NUMERICAL_CHECKS_PASSED','checks':checks,'seed':20261002,
            'checks_count':len(checks),'reml_single_generated_case':{'selected_rho':rho,**selection},
            'analytic_example':{'diagonal_variance':10.,'gls_variance':1.9},
            'elapsed_seconds':time.monotonic()-start,
            'real_matcher_evaluation':'NOT_RUN_NO_EXISTING_CACHE',
            'scope':'independent numerical implementation of report; no real RoMa/pose AUC results or evidence that actual errors follow this covariance model'}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);run(ap.parse_args().out)
