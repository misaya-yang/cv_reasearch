"""Check two-end marginals and spectral REML against independent dense forms."""
import json,time
from pathlib import Path
import numpy as np
from gic import two_end_field,profiled_reml,SpectralREML


def run():
    rng=np.random.default_rng(2033);n=180;p=5
    xy=rng.random((2,n,2));grad=rng.normal(size=(2,n,2))
    a=rng.normal(size=(2,n,2,2));cov=a@a.swapaxes(-1,-2)+np.eye(2)[None,None]
    var,b=two_end_field(xy[0],xy[1],cov[0],cov[1],grad[0],grad[1])
    checks={}
    def ck(name,error,tol=1e-8):
        checks[name]={'max_error':float(error),'passed':float(error)<tol,'tolerance':tol}
        assert float(error)<tol,(name,error)
    for rho in [0.,.1,.6,.95]:
        s=np.diag((1-rho)*var)+rho*b@b.T
        ck(f'both_endpoint_marginal_rho{rho}',np.max(np.abs(np.diag(s)-var)))
    j=rng.normal(size=(n,p));r=rng.normal(size=n)
    t=time.perf_counter();spec=SpectralREML(r,j,var,b);construction=time.perf_counter()-t
    f0=profiled_reml(r,j,var,np.zeros_like(b));g0=spec.objective(0)
    errors=[];fullstart=time.perf_counter()
    for rho in np.linspace(0,.95,40):
        f=profiled_reml(r,j,(1-rho)*var,np.sqrt(rho)*b)
        errors.append(abs((f-f0)-(spec.objective(rho)-g0)))
    fulltime=time.perf_counter()-fullstart
    ck('spectral_vs_full_reml_differences',max(errors))
    t=time.perf_counter()
    for rho in np.linspace(0,.95,40):spec.objective(rho)
    scalar=time.perf_counter()-t
    spec2=SpectralREML(r+j@rng.normal(size=p)*30,j,var,b)
    ck('spectral_geometry_shift_invariance',max(abs(spec.objective(rho)-spec2.objective(rho)) for rho in [0,.4,.95]))
    rho,gate=spec.select()
    result={'state':'NUMERICAL_CHECKS_PASSED','checks':checks,'checks_count':len(checks),
            'field_rank_bound':b.shape[1],'eigen_build_seconds':construction,
            'full_40_evaluations_seconds':fulltime,'scalar_40_evaluations_seconds':scalar,
            'rho_sample':rho,'gate':gate,'scope':'algebra and implementation only; CPU timings at n180 do not predict benchmark runtime; not real pose AUC'}
    pth=Path(__file__).parent/'results/two_end_checks.json';pth.parent.mkdir(exist_ok=True);pth.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
