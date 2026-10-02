"""Run the user-provided extension against independent numerical references."""
import sys,json
from pathlib import Path
from pose_eval import update
sys.path.insert(0,str(Path(__file__).parent/'sources'))
import numpy as np
from scipy.spatial.transform import Rotation
from provided_extension.gic import core as c,pose as p
from gic import grid_basis,two_end_field,profiled_reml

rng=np.random.default_rng(20261003);n=180
xyA=rng.uniform(0,1,(n,2));xyB=rng.uniform(0,1,(n,2))
bA=grid_basis(xyA);bB=grid_basis(xyB)
q=rng.normal(size=(n,2,2));sA=q@q.transpose(0,2,1)+.5*np.eye(2)
q=rng.normal(size=(n,2,2));sB=q@q.transpose(0,2,1)+.5*np.eye(2)
ga=rng.normal(size=(n,2));gb=rng.normal(size=(n,2))
d,f=two_end_field(xyA,xyB,sA,sB,ga,gb)
dd,ff=c.two_endpoint_factors(sA,sB,ga,gb,bA,bB)
checks={'factors_maxdiff':float(max(np.max(abs(d-dd)),np.max(abs(f-ff))))}
assert checks['factors_maxdiff']<1e-12
J=rng.normal(size=(n,5));r=rng.normal(size=n)
spec=c.prepare_normal_reml(J,r,d,f)
offset=profiled_reml(r,J,d,np.zeros_like(f))-spec.objective(0)
checks['spectral_direct_maxdiff']=max(abs(profiled_reml(r,J,(1-z)*d,np.sqrt(z)*f)-spec.objective(z)-offset) for z in np.linspace(0,.95,40))
assert checks['spectral_direct_maxdiff']<1e-9
for z in [0,.2,.95]:
 S=c.two_endpoint_covariance(sA,sB,ga,gb,bA,bB,z)
 checks['marginals_'+str(z)]=float(np.max(abs(np.diag(S.dense())-d)))
 assert checks['marginals_'+str(z)]<1e-11
 checks['woodbury_'+str(z)]=float(np.max(abs(S.solve(r)-np.linalg.solve(S.dense(),r))))
 assert checks['woodbury_'+str(z)]<1e-9
for label,F in [('zero',np.zeros_like(f)),('tangent',J@rng.normal(size=(5,100)))]:
 rho,meta=c.estimate_rho_from_factors(J,r,np.ones(n),F)
 checks['fallback_'+label]=dict(rho=rho,reason=meta['reason']);assert rho==0
K1=np.array([[910,0,620],[0,1030,470],[0,0,1.]])
K2=np.array([[1210,0,760],[0,870,570],[0,0,1.]])
a=rng.uniform([0,0],[1240,940],(n,2));b=rng.uniform([0,0],[1520,1140],(n,2))
R=Rotation.from_rotvec([.12,-.07,.01]).as_matrix();t=np.array([.2,.3,.8]);t/=np.linalg.norm(t)
e,ga,gb,j=p.linearize_dual_endpoint(R,t,a,b,K1,K2)
eps=1e-6
cols=[]
for i in range(5):
 delta=np.zeros(5);delta[i]=eps;rp,tp=p.retract(R,t,delta);rm,tm=p.retract(R,t,-delta)
 cols.append((p.epipolar_algebraic_residual(rp,tp,a,b,K1,K2)[0]-p.epipolar_algebraic_residual(rm,tm,a,b,K1,K2)[0])/(2*eps))
checks['analytic_pose_jacobian']=float(np.max(abs(np.column_stack(cols)-j)));assert checks['analytic_pose_jacobian']<1e-8
for side,analytic in [(0,ga),(1,gb)]:
 for axis in range(2):
  ap=[a.copy(),b.copy()];am=[a.copy(),b.copy()];ap[side][:,axis]+=.001;am[side][:,axis]-=.001
  numeric=(p.epipolar_algebraic_residual(R,t,*ap,K1,K2)[0]-p.epipolar_algebraic_residual(R,t,*am,K1,K2)[0])/.002
  v=float(np.max(abs(numeric-analytic[:,axis])));checks[f'endpoint_{side}_{axis}']=v;assert v<1e-9
report=dict(state='PASSED',checks=checks,scope='User supplied static-only code now runtime/import/independent numeric checked; no real quality claim')
Path('results/extension_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
