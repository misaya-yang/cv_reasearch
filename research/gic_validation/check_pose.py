"""Independent finite differences for the two-end pixel residual interface."""
import json
from pathlib import Path
from pose_eval import residual,update,jacobian
import numpy as np
from scipy.spatial.transform import Rotation

rng=np.random.default_rng(20261002)
K1=np.array([[950.,0.,620.],[0.,1070.,470.],[0.,0.,1.]])
K2=np.array([[1170.,0.,780.],[0.,880.,580.],[0.,0.,1.]])
R=Rotation.from_rotvec([.11,-.17,.03]).as_matrix()
t=np.array([.3,-.2,.8]);t/=np.linalg.norm(t)
x1=rng.uniform([0,0],[1240,940],(120,2))
x2=rng.uniform([0,0],[1560,1160],(120,2))
r,ga,gb=residual(R,t,x1,x2,K1,K2,True)
checks={}
for side,analytic in [(0,ga),(1,gb)]:
    for axis in range(2):
        plus=[x1.copy(),x2.copy()];minus=[x1.copy(),x2.copy()]
        plus[side][:,axis]+=.001;minus[side][:,axis]-=.001
        numeric=(residual(R,t,*plus,K1,K2)-residual(R,t,*minus,K1,K2))/.002
        checks[f'pixel_end{side}_axis{axis}']=float(np.max(np.abs(numeric-analytic[:,axis])))
        assert np.allclose(numeric,analytic[:,axis],rtol=1e-6,atol=1e-7)
d=rng.normal(size=5)*1e-7;rn,tn=update(R,t,d)
checks['pose_directional_linearization']=float(np.max(np.abs(residual(rn,tn,x1,x2,K1,K2)-r-jacobian(R,t,x1,x2,K1,K2)@d)))
assert checks['pose_directional_linearization']<1e-7
checks['rotation_orthogonality']=float(np.max(np.abs(rn.T@rn-np.eye(3))))
checks['translation_norm']=float(abs(np.linalg.norm(tn)-1))
assert checks['rotation_orthogonality']<1e-12 and checks['translation_norm']<1e-12
# Coordinate-scale contract: a global pixel resize changes residuals, not pose.
scale=2.3;A=np.diag([scale,scale,1.])
rs,gas,gbs=residual(R,t,x1*scale,x2*scale,A@K1,A@K2,True)
checks['pixel_scale_residual']=float(np.max(np.abs(rs-scale*r)))
checks['pixel_scale_gradients']=float(max(np.max(np.abs(gas-ga)),np.max(np.abs(gbs-gb))))
assert np.allclose(rs,scale*r,rtol=1e-10,atol=1e-10)
assert np.allclose(gas,ga,rtol=1e-10,atol=1e-10) and np.allclose(gbs,gb,rtol=1e-10,atol=1e-10)
out=Path('results');out.mkdir(exist_ok=True)
report=dict(state='PASSED',checks=checks,scope='numerical geometry interface only; not task quality evidence')
(out/'pose_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
