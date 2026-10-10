"""Connect observation and BP on ambiguous synthetic evidence.

Checks component compatibility and meaningful invariants; no real inputs,
encoders, query labels, masks or scores are produced.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'2')
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))

def main(output):
    output=Path(output)
    if output.exists():
        raise ValueError('Preserve the existing contract receipt; choose a new output path')
    sources={name:REPO/path for name,path in (
        ('observation','src/ics/methods/joint_reference_observation.py'),
        ('BP','src/ics/methods/joint_reference_bp.py'),
        ('check','scripts/check_joint_reference_contract.py'))}
    source_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()}
    from ics.methods.joint_reference_observation import build_observation
    from ics.methods.joint_reference_bp import solve, BPConfig
    torch.set_num_threads(2);start=time.monotonic()
    # Identical appearance on both roles must stay ambiguous. Coordinate-only
    # role shifts have identical within-role structure; geometry must not
    # fabricate a unique label explanation for this symmetric fixture.
    raw=torch.zeros(4096,4,dtype=torch.float32);raw[:,0]=1
    mask=torch.zeros(1024,1024,dtype=torch.bool);mask[:,512:]=True
    observation=build_observation(raw,mask,raw,[raw.clone() for _ in range(4)],feature_dim=4)
    indices=torch.tensor([0,1,128,129],dtype=torch.int64)
    selected={k:torch.as_tensor(observation[k])[indices] for k in ('b0','valid','role','ref_xy','query_xy','P0')}
    identity=solve(selected['b0'],selected['valid'],selected['role'],selected['ref_xy'],selected['query_xy'],(2,2),BPConfig(lambda_=0.))
    candidate=solve(selected['b0'],selected['valid'],selected['role'],selected['ref_xy'],selected['query_xy'],(2,2))
    p0=selected['P0'][:,1].numpy()
    p_identity=np.asarray(identity.p_fg)
    p_candidate=np.asarray(candidate.p_fg)
    checks=dict(component_probability_and_coordinate_contract=identity.p_fg.shape==(4,),
        lambda_zero_preserves_complete_role_mass=np.allclose(p_identity,p0,rtol=0,atol=1e-8),
        appearance_alias_is_half_mass=np.allclose(p0,.5,rtol=0,atol=1e-6),
        symmetric_structures_do_not_fabricate_identity=np.allclose(p_candidate,.5,rtol=0,atol=1e-8),
        finite_joint_output=bool(np.isfinite(p_candidate).all()))
    if not all(checks.values()):raise AssertionError(checks)
    if source_sha256!={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()}:
        raise RuntimeError('Source changed during contract validation')
    result=dict(state='SYNTHETIC_COMPONENT_CONTRACT_PASSED',checks=checks,
        source_sha256=source_sha256,synthetic_fixture='Identical four-dimensional appearance; reference right half FG; query nodes [0,1,128,129]',
        P0=p0.tolist(),lambda_zero=p_identity.tolist(),faithful_joint=p_candidate.tolist(),
        seconds=time.monotonic()-start,encoder_forward=0,real_image_reads=0,raw_cache_reads=0,query_GT_reads=0,
        limits='Four synthetic nodes. Pro model known collapse/limited odds correction remains exposed; not a repaired model, real segmentation or mIoU.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: check_joint_reference_contract.py OUTPUT_JSON')
    main(sys.argv[1])
