"""Verify one legal cached observation, without segmentation or query labels.

Uses the first already-frozen DeepGlobe pair, not a selected good example.
There is no encoder, FoRIS host, solver, mask threshold or scoring entry point.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'2')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def main(assets,output):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.official_data import load_inputs
    from raw_feature_cache import RawFeatureCache
    from ics.methods.joint_reference_observation import build_observation
    torch.set_num_threads(2)
    assets=Path(assets);output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'input_receipt.json').exists():
        raise ValueError('Input probe already completed; inspect receipt instead of repeating')
    catalogue_path=assets/'a/pro_joint_correspondence_review_20261010/assets_availability.json'
    catalogue=json.loads(catalogue_path.read_text())
    source=assets/'a/deepglobe_component100_20261009'
    rows=json.loads((source/'manifest.json').read_text())
    row=rows[0]
    expected=next(r for r in catalogue['datasets']['DeepGlobe100']['records'] if r['episode_id']==row['episode_id'])
    profile=Path(catalogue['profile_path'])
    if sha(profile)!=catalogue['profile_sha256']:raise ValueError('Raw profile changed')
    cache=RawFeatureCache(profile.parent.parent,json.loads(profile.read_text()))
    sys.path.insert(0,str(assets/'third_party/foris_official'))
    from utils.data import build_transform
    transform=build_transform(1024)
    # load_inputs opens RGB R/Q and the legal reference mask only.
    reference,mask,query=load_inputs(row,assets)
    canonical=transform.transforms[0](query)
    requests=[('reference_full',reference,expected['roles']['reference_full']['metadata_candidates'][0]),
              ('query_full',query,expected['roles']['query_full']['metadata_candidates'][0])]
    for corner in expected['corners']:
        requests.append(('query_window_'+str(corner['window_index']),canonical.crop(tuple(corner['box_xyxy'])),corner['key']))
    arrays=[];receipts=[];started=time.monotonic()
    for role,image,expected_key in requests:
        value=transform(image).numpy();key=cache.key(value)
        if key!=expected_key:raise ValueError('Actual transformed input differs from recorded '+role)
        raw=cache.read(value,('O/24',))['O/24']
        if raw.dtype!=np.float32 or raw.shape!=(4096,1024):raise ValueError('Wrong actual feature profile')
        arrays.append(torch.from_numpy(raw))
        entry_path=cache.folder/key/'entry.json';entry=json.loads(entry_path.read_text())
        receipts.append(dict(role=role,key=key,entry_sha256=sha(entry_path),
            payload_sha256=entry['file_sha256'],O24_tensor_sha256=entry['features']['O/24']['tensor_sha256']))
    load_seconds=time.monotonic()-started
    mask1024=F.interpolate(mask.float()[None,None],(1024,1024),mode='nearest')[0,0]
    observation_started=time.monotonic()
    observed=build_observation(arrays[0],mask1024,arrays[1],arrays[2:])
    observation_seconds=time.monotonic()-observation_started
    # Store only the legal candidate representation, never an output mask.
    keys=('b0','unary_cost','valid','role','refpatch','ref_xy','query_xy','P0','alpha')
    packed={key:torch.as_tensor(observed[key]).cpu().numpy() for key in keys}
    for key in ('b0','P0','alpha','ref_xy','query_xy'):
        if not np.isfinite(packed[key]).all():raise ValueError('Nonfinite actual observation: '+key)
    b0,valid,role=packed['b0'],packed['valid'].astype(bool),packed['role']
    if b0.shape!=(16384,16) or np.any(b0[~valid]!=0):raise ValueError('Wrong padded candidate representation')
    mass=np.stack([np.where(valid&(role==y),b0,0).sum(1) for y in (0,1)],1)
    role_error=float(np.max(np.abs(mass-packed['P0'])))
    if role_error>1e-5:raise ValueError('Candidate truncation changed complete role mass')
    field_path=output/'legal_observation.npz'
    np.savez_compressed(field_path,**packed)
    pfg=packed['P0'][:,1]
    receipt=dict(state='INPUT_AND_UNARY_PREPARED_ONLY',episode_id=row['episode_id'],
        source_manifest_sha256=sha(source/'manifest.json'),catalogue_sha256=sha(catalogue_path),
        profile_sha256=sha(profile),source_sha256=sha(REPO/'src/ics/methods/joint_reference_observation.py'),
        probe_source_sha256=sha(Path(__file__)),reference_mask_hash=row['reference_mask_hash'],
        raw_inputs=receipts,raw_requests=6,load_seconds=load_seconds,observation_seconds=observation_seconds,
        observation_sha256=sha(field_path),nodes=16384,candidate_slots=16,
        candidate_role_mass_max_error=role_error,
        alpha_summary={str(y):dict(min=float(packed['alpha'][:,y].min()),median=float(np.median(packed['alpha'][:,y]))) for y in (0,1)},
        unary_fg_above_0731_fraction=float((pfg>0.73105857863).mean()),
        unary_fg_below_0269_fraction=float((pfg<0.26894142137).mean()),
        encoder_constructions=0,encoder_forward=0,FoRIS_calls=0,CRF_calls=0,
        query_GT_reads=0,parent_GT_reads=0,solver_calls=0,new_segmentation_masks=0,
        scope='One original first-pair input/candidate preparation. Confidence fractions are unlabeled, not error rates. No mIoU, new segmentation, sample queue or cold endpoint claim.')
    write(output/'input_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.assets,args.output)
