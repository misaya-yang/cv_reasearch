#!/usr/bin/env python3
"""CPU audit of frozen metric reach, using saved masks and deployed weight ranges.

No encoder, checkpoint, feature regeneration, query GT or parameter fitting.
The analytic bound assumes the method's ideal unit-feature/orthogonal-U contract;
it is not a certified floating-point bound for the full implementation.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import numpy as np


def unpack(record):
    raw=base64.b64decode(record.get('data',record.get('base64')))
    return np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='big').reshape(record['shape']).astype(bool)


def field_from_patch(patch,height,width):
    def coordinate(n,m):
        z=np.clip((np.arange(n)+.5)*m/n-.5,0,m-1)
        low=np.floor(z).astype(int)
        return low,np.minimum(low+1,m-1),z-low
    y0,y1,wy=coordinate(height,patch.shape[0]);x0,x1,wx=coordinate(width,patch.shape[1])
    horizontal=patch[:,x0]*(1-wx)+patch[:,x1]*wx
    return horizontal[y0]*(1-wy[:,None])+horizontal[y1]*wy[:,None]-.5


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Preserve existing audit')
    source=json.loads(a.report.read_text())
    if source['state']!='COMPLETED':raise ValueError('Completed frozen inference required')
    rows=[]
    for row in source['records']:
        host=unpack(row['prediction_bits']['insid3_native'])
        if host.shape!=(1024,1024):raise ValueError('Only known native64 to1024 contract supported')
        # A center sample has >.93 weight from its parent, so recovers that
        # binary parent regardless of neighbours. Verify the whole output too.
        patch=host[8::16,8::16].astype(float)
        h0=field_from_patch(patch,1024,1024)
        if not np.array_equal(h0>0,host):raise ValueError('Native hard-field reconstruction failed')
        info=dict(e=row['e'],c=row['c'],host_field_reconstruction_mask_exact=True,
                  hard_interior_pixels=int((np.abs(h0)==.5).sum()),arms={})
        for arm in ('metric_protected','metric_unprotected','metric_fixed_global'):
            audit=row['methods'][arm]['audit']
            epsilon=max(abs(audit['min_weight']-1),abs(audit['max_weight']-1))
            bound=4*audit['gain']*epsilon
            changed=unpack(row['prediction_bits'][arm])!=host
            eligible=np.abs(h0)<=bound
            info['arms'][arm]=dict(epsilon=epsilon,gain=audit['gain'],alpha=audit['alpha'],
                ideal_delta_infinity_bound=bound,ideal_eligible_pixels=int(eligible.sum()),
                eligible_percent=100*float(eligible.mean()),actual_changed_pixels=int(changed.sum()),
                actual_changes_outside_ideal_bound=int((changed&~eligible).sum()))
        rows.append(info)
    output=dict(state='COMPLETED_LOCAL_ANALYTIC_REACH_AUDIT',records=rows,
        source_sha256=hashlib.sha256(a.report.read_bytes()).hexdigest(),
        derivation='Unit differences have squared norm<=4; |dM^2-dI^2|<=4epsilon; each logmeanexp changes<=2epsilon/tau; FG-BG ratio<=4epsilon/tau; beta*tau and convex bilinear interpolation give |deltaH|<=4beta*epsilon.',
        scope='Ideal arithmetic reach of already frozen deployed metrics, not actual correction direction or a new segmentation result',
        FP32_certified=False,query_GT_used=False,encoder_or_GPU_used=False,deleted_features_restored=False,
        selected_epoch={k:v['selected_epoch'] for k,v in source['checkpoint_metadata'].items()})
    a.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(state=output['state'],cases=len(rows),max_adaptive_bound=max(r['arms']['metric_protected']['ideal_delta_infinity_bound'] for r in rows),
        adaptive_eligible_pixels=sum(r['arms']['metric_protected']['ideal_eligible_pixels'] for r in rows),
        total_pixels=len(rows)*1024**2,adaptive_changed_pixels=sum(r['arms']['metric_protected']['actual_changed_pixels'] for r in rows))))


if __name__=='__main__':main()
