#!/usr/bin/env python3
"""Post-seal, post-score pixel and producer audit for the one RCG+RGB composition."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))


def main():
    import numpy as np
    from ics.experiment import sha,unpack,metric
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out
    seal=json.loads((out/'sealed.json').read_text())
    report=json.loads((out/'score/report.json').read_text())
    if seal['state']!='ALL_PREDICTIONS_SEALED' or seal['n']!=600 or report['n']!=600:
        raise ValueError('Require sealed, separately scored public600')
    for key in ('config','inference_manifest','evaluation_manifest'):
        if sha(out/(key+'.json'))!=seal[key+'_sha256']:raise ValueError('Sealed metadata changed')
    rows=json.loads((out/'evaluation_manifest.json').read_text())
    totals=dict(added_TP=0,deleted_FP=0,deleted_TP=0,added_FP=0)
    counts={'rcg_current.control':[],'rcg_rgb_proposal_v2':[]};details=[];old_xor=0
    durations={key:[] for key in ('wall_seconds','cpu_seconds','rcg_recompute_seconds','mean_recompute_seconds')}
    rgb_seconds=[];d_total=0;outside_xor=0;started=time.perf_counter()
    for row in rows:
        key=row['occurrence_id'];rp=out/'receipts'/(key+'.json');pp=out/'predictions'/(key+'.npz')
        if sha(rp)!=seal['receipts'][key]:raise ValueError('Receipt hash changed')
        receipt=json.loads(rp.read_text())
        if sha(pp)!=receipt['prediction_sha256'] or sha(row['packet_export'])!=receipt['inputs']['packet_sha256']:
            raise ValueError('Prediction/GT packet binding changed')
        with np.load(pp,allow_pickle=False) as archive:
            base=unpack(archive['rcg_current.control']);final=unpack(archive['rcg_rgb_proposal_v2'])
            cut=unpack(archive['rcg_rgb_full_cut.control']);unary=unpack(archive['rcg_rgb_same_unary.control'])
        d=cut^unary;d_total+=int(d.sum());outside_xor+=int(np.count_nonzero(final[~d]!=base[~d]))
        if not np.array_equal(final,np.where(d,cut,base)):raise ValueError('Specified local composition mismatch')
        # This is a distinct, post-seal evaluation phase. Query labels never
        # enter the producer or influence any constants/readout decisions.
        with np.load(row['packet_export'],allow_pickle=False) as archive:truth=unpack(archive['truth'])
        added=final&~base;deleted=base&~final
        delta=dict(added_TP=int((added&truth).sum()),deleted_FP=int((deleted&~truth).sum()),
                   deleted_TP=int((deleted&truth).sum()),added_FP=int((added&~truth).sum()))
        for name in totals:totals[name]+=delta[name]
        for name,mask in [('rcg_current.control',base),('rcg_rgb_proposal_v2',final)]:
            counts[name].append([int((mask&truth).sum()),int((mask|truth).sum())])
        historical_xor=None
        if row.get('evaluation_controls',{}).get('rcg.control'):
            spec=row['evaluation_controls']['rcg.control']
            if sha(spec['path'])!=spec['sha256']:raise ValueError('Historical RCG control changed')
            with np.load(spec['path'],allow_pickle=False) as archive:old=unpack(archive[spec['key']])
            historical_xor=int(np.count_nonzero(old!=base));old_xor+=historical_xor
        details.append(dict(occurrence_id=key,**delta,disagreement_pixels=int(d.sum()),
                            historical_rcg_mask_xor=historical_xor))
        for name in durations:durations[name].append(receipt[name])
        rgb_seconds.append(receipt['simple_control']['rgb_component_seconds'])
    for name,values in counts.items():
        value=metric(np.asarray(values),np.array([row['c'] for row in rows]))
        if abs(value-report['scores'][name])>1e-10:raise ValueError('Independent complete-count score mismatch')
    result=dict(state='POST_SEAL_COMPLETE_COUNTS_AND_LOCAL_RULE_AUDITED',n=600,
                reference_baseline='current recomputed RCG',pixel_changes=totals,
                corrected_pixels=totals['added_TP']+totals['deleted_FP'],
                harmed_pixels=totals['deleted_TP']+totals['added_FP'],
                net_correct_pixels=totals['added_TP']+totals['deleted_FP']-totals['deleted_TP']-totals['added_FP'],
                disagreement_pixels=d_total,outside_disagreement_baseline_xor=outside_xor,
                historical_rcg_current_mask_xor=old_xor,historical_parity_forced=False,
                independently_recounted_scores={name:report['scores'][name] for name in counts},
                production_wall_seconds=seal['elapsed_seconds'],
                conservative_collective_peak_rss_bytes=seal['peak_owned_rss_bytes'],
                serial_episode_costs={name:dict(sum_seconds=float(np.sum(values)),mean_seconds=float(np.mean(values)),
                                              p95_seconds=float(np.percentile(values,95))) for name,values in durations.items()},
                rgb_cost=dict(sum_seconds=float(np.sum(rgb_seconds)),mean_seconds=float(np.mean(rgb_seconds)),
                              p95_seconds=float(np.percentile(rgb_seconds,95))),
                scoring_audit_seconds=time.perf_counter()-started,
                exposure='reused public600 exploratory composition; not independent confirmation',
                independent_method_increment=0,no_new_information=True,new_encoder_forwards=0,
                query_gt_opened_in_producer=False,query_gt_opened_only_after_seal=True,
                details=details)
    (out/'score/composition_pixel_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='details'},indent=2,allow_nan=False))


if __name__=='__main__':main()
