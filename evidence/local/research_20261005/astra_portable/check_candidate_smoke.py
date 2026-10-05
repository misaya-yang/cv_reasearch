#!/usr/bin/env python3
"""Check a five-case source replay against bundled original frozen evidence.

No feature or GT read. The fixed-fraction control additionally checks the full
220 label-free budget calculation: a five-case smoke has different calibration
data and must not be mislabeled as a full-220 cross-fit reproduction.
"""
import argparse, json
from pathlib import Path
import numpy as np
import adaptive_deletion as deletion
from statistics_randomstate import episode_key

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--evidence', type=Path, default=Path(__file__).parent/'evidence')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('No overwrite')
    f=json.loads((a.run/'freeze.json').read_text())
    rows=sorted(json.loads((a.evidence/'prediction_budgets220.json').read_text()),
                key=lambda r:(r['fold'],r['e'],r['c']))
    budgets={episode_key(r):r for r in rows}
    fractions=deletion.crossfit_fractions(rows,budgets)
    old=json.loads((a.evidence/'original_crossfit_parameters220.json').read_text())
    for fold in old:
        for field in ['fraction','train_BG_gate_deletions','train_domain_pixels',
                      'train_episodes','test_episodes','train_keys',
                      'excluded_other_fold_shared_photo_keys']:
            assert fractions[fold][field]==old[fold][field],(fold,field)
    result=dict(five_case_execution_only=True,full220_budget_parameters_exact=True,
                fixed_fraction_check='Use full220 original label-free fractions for original-mask parity; five-case own cross-fit is different by definition.',episodes={})
    with np.load(a.evidence/'original_reference_masks5.npz',allow_pickle=False) as original:
        for k,rec in f['records'].items():
            with np.load(a.run/rec['path'],allow_pickle=False) as z:
                mismatches={};field_errors={}
                for n in z.files:
                    full=k+'__'+n
                    if full not in original or n==deletion.FIXED_ARM:continue
                    if n.endswith('_field'):
                        field_errors[n]=float(np.abs(z[n]-original[full]).max())
                    else:
                        mismatches[n]=int(np.unpackbits(z[n]^original[full]).sum())
                fixed,K=deletion.fixed_fraction_mask(z,z,fractions[str(budgets[k]['fold'])]['fraction'])
                mismatches['fixed_fraction_with_original220_calibration']=int(np.unpackbits(fixed^original[k+'__'+deletion.FIXED_ARM]).sum())
                result['episodes'][k]=dict(mask_mismatches=mismatches,field_max_abs_error=field_errors,
                    original220_fixed_K=K,evidence_drift=f['audit'][k]['quantized_feature_evidence_audit'])
                assert max(mismatches.values(),default=0)==0,(k,mismatches)
    result['all_compared_masks_exact']=True
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PASS: all available original masks exact; full220 cross-fit budget parameters exact')
if __name__=='__main__':main()
