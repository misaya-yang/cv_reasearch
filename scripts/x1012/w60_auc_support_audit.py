"""Post-hoc denominator audit of W60's fixed near-pure-token AUC; no inference."""
from pathlib import Path
import gzip
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/x1012'),str(ROOT/'scripts/x1013')]
import common as c
import plan_common as pc
import w31_demonstration as w31
import w59_normalization_causal as run


def main():
    out=c.task_path('W60'); assert (out/'scored.json').exists()
    with gzip.open(ROOT/'evidence/research/w60_pure_background_S.jsonl.gz','rt') as f:
        old={r['index']:r for r in map(json.loads,f)}
    rows=[]
    for b in w31.rows():
        r=old[str(b['index'])];gt,valid=pc.truth(b)
        cov=gt.reshape(64,16,64,16).mean((1,3)).ravel()
        vf=valid.reshape(64,16,64,16).mean((1,3)).ravel()
        pos=(cov>=.9)&(vf==1);neg=(cov<=.1)&(vf==1)
        middle=(cov>.1)&(cov<.9)&(vf==1);invalid=vf<1
        assert np.all(pos.astype(int)+neg+middle+invalid==1)
        mass=float(cov.sum())
        result=dict(index=r['index'],dataset=r['dataset'],auc_available=r['auc_B'] is not None,
                    positive_tokens=int(pos.sum()),negative_tokens=int(neg.sum()),
                    foreground_mass=mass,
                    foreground_mass_fraction={k:float(cov[mask].sum()/max(mass,1e-12))
                                              for k,mask in (('positive_label',pos),('negative_label',neg),('unselected_mixed',middle),('unselected_ignore',invalid))})
        assert pos.sum()+neg.sum()==r['pure_query_tokens']
        assert result['auc_available']==bool(pos.any() and neg.any())
        rows.append(result)
    summary=dict(n=len(rows),scope='Post-hoc support audit of pre-fixed W60 AUC labels (coverage >=.9 or <=.1); no new method, GT used only for scoring',domains={})
    for d in c.DOMAINS:
        sel=[r for r in rows if r['dataset']==d];scored=[r for r in sel if r['auc_available']]
        summary['domains'][d]=dict(n=len(sel),auc_n=len(scored),
                                  median_positive_tokens=float(np.median([r['positive_tokens'] for r in scored])),
                                  median_positive_foreground_fraction=float(np.median([r['foreground_mass_fraction']['positive_label'] for r in scored])),
                                  median_negative_foreground_fraction=float(np.median([r['foreground_mass_fraction']['negative_label'] for r in scored])),
                                  median_mixed_foreground_fraction=float(np.median([r['foreground_mass_fraction']['unselected_mixed'] for r in scored])))
    run.write(out/'auc_support_cases.json',rows);run.write(out/'auc_support_summary.json',summary)
    run.write(ROOT/'evidence/research/w60_auc_support.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
