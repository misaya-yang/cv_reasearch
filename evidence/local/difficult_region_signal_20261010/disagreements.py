"""Measure saved signal in actual FoRIS/native9 disagreements; no new models."""
import importlib.util
import json
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('diagnostic_core',ROOT/'raw/frozen/study.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
np=m.np
DATA=ROOT.parents[2].parent/'cv_data'
PILOT=DATA/'a/joint_role_pilot200_20261010'
OUT=ROOT/'disagreement';OUT.mkdir(exist_ok=True)

manifest=m.read(PILOT/'manifest.json')
raw=m.read(ROOT/'raw/sealed.json')
old={r['episode_id']:r for r in m.lines(PILOT/'inference.jsonl')}
rows=[];summary=defaultdict(list)
for row,rec in zip(manifest,raw['receipts']):
    assert row['episode_id']==rec['episode_id']
    src=old[row['episode_id']]
    path=PILOT/'predictions'/src['filename'];assert m.sha(path)==src['prediction_sha256']
    with np.load(path,allow_pickle=False) as z:
        masks={a:np.unpackbits(z['cli1024/'+a],count=1024**2).reshape(1024,1024).astype(bool)
               for a in ('foris.crf','region.fast')}
    path=ROOT/'raw/fields'/rec['filename'];assert m.sha(path)==rec['sha256']
    with np.load(path,allow_pickle=False) as z:
        fields={v:z['source_apd.ridge.'+v].ravel() for v in ('global','local4')}
    truth=m.mask(row,'query')
    b,n=masks['foris.crf'],masks['region.fast']
    rois={'disagreement':b^n,'native_only':n&~b,'whole_only':b&~n,'agree_fg':b&n,'agree_bg':~b&~n}
    r=dict(episode_id=row['episode_id'],dataset=row['dataset'],rois={})
    for name,roi in rois.items():
        fg,bg=m.mass128(roi&truth),m.mass128(roi&~truth)
        r['rois'][name]={v:m.weighted_rank(score,fg,bg) for v,score in fields.items()}
        for v,value in r['rois'][name].items():summary[row['dataset']+'/'+name+'/'+v].append(value)
    rows.append(r)
(OUT/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
report={}
for key,values in summary.items():
    valid=[v for v in values if v['auc'] is not None]
    report[key]=dict(n_valid=len(valid),macro={metric:float(np.mean([v[metric] for v in valid])) if valid else None
        for metric in ('auc','ap','tpr_at_fpr05','prevalence')},fg=sum(v['fg'] for v in values),bg=sum(v['bg'] for v in values))
m.write(OUT/'report.json',dict(n=len(rows),summary=report,script_sha256=m.sha(__file__),
    raw_seal_sha256=m.sha(ROOT/'raw/sealed.json'),manifest_sha256=m.sha(PILOT/'manifest.json'),
    scope='Fixed saved model fields and masks; fine128 score with canonical1024 pixel mass; query GT diagnostic only.'))
for key,value in report.items():
    if any('/'+roi+'/' in key for roi in ('disagreement','native_only','whole_only')):
        print(key,value['n_valid'],{k:round(v,4) if v is not None else None for k,v in value['macro'].items()})
