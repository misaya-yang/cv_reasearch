"""Compact source-identity and influence diagnostics; inputs stay read-only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
DATA=REPO.parent/'cv_data'
STUDY=REPO/'evidence/local/difficult_region_signal_20261010'
OLD=REPO/'evidence/local/cpu100_20261006/server/screen12_native200_v3/screen12_native200_v3/score/report.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
old=read(OLD)
sources={}
for fn in ('src/ics/cpu100/invariance_support.py','src/ics/cpu100/common.py','src/ics/cpu100/decision_risk.py','src/ics/methods/direct_dino_features.py'):
    content=subprocess.check_output(['git','show','6e986b9^:'+fn],cwd=REPO)
    digest=hashlib.sha256(content).hexdigest()
    assert digest==old['method_source_hashes'][fn]
    dest=OUT/('historical_'+Path(fn).name);dest.write_bytes(content)
    sources[fn]=dict(git_ref='6e986b9^',sha256=digest,bound_to_native200=True,snapshot=str(dest))

h=old['class_actions_vs_dino_prototype']['inv_huber_reference_readout']
r=old['class_actions_vs_dino_prototype']['inv_huber_ridge']
deltas=sorted([dict(class_id=c,delta=100*(h[c]['final_I']/h[c]['final_U']-r[c]['final_I']/r[c]['final_U']),
    huber_I=h[c]['final_I'],huber_U=h[c]['final_U'],ridge_I=r[c]['final_I'],ridge_U=r[c]['final_U']) for c in h],key=lambda x:x['delta'],reverse=True)
sumd=sum(d['delta'] for d in deltas)
huber_influence=dict(top_classes=deltas[:5],top1_share_of_net_macro_gain=deltas[0]['delta']/sumd,
    excluding_largest_class_gain=sum(d['delta'] for d in deltas[1:])/(len(deltas)-1),
    note='Posthoc class influence; does not replace the full declared metric or establish a new cohort.')

raw={r['episode_id']:r for r in lines(STUDY/'raw/episodes.jsonl')}
prior={r['episode_id']:r for r in lines(STUDY/'existing/episodes.jsonl')}
valid=[]
for eid,row in raw.items():
    if row['dataset']!='paco_part':continue
    a=row['rois']['new_unary_foreground']['source_apd.ridge.global'];b=prior[eid]['rois']['new_unary_foreground']['P0']
    if a['auc'] is None:continue
    valid.append(dict(episode_id=eid,fg=a['fg'],bg=a['bg'],weight=a['fg']*a['bg'],auc_delta=a['auc']-b['auc']))
total=sum(r['weight'] for r in valid)
for r in valid:r['pairmass_share']=r['weight']/total;r['weighted_contribution']=r['pairmass_share']*r['auc_delta']
valid.sort(key=lambda x:x['weight'],reverse=True)
mass_influence=dict(top_regions=valid[:5],effective_episode_count=total**2/sum(r['weight']**2 for r in valid),
    within_image_pairmass_weighted_auc_delta=sum(r['weighted_contribution'] for r in valid),
    macro_auc_delta=float(np.mean([r['auc_delta'] for r in valid])),
    note='Diagnostic mean of within-image AUC weighted by FG*BG; not pooled cross-image AUC or official mIoU.')

cfg=read(DATA/'a/joint_role_pilot200_20261010/config.json');profile=read(cfg['raw_profile'])
assert sha(cfg['raw_profile'])==cfg['raw_profile_sha256']
assert profile['weights_sha256'] in old['checkpoint_hashes']
old_actions={}
for method in ('inv_huber_reference_readout','inv_huber_ridge','DR_control_average_logistic','inv_adversarial_constant'):
    rows=old['actions_vs_dino_prototype'][method]
    old_actions[method]={k:sum(r[k] for r in rows) for k in ('add_TP','add_FP','delete_TP','delete_FP')}
legacy={}
for family,targets in [
    ('nine_public600_v2',{'reference_quadratic':['quadratic_linear.control','quadratic_homogeneous.control','mean.control'],
                        'query_recurrence':['recurrence_all_seed.control','recurrence_single_seed.control','mean.control'],
                        'reference_prior_shift':['prior_reference_fraction.control','prior_balanced.control','mean.control']}),
    ('extensions_public600_v2',{'reference_hull':['hull_affine.control','hull_subspace.control','mean.control'],
                               'reference_triplet_relations':['triplet_zero.control','triplet_no_third.control','mean.control']})]:
    p=REPO/'evidence/local/research_20261006/server_prepared_01a1100b'/family/'score/report.json'
    v=read(p);selected={}
    for method,controls in targets.items():
        selected[method]=dict(score=v['scores'][method],controls={})
        for control in controls:
            delta=v['contrasts'][method][control]
            assert abs(delta['gain']-(v['scores'][method]-v['scores'][control]))<1e-12
            selected[method]['controls'][control]=dict(score=v['scores'][control],**delta)
    legacy[family]=dict(source=str(p),sha256=sha(p),n=v['n'],classes=v['classes'],bootstrap=v['bootstrap'],
        photo_groups=v['photo_groups'],largest_photo_group=v['largest_photo_group'],methods=selected,
        scope='Historical processed-cache canonical1024 scores read from original report; score subtraction checked, no raw I/U or CI recalculation.')
report=dict(source_identity=sources,current_profile=dict(path=cfg['raw_profile'],sha256=cfg['raw_profile_sha256'],
    checkpoint=profile['weights_sha256'],preprocessing=profile['preprocessing'],feature_definition=profile['features']),
    old_huber_influence=huber_influence,old_actions=old_actions,paco_added_roi_influence=mass_influence,legacy=legacy,
    script_sha256=sha(__file__),input_sha256={str(p):sha(p) for p in [OLD,STUDY/'raw/episodes.jsonl',STUDY/'existing/episodes.jsonl']})
(OUT/'supplement.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(sources_verified=len(sources),legacy_reports_checked=len(legacy),output=str(OUT/'supplement.json')),ensure_ascii=False))
