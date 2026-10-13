#!/usr/bin/env python3
"""Can cached physical reference detail distinguish R-equivalent oracle rules?

Candidate directions are derived from query-label oracles. This study only
tests an observer premise; it is never an inference method or benchmark.
"""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import task_transfer_source_span_audit as span

d,base,np,torch,F=span.d,span.base,span.np,span.torch,span.F
PRODUCER=base.ASSETS/'a/reference_focus200_20261010'
OUT=base.ASSETS/'a/task_transfer_focus_observer_20261010'
OBSERVERS=['whole_reference','interpolated_focus','actual_focus','shuffled_actual_focus']


def prepare():
    if (OUT/'config.json').exists():return
    pairs=d.read(span.OUT/'pairs.json');bindings=d.read(PRODUCER/'input_bindings.json')
    tasks=[]
    for i,p in enumerate(pairs):
        matches=[r for r in bindings if r['reference_rgb_hash']==p['row']['reference_rgb_hash'] and
                 r['reference_mask_hash']==p['row']['reference_mask_hash']]
        if not matches:continue
        focus=min(matches,key=lambda r:r['index'])
        assert focus['dataset']==p['dataset']=='paco_part'
        assert focus['reference_path']==p['row']['reference_path']
        assert focus['reference_crop']==p['row'].get('reference_crop')
        tasks.append(dict(pair=p,parent_index=i,focus=focus))
    assert len(tasks)>0
    OUT.mkdir(parents=True,exist_ok=True)
    base.write(OUT/'tasks.json',tasks)
    base.write(OUT/'config.json',dict(n_pairs=len(tasks),observers=OBSERVERS,new_encodings=0,
        source_sha256=base.sha(Path(__file__)),helper_sha256=base.sha(Path(span.__file__)),
        parent_sha256={str(p):base.sha(p) for p in [span.OUT/'report.json',span.OUT/'verification.json',
            PRODUCER/'input_bindings.json',PRODUCER/'COMPLETE.json']},tasks_sha256=base.sha(OUT/'tasks.json'),
        candidates='theta_R +/- beta*oracle_null_delta; beta uses whole reference only and preserves all4096 R token binary decisions for both signs',
        criterion='Legitimate reference held-block balanced squared risk, with native focus physical pixel mass and whole-reference role denominators',
        control='Same physical centers/coverage with bilinear whole-unit-descriptor sampling and normalization; actual descriptor fields shuffled with seed0, preserving complete field/label marginals',
        view_selection='Existing original-task-A focus view; auxiliaryB keeps that same fixed view. Eligible pairs chosen by exact RGB/maskA identity and cache availability, never outcomes.',
        query_GT_in_directions=True,query_GT_in_new_criterion=False,
        warning='Query-trained directions are oracle counterfactuals. Observer selection cannot be deployed without a legal way to construct candidates; not a segmentation algorithm or originality claim.'))


def crop_coverage(pair,variant,box):
    binding=pair['variants'][variant]
    assert base.sha(binding['path'])==binding['file_sha256']
    mask=base.read_mask(binding['path'])
    whole=F.interpolate(torch.from_numpy(mask.astype('float32'))[None,None],(1024,1024),mode='nearest')[0,0]
    x0,y0,x1,y1=box
    cropped=whole[y0:y1,x0:x1]
    return F.avg_pool2d(F.interpolate(cropped[None,None],(1024,1024),mode='nearest'),16)[0,0].numpy().reshape(-1)


def risk(score,c,keep,denominator_fg,denominator_bg,physical_weight=1.):
    s=score[keep];fg=c[keep];bg=1-fg
    return float(physical_weight*(.5*np.sum(fg*(s-1)**2)/denominator_fg+.5*np.sum(bg*(s+1)**2)/denominator_bg))


def criterion_stage():
    from raw_feature_cache import tensor_hash
    from ics.official_data import decoded_rgb,array_hash
    from utils.data import build_transform
    prepare();cfg=d.read(OUT/'config.json')
    assert base.sha(Path(__file__))==cfg['source_sha256'] and base.sha(OUT/'tasks.json')==cfg['tasks_sha256']
    for p,h in cfg['parent_sha256'].items():assert base.sha(p)==h
    if (OUT/'criterion_sealed.json').exists():return
    tasks=d.read(OUT/'tasks.json');parent={r['id']:r for r in d.read(span.OUT/'report.json')['records']}
    source=d.read(d.OUT/'source_sealed.json');folds=d.split_grid();records=[];start=time.monotonic()
    transform=build_transform(1024);permutation=np.random.default_rng(0).permutation(4096)
    for task in tasks:
        p=task['pair'];i=task['parent_index'];foc=task['focus'];rec=parent[p['id']]
        assert base.sha(span.OUT/'oracle_fields'/rec['file'])==rec['fields_sha256']
        old=source['records'][i];assert old['id']==p['id']
        assert base.sha(d.OUT/'source_fields'/old['file'])==old['sha256']
        attempts=[]
        with base.deny_labels(base.OUT,attempts):
            assert base.sha(foc['entry_path'])==foc['entry_sha256']
            assert base.sha(foc['payload_path'])==foc['payload_sha256']
            entry=d.read(foc['entry_path'])
            with np.load(foc['payload_path'],allow_pickle=False) as z:actual=z['O/24'].copy()
            assert actual.dtype==np.float32 and actual.shape==(4096,1024)
            assert tensor_hash(actual)==entry['features']['O/24']['tensor_sha256']==foc['O24']['tensor_sha256']
            assert base.sha(foc['focus_rgb_path'])==foc['focus_rgb_file_sha256']
            rgb=decoded_rgb(base.ASSETS/p['row']['reference_path'],p['row'].get('reference_crop'))
            assert array_hash(np.asarray(rgb))==p['row']['reference_rgb_hash']
            focus_rgb=transform.transforms[0](rgb).crop(tuple(foc['box_xyxy']))
            assert array_hash(np.asarray(focus_rgb))==foc['focus_rgb_hash']
            assert tensor_hash(transform(focus_rgb).numpy())==foc['input_tensor_hash']==entry['input_tensor_hash']
            actual=F.normalize(torch.from_numpy(actual),dim=1).numpy().astype(float)
            (r,_),_=d.load_raw(p)
            x0,y0,x1,y1=foc['box_xyxy'];assert x1-x0==y1-y0==512
            centers_x=x0+(np.arange(64)+.5)*8;centers_y=y0+(np.arange(64)+.5)*8
            gx,gy=np.meshgrid(2*centers_x/1024-1,2*centers_y/1024-1)
            grid=torch.from_numpy(np.stack([gx,gy],-1).astype('float32'))[None]
            sampled=F.grid_sample(r.T.reshape(1,1024,64,64),grid,mode='bilinear',padding_mode='border',align_corners=False)
            derived=F.normalize(sampled[0].reshape(1024,-1).T,dim=1).numpy().astype(float)
            mapped=folds.reshape(64,64)[(centers_y//16).astype(int)[:,None],(centers_x//16).astype(int)[None,:]].reshape(-1)
            values=[]
            with np.load(span.OUT/'oracle_fields'/rec['file']) as fields,np.load(d.OUT/'source_fields'/old['file']) as weights:
                for m in rec['measures']:
                    if not m['valid']:continue
                    v,fold=m['variant'],m['fold'];prefix=f'{v}/{fold}/';old_key=f'{v}/unit_raw/{fold}/'
                    cr=d.coverage(p['variants'][v]['path'],p['variants'][v]['file_sha256'])
                    cf=crop_coverage(p,v,foc['box_xyxy'])
                    if v=='A':
                        assert base.sha(foc['coverage64_path'])==foc['coverage64_file_sha256']
                        assert np.array_equal(cf.reshape(64,64),np.load(foc['coverage64_path']))
                    reference=fields[prefix+'sparse_reference/r'];change=fields[prefix+'joint_sampled_nullspace/r']-reference
                    beta=min(span.safe_scale(reference,change),span.safe_scale(reference,-change))
                    assert np.array_equal(reference+beta*change>0,reference>0)
                    assert np.array_equal(reference-beta*change>0,reference>0)
                    null=fields[prefix+'null_coefficients'];theta=np.r_[weights[old_key+'coefficient'],weights[old_key+'bias']]
                    keep=folds!=fold;den_fg=cr[keep].sum();den_bg=(1-cr[keep]).sum();assert den_fg>0 and den_bg>0
                    scores={'whole_reference':(reference,change,cr,keep,1.)}
                    for name,x in [('interpolated_focus',derived),('actual_focus',actual)]:
                        s=x@theta[:-1]+theta[-1];delta=x@null[:-1]+null[-1]
                        scores[name]=(s,delta,cf,mapped!=fold,.25)
                    s,delta,c,mask,weight=scores['actual_focus']
                    scores['shuffled_actual_focus']=(s[permutation],delta[permutation],c,mask,weight)
                    observations={}
                    for name,(s,delta,c,mask,weight) in scores.items():
                        plus=risk(s+beta*delta,c,mask,den_fg,den_bg,weight)
                        minus=risk(s-beta*delta,c,mask,den_fg,den_bg,weight)
                        observations[name]=dict(plus_risk=plus,minus_risk=minus,risk_delta=plus-minus,
                            choose_plus=plus<minus,foreground_mass=float(weight*c[mask].sum()),background_mass=float(weight*(1-c[mask]).sum()))
                    values.append(dict(variant=v,fold=fold,beta=beta,whole_reference_binary_identity_both_signs=True,observers=observations))
            assert not attempts
        records.append(dict(id=p['id'],dataset=p['dataset'],parent_file=rec['file'],parent_fields_sha256=rec['fields_sha256'],
            focus_key=foc['key'],focus_payload_sha256=foc['payload_sha256'],criteria=values,new_query_label_access_attempts=attempts))
    base.write(OUT/'criteria.json',records)
    base.write(OUT/'criterion_sealed.json',dict(n_pairs=len(records),criteria_sha256=base.sha(OUT/'criteria.json'),
        config_sha256=base.sha(OUT/'config.json'),seconds=time.monotonic()-start,new_encodings=0,
        warning='No new query-label reads in reference criterion; candidate directions already contain parent query-label supervision.'))


def score():
    from scipy.stats import spearmanr
    seal=d.read(OUT/'criterion_sealed.json');assert base.sha(OUT/'criteria.json')==seal['criteria_sha256']
    truth=d.read(base.OUT/'query_truth.json');tasks={t['pair']['id']:t for t in d.read(OUT/'tasks.json')}
    folds=d.split_grid();records=[]
    for rec in d.read(OUT/'criteria.json'):
        with np.load(span.OUT/'oracle_fields'/rec['parent_file']) as fields:
            values=[]
            for criterion in rec['criteria']:
                v,fold=criterion['variant'],criterion['fold'];prefix=f'{v}/{fold}/'
                qb=truth[rec['id']]['masks'][v];c=d.coverage(qb['path'],qb['sha256']);keep=folds!=fold
                source=fields[prefix+'sparse_reference/q'];delta=fields[prefix+'joint_sampled_nullspace/q']-source
                plus=d.metrics(source+criterion['beta']*delta,c,keep);minus=d.metrics(source-criterion['beta']*delta,c,keep)
                chosen={name:plus if obs['choose_plus'] else minus for name,obs in criterion['observers'].items()}
                values.append(criterion|dict(query_plus=plus,query_minus=minus,selected_query=chosen))
        records.append(rec|dict(scored_criteria=values))
    summary={}
    for observer in OBSERVERS:
        photo_auc=[np.mean([m['selected_query'][observer]['auc'] for m in r['scored_criteria']]) for r in records]
        compared={}
        for control in OBSERVERS:
            compared[control]=base.bootstrap([np.mean([m['selected_query'][observer]['auc']-m['selected_query'][control]['auc'] for m in r['scored_criteria']]) for r in records])
        margins=[np.mean([-m['observers'][observer]['risk_delta'] for m in r['scored_criteria']]) for r in records]
        gains=[np.mean([m['query_plus']['auc']-m['query_minus']['auc'] for m in r['scored_criteria']]) for r in records]
        rho=spearmanr(margins,gains).statistic
        summary[observer]=dict(n_pairs=len(records),selected_query_auc=float(np.mean(photo_auc)),
            selected_query_auc_minus=compared,photo_risk_margin_vs_query_auc_spearman=float(rho) if np.isfinite(rho) else None,
            choose_plus_fraction=float(np.mean([m['observers'][observer]['choose_plus'] for r in records for m in r['scored_criteria']])))
    reference=dict(always_plus_query_auc=float(np.mean([np.mean([m['query_plus']['auc'] for m in r['scored_criteria']]) for r in records])),
        always_minus_query_auc=float(np.mean([np.mean([m['query_minus']['auc'] for m in r['scored_criteria']]) for r in records])),
        mean_symmetric_safe_beta=float(np.mean([m['beta'] for r in records for m in r['scored_criteria']])))
    base.write(OUT/'report.json',dict(state='COMPLETE_ORACLE_FOCUS_OBSERVER_PREMISE',observers=summary,oracle_controls=reference,
        records=records,criterion_seal_sha256=base.sha(OUT/'criterion_sealed.json'),new_encodings=0,
        warning='PACO-only17 cache-eligible pairs; all candidates contain query-label oracle directions. Not a deployable selector or new algorithm; shuffled control breaks feature/label physical alignment, not a pure higher-order interaction proof.'))
    lines=['# 真实参考focus能否选择同R判断的不同Q解释','',
        '仅17个PACO exact RGB＋maskA绑定的已有真实focus；A/B共享固定原A裁剪。候选±方向来自query标签oracle，两者保留4096 R token全部二值判断。观察判据只用合法R标签；这仍是oracle候选辨识前提，不是可部署方法。','',
        '|判据|选择后的Q AUC|偏好+比例|与Q方向收益的photo rho|','|---|---:|---:|---:|']
    for name,s in summary.items():lines.append(f"|{name}|{s['selected_query_auc']:.5f}|{s['choose_plus_fraction']:.3f}|{s['photo_risk_margin_vs_query_auc_spearman']}|")
    lines+=['',f"始终选+ oracle对照AUC={reference['always_plus_query_auc']:.5f}，始终选−={reference['always_minus_query_auc']:.5f}；不能忽略候选方向由query标签生成的乐观性。"]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    base.write(OUT/'activity.json',dict(state='DIAGNOSED',n_pairs=len(records),new_encodings=0))
    print(json.dumps(dict(observers=summary,oracle_controls=reference)),flush=True)


if __name__=='__main__':
    criterion_stage();score()
