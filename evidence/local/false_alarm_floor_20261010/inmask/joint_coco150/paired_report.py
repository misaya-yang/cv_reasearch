"""Post-seal raster masks and paired comparisons for the frozen COCO150 probe.

Uses exactly the frozen helper's field, zero and10%-area abstention rules.
No encoder, no fitted parameter, and no query label is used in mask creation.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
import inmask_evidence as base


def main():
    args = base.arguments(['evaluate']).parse_args(['evaluate','--cohort','a/coco_role_competition200_20261010','--name','joint_coco150'])
    source,out = base.opened(args)
    plan = base.read(out/'config.json')
    seal = base.read(out/'sealed.json')
    assert seal['n'] == len(plan['ids']) == 150
    assert base.sha(Path(base.__file__)) == plan['instrument_sha256'] == 'ba73a7b1d79ba5e03ad21e3a6fbd4fea192f92e9b521d2f168417d81e939da2e'
    assert base.sha(out/'config.json') == seal['config_sha256']
    episodes = base.lines(out/'episodes.jsonl')
    report = base.read(out/'report.json')
    rows = source.rows[:150]
    assert [e['id'] for e in episodes] == plan['ids']
    fields = {k for e in episodes for k in e['fields']}
    stems = {k.split('.',1)[1] for k in fields if k.startswith('separate.') or k.startswith('joint.')}
    fields = sorted(fields | {arm+'.'+name for arm in ['separate','joint'] for name in stems})
    (out/'predictions').mkdir(exist_ok=True)
    details = []
    for i,(row,e) in enumerate(zip(rows,episodes)):
        field_path = out/'fields'/f'{i:06d}.npz'
        assert base.sha(field_path) == seal['fields'][e['id']]
        prediction = source.prediction(row)
        predicted = base.token_mass(prediction,plan['grid'])
        h,w = prediction.shape
        index = (np.arange(h)*plan['grid']//h)[:,None]*plan['grid']+(np.arange(w)*plan['grid']//w)[None,:]
        packed = dict(original_hw=np.asarray((h,w)),foris=np.packbits(prediction.reshape(-1)))
        info = dict(id=e['id'],cls=e['cls'],foris_iu=[e['hit'],e['g']+e['false_alarm']],fields={})
        with np.load(field_path,allow_pickle=False) as z:
            for name in fields:
                missing = name not in z.files
                low_area = False
                if missing:
                    keep = np.ones(plan['grid']**2,bool)
                else:
                    value = z[name].astype(np.float64)
                    proposal = value > base.ZERO(name)
                    low_area = bool(predicted[proposal].sum() < .1*predicted.sum())
                    keep = base.rule(value,base.ZERO(name),predicted)
                mask = prediction & keep[index]
                packed[name] = np.packbits(mask.reshape(-1))
                field = e['fields'].get(name)
                iu = [e['hit'],e['g']+e['false_alarm']] if field is None else [field['rule_hit'],e['g']+field['rule_false_alarm']]
                info['fields'][name] = dict(iu=iu,abstain_missing=missing,abstain_under10pct=low_area,
                    no_candidate=bool(predicted.sum()==0),changed_pixels=int((mask != prediction).sum()),
                    auc=None if field is None else field['auc'],
                    kept95=None if field is None or field['auc'] is None else field['kept95_mass']/max(field['false_alarm'],1))
        path = out/'predictions'/f'{i:06d}.npz'
        np.savez_compressed(path,**packed)
        info['prediction_sha256'] = base.sha(path)
        details.append(info)
    cls = np.asarray([e['cls'] for e in episodes])
    I0 = np.asarray([e['hit'] for e in episodes]);U0 = np.asarray([e['g']+e['false_alarm'] for e in episodes])
    names = sorted({k.split('.',1)[1] for k in fields if k.startswith('separate.') or k.startswith('joint.')})
    table = {}
    for name in names:
        arms,values = {},{}
        for arm in ['separate','joint']:
            key = arm+'.'+name
            values[arm] = tuple(np.asarray([d['fields'][key]['iu'][j] for d in details]) for j in [0,1])
            I,U = values[arm]
            entries = [d['fields'][key] for d in details]
            arms[arm] = dict(miou=base.macro(I,U,cls),delta_vs_foris=base.macro(I,U,cls)-base.macro(I0,U0,cls),
                abstain_missing_count=sum(e['abstain_missing'] for e in entries),
                abstain_under10pct_count=sum(e['abstain_under10pct'] for e in entries),
                no_candidate_count=sum(e['no_candidate'] for e in entries),
                changed_episodes=sum(e['changed_pixels']>0 for e in entries),
                instrument=report['datasets']['coco']['fields'].get(key))
        paired = [(d['fields']['joint.'+name]['auc'],d['fields']['separate.'+name]['auc']) for d in details
                  if d['fields']['joint.'+name]['auc'] is not None and d['fields']['separate.'+name]['auc'] is not None]
        table[name] = dict(**arms,joint_minus_separate_miou=arms['joint']['miou']-arms['separate']['miou'],
            paired_episode_bootstrap_interval=base.interval(*values['separate'],*values['joint'],cls),
            paired_auc_episodes=len(paired),paired_mean_auc_delta=None if not paired else float(np.mean([j-s for j,s in paired])))
    records = [base.read(out/'records'/f'{i:06d}.json') for i in range(150)]
    requests = [r for e in records for r in e['raw_requests']]
    def times(operation):
        now = [r['telemetry']['synchronized_forward_seconds'] for r in requests if r['operation']==operation and r['telemetry']['encoder_calls']]
        return dict(actual_forwards=len(now),mean_seconds=None if not now else float(np.mean(now)),sum_seconds=float(sum(now)))
    result = dict(state='COMPLETE_FROZEN_COCO150_PAIRED_REPORT',n=150,ids=plan['ids'],frame='original',grid=plan['grid'],
        metric='Observed38 fold/class pooled I/U; full150 original-prefix episodes; exposed development only',
        full_foris_miou=base.macro(I0,U0,cls),fields=table,
        policy='Same fixed fields; score>natural zero (back_auc .5, others0); under10% FoRIS area or missing evidence retains fullFoRIS',
        cost={name:times(name) for name in ['separate.reference','separate.query','joint.canvas']},
        actual_encoder_calls=sum(r['telemetry']['encoder_calls'] for r in requests),
        legacy_reuse_requests=sum(r['reused_legacy'] for r in requests),
        same_run_cache_hits=sum(r['telemetry']['state']=='CACHE_HIT' for r in requests),
        mask_records=len(details),masks_include_all_predeclared_fields=True,
        seal_sha256=base.sha(out/'sealed.json'),instrument_report_sha256=base.sha(out/'report.json'),
        report_source_sha256=base.sha(__file__),mask_rule_query_GT_reads=0,
        interval_scope='Paired exposed-episode bootstrap; no independent-generalization claim')
    base.write(out/'paired_mask_metrics.json',details)
    base.write(out/'paired_comparison.json',result)
    base.write(out/'paired_masks_sealed.json',dict(n=150,predictions=[d['prediction_sha256'] for d in details],
        paired_metrics_sha256=base.sha(out/'paired_mask_metrics.json'),report_sha256=base.sha(out/'paired_comparison.json')))
    print(json.dumps(dict(state=result['state'],n=150,foris=result['full_foris_miou'],encoder_calls=result['actual_encoder_calls']),indent=2))


if __name__ == '__main__':main()
