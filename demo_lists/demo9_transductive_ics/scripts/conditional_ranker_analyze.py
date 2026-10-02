#!/usr/bin/env python3
"""CPU-only matched full-token ranker analysis with fixed strong scalar controls.

Requires healthy COMPLETED train status before fitting. Ridge/HGB consume exactly
8 label-free scores + within-query ranks/zscores + direct flag (25 features).
Fit train only; dev is diagnostic, never selects hyperparameters/checkpoints.
Native and original-photo I/U stay separate, with the SAME chosen candidate.
All-role-photo connected test components are the primary bootstrap units;
episode/class intervals are explicitly non-independent sensitivity analyses.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from conditional_ranker_train import (apply_manifest, load_records, merge_controls_overlay,
    photo_id, purge_split, validate_record, append_counterfactual_training)
from offline_selector import SCORES, candidate_features, fixed_models

SEED = 2044
BASELINES = ('direct', 'native_naive', 'cached_naive', 'cached_ag4', 'candidate_oracle')
LEARNED = ('stat_mlp', 'rq', 'rqdonor', 'ridge', 'hist_gradient_boosting')


def candidate_mapping(row):
    """Only frozen provenance, never GT quality, defines baseline candidate IDs."""
    expected = dict(direct='direct_1shot', native_naive='naive_native_multi',
                    cached_naive='naive_cached_native_p1', cached_ag4='current_ag4_cached_native_p1')
    mapping = {}
    for name, kind in expected.items():
        indices = [i for i, p in enumerate(row['candidate_provenance']) if p.get('kind') == kind]
        if len(indices) != 1:
            raise ValueError(f'e={row["e"]}: need exactly one {kind}, found {indices}')
        mapping[name] = indices[0]
    if mapping['direct'] != 0 or row['direct_index'] != 0:
        raise ValueError('predeclared direct candidate must be index zero')
    return mapping


def scalar_features(row):
    names = row.get('scalar_feature_names')
    if names is None or list(names) != list(SCORES):
        raise ValueError('strong controls require named original eight scores in exact order')
    values = row['scalar_features'].numpy()
    if values.shape != (len(row['candidate_masks']), 8):
        raise ValueError('strong scalar input must be [K,8]')
    result = candidate_features({key: values[:, i] for i, key in enumerate(SCORES)})
    if result.shape[1] != 25:
        raise AssertionError('fixed scalar feature pipeline changed')
    return result


def connected_photo_components(rows):
    """Union episodes sharing ANY support/query/donor canonical photo ID."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    owner = {}
    for i, row in enumerate(rows):
        for identity in row['_photos']:
            if identity in owner:
                union(i, owner[identity])
            else:
                owner[identity] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    groups = sorted(groups.values(), key=lambda values: values[0])
    membership = np.empty(len(rows), dtype=int)
    for component, indices in enumerate(groups):
        membership[indices] = component
    return membership, [dict(episodes=[int(rows[i]['e']) for i in group],
        classes=sorted({int(rows[i]['c']) for i in group}), size=len(group),
        photo_ids=len(set().union(*(rows[i]['_photos'] for i in group)))) for group in groups]


def native_score(classes, iu):
    return 100 * float(np.mean([iu[classes == c, 0].sum() / max(iu[classes == c, 1].sum(), 1)
                               for c in np.unique(classes)]))


def weighted_score_draws(classes, arrays, weights):
    """Component draws macro-average represented classes; absent classes are omitted."""
    unique = np.unique(classes)
    active = np.stack([weights[:, classes == c].sum(1) > 0 for c in unique], axis=1)
    denominator = active.sum(1).clip(min=1)
    draws = {}
    for name, iu in arrays.items():
        ratios = []
        for c in unique:
            take = classes == c
            intersection = weights[:, take] @ iu[take, 0]
            union = weights[:, take] @ iu[take, 1]
            ratios.append(intersection / np.maximum(union, 1))
        draws[name] = (np.stack(ratios, axis=1) * active).sum(1) / denominator * 100
    return draws


def paired_bootstrap(classes, arrays, components, repetitions):
    unique = np.unique(classes)
    rng = np.random.default_rng(SEED)
    component_count = int(components.max()) + 1
    counts = rng.multinomial(component_count, np.full(component_count, 1/component_count), size=repetitions)
    component_draws = weighted_score_draws(classes, arrays, counts[:, components])
    # Within-class paired episode resampling is a sensitivity, not independence.
    episode_weights = np.zeros((repetitions, len(classes)))
    for c in unique:
        take = np.flatnonzero(classes == c)
        episode_weights[:, take] = rng.multinomial(len(take), np.full(len(take), 1/len(take)), size=repetitions)
    episode_draws = weighted_score_draws(classes, arrays, episode_weights)
    sampled_classes = rng.integers(0, len(unique), size=(repetitions, len(unique)))
    class_draws = {}
    for name, iu in arrays.items():
        ratios = np.array([100 * iu[classes == c, 0].sum()/max(iu[classes == c, 1].sum(), 1) for c in unique])
        class_draws[name] = ratios[sampled_classes].mean(1)
    output = {}
    for name, iu in arrays.items():
        point = native_score(classes, iu)
        output[name] = dict(class_miou=point)
        for label, draws in [('all_role_photo_component_bootstrap', component_draws),
                             ('episode_stratified_sensitivity_NOT_independent', episode_draws),
                             ('class_block_sensitivity_NOT_independent', class_draws)]:
            result = dict(score_95=list(np.quantile(draws[name], [.025, .975])))
            for control in ('direct', 'cached_ag4', 'stat_mlp', 'ridge', 'hist_gradient_boosting'):
                if control in arrays:
                    result['gain_vs_' + control + '_95'] = list(np.quantile(draws[name] - draws[control], [.025, .975]))
            output[name][label] = result
        output[name]['gain_vs_direct_pp'] = point - native_score(classes, arrays['direct'])
        output[name]['gain_vs_cached_ag4_pp'] = point - native_score(classes, arrays['cached_ag4'])
    return output


def selection_diagnostics(rows, selections):
    result = {}
    quality = [r['candidate_iou'].numpy() for r in rows]
    direct = np.array([values[0] for values in quality])
    current = np.array([values[candidate_mapping(r)['cached_ag4']] for r, values in zip(rows, quality)])
    oracle = np.array([values.max() for values in quality])
    for name, choices in selections.items():
        chosen = np.array([values[i] for values, i in zip(quality, choices)])
        regret = oracle - chosen
        row = dict(mean_episode_native_iou=float(chosen.mean()), mean_episode_native_regret=float(regret.mean()),
            max_native_regret=float(regret.max()), oracle_match_fraction=float(np.mean(regret <= 1e-8)))
        for baseline, values in [('direct', direct), ('cached_ag4', current)]:
            delta = chosen - values
            row['vs_' + baseline] = dict(wins=int((delta > 1e-8).sum()), losses=int((delta < -1e-8).sum()),
                ties=int((np.abs(delta) <= 1e-8).sum()), mean_episode_iou_gain=float(delta.mean()),
                gain_episodes=[int(r['e']) for r, value in zip(rows, delta) if value > 1e-8],
                loss_episodes=[int(r['e']) for r, value in zip(rows, delta) if value < -1e-8])
        result[name] = row
    return result


def control_evaluate(model, rows):
    chosen = [int(np.argmax(model.predict(scalar_features(row)))) for row in rows]
    return dict(episodes=len(rows), mean_episode_native_iou=float(np.mean([
        float(row['candidate_iou'][index]) for row, index in zip(rows, chosen)])))


def read_healthy_status(path, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    while True:
        status = json.loads(path.read_text()) if path.exists() else None
        if status and status.get('state') == 'COMPLETED':
            if not all(status['arms'].get(name, {}).get('state') == 'COMPLETED' for name in ('scalar', 'rq', 'rqdonor')):
                raise ValueError('healthy complete scalar/RQ/RQdonor required')
            return status
        if status and status.get('state') in ('ERROR', 'NOT_EVALUABLE', 'PARTIALLY_EVALUABLE'):
            raise ValueError('training did not complete a healthy matched comparison: '+status['state'])
        if time.monotonic() >= deadline:
            raise ValueError('NOT_READY: training must complete before fitting or exposing test results')
        time.sleep(min(5, max(0, deadline-time.monotonic())))


def verify_training_identity(status, index, manifest, audit, counterfactual_report=None):
    source = {str(Path(key).resolve()): value for key, value in status['source_sha256'].items()}
    for path in (index, manifest) + ((counterfactual_report,) if counterfactual_report else ()):
        if source.get(str(path.resolve())) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('training input identity mismatch: '+str(path))
    if status['split_audit']['retained_episode_ids'] != audit['retained_episode_ids']:
        raise ValueError('retained split episodes differ from trained models')


def learned_choices(status, test):
    choices = {}
    expected = {int(r['e']) for r in test}
    for output, arm in [('stat_mlp', 'scalar'), ('rq', 'rq'), ('rqdonor', 'rqdonor')]:
        saved = status['arms'][arm]['test_records']
        by_e = {int(r['e']): r for r in saved}
        if len(by_e) != len(saved) or set(by_e) != expected:
            raise ValueError('trained test coverage mismatch')
        indices = []
        for row in test:
            record = by_e[int(row['e'])]
            index = int(record['chosen'])
            if int(record['c']) != int(row['c']) or not 0 <= index < len(row['candidate_masks']):
                raise ValueError('trained candidate metadata mismatch')
            if not np.array_equal(np.asarray(record['chosen_iu']), row['candidate_iu'][index].numpy()):
                raise ValueError('trained choice native I/U mismatch')
            indices.append(index)
        choices[output] = indices
    return choices


def self_check():
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    rows = []
    provenance = [{'kind':'direct_1shot'}] + [{'kind':'donor_two_hop','donor_slot':j} for j in range(3)] + [
        {'kind':'naive_native_multi'}, {'kind':'naive_cached_native_p1'}, {'kind':'current_ag4_cached_native_p1'}]
    for e in range(10):
        raw = rng.uniform(size=(7,8)).astype(np.float32)
        iu = torch.tensor([[30+j*5,100] for j in range(7)])
        ids = frozenset({str(1000+e),str(2000+e),str(3000+e//2)})
        row = dict(e=e,c=e//2,direct_index=0,photo_ids=list(ids),_photos=ids,
            candidate_masks=torch.rand(7,4,4)>.5,candidate_provenance=provenance,
            candidate_iu=iu,candidate_iou=(iu[:,0]/iu[:,1]).float(),candidate_original_iu=iu*2,
            scalar_features=torch.from_numpy(raw),scalar_feature_names=list(SCORES))
        rows.append(row)
        assert scalar_features(row).shape == (7,25)
        assert candidate_mapping(row)==dict(direct=0,native_naive=4,cached_naive=5,cached_ag4=6)
        changed=dict(row,candidate_iou=1-row['candidate_iou'],c=999,e=999)
        assert np.array_equal(scalar_features(row),scalar_features(changed))
        assert candidate_mapping(row)==candidate_mapping(changed)
    membership, groups=connected_photo_components(rows)
    assert len(groups)==5 and all(x['size']==2 for x in groups)
    arrays={k:np.array([r['candidate_iu'][0].numpy() for r in rows]) for k in ('direct','cached_ag4','stat_mlp','ridge','hist_gradient_boosting')}
    boot=paired_bootstrap(np.array([r['c'] for r in rows]),arrays,membership,100)
    assert boot['ridge']['all_role_photo_component_bootstrap']['gain_vs_direct_95']==[0.,0.]
    try:
        models=fixed_models()
    except ModuleNotFoundError:
        model_test='SKIPPED_local_sklearn_unavailable'
    else:
        X=np.concatenate([scalar_features(r) for r in rows[:6]])
        y=np.concatenate([r['candidate_iou'].numpy() for r in rows[:6]])
        for model in models.values():
            model.fit(X,y)
            assert np.isfinite(model.predict(scalar_features(rows[-1]))).all()
        model_test='PASSED_fixed_Ridge_HGB_synthetic_fit'
    # Integration through the actual source loader: no real token tensors are copied.
    with tempfile.TemporaryDirectory(prefix='conditional_analysis_loader_') as folder:
        root=Path(folder);base_dir=root/'base';controls=root/'controls';base_dir.mkdir();controls.mkdir()
        base=dict(e=0,c=0,split='train',photo_ids=['100','200','300','400','500'],
            reference_tokens=torch.randn(16,32).half(),query_tokens=torch.randn(16,32).half(),
            reference_mask=torch.rand(4,4)>.5,donor_tokens=torch.randn(3,16,32).half(),
            donor_masks=torch.rand(3,4,4)>.5,candidate_masks=rows[0]['candidate_masks'][:4],
            candidate_iu=rows[0]['candidate_iu'][:4],candidate_iou=rows[0]['candidate_iou'][:4],candidate_provenance=provenance[:4])
        base_path=base_dir/'episode.pt';torch.save(base,base_path)
        source=base_dir/'report.json';source.write_text(json.dumps({'records':[{'path':str(base_path)}]}))
        overlay={k:base[k] for k in ('e','c','split','photo_ids')}
        overlay.update(source_episode_path=str(base_path),source_candidate_count=4,candidate_state='COMPLETE',
            candidate_masks=rows[0]['candidate_masks'],candidate_iu=rows[0]['candidate_iu'],candidate_iou=rows[0]['candidate_iou'],
            candidate_provenance=provenance,scalar_features=rows[0]['scalar_features'],scalar_feature_names=list(SCORES),audit={'core_native_iu_exact':True})
        overlay_path=controls/'overlay.pt';torch.save(overlay,overlay_path)
        index=controls/'report.json';index.write_text(json.dumps(dict(source_report=str(source),records=[dict(path=str(base_path),overlay_path=str(overlay_path))])))
        loaded,_,_=load_records(index)
        validate_record(loaded[0],strict_tokens=False)
        assert scalar_features(loaded[0]).shape==(7,25)
        assert loaded[0]['_overlay_prefix_verified']
    print(json.dumps(dict(cpu_10_synthetic='PASSED',candidate_mapping_GT_free=True,
        scalar_features_GT_and_ID_free=True,component_bootstrap='PASSED',source_overlay_loader='PASSED',
        fixed_models=model_test)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--file',type=Path)
    ap.add_argument('--manifest',type=Path)
    ap.add_argument('--counterfactual-report',type=Path)
    ap.add_argument('--train-status',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--max-bytes',type=int,default=6_000_000_000)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--wait-seconds',type=int,default=0)
    ap.add_argument('--self-check',action='store_true')
    a=ap.parse_args()
    if a.self_check:self_check();return
    if not all((a.file,a.manifest,a.train_status,a.out)) or a.bootstrap<100 or not 0<a.max_bytes<=6_000_000_000 or a.wait_seconds<0:
        ap.error('require --file --manifest --train-status --out; bootstrap>=100; max-bytes<=6GB')
    status=read_healthy_status(a.train_status,a.wait_seconds)
    started=time.monotonic()
    report=dict(state='RUNNING',args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.file,a.manifest,a.train_status,Path(__file__))},
        fixed_control_protocol='Ridge(alpha1)+train StandardScaler; HistGB(max_iter200,max_depth3,leaf20,lr.05,l2=1,random2044,early_stopFalse); no search; train only, dev diagnostic only',
        feature_protocol='Original named8 scores + within-query average tied ranks + within-query zscores + direct flag =25; no class/episode/GT inputs',
        interpretation='All fold0 development pilot; additional supervised controls, exploratory comparisons, no independent/blind method claim.',
        metric_protocol='Native model-size I/U and original-photo I/U reported separately using identical choices; oracle chooses on native candidate IoU.',
        uncertainty_protocol='Primary paired resampling over connected components of all-role test photo overlap; episode/class intervals are non-independent sensitivities. Models are not refit per bootstrap; no multiplicity correction.',
        controls={})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    def write():
        report['elapsed_s']=time.monotonic()-started
        tmp=a.out.with_suffix(a.out.suffix+'.tmp');tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(a.out)
    def emit(value):print(json.dumps(value,allow_nan=False),flush=True)
    if a.counterfactual_report:
        report['source_sha256'][str(a.counterfactual_report)]=hashlib.sha256(a.counterfactual_report.read_bytes()).hexdigest()
    write()
    try:
        rows,_,assets=load_records(a.file,a.max_bytes)
        apply_manifest(rows,a.manifest)
        for row in rows:
            validate_record(row)
        if a.counterfactual_report:
            rows,cf_audit,assets=append_counterfactual_training(rows,a.counterfactual_report,a.manifest,a.max_bytes,assets)
            report['counterfactual_append']=cf_audit
        for row in rows:
            if row['candidate_iou_metric']!='native_iou' or 'candidate_original_iu' not in row:
                raise ValueError('both native target and separately saved original-photo I/U required')
            scalar_features(row)
        groups,audit=purge_split(rows)
        verify_training_identity(status,a.file,a.manifest,audit,a.counterfactual_report)
        train,dev,test=groups['train'],groups['dev'],groups['test']
        if not train or not dev or not test:raise ValueError('same trained nonempty split required')
        report['split_audit']=audit
        report['supervision_counts']=dict(train_episodes=len(train),train_candidates=sum(len(r['candidate_masks']) for r in train),
            dev_episodes=len(dev),test_episodes=len(test))
        selections=learned_choices(status,test)
        for baseline in ('direct','native_naive','cached_naive','cached_ag4'):
            selections[baseline]=[candidate_mapping(row)[baseline] for row in test]
        # Diagnostic evaluator intervention only, after all legal baseline mappings freeze.
        selections['candidate_oracle']=[int(row['candidate_iou'].argmax()) for row in test]
        X=np.concatenate([scalar_features(row) for row in train])
        y=np.concatenate([row['candidate_iou'].numpy() for row in train])
        for name,model in fixed_models().items():
            emit(dict(event='control_fit_start',model=name,train_episodes=len(train),candidate_rows=len(y)))
            tick=time.monotonic();model.fit(X,y)
            dev_result=control_evaluate(model,dev)
            # All settings fixed; these diagnostics never change a setting or select a model.
            choices=[int(np.argmax(model.predict(scalar_features(row)))) for row in test]
            selections[name]=choices
            report['controls'][name]=dict(state='COMPLETED',fit_s=time.monotonic()-tick,
                train=control_evaluate(model,train),dev=dev_result,dev_used_for_tuning=False)
            emit(dict(event='control_fit_done',model=name,dev=dev_result,fit_s=report['controls'][name]['fit_s']));write()
        membership,components=connected_photo_components(test)
        report['test_component_audit']=dict(effective_components=len(components),episodes=len(test),classes=len({int(r['c']) for r in test}),
            component_sizes=[g['size'] for g in components],components=components,
            CI_is_weak=len(components)<10,warning='Few connected photo components: exploratory interval, weak evidence' if len(components)<10 else None)
        classes=np.array([int(r['c']) for r in test])
        for metric,key in [('native_model_resolution','candidate_iu'),('original_photo_resolution','candidate_original_iu')]:
            arrays={name:np.array([row[key][index].numpy() for row,index in zip(test,choices)]) for name,choices in selections.items()}
            report[metric]=paired_bootstrap(classes,arrays,membership,a.bootstrap)
        report['episode_selection_diagnostics']=selection_diagnostics(test,selections)
        report['records']=[dict(e=int(row['e']),c=int(row['c']),photo_component=int(membership[i]),
            choices={name:int(indices[i]) for name,indices in selections.items()},
            candidate_provenance=row['candidate_provenance'],
            native_candidate_iu=row['candidate_iu'].tolist(),original_candidate_iu=row['candidate_original_iu'].tolist()) for i,row in enumerate(test)]
        report['native_candidate_headroom_pp']=report['native_model_resolution']['candidate_oracle']['gain_vs_direct_pp']
        report['state']='COMPLETED';write()
        emit(dict(event='analysis_complete',effective_components=len(components),
            native_scores={name:value['class_miou'] for name,value in report['native_model_resolution'].items()},
            candidate_headroom_pp=report['native_candidate_headroom_pp']))
    except BaseException as ex:
        report.update(state='ERROR',error=repr(ex));write();raise


if __name__=='__main__':main()
