#!/usr/bin/env python3
"""Read frozen membership outputs and small source traces; no model/fitting.

Native grid transitions and model1024 refinement transitions are separate.
GT threshold-family diagnostics never select a deployed threshold or method.
"""
import argparse
import base64
import json
from pathlib import Path
import sys
import time
import zipfile
import zlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_rice_core import digest, task, counts, class_table, aggregate, paired
from analyze_native_angular import unpack_mask
from native_angular_experiment import query_score_audit

ARMS = ('foris_crf', 'native_identity', 'part2_native_direct', 'kernel_svm',
        'complete_frost', 'foris_stateful')
LEDGER = ('recovered_fn', 'lost_tp', 'added_fp', 'removed_fp')
REQUIRED_TRACE = ('part2_score', 'part3_score', 'part4_score', 'pre_refinement',
                  'post_refinement', 'coverage', 'truth_model')
EXTRA_GRID_EVIDENCE = ('part2_sf', 'part2_sbn', 'candidate_hard', 'candidate_vote',
                      'seed_prior', 'part3_candidate_vote', 'part3_seed_prior',
                      'part4_penalty', 'part4_cluster_delta')
ALLOWED_TRACE = set(REQUIRED_TRACE) | {'svm_grid_margin', 'frost_grid_posterior', 'frost_candidate'} | set(EXTRA_GRID_EVIDENCE)


def classification(prediction, truth, valid=None):
    p, y = np.asarray(prediction), np.asarray(truth)
    if p.shape != y.shape or p.dtype != np.bool_ or y.dtype != np.bool_:
        raise ValueError('Matching Boolean prediction/truth grids required')
    valid = np.ones(p.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != p.shape:
        raise ValueError('Validity grid shape mismatch')
    tp = int((p & y & valid).sum()); fp = int((p & ~y & valid).sum())
    fn = int((~p & y & valid).sum()); tn = int((~p & ~y & valid).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, I=tp, U=tp+fp+fn,
                iou=tp/max(tp+fp+fn, 1), tpr=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(fp+tn) if fp+tn else None,
                precision=tp/(tp+fp) if tp+fp else None)


def transition(before, after, truth):
    a, b, y = np.asarray(before), np.asarray(after), np.asarray(truth)
    if a.shape != b.shape or a.shape != y.shape:
        raise ValueError('No cross-resolution transition ledger is permitted')
    return dict(recovered_fn=int((~a & b & y).sum()), lost_tp=int((a & ~b & y).sum()),
                added_fp=int((~a & b & ~y).sum()), removed_fp=int((a & ~b & ~y).sum()))


def source_threshold(score):
    s = np.asarray(score)
    if s.ndim != 2 or not np.isfinite(s).all():
        raise ValueError('Finite original native grid score required')
    low, high = float(s.min()), float(s.max())
    return low + .5 * max(high-low, 1e-6)


def threshold_family_diagnostic(score, coverage, actual_prediction):
    """Exact max-IoU threshold family on coverage>.5 hard PATCH labels.

    Includes empty/all-positive predictions and admits only complete equal-
    score groups. Its upper bound applies ONLY to this grid threshold family;
    it is neither a full-image oracle nor a deployable threshold calibration.
    """
    s, coverage = np.asarray(score, dtype=np.float64), np.asarray(coverage)
    if s.shape != coverage.shape or not np.isfinite(s).all():
        raise ValueError('Finite matching score/coverage grid required')
    y = (coverage > .5).ravel()
    order = np.argsort(-s.ravel(), kind='stable')
    sorted_scores, sorted_y = s.ravel()[order], y[order]
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])+1
    selected = np.r_[0, ends]
    intersections = np.r_[0, np.cumsum(sorted_y, dtype=np.int64)[ends-1]]
    unions = int(y.sum()) + selected - intersections
    values = intersections/np.maximum(unions, 1)
    best = int(values.argmax())
    prefix = int(selected[best])
    lower = classification(np.asarray(actual_prediction, dtype=bool), y.reshape(s.shape))['iou']
    upper = float(values[best])
    if lower > upper+1e-12:
        raise ValueError('Actual fixed score-threshold mask is outside its declared threshold-family bound')
    return dict(state='GT_GRID_THRESHOLD_FAMILY_DIAGNOSTIC_ONLY', grid=list(s.shape),
                truth_rule='coverage>0.5, one hard label per grid patch; mixed patches included',
                ties='whole equal-score groups only', evaluated_prefixes=len(selected),
                best_I=int(intersections[best]), best_U=int(unions[best]), selected_patches=prefix,
                lower_bound_actual_fixed_grid_iou=lower, upper_bound_threshold_family_grid_iou=upper,
                threshold_interval=dict(selected_min_score=float(sorted_scores[prefix-1]) if prefix else None,
                    rejected_max_score=float(sorted_scores[prefix]) if prefix < len(y) else None,
                    rule='score>threshold; threshold>=rejected_max and <selected_min when both exist'),
                not_full_mask_oracle=True, no_GT_threshold_deployed=True)


def score_stage(score, coverage, threshold, rule):
    s = np.asarray(score)
    if s.shape != coverage.shape or not np.isfinite(s).all() or not np.isfinite(threshold):
        raise ValueError('Finite compatible grid score and fixed threshold required')
    prediction = s > threshold
    hard_truth = coverage > .5
    pure = (coverage >= .9) | (coverage <= .1)
    roc = query_score_audit(s, coverage)
    if roc['state'] == 'SCORED':
        positive, negative = roc['positive_patches'], roc['negative_patches']
        tp = roc['actual_tpr']*positive; fp = roc['fpr_at_at_least_95tpr']*negative
        roc.update(pure_foreground_fraction=positive/(positive+negative),
                   diagnostic_FP_per_foreground=fp/positive,
                   diagnostic95_precision=tp/(tp+fp) if tp+fp else None,
                   diagnostic95_grid_pure_iou=tp/(positive+fp),
                   diagnostic_operating_point_not_deployable=True)
    return dict(state='GRID_SCORE_ANALYZED', grid=list(s.shape), threshold=float(threshold),
                decision_rule=rule, score_ROC_on_pure_patches=roc,
                actual_grid_all_hard_patch_metrics=classification(prediction, hard_truth),
                actual_grid_pure_patch_metrics=classification(prediction, hard_truth, pure),
                threshold_family_diagnostic=threshold_family_diagnostic(s, coverage, prediction)), prediction


def _trace_path(row, directory):
    expected = 'trace_%d_%d_%d.npz' % task(row)
    declared = Path(row.get('trace_archive', ''))
    if declared.name != expected:
        raise ValueError('Trace filename must encode exact frozen fold/e/class identity')
    path = declared if declared.is_absolute() else directory / declared
    if path.resolve().parent != directory.resolve():
        raise ValueError('Trace must be in the same folder as report; no arbitrary asset reads')
    return path


def load_trace(path):
    # Never decompress raw features or unknown arrays. A legal trace is about
    # 3MB of Boolean masks plus tiny score grids, independent of DINO dimension.
    with zipfile.ZipFile(path) as zipped:
        if sum(item.file_size for item in zipped.infolist()) > 16*1024*1024:
            raise ValueError('Trace exceeds the declared small-map/mask CPU budget')
    with np.load(path, allow_pickle=False) as archive:
        if not set(archive.files).issubset(ALLOWED_TRACE) or not set(REQUIRED_TRACE).issubset(archive.files):
            raise ValueError('Missing required trace arrays or undeclared feature payload')
        return {key: archive[key] for key in archive.files}


def trace_analysis(data, row, native_stateful_mask, evaluator_mask):
    if not set(REQUIRED_TRACE).issubset(data) or not set(data).issubset(ALLOWED_TRACE):
        raise ValueError('Trace schema mismatch')
    coverage = np.asarray(data['coverage'])
    if coverage.shape != (64,64) or not np.isfinite(coverage).all() or ((coverage < 0) | (coverage > 1)).any():
        raise ValueError('Original-query area coverage must be finite lawful64x64, evaluation only')
    extra_evidence = []
    for key in EXTRA_GRID_EVIDENCE:
        if key not in data:
            continue
        value = np.asarray(data[key])
        if value.shape != coverage.shape or not np.isfinite(value).all():
            raise ValueError('Native small-grid evidence must be finite64x64: '+key)
        if key == 'candidate_hard' and value.dtype != np.bool_:
            raise ValueError('Native candidate_hard must be a Boolean grid')
        extra_evidence.append(key)
    truth = np.asarray(data['truth_model'])
    masks = {key: np.asarray(data[key]) for key in ('pre_refinement','post_refinement')}
    if truth.shape != (1024,1024) or truth.dtype != np.bool_:
        raise ValueError('Boolean model1024 truth required')
    for value in masks.values():
        if value.shape != truth.shape or value.dtype != np.bool_:
            raise ValueError('Boolean model1024 refinement mask required')
    if not np.array_equal(truth,evaluator_mask) or not np.array_equal(masks['post_refinement'],native_stateful_mask):
        raise ValueError('Trace truth/post mask differs from frozen evaluator/stateful source output')
    stages, grid_masks = {}, {}
    reported = row.get('native_binarization', {}).get('threshold_in_source_score_units')
    if reported is None or not np.isfinite(reported):
        raise ValueError('Actual source Part4 binarization threshold must be recorded')
    for part in ('part2','part3','part4'):
        score = np.asarray(data[part+'_score'])
        threshold = source_threshold(score)
        if part == 'part4':
            # Native FP32 arithmetic can round the scalar expression slightly.
            tolerance = 8*np.finfo(score.dtype if np.issubdtype(score.dtype,np.floating) else np.float64).eps*max(1.,abs(threshold))
            if abs(float(reported)-threshold) > tolerance:
                raise ValueError('Recorded source threshold disagrees with source min+.5*max(range,1e-6)')
            threshold = float(reported)
        stages[part], grid_masks[part] = score_stage(score,coverage,threshold,
            'source min+.5*max(max-min,1e-6), strict greater; grid probe before upsampling')
    svma = row.get('svm_audit', {})
    if 'svm_grid_margin' in data:
        if not (svma.get('trained') is True or svma.get('state') in ('REFERENCE_KERNEL_SVM_FITTED','FITTED','COMPLETED')):
            raise ValueError('A nonfitted SVM fallback cannot supply a scorer ROC')
        stages['kernel_svm'], _ = score_stage(data['svm_grid_margin'],coverage,0.,'complete SVM signed margin>0')
    else:
        if not svma.get('state') or svma.get('trained') is True or svma.get('state') == 'REFERENCE_KERNEL_SVM_FITTED':
            raise ValueError('Missing SVM score requires an explicit source fallback state')
        stages['kernel_svm'] = dict(state='SOURCE_FALLBACK_NO_SCORER_ROC',source_audit=svma)
    frosta = row.get('frost_audit', {})
    if 'frost_grid_posterior' in data:
        if frosta.get('continuous_density_available') is not True:
            raise ValueError('FROST continuous map requires an explicit availability audit')
        log_density_ratio = np.asarray(data['frost_grid_posterior'])
        # Historical NPZ field name is retained, but source kde_log_posterior
        # returns log pFG-log pBG, NOT a probability or sigmoid output.
        threshold = frosta.get('threshold')
        if threshold is None or not np.isfinite(threshold) or float(threshold) != 0.:
            raise ValueError('FROST source constructor contract requires fixed signed-LLR threshold0')
        stages['complete_frost_density'], frost_mask = score_stage(log_density_ratio,coverage,float(threshold),
            'official source smoothed signed log-density ratio>tau, fixed tau=0; continuous grid probe')
        stages['complete_frost_density']['score_units'] = 'signed_log_pFG_minus_log_pBG_unbounded'
        if 'frost_candidate' in data:
            candidate = np.asarray(data['frost_candidate'])
            if candidate.shape != coverage.shape or candidate.dtype != np.bool_:
                raise ValueError('FROST Boolean candidate grid required')
            stages['frost_candidate_grid'] = dict(candidate_metrics=classification(candidate,coverage>.5),
                signed_LLR_zero_intersection_metrics=classification(frost_mask & candidate,coverage>.5),
                scope='Descriptive candidate intersection; not presumed identical to final official source mask')
    else:
        if frosta.get('continuous_density_available') is not False or not frosta.get('state'):
            raise ValueError('Missing FROST continuous map requires explicit source fallback')
        stages['complete_frost_density'] = dict(state='SOURCE_FALLBACK_NO_SCORER_ROC',source_audit=frosta)
        if 'frost_candidate' in data:
            candidate = np.asarray(data['frost_candidate'])
            if candidate.shape != coverage.shape or candidate.dtype != np.bool_:
                raise ValueError('FROST Boolean candidate grid required')
            stages['frost_candidate_grid'] = dict(candidate_metrics=classification(candidate,coverage>.5),
                scope='Official-source fallback candidate only; no continuous density ROC')
    return dict(trace_source_arm='foris_stateful', stateful_replay_exact=True,
                native_source_context='public set_reference/set_target/segment; RGB/position seed context enabled',
                grid_stages=stages, grid_transitions={
                    'part2_to_part3':transition(grid_masks['part2'],grid_masks['part3'],coverage>.5),
                    'part3_to_part4':transition(grid_masks['part3'],grid_masks['part4'],coverage>.5)},
                grid_transition_units='64x64 hard patch labels coverage>0.5; same resolution only',
                model1024_stages={key:classification(value,truth) for key,value in masks.items()},
                model1024_transition=transition(masks['pre_refinement'],masks['post_refinement'],truth),
                model1024_transition_units='1024x1024 pixels; source refinement pre-to-post only',
                grid_to_model_transition_computed=False,
                source_small_grid_evidence_fields_verified=extra_evidence,
                source_threshold_roundoff_scope='Recorded1024 masks authoritative; grid scalar comparison subject to original floating arithmetic')


def analyze(report, directory, manifest=None, report_sha=None, manifest_sha=None, trace_reader=None):
    if report.get('state') != 'COMPLETED' or tuple(report.get('arms', [])) != ARMS:
        raise ValueError('Completed frozen six-arm membership report required')
    if report.get('query_GT_used_in_prediction',False) is not False:
        raise ValueError('Query GT must not enter prediction')
    rows = report.get('records', [])
    if len(rows) != 40 or len({task(row) for row in rows}) != 40:
        raise ValueError('Exactly forty unique records required')
    if any(sum(row['fold'] == fold for row in rows) != 10 for fold in range(4)):
        raise ValueError('Exactly ten records per fold0..3 required')
    expected = None
    if manifest is not None:
        if manifest.get('state') != 'PREPARED_ASSETS' or tuple(manifest.get('arms',[])) != ARMS:
            raise ValueError('Frozen six-arm prepared manifest required')
        expected = {task(row):row for row in manifest['frozen_episodes']}
        if len(expected) != 40 or set(expected) != {task(row) for row in rows}:
            raise ValueError('Frozen task identities differ')
        if manifest_sha and report.get('manifest_sha256') != manifest_sha:
            raise ValueError('Report manifest digest differs from supplied frozen manifest')
        declared = manifest.get('source_hashes',{})
        observed = report.get('source_hashes',{})
        if not declared or not observed or any(observed.get(k) != v for k,v in declared.items()):
            raise ValueError('Report source hashes do not match prepared source provenance')
    directory = Path(directory)
    trace_reader = load_trace if trace_reader is None else trace_reader
    trajectories, photo_tasks, harm = [], {}, {arm:{key:0 for key in LEDGER} for arm in ARMS}
    for row in rows:
        if row.get('query_GT_opened_after_all_predictions') is not True:
            raise ValueError('Every record must freeze all predictions before query GT opens')
        if row.get('replay_identity_exact') is not True or row.get('stateful_replay_exact') is not True:
            raise ValueError('Both historical and stateful no-op replays must assert exactness')
        if row.get('trace_source_arm') != 'foris_stateful':
            raise ValueError('Source trace must explicitly describe the stateful public context')
        for role in ('support','query'):
            if not row.get(role) or expected is not None and row[role] != expected[task(row)][role]:
                raise ValueError('Missing or mismatched frozen all-role photo')
            photo_tasks.setdefault(row[role],set()).add(task(row))
        for field in ('iu','original_iu'):
            if set(row.get(field,{})) != set(ARMS):
                raise ValueError('Missing/extra scored complete arm')
            for arm in ARMS:counts(row[field][arm])
            if row[field]['foris_crf'] != row[field]['native_identity']:
                raise ValueError('Historical identity counts differ')
        bits = row.get('prediction_bits',{})
        if set(bits) != set(ARMS) or not row.get('evaluator_bits'):
            raise ValueError('All frozen packed predictions and model GT must be present')
        truth = unpack_mask(row['evaluator_bits']).astype(bool)
        predictions = {arm:unpack_mask(bits[arm]).astype(bool) for arm in ARMS}
        if truth.shape != (1024,1024) or any(p.shape != truth.shape for p in predictions.values()):
            raise ValueError('Frozen model masks must all be1024x1024')
        if not np.array_equal(predictions['foris_crf'],predictions['native_identity']):
            raise ValueError('Packed historical native identity masks differ, even if I/U agree')
        for arm,prediction in predictions.items():
            measured = classification(prediction,truth)
            if row['iu'][arm] != [measured['I'],measured['U']]:
                raise ValueError('Frozen model I/U differs from packed predictions')
            ledger = row.get('ledger',{}).get(arm)
            computed = transition(predictions['foris_crf'],prediction,truth)
            if ledger != computed or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in computed.values()):
                raise ValueError('Frozen method harm ledger differs from packed predictions')
            for key in LEDGER:harm[arm][key]+=computed[key]
        path = _trace_path(row,directory)
        if trace_reader is load_trace:
            if not row.get('trace_sha256') or digest(path)!=row['trace_sha256']:
                raise ValueError('Frozen small trace checksum missing or different')
        data = trace_reader(path)
        if 'svm_grid_margin' not in data and not np.array_equal(predictions['kernel_svm'],predictions['foris_crf']):
            raise ValueError('Declared missing-score SVM fallback must be exact historical native output')
        trajectories.append(dict(fold=row['fold'],e=row['e'],c=row['c'],trace_archive=path.name,
                                 **trace_analysis(data,row,predictions['foris_stateful'],truth)))
    scores, comparisons = {}, {}
    for field in ('iu','original_iu'):
        scores[field] = {}
        for arm in ARMS:
            table, totals = class_table(rows,arm,field)
            scores[field][arm] = dict(**aggregate(table), per_class=[dict(fold=k[0],c=k[1],
                I=int(totals[k][0]),U=int(totals[k][1]),iou=table[k]) for k in sorted(table)])
        comparisons[field] = {arm+' vs '+base:paired(rows,arm,base,field,repetitions=2000,seed=2053)
                              for base in ('foris_crf','foris_stateful') for arm in ARMS if arm != base}
    repeated = [dict(photo=p,tasks=[list(k) for k in sorted(tasks)])
                for p,tasks in sorted(photo_tasks.items()) if len(tasks)>1]
    return dict(state='CPU_NATIVE_MEMBERSHIP_ANALYSIS',records=40,arms=list(ARMS),
                report_sha256=report_sha,manifest_sha256=manifest_sha,
                manifest_IDs_and_source_hashes_verified=manifest is not None,
                scores=scores,comparisons=comparisons,harm_ledger_model1024_vs_historical_native=harm,
                native_identity=dict(packed_mask_exact_checks=40,runtime_replay_assertions=40),
                stateful_identity=dict(runtime_replay_assertions=40,
                    trace_post_matches_frozen_stateful_masks=40,
                    qualification='Stateful replay assertion is runtime evidence; no separate packed replay arm stored'),
                primary_trace_context='foris_stateful; not equated with historical direct-predict context',
                descriptive_source_trajectories=trajectories,
                model_and_original_resolution_separate=True,grid_and_model_transitions_separate=True,
                GT_diagnostic_thresholds_never_deployed=True,
                repeated_all_role_photos=repeated,independent_class_bootstrap_photo_warning=bool(repeated),
                scope='Reused forty development episodes; paired intervals descriptive, not independent validation or full benchmark')


def self_check():
    begin = time.monotonic()
    checks = []
    tied = np.ones((2,2)); labels=np.array([[1.,0.],[0.,0.]])
    diagnostic=threshold_family_diagnostic(tied,labels,np.zeros((2,2),dtype=bool))
    assert diagnostic['evaluated_prefixes']==2 and diagnostic['upper_bound_threshold_family_grid_iou']==.25
    checks.append('oracle_whole_tie_groups')
    rng=np.random.default_rng(2071)
    for _ in range(20):
        s=rng.integers(-2,3,(3,4)).astype(float); c=rng.integers(0,2,(3,4)).astype(float)
        d=threshold_family_diagnostic(s,c,s>source_threshold(s))
        brute=max(classification(s>threshold,c>.5)['iou'] for threshold in np.r_[np.unique(s),s.min()-1])
        assert d['upper_bound_threshold_family_grid_iou']==brute
    checks.append('prefix_oracle_matches_brute_thresholds')
    rare=np.full((20,50),.1);rare[0,0]=.2;rare[-1,-1]=-1
    c=np.zeros_like(rare);c[0,0]=1
    stage,_=score_stage(rare,c,source_threshold(rare),'source')
    assert stage['score_ROC_on_pure_patches']['auc']==1
    assert stage['actual_grid_all_hard_patch_metrics']['iou']==1/999
    checks.append('perfect_auc_rare_foreground_low_actual_iou')
    tiny=np.full((2,2),5.)
    assert source_threshold(tiny)==5.+.5e-6 and not (tiny>source_threshold(tiny)).any()
    checks.append('native_degenerate_floor_not_midpoint')
    try:transition(np.zeros((2,2),bool),np.zeros((3,3),bool),np.zeros((2,2),bool))
    except ValueError:pass
    else:raise AssertionError('cross-resolution ledger accepted')
    checks.append('cross_resolution_ledger_rejected')
    grid=np.zeros((64,64),dtype=bool);grid[:,:32]=True
    truth=np.repeat(np.repeat(grid,16,0),16,1)
    def packed(mask):return dict(shape=list(mask.shape),codec='zlib_np_packbits_big',
        data=base64.b64encode(zlib.compress(np.packbits(mask.ravel()).tobytes())).decode())
    bits=packed(truth)
    trace=dict(part2_score=grid.astype(float),part3_score=grid.astype(float),part4_score=grid.astype(float),
               coverage=grid.astype(float),truth_model=truth,pre_refinement=truth,post_refinement=truth,
               svm_grid_margin=2*grid.astype(float)-1,frost_grid_posterior=np.where(grid,.5,-2.),
               frost_candidate=np.ones((64,64),bool),
               **{key:(grid.copy() if key=='candidate_hard' else grid.astype(float)) for key in EXTRA_GRID_EVIDENCE})
    rows=[]
    for fold in range(4):
        for e in range(10):
            rows.append(dict(fold=fold,e=e,c=4*e+fold,support='s%d_%d'%(fold,e),query='q%d_%d'%(fold,e),
                trace_archive='trace_%d_%d_%d.npz'%(fold,e,4*e+fold),trace_source_arm='foris_stateful',
                replay_identity_exact=True,stateful_replay_exact=True,query_GT_opened_after_all_predictions=True,
                native_binarization={'threshold_in_source_score_units':.5},
                svm_audit={'state':'REFERENCE_KERNEL_SVM_FITTED','trained':True},
                frost_audit={'state':'SCORED','continuous_density_available':True,'threshold':0.},
                iu={a:[int(truth.sum()),int(truth.sum())] for a in ARMS},original_iu={a:[2,4] for a in ARMS},
                ledger={a:{k:0 for k in LEDGER} for a in ARMS},
                prediction_bits={a:bits for a in ARMS},evaluator_bits=bits))
    report=dict(state='COMPLETED',records=rows,arms=list(ARMS),manifest_sha256='synthetic',source_hashes={'source':'fixture'})
    manifest=dict(state='PREPARED_ASSETS',arms=list(ARMS),frozen_episodes=[dict(r) for r in rows],source_hashes={'source':'fixture'})
    result=analyze(report,Path('/tmp/membership_fixture'),manifest,manifest_sha='synthetic',trace_reader=lambda p:trace)
    assert result['scores']['iu']['foris_stateful']['fold_mean_class_miou']==100
    assert result['scores']['original_iu']['foris_stateful']['fold_mean_class_miou']==50
    assert result['native_identity']['packed_mask_exact_checks']==40
    assert result['comparisons']['iu']['kernel_svm vs foris_stateful']['exploratory_delta95_pp']==[0.,0.]
    frost_stage=result['descriptive_source_trajectories'][0]['grid_stages']['complete_frost_density']
    assert frost_stage['threshold']==0. and frost_stage['actual_grid_all_hard_patch_metrics']['iou']==1.
    assert frost_stage['score_units']=='signed_log_pFG_minus_log_pBG_unbounded'
    assert set(result['descriptive_source_trajectories'][0]['source_small_grid_evidence_fields_verified'])==set(EXTRA_GRID_EVIDENCE)
    # A probability-style threshold would exclude these exact .5 foreground
    # values under strict>; that must never pass as the official source rule.
    wrong_row={**rows[0],'frost_audit':{'state':'SCORED','continuous_density_available':True,'threshold':.5}}
    try:trace_analysis(trace,wrong_row,truth,truth)
    except ValueError:pass
    else:raise AssertionError('probability0.5 threshold substituted for signed source LLR0')
    checks.append('forty_trace_packed_identity_class_sumIU_dual_baseline_bootstrap')
    for key,value in [('query_GT_opened_after_all_predictions',False),('trace_source_arm','foris_crf'),('stateful_replay_exact',False)]:
        saved=rows[0][key];rows[0][key]=value
        try:analyze(report,Path('/tmp/membership_fixture'),trace_reader=lambda p:trace)
        except ValueError:pass
        else:raise AssertionError('illegal source/GT flag accepted')
        rows[0][key]=saved
    saved=report['source_hashes'];report['source_hashes']={'source':'different'}
    try:analyze(report,Path('/tmp/membership_fixture'),manifest,manifest_sha='synthetic',trace_reader=lambda p:trace)
    except ValueError:pass
    else:raise AssertionError('different source hashes accepted')
    report['source_hashes']=saved
    saved=rows[0]['query'];rows[0]['query']='different_photo'
    try:analyze(report,Path('/tmp/membership_fixture'),manifest,manifest_sha='synthetic',trace_reader=lambda p:trace)
    except ValueError:pass
    else:raise AssertionError('different frozen role photo accepted')
    rows[0]['query']=saved
    checks.append('GT_context_frozen_photos_and_source_hashes_enforced')
    saved=rows[0]['prediction_bits'];rows[0]['prediction_bits']={**saved,'native_identity':packed(~truth)}
    try:analyze(report,Path('/tmp/membership_fixture'),trace_reader=lambda p:trace)
    except ValueError:pass
    else:raise AssertionError('equal-count different replay accepted')
    rows[0]['prediction_bits']=saved
    checks.append('different_packed_replay_rejected')
    fallback_trace={k:v for k,v in trace.items() if k not in ('svm_grid_margin','frost_grid_posterior')}
    fallback_row={**rows[0], 'svm_audit':{'state':'SKLEARN_UNAVAILABLE','trained':False},
                  'frost_audit':{'state':'SOURCE_FALLBACK','continuous_density_available':False}}
    fallback=trace_analysis(fallback_trace,fallback_row,truth,truth)
    assert fallback['grid_stages']['kernel_svm']['state']=='SOURCE_FALLBACK_NO_SCORER_ROC'
    assert fallback['grid_stages']['complete_frost_density']['state']=='SOURCE_FALLBACK_NO_SCORER_ROC'
    checks.append('source_fallback_has_no_fabricated_scorer_ROC')
    malformed={**trace,'post_refinement':truth.astype(float)}
    try:trace_analysis(malformed,rows[0],truth,truth)
    except ValueError:pass
    else:raise AssertionError('floating mask silently accepted')
    for malformed in ({**trace,'candidate_hard':grid.astype(float)},
                      {**trace,'part4_penalty':np.ones((3,3))},
                      {**trace,'part2_sf':np.full((64,64),np.nan)},
                      {**trace,'raw_DINO_features':np.zeros((1,2,3))}):
        try:trace_analysis(malformed,rows[0],truth,truth)
        except ValueError:pass
        else:raise AssertionError('invalid evidence/rawfeature payload accepted')
    checks.append('shape_and_boolean_mask_contract')
    return dict(state='CPU_NATIVE_MEMBERSHIP_ANALYSIS_PASSED',checks=checks,elapsed_seconds=time.monotonic()-begin,
                no_GPU_or_real_task=True,oracle_grid_only=True,source_contexts_not_equated=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--out',type=Path);parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.out and args.out.exists():raise ValueError('Preserve existing analyses; choose a fresh output')
    begin=time.monotonic()
    if args.self_check:
        result=self_check()
    else:
        if args.report is None or args.out is None:parser.error('--report and --out required')
        manifest=json.loads(args.manifest.read_text()) if args.manifest else None
        result=analyze(json.loads(args.report.read_text()),args.report.resolve().parent,manifest,
            digest(args.report),digest(args.manifest) if args.manifest else None)
        result['analysis_seconds']=time.monotonic()-begin
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(state=result['state'],records=result.get('records'),
                         checks=len(result.get('checks',[])),elapsed_seconds=time.monotonic()-begin)))


if __name__=='__main__':main()
