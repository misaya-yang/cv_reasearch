#!/usr/bin/env python3
"""FIXED ten-pair availability card; reuse first result, no reordered replacements.

Uses unchanged conditional_counterfactual_prepare.py separately for each of the
nine unmeasured pairs, with one persistent frozen DINO instance. Scientific K8,
P1 union, legal gold switch, native IU, original IU, .5 coverage and .1 regret
rules stay unchanged. A deterministic NEW card asks how many of the PRESELECTED
10 pairs are usable training data; not benchmark performance or a rule variant.
Continuation requires >=3/10 usable pairs across >=2 concept pairs. Any valid
extra-GT subset must subsequently be given equally to all learned controls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def summarize(pairs):
    valid=[r for r in pairs if r['diagnostic']['pair_teaches_prespecified_flip']]
    concepts=sorted(set(tuple(sorted(r['classes'])) for r in valid))
    return dict(observed_pairs=len(pairs),usable_pairs=len(valid),usable_pair_indices=[r['pair_index'] for r in valid],
                usable_fraction=len(valid)/len(pairs) if pairs else None,
                usable_concept_pairs=[list(x) for x in concepts],usable_concept_pair_count=len(concepts),
                coverage_failures=sum(not r['diagnostic']['checks']['both_oracles_ge_50'] for r in pairs),
                overlapping_best_set_failures=sum(not r['diagnostic']['checks']['optimal_sets_disjoint'] for r in pairs),
                cross_regret_failures=sum(not r['diagnostic']['checks']['both_cross_regrets_ge_10'] for r in pairs),
                continuation_eligible=len(pairs)==10 and len(valid)>=3 and len(concepts)>=2,
                criterion='Exactly fixed10, usable >=3 and concept-pair diversity >=2; individual oracle>=.5/disjoint best sets/cross regret>=.1 unchanged')


def verify_reused(first_report,first_pair,manifest_sha):
    if first_report['state']!='COMPLETED':raise ValueError('Reuse source must have completed its original interface check')
    if first_report['pair']!=first_pair:raise ValueError('Reuse source differs from the fixed FIRST pair')
    if first_report['source_manifest_sha256']!=manifest_sha:raise ValueError('Reuse source is not the new cohort')
    if not first_report['audit']['same_rgb_patch_features_exact']:raise ValueError('Reuse source has invalid RGB-feature interface')
    if len(first_report['records'])!=2:raise ValueError('Reuse source must have both gold task overlays')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--feasibility',default='/tmp/demo9_counterfactual_feasibility120_v2.json')
    ap.add_argument('--reuse-firstcompletedsource');ap.add_argument('--out')
    ap.add_argument('--prepared-root',default='/root/autodl-tmp/demo9');ap.add_argument('--limit',type=int,default=10)
    ap.add_argument('--max-bytes',type=int,default=500_000_000);ap.add_argument('--self-check',action='store_true')
    a=ap.parse_args()
    if a.self_check:self_check();return
    if a.limit!=10:ap.error('This availability card requires exactly the fixed10; no other sweep size')
    if not a.out or not a.reuse_firstcompletedsource:ap.error('--out and --reuse-firstcompletedsource are required')
    feasibility=json.loads(Path(a.feasibility).read_text())
    fixed=feasibility['minimal_recombined_examples']
    if len(fixed)!=10:raise ValueError('Need exactly the already preselected ten pairs')
    manifest_path=Path(feasibility['source_manifest']);manifest=json.loads(manifest_path.read_text())
    manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if feasibility['source_manifest_sha256']!=manifest_sha:raise ValueError('Cohort manifest drift')
    import conditional_counterfactual_prepare as single
    for pair in fixed:single.validate_pair(dict(minimal_recombined_examples=[pair]),manifest)
    reused_path=Path(a.reuse_firstcompletedsource)
    if reused_path.is_dir():reused_path=reused_path/'report.json'
    first=json.loads(reused_path.read_text());verify_reused(first,fixed[0],manifest_sha)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);report_path=out/'report.json'
    if report_path.exists():raise ValueError('Use a fresh own availability output; original diagnostic is never overwritten')
    report=dict(state='RUNNING',args=vars(a),dataset='COCO-20i',fold=0,split='new_train_only',fixed_pair_count=10,
        source_manifest_sha256=manifest_sha,source_feasibility_sha256=hashlib.sha256(Path(a.feasibility).read_bytes()).hexdigest(),
        scientific_kernel_sha256=hashlib.sha256(Path(single.__file__).read_bytes()).hexdigest(),
        fixed_pairs=fixed,pairs=[],records=[],reused_first_report=str(reused_path),
        contract='Same existing scientific single-pair kernel, fixed10 in original order; no replacements/reordering/query-GT pair selection. Shared full tensors once per RGB pair, same K8/P1/provenance/direct0 across its two task overlays.',
        four_lines=dict(assumption='The fixed eligible annotated training cohort supplies enough GOOD both-semantic flips to teach reference conditioning.',
            prediction='At least3/10 fixed pairs pass unchanged coverage>=.5, disjoint oracle best sets and both cross regrets>=.1, across>=2 concept pairs.',
            match='Freeze valid training-only subset, then supply identical extra GT to scalar/full-token controls; a later matched training experiment still needed.',
            mismatch='Do not train this small CF construct; report fixed10 coverage failure causes. This says nothing about all1061 eligible pairs; no replacement search or criterion relaxation.'),
        forbidden_model_metadata=first['forbidden_model_metadata'],
        inference_allowlist=first['inference_allowlist'],
        original_first_diagnostic_preserved=True)
    start=time.time();shared_paths=set()
    def save():
        report.update(elapsed_s=time.time()-start,summary=summarize(report['pairs']))
        report['total_shared_and_task_bytes']=sum(Path(p).stat().st_size for p in shared_paths)
        tmp=report_path.with_suffix('.tmp');tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(report_path)
    def append(index,completed,path,reused=False):
        pair=fixed[index]
        report['pairs'].append(dict(pair_index=index,classes=pair['classes'],report_path=str(path),
            reused=reused,diagnostic=completed['diagnostic'],audit=completed['audit'],elapsed_s=completed['elapsed_s']))
        for condition,entry in enumerate(completed['records']):
            # Records have globally unique audit IDs; payload source identities
            # remain unchanged/audit-only. A future reader must merge the shared
            # object and task overlay, then use only the explicit input allowlist.
            report['records'].append(dict(entry,dataset_index=2*index+condition,counterfactual_pair_index=index,
                reused_source=reused,usable_training_pair=completed['diagnostic']['pair_teaches_prespecified_flip']))
            shared_paths.add(entry['path']);shared_paths.add(entry['overlay_path'])
        save()
        print(f'pair {index+1}/10 reused={reused} oracle={completed["diagnostic"]["oracle_iou"]} usable={completed["diagnostic"]["pair_teaches_prespecified_flip"]} summary={report["summary"]}',flush=True)
    save();append(0,first,reused_path,reused=True)
    try:
        os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
        sys.path.insert(0,str(Path(a.prepared_root)/'scripts'));import _paths
        sys.path.insert(0,_paths.DEMO4)
        import icx.common as common
        original_builder=common.build_model;frozen=[None]
        def shared_builder(*args,**kwargs):
            if args or kwargs:raise ValueError('Frozen single-pair kernel must use unchanged default model parameters')
            if frozen[0] is None:frozen[0]=original_builder()
            return frozen[0]
        common.build_model=shared_builder
        original_argv=sys.argv[:]
        try:
            for index in range(1,10):
                # Each single-pair full object has measured size~43MB. Budget
                # reserve is conservative; existing first tensors are counted.
                remaining=a.max_bytes-report['total_shared_and_task_bytes']
                if remaining<45_000_000:raise RuntimeError('Frozen500MB total cap cannot accommodate another complete pair')
                pair_out=out/f'pair_{index:02d}';pair_feas=dict(feasibility,minimal_recombined_examples=[fixed[index]])
                input_path=out/f'feasibility_pair_{index:02d}.json';input_path.write_text(json.dumps(pair_feas))
                sys.argv=[str(Path(single.__file__)),'--feasibility',str(input_path),'--manifest',str(manifest_path),
                          '--prepared-root',a.prepared_root,'--out',str(pair_out),'--limit','1','--max-bytes',str(min(remaining,100_000_000))]
                single.main()
                completed_path=pair_out/'report.json';completed=json.loads(completed_path.read_text())
                if completed['state']!='COMPLETED':raise RuntimeError(f'Fixed pair{index} interface/generation failed; no replacement')
                if completed['pair']!=fixed[index]:raise AssertionError('Fixed pair order drifted')
                append(index,completed,completed_path)
                if report['total_shared_and_task_bytes']>a.max_bytes:raise RuntimeError('Total byte cap exceeded')
        finally:common.build_model=original_builder;sys.argv=original_argv
        report['state']='COMPLETED';save()
        report['frozen_valid_training_records']=[r for r in report['records'] if r['usable_training_pair']]
        report['next_action']='MATCHED_EXTRA_GT_TRAINING_CARD_REQUIRES_PRIMARY_DECISION' if report['summary']['continuation_eligible'] else 'DO_NOT_TRAIN_THIS_SMALL_CF_CONSTRUCT'
        save();print(json.dumps(dict(state=report['state'],elapsed_s=report['elapsed_s'],summary=report['summary'],total_bytes=report['total_shared_and_task_bytes'],next_action=report['next_action'])),flush=True)
    except BaseException as exc:
        report.update(state='ERROR',error=repr(exc),next_action='STOP_NO_REPLACEMENT_NO_CRITERION_CHANGE');save();raise


def self_check():
    from conditional_counterfactual_prepare import oracle_diagnostic
    fail=oracle_diagnostic([.27,0],[0,.68]);good=oracle_diagnostic([.8,.1],[.1,.9])
    rows=[dict(pair_index=i,classes=[0,4] if i<2 else [8,16],diagnostic=good if i<3 else fail) for i in range(10)]
    assert summarize(rows)['continuation_eligible']
    assert not summarize(rows[:9])['continuation_eligible']
    rows[2]['diagnostic']=fail;assert not summarize(rows)['continuation_eligible']
    rows[2]['diagnostic']=good;rows[2]['classes']=[0,4];assert not summarize(rows)['continuation_eligible']
    pair=dict(classes=[0,4]);first=dict(state='COMPLETED',pair=pair,source_manifest_sha256='same',audit=dict(same_rgb_patch_features_exact=True),records=[{},{}])
    verify_reused(first,pair,'same')
    try:verify_reused(first,dict(classes=[4,8]),'same');raise AssertionError('Reuse drift passed')
    except ValueError:pass
    print('CPU smoke passed: immutable first-pair reuse/schema, fixed10 denominator, unchanged per-pair gates and >=3/10 across >=2 concept-pair continuation, no replacement or reordering.')


if __name__=='__main__':main()
