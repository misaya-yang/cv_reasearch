"""Recover historical Strong S stages from saved masks and selections only.

No encoder, feature reads, FoRIS/CRF, new method or final mask is produced.
First verify every historical output by deterministic reconstruction; then
read query labels for attribution. Labels never change selections.
"""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.ndimage import label
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/local/strong_stage600_20261011'
ALIGN = ROOT / 'evidence/local/strong600_conflict_alignment_20261011'
OLD = ROOT / 'evidence/local/task_conflict_execution6200_20261011'
sys.path.insert(0, str(ROOT / 'scripts'))
from run_m4_baselines import render

ARMS = ('component.query_context', 'component.global_agreement_budget')
WINDOWS = [[x, y, x+512, y+512] for y in (0, 256, 512) for x in (0, 256, 512)]
COVER = np.zeros((1024, 1024), np.uint8)
for x0, y0, x1, y1 in WINDOWS:
    COVER[y0:y1, x0:x1] += 1


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def helper_module():
    path = ROOT / 'evidence/local/reference_background_conflict6200_20261011/score_and_verify.py'
    spec = importlib.util.spec_from_file_location('saved_stage_scorer', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def inventory():
    start = time.monotonic()
    assert not (OUT / 'protocol.json').exists(), 'Prepared already; do not replace inventory'
    OUT.mkdir(parents=True, exist_ok=True)
    members, bindings, inputs = read(ALIGN/'membership.json'), lines(OLD/'score_bindings.jsonl'), lines(OLD/'inputs.jsonl')
    assert len(members) == 600
    common = [ALIGN/'membership.json', OLD/'score_bindings.jsonl', OLD/'inputs.jsonl', Path(__file__), ROOT/'scripts/run_m4_baselines.py', ROOT/'evidence/local/reference_background_conflict6200_20261011/score_and_verify.py']
    cached, records, hashes = {}, [], {str(p): sha(p) for p in common}
    for i, member in enumerate(members):
        region = Path(member['old_region']['old_region_path'])
        cohort = region.parent.parent
        source = (ROOT.parent/'cv_data/a/lvis_fixedwindows200_20261009') if cohort.name == 'lvis_component_presence200_20261009' else cohort.parent
        if str(cohort) not in cached:
            cr = {r['episode_id']: r for r in lines(cohort/'inference.jsonl')}
            sr = {r['episode_id']: r for r in lines(source/'inference.jsonl')}
            manifests = {r['episode_id']: r for r in read(source/'manifest.json')}
            cached[str(cohort)] = (cr, sr, manifests)
            for p in (cohort/'inference.jsonl', source/'inference.jsonl', source/'manifest.json', cohort/'sealed.json', source/'sealed.json'):
                hashes[str(p)] = sha(p)
        cr, sr, manifests = cached[str(cohort)]
        eid = member['episode_id']
        source_row, region_row, manifest = sr[eid], cr[eid], manifests[eid]
        binding, input_row = bindings[member['source_index']], inputs[member['source_index']]
        assert binding['row']['episode_id'] == input_row['row']['episode_id'] == eid
        assert binding['binding_index'] == member['binding_index']
        for key in ('reference_rgb_hash', 'query_rgb_hash', 'reference_mask_hash'):
            assert member[key] == binding['row'][key] == input_row['row'][key] == manifest[key]
        assert region_row['filename'] == source_row['filename'] == region.name
        assert source_row['window_boxes'] == WINDOWS
        assert region_row['query_window_raw_keys'] == source_row['raw_keys']
        assert region_row['query_full_raw_key'] == input_row['raw']['query_full']['key']
        assert len(region_row['regions']) == 9
        assert sha(region) == member['old_region']['old_region_current_sha256']
        paths = {'region': region, 'windows': source/'window_masks'/region.name, 'source_prediction': source/'predictions'/region.name, 'B': Path(binding['old_final_masks']['B']['path'])}
        assets = {k: {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for k,p in paths.items()}
        assert assets['B']['sha256'] == binding['old_final_masks']['B']['sha256']
        records.append(dict(index=i, source_index=member['source_index'], episode_id=eid, cohort=str(cohort), source=str(source), assets=assets,
                            region_receipt_sha256=canonical_sha(region_row), source_receipt_sha256=canonical_sha(source_row), reference_box=source_row['reference_box']))
    for p in (ROOT.parent/'cv_data/a/lvis_component_presence200_20261009/run_component_presence.py', ROOT.parent/'cv_data/a/lvis_fixedwindows200_20261009/run_windows.py', ROOT.parent/'cv_data/a/lvis_component_replication200_20261009/run_replication.py', ROOT.parent/'cv_data/a/lvis_component_completion600_20261009/run_complete600.py'):
        hashes[str(p)] = sha(p)
    write(OUT/'inventory.json', dict(n=600, records=records, source_sha256=hashes, wall_seconds=time.monotonic()-start, query_GT_reads=0, feature_reads=0))
    write(OUT/'protocol.json', dict(state='FROZEN_POSTSEAL_SAVED_STRONG_STAGE_DIAGNOSIS', n=600, inventory_sha256=sha(OUT/'inventory.json'), source_sha256=hashes,
                                  gate='Use historical selections and strict margin>0; never refit or recalculate descriptors', fusion='2*votes>all_view_coverage; ties BG; historical float32 bilinear align_corners=False >.5',
                                  attribution='B&~S: no_any_native_support; positive_views_fail_raw_majority; raw_majority_but_gate_veto. Monotone stages, each independently rendered to each frame.',
                                  role='Necessary saved-mask counterfactual diagnosis only, not a new candidate or data-selected repair. Local observation includes reference crop, FoRIS and CRF; these are not separated.',
                                  constraints=dict(new_encoder=0, new_features=0, new_FoRIS=0, new_CRF=0, new_views=0, new_candidates=0, dense_writes=0)))
    print(json.dumps(dict(state='PREPARED600_ALL_ASSETS_BOUND_NO_GT', n=len(records), cohorts=dict(Counter(Path(r['cohort']).parent.name if Path(r['cohort']).name=='component' else Path(r['cohort']).name for r in records)), wall_seconds=time.monotonic()-start)))


def reconstruct(entry, region_receipt, helper, checks, target=None):
    path = entry['assets']['source_prediction']['path']
    with np.load(path, allow_pickle=False) as z:
        shape = tuple(map(int, z['original_hw']))
        base = helper.unpack(z, 'original/foris.crf', shape, checks)
        saved_native = helper.unpack(z, 'original/window.native', shape, checks)
    full = render(base, (1024, 1024))
    raw = np.zeros_like(COVER)
    votes = {arm: np.zeros_like(COVER) for arm in ARMS}
    components = []
    with np.load(entry['assets']['windows']['path'], allow_pickle=False) as z:
        for i, (box, receipt) in enumerate(zip(WINDOWS, region_receipt['regions'])):
            x0, y0, x1, y1 = box
            local = helper.unpack(z, 'window.native/'+str(i), (512,512), checks)
            ids,n = label(local, np.ones((3,3), dtype=bool))
            areas = np.bincount(ids.ravel(), minlength=n+1)[1:]
            overlap = np.bincount(ids.ravel(), weights=full[y0:y1,x0:x1].ravel(), minlength=n+1)[1:]
            agreement = overlap / np.maximum(areas+int(full[y0:y1,x0:x1].sum())-overlap, 1)
            assert receipt['window'] == i and receipt['components'] == n
            assert areas.tolist() == receipt['component_sizes']
            assert agreement.tolist() == receipt['global_agreement']
            accept = receipt['selections'][ARMS[0]]
            expected = [True]*n if region_receipt['missing_query_role'] else [m>0 for m in receipt['context_margin']]
            assert accept == expected
            order = sorted(range(n), key=lambda j: (-agreement[j], j))
            chosen = set(order[:sum(accept)])
            assert receipt['selections'][ARMS[1]] == [j in chosen for j in range(n)]
            checks['component_labels_areas_margin_budget'] += n+1
            raw[y0:y1,x0:x1] += local
            for arm in ARMS:
                votes[arm][y0:y1,x0:x1] += np.asarray([False]+receipt['selections'][arm], bool)[ids]
            if target is not None:
                tp = np.bincount(ids.ravel(), weights=target[y0:y1,x0:x1].ravel(), minlength=n+1)[1:].astype(np.int64)
                for j in range(n):
                    components.append(dict(window=i, component=j+1, area=int(areas[j]), TP=int(tp[j]), FP=int(areas[j]-tp[j]), margin=receipt['context_margin'][j], accepted=accept[j], global_agreement=receipt['global_agreement'][j]))
    stages = dict(any_native=raw>0, native_majority=2*raw>COVER, S=2*votes[ARMS[0]]>COVER, budget=2*votes[ARMS[1]]>COVER)
    assert not np.any(stages['native_majority'] & ~stages['any_native'])
    assert not np.any(stages['S'] & ~stages['native_majority'])
    assert np.array_equal(render(stages['native_majority'], shape), saved_native)
    with np.load(entry['assets']['region']['path'], allow_pickle=False) as z:
        assert tuple(z['original_hw']) == shape
        for arm, name in zip(ARMS, ('S','budget')):
            for frame, size in (('canvas',(1024,1024)),('original',shape)):
                expected = stages[name] if frame=='canvas' else render(stages[name],size)
                assert np.array_equal(expected, helper.unpack(z, frame+'/'+arm,size,checks)), (entry['index'],arm,frame)
                checks['historical_mask_bitwise_reconstruction'] += 1
    return shape, base, stages, raw, votes[ARMS[0]], components


def iu(pred, target):
    return dict(I=int((pred & target).sum()), U=int((pred | target).sum()))


def diagnose():
    start, cpu = time.monotonic(), time.process_time()
    assert not (OUT/'report.json').exists(), 'Closed diagnosis must not be repeated'
    torch.set_num_threads(2)
    protocol, inventory = read(OUT/'protocol.json'), read(OUT/'inventory.json')
    assert sha(OUT/'inventory.json') == protocol['inventory_sha256']
    for p,digest in protocol['source_sha256'].items():
        assert sha(p) == digest, p
    helper, checks = helper_module(), Counter()
    bindings = lines(OLD/'score_bindings.jsonl')
    regions = {cohort: {x['episode_id']:x for x in lines(Path(cohort)/'inference.jsonl')} for cohort in {r['cohort'] for r in inventory['records']}}
    # All 600 historical S/budget/native outputs reconstructed before any GT.
    for entry in inventory['records']:
        for asset in entry['assets'].values():
            assert sha(asset['path']) == asset['sha256'], asset['path']
            checks['pre_GT_old_asset_SHA256'] += 1
        receipt = regions[entry['cohort']][entry['episode_id']]
        assert canonical_sha(receipt) == entry['region_receipt_sha256']
        reconstruct(entry,receipt,helper,checks)
        if (entry['index']+1)%100==0:
            print(json.dumps(dict(stage='PRE_GT_SAVED_MASK_RECONSTRUCTION',done=entry['index']+1,wall_seconds=time.monotonic()-start)),flush=True)
    pregt_seconds = time.monotonic()-start
    write(OUT/'reconstruction_seal.json', dict(state='ALL600_OLD_S_BUDGET_NATIVE_BITWISE_RECONSTRUCTED_BEFORE_GT', n=600, query_GT_reads=0, protocol_sha256=sha(OUT/'protocol.json'), inventory_sha256=sha(OUT/'inventory.json'), checks=dict(checks), wall_seconds=pregt_seconds))
    totals = {frame: defaultdict(Counter) for frame in ('cli','original')}
    cases, cc_totals, requested_bytes = [], Counter(), 0
    with (OUT/'components.jsonl').open('w') as ledger:
        for entry in inventory['records']:
            row = bindings[entry['source_index']]['row']
            assert row['episode_id'] == entry['episode_id']
            gt = helper.truth(row,checks)
            cli_gt = helper.resize(gt,(1024,1024),'nearest') > .5
            shape, base, stages, raw, accepted, components = reconstruct(entry,regions[entry['cohort']][entry['episode_id']],helper,checks,cli_gt)
            ledger.write(json.dumps(dict(index=entry['index'], episode_id=row['episode_id'], components=components),allow_nan=False)+'\n')
            for cc in components:
                name = 'accepted' if cc['accepted'] else 'rejected'
                cc_totals[name+'_components'] += 1
                cc_totals[name+'_TP_view_pixels'] += cc['TP']
                cc_totals[name+'_FP_view_pixels'] += cc['FP']
                cc_totals[name+'_mixed_components'] += int(cc['TP']>0 and cc['FP']>0)
                cc_totals[name+'_GT_empty_components'] += int(cc['TP']==0)
                cc_totals[name+'_all_GT_components'] += int(cc['FP']==0)
            case = dict(index=entry['index'],source_index=entry['source_index'],episode_id=row['episode_id'],fold=row['fold'],loader_class_id=row['loader_class_id'],frames={})
            with np.load(entry['assets']['B']['path'], allow_pickle=False) as z:
                for frame, size in (('cli',(1024,1024)),('original',shape)):
                    truth = cli_gt if frame=='cli' else helper.resize(gt,size,'nearest')>.5
                    fields = stages if frame=='cli' else {k:render(v,size) for k,v in stages.items()}
                    bkey = bindings[entry['source_index']]['old_final_masks']['B'][frame+'_key']
                    b = helper.unpack(z,bkey,size,checks)
                    if frame=='original':
                        assert np.array_equal(b,base)
                    any_fg,native,strong = (fields[k] for k in ('any_native','native_majority','S'))
                    deleted = b & ~strong
                    partitions = dict(local_no_native_support=deleted & ~any_fg,
                                      positive_views_fail_raw_majority=deleted & any_fg & ~native,
                                      raw_majority_lost_by_region_gate=deleted & native)
                    stats = {name:dict(TP=int((mask & truth).sum()),FP=int((mask & ~truth).sum())) for name,mask in partitions.items()}
                    assert sum(x['TP'] for x in stats.values()) == int((deleted & truth).sum())
                    assert sum(x['FP'] for x in stats.values()) == int((deleted & ~truth).sum())
                    added = strong & ~b
                    edits = dict(add_TP=int((added & truth).sum()),add_FP=int((added & ~truth).sum()),delete_TP=int((deleted & truth).sum()),delete_FP=int((deleted & ~truth).sum()))
                    # Strict-majority counterfactual includes accepted tie and
                    # accepted positive minority without altering any gate.
                    accepted_any = accepted>0
                    accepted_tie = (2*accepted==COVER) & (accepted>0)
                    raw_tie = (2*raw==COVER) & (raw>0)
                    masks = dict(accepted_any=accepted_any,accepted_tie=accepted_tie,raw_tie=raw_tie)
                    if frame=='original':
                        masks = {k:render(v,size) for k,v in masks.items()}
                    details = {name:dict(B_deleted_TP=int((deleted & mask & truth).sum()),B_deleted_FP=int((deleted & mask & ~truth).sum()),all_FN=int((~strong & mask & truth).sum()),all_TN=int((~strong & mask & ~truth).sum())) for name,mask in masks.items()}
                    stage_iu = {k:iu(p,truth) for k,p in dict(B=b,**fields).items()}
                    gate_removed = native & ~strong
                    gate = dict(TP=int((gate_removed & truth).sum()),FP=int((gate_removed & ~truth).sum()),inside_B_TP=int((gate_removed & b & truth).sum()),inside_B_FP=int((gate_removed & b & ~truth).sum()))
                    case['frames'][frame] = dict(S_vs_B=edits,B_deleted_partition=stats,strict_majority_details=details,stage_iu=stage_iu,gate_removed_from_native=gate)
                    for group, values in (('S_vs_B',edits),('gate_removed_from_native',gate)):
                        totals[frame][group].update(values)
                    for name, values in stats.items(): totals[frame]['partition/'+name].update(values)
                    for name, values in details.items(): totals[frame]['details/'+name].update(values)
            cases.append(case)
            if (entry['index']+1)%100==0:
                print(json.dumps(dict(stage='POSTSEAL_SAVED_STAGE_ATTRIBUTION',done=entry['index']+1,wall_seconds=time.monotonic()-start)),flush=True)
    # One episode per observed fold/class; official missing classes still have
    # zero contribution and do not change the ranking of these same 600.
    expected = helper.expected_classes()['lvis']
    summary = {}
    for frame in ('cli','original'):
        metrics = {}
        for arm in ('B','any_native','native_majority','S','budget'):
            grouped = defaultdict(lambda:[0,0])
            for case in cases:
                key = (str(case['fold']),str(case['loader_class_id']))
                a = case['frames'][frame]['stage_iu'][arm]
                grouped[key][0] += a['I']; grouped[key][1] += a['U']
            assert len(grouped)==600 and all(k[1] in expected[k[0]] for k in grouped)
            metrics[arm] = dict(observed_600_mIoU_percent=100*sum(i/max(u,1) for i,u in grouped.values())/600,
                                official_920_zero_missing_mIoU_percent=100*sum(i/max(u,1) for i,u in grouped.values())/920)
        summary[frame] = dict(totals={k:dict(v) for k,v in totals[frame].items()},metrics=metrics)
    write(OUT/'episodes.json', cases)
    # Counts distinguish payload decodes from SHA reads, not physical disk I/O.
    requested_bytes = sum(2*sum(e['assets'][k]['bytes'] for k in ('windows','source_prediction','region'))+e['assets']['B']['bytes'] for e in inventory['records'])
    mask_sha_bytes = sum(3*e['assets']['region']['bytes']+2*sum(e['assets'][k]['bytes'] for k in ('windows','source_prediction','B')) for e in inventory['records'])
    report = dict(state='COMPLETE600_HISTORICAL_STRONG_STAGE_ATTRIBUTION',n=600,summary=summary,components=dict(cc_totals),checks=dict(checks),query_GT_reads=600,
                  wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  pre_GT_reconstruction_wall_seconds=pregt_seconds,mask_payload_container_bytes_requested=requested_bytes,mask_payload_decode_requests=4200,
                  inventory_and_diagnosis_mask_SHA_bytes_requested=mask_sha_bytes,inventory_and_diagnosis_mask_SHA_requests=5400,physical_IO_bytes=None,
                  resources=dict(new_encoder_calls=0,raw_feature_loads=0,new_FoRIS_calls=0,new_CRF_calls=0,new_predictions=0,new_candidate_arms=0,dense_archive_writes=0,MPS_calls=0),
                  protocol_sha256=sha(OUT/'protocol.json'),inventory_sha256=sha(OUT/'inventory.json'),seal_sha256=sha(OUT/'reconstruction_seal.json'),
                  scope='Post-seal LVIS600 decomposition. Saved local observation includes reference crop and complete window FoRIS/CRF; those steps cannot be separated here. Region-gate effects established by exact all-accept counterfactual at unchanged coverage; vote/gate errors can coexist. Connected regions are not instances. Diagnostic any-support is not a proposed method.')
    write(OUT/'report.json',report)
    print(json.dumps(dict(state=report['state'],wall_seconds=report['wall_seconds'],summary=summary,components=dict(cc_totals)),allow_nan=False),flush=True)


if __name__=='__main__':
    if sys.argv[1:] == ['prepare']:
        inventory()
    elif sys.argv[1:] == ['diagnose']:
        diagnose()
    else:
        raise SystemExit('Use prepare then diagnose, only once.')
