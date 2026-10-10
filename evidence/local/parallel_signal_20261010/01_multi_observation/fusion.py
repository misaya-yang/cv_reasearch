"""One frozen minority-support restoration with two matched simple controls.

All new complete masks are sealed before this process opens query truth.
No model/source head/raw read. One CPU, no parameter or threshold search.
"""
import hashlib
import json
from pathlib import Path
import time
from collections import defaultdict
import numpy as np
import study as s

OUT = s.OUT/'fusion_v1'
ARMS = ('restore.accepted', 'restore.raw_support', 'restore.whole')


def point(records, expected=None):
    arms = sorted(records[0]['iu'])
    sums = defaultdict(lambda: np.zeros((len(arms), 2), np.int64))
    for r in records:
        sums[str(r['fold']), str(r['class_id'])] += [r['iu'][a] for a in arms]
    if expected is None:
        expected = {f: sorted(c for ff, c in sums if ff == f) for f in sorted({f for f, _ in sums})}
    folds = {}
    for f, classes in expected.items():
        values = []
        for c in classes:
            v = sums[str(f), str(c)]
            values.append(100*v[:, 0]/np.maximum(v[:, 1], 1))
        folds[str(f)] = dict(zip(arms, np.mean(values, axis=0).tolist()))
    return dict(n=len(records), observed_fold_classes=len({(r['fold'], r['class_id']) for r in records}),
                metric_fold_classes=sum(len(x) for x in expected.values()),
                miou={a: float(np.mean([v[a] for v in folds.values()])) for a in arms}, per_fold=folds)


def counts(mask, truth):
    tp = int((mask & truth).sum())
    fp = int((mask & ~truth).sum())
    fn = int((~mask & truth).sum())
    return dict(tp=tp, fp=fp, fn=fn, intersection=tp, union=tp+fp+fn)


def gross(before, after, truth):
    add, delete = after & ~before, before & ~after
    return dict(added_tp=int((add & truth).sum()), added_fp=int((add & ~truth).sum()),
                deleted_tp=int((delete & truth).sum()), deleted_fp=int((delete & ~truth).sum()))


def load_fields(rec):
    p = s.OUT/'reconstructed'/rec['filename']
    assert s.sha(p) == rec['sha']
    with np.load(p, allow_pickle=False) as z:
        raw, accepted = z['raw_votes'].copy(), z['accepted_votes'].copy()
        shape = tuple(z['original_hw'])
        f = dict(raw=raw, accepted=accepted, region=2*accepted>s.COVER,
                 unfiltered=2*raw>s.COVER, guide=s.unpack(z, 'guide', s.COVER.shape),
                 base_canvas=s.unpack(z, 'base_canvas', s.COVER.shape),
                 base_original=s.unpack(z, 'base_original', shape))
        if 'geometry' in z:
            f['geometry'] = s.unpack(z, 'geometry', s.COVER.shape)
    return f, shape


def new_masks(f):
    return {'restore.accepted': f['region'] | (f['guide'] & (f['accepted'] > 0)),
            'restore.raw_support': f['region'] | (f['guide'] & (f['raw'] > 0)),
            'restore.whole': f['region'] | f['guide']}


def main():
    start, cpu = time.monotonic(), time.process_time()
    OUT.mkdir(exist_ok=True)
    (OUT/'predictions').mkdir(exist_ok=True)
    seal = json.loads((s.OUT/'reconstructed_seal.json').read_text())
    records = seal['files']
    config = dict(candidate='M union (B intersect A)',
        M='strict majority of accepted component masks on canonical1024',
        B='original complete FoRIS mask rendered back to canonical1024 exactly as source guide',
        A='union of accepted component masks',
        controls={'restore.raw_support':'M union (B intersect U), U=raw local mask union',
                  'restore.whole':'M union B'},
        output='canonical first; original via bilinear align_corners=False, strict >.5',
        tuning='none; same formula across source groups; diagnostics already exposed',
        intended_scope='repair only majority loss with simultaneous whole and accepted-local support',
        code_sha=s.sha(__file__), study_sha=s.sha(s.__file__), field_seal_sha=s.sha(s.OUT/'reconstructed_seal.json'),
        diagnostic_sha=s.sha(s.OUT/'diagnostic_report.json'), cpu_threads=1, n=len(records))
    s.write(OUT/'config.json', config)
    output_seals = []
    # No call to load_truth or other query label path occurs before this seal.
    for i, rec in enumerate(records):
        f, shape = load_fields(rec)
        masks = new_masks(f)
        assert not np.any(f['region'] & ~masks['restore.accepted'])
        assert not np.any(masks['restore.accepted'] & ~masks['restore.raw_support'])
        assert not np.any(masks['restore.raw_support'] & ~masks['restore.whole'])
        payload = dict(original_hw=np.asarray(shape))
        for name, mask in masks.items():
            payload['canvas/'+name] = np.packbits(mask)
            payload['original/'+name] = np.packbits(s.render(mask, shape))
        path = OUT/'predictions'/rec['filename']
        np.savez_compressed(path, **payload)
        output_seals.append(dict(group=rec['group'], episode_id=rec['episode_id'], filename=rec['filename'], sha=s.sha(path)))
    inference_seconds = time.monotonic()-start
    s.write(OUT/'sealed.json', dict(state='ALL_NEW_PREDICTIONS_SEALED_BEFORE_SCORING_GT',
        n=len(output_seals), config_sha=s.sha(OUT/'config.json'), files=output_seals,
        seconds=inference_seconds, cpu_seconds=time.process_time()-cpu))
    print(json.dumps(dict(stage='sealed_all_new_masks', n=len(output_seals), seconds=inference_seconds)), flush=True)
    scored = []
    gold = {
        'lvis_original600': (s.A/'lvis_component_completion600_20261009/score/cumulative600_episodes.jsonl',
                              {'foris':'foris.crf', 'region':'component.query_context', 'geometry':'component.global_agreement_budget'}),
        'deep_original100': (s.A/'deepglobe_component100_20261009/score/scored_episodes.jsonl',
                              {'foris':'foris.crf', 'region':'component.query_context', 'geometry':'component.global_agreement_budget', 'unfiltered':'window.native'}),
        'deep_fast100': (s.A/'deepglobe_single_final_crf100_20261009/score/scored_episodes.jsonl',
                          {'foris':'foris.crf', 'region':'fast.no_crf', 'geometry':'fast.geometry_no_crf'}),
        'paco_fast600': (s.A/'paco_fast9_600_20261009/score/scored_episodes.jsonl',
                          {'foris':'foris.crf', 'region':'region.fast'}),
    }
    gold = {g:(s.index(p), names) for g, (p, names) in gold.items()}
    checks = 0
    with (OUT/'episodes.jsonl').open('w', buffering=1) as log:
        for rec, frozen in zip(records, output_seals):
            f, shape = load_fields(rec)
            row, group = rec['row'], rec['group']
            raw_gt = s.load_truth(row)
            path = OUT/'predictions'/rec['filename']
            assert s.sha(path) == frozen['sha']
            per_frame, confusion, edits = {}, {}, {}
            with np.load(path, allow_pickle=False) as z:
                for frame in ('canvas', 'original'):
                    size = s.COVER.shape if frame == 'canvas' else shape
                    gt = s.nearest(raw_gt, size)
                    masks = {name:s.unpack(z, frame+'/'+name, size) for name in ARMS}
                    masks['foris'] = f['base_canvas'] if frame == 'canvas' else f['base_original']
                    masks['region'] = s.render(f['region'], size)
                    masks['unfiltered'] = s.render(f['unfiltered'], size)
                    if 'geometry' in f:
                        masks['geometry'] = s.render(f['geometry'], size)
                    confusion[frame] = {name:counts(m, gt) for name,m in masks.items()}
                    per_frame[frame] = {name:[c['intersection'], c['union']] for name,c in confusion[frame].items()}
                    edits[frame] = {name:gross(masks['region'], masks[name], gt) for name in ARMS}
                    if frame == 'original':
                        old, names = gold[group]
                        iu = old[row['episode_id']]['iu']
                        if group == 'paco_fast600':
                            iu = iu['original']
                        for new, previous in names.items():
                            assert per_frame[frame][new] == iu[previous], (group,row['episode_id'],new)
                            checks += 1
            r = dict(group=group, episode_id=row['episode_id'], fold=row['fold'], class_id=row['loader_class_id'],
                     query_photo_id=row.get('query_photo_id'), iu=per_frame, confusion=confusion, edits_vs_region=edits)
            scored.append(r)
            log.write(json.dumps(r)+'\n')
    expected = json.loads((s.A/'paco_fast9_600_20261009/config.json').read_text())['expected_classes']
    pilot_path = s.A/'joint_role_pilot200_20261010/manifest.json'
    pilot = {r['episode_id']:r for r in json.loads(pilot_path.read_text())}
    summaries = {}
    for group in sorted({r['group'] for r in scored}):
        rs = [r for r in scored if r['group'] == group]
        for label, selected, classes in [(group, rs, expected if group=='paco_fast600' else None)]+(
            [('paco_pilot100', [r for r in rs if r['episode_id'] in pilot], None)] if group=='paco_fast600' else []):
            if label == 'paco_pilot100':
                assert len(selected) == 100
                for r in records:
                    if r['group']==group and r['episode_id'] in pilot:
                        for key in ('query_mask_hash','reference_mask_hash','query_rgb_hash','reference_rgb_hash'):
                            assert r['row'][key] == pilot[r['episode_id']][key]
            frames = {}
            for frame in ('canvas','original'):
                points = [dict(r,iu=r['iu'][frame]) for r in selected]
                q = point(points, classes)
                q['edits_vs_region'] = {name:{k:sum(r['edits_vs_region'][frame][name][k] for r in selected)
                                               for k in selected[0]['edits_vs_region'][frame][name]} for name in ARMS}
                q['confusion'] = {name:{k:sum(r['confusion'][frame][name][k] for r in selected)
                                         for k in selected[0]['confusion'][frame][name]} for name in selected[0]['confusion'][frame]}
                q['delta_vs_region'] = {a:v-q['miou']['region'] for a,v in q['miou'].items()}
                frames[frame] = q
            summaries[label] = frames
    s.write(OUT/'report.json', dict(groups=summaries, original_reference_iu_checks=checks,
        inference_seconds=inference_seconds, wall_seconds=time.monotonic()-start,
        cpu_seconds=time.process_time()-cpu, cpu_threads=1, pilot_manifest_sha=s.sha(pilot_path),
        all_mask_seal_sha=s.sha(OUT/'sealed.json'), episodes_sha=s.sha(OUT/'episodes.jsonl'),
        note='Deep means road total I/U because all belong to one fold/class. PACO600 includes official missing slots; pilot100 observed87 only. Legacy canvas foris is round-trip guide; original endpoint is exact. No bootstrap or held-out confirmation.'))
    print(json.dumps({g:z['original']['miou'] for g,z in summaries.items()}), flush=True)


if __name__ == '__main__':
    main()
