"""Read-only joint error/edit structure of existing exposed full predictions.

All GT use is offline counting. No encoder, fitting, prediction generation,
selector, threshold selection or Boolean-combination mask is executed.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent/'cv_data'
OUT = ASSETS/'a/x1012s2_error_overlap'
R1 = ASSETS/'a/x1012_R1'
R4 = ASSETS/'a/x1012_R4_global'
R5 = ASSETS/'a/x1012_R5_candidate/final_tau0475'
R5RESULT = ASSETS/'a/x1012_R5_results'
INS = ASSETS/'a/x1012_R6_insid3'
COMMON = ('FoRIS', 'INS_bilinear', 'INS_crf640', 'UN_midrange', 'UN_final475')
DEV_EXTRA = ('raw_UN', 'T')
EDITS = ('recovered_TP', 'added_FP', 'deleted_TP', 'removed_FP')
CHINESE_EDITS = ('补回TP', '加入FP', '删除TP', '移除FP')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def helpers():
    modules = {}
    for name, source in [('lab', R1/'frozen/3_lab.py'), ('plan_common', R1/'frozen/1_plan_common.py')]:
        target = OUT/'runtime'/(name+'.py')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        assert source.read_bytes() == target.read_bytes()
        spec = importlib.util.spec_from_file_location(name, target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if name == 'lab':
            module.ROOT, module.ASSETS = ROOT, ASSETS
            module.DEV = torch.device('cpu')
        modules[name] = module
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return modules['lab'], modules['plan_common']


lab, pc = helpers()


def prediction(path, key, digest):
    return {'path': str(path), 'key': key, 'sha256': digest}


def prepare():
    dependencies = [R1/'manifest.json', R1/'sealed.json', R1/'report.json',
                    R4/'sealed.json', R4/'report.json', R5/'sealed.json',
                    R5RESULT/'report.json', ASSETS/'a/x1012_R5_validation/manifest.baselines.json']
    for c in ('development1360', 'r5_500'):
        dependencies += [INS/c/'manifest.json', INS/c/'sealed.json', INS/c/'episodes.json', INS/c/'report.json']
    spec = {
        'state': 'FIXED_EXISTING_OUTPUT_DIAGNOSTIC', 'cohorts': {'development1360': 1360, 'r5_500': 500},
        'main_question': 'Which full paths correct existing FoRIS errors, at what introduced-error cost, and which second paths reject first-path side effects versus rejecting first-path benefits?',
        'common_paths': COMMON, 'development_only_paths': DEV_EXTRA,
        'path_identity': {'UN_midrange': 'R1 VP_UN / R5 UN_midrange with original .5 readout',
                          'UN_final475': 'frozen R4/R5 tau=.475 with rho>.9 unchanged T fallback',
                          'raw_UN': 'R1 independent c field and reliability graph, without VP additions; not R6 raw-query graph',
                          'T': 'R1 original saved independent c readout',
                          'INS_endpoints': 'same released INSID3 full path; bilinear and official CRF640 separately, not independent votes'},
        'missing_path_policy': 'R5 has no saved complete raw_UN/T natural-mask cohort; do not generate it from fields. Joint common failures always name the available path set.',
        'edit_columns': EDITS, 'edit_columns_zh': CHINESE_EDITS,
        'edit_definitions': ['FoRIS0/path1/GT1', 'FoRIS0/path1/GT0', 'FoRIS1/path0/GT1', 'FoRIS1/path0/GT0'],
        'ordered_overlap': 'For first path A and second C, both_share means both change FoRIS; second_vetoes_first means only A changes FoRIS; second_only means only C changes FoRIS. Report four edit labels for each subset. These are masks already produced, not an executed composite.',
        'complementarity': 'On FoRIS errors, at least one other saved path correct versus all saved paths wrong; per-path unique correct and unique harmful changes reported separately.',
        'GT_role': 'offline pixel/error-contingency counting only, no GT selector and no new method scores',
        'metric': 'CLI1024, original per-case valid/ignore masks; class scores pool I/U per observed (fold,class) then equal slots',
        'resources': {'device': 'cpu', 'workers': 1, 'threads': 1},
        'new_encoder_calls': 0, 'fitted_models': 0, 'new_predictions': 0,
        'source_sha256': sha(__file__),
        'helper_sha256': {str(OUT/'runtime'/(n+'.py')): sha(OUT/'runtime'/(n+'.py')) for n in ('lab', 'plan_common')},
        'input_sha256': {str(p): sha(p) for p in dependencies},
    }
    if (OUT/'config.json').exists():
        assert read(OUT/'config.json') == json.loads(json.dumps(spec)), 'Diagnostic contract/input drift'
    else:
        write(OUT/'config.json', spec)
        (OUT/'frozen').mkdir(exist_ok=True)
        shutil.copyfile(__file__, OUT/'frozen/x1012s2_error_overlap.py')
    r1p = {r['index']: r for r in read(R1/'sealed.json')['records']}
    r1s = {r['index']: r for r in read(R1/'report.json')['records']}
    r4p = {r['index']: r for r in read(R4/'sealed.json')['records']}
    r4s = {r['index']: r for r in read(R4/'report.json')['records']}
    r5p = {r['index']: r for r in read(R5/'sealed.json')['records']}
    r5s = {r['index']: r for r in read(R5RESULT/'report.json')['records']}
    r5b = {pc.key(b): b for b in read(ASSETS/'a/x1012_R5_validation/manifest.baselines.json')}
    jobs = []
    for cohort in ('development1360', 'r5_500'):
        ins_p = {r['index']: r for r in read(INS/cohort/'sealed.json')['records']}
        ins_s = {r['index']: r for r in read(INS/cohort/'episodes.json')}
        rows = read(INS/cohort/'manifest.json')
        for b in rows:
            key = pc.key(b)
            bas = (b if cohort == 'development1360' else r5b[key])['old_arms']['B']
            # Legacy extended manifests omit a payload digest. Bind its current
            # bytes here; all seven original scored integers must still match.
            sources = {'FoRIS': prediction(bas['path'], bas['cli_key'], bas.get('sha256') or sha(bas['path']))}
            expected = {'FoRIS': ins_s[key]['metrics']['FoRIS']}
            ip = ins_p[key]
            for alias, actual in (('INS_bilinear', 'insid3.bilinear'), ('INS_crf640', 'insid3.crf640')):
                sources[alias] = prediction(ip['prediction_path'], 'cli/'+actual, ip['prediction_sha256'])
                expected[alias] = ins_s[key]['metrics'][actual]
            if cohort == 'development1360':
                a, c = r1p[key], r4p[key]
                for alias, actual in (('UN_midrange', 'VP_UN'), ('raw_UN', 'raw_UN'), ('T', 'T')):
                    sources[alias] = prediction(a['prediction_path'], actual, a['prediction_sha256'])
                    expected[alias] = r1s[key]['metrics'][actual]
                sources['UN_final475'] = prediction(c['prediction_path'], 'UN_tau475', c['prediction_sha256'])
                expected['UN_final475'] = r4s[key]['metrics']['UN_tau475']
                arms = COMMON+DEV_EXTRA
            else:
                a = r5p[key]
                for alias, actual in (('UN_midrange', 'UN_midrange'), ('UN_final475', 'candidate')):
                    sources[alias] = prediction(a['prediction_path'], actual, a['prediction_sha256'])
                    expected[alias] = r5s[key]['metrics'][actual]
                arms = COMMON
            jobs.append({'binding': b, 'cohort': cohort, 'arms': arms, 'sources': sources, 'expected': expected})
    assert len(jobs) == 1860
    return jobs, spec


def bits_for(arms):
    return ((np.arange(1 << len(arms))[:, None] >> np.arange(len(arms))) & 1).astype(bool)


def metrics_from_counts(counts, arms):
    bits = bits_for(arms)
    result = {}
    for j, arm in enumerate(arms):
        tp, fp = int(counts[1, bits[:, j]].sum()), int(counts[0, bits[:, j]].sum())
        fn = int(counts[1, ~bits[:, j]].sum())
        result[arm] = {'I': tp, 'U': tp+fp+fn, 'TP': tp, 'FP': fp, 'FN': fn,
                       'GT': tp+fn, 'predicted': tp+fp, 'IoU': 100*tp/max(tp+fp+fn, 1)}
    return result


def vector(counts, base, changed):
    return [int(counts[1, ~base & changed].sum()), int(counts[0, ~base & changed].sum()),
            int(counts[1, base & changed].sum()), int(counts[0, base & changed].sum())]


def divide(a, b):
    return float(a/b) if b else None


def structure(counts, arms):
    bits = bits_for(arms)
    base = bits[:, 0]
    metrics = metrics_from_counts(counts, arms)
    bf, bn = metrics['FoRIS']['FP'], metrics['FoRIS']['FN']
    changes = {a: bits[:, i] != base for i, a in enumerate(arms) if i}
    individual, ordered = {}, {}
    for a, changed in changes.items():
        v = vector(counts, base, changed)
        others = np.any(np.stack([x for k, x in changes.items() if k != a]), axis=0)
        individual[a] = {'edits': v, 'recovered_FoRIS_FN_fraction': divide(v[0], bn),
                         'removed_FoRIS_FP_fraction': divide(v[3], bf),
                         'addition_correct_fraction': divide(v[0], v[0]+v[1]),
                         'deletion_correct_fraction': divide(v[3], v[3]+v[2]),
                         'new_errors_per_corrected_error': divide(v[1]+v[2], v[0]+v[3]),
                         'unique_vs_all_other_saved_paths': vector(counts, base, changed & ~others)}
        am = metrics[a]
        bm = metrics['FoRIS']
        assert am['TP'] == bm['TP']+v[0]-v[2]
        assert am['FP'] == bm['FP']+v[1]-v[3]
        assert am['FN'] == bm['FN']-v[0]+v[2]
        for c, second in changes.items():
            if c == a:
                continue
            share = vector(counts, base, changed & second)
            veto = vector(counts, base, changed & ~second)
            only = vector(counts, base, ~changed & second)
            ordered[a+' -> '+c] = {
                'both_share': share, 'second_vetoes_first': veto, 'second_only': only,
                'first_harm_corrected_by_second_fraction': divide(veto[1]+veto[2], v[1]+v[2]),
                'first_benefit_undone_by_second_fraction': divide(veto[0]+veto[3], v[0]+v[3]),
                'second_veto_harm_to_benefit_ratio': divide(veto[1]+veto[2], veto[0]+veto[3])}
    joint = {}
    for label, names in (('common_paths', COMMON[1:]), ('all_available_paths', arms[1:])):
        any_change = np.any(np.stack([changes[a] for a in names]), axis=0)
        v = vector(counts, base, any_change)
        joint[label] = {'other_paths': names, 'at_least_one_other_changes_FoRIS': v,
                        'recoverable_FN_fraction': divide(v[0], bn), 'recoverable_FP_fraction': divide(v[3], bf),
                        'all_paths_wrong_FN': bn-v[0], 'all_paths_wrong_FP': bf-v[3],
                        'note': 'recovery means an existing other path is correct on this GT-labeled pixel; GT cannot choose that path at inference'}
    if 'raw_UN' in arms:
        core = np.any(np.stack([changes[a] for a in COMMON[1:]]), axis=0)
        raw = changes['raw_UN'] | changes['T']
        joint['raw_T_changes_on_common_path_consensus'] = vector(counts, base, ~core & raw)
    return {'valid_pixels': int(counts.sum()), 'metrics': metrics,
            'per_path_vs_FoRIS': individual, 'ordered_path_edit_overlap': ordered, 'joint': joint}


def episode(job):
    b, arms = job['binding'], job['arms']
    gt, valid = pc.truth(b)
    code = np.zeros((1024, 1024), dtype=np.uint8)
    by_path = defaultdict(list)
    for j, arm in enumerate(arms):
        source = job['sources'][arm]
        by_path[source['path']].append((j, arm, source))
    for path, items in by_path.items():
        assert all(sha(path) == source['sha256'] for _, _, source in items[:1])
        with np.load(path, allow_pickle=False) as z:
            for j, arm, source in items:
                pred = pc.unpack(z[source['key']])
                code |= pred.astype(np.uint8) << j
    states = 1 << len(arms)
    labeled = code[valid].astype(np.uint16) + states*gt[valid].astype(np.uint16)
    counts = np.bincount(labeled, minlength=states*2).reshape(2, states)
    metrics = metrics_from_counts(counts, arms)
    for a in arms:
        assert all(metrics[a][k] == job['expected'][a][k] for k in ('I', 'U', 'TP', 'FP', 'FN', 'GT', 'predicted')), (pc.key(b), a, 'saved integer mismatch')
    nonzero = np.flatnonzero(counts.sum(0))
    return {'cohort': job['cohort'], 'index': pc.key(b), 'dataset': b['dataset'],
            'fold': b['row']['fold'], 'class_id': b['row']['global_class_id'], 'arms': arms,
            'query_path': str(ASSETS/b['row']['query_path']), 'query_crop': b['row'].get('query_crop'),
            'reference_path': str(ASSETS/b['row']['reference_path']), 'reference_crop': b['row'].get('reference_crop'),
            'query_mask_path': b['row']['query_mask_path'], 'reference_mask_path': b['row']['reference_mask_path'],
            'sources': job['sources'], 'metrics': metrics,
            'patterns': [[int(k), int(counts[0, k]), int(counts[1, k])] for k in nonzero]}


def restore_counts(record):
    value = np.zeros((2, 1 << len(record['arms'])), dtype=np.int64)
    for state, bg, fg in record['patterns']:
        value[:, state] = (bg, fg)
    return value


def main():
    jobs, spec = prepare()
    started, records = time.monotonic(), []
    for i, job in enumerate(jobs, 1):
        records.append(episode(job))
        if i <= 2 or i % 200 == 0:
            print({'phase': 'existing-mask-counting', 'completed': i, 'n': len(jobs), 'seconds': round(time.monotonic()-started, 1)}, flush=True)
    write(OUT/'episodes.json', records)
    summary = {}
    for cohort in ('development1360', 'r5_500'):
        cohort_summary = {}
        for domain in lab.ALL_DOMAINS:
            rows = [r for r in records if r['cohort'] == cohort and r['dataset'] == domain]
            arms = tuple(rows[0]['arms'])
            aggregate = np.stack([restore_counts(r) for r in rows]).sum(0)
            detail = structure(aggregate, arms)
            slots = defaultdict(lambda: {a: [0, 0] for a in arms})
            for r in rows:
                for a in arms:
                    slot = slots[(str(r['fold']), str(r['class_id']))][a]
                    slot[0] += r['metrics'][a]['I']
                    slot[1] += r['metrics'][a]['U']
            for a in arms:
                detail['metrics'][a]['class_miou'] = float(np.mean([100*v[a][0]/max(v[a][1], 1) for v in slots.values()]))
                detail['metrics'][a]['episode_mean_iou'] = float(np.mean([r['metrics'][a]['IoU'] for r in rows]))
                detail['metrics'][a]['class_delta_to_FoRIS_pp'] = detail['metrics'][a]['class_miou']-detail['metrics']['FoRIS']['class_miou']
            nz = np.flatnonzero(aggregate.sum(0))
            cohort_summary[domain] = dict(n=len(rows), arms=arms, **detail,
                contingency_patterns=[[int(k), int(aggregate[0, k]), int(aggregate[1, k])] for k in nz],
                class_slots=[{'fold': f, 'class_id': c, 'I_U': v} for (f, c), v in slots.items()])
        summary[cohort] = cohort_summary
    result = {'state': 'COMPLETE_EXISTING_OUTPUT_JOINT_DIAGNOSTIC', 'n': len(records),
              'config': spec, 'summary': summary, 'seconds': time.monotonic()-started,
              'all_saved_integer_metrics_reproduced': True, 'new_predictions': 0, 'fitted_models': 0,
              'new_encoder_calls': 0, 'episodes_path': str(OUT/'episodes.json'),
              'episodes_sha256': sha(OUT/'episodes.json'), 'config_sha256': sha(OUT/'config.json'),
              'limitations': ['Pixel- or episode-oracle selection is not an inference rule; none is deployed here.',
                              'INSID3 endpoints share one inference path and UN endpoints share one field; do not count them as independent evidence.',
                              'R5 raw_UN/T masks are absent; no such outputs were generated or silently substituted.',
                              'Ordered edit support is a structural diagnostic; a new natural composite must be explicitly implemented and evaluated before claiming its performance.']}
    write(OUT/'summary.json', result)
    print({'phase': 'complete', 'n': len(records), 'seconds': round(time.monotonic()-started, 1),
           'summary': str(OUT/'summary.json')}, flush=True)


if __name__ == '__main__':
    main()
