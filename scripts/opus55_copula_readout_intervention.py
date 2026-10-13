#!/usr/bin/env python3
"""Exchange saved score ranks and marginals to isolate spatial discrimination.

This is a diagnostic intervention on already exposed development pairs, not a
new deployable method. Swaps occur after native minmax and bilinear resizing;
the fixed .5 threshold and original CRF then act without renormalization.
"""
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opus55_reference_mask_trace as base

OUT = base.OUT / 'copula_readout_intervention'
CANDIDATE = base.OUT / 'query_partition_intervention'
ARMS = ['native', 'candidate', 'native_rank_candidate_distribution',
        'candidate_rank_native_distribution']


def full_field(score):
    import torch
    import torch.nn.functional as f
    x = torch.from_numpy(score.copy())
    x = x - x.min()
    x = x / x.max().clamp_min(1e-6)
    return f.interpolate(x[None, None], (1024, 1024), mode='bilinear',
                         align_corners=False)[0, 0].numpy()


def swap_distribution(spatial, marginal):
    import numpy as np
    from scipy.stats import rankdata
    ranks = rankdata(spatial.ravel(), method='average') - 1
    result = np.interp(ranks, np.arange(ranks.size), np.sort(marginal.ravel()))
    return result.astype('float32').reshape(spatial.shape)


def self_check():
    import numpy as np
    x = np.array([3, 1, 4, 2], dtype='float32')
    y = np.array([.1, .4, .7, .9], dtype='float32')
    got = swap_distribution(x, y)
    assert np.array_equal(got, np.array([.7, .1, .9, .4], dtype='float32'))
    tied = swap_distribution(np.array([0, 0, 1, 2], dtype='float32'), y)
    assert tied[0] == tied[1] and np.all(np.diff(tied) >= 0)
    assert np.array_equal(swap_distribution(x, x), x)


def prepare(out):
    self_check()
    if (out / 'config.json').exists():
        return
    pairs = json.loads((base.OUT / 'pairs.json').read_text())
    indexes = [{r['id']: r for r in map(json.loads, (p / 'inference.jsonl').read_text().splitlines())}
               for p in [base.OUT, CANDIDATE]]
    config = dict(n_pairs=len(pairs), arms=ARMS, new_encodings=0, new_raw_reads=0,
        query_GT_in_inference=False, thresholds_swept=0,
        candidate='task_axis_native_query_partitions; chosen to remove changed query membership as a confound, not by best score',
        intervention='Tie-aware average-rank transport of full1024 native/candidate score marginals, followed directly by >.5 and original CRF',
        tie_policy='Equal source scores remain equal. Exact target histogram and area require no ties; record deviations rather than spatially breaking ties',
        exposure='Same exposed60 development pairs; mechanistic diagnostic, not held-out confirmation or deployable selector',
        source_sha256={str(p): base.sha(p) for p in [Path(__file__), Path(base.__file__),
            base.REPO / 'scripts/cached_dino.py', base.REPO / 'src/ics/m4_crf_cached.py',
            base.ASSETS / 'third_party/foris_official/models/foris.py']},
        parent_sha256={str(p / n): base.sha(p / n) for p in [base.OUT, CANDIDATE]
                       for n in ['sealed.json', 'inference.jsonl']} |
                      {str(base.OUT / 'pairs.json'): base.sha(base.OUT / 'pairs.json')})
    seen = set()
    tasks = []
    for p in pairs:
        identity = p['dataset'] not in seen
        seen.add(p['dataset'])
        tasks.append(dict(pair=p, native=indexes[0][p['id']], candidate=indexes[1][p['id']],
                          identity_probe=identity))
    base.write(out / 'config.json', config)
    base.write(out / 'tasks.json', tasks)


def initialize(out):
    global NP, TORCH, HOST, ROOT
    import numpy as np
    import torch
    from cached_dino import CachedDINO, cache_host
    torch.set_num_threads(2)
    torch.manual_seed(0)
    NP, TORCH, HOST, ROOT = np, torch, cache_host(base.ASSETS, CachedDINO(None),
                                               mask_refiner='crf'), Path(out)


def infer_one(task):
    from ics.official_data import decoded_rgb, array_hash
    start = time.monotonic()
    p = task['pair']
    for r in [task['native'], task['candidate']]:
        assert base.sha(r['file']) == r['sha256']
    attempts, packed, meta = [], {}, {}
    with base.deny_labels(base.OUT, attempts), TORCH.inference_mode():
        q = decoded_rgb(base.ASSETS / p['row']['query_path'], p['row'].get('query_crop'))
        assert array_hash(NP.asarray(q)) == p['row']['query_rgb_hash']
        tgt = HOST._transform(q).unsqueeze(0)
        with NP.load(task['native']['file']) as native, NP.load(task['candidate']['file']) as candidate:
            for v in ['A', 'B']:
                source = dict(native=(native, v + '/'),
                              candidate=(candidate, v + '/task_axis_native_query_partitions/'))
                fields = {}
                for arm, (z, prefix) in source.items():
                    fields[arm] = full_field(z[prefix + 'score'])
                    pre = fields[arm] > .5
                    assert NP.array_equal(NP.packbits(pre), z[prefix + 'pre'])
                    for stage in ['pre', 'crf']:
                        packed[v + '/' + arm + '/' + stage] = z[prefix + stage].copy()
                    if task['identity_probe']:
                        replay = HOST._finalize_mask(TORCH.from_numpy(pre), tgt)
                        assert NP.array_equal(NP.packbits(replay.numpy()), z[prefix + 'crf'])
                meta[v] = dict(source_distinct_values={a: int(NP.unique(x).size) for a, x in fields.items()},
                               source_pre_pixels={a: int((x > .5).sum()) for a, x in fields.items()},
                               swapped_pre_pixels={}, marginal_area_errors={})
                for spatial, marginal in [('native', 'candidate'), ('candidate', 'native')]:
                    arm = spatial + '_rank_' + marginal + '_distribution'
                    swapped = swap_distribution(fields[spatial], fields[marginal])
                    pre = swapped > .5
                    crf = HOST._finalize_mask(TORCH.from_numpy(pre), tgt)
                    packed[v + '/' + arm + '/pre'] = NP.packbits(pre)
                    packed[v + '/' + arm + '/crf'] = NP.packbits(crf.numpy())
                    meta[v]['swapped_pre_pixels'][arm] = int(pre.sum())
                    meta[v]['marginal_area_errors'][arm] = int(pre.sum()) - meta[v]['source_pre_pixels'][marginal]
        assert not attempts and HOST.encoder.encoder is None
        assert HOST.encoder.encoder_calls == 0 and HOST.encoder.cache_reads == 0
    file = ROOT / 'predictions' / (p['id'] + '.npz')
    file.parent.mkdir(parents=True, exist_ok=True)
    NP.savez_compressed(file, **packed)
    return dict(id=p['id'], dataset=p['dataset'], file=str(file), sha256=base.sha(file),
        meta=meta, native_and_candidate_pre_bit_exact=True,
        crf_renderer_identity_probe=task['identity_probe'], encoder_calls=0, raw_reads=0,
        query_label_access_attempts=attempts, seconds=time.monotonic() - start, worker_pid=os.getpid())


def infer(out, workers):
    config = json.loads((out / 'config.json').read_text())
    for p, digest in (config['source_sha256'] | config['parent_sha256']).items():
        assert base.sha(p) == digest
    if (out / 'sealed.json').exists():
        return
    tasks = json.loads((out / 'tasks.json').read_text())
    done, ledger, start = {}, out / 'inference.jsonl', time.monotonic()
    if ledger.exists():
        for r in map(json.loads, ledger.read_text().splitlines()):
            assert base.sha(r['file']) == r['sha256']
            done[r['id']] = r
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
        mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
        initargs=(str(out),)) as pool, ledger.open('a', buffering=1) as handle:
        futures = [pool.submit(infer_one, t) for t in tasks if t['pair']['id'] not in done]
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            handle.write(json.dumps(r) + '\n')
            done[r['id']] = r
            base.write(out / 'activity.json', dict(state='COPULA_READOUT_INTERVENTION',
                completed=len(done), n=len(tasks), controller_pid=os.getpid(),
                worker_pids=[p.pid for p in pool._processes.values()],
                elapsed_seconds=time.monotonic() - start, new_encodings=0))
    assert len(done) == len(tasks)
    base.write(out / 'sealed.json', dict(state='ALL_COPULA_PREDICTIONS_SEALED', n_pairs=len(tasks),
        arms=ARMS, new_encodings=0, raw_reads=0, query_GT_in_inference=False,
        config_sha256=base.sha(out / 'config.json'), tasks_sha256=base.sha(out / 'tasks.json'),
        inference_sha256=base.sha(ledger)))


def diagnose(out):
    import numpy as np
    import torch
    import torch.nn.functional as f
    torch.set_num_threads(2)
    seal = json.loads((out / 'sealed.json').read_text())
    for name, key in [('config.json', 'config_sha256'), ('tasks.json', 'tasks_sha256'),
                      ('inference.jsonl', 'inference_sha256')]:
        assert base.sha(out / name) == seal[key]
    truth = json.loads((base.OUT / 'query_truth.json').read_text())
    records = []
    for r in map(json.loads, (out / 'inference.jsonl').read_text().splitlines()):
        assert base.sha(r['file']) == r['sha256']
        targets = []
        for v in ['A', 'B']:
            b = truth[r['id']]['masks'][v]
            assert base.sha(b['path']) == b['sha256']
            target = torch.from_numpy(base.read_mask(b['path']).astype('float32'))
            targets.append(f.interpolate(target[None, None], (1024, 1024), mode='nearest')[0, 0].numpy().astype(bool))
        ex = [targets[0] & ~targets[1], targets[1] & ~targets[0]]
        keep, measures = ex[0] | ex[1], {}
        with np.load(r['file']) as z:
            for arm in ARMS:
                measures[arm] = {}
                for stage in ['pre', 'crf']:
                    values, aucs = [], []
                    for i, v in enumerate(['A', 'B']):
                        pred = np.unpackbits(z[v + '/' + arm + '/' + stage], count=1024**2).reshape(1024, 1024).astype(bool)
                        hit = int((pred & targets[i]).sum())
                        union = int((pred | targets[i]).sum())
                        values.append(dict(I=hit, U=union, TP=hit, FP=int((pred & ~targets[i]).sum()),
                                           FN=int((~pred & targets[i]).sum()), predicted=int(pred.sum()),
                                           target=int(targets[i].sum()), iou=100 * hit / max(union, 1)))
                        aucs.append(base.weighted_auc(pred[keep], ex[i][keep], ex[1-i][keep]))
                    measures[arm][stage] = dict(tasks=values, task_mean_iou=float(np.mean([v['iou'] for v in values])),
                                               switch_auc=float(np.mean(aucs)))
        records.append(r | dict(classes=truth[r['id']]['classes'], measures=measures))
    summary = {}
    for d in sorted({r['dataset'] for r in records}):
        sub = [r for r in records if r['dataset'] == d]
        arms = {}
        for arm in ARMS:
            arms[arm] = {}
            for stage in ['pre', 'crf']:
                arms[arm][stage] = dict(task_mean_iou=float(np.mean([r['measures'][arm][stage]['task_mean_iou'] for r in sub])),
                    delta_native=base.bootstrap([r['measures'][arm][stage]['task_mean_iou'] - r['measures']['native'][stage]['task_mean_iou'] for r in sub]),
                    switch_auc=float(np.mean([r['measures'][arm][stage]['switch_auc'] for r in sub])),
                    per_role_delta_native={v: base.bootstrap([r['measures'][arm][stage]['tasks'][i]['iou'] - r['measures']['native'][stage]['tasks'][i]['iou'] for r in sub]) for i, v in enumerate(['A', 'B'])},
                    mean_predicted_pixels=float(np.mean([t['predicted'] for r in sub for t in r['measures'][arm][stage]['tasks']])))
        summary[d] = dict(n=len(sub), arms=arms,
            max_tie_induced_area_error_pixels=max(abs(n) for r in sub for v in r['meta'].values() for n in v['marginal_area_errors'].values()))
    base.write(out / 'diagnosis.json', dict(state='POST_SEAL_COPULA_CAUSAL_DIAGNOSIS',
        datasets=summary, records=records, seal_sha256=base.sha(out / 'sealed.json'),
        new_encodings=0, raw_reads=0,
        limits='Swap separates saved spatial ordering and marginal readout, with ties reported. No GT threshold or new deployable candidate; same exposed development pairs, not official class mIoU.'))
    lines = ['# 排序与分数分布交换：空间判别和读出面积拆分', '',
        '原FoRIS与固定native查询分区的task-axis场，先分别执行原minmax及bilinear1024，再交换排序与分数边际分布。直接>.5及原CRF，不再次归一化。相同分数保留ties，不按像素位置打破。',
        '60对开发照片、每对两个真实任务；零编码、零raw读取，全部预测封存后才读取query GT。IoU是每照片两任务均值，不是官方class mIoU。', '',
        '|域|原FoRIS|原候选|原排序＋候选分布|候选排序＋原分布|', '|---|---:|---:|---:|---:|']
    for d, s in summary.items():
        lines.append('|' + d + '|' + '|'.join(f"{s['arms'][a]['crf']['task_mean_iou']:.3f}" for a in ARMS) + '|')
    lines += ['', '配对2000次photo-bootstrap区间、原任务A/辅助任务B分开结果、TP/FP/FN、pre与CRF像素面积以及ties引起的面积误差完整保存在diagnosis.json。',
              '若仅交换分布复现收益，不能归因于更准确的空间身份排序；若仅交换排序仍受益，才支持空间判别改善。CRF非线性仍可能改变交换后的面积，因此同时报告pre。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    base.write(out / 'activity.json', dict(state='DIAGNOSED', n_pairs=len(records), new_encodings=0, raw_reads=0))
    print(json.dumps({d: {a: s['arms'][a]['crf'] for a in ARMS} for d, s in summary.items()}), flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    prepare(OUT)
    infer(OUT, a.workers)
    diagnose(OUT)


if __name__ == '__main__':
    main()
