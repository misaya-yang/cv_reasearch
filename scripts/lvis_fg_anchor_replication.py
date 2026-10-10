#!/usr/bin/env python3
"""One new, photo-disjoint development200: frozen FG-anchor vs full FoRIS.

Existing score/FG fields and FoRIS masks are reused. Only the original CRF for
FG-anchor runs. Query labels are confined to score, after all masks are sealed.
"""
import argparse
import concurrent.futures
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '2')
from lvis_atomic_study import ASSETS, DEFAULT_OUT as ATOMIC, point, write
from raw_feature_cache import file_hash

PILOT = ATOMIC / 'pilot100'
ROOT = ASSETS / 'a/lvis_fg_anchor200_20261009'
ARMS = ['foris.crf', 'foris.fg_anchor.crf']
BASE, ANCHOR = ARMS


def index(path):
    return {r['episode_id']: r for r in map(json.loads, Path(path).read_text().splitlines())}


def prepare():
    if (ROOT / 'config.json').exists():
        return validate()
    assert subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip() == 'codex_m4'
    rows = json.loads((ATOMIC / 'manifest.json').read_text())
    old = json.loads((PILOT / 'manifest.json').read_text())
    seen = {r['query_photo_id'] for r in old}
    selected = []
    for fold in range(10):
        eligible = [r for r in rows if r['fold'] == fold and r['query_photo_id'] not in seen]
        for row in eligible[:20]:
            assert row['query_photo_id'] not in seen
            selected.append(row)
            seen.add(row['query_photo_id'])
        assert len(eligible[:20]) == 20
    assert len(selected) == len({r['query_photo_id'] for r in selected}) == 200
    assert all(r['query_crop'] is None for r in selected + old)
    assert not {r['query_photo_id'] for r in old} & {r['query_photo_id'] for r in selected}
    atomic = index(ATOMIC / 'inference.jsonl')
    def task(row):
        rec = atomic[row['episode_id']]
        assert (ATOMIC / 'fields' / rec['filename']).is_file()
        assert (ATOMIC / 'predictions' / rec['filename']).is_file()
        return dict(row=row, filename=rec['filename'], field_sha256=rec['field_sha256'],
                    prediction_sha256=rec['prediction_sha256'])
    tasks = [task(r) for r in selected]
    sources = [Path(__file__), REPO / 'src/ics/m4_crf.py',
               REPO / 'src/ics/official_data.py', REPO / 'scripts/lvis_atomic_study.py',
               REPO / 'scripts/run_m4_baselines.py', REPO / 'src/ics/metrics.py',
               ASSETS / 'third_party/foris_official/utils/data.py',
               ASSETS / 'third_party/foris_official/utils/refinement.py']
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ('fields', 'predictions', 'source'):
        (ROOT / name).mkdir(exist_ok=True)
    write(ROOT / 'manifest.json', selected)
    write(ROOT / 'tasks.json', tasks)
    write(ROOT / 'probe_task.json', task(old[0]))
    config = dict(n=200, arms=ARMS, branch='codex_m4',
        selection='first20 per fold in existing1400 manifest after excluding old100 query photos; no scores or labels used',
        exposure='new200 candidate replication from previously diagnosed1400 development pool; not independent confirmation',
        old100_query_photo_overlap=0, unique_query_photos=200,
        candidate_output_exposure='FG-anchor had been run only on old100, not this200',
        cache_coverage=dict(score=200, fg=200, full_FoRIS_masks=200, missing=0),
        anchor='(full original final score - actual original FG LSE min)/(FG max-FG min).clamp_min(1e-6); bilinear1024 >.5; original CRF',
        encoder_calls=0, query_GT_in_inference=False, parameter_search=False,
        scope='exactly one200; stop on completion; no additional layers/views/graph candidates/full runs',
        source_sha256={str(p): file_hash(p) for p in sources},
        manifest_sha256=file_hash(ROOT / 'manifest.json'), tasks_sha256=file_hash(ROOT / 'tasks.json'),
        parent_seal_sha256=file_hash(ATOMIC / 'sealed.json'))
    write(ROOT / 'config.json', config)
    for p in sources:
        relative = p.relative_to(REPO) if p.is_relative_to(REPO) else Path('assets') / p.relative_to(ASSETS)
        target = ROOT / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    write(ROOT / 'activity.json', dict(state='PREPARED_ONLY', n=200, completed=0, encoder_calls=0))
    return config


def validate():
    config = json.loads((ROOT / 'config.json').read_text())
    for name in ('manifest', 'tasks'):
        assert file_hash(ROOT / (name + '.json')) == config[name + '_sha256']
    for p, digest in config['source_sha256'].items():
        assert file_hash(p) == digest, p
    assert file_hash(ATOMIC / 'sealed.json') == config['parent_seal_sha256']
    return config


def initialize():
    global CFG, TRANSFORM, CRF, BAND, CORE, REFINE
    import torch
    from ics.m4_crf import install
    torch.set_num_threads(2)
    torch.manual_seed(0)
    CFG = validate()
    sys.path.insert(0, str(ASSETS / 'third_party/foris_official'))
    with (ROOT / 'crf_install.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        install(ASSETS / 'third_party/crf_source', ASSETS / 'runtime/macos/crf')
    from utils.data import build_transform
    from utils.refinement import init_crf, crf_refine
    TRANSFORM = build_transform(1024)
    CRF, BAND, CORE = init_crf(1024, 'cpu')
    CRF.eval().requires_grad_(False)
    REFINE = crf_refine


def prediction(task):
    import numpy as np
    import torch
    import torch.nn.functional as f
    from PIL import Image
    from ics.official_data import array_hash
    row = task['row']
    path = ATOMIC / 'fields' / task['filename']
    assert file_hash(path) == task['field_sha256']
    with np.load(path) as saved:
        fg, score = (saved[k].copy() for k in ('fg', 'score'))
    with Image.open(ASSETS / row['query_path']) as opened:
        query = opened.convert('RGB')
    assert array_hash(np.asarray(query)) == row['query_rgb_hash']
    with torch.inference_mode():
        q_input = TRANSFORM(query)
        parent, child = torch.from_numpy(fg), torch.from_numpy(score)
        anchor = ((child - parent.min()) / (parent - parent.min()).max().clamp_min(1e-6)).numpy()
        initial = f.interpolate(torch.from_numpy(anchor)[None, None], (1024, 1024),
                                mode='bilinear', align_corners=False)[0, 0] > .5
        start = time.monotonic()
        mask = REFINE(CRF, BAND, CORE, q_input[None], initial).numpy()
        crf_seconds = time.monotonic() - start
    return {ANCHOR: mask}, dict(foreground_anchor_normalized_score=anchor), dict(crf_seconds=crf_seconds)


def probe():
    import numpy as np
    initialize()
    task = json.loads((ROOT / 'probe_task.json').read_text())
    masks, fields, info = prediction(task)
    rec = index(PILOT / 'inference.jsonl')[task['row']['episode_id']]
    with np.load(PILOT / 'fields' / rec['filename']) as saved:
        assert np.array_equal(fields['foreground_anchor_normalized_score'], saved['foreground_anchor_normalized_score'])
    with np.load(PILOT / 'predictions' / rec['filename']) as saved:
        assert np.array_equal(np.packbits(masks[ANCHOR]), saved['cli/' + ANCHOR])
    write(ROOT / 'probe_receipt.json', dict(state='EXISTING_ANCHOR_BIT_EXACT',
          old_episode=task['row']['episode_id'], new200_predictions=0, query_GT_opened=False,
          encoder_calls=0, diagnostics=info))
    print('Old-case anchor field and CRF mask replay bit exact.', flush=True)


def one(task):
    import numpy as np
    from run_m4_baselines import render
    started = time.monotonic()
    masks, fields, info = prediction(task)
    name = task['filename']
    path = ATOMIC / 'predictions' / name
    assert file_hash(path) == task['prediction_sha256']
    packed = dict(original_hw=np.asarray(task['row']['query_size_hw']))
    with np.load(path) as saved:
        for frame in ('cli', 'original'):
            for arm in [BASE]:
                packed[frame + '/' + arm] = saved[frame + '/' + arm].copy()
    for arm, mask in masks.items():
        packed['cli/' + arm] = np.packbits(mask)
        packed['original/' + arm] = np.packbits(render(mask, tuple(task['row']['query_size_hw'])))
    np.savez_compressed(ROOT / 'predictions' / name, **packed)
    np.savez_compressed(ROOT / 'fields' / name, **fields)
    return dict(episode_id=task['row']['episode_id'], filename=name,
                prediction_sha256=file_hash(ROOT / 'predictions' / name),
                field_sha256=file_hash(ROOT / 'fields' / name), encoder_calls=0,
                raw_O24_reads=0, diagnostics=info, seconds=time.monotonic() - started)


def infer():
    validate()
    assert (ROOT / 'probe_receipt.json').exists()
    if (ROOT / 'sealed.json').exists():
        print('Already sealed; no restart.')
        return
    tasks = json.loads((ROOT / 'tasks.json').read_text())
    ledger = ROOT / 'inference.jsonl'
    done = index(ledger) if ledger.exists() else {}
    for rec in done.values():
        assert file_hash(ROOT / 'predictions' / rec['filename']) == rec['prediction_sha256']
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, initializer=initialize,
            mp_context=multiprocessing.get_context('spawn')) as pool, ledger.open('a', buffering=1) as log:
        jobs = [pool.submit(one, t) for t in tasks if t['row']['episode_id'] not in done]
        for future in concurrent.futures.as_completed(jobs):
            rec = future.result()
            done[rec['episode_id']] = rec
            log.write(json.dumps(rec) + '\n')
            write(ROOT / 'activity.json', dict(state='INFERENCE', n=200, completed=len(done),
                  controller_pid=os.getpid(), worker_pids=[p.pid for p in pool._processes.values()],
                  encoder_calls=0, elapsed_seconds=time.monotonic() - started))
            if len(done) % 20 == 0:
                print(json.dumps(dict(completed=len(done), n=200, seconds=time.monotonic() - started)), flush=True)
    assert len(done) == 200
    write(ROOT / 'sealed.json', dict(state='ALL_PREDICTIONS_SEALED', n=200,
          config_sha256=file_hash(ROOT / 'config.json'), manifest_sha256=file_hash(ROOT / 'manifest.json'),
          inference_index_sha256=file_hash(ledger), query_GT_in_inference=False, encoder_calls=0,
          wall_seconds=time.monotonic() - started))
    write(ROOT / 'activity.json', dict(state='INFERENCE_COMPLETE', n=200, completed=200, encoder_calls=0, worker_pids=[]))


def contributions(flat, arm, base):
    values = {}
    for subset in range(16):
        rows = []
        for r in flat:
            i, u = r['iu'][base]
            at, af, dt, df = r['edits'][arm][base]
            i += (at if subset & 1 else 0) - (dt if subset & 4 else 0)
            u += (af if subset & 2 else 0) - (df if subset & 8 else 0)
            rows.append(dict(fold=r['fold'], class_id=r['class_id'], iu={'value': [i, u]}))
        values[subset] = point(rows, ['value'])['value']
    out = []
    for bit in range(4):
        total = 0.
        for subset in range(16):
            if subset & (1 << bit):
                continue
            k = subset.bit_count()
            total += math.factorial(k) * math.factorial(3-k) / 24 * (values[subset | (1 << bit)] - values[subset])
        out.append(total)
    assert abs(sum(out) - (point(flat, [arm])[arm] - point(flat, [base])[base])) < 1e-9
    return out


def score():
    import numpy as np
    import torch
    import torch.nn.functional as f
    from PIL import Image
    from ics.official_data import array_hash
    from ics.metrics import counts, gross_edits
    torch.set_num_threads(2)
    seal = json.loads((ROOT / 'sealed.json').read_text())
    for name, key in [('config.json', 'config_sha256'), ('manifest.json', 'manifest_sha256'),
                      ('inference.jsonl', 'inference_index_sha256')]:
        assert file_hash(ROOT / name) == seal[key]
    source = index(ATOMIC / 'episode_metrics.jsonl')
    predictions = index(ROOT / 'inference.jsonl')
    rows = json.loads((ROOT / 'manifest.json').read_text())
    metrics = []
    for row in rows:
        rec = predictions[row['episode_id']]
        path = ROOT / 'predictions' / rec['filename']
        assert file_hash(path) == rec['prediction_sha256']
        with Image.open(row['query_mask_path']) as opened:
            raw_truth = (np.asarray(opened.convert('L')) > 0).astype(np.uint8)
        assert array_hash(raw_truth) == row['query_mask_hash']
        item = dict(episode_id=row['episode_id'], query_photo_id=row['query_photo_id'],
                    fold=row['fold'], class_id=row['loader_class_id'], frames={})
        with np.load(path) as saved:
            for frame, shape in [('cli', (1024, 1024)), ('original', tuple(row['query_size_hw']))]:
                truth = f.interpolate(torch.from_numpy(raw_truth)[None, None].float(), shape, mode='nearest')[0, 0].numpy() > .5
                masks = {a: np.unpackbits(saved[frame + '/' + a], count=int(np.prod(shape))).reshape(shape).astype(bool) for a in ARMS}
                iu = {a: counts(m, truth) for a, m in masks.items()}
                for a in [BASE]:
                    assert iu[a] == source[row['episode_id']]['frames'][frame]['iu'][a]
                item['frames'][frame] = dict(iu=iu, truth_pixels=int(truth.sum()),
                    predicted_pixels={a: int(m.sum()) for a, m in masks.items()},
                    edits={ANCHOR: {BASE: gross_edits(masks[ANCHOR], masks[BASE], truth)}})
        metrics.append(item)
    (ROOT / 'episode_metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metrics))
    report = dict(state='COMPLETE', n=200, encoder_calls=0, development_exposed=True, frames={},
                  inference_wall_seconds=seal['wall_seconds'], scope='one photo-disjoint candidate replication; previously diagnosed development pool')
    for frame in ('cli', 'original'):
        flat = [dict(r, **r['frames'][frame]) for r in metrics]
        miou = point(flat, ARMS)
        pairs = {}
        for a, b in [(ANCHOR, BASE)]:
            delta = [r['iu'][a][0] / max(r['iu'][a][1], 1) - r['iu'][b][0] / max(r['iu'][b][1], 1) for r in flat]
            pairs[a + ' vs ' + b] = dict(delta_pp=miou[a] - miou[b],
                cases_up=sum(d > 1e-10 for d in delta), cases_down=sum(d < -1e-10 for d in delta),
                cases_equal=sum(abs(d) <= 1e-10 for d in delta),
                edits=np.sum([r['edits'][a][b] for r in flat], axis=0).tolist(),
                exact_miou_contributions=contributions(flat, a, b), edit_order=['add_TP', 'add_FP', 'delete_TP', 'delete_FP'])
        pixel = {}
        for a in ARMS:
            tp = sum(r['iu'][a][0] for r in flat)
            pred = sum(r['predicted_pixels'][a] for r in flat)
            gt = sum(r['truth_pixels'] for r in flat)
            pixel[a] = dict(precision=tp / max(pred, 1), recall=tp / max(gt, 1))
        report['frames'][frame] = dict(miou=miou, pairs=pairs, pixel=pixel,
            zero_intersection={a: sum(r['iu'][a][0] == 0 for r in flat) for a in ARMS},
            empty_predictions={a: sum(r['predicted_pixels'][a] == 0 for r in flat) for a in ARMS},
            per_fold={str(fold): point([r for r in flat if r['fold'] == fold], ARMS) for fold in range(10)})
    write(ROOT / 'report.json', report)
    write(ROOT / 'activity.json', dict(state='COMPLETE', n=200, completed=200, encoder_calls=0, worker_pids=[]))
    print(json.dumps({k: v['miou'] for k, v in report['frames'].items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'probe', 'infer', 'score'))
    action = parser.parse_args().action
    globals()[action]()
