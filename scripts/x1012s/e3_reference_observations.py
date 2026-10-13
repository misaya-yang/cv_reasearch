"""E3: one reference's actual focus versus raw-feature interpolation in IND0.

Only the reference observations change. Whole query, mu1, temperature, cached
K64, physical FG/BG mass and readout are shared. All natural outputs are sealed
before query labels or the exact archived FoRIS/MEAN masks are opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/x1012'), str(ROOT / 'scripts'), str(ROOT / 'src')]

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import b1_kshot as b1
import plan_report as pr
from raw_feature_cache import tensor_hash
from run_reference_focus200 import deny_paths
from ics.methods.reference_focus_head import _derived_focus, _reference_measure

lab, pc = b1.lab, b1.pc
OUT = lab.ASSETS / 'a/x1012s_E3'
SOURCE = lab.ASSETS / 'a/reference_focus_head200_20261010'
PRODUCER = lab.ASSETS / 'a/reference_focus200_20261010'
P0 = lab.ASSETS / 'a/x1012_P0/fields'
PLAN = ROOT / 'docs/research/SINGLE_REFERENCE_OBSERVATION_PLAN_20261012.md'
ARMS = ('whole', 'derived', 'actual')
CONTROLS = {'FoRIS': 'cli1024/foris.crf', 'MEAN': 'cli1024/mean'}
TEMPERATURE = .05
TOL = 1e-5


def setup(device='cpu', threads=2):
    pc.setup(device)
    torch.set_num_threads(threads)
    lab.basis.cache_clear()


def prepare(args):
    assert PLAN.is_file(), 'E3 plan must exist before preparation'
    if (OUT / 'config.json').exists():
        check_config()
        print({'state': 'ALREADY_PREPARED', 'out': str(OUT)}, flush=True)
        return
    assert not OUT.exists() or not any(OUT.iterdir()), 'Preserve existing E3 outputs'
    old = pc.read(SOURCE / 'sealed.json')
    assert old['state'] == 'ALL_PREDICTIONS_SEALED' and old['n'] == 200
    for name in ('config', 'manifest', 'tasks'):
        assert pc.sha(SOURCE / (name + '.json')) == old[name + '_sha256']
    producer = pc.read(PRODUCER / 'COMPLETE.json')
    assert producer['state'] == 'REFERENCE_FOCUS_O24_COMPLETE' and producer['n'] == 200
    assert pc.sha(PRODUCER / 'input_bindings.json') == producer['input_bindings_sha256']
    rows, tasks = pc.read(SOURCE / 'manifest.json'), pc.read(SOURCE / 'tasks.json')
    focus_rows = pc.read(PRODUCER / 'input_bindings.json')
    assert pc.sha(SOURCE / 'manifest.json') == producer['source_manifest_sha256']
    assert Counter(r['dataset'] for r in rows) == {'deepglobe_road': 100, 'paco_part': 100}
    source_cfg = pc.read(SOURCE / 'config.json')
    pilot = Path(source_cfg['source_pilot_root'])
    assert pc.sha(pilot / 'sealed.json') == source_cfg['source_pilot_seal_sha256']
    pilot_seal = pc.read(pilot / 'sealed.json')
    assert pc.sha(pilot / 'manifest.json') == pilot_seal['manifest_sha256'] == old['manifest_sha256']
    assert pc.sha(pilot / 'inference.jsonl') == pilot_seal['inference_sha256']
    import json
    baseline_records = {r['episode_id']: r for r in
                        (json.loads(line) for line in (pilot / 'inference.jsonl').read_text().splitlines())}
    p0seal = pc.read(P0 / 'sealed.json')
    assert p0seal['state'] == 'STANDARD_FIELDS_SEALED'
    assert pc.sha(P0 / 'manifest.json') == p0seal['manifest_sha256']
    p0records = {r['index']: r for r in p0seal['records']}
    partitions = {}
    for b in pc.read(P0 / 'manifest.json'):
        partitions.setdefault(b['fit']['query_raw_identity']['tensor_sha256'], p0records[pc.key(b)])
    manifest = []
    for i, (row, source, focus) in enumerate(zip(rows, tasks, focus_rows)):
        assert i == source['index'] == focus['index']
        assert row['episode_id'] == source['episode_id'] == focus['episode_id']
        assert source['focus'] == focus
        for key in ('reference_path', 'reference_crop', 'reference_rgb_hash', 'reference_mask_path', 'reference_mask_hash'):
            assert focus[key] == row.get(key), (i, key)
        x0, y0, x1, y1 = focus['box_xyxy']
        assert x1 - x0 == y1 - y0 == 512 and all(v % 16 == 0 for v in (x0, y0, x1, y1))
        assert 0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 1024
        raw = {r['role']: r for r in source['raw']}
        assert raw['query_full']['tensor_sha256'] in partitions, 'Missing exact cached query K64'
        old_record = baseline_records[row['episode_id']]
        baseline = source['baseline']
        assert Path(baseline['path']) == pilot / 'predictions' / old_record['filename']
        assert baseline['sha256'] == old_record['prediction_sha256'] == pc.sha(baseline['path'])
        for role in ('reference_full', 'query_full'):
            identity = next(r for r in old_record['raw_inputs'] if r['role'] == role)
            assert all(raw[role][k] == identity[k] for k in ('key', 'payload_sha256', 'tensor_sha256'))
        with np.load(baseline['path'], allow_pickle=False) as z:
            assert all(key in z.files for key in CONTROLS.values()), 'Exact complete controls unavailable'
        focus_raw = {k: focus[k] for k in ('key', 'entry_path', 'entry_sha256', 'payload_path', 'payload_sha256', 'input_tensor_hash')}
        focus_raw['tensor_sha256'] = focus['O24']['tensor_sha256']
        manifest.append({'index': f'{i:06d}', 'dataset': row['dataset'], 'row': row,
                         'fit': {'reference_raw': raw['reference_full'], 'query_raw_identity': raw['query_full']},
                         'focus': focus, 'focus_raw': focus_raw, 'baseline': baseline,
                         'partition': partitions[raw['query_full']['tensor_sha256']]})
    assert len(manifest) == 200
    files = [Path(__file__), Path(b1.__file__), Path(pc.__file__), Path(pr.__file__), Path(lab.__file__),
             ROOT / 'scripts/raw_feature_cache.py', ROOT / 'scripts/run_reference_focus200.py',
             ROOT / 'src/ics/methods/reference_focus_head.py', lab.BASIS, PLAN]
    files += [SOURCE / (n + '.json') for n in ('config', 'manifest', 'tasks', 'sealed')]
    files += [PRODUCER / 'COMPLETE.json', PRODUCER / 'input_bindings.json',
              P0 / 'manifest.json', P0 / 'sealed.json', pilot / 'manifest.json',
              pilot / 'sealed.json', pilot / 'inference.jsonl']
    pc.write(OUT / 'manifest.json', manifest)
    pc.write(OUT / 'config.json', {
        'state': 'FROZEN', 'task': 'E3', 'n': 200, 'device': args.device, 'threads': args.threads,
        'query_block': 256, 'temperature': TEMPERATURE, 'arms': list(ARMS),
        'smoke_indices': ['000000', '000100'], 'source_sha256': {str(p): pc.sha(p) for p in files},
        'manifest_sha256': pc.sha(OUT / 'manifest.json'),
        'partition': 'Exact bound P0 regions64 by whole-query tensor SHA; no reclustering or B1 reproduction claim',
        'projection': 'raw O24 -> unit -> fixed positional projection -> unit -> subtract fixed mu1 -> unit',
        'mu1': '.5*(mean(debiased whole R)+mean(debiased whole Q)); reused for all focus arms',
        'weights': 'physical inverse overlap: whole outside256/inside128, focus32, divided globally by256',
        'derived': 'crop whole raw32x32 then bilinear64 align_corners=False; same subsequent transforms as actual',
        'nearest_null': 'exact parent-token repeat, fine FG/BG mass folded back into parent kernels',
        'readout': 'fixed K64 region mean; bilinear1024; minmax; strict >.5',
        'controls': CONTROLS, 'query_GT_in_inference': False, 'new_encoder_calls': 0,
        'exposure': 'Existing PACO100/Deep100 development tasks; not fresh confirmation or ten-domain result',
        'cost_scope': 'cached three raw observations plus three KDE arms; no cold encoder timing'})
    print({'state': 'PREPARED', 'n': 200, 'out': str(OUT), 'device': args.device, 'threads': args.threads}, flush=True)


def check_config():
    cfg = pc.read(OUT / 'config.json')
    assert pc.sha(OUT / 'manifest.json') == cfg['manifest_sha256']
    for path, digest in cfg['source_sha256'].items():
        assert pc.sha(path) == digest, 'Frozen E3 dependency changed: ' + path
    return cfg


def reference_measure(b):
    focus, row = b['focus'], b['row']
    with Image.open(row['reference_mask_path']) as image:
        raw = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
    assert tensor_hash(raw) == row['reference_mask_hash']
    mask = F.interpolate(torch.from_numpy(raw)[None, None].float(), (1024, 1024), mode='nearest')[0, 0]
    assert tensor_hash(mask.numpy().astype(np.uint8)) == focus['canonical_mask_uint8_array_sha256']
    box = tuple(focus['box_xyxy'])
    measure = _reference_measure(mask, box)
    assert pc.sha(focus['coverage64_path']) == focus['coverage64_file_sha256']
    saved = np.load(focus['coverage64_path'], allow_pickle=False)
    assert tensor_hash(saved) == focus['coverage64_tensor_sha256']
    assert np.array_equal(saved.reshape(-1), measure['coverage'][4096:].numpy())
    y = measure['coverage'].float().to(lab.DEV)
    area = (measure['physical_area'] / 256.).float().to(lab.DEV)
    fg, bg = area * y, area * (1 - y)
    y0 = y[:4096]
    assert abs(float(fg.sum() - y0.sum())) < TOL
    assert abs(float(bg.sum() - (1 - y0).sum())) < TOL
    x0, z0, x1, z1 = box
    parent = torch.arange(4096, device=lab.DEV).reshape(64, 64)[z0 // 16:z1 // 16, x0 // 16:x1 // 16]
    parent = parent.repeat_interleave(2, 0).repeat_interleave(2, 1).flatten()
    # This is the exact nearest-copy kernel contribution, without an unnecessary
    # duplicate 4096x1024 matrix multiplication. It checks each parent, not just totals.
    repeat_fg = fg[:4096].clone().index_add_(0, parent, fg[4096:])
    repeat_bg = bg[:4096].clone().index_add_(0, parent, bg[4096:])
    assert torch.equal(repeat_fg, y0) and torch.equal(repeat_bg, 1 - y0)
    return y0, fg, bg, repeat_fg, repeat_bg, measure['diagnostics']


@torch.inference_mode()
def observation_fields(raw_r, raw_focus, raw_q, box, y0, fg, bg, repeat_fg, repeat_bg, block=256):
    r, q = lab.debias(raw_r.to(lab.DEV)), lab.debias(raw_q.to(lab.DEV))
    mu = .5 * (r.mean(0) + q.mean(0))
    rc, qc = lab.unit(r - mu), lab.unit(q - mu)
    variants = {'derived': _derived_focus(raw_r, box), 'actual': raw_focus}
    focus_features = {name: lab.unit(lab.debias(value.to(lab.DEV)) - mu) for name, value in variants.items()}
    out = {name: [] for name in ARMS}
    repeated = []
    weights = torch.stack((y0, 1 - y0, fg[:4096], bg[:4096], repeat_fg, repeat_bg), 1)
    for start in range(0, len(q), block):
        query = qc[start:start + block]
        sim = query @ rc.T
        shift = sim.max(1).values
        kernel = torch.exp((sim - shift[:, None]) / TEMPERATURE)
        values = kernel @ weights
        denominator = kernel.sum(1).clamp_min(1e-30)
        out['whole'].append(values[:, 0] / denominator)
        # repeat_fg+repeat_bg is exactly one per parent, so the same denominator
        # implements the nearest-copy null with the original reduction order.
        repeated.append(values[:, 4] / denominator)
        whole_logs = values[:, 2:4].clamp_min(1e-30).log() + shift[:, None] / TEMPERATURE
        for name, ref in focus_features.items():
            fsim = query @ ref.T
            fshift = fsim.max(1).values
            fk = torch.exp((fsim - fshift[:, None]) / TEMPERATURE)
            fvalues = fk @ torch.stack((fg[4096:], bg[4096:]), 1)
            flog = fvalues.clamp_min(1e-30).log() + fshift[:, None] / TEMPERATURE
            logs = torch.logaddexp(whole_logs, flog)
            out[name].append(torch.sigmoid(logs[:, 0] - logs[:, 1]))
    result = {name: torch.cat(parts) for name, parts in out.items()}
    null = torch.cat(repeated)
    assert all(torch.isfinite(value).all() for value in result.values())
    assert float((null - result['whole']).abs().max()) <= TOL
    return result, null


def compute(b, cfg):
    started, counters = time.monotonic(), {}
    with b1.legal_inference(b), deny_paths([b['baseline']['path']]):
        y0, fg, bg, repeat_fg, repeat_bg, measure = reference_measure(b)
        partition = b['partition']
        assert pc.sha(partition['path']) == partition['sha256']
        with np.load(partition['path'], allow_pickle=False) as z:
            labels = torch.from_numpy(z['regions64'].copy().astype(np.int64)).to(lab.DEV)
        assert labels.shape == (4096,) and labels.min() >= 0 and labels.max() < 64
        raw_r = b1.read_raw(b['fit']['reference_raw'], counters)
        raw_q = b1.read_raw(b['fit']['query_raw_identity'], counters)
        raw_focus = b1.read_raw(b['focus_raw'], counters)
        kernel_start = time.monotonic()
        fields, null = observation_fields(raw_r, raw_focus, raw_q, tuple(b['focus']['box_xyxy']),
                                          y0, fg, bg, repeat_fg, repeat_bg, cfg['query_block'])
        if lab.DEV.type == 'mps':
            torch.mps.synchronize()
        counters['kernel_seconds'] = time.monotonic() - kernel_start
        token_error = float((fields['whole'] - null).abs().max())
        region_fields = {name: b1.region_mean(value, labels) for name, value in fields.items()}
        null_region = b1.region_mean(null, labels)
        masks = {name: pc.readout(value.cpu()) for name, value in region_fields.items()}
        null_mask = pc.readout(null_region.cpu())
        xor = int(np.count_nonzero(null_mask ^ masks['whole']))
        assert xor == 0, 'Nearest-copy null changed the whole natural mask'
        artifacts = {**{'field/' + name: value.cpu().numpy() for name, value in region_fields.items()},
                     **{'token/' + name: value.cpu().numpy() for name, value in fields.items()},
                     **{'mask/' + name: pc.pack(value) for name, value in masks.items()},
                     'regions64': labels.cpu().numpy().astype(np.int16)}
    return artifacts, {'physical_measure': measure, 'nearest_token_max_abs': token_error,
                       'nearest_region_max_abs': float((region_fields['whole'] - null_region).abs().max()),
                       'nearest_mask_xor': xor, 'query_partition_sha256': partition['sha256'],
                       'query_partition_index': partition['index'], 'new_encoder_calls': 0,
                       'query_GT_reads': 0, 'baseline_mask_reads': 0, **counters,
                       'compute_seconds': time.monotonic() - started}


def produce(smoke=False):
    cfg = check_config()
    setup(cfg['device'], cfg['threads'])
    rows = pc.read(OUT / 'manifest.json')
    selected = [b for b in rows if not smoke or pc.key(b) in cfg['smoke_indices']]
    started, records = time.monotonic(), []
    for number, b in enumerate(selected, 1):
        key = pc.key(b)
        receipt_path, path = OUT / 'records' / (key + '.json'), OUT / 'artifacts' / (key + '.npz')
        if receipt_path.exists():
            rec = pc.read(receipt_path)
            assert rec['config_sha256'] == pc.sha(OUT / 'config.json')
            assert pc.sha(rec['artifact_path']) == rec['artifact_sha256']
        else:
            artifacts, diagnostics = compute(b, cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **artifacts)
            temporary.replace(path)
            rec = {'index': key, 'episode_id': b['row']['episode_id'], 'dataset': b['dataset'],
                   'artifact_path': str(path), 'artifact_sha256': pc.sha(path),
                   'config_sha256': pc.sha(OUT / 'config.json'), 'diagnostics': diagnostics}
            pc.write(receipt_path, rec)
        records.append(rec)
        if number <= 2 or number % 10 == 0:
            progress = {'state': 'NATURAL_PRODUCTION', 'completed': number, 'n': len(selected),
                        'seconds': time.monotonic() - started, 'latest_compute_seconds': rec['diagnostics']['compute_seconds']}
            pc.write(OUT / 'activity.json', progress)
            print(progress, flush=True)
    check_config()
    state = 'E3_NATURAL_SMOKE_COMPLETE' if smoke else 'ALL_E3_NATURAL_PREDICTIONS_SEALED'
    seal = {'state': state, 'n': len(records), 'config_sha256': pc.sha(OUT / 'config.json'),
            'manifest_sha256': cfg['manifest_sha256'], 'production_wall_seconds': time.monotonic() - started,
            'device': cfg['device'], 'threads': cfg['threads'], 'records': records,
            'new_encoder_calls': 0, 'query_GT_reads': 0, 'baseline_mask_reads': 0,
            'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    pc.write(OUT / ('smoke.json' if smoke else 'sealed.json'), seal)
    elapsed = sum(r['diagnostics']['compute_seconds'] for r in records)
    print({'state': state, 'n': len(records), 'mean_compute_seconds': elapsed / len(records),
           'projected_200_minutes': elapsed / len(records) * 200 / 60}, flush=True)


def summarize(records, bootstrap=1000):
    rng, result = np.random.default_rng(20261012), {}
    for dataset in ('deepglobe_road', 'paco_part'):
        rs = [r for r in records if r['dataset'] == dataset]
        photos = sorted({r['query_photo'] for r in rs})
        slots = sorted({(str(r['fold']), str(r['class_id'])) for r in rs})
        pi, si = {x: i for i, x in enumerate(photos)}, {x: i for i, x in enumerate(slots)}
        indices = ([pi[r['query_photo']] for r in rs], [si[(str(r['fold']), str(r['class_id']))] for r in rs])
        shape = len(photos), len(slots)
        presence = np.zeros(shape)
        np.add.at(presence, indices, 1)
        weights = np.stack([np.bincount(rng.integers(0, len(photos), len(photos)), minlength=len(photos))
                            for _ in range(bootstrap)])
        represented = (weights @ presence) > 0
        modes = {}
        for suffix, mode, names in (('', 'natural', (*CONTROLS, *ARMS)),
                                    ('/oracle_threshold', 'strict_oracle', ARMS)):
            table, boot = {}, {}
            for arm in names:
                ms = [r['metrics'][arm + suffix] for r in rs]
                inter, union = np.zeros(shape), np.zeros(shape)
                np.add.at(inter, indices, [m['I'] for m in ms])
                np.add.at(union, indices, [m['U'] for m in ms])
                values = 100 * (weights @ inter) / np.maximum(weights @ union, 1)
                boot[arm] = (values * represented).sum(1) / represented.sum(1)
                table[arm] = {'class_miou': float(np.mean(100 * inter.sum(0) / np.maximum(union.sum(0), 1))),
                              'episode_mean_iou': float(np.mean([m['IoU'] for m in ms])),
                              'FN_GT': sum(m['FN'] for m in ms) / max(sum(m['GT'] for m in ms), 1),
                              'FP_GT': sum(m['FP'] for m in ms) / max(sum(m['GT'] for m in ms), 1)}
            comparisons = [('actual', 'whole'), ('actual', 'derived'), ('derived', 'whole')]
            if not suffix:
                comparisons += [('actual', name) for name in CONTROLS]
            contrasts = {a + '-' + b: {'delta_pp': table[a]['class_miou'] - table[b]['class_miou'],
                          'paired_query_photo_bootstrap95_pp': np.quantile(boot[a] - boot[b], [.025, .975]).tolist()}
                         for a, b in comparisons}
            modes[mode] = {'arms': table, 'contrasts': contrasts}
        edits = {arm: {base: {k: sum(r['transitions'][arm][base][k] for r in rs)
                              for k in ('added_TP', 'added_FP', 'removed_TP', 'removed_FP')}
                       for base in ('whole', 'derived')} for arm in ('derived', 'actual')}
        result[dataset] = {'n': len(rs), 'query_photo_n': len(photos), 'class_slot_n': len(slots),
                           'transitions': edits, **modes}
    return result


def score():
    cfg = check_config()
    setup('cpu', cfg['threads'])
    seal = pc.read(OUT / 'sealed.json')
    rows = pc.read(OUT / 'manifest.json')
    assert seal['state'] == 'ALL_E3_NATURAL_PREDICTIONS_SEALED' and seal['n'] == len(rows) == 200
    assert seal['config_sha256'] == pc.sha(OUT / 'config.json')
    receipts = {r['index']: r for r in seal['records']}
    for b in rows:
        rec = receipts[pc.key(b)]
        assert pc.sha(rec['artifact_path']) == rec['artifact_sha256']
        assert pc.sha(b['baseline']['path']) == b['baseline']['sha256']
    # Every natural prediction has been verified before the first GT/control read.
    started, records = time.monotonic(), []
    for i, b in enumerate(rows, 1):
        rec, row = receipts[pc.key(b)], b['row']
        with Image.open(row['query_mask_path']) as image:
            raw = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
        assert tensor_hash(raw) == row['query_mask_hash']
        gt = F.interpolate(torch.from_numpy(raw)[None, None].float(), (1024, 1024), mode='nearest')[0, 0].numpy() > .5
        valid = np.ones_like(gt, dtype=bool)
        metrics, thresholds, transitions = {}, {}, {}
        with np.load(rec['artifact_path'], allow_pickle=False) as z, np.load(b['baseline']['path'], allow_pickle=False) as old:
            masks = {a: pc.unpack(z['mask/' + a]) for a in ARMS}
            masks.update({a: pc.unpack(old[key]) for a, key in CONTROLS.items()})
            for a, pred in masks.items():
                metrics[a] = pr.metric(pred, gt, valid)
            for a in ARMS:
                oracle, threshold = b1.threshold_oracle(z['field/' + a], gt, valid)
                metrics[a + '/oracle_threshold'] = pr.metric(oracle, gt, valid)
                thresholds[a] = threshold
            for a in ('derived', 'actual'):
                transitions[a] = {}
                for base in ('whole', 'derived'):
                    added, removed = masks[a] & ~masks[base], masks[base] & ~masks[a]
                    transitions[a][base] = {'added_TP': int((added & gt).sum()), 'added_FP': int((added & ~gt).sum()),
                                            'removed_TP': int((removed & gt).sum()), 'removed_FP': int((removed & ~gt).sum())}
        records.append({'index': pc.key(b), 'episode_id': row['episode_id'], 'dataset': b['dataset'],
                        'fold': row['fold'], 'class_id': row['global_class_id'],
                        'query_photo': str(row['query_photo_id']), 'metrics': metrics,
                        'oracle_thresholds': thresholds, 'transitions': transitions})
        if i <= 2 or i % 25 == 0:
            print({'state': 'SCORING', 'completed': i, 'n': 200}, flush=True)
    summary = summarize(records)
    cost = {'compute_seconds': sum(r['diagnostics']['compute_seconds'] for r in receipts.values()),
            'scoring_seconds': time.monotonic() - started, 'new_encoder_calls': 0,
            'logical_payload_bytes': sum(r['diagnostics']['logical_payload_bytes'] for r in receipts.values()),
            'peak_RSS_bytes': seal['peak_RSS_bytes']}
    checks = {'nearest_token_max_abs': max(r['diagnostics']['nearest_token_max_abs'] for r in receipts.values()),
              'nearest_region_max_abs': max(r['diagnostics']['nearest_region_max_abs'] for r in receipts.values()),
              'nearest_mask_xor': sum(r['diagnostics']['nearest_mask_xor'] for r in receipts.values())}
    pc.write(OUT / 'report.json', {'state': 'COMPLETE', 'task': 'E3', 'n': 200, 'summary': summary,
                                  'records': records, 'cost': cost, 'controls': checks,
                                  'seal_sha256': pc.sha(OUT / 'sealed.json'), 'config_sha256': pc.sha(OUT / 'config.json')})
    lines = ['# E3 同一参考真实 focus 的双角色 KDE', '',
             '固定PACO100/Deep100已有曝光任务；CLI1024，观察(fold,global_class)池化I/U后等权。',
             '三个实质臂仅改变参考观察。固定whole query、μ1、温度.05、同一个已绑定P0 K64和IND0读出；0新增编码。',
             'whole/derived/actual完整自然输出全部封存后评分，原同配对FoRIS与MEAN仅用于评分。',
             'nearest父token重复只检查退化；每父块和全图FG/BG质量均精确保持。此处不声称P0与B1曾使用相同K64。', '',
             '|域|n|FoRIS|MEAN|whole|derived|actual|actual−whole|actual−derived|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset, s in summary.items():
        table, contrasts = s['natural']['arms'], s['natural']['contrasts']
        values = [table[a]['class_miou'] for a in (*CONTROLS, *ARMS)]
        lines.append(f"|{dataset}|{s['n']}|" + '|'.join(f'{v:.3f}' for v in values) +
                     f"|{contrasts['actual-whole']['delta_pp']:+.3f}|{contrasts['actual-derived']['delta_pp']:+.3f}|")
    lines += ['', '|域|对照|自然差pp / 95%照片配对区间|strict oracle差pp / 区间|', '|---|---|---|---|']
    for dataset, s in summary.items():
        for name in ('actual-whole', 'actual-derived'):
            cells = []
            for mode in ('natural', 'strict_oracle'):
                d = s[mode]['contrasts'][name]
                lo, hi = d['paired_query_photo_bootstrap95_pp']
                cells.append(f"{d['delta_pp']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            lines.append(f'|{dataset}|{name}|' + '|'.join(cells) + '|')
    lines += ['', '阈值oracle只用于排序诊断，没有参与自然阈值或候选选择。区间按原query照片配对bootstrap 1000次；本队列仅为开发证据。',
              f'退化控制：`{checks}`。', f'缓存增量成本：`{cost}`；不代表RGB端到端推理延迟。',
              f'逐例I/U、TP/FP增删、阈值、完整配对区间：`{OUT / "report.json"}`。']
    path = ROOT / 'docs/research/x1012s_tasks/E3.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    print({'state': 'COMPLETE', 'n': 200, 'report': str(path), 'summary': summary}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'produce', 'score'))
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu', help='Frozen by prepare')
    parser.add_argument('--threads', type=int, choices=(2, 4), default=2, help='Frozen by prepare')
    parser.add_argument('--smoke', action='store_true', help='Produce first task in each domain, without scoring')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args)
    elif args.mode == 'produce':
        produce(args.smoke)
    else:
        score()
