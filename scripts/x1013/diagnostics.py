"""T8 prefix attribution and T9's fixed S-only assembly, without new arms."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import time
import numpy as np
from scipy import ndimage
from scipy.stats import rankdata
import torch
import common as c
import produce
import score

FORIS_PREFIXES = ('fg', 'bg', 'vote', 'prior', 'penalty', 'pre', 'FoRIS')
CANDIDATE_PREFIXES = ('c', 'c_mod', 'c_mod+V', 'c_mod+V+P', 'no_e', 'candidate')
ORDER = ('G1', 'F6a', 'F1', 'S-self', 'C1')


def t8_predict(binding):
    destination = c.prediction_path('T8', binding)
    if destination.exists():
        return str(binding['index'])
    global HOST, ADAPTER
    if 'HOST' not in globals():
        HOST, ADAPTER = c.rc.pc.host_cpu()
        torch.set_num_threads(2)
    from ics.foris import run_foris
    from ics.official_data import load_inputs
    from lvis_atomic_study import return_taps
    ADAPTER.bind(binding)
    functions = {'stage2': (HOST._part2_stage2_contrastive_score, ['sim_fg_hw']),
                 'stage3': (HOST._part3_clustering, ['vote_soft', 'prior']),
                 'penalty': (HOST._semantic_disagreement_penalty, ['penalty']),
                 'delta': (HOST._semantic_cluster_reweight_map, ['labels_t', 'delta_cluster'])}
    with c.rc.pc.inference_only(binding), return_taps(functions) as taps:
        reference, mask, query = load_inputs(binding['row'], c.ASSETS)
        final, got, _, _ = run_foris(HOST, reference, mask, query)
        s2, s3 = got['s2'], got['s3']
        voted = s2 + HOST.candidate_boost * (taps['stage3']['vote_soft'] - .5)
        assert torch.equal(voted + HOST.seed_cluster_boost * (taps['stage3']['prior'] - .5), s3)
        penalty = taps['penalty']['penalty'].clamp_max(HOST.semantic_penalty_max)
        delta = taps['delta']['delta_cluster'][taps['delta']['labels_t']].reshape(64, 64)
        assert torch.equal(s3 - penalty + delta, got['score'])
        prefixes = {'fg': taps['stage2']['sim_fg_hw'], 'bg': s2, 'vote': voted,
                    'prior': s3, 'penalty': s3 - penalty, 'pre': got['score']}
        predictions = {'foris/' + name: c.pack(HOST._binarize_response(value, target_hw=(1024, 1024)).numpy())
                       for name, value in prefixes.items()}
        predictions['foris/FoRIS'] = c.pack(final.numpy())
        if not np.array_equal(predictions['foris/FoRIS'], c.pack(c.rc.lab.arm_cli(binding, 'B').numpy())):
            c.write(c.task_path('T8') / 'gate_failure.json',
                    {'index': str(binding['index']), 'gate': 'full FoRIS exact replay'})
            raise RuntimeError(f"T8 full FoRIS identity failed: {binding['index']}")
        fields = c.load_fields(binding)
        kernel = fields['c32'].reshape(64, 64)
        modulated = fields['c_mod'].reshape(64, 64)
        # Prefixes expose the operations even when the final rule falls back.
        own = {'c': kernel, 'c_mod': modulated, 'c_mod+V': modulated + fields['addition_V'],
               'c_mod+V+P': fields['primary'],
               'no_e': c.solve(fields['primary'], np.ones_like(fields['explanation']), c.laplacian(fields)),
               'candidate': c.foreground_field(fields)}
        predictions.update({'own/' + name: c.pack(c.readout(value)) for name, value in own.items()})
        c.save_npz(destination, **predictions)
        if ADAPTER.encoder is not None or ADAPTER.encoder_calls:
            raise RuntimeError('T8 unexpectedly invoked an encoder')
    ADAPTER.memo.clear()
    return str(binding['index'])


def semantic_codes(binding, region):
    """COCO's original per-pixel category encoding, aligned with the query crop."""
    from PIL import Image
    query = c.ASSETS / binding['row']['query_path']
    matches = [path for path in binding['row'].get('source_files', {})
               if '/annotations/' in path and c.Path(path).stem == query.stem]
    path = c.ASSETS / matches[0] if matches else query.parent.parent / 'annotations' / query.parent.name / (query.stem + '.png')
    if not path.exists():
        raise FileNotFoundError(f'COCO semantic annotation unavailable: {path}')
    with Image.open(path) as opened:
        codes = np.asarray(opened).copy()
    crop = binding['row'].get('query_crop')
    if crop:
        x0, y0, x1, y1 = crop
        codes = codes[y0:y1, x0:x1]
    codes = np.asarray(Image.fromarray(codes).resize((1024, 1024), Image.Resampling.NEAREST))
    values, counts = np.unique(codes[region], return_counts=True)
    selected = np.argsort(-counts)[:5]
    return [{'code': int(values[i]), 'pixels': int(counts[i])} for i in selected]


def t8_describe(job):
    binding, group, base = job
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    with np.load(c.prediction_path('T0', binding), allow_pickle=False) as archive:
        candidate = c.unpack(archive['candidate']) & valid
    foris = c.rc.lab.arm_cli(binding, 'B').numpy() & valid
    regions = {'E_FP': candidate & ~foris & ~gt, 'E_FN': ~candidate & foris & gt,
               'TP': candidate & gt}
    with np.load(c.prediction_path('T8', binding), allow_pickle=False) as archive:
        predictions = {name: c.unpack(archive[name]) for name in archive.files}
    counts = {}
    for region_name in ('E_FP', 'E_FN'):
        region = regions[region_name]
        counts[region_name] = {'pixels': int(region.sum()),
            'foreground': {name: int(np.count_nonzero(mask & region)) for name, mask in predictions.items()}}
    # Connected components of the predicted foreground, with 8-neighbor adjacency.
    components, _ = ndimage.label(candidate, structure=np.ones((3, 3), bool))
    touching = np.unique(components[regions['TP']])
    touching = touching[touching != 0]
    connected = int(np.count_nonzero(regions['E_FP'] & np.isin(components, touching))) / max(int(regions['E_FP'].sum()), 1)
    arrays = {'e': c.up(fields['explanation']), 'c': c.up(fields['c32']),
              'c_mod': c.up(fields['c_mod']), 'V': c.up(fields['addition_V']), 'P': c.up(fields['addition_P'])}
    quantile_e = np.zeros((1024, 1024), np.float32)
    quantile_e[valid] = ((rankdata(arrays['e'][valid], method='average') - .5) / int(valid.sum())).astype(np.float32)
    measurements = {}
    for name, region in regions.items():
        measurements[name] = {'pixels': int(region.sum()),
            'e_relative_quantiles': np.percentile(quantile_e[region], [0, 10, 25, 50, 75, 90, 100]).tolist() if region.any() else None,
            'means': {key: float(values[region].mean()) if region.any() else None for key, values in arrays.items()}}
    oracle = score.strict_oracle(c.foreground_field(fields), gt, valid)
    record = dict(c.metadata(binding, fields), group=group, gt_fraction=float(gt.sum()) / 1048576,
                  reference_fraction=float(fields['reference_fraction']),
                  foris_iou=score.episode_iou(base, 'foris'), candidate_iou=score.episode_iou(base, 'candidate'),
                  candidate_oracle_iou=100 * oracle[0] / max(oracle[1], 1),
                  E_FP_connected_to_TP=connected, regions=measurements, prefix_counts=counts)
    if binding['dataset'] == 'coco':
        record['E_FP_semantic_top5'] = semantic_codes(binding, regions['E_FP'])
    return record


def run_t8(workers):
    out = c.task_path('T8')
    if (out / 'scored.json').exists():
        return
    began = time.monotonic()
    # Keep this optional post-E1 diagnostic on the selection half. The older
    # T8 path sampled both halves and would otherwise reopen report-half GT.
    base = [row for row in score.read_records('T0') if row['split'] == 'S']
    if not base:
        raise RuntimeError('T8 needs the completed T0 identity gate')
    all_bindings = {str(b['index']): b for b in c.rows()}
    rng = np.random.default_rng(c.SEED)
    selected = []
    for domain in ('coco', 'isic', 'suim', 'lvis'):
        eligible = [r for r in base if r['dataset'] == domain and
                    (domain != 'lvis' or r['gt'] / 1048576 > .2)]
        eligible.sort(key=lambda r: -((r['candidate'][1] - r['candidate'][0]) -
                                     (r['foris'][1] - r['foris'][0])))
        n = max(1, int(np.ceil(.1 * len(eligible))))
        tail, remaining = eligible[:n], eligible[n:]
        control = [remaining[i] for i in rng.choice(len(remaining), n, replace=False)]
        selected.extend((all_bindings[r['index']], group, r)
                        for group, values in (('tail', tail), ('control', control)) for r in values)
    c.write(out / 'selection.json', [{'index': str(b['index']), 'dataset': b['dataset'], 'group': group}
                                     for b, group, _ in selected])
    print(f'T8 replay: {len(selected)} tasks', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool:
        for number, _ in enumerate(pool.map(t8_predict, (b for b, _, _ in selected), chunksize=1), 1):
            if number % 25 == 0 or number == len(selected):
                print(f'T8 replay: {number}/{len(selected)}, {time.monotonic() - began:.1f}s', flush=True)
        records = list(pool.map(t8_describe, selected, chunksize=2))
    path = out / 'descriptions.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records))
    tables = {}
    for domain in ('coco', 'isic', 'suim', 'lvis'):
        cells = []
        for group in ('tail', 'control'):
            subset = [r for r in records if r['dataset'] == domain and r['group'] == group]
            for region in ('E_FP', 'E_FN'):
                pixels = sum(r['prefix_counts'][region]['pixels'] for r in subset)
                values = {name: sum(r['prefix_counts'][region]['foreground'][name] for r in subset) / max(pixels, 1)
                          for name in subset[0]['prefix_counts'][region]['foreground']}
                cells.append({'group': group, 'region': region, 'n': len(subset), 'pixels': pixels, **values})
        tables[domain] = cells
    c.write(out / 'prefix_tables.json', tables)
    from report import case_sheets
    c.initialize()
    paths = case_sheets(selected)
    c.write(out / 'scored.json', {'n': len(selected), 'scope': 'S only', 'T_scored': False,
                                'seconds': time.monotonic() - began,
                                'identity_gate': 'PASS', 'case_paths': paths})


def chain_field(fields, retained, binding, background_role=False):
    original_fallback = produce.fallback(fields)
    is_fallback = False if background_role else original_fallback and (
        'G1' not in retained or int(fields['touched']) >= 64)
    if is_fallback:
        return fields['c32'].reshape(64, 64), True
    field = c.solve(fields['primary'], fields['explanation'], c.laplacian(fields),
                    fields['guide'] if 'F6a' in retained else None)
    if 'F1' in retained:
        with np.load(c.task_path('T5') / 'clusters' / (c.key(binding) + '.npz'), allow_pickle=False) as archive:
            field = produce.cluster_mean(field, archive['labels'])
    return field, False


def chain_prediction(binding, retained):
    fields = c.load_fields(binding)
    front, is_fallback = chain_field(fields, retained, binding)
    prediction = c.readout(front)
    if 'S-self' in retained and not produce.fallback(fields):
        with np.load(c.task_path('T6') / 'fields' / (c.key(binding) + '.npz'), allow_pickle=False) as archive:
            bg = {name: archive[name] for name in archive.files}
        back, _ = chain_field(bg, retained, binding, background_role=True)
        prediction = produce.symmetric(front, back)[2]
    # C1's prespecified scope is original rho <= .9, including in a G1 chain.
    if 'C1' in retained and not produce.fallback(fields):
        prediction = produce.crf(binding, prediction)
    return c.pack(prediction)


def chain_worker(job):
    phase, binding, retained = job
    path = c.prediction_path('T9', binding).parent.parent / phase / (c.key(binding) + '.npz')
    if not path.exists():
        with c.rc.pc.inference_only(binding):
            c.save_npz(path, final=chain_prediction(binding, retained))
    return str(binding['index'])


def chain_score(job):
    phase, binding = job
    path = c.prediction_path('T9', binding).parent.parent / phase / (c.key(binding) + '.npz')
    with np.load(path, allow_pickle=False) as archive:
        prediction = c.unpack(archive['final'])
    gt, valid = c.truth(binding)
    return {'index': str(binding['index']), 'dataset': binding['dataset'], 'split': c.half(binding),
            'fold': binding['row']['fold'], 'cls': binding['row']['global_class_id'], 'photo': c.photo(binding),
            'final': c.iu(prediction, gt, valid)}


def run_chain_phase(phase, bindings, retained, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool:
        for number, _ in enumerate(pool.map(chain_worker, ((phase, b, retained) for b in bindings), chunksize=4), 1):
            if number % 1000 == 0 or number == len(bindings):
                print(f'T9 {phase} predict: {number}/{len(bindings)}', flush=True)
        # All masks for this phase exist before its GT is opened.
        records = list(pool.map(chain_score, ((phase, b) for b in bindings), chunksize=8))
    score.save_records(c.task_path('T9') / (phase + '_iu.jsonl.gz'), records)
    return records


def run_t9(workers):
    out = c.task_path('T9')
    if (out / 'scored.json').exists():
        return
    started = time.monotonic()
    for task in ('T0', 'T1', 'T5', 'T6'):
        if not (c.task_path(task) / 'scored.json').exists():
            raise RuntimeError(f'T9 dependency incomplete: {task}')
    bindings = c.rows()
    selection_rows = [b for b in bindings if c.half(b) == 'S']
    selection_base = [r for r in score.read_records('T0') if r['split'] == 'S']
    previous = score.domain_scores(selection_base, 'candidate')
    state_path = out / 'selection.json'
    state = c.read(state_path) if state_path.exists() else {'retained': [], 'steps': []}
    retained = state['retained']
    for step in state['steps']:
        previous = step['after'] if step['keep'] else step['before']
    for number, name in enumerate(ORDER):
        if number < len(state['steps']):
            continue
        proposed = retained + [name]
        print(f'T9 S trial {name}: {proposed}', flush=True)
        records = run_chain_phase('step_' + name, selection_rows, proposed, workers)
        after = score.domain_scores(records, 'final')
        deltas = {d: after[d] - previous[d] for d in c.DOMAINS}
        mean_delta, worst = float(np.mean(list(deltas.values()))), float(min(deltas.values()))
        keep = mean_delta >= 0 and worst >= -.3
        state['steps'].append({'name': name, 'before': previous, 'after': after,
                               'delta': deltas, 'equal_domain_delta': mean_delta,
                               'worst_domain_delta': worst, 'keep': bool(keep)})
        if keep:
            retained.append(name)
            previous = after
        c.write(state_path, state)
        print(f'T9 {name}: mean={mean_delta:+.4f}, min={worst:+.4f}, keep={keep}', flush=True)
    # Freeze the S-only decisions before opening any final T prediction/score.
    state['selection_complete'] = True
    c.write(state_path, state)
    records = run_chain_phase('final', bindings, retained, workers)
    score.save_records(out / 'iu.jsonl.gz', records)
    c.write(out / 'scored.json', {'n': len(records), 'seconds': time.monotonic() - started,
                                'identity_gate': 'N/A', 'retained': retained,
                                'T_evaluations': 1})
