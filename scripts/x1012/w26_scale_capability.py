"""W26 route C: one frozen, cache-only scale-capability experiment.

W24 trades scale between domains; W25 leaves oracle headroom but max-J fails.
W26-0's existing quantities fail, motivating exactly the A/B/C/D descriptors
in W26_scale_PLAN.md 5.2. No descriptor, model setting, or gate is changed after
looking at outcomes. Positive capability opens the single-descriptor rule in
section 6; failure follows the predeclared local-signal/closure reading.

produce: S-W26 only (W25's 1,285 + all remaining S COCO/ISIC = 1,750).
check: all three implementation gates in 5.3, before any outcome report.
report: weighted LODO GBR, binary AFTER CRF, five feature sets; the complete
        three-way BEFORE-CRF companion, references, and rank correlations.
rule: section 6 only if the full binary feature set passes (a), (b), and (c).

Only rc.independent's cached O/24 is used. Query GT is inaccessible while heads
and label-free descriptors run; it is opened afterwards only for I/U records.
Fallbacks keep SH's unrefined mask, never select a scale, and have null unused
descriptors. All generated run files are in ../cv_data/a/x1013_W26; the compact
S evidence is exported only after checks pass. No T or new-photo data is read
by produce/check/report, or by rule when the capability/rule gate fails.

Run from the repository root with ~/miniconda3/envs/aidemo/bin/python -B:
    scripts/x1012/w26_scale_capability.py produce [workers]
    scripts/x1012/w26_scale_capability.py check
    scripts/x1012/w26_scale_capability.py report
    scripts/x1012/w26_scale_capability.py rule
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/x1013'), str(ROOT / 'scripts/x1012')]
import common as c
import score

OUT = c.task_path('W26')
EVIDENCE = ROOT / 'evidence/research/w26_scale_capability_S.jsonl.gz'
PLAN = ROOT / 'docs/research/x1012_tasks/W26_scale_PLAN.md'
LAMS = (4, 16, 64)
PRE = {4: 'L4', 16: 'SH', 64: 'L64'}
POST = {16: 'SH+', 64: 'L64+'}
PIXELS = 1024 ** 2
GBR_PARAMS = dict(n_estimators=150, max_depth=2, learning_rate=.05,
                  subsample=.8, random_state=0)
EXPECTED = dict(zip(c.DOMAINS, (456, 150, 150, 150, 150, 110, 309, 105, 108, 62)))
EXPECTED_ACTIVE = dict(zip(c.DOMAINS, (456, 116, 150, 150, 146, 107, 309, 105, 0, 16)))
A_NAMES = ('log_a16', 'log_a64_over_a16', 'log_a4_over_a16', 'J4', 'J16',
           'J64', 'J64_minus_J16', 'J4_minus_J16', 'rho')
B_BASE = ('guided_iou', 'crf_iou', 'band_guided_abs_delta')
C_BASE = ('outside_largest_fraction', 'hole_fraction', 'isoperimetric', 'log_components')
D_BASE = ('ncut', 'cross_over_internal_weight', 'inner_minus_ring_field')


def family_names(base, lams):
    return (tuple(f'{name}_{lam}' for lam in lams for name in base) +
            tuple(f'{name}_delta{lam}_16' for lam in lams if lam != 16 for name in base))


FAMILIES = {'A': A_NAMES, 'B': family_names(B_BASE, (16, 64)),
            'C': family_names(C_BASE, LAMS), 'D': family_names(D_BASE, LAMS)}
FEATURE_SETS = {'A': 'A', 'A+B': 'AB', 'A+C': 'AC', 'A+D': 'AD', 'A+B+C+D': 'ABCD'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    c.write(OUT / name, obj)


def load(name):
    return c.read(OUT / name)


def compressed(name):
    with gzip.open(ROOT / f'evidence/research/{name}.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def sample():
    import w13_plugin as w13
    s = [b for b in c.rows() if c.half(b) == 'S']
    old = w13.subset(s, 150)
    ids = {str(b['index']) for b in old}
    rows = old + [b for b in s if b['dataset'] in ('coco', 'isic') and str(b['index']) not in ids]
    assert dict(Counter(b['dataset'] for b in rows)) == EXPECTED
    assert len(rows) == len({str(b['index']) for b in rows}) == 1750
    return rows, ids


def run_identity(rows):
    frozen = [ROOT / 'scripts/x1012' / name for name in
              ('r5_candidate.py', 'w23_simple_pipeline.py', 'w25_j_scale.py')]
    frozen += sorted((ROOT / 'scripts/x1013').glob('*.py'))
    frozen += [ROOT / 'docs/research/x1013_tasks/REPORT.md',
               ROOT / 'scripts/x1012s/photo_holdout.py']
    # No evaluation or holdout files are opened for this frozen-source snapshot.
    text = PLAN.read_text()
    return {'source_sha256': sha(__file__), 'plan_sections_1_6_sha256':
            hashlib.sha256(text.split('## 7 执行约束与交付')[0].encode()).hexdigest(),
            'frozen_source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in frozen},
            'ordered_indices': [str(b['index']) for b in rows],
            'counts': EXPECTED, 'active_counts': EXPECTED_ACTIVE,
            'feature_names': {k: list(v) for k, v in FAMILIES.items()}, 'gbr_parameters': GBR_PARAMS, 'seed': c.SEED,
            'descriptor_conventions': {
                'components': '8-connectivity', 'holes': 'scipy binary_fill_holes default 4-connectivity',
                'perimeter': 'm minus scipy binary_erosion once, default 4-connectivity',
                'band': 'inside/outside Euclidean distance to opposite mask <=16 pixels',
                'ring': 'scipy binary_dilation once, default 4-connectivity',
                'graph_edge_means': 'nonzero graph edges; symmetric internal edges counted in both directions',
                'empty_denominators': 'area clamped to 1 pixel, graph ratios to 1e-12; empty field mean=0'},
            'data_scope': 'S-W26 only; no T features/labels/evidence and no new-photo holdout'}


def verify_frozen():
    manifest = load('manifest.json')
    for path, digest in manifest['frozen_source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError(f'Frozen source changed: {path}')
    text = PLAN.read_text().split('## 7 执行约束与交付')[0]
    if hashlib.sha256(text.encode()).hexdigest() != manifest['plan_sections_1_6_sha256']:
        raise RuntimeError('Plan sections 1-6 changed; stop')


def initialize():
    c.initialize()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)


def heads(binding):
    """Exact W25 calculations, with fields retained for the frozen descriptors."""
    import torch
    import torch.nn.functional as F
    from w09_integration_lambda import solve
    from w24_sensitivity import mean_solve_lam
    rc, lab = c.rc, c.rc.lab
    counters = {}
    r, q, _, y, f = rc.independent(binding, counters)
    rho = float(f['reference_rho'][0])
    if rho > .9:
        field = lab.minmax(lab.up(torch.from_numpy(f['c32']).reshape(64, 64))).numpy()
        mask = field > .5
        return {lam: field for lam in LAMS}, {lam: mask for lam in LAMS}, {}, None, rho, counters
    c_mod, _ = rc.modulate(torch.from_numpy(f['c32']), torch.from_numpy(f['regions64'].astype(np.int64)),
                          f['ind0_32'], f['indn_32'], f['sim_max'], f['back_f'])
    graph = rc.graph(q, f['query_neighbors20']).tocsr()
    cov = y.numpy()
    qn, rn = F.normalize(q.float(), dim=1), F.normalize(r.float(), dim=1)
    fi = np.flatnonzero(cov >= .9)
    if not len(fi):
        fi = np.flatnonzero(cov == cov.max())
    guide = (qn @ F.normalize(rn[fi].mean(0), dim=0)).numpy()
    fg = torch.from_numpy(cov >= .5 if (cov >= .5).any() else cov == cov.max())
    mu_fg = F.normalize(rn[fg].mean(0), dim=0)
    bg = rn[~fg]
    if len(bg):
        hard = bg[(bg @ mu_fg).topk(max(1, int(.2 * len(bg)))).indices]
        mu_bg = F.normalize(hard.mean(0), dim=0)
        mu_bg = mu_bg - (mu_bg @ mu_fg) * mu_fg
        mu_bg = mu_bg / mu_bg.norm().clamp_min(1e-8)
    else:
        mu_bg = torch.zeros_like(mu_fg)
    p = (qn @ mu_fg - .55 * (qn @ mu_bg)).numpy().reshape(64, 64)
    up = lambda z: lab.up(torch.from_numpy(np.asarray(z, np.float32).reshape(64, 64))).numpy().astype(np.float64)
    fields, masks, j = {}, {}, {}
    for lam in LAMS:
        u = lab.minmax(torch.from_numpy(up(solve(c_mod.numpy(), f['log_e'], graph, lam)))).numpy()
        m = up(mean_solve_lam(p, guide, graph, lam))
        fields[lam] = .5 * m + .5 * u
        masks[lam] = fields[lam] > .5
        j[lam] = rc.quality(masks[lam], f['c32'], f['back_f'].astype(np.float64))
    return fields, masks, j, graph, rho, counters


def overlap(a, b):
    return float(np.count_nonzero(a & b) / max(np.count_nonzero(a | b), 1))


def shape_features(mask):
    from scipy import ndimage as ndi
    labels, count = ndi.label(mask, structure=np.ones((3, 3), bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    area = int(mask.sum())
    perimeter = int(np.count_nonzero(mask & ~ndi.binary_erosion(mask)))
    return [float((area - sizes.max(initial=0)) / max(area, 1)),
            float(np.count_nonzero(ndi.binary_fill_holes(mask) & ~mask) / max(area, 1)),
            float(perimeter ** 2 / (4 * np.pi * max(area, 1))), float(np.log(count + 1))]


def graph_features(field, weights):
    from scipy import ndimage as ndi
    token_field = field.reshape(64, 16, 64, 16).mean(axis=(1, 3))
    mask = token_field > .5
    t = mask.ravel()
    cross = weights[t][:, ~t]
    internal = weights[t][:, t]
    cut = float(cross.sum())
    vol = float(weights[t].sum())
    cross_mean = cut / max(cross.nnz, 1)
    internal_mean = float(internal.sum()) / max(internal.nnz, 1)
    ring = ndi.binary_dilation(mask) & ~mask
    contrast = (float(token_field[mask].mean()) - float(token_field[ring].mean())) if mask.any() and ring.any() else 0.
    return [cut / max(vol, 1e-12), cross_mean / max(internal_mean, 1e-12), contrast]


def add_family(destination, family, base, lams, values):
    vector = [x for lam in lams for x in values[lam]]
    vector += [values[lam][i] - values[16][i]
               for lam in lams if lam != 16 for i in range(len(base))]
    destination.update({family + '.' + name: float(v)
                        for name, v in zip(FAMILIES[family], vector)})


def descriptors(binding, fields, masks, refined, j, graph, rho):
    import torch
    import produce
    from scipy import ndimage as ndi
    from ics.official_data import decoded_rgb
    from utils.data import denormalize
    from w21_refine import guided
    a = {lam: max(int(masks[lam].sum()), 1) / PIXELS for lam in LAMS}
    vector = [np.log(a[16]), np.log(a[64] / a[16]), np.log(a[4] / a[16]),
              j[4], j[16], j[64], j[64] - j[16], j[4] - j[16], rho]
    desc = {'A.' + name: float(value) for name, value in zip(A_NAMES, vector)}
    host = produce.CRF_HOST
    host.set_target(decoded_rgb(c.ASSETS / binding['row']['query_path'], binding['row'].get('query_crop')))
    image = host._tgt_image.clone()
    host._tgt_image = host._orig_tgt_size = None
    rgb = denormalize(image).clamp(0, 1).float()
    b = {}
    for lam in (16, 64):
        mask = masks[lam]
        filt = guided(rgb, torch.from_numpy(fields[lam]).float(), r=8, eps=1e-3).numpy()
        band = ((mask & (ndi.distance_transform_edt(mask) <= 16)) |
                (~mask & (ndi.distance_transform_edt(~mask) <= 16)))
        b[lam] = [overlap(mask, filt > .5), overlap(mask, refined[lam]),
                  float(np.abs(filt - fields[lam])[band].mean()) if band.any() else 0.]
    add_family(desc, 'B', B_BASE, (16, 64), b)
    add_family(desc, 'C', C_BASE, LAMS, {lam: shape_features(masks[lam]) for lam in LAMS})
    weights = -graph.copy()
    weights.setdiag(0)
    weights.eliminate_zeros()
    assert weights.nnz == 0 or weights.data.min() >= 0
    add_family(desc, 'D', D_BASE, LAMS, {lam: graph_features(fields[lam], weights) for lam in LAMS})
    if not np.isfinite(list(desc.values())).all():
        raise RuntimeError('Nonfinite descriptor')
    return desc


def one(binding):
    import produce
    import psutil
    if c.half(binding) != 'S':
        raise RuntimeError('W26-1 cannot read a T episode')
    started = time.perf_counter()
    with c.rc.pc.inference_only(binding):
        fields, masks, j, graph, rho, counters = heads(binding)
        headed = time.perf_counter()
        if rho > .9:
            refined = {16: masks[16], 64: masks[64]}
            desc = {family + '.' + name: None for family, names in FAMILIES.items() for name in names}
            crfed = time.perf_counter()
        else:
            refined = {lam: produce.crf(binding, masks[lam]) for lam in (16, 64)}
            crfed = time.perf_counter()
            desc = descriptors(binding, fields, masks, refined, j, graph, rho)
    described = time.perf_counter()
    # Annotation access starts only after every mask and descriptor is fixed.
    gt, valid = c.truth(binding)
    row = binding['row']
    record = {'index': str(binding['index']), 'dataset': binding['dataset'], 'fold': row['fold'],
              'cls': row['global_class_id'], 'photo': c.photo(binding), 'split': 'S',
              'gt': int(gt.sum()), 'rho': rho, 'fallback': rho > .9, 'descriptors': desc,
              'raw_area': {str(lam): int(masks[lam].sum()) for lam in LAMS}, 'raw_counters': counters,
              'encoder_calls': int(getattr(getattr(produce, 'CRF_ADAPTER', None), 'encoder_calls', 0)),
              'sec': {'heads': headed - started, 'crf': crfed - headed, 'descriptors': described - crfed,
                      'total': time.perf_counter() - started},
              'worker_rss_mib': psutil.Process().memory_info().rss / 1024 ** 2}
    record.update({PRE[lam]: c.iu(masks[lam], gt, valid) for lam in LAMS})
    record.update({POST[lam]: c.iu(refined[lam], gt, valid) for lam in (16, 64)})
    assert record['encoder_calls'] == 0 and counters['raw_requests'] == 2
    return record


def records():
    manifest = load('manifest.json')
    path = OUT / 'episodes.jsonl'
    indexed = {}
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e['index'] in indexed:
            raise RuntimeError(f'Duplicate record {e["index"]}')
        indexed[e['index']] = e
    if set(indexed) != set(manifest['ordered_indices']):
        raise RuntimeError('Production incomplete or wrong sample; no analysis allowed')
    rows = [indexed[i] for i in manifest['ordered_indices']]
    active_counts = Counter(e['dataset'] for e in rows if not e['fallback'])
    assert {d: active_counts[d] for d in c.DOMAINS} == EXPECTED_ACTIVE
    assert all(e['split'] == 'S' and e['encoder_calls'] == 0 for e in rows)
    return rows


def resource():
    import psutil
    m = psutil.virtual_memory()
    return {'cpu_percent': psutil.cpu_percent(interval=.2), 'available_gib': m.available / 1024 ** 3,
            'memory_percent': m.percent, 'swap': psutil.swap_memory()._asdict()}


def produce_all(workers):
    if not 1 <= workers <= 4:
        raise ValueError('W26 permits 1-4 workers only')
    OUT.mkdir(parents=True, exist_ok=True)
    rows, _ = sample()
    identity = run_identity(rows)
    if (OUT / 'manifest.json').exists():
        if identity != load('manifest.json'):
            raise RuntimeError('Resume identity mismatch; preserve existing run and stop')
    else:
        save('manifest.json', identity)
    top = subprocess.run(['top', '-l', '1', '-n', '0'], capture_output=True, text=True, check=True).stdout
    stamp = time.time_ns()
    (OUT / f'top_{stamp}.txt').write_text(top)
    print(top, flush=True)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(line)['index'] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in rows if str(b['index']) not in done]
    started = time.perf_counter()
    snapshots = [{'completed': len(done), **resource()}]
    print(f'S-W26 {len(rows)}, saved {len(done)}, todo {len(todo)}, workers {workers}, torch threads 2', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize) as pool, path.open('a') as stream:
        for n, result in enumerate(pool.map(one, todo, chunksize=1), 1):
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            if n % 20 == 0 or n == len(todo):
                os.fsync(stream.fileno())
                snapshots.append({'completed': len(done) + n, **resource()})
                print(f'produced {len(done) + n}/{len(rows)}, elapsed {time.perf_counter() - started:.1f}s, '
                      f'CPU {snapshots[-1]["cpu_percent"]:.1f}%, available {snapshots[-1]["available_gib"]:.2f} GiB', flush=True)
                save(f'resources_{stamp}.json', {'workers': workers, 'torch_threads': 2, 'snapshots': snapshots})
    verify_frozen()
    complete = records()
    save(f'production_{stamp}.json', {'n': len(complete), 'produced': len(todo), 'workers': workers,
         'torch_threads': 2, 'wall_seconds': time.perf_counter() - started, 'started_unix_ns': stamp,
         'worker_rss_mib_max': max(e['worker_rss_mib'] for e in complete),
         'episode_seconds_mean': float(np.mean([e['sec']['total'] for e in complete]))})
    print('Production complete. Run check before report.', flush=True)


def iou(values):
    return values[0] / max(values[1], 1)


def matrix(rows, families):
    names = [family + '.' + name for family in families for name in FAMILIES[family]]
    return np.asarray([[e['descriptors'][name] for name in names] for e in rows], np.float64)


def training_weights(rows, arm):
    weights = np.asarray([e[arm][1] for e in rows], np.float64)
    for domain in sorted({e['dataset'] for e in rows}):
        ids = np.asarray([e['dataset'] == domain for e in rows])
        if weights[ids].sum() <= 0:
            raise RuntimeError('Training domain has zero union weight')
        weights[ids] /= weights[ids].sum()
    return weights


def lodo(rows, families, *, post=True, model='gbr', weighted=True):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    arms = POST if post else PRE
    base = arms[16]
    eligible = [e for e in rows if not e['fallback']]
    picks = {e['index']: 16 for e in rows}
    predictions = {}
    for domain in sorted({e['dataset'] for e in eligible}):
        tr = [e for e in eligible if e['dataset'] != domain]
        te = [e for e in eligible if e['dataset'] == domain]
        xtr, xte = matrix(tr, families), matrix(te, families)
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
        weight = training_weights(tr, base) if weighted else None
        pred = {}
        for lam in (64,) if post else (4, 64):
            m = Ridge(1.0) if model == 'ridge' else GradientBoostingRegressor(**GBR_PARAMS)
            target = [iou(e[arms[lam]]) - iou(e[base]) for e in tr]
            m.fit((xtr - mu) / sd, target, sample_weight=weight)
            pred[lam] = m.predict((xte - mu) / sd)
        for n, e in enumerate(te):
            pick = 64 if pred[64][n] > 0 else 16
            if not post:
                pick = max((0., 16), (pred[4][n], 4), (pred[64][n], 64))[1]
            picks[e['index']] = pick
            predictions[e['index']] = {str(lam): float(values[n]) for lam, values in pred.items()}
    return picks, predictions


def selected(rows, picks, post):
    arms = POST if post else PRE
    return [dict(e, sel=e[arms[picks[e['index']]]]) for e in rows]


def check():
    verify_frozen()
    rows = records()
    old = compressed('w25_j_scale_S')
    sh = {e['index']: e for e in compressed('w23_simple_pipeline_S')}
    current = {e['index']: e for e in rows}
    w = [current[e['index']] for e in old]
    assert len(w) == 1285
    same_pre = sum(e['SH'][:2] == current[e['index']]['SH'][:2] for e in old)
    same_post = sum(e['SH+'][:2] == sh[e['index']]['SH+'][:2] for e in w)
    result = {'records_sha256': sha(OUT / 'episodes.jsonl'), 'n': len(w),
              'm16_equal_W25': same_pre, 'm16plus_equal_W23': same_post, 'minimum_matches': 1280}
    print(f'Check 1: {same_pre}/1285 (required >=1280)', flush=True)
    print(f'Check 2: {same_post}/1285 (required >=1280)', flush=True)
    if same_pre < 1280 or same_post < 1280:
        result['passed'] = False
        save('checks.json', result)
        raise RuntimeError('Mask implementation gate failed; stop before analysis')
    means = {}
    for model in ('ridge', 'gbr'):
        picks, _ = lodo(w, 'A', post=False, model=model, weighted=False)
        chosen = selected(w, picks, False)
        deltas = {d: score.miou([e for e in chosen if e['dataset'] == d], 'sel') -
                     score.miou([e for e in chosen if e['dataset'] == d], 'SH') for d in c.DOMAINS}
        means[model] = {'mean_pp': float(np.mean(list(deltas.values()))), 'domains_pp': deltas}
    passed = abs(means['ridge']['mean_pp'] - (-.19)) <= .05 and abs(means['gbr']['mean_pp'] - (-.10)) <= .05
    result.update(reproduction=means, expected_means_pp={'ridge': -.19, 'gbr': -.10}, tolerance_pp=.05, passed=passed)
    save('checks.json', result)
    print('Check 3:', {k: v['mean_pp'] for k, v in means.items()}, 'PASS' if passed else 'FAIL', flush=True)
    if not passed:
        raise RuntimeError('Selector reproduction gate failed; stop before analysis')
    with gzip.open(EVIDENCE, 'wt') as stream:
        for e in rows:
            stream.write(json.dumps(e, ensure_ascii=False, allow_nan=False) + '\n')
    print('All three checks passed; S evidence exported. Analysis is now allowed.', flush=True)


def require_checks():
    verify_frozen()
    checks = load('checks.json')
    if not checks['passed'] or checks['records_sha256'] != sha(OUT / 'episodes.jsonl'):
        raise RuntimeError('All three checks must pass for these exact records before analysis')


def effect(rows, a, b, rng):
    lo, hi = score.paired_ci(rows, a, b, rng)
    return {'pp': score.miou(rows, a) - score.miou(rows, b), 'lo': lo, 'hi': hi}


def reading(summary):
    table = summary['domains']
    a = summary['mean_pp'] >= .20
    b = all(table[d]['sel_minus_SH']['pp'] >= .50 and
            table[d]['sel_minus_FoRIS']['pp'] >= 0 for d in ('coco', 'isic'))
    others = [d for d in c.DOMAINS if d not in ('coco', 'isic')]
    cc = all(table[d]['sel_minus_SH']['pp'] >= -.30 and table[d]['sel_minus_SH']['hi'] >= 0 for d in others)
    return {'a': a, 'b': b, 'c': cc, 'passed': a and b and cc}


def reference_rows(rows, post, rng):
    arms = POST if post else PRE
    base = arms[16]
    per_domain = {}
    augmented = []
    for d in c.DOMAINS:
        dr = [e for e in rows if e['dataset'] == d]
        # Fixed lambda is chosen by the required pooled metric, never episode mean.
        best = max(arms, key=lambda lam: (score.miou(dr, arms[lam]), lam == 16))
        for e in dr:
            oracle = max(arms, key=lambda lam: (iou(e[arms[lam]]), lam == 16))
            augmented.append(dict(e, oracle=e[arms[oracle]], domain_fixed=e[arms[best]]))
        rr = augmented[-len(dr):]
        per_domain[d] = {'oracle_minus_SH': effect(rr, 'oracle', base, rng),
                         'fixed_minus_SH': effect(rr, 'domain_fixed', base, rng),
                         'fixed_lambda': best, 'fixed_miou': score.miou(rr, 'domain_fixed'),
                         'SH_miou': score.miou(rr, base)}
    by_index = {e['index']: e for e in augmented}
    return [by_index[e['index']] for e in rows], per_domain


def correlations(rows, post):
    import warnings
    from scipy.stats import spearmanr
    arms = POST if post else PRE
    result = {}
    for family in 'BCD':
        for name in FAMILIES[family]:
            key = family + '.' + name
            result[key] = {}
            for d in c.DOMAINS:
                dr = [e for e in rows if e['dataset'] == d and not e['fallback']]
                if not dr:
                    result[key][d] = None
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    v = spearmanr([e['descriptors'][key] for e in dr],
                                  [iou(e[arms[64]]) - iou(e[arms[16]]) for e in dr]).statistic
                result[key][d] = float(v) if np.isfinite(v) else None
    return result


def analyse(rows, post, rng):
    arms = POST if post else PRE
    base = arms[16]
    rows, references = reference_rows(rows, post, rng)
    result = {'references': references, 'feature_sets': {}}
    for label, families in FEATURE_SETS.items():
        picks, predictions = lodo(rows, families, post=post)
        chosen = selected(rows, picks, post)
        domains = {}
        for d in c.DOMAINS:
            dr = [e for e in chosen if e['dataset'] == d]
            active = [e for e in dr if not e['fallback']]
            domains[d] = {'sel_minus_SH': effect(dr, 'sel', base, rng),
                          'sel_minus_fixed': effect(dr, 'sel', 'domain_fixed', rng),
                          'p64_all': sum(picks[e['index']] == 64 for e in dr) / len(dr),
                          'p64_active': sum(picks[e['index']] == 64 for e in active) / len(active) if active else 0.,
                          'p4_all': sum(picks[e['index']] == 4 for e in dr) / len(dr),
                          'n': len(dr), 'active_n': len(active)}
            if d in ('coco', 'isic'):
                domains[d]['sel_minus_FoRIS'] = effect(dr, 'sel', 'foris', rng)
        summary = {'domains': domains, 'mean_pp': float(np.mean([v['sel_minus_SH']['pp'] for v in domains.values()])),
                   'picks': picks, 'predictions': predictions}
        if post:
            summary['reading'] = reading(summary)
        result['feature_sets'][label] = summary
        print(f'Finished {"post" if post else "pre"}-CRF {label}', flush=True)
    result['spearman'] = correlations(rows, post)
    return result


def cell(value):
    return f'{value["pp"]:+.2f} [{value["lo"]:+.2f}, {value["hi"]:+.2f}]'


def markdown(result):
    checks = load('checks.json')
    lines = ['## 结果（W26-1，2026-10-14）', '',
             'S-W26：1,750 例，非回退 1,555 例；只读 S，编码器调用 0。所有差值均为 pp，区间为 '
             '`score.paired_ci` 的照片成组 95% bootstrap（500 次，rng=`c.SEED`）。'
             '按 (fold,class) 池化 I/U 后槽等权，十域均值等权；Fundus 的尺度差值恒为 0。', '',
             f'实现检查 1：m16 对 W25 SH，{checks["m16_equal_W25"]}/1,285 ≥ 1,280，PASS。',
             f'实现检查 2：m16+ 对 W23 SH+，{checks["m16plus_equal_W23"]}/1,285 ≥ 1,280，PASS。',
             f'实现检查 3：W25 样本、仅 A、CRF 前三选一、不加权留一域，ridge '
             f'{checks["reproduction"]["ridge"]["mean_pp"]:+.4f}、GBR '
             f'{checks["reproduction"]["gbr"]["mean_pp"]:+.4f}；分别复现 −0.19/−0.10（±0.05），PASS。', '',
             '三项检查全部通过后才运行以下分析；预测标准化仅用训练折，GBR 固定 150 棵、深 2、'
             '学习率 .05、subsample .8、random_state 0。权重为该读出阶段 λ16 的 GT 并集像素，域内归一化。']
    for mode, title in (('binary_post', '主报告：CRF 后二选一 {16,64}'),
                        ('ternary_pre', '次要报告：CRF 前三选一 {4,16,64}')):
        section = result[mode]
        lines += ['', f'### {title}', '', 'sel − SH（主报告中的 sel/SH 均带 CRF）', '',
                  '| 域 | ' + ' | '.join(FEATURE_SETS) + ' |',
                  '|---|' + '---:|' * len(FEATURE_SETS)]
        for d, name in zip(c.DOMAINS, c.DOMAIN_NAMES):
            lines.append('| ' + name + ' | ' + ' | '.join(cell(s['domains'][d]['sel_minus_SH'])
                         for s in section['feature_sets'].values()) + ' |')
        lines.append('| 十域均值 | ' + ' | '.join(f'{s["mean_pp"]:+.3f}' for s in section['feature_sets'].values()) + ' |')
        lines += ['', '| 特征集 | COCO sel − FoRIS | ISIC sel − FoRIS |', '|---|---:|---:|']
        for label, summary in section['feature_sets'].items():
            lines.append(f'| {label} | {cell(summary["domains"]["coco"]["sel_minus_FoRIS"])} | '
                         f'{cell(summary["domains"]["isic"]["sel_minus_FoRIS"])} |')
        lines += ['', '参照行（每域固定 λ 是诊断，不是合法方法）：', '',
                  '| 域 | oracle − SH | 最优固定 λ | 固定 λ − SH | SH mIoU | 固定 λ mIoU |',
                  '|---|---:|---:|---:|---:|---:|']
        for d, name in zip(c.DOMAINS, c.DOMAIN_NAMES):
            ref = section['references'][d]
            lines.append(f'| {name} | {cell(ref["oracle_minus_SH"])} | {ref["fixed_lambda"]} | '
                         f'{cell(ref["fixed_minus_SH"])} | {ref["SH_miou"]:.2f} | {ref["fixed_miou"]:.2f} |')
        lines += ['', 'sel − 每域最优固定 λ（衡量域内逐例信息）：', '',
                  '| 域 | ' + ' | '.join(FEATURE_SETS) + ' |',
                  '|---|' + '---:|' * len(FEATURE_SETS)]
        for d, name in zip(c.DOMAINS, c.DOMAIN_NAMES):
            lines.append('| ' + name + ' | ' + ' | '.join(cell(s['domains'][d]['sel_minus_fixed'])
                         for s in section['feature_sets'].values()) + ' |')
        lines += ['', '选择 64 的比例：全域 / 非回退例；回退例固定 16。', '',
                  '| 域 | ' + ' | '.join(FEATURE_SETS) + ' |',
                  '|---|' + '---:|' * len(FEATURE_SETS)]
        for d, name in zip(c.DOMAINS, c.DOMAIN_NAMES):
            lines.append('| ' + name + ' | ' + ' | '.join(f'{100*s["domains"][d]["p64_all"]:.1f}% / '
                         f'{100*s["domains"][d]["p64_active"]:.1f}%' for s in section['feature_sets'].values()) + ' |')
        lines += ['', 'B/C/D 域内 Spearman，与该阶段 IoU64 − IoU16 计算；仅非回退例，'
                  '空域或常数列记 —，Road 仅 16 例只报告。', '',
                  '| 描述子 | ' + ' | '.join(c.DOMAIN_NAMES) + ' |',
                  '|---|' + '---:|' * len(c.DOMAINS)]
        for key, corr in section['spearman'].items():
            lines.append('| ' + key + ' | ' + ' | '.join(f'{corr[d]:+.3f}' if corr[d] is not None else '—'
                         for d in c.DOMAINS) + ' |')
    gate = result['gate']
    lines += ['', '### 按 5.5 的结论', '', f'完整 A+B+C+D：条件 (a) {gate["a"]}；'
              f'(b) {gate["b"]}；(c) {gate["c"]}。', '', result['conclusion'], '',
              '监督选择器只作能力诊断，不进入免训练方法。没有在结果后增加描述子、调 GBR 或更改门槛。', '',
              '### 运行与资源', '', '```json', json.dumps(result['runtime'], ensure_ascii=False, indent=2), '```', '',
              '代码：`scripts/x1012/w26_scale_capability.py`；证据：'
              '`evidence/research/w26_scale_capability_S.jsonl.gz`；完整结构化分析、选择记录、'
              '分阶段计时和资源采样：`../cv_data/a/x1013_W26/`。']
    return '\n'.join(lines) + '\n'


def report():
    require_checks()
    started = time.perf_counter()
    base = {e['index']: e for e in compressed('x1013_E1_iu')}
    rows = [dict(e, foris=base[e['index']]['foris']) for e in records()]
    rng = np.random.default_rng(c.SEED)
    result = {'records_sha256': sha(OUT / 'episodes.jsonl'),
              'binary_post': analyse(rows, True, rng), 'ternary_pre': analyse(rows, False, rng)}
    full = result['binary_post']['feature_sets']['A+B+C+D']
    gate = full['reading']
    local = [label for label in ('A+B', 'A+C', 'A+D')
             if result['binary_post']['feature_sets'][label]['reading']['b']]
    result['gate'] = gate
    result['local_signal_sets'] = local
    if gate['passed']:
        task_only = all(full['domains'][d]['sel_minus_fixed']['pp'] <= 0 for d in ('coco', 'isic'))
        result['conclusion'] = 'W26-1 通过，进入 W26-2。' + ('COCO、ISIC 收益均不超过每域固定 λ，'
                 '只支持任务级区分；不能声称逐例自适应，规则须给物理理由。' if task_only else
                 '是否能形成免训练的单描述子阈值规则，按第 6 节继续检验。')
    elif local:
        result['conclusion'] = 'W26-1 未通过；' + '、'.join(local) + ' 单独满足 (b)，记录局部信号后停止，交回用户决定。W26-2 未执行，T 半未读。'
    else:
        result['conclusion'] = 'W26-1 未通过，也没有 A+X 单族满足 (b)。按 5.5 关闭路线 C；'
        result['conclusion'] += 'SH+ 维持主方法，论文维持“无一域显著低于 FoRIS”，服务器计划不变。W26-2 未执行，T 半未读。'
    productions = [c.read(path) for path in sorted(OUT.glob('production_*.json'))]
    resources = [c.read(path) for path in sorted(OUT.glob('resources_*.json'))]
    snapshots = [r for run in resources for r in run['snapshots']]
    result['runtime'] = {'production_runs': productions, 'analysis_wall_seconds': time.perf_counter() - started,
                         'sampled_cpu_percent_max': max(r['cpu_percent'] for r in snapshots),
                         'sampled_available_gib_min': min(r['available_gib'] for r in snapshots),
                         'source_sha256': sha(__file__), 'frozen_sources_verified': True,
                         'encoder_calls': 0, 'T_read': False, 'new_photo_holdout_read': False}
    save('report.json', result)
    (OUT / 'result_section.md').write_text(markdown(result))
    print(result['conclusion'], flush=True)
    print('Gate:', gate, 'mean:', full['mean_pp'], flush=True)


def rule():
    require_checks()
    result = load('report.json')
    if result['records_sha256'] != sha(OUT / 'episodes.jsonl'):
        raise RuntimeError('Report does not match current records')
    if not result['gate']['passed']:
        save('rule.json', {'status': 'NOT_ENTERED', 'reason': result['conclusion'],
                           'T_read': False, 'new_photo_holdout_read': False})
        print(result['conclusion'], flush=True)
        return
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    base = {e['index']: e for e in compressed('x1013_E1_iu')}
    rows = [dict(e, foris=base[e['index']]['foris']) for e in records()]
    active = [e for e in rows if not e['fallback']]
    names = [f + '.' + name for f in 'ABCD' for name in FAMILIES[f]]
    x = matrix(active, 'ABCD')
    mu, sd = x.mean(0), x.std(0) + 1e-9
    normalized = (x - mu) / sd
    y = np.asarray([iou(e['L64+']) - iou(e['SH+']) for e in active])
    weights = training_weights(active, 'SH+')
    model = GradientBoostingRegressor(**GBR_PARAMS).fit(normalized, y, sample_weight=weights)
    importance = permutation_importance(model, normalized, y, n_repeats=5,
                                       random_state=0, sample_weight=weights, n_jobs=1)
    winner = max((i for i, name in enumerate(names) if not name.startswith('A.')),
                 key=lambda i: importance.importances_mean[i])
    feature = names[winner]
    corr = result['binary_post']['spearman'][feature]
    positive = sum(v > 0 for v in corr.values() if v is not None)
    negative = sum(v < 0 for v in corr.values() if v is not None)
    if positive == negative:
        raise RuntimeError('Section 6 majority Spearman direction is tied; plan cannot define direction')
    direction = 1 if positive > negative else -1
    picks = {e['index']: 16 for e in rows}
    thresholds = {}
    for domain in c.DOMAINS:
        tr = [e for e in active if e['dataset'] != domain]
        te = [e for e in active if e['dataset'] == domain]
        threshold = fit_threshold(tr, feature, direction)
        thresholds[domain] = threshold
        for e in te:
            picks[e['index']] = 64 if direction * e['descriptors'][feature] > direction * threshold else 16
    rng = np.random.default_rng(c.SEED)
    rr, _ = reference_rows(rows, True, rng)
    chosen = selected(rr, picks, True)
    domains = {}
    for d in c.DOMAINS:
        dr = [e for e in chosen if e['dataset'] == d]
        domains[d] = {'sel_minus_SH': effect(dr, 'sel', 'SH+', rng),
                      'sel_minus_FoRIS': effect(dr, 'sel', 'foris', rng),
                      'sel_minus_fixed': effect(dr, 'sel', 'domain_fixed', rng),
                      'p64_all': sum(picks[e['index']] == 64 for e in dr) / len(dr)}
    summary = {'domains': domains, 'mean_pp': float(np.mean([v['sel_minus_SH']['pp'] for v in domains.values()]))}
    gate = reading(summary)
    fraction = sum(v == 64 for v in picks.values()) / len(rows)
    gate['nonconstant'] = .10 <= fraction <= .90
    gate['passed'] = gate['passed'] and gate['nonconstant']
    details = {'feature': feature, 'direction': '>' if direction > 0 else '<',
               'direction_sign': direction, 'spearman_positive_domains': positive,
               'spearman_negative_domains': negative,
               'permutation_importance': dict(zip(names, map(float, importance.importances_mean))),
               'importance_protocol': 'all active S-W26; weighted fitted GBR; default R2 score; 5 repeats; seed 0',
               'lodo_thresholds': thresholds, 'lodo': summary, 'fraction64_all': fraction,
               'gate': gate, 'T_read': False, 'new_photo_holdout_read': False}
    text = ['\n### W26-2：单描述子规则', '',
            f'置换重要性最大的 B/C/D 描述子：`{feature}`；方向 `{details["direction"]}` '
            f'（域内 Spearman 正/负：{positive}/{negative}）。', '',
            '| 域 | 规则+ − SH+ | 规则+ − FoRIS | 规则+ − 每域固定 λ | τ（留一域） | 选择64 |',
            '|---|---:|---:|---:|---:|---:|']
    for d, name in zip(c.DOMAINS, c.DOMAIN_NAMES):
        v = domains[d]
        text.append(f'| {name} | {cell(v["sel_minus_SH"])} | {cell(v["sel_minus_FoRIS"])} | '
                    f'{cell(v["sel_minus_fixed"])} | {thresholds[d]:.8g} | {100*v["p64_all"]:.1f}% |')
    text += ['', f'十域均值 {summary["mean_pp"]:+.3f} pp；全体选择64比例 {100*fraction:.2f}%；'
             f'门槛：`{gate}`。']
    if not gate['passed']:
        details['status'] = 'CAPABILITY_WITHOUT_SIMPLE_RULE'
        details['conclusion'] = '有能力，但没有简单规则；停止，不把训练好的选择器放进方法。T 半未读。'
        text += ['', details['conclusion']]
        save('rule.json', details)
        with (OUT / 'result_section.md').open('a') as stream:
            stream.write('\n'.join(text) + '\n')
        print(details['conclusion'], flush=True)
        return
    threshold = fit_threshold(active, feature, direction)
    details.update(global_threshold=threshold, status='RULE_FROZEN')
    rule_spec = {'feature': feature, 'direction_sign': direction, 'threshold': threshold}
    head_path = ROOT / 'scripts/x1012/w26_scale_head.py'
    source = ('"""Frozen W26-2 training-free single-descriptor scale rule."""\n'
              'from w26_scale_capability import rule_head\n'
              f'RULE = {rule_spec!r}\n\n'
              'def scale_head(binding, return_sh=False):\n'
              '    return rule_head(binding, RULE, return_sh=return_sh)\n')
    if head_path.exists() and head_path.read_text() != source:
        raise RuntimeError('A different scale_head already exists; preserve it and stop')
    head_path.write_text(source)
    details['scale_head_sha256'] = sha(head_path)
    details['capability_source_sha256'] = sha(__file__)
    save('rule.json', details)
    text += ['', f'全局 τ={threshold:.10g}；冻结代码 SHA256：`{details["scale_head_sha256"]}`。',
             '物理依据：所选描述子按计划定义测量颜色一致度、形状整合或图边界强度；'
             '方向使用各域的多数关联，单个阈值仅决定是否增强图整合。'
             '它是实测的关联规则，不作因果主张；若能力收益只来自任务级区分，不称逐例自适应。']
    with (OUT / 'result_section.md').open('a') as stream:
        stream.write('\n'.join(text) + '\n')
    evaluate_rule_t(details)
    time_rule()


def fit_threshold(rows, feature, direction):
    """Exactly maximize weighted post-CRF gain, preserving all tied levels."""
    x = direction * np.asarray([e['descriptors'][feature] for e in rows])
    gain = np.asarray([iou(e['L64+']) - iou(e['SH+']) for e in rows]) * training_weights(rows, 'SH+')
    order = np.argsort(x, kind='stable')
    levels, starts = np.unique(x[order], return_index=True)
    totals = np.add.reduceat(gain[order], starts)
    # Include the all-64 and all-16 endpoints. Ties prefer the larger threshold,
    # hence fewer changes to lambda16; this does not change the objective.
    thresholds = np.r_[np.nextafter(levels[0], -np.inf),
                       levels[:-1] + (levels[1:] - levels[:-1]) / 2, levels[-1]]
    objectives = np.r_[np.cumsum(totals[::-1])[::-1], 0.]
    best = max(range(len(thresholds)), key=lambda i: (objectives[i], thresholds[i]))
    return float(direction * thresholds[best])


def rule_head(binding, spec, return_sh=False):
    """Online rule: compute only the chosen descriptor family; no query GT."""
    import torch
    import produce
    from scipy import ndimage as ndi
    family, name = spec['feature'].split('.', 1)
    with c.rc.pc.inference_only(binding):
        fields, masks, j, graph, rho, _ = heads(binding)
        if rho > .9:
            return masks[16], masks[16], rho, 16
        refined, desc = {}, {}
        if family == 'C':
            add_family(desc, 'C', C_BASE, LAMS, {lam: shape_features(masks[lam]) for lam in LAMS})
        elif family == 'D':
            weights = -graph.copy()
            weights.setdiag(0)
            weights.eliminate_zeros()
            add_family(desc, 'D', D_BASE, LAMS, {lam: graph_features(fields[lam], weights) for lam in LAMS})
        elif family == 'B':
            from ics.official_data import decoded_rgb
            from utils.data import denormalize
            from w21_refine import guided
            # Only the single selected type is computed, at the needed scales.
            lams = (16, 64) if '_delta64_16' in name else (int(name.rsplit('_', 1)[1]),)
            vals = {}
            if name.startswith('crf_iou'):
                for lam in lams:
                    refined[lam] = produce.crf(binding, masks[lam])
                    vals[lam] = overlap(masks[lam], refined[lam])
            else:
                if not hasattr(produce, 'CRF_HOST'):
                    produce.CRF_HOST, produce.CRF_ADAPTER = c.rc.pc.host_cpu()
                host = produce.CRF_HOST
                host.set_target(decoded_rgb(c.ASSETS / binding['row']['query_path'], binding['row'].get('query_crop')))
                image = host._tgt_image.clone()
                host._tgt_image = host._orig_tgt_size = None
                rgb = denormalize(image).clamp(0, 1).float()
                for lam in lams:
                    filt = guided(rgb, torch.from_numpy(fields[lam]).float(), r=8, eps=1e-3).numpy()
                    mask = masks[lam]
                    if name.startswith('guided_iou'):
                        vals[lam] = overlap(mask, filt > .5)
                    else:
                        band = ((mask & (ndi.distance_transform_edt(mask) <= 16)) |
                                (~mask & (ndi.distance_transform_edt(~mask) <= 16)))
                        vals[lam] = float(np.abs(filt - fields[lam])[band].mean()) if band.any() else 0.
            desc[spec['feature']] = vals[64] - vals[16] if len(lams) == 2 else vals[lams[0]]
        else:
            raise RuntimeError('Rule feature must be from B/C/D')
        pick = 64 if spec['direction_sign'] * desc[spec['feature']] > spec['direction_sign'] * spec['threshold'] else 16
        if pick not in refined:
            refined[pick] = produce.crf(binding, masks[pick])
        if return_sh and 16 not in refined:
            refined[16] = produce.crf(binding, masks[16])
        return refined.get(16), refined[pick], rho, pick


def t_one(binding):
    from w26_scale_head import scale_head
    started = time.perf_counter()
    sh, final, rho, pick = scale_head(binding, return_sh=True)
    gt, valid = c.truth(binding)
    row = binding['row']
    return {'index': str(binding['index']), 'dataset': binding['dataset'], 'fold': row['fold'],
            'cls': row['global_class_id'], 'photo': c.photo(binding), 'gt': int(gt.sum()),
            'SH+': c.iu(sh, gt, valid), 'rule+': c.iu(final, gt, valid),
            'rho': rho, 'pick': pick, 'sec': time.perf_counter() - started}


def evaluate_rule_t(details):
    import w23_simple_pipeline as w23
    # Section 6 step 4: first and only T evidence read. Save a checkpoint so
    # interrupted evaluation can resume without reopening the original records.
    reference = OUT / 'T_reference.json'
    if not reference.exists():
        with (OUT / 'T_once_started.json').open('x') as stream:
            json.dump({'started_unix_ns': time.time_ns(), 'rule_sha256': details['scale_head_sha256']}, stream)
        old = compressed('w23_simple_pipeline_T')
        base = {e['index']: e for e in compressed('w11_t_ablation')}
        save('T_reference.json', {e['index']: {'foris': base[e['index']]['foris'],
                  'MFU': base[e['index']]['MFU'], 'old_SH+': e['SH+']} for e in old})
    reference_data = load('T_reference.json')
    bindings = w23.rows_for('T')
    assert len(bindings) == 1255 and {str(b['index']) for b in bindings} == set(reference_data)
    path = OUT / 'T_episodes.jsonl'
    done = {json.loads(line)['index'] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in bindings if str(b['index']) not in done]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize) as pool, path.open('a') as stream:
        for n, record in enumerate(pool.map(t_one, todo, chunksize=1), 1):
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            if n % 20 == 0:
                print(f'W26-2 T produced {len(done)+n}/1255', flush=True)
    rr = [dict(e, **reference_data[e['index']]) for e in map(json.loads, path.read_text().splitlines())]
    assert len(rr) == 1255
    rng = np.random.default_rng(c.SEED)
    tab = {}
    for d in c.DOMAINS:
        dr = [e for e in rr if e['dataset'] == d]
        tab[d] = {ref: effect(dr, 'rule+', ref, rng) for ref in ('SH+', 'foris', 'MFU')}
    mean = float(np.mean([v['SH+']['pp'] for v in tab.values()]))
    primary = (all(tab[d]['foris']['pp'] >= 0 for d in ('coco', 'isic')) and mean >= 0 and
               all(v['SH+']['hi'] >= 0 for v in tab.values()))
    details.update(T_read=True, T_domains=tab, T_mean_minus_SH_pp=mean,
                   T_wall_seconds=time.perf_counter() - started,
                   status='RULE_PRIMARY' if primary else ('RULE_SECONDARY' if mean >= 0 else 'SH_PRIMARY_ONLY'))
    save('rule.json', details)
    lines = ['\n#### T 半一次性结果', '', '| 域 | 规则+ − SH+ | 规则+ − FoRIS | 规则+ − MFU |',
             '|---|---:|---:|---:|---:|']
    for d, name in zip(c.DOMAINS, c.DOMAIN_NAMES):
        lines.append('| ' + name + ' | ' + ' | '.join(cell(tab[d][ref]) for ref in ('SH+', 'foris', 'MFU')) + ' |')
    lines += ['', f'十域均值规则+ − SH+={mean:+.3f} pp；结论：`{details["status"]}`。']
    with (OUT / 'result_section.md').open('a') as stream:
        stream.write('\n'.join(lines) + '\n')


def time_rule():
    import torch
    from w26_scale_head import scale_head
    bindings = json.loads((c.task_path('W19') / 'time_bindings.json').read_text())
    initialize()
    store, timings = {}, []
    original = c.rc.raw
    c.rc.raw = lambda metadata, counters: store[metadata['payload_path']]
    try:
        for n, binding in enumerate([bindings[0], bindings[1]] + bindings):
            store.clear()
            for key in ('reference_raw', 'query_raw_identity'):
                path = binding['fit'][key]['payload_path']
                with np.load(path, allow_pickle=False) as archive:
                    store[path] = torch.from_numpy(archive['O/24'].copy())
            start = time.perf_counter()
            scale_head(binding)
            elapsed = time.perf_counter() - start
            if n >= 2:
                timings.append({'index': str(binding['index']), 'dataset': binding['dataset'], 'seconds': elapsed})
    finally:
        c.rc.raw = original
    values = [t['seconds'] for t in timings]
    result = {'n': len(values), 'threads': 2, 'warmup_discarded': 2, 'in_memory_O24': True,
              'mask_RGB_and_CRF_included': True, 'mean_seconds': float(np.mean(values)),
              'median_seconds': float(np.median(values)), 'slower_than_1_56': np.mean(values) > 1.56,
              'episodes': timings}
    save('rule_timing.json', result)
    with (OUT / 'result_section.md').open('a') as stream:
        stream.write(f'\n#### 同协议计时\n\n规则+ 头部均值 {result["mean_seconds"]:.3f}s，'
                     f'中位 {result["median_seconds"]:.3f}s，27 例、2 次预热丢弃、CPU 2 线程、'
                     'O24 在内存，包含 mask/RGB 读取与 CRF。'
                     + ('超过 FoRIS 的 1.56s，准确率与速度的交换如实保留，由用户决定速度主张。'
                        if result['slower_than_1_56'] else '不超过 FoRIS 的 1.56s。') + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('produce')
    p.add_argument('workers', nargs='?', type=int, default=2)
    for name in ('check', 'report', 'rule'):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == 'produce':
        produce_all(args.workers)
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
