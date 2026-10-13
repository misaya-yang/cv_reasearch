"""R7: one-pass, training-free query support with immutable reference priors.

One real masked reference, cached O24 only. No gradients, fitting, pseudo-label
iterations or query annotations in inference. All natural outputs precede scoring.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_env] = '2'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent/'cv_data'
OUT = ASSETS/'a/x1012_R7'
R1 = ASSETS/'a/x1012_R1'
P0 = ASSETS/'a/x1012_P0/fields'
RUNTIME = OUT/'runtime'
ARMS = ('R_only', 'R_QF', 'R_QB', 'R_QFB', 'R_QFB_roleperm')
SALT = 'R7_query_seed_role_permutation_20261012'
TAU = .05


def load_helpers():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    modules = {}
    for name, source in (('lab', R1/'frozen/3_lab.py'),
                         ('plan_common', R1/'frozen/1_plan_common.py'),
                         ('plan_report', R1/'frozen/2_plan_report.py')):
        dest = RUNTIME/(name+'.py')
        if not dest.exists():
            shutil.copyfile(source, dest)
        assert dest.read_bytes() == source.read_bytes(), 'R7 helper drift: '+name
        spec = importlib.util.spec_from_file_location(name, dest)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        if name == 'lab':
            m.ROOT, m.ASSETS = ROOT, ASSETS
            m.BINDINGS = ASSETS/'a/consensus_reference_calibration6380_v1_20261011/source_bindings.json'
            m.BASIS = ASSETS/'native_assets/positional_basis.pt'
        modules[name] = m
    return modules['lab'], modules['plan_common'], modules['plan_report']


lab, pc, pr = load_helpers()


def initialize(device='cpu'):
    assert device in ('cpu', 'mps')
    if device == 'mps':
        assert torch.backends.mps.is_available()
    pc.setup(device)
    torch.set_num_threads(2)


def role_posterior(likelihood, prior):
    mass = likelihood * prior[None, :]
    return mass[:, 0] / mass.sum(1).clamp_min(1e-30)


def mathematical_check():
    kernel = np.array([[.9, .2, .1, .4], [.1, .4, .8, .3]], np.float64)
    y = np.array([1., .25, 0., .75])
    pi = np.array([y.mean(), 1-y.mean()])
    role_weights = np.stack([y/y.sum(), (1-y)/(1-y).sum()], axis=1)
    lr = kernel@role_weights
    explicit = lr*pi
    explicit = explicit[:, 0]/explicit.sum(1)
    old = (kernel@y)/kernel.sum(1)
    np.testing.assert_allclose(explicit, old, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(role_weights.sum(0)*pi, pi, rtol=1e-14)
    return {'state': 'PASS', 'checks': ['role_normalization_plus_prior_recovers_K1_c',
                                      'each_role_total_prior_mass_unchanged'],
            'maximum_K1_absolute_error': float(np.max(np.abs(explicit-old)))}


def prepare(cohort):
    assert cohort in ('screen40', 'full1360')
    (OUT/'frozen').mkdir(parents=True, exist_ok=True)
    config_path = OUT/'config.json'
    if not config_path.exists():
        sources = [Path(__file__), *(RUNTIME/(n+'.py') for n in ('lab', 'plan_common', 'plan_report'))]
        config = {
            'state': 'FROZEN_BEFORE_NATURAL_OUTPUTS', 'arms': ARMS,
            'reference_count': 1, 'new_encoder_calls': 0, 'training': False,
            'gradients': False, 'parameter_fitting': False, 'pseudo_label_iterations': 0,
            'query_GT_in_inference': False, 'FoRIS_final_score_in_inference': False,
            'domain_branch': False, 'extra_real_reference': False,
            'features': 'cached O/24; original positional projection; unit normalization',
            'center': 'fixed K1 mu=.5*(mean(debiased_R)+mean(debiased_Q)); unit(R-mu),unit(Q-mu); never recompute after query augmentation',
            'kernel': 'exp((cosine-1)/.05); common shift changes no posterior',
            'reference_roles': 'L_RF=sum(y*k)/sum(y); L_RB=sum((1-y)*k)/sum(1-y); zero reference role receives zero likelihood',
            'prior': 'pi_F=mean(reference soft coverage), pi_B=1-pi_F; identical for every arm',
            'seeds': 'minmax(R1 VP_UN continuous64 field); foreground>=.9, background<=.1; other query tokens unknown',
            'seed_dependency': 'R1 VP_UN uses independent max-J modulation, V/P (2 borrowed stages), and explanation-weighted reciprocal graph; no FoRIS final score',
            'query_likelihood': 'mean kernel over the selected role only; per query token exclude self and all Chebyshev<=1 grid neighbors; renormalize by the remaining role count',
            'empty_query_role': 'return the corresponding reference likelihood L_R',
            'augmentation': 'for each augmented role L=.5*L_R+.5*L_Q; other role stays L_R; original reference always retained',
            'posterior': 'pi_F*L_F/(pi_F*L_F+pi_B*L_B)',
            'permutation': 'fixed SHA256(salt+NUL+episode index) seed; shuffle F/B assignments within the same seed union, preserving both role counts and seed locations',
            'permutation_salt': SALT, 'readout': 'token64 -> bilinear1024 -> minmax -> strict>.5; no K64 regions',
            'screen': 'deterministic stride40 per domain from the original fixed R1 1360 manifest; exactly400; no R5 data',
            'resources': 'one producer; CPU2 threads; MPS only after coordinator release; score on CPU2',
            'runtime_root_override': {'lab.ROOT': str(ROOT), 'lab.ASSETS': str(ASSETS)},
            'runtime_source_provenance': {str(RUNTIME/'lab.py'): str(R1/'frozen/3_lab.py'),
                                          str(RUNTIME/'plan_common.py'): str(R1/'frozen/1_plan_common.py'),
                                          str(RUNTIME/'plan_report.py'): str(R1/'frozen/2_plan_report.py')},
            'source_sha256': {str(p): pc.sha(p) for p in sources},
            'dependency_sha256': {str(p): pc.sha(p) for p in (R1/'sealed.json', R1/'manifest.json', P0/'sealed.json')},
            'mathematical_check': mathematical_check(),
        }
        pc.write(config_path, config)
        shutil.copyfile(Path(__file__), OUT/'frozen/r7_anchored_support.py')
        (OUT/'SPEC.md').write_text(
            '# R7 frozen one-pass anchored query support\n\n'
            'One true masked reference, no training/fitting/gradients, no new DINO encoding. '
            'Keep the original K1 joint center and temperature .05. Each reference role likelihood '
            'is its own normalized kernel mean; keep pi_F=mean(reference coverage) unchanged. '
            'High/low cores of the frozen independent R1 VP_UN field supply query foreground/background '
            'support at fixed thresholds .9/.1; all other query tokens remain unknown. '
            'Exclude each predicted query token and its 3x3 neighborhood from support, normalize '
            'each remaining query role independently, and average an augmented role likelihood '
            'equally with the immutable reference role. Empty support returns the reference likelihood.\n\n'
            'Five arms: R only, R+Q foreground, R+Q background, R+both, and R+both with SHA-seeded '
            'role permutation preserving seed union and role counts. One pass only. Full continuous '
            'token predictions use bilinear1024/minmax>.5. All natural predictions seal before GT scoring. '
            'Screen400 is stride40/domain within the original R1 1360; no new R5 episodes. '
            'Cached seed construction adds R1 max-J/V/P/graph work; reported R7 runtime excludes '
            'cached seed production and is not an RGB-to-mask online latency claim.\n', encoding='utf-8')
    config = pc.read(config_path)
    assert all(pc.sha(p) == h for p, h in config['source_sha256'].items()), 'R7 source drift'
    assert all(pc.sha(p) == h for p, h in config['dependency_sha256'].items()), 'R7 dependency drift'
    run = OUT/cohort
    for folder in ('records', 'predictions', 'fields'):
        (run/folder).mkdir(parents=True, exist_ok=True)
    rows = pc.read(R1/'manifest.json')
    if cohort == 'screen40':
        selected = []
        for domain in lab.ALL_DOMAINS:
            part = [r for r in rows if r['dataset'] == domain]
            assert len(part) >= 40
            selected.extend(part[i] for i in (np.arange(40)*len(part)/40).astype(int))
        rows = selected
    assert len(rows) == (400 if cohort == 'screen40' else 1360)
    if not (run/'manifest.json').exists():
        pc.write(run/'manifest.json', rows)
    else:
        assert pc.read(run/'manifest.json') == rows
    return run, rows


@lru_cache(2)
def query_exclusion(device):
    ids = torch.arange(4096, device=device)
    row, col = ids//64, ids % 64
    return ((row[:, None]-row[None, :]).abs() <= 1) & ((col[:, None]-col[None, :]).abs() <= 1)


@torch.inference_mode()
def compute(binding):
    name = pc.key(binding)
    r1 = pc.read(R1/'records'/f'{name}.json')
    p0 = pc.read(P0/'records'/f'{name}.json')
    assert pc.sha(r1['field_path']) == r1['field_sha256']
    assert pc.sha(p0['path']) == p0['sha256']
    with np.load(r1['field_path'], allow_pickle=False) as z:
        seed_field = z['VP_UN'].copy()
    with np.load(p0['path'], allow_pickle=False) as z:
        control_c = z['c32'].copy()
    s = (seed_field-seed_field.min())/max(float(seed_field.max()-seed_field.min()), 1e-6)
    sf, sb = s.ravel() >= .9, s.ravel() <= .1
    seed_ids = np.flatnonzero(sf | sb)
    rng_seed = int(hashlib.sha256((SALT+'\0'+name).encode()).hexdigest()[:16], 16)
    permuted_ids = np.random.default_rng(rng_seed).permutation(seed_ids)
    roles = np.zeros((4096, 4), np.float32)
    roles[:, 0], roles[:, 1] = sf, sb
    roles[permuted_ids[:int(sf.sum())], 2] = 1
    roles[permuted_ids[int(sf.sum()):], 3] = 1
    assert np.array_equal(roles[:, :2].sum(0), roles[:, 2:].sum(0))
    assert np.array_equal(roles[:, :2].sum(1), roles[:, 2:].sum(1))
    mask = lab.mask1024(binding['row']['reference_mask_path']).to(lab.DEV)
    y = lab.coverage64(mask).flatten()
    reference_weights = torch.stack((y, 1-y), dim=1)
    role_mass = reference_weights.sum(0)
    prior = role_mass/4096
    reference_weights /= role_mass.clamp_min(1e-30)[None, :]
    raw_digests = {}
    for role in ('reference_raw', 'query_raw_identity'):
        entry = binding['fit'][role]
        expected = entry.get('payload_sha256')
        if expected is None:
            expected = pc.read(Path(entry['payload_path']).parent/'entry.json')['file_sha256']
        assert pc.sha(entry['payload_path']) == expected
        raw_digests[role] = expected
    r, q = lab.debias(lab.feats(binding, 'reference')), lab.debias(lab.feats(binding, 'query'))
    mu = .5*(r.mean(0)+q.mean(0))
    rc, qc = lab.unit(r-mu), lab.unit(q-mu)
    reference_kernel = torch.exp((qc@rc.T-1)/TAU)
    lr = reference_kernel@reference_weights
    base = role_posterior(lr, prior)
    c_difference = float(np.max(np.abs(base.cpu().numpy()-control_c)))
    assert c_difference < 1e-4, (name, 'role normalization failed K1 identity', c_difference)
    del reference_kernel, r, rc
    query_kernel = torch.exp((qc@qc.T-1)/TAU)
    excluded = query_exclusion(str(lab.DEV))
    query_kernel.masked_fill_(excluded, 0.)
    role_weights = torch.as_tensor(roles, device=lab.DEV)
    # Fixed tensor shapes bound MPS graph caches despite variable seed counts.
    available = (~excluded).float()@role_weights
    lq = (query_kernel@role_weights)/available.clamp_min(1.)
    repeated_reference = lr[:, [0, 1, 0, 1]]
    lq = torch.where(available > 0, lq, repeated_reference)
    fields = {'R_only': base.reshape(64, 64).cpu().numpy()}
    for arm, columns, augment in (('R_QF', (0, 1), (True, False)),
                                  ('R_QB', (0, 1), (False, True)),
                                  ('R_QFB', (0, 1), (True, True)),
                                  ('R_QFB_roleperm', (2, 3), (True, True))):
        likelihood = torch.stack([.5*(lr[:, k]+lq[:, columns[k]]) if augment[k] else lr[:, k]
                                  for k in (0, 1)], dim=1)
        fields[arm] = role_posterior(likelihood, prior).reshape(64, 64).cpu().numpy()
    info = {'foreground_seed_n': int(sf.sum()), 'background_seed_n': int(sb.sum()),
            'unknown_query_n': int(4096-len(seed_ids)), 'role_permutation_seed': rng_seed,
            'prior_foreground': float(prior[0]), 'prior_background': float(prior[1]),
            'reference_only_max_abs_delta_to_P0_c': c_difference,
            'query_rows_without_role_support': (available == 0).sum(0).cpu().numpy().tolist(),
            'R1_field_sha256': r1['field_sha256'], 'P0_field_sha256': p0['sha256'],
            'raw_input_sha256': raw_digests,
            'reference_mask_sha256': pc.sha(binding['row']['reference_mask_path']),
            'cached_R1_eight_arm_case_seconds': r1['seconds'], 'seed_production_in_R7_seconds': False}
    return fields, info


def produce(cohort, device, limit=None):
    initialize(device)
    run, rows = prepare(cohort)
    if (run/'sealed.json').exists():
        print({'phase': 'R7-produce', 'cohort': cohort, 'state': 'ALREADY_SEALED'}, flush=True)
        return
    total = len(rows)
    selected = rows if limit is None else rows[:limit]
    began, records = time.monotonic(), []
    for i, binding in enumerate(selected, 1):
        name = pc.key(binding)
        record_path = run/'records'/f'{name}.json'
        if record_path.exists():
            record = pc.read(record_path)
            for kind in ('prediction', 'field'):
                assert pc.sha(record[kind+'_path']) == record[kind+'_sha256']
            records.append(record)
            continue
        started = time.monotonic()
        with pc.inference_only(binding):
            fields, info = compute(binding)
            predictions = {arm: pc.pack(pc.readout(field)) for arm, field in fields.items()}
            pp, fp = run/'predictions'/f'{name}.npz', run/'fields'/f'{name}.npz'
            assert not pp.exists() and not fp.exists(), 'Preserve orphaned R7 output '+name
            np.savez_compressed(pp, **predictions)
            np.savez_compressed(fp, **fields)
        record = {'index': name, 'dataset': binding['dataset'], 'episode_id': binding['row']['episode_id'],
                  'prediction_path': str(pp), 'prediction_sha256': pc.sha(pp),
                  'field_path': str(fp), 'field_sha256': pc.sha(fp), 'support': info,
                  'seconds': time.monotonic()-started, 'device': device, 'threads': 2,
                  'new_encoder_calls': 0, 'inference_query_GT_reads': 0, 'training': False,
                  'parameter_fitting': False, 'real_reference_count': 1}
        pc.write(record_path, record)
        records.append(record)
        lab.free()
        if i <= 3 or i % 20 == 0:
            print({'phase': 'R7-produce', 'cohort': cohort, 'completed': i, 'n': total,
                   'device': device, 'seconds': round(time.monotonic()-began, 2)}, flush=True)
    if len(selected) != total:
        pc.write(run/'throughput.json', {'state': 'PARTIAL_NOT_SCORED', 'n': len(selected),
                                       'wall_seconds': time.monotonic()-began, 'device': device,
                                       'new_encoder_calls': 0, 'inference_query_GT_reads': 0})
        print({'state': 'PARTIAL_NOT_SCORED', 'n': len(selected), 'total': total}, flush=True)
        return
    seal = {'state': 'ALL_NATURAL_OUTPUTS_SEALED', 'n': total, 'records': records,
            'manifest_sha256': pc.sha(run/'manifest.json'), 'config_sha256': pc.sha(OUT/'config.json'),
            'wall_seconds_this_invocation': time.monotonic()-began,
            'case_seconds_sum': sum(r['seconds'] for r in records),
            'cached_seed_production_excluded': True, 'workers': 1, 'cpu_threads': 2,
            'new_encoder_calls': 0, 'inference_query_GT_reads': 0, 'training': False}
    pc.write(run/'sealed.json', seal)
    print({'phase': 'R7-produce', 'cohort': cohort, 'state': seal['state'], 'n': total}, flush=True)


def score(cohort, baseline_report=None):
    initialize('cpu')
    run, rows = prepare(cohort)
    seal = pc.read(run/'sealed.json')
    assert seal['state'] == 'ALL_NATURAL_OUTPUTS_SEALED' and seal['n'] == len(rows)
    assert pc.sha(run/'manifest.json') == seal['manifest_sha256']
    assert pc.sha(OUT/'config.json') == seal['config_sha256']
    receipts = {r['index']: r for r in seal['records']}
    old = {r['index']: r for r in pc.read(R1/'report.json')['records']}
    extra = {} if baseline_report is None else {r['index']: r for r in pc.read(baseline_report)['records']}
    extra_arms = () if not extra else tuple(a for a in next(iter(extra.values()))['metrics'] if 'insid3' in a.lower())
    if baseline_report is not None:
        assert extra_arms, 'Supplied baseline report has no INSID3 arm'
    controls = ('FoRIS', 'R1_VP_UN') + extra_arms
    records, began = [], time.monotonic()
    for i, binding in enumerate(rows, 1):
        name = pc.key(binding)
        receipt = receipts[name]
        assert pc.sha(receipt['prediction_path']) == receipt['prediction_sha256']
        with np.load(receipt['prediction_path'], allow_pickle=False) as z:
            record = pr.score_record(binding, {a: pc.unpack(z[a]) for a in ARMS})
        assert old[name]['query_photo'] == record['query_photo']
        record['metrics']['FoRIS'] = old[name]['metrics']['FoRIS']
        record['metrics']['R1_VP_UN'] = old[name]['metrics']['VP_UN']
        for arm in extra_arms:
            assert extra[name]['query_photo'] == record['query_photo']
            record['metrics'][arm] = extra[name]['metrics'][arm]
        records.append(record)
        if i <= 3 or i % 100 == 0:
            print({'phase': 'R7-score', 'cohort': cohort, 'completed': i, 'n': len(rows)}, flush=True)
    notes = [
        '严格免训练：单张真实参考，已有O24，不做梯度、参数拟合或伪标签迭代；没有额外真实参考、查询GT推理、FoRIS最终score输入。',
        'K1联合中心及温度.05不变；参考与查询均按FG/BG角色单独归一，显式保留原参考pi；增补角色用R/Q等权似然，原R始终保留，无种子角色回到L_R。',
        'Q种子仅取冻结R1 VP_UN的minmax>=.9与<=.1，未知不当BG；排除自身及3x3邻域；置换保留种子集合与两侧数量。五臂连续token输出使用同一bilinear1024/minmax>.5。',
        '全部自然预测先封存后评分；不进行监督能力诊断或GT阈值选择。screen40固定原R1队列每域40例，共400；full1360仍为原开发队列，未使用R5新500。',
        '增补臂额外依赖R1独立种子（max-J、V/P两步和图解）；R7耗时只计缓存后的核增补/读出，排除缓存种子产生，不能宣称RGB到mask在线延迟。',
        'INSID3封存对照已绑定。' if extra_arms else 'INSID3对照尚待独立基线产物绑定；当前报告不声称胜过INSID3。',
    ]
    result = pr.report('R7_'+cohort, run, records, controls+ARMS, notes,
                       extra={'production': {k: v for k, v in seal.items() if k != 'records'},
                              'baseline_report': str(baseline_report) if baseline_report else None,
                              'baseline_report_sha256': pc.sha(baseline_report) if baseline_report else None})
    contrasts = {}
    for baseline in ('R_only', 'R_QFB_roleperm'):
        projected = [dict(r, metrics={'FoRIS': r['metrics'][baseline],
                                     **{a: r['metrics'][a] for a in ('R_QF', 'R_QB', 'R_QFB')}}) for r in records]
        contrasts['versus_'+baseline] = pr.summary(projected, ('FoRIS', 'R_QF', 'R_QB', 'R_QFB'))
    pc.write(run/'paired_contrasts.json', contrasts)
    pc.write(run/'completion.json', {'state': 'COMPLETE', 'n': len(rows), 'training': False,
                                    'new_encoder_calls': 0, 'inference_query_GT_reads': 0,
                                    'score_seconds': time.monotonic()-began,
                                    'INSID3_baseline_bound': bool(extra_arms),
                                    'equal_domain_means': {a: float(np.mean([t[a]['class_miou'] for t in result['summary'].values()])) for a in controls+ARMS}})
    print({'phase': 'R7-complete', 'cohort': cohort, 'n': len(rows),
           'seconds': round(time.monotonic()-began, 2)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'produce', 'score'))
    parser.add_argument('--cohort', choices=('screen40', 'full1360'), default='screen40')
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--baseline-report', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze':
        initialize('cpu')
        run, rows = prepare(args.cohort)
        print({'state': 'FROZEN', 'cohort': args.cohort, 'n': len(rows), 'path': str(run)}, flush=True)
    elif args.mode == 'produce':
        produce(args.cohort, args.device, args.limit)
    else:
        score(args.cohort, args.baseline_report)
