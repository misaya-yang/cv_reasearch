"""B4: independent c plus at most two actual FoRIS stage dependencies.

produce seals every natural mask; controls separately seals complete FoRIS and
matched MEAN benchmarks; score reads GT only after validating both seals.
No complete FoRIS pipeline is run to obtain candidate fields.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
import torch
import torch.nn.functional as F
import lab
import plan_common as pc
import a2_exchange as workflow

REQUIRED = ('c32', 'ind0_32', 'mask/IND0', 'mask/INDN', 'mask/T', 'regions64')
ACTIVE = {
    'IND0': (), 'B': ('B',), 'V': ('V',), 'P_only': ('P',),
    'K_prob': ('K_prob',), 'K_minmax': ('K_minmax',),
    'BV': ('B', 'V'), 'VP': ('V', 'P'),
    'BK_prob': ('B', 'K_prob'), 'BK_minmax': ('B', 'K_minmax'),
    'VK_prob': ('V', 'K_prob'), 'VK_minmax': ('V', 'K_minmax'),
}
DEPENDENCY_ORDER = ('B', 'V', 'P', 'K_prob', 'K_minmax')
FUNCTIONS = {
    'B': ['b4_minimal.background_term (FG mean, top20% R hard-negative mean, orthogonalization, Q dot)'],
    'V': ['FoRIS._locate_candidates', 'utils.data.downsample_mask'],
    'P': ['FoRIS._build_seed_cluster_prior', 'utils.data.downsample_mask',
          'utils.clustering.agglomerative_clustering', 'utils.clustering.compute_cluster_prototypes'],
    'K_prob': ['FoRIS._semantic_cluster_reweight_map', 'utils.clustering.agglomerative_clustering'],
    'K_minmax': ['FoRIS._semantic_cluster_reweight_map', 'utils.clustering.agglomerative_clustering'],
}


def dependencies(active):
    required = set(active)
    if 'P' in required:
        required.add('V')
    return tuple(k for k in DEPENDENCY_ORDER if k in required)


def budget(active):
    deps = dependencies(active)
    count = len(deps)
    assert count <= 2, 'Illegal borrowed-stage dependency closure'
    return {'active_score_terms': list(active), 'computed_dependencies': list(deps),
            'FoRIS_stage_count': count, 'functions': {k: FUNCTIONS[k] for k in deps},
            'CRF_counted_in_two_step_budget': False}


def natural_arms(with_crf):
    raw = tuple(ACTIVE) + ('INDN', 'T')
    return raw + tuple(a + '_crf' for a in raw) if with_crf else raw


def background_term(r, q, native_fg):
    """Only the source background component, never source foreground scores."""
    fg = torch.as_tensor(native_fg, dtype=torch.bool).reshape(-1)
    shape = (64, 64)
    info = {'native_fg_tokens': int(fg.sum()), 'native_bg_tokens': int((~fg).sum()),
            'hard_negative_tokens': 0, 'neutral_reason': None, 'weight': .55}
    if not bool(fg.any()) or not bool((~fg).any()):
        info['neutral_reason'] = 'no_native_FG' if not bool(fg.any()) else 'no_native_BG'
        return torch.zeros(shape, dtype=torch.float32), info
    mu_fg = F.normalize(r[fg].mean(0), p=2, dim=0)
    bg = r[~fg]
    k = max(1, int(.2 * len(bg)))
    mu_bg = F.normalize(bg[(bg @ mu_fg).topk(k).indices].mean(0), p=2, dim=0)
    orth = mu_bg - (mu_bg * mu_fg).sum() * mu_fg
    mu_bg = orth / orth.norm().clamp_min(1e-8)
    info.update(hard_negative_tokens=k, orthogonal_background_norm=float(orth.norm()))
    return (-.55 * (q @ mu_bg)).reshape(shape), info


def freeze_method(out, config, with_crf):
    definition = {
        'state': 'METHOD_FROZEN_BEFORE_GT', 'main_evidence': 'P0 FP32 independent c32; never FoRIS score',
        'geometry_features': 'unconditionally IND-debiased unit R/Q O24, no new encoder',
        'readout': 'cached ind0_32 + IND K64 region_mean(sum of score additions); bilinear1024 -> minmax -> >.5',
        'zero_step': 'byte-exact reuse of sealed P0 mask/IND0; CPU/MPS interpolation rounding is not an extra gate',
        'background_native_FG': 'first official center-bilinear mask64 >.5, no fallback/invented FG for B; absent FG or BG gives zero B',
        'background': 'unit(mean native R FG); top20% R BG by similarity; unit hard-negative mean; orthogonalize against FG; -.55*Q dot; no FG clustering',
        'vote': 'direct source _locate_candidates, preserving reference normalization axis and source mask fallback',
        'prior': 'direct source _build_seed_cluster_prior with source V candidate | [::2,::2] grid and transformed query RGB; P requires V',
        'K_semantics': {'prob': 'sf=c32, sbn=1-c32', 'minmax': 'sf=minmax(c32), sbn=1-minmax(c32)',
                        'note': 'both are complementary independent adapters, not original separately normalized FG/hard-negative BG; sf is never updated by B/V/P'},
        'coefficients': {'background': .55, 'vote': .20, 'prior': .25, 'K_fg_boost': .20,
                         'K_conflict_suppress': .18, 'K_neg_cap': .12, 'tau': .6},
        'candidates': {a: budget(v) for a, v in ACTIVE.items()},
        'excluded': ['B+P needs B,V,P=3', 'P+K needs V,P,K=3', 'penalty requires V,P,penalty=3'],
        'common_CRF_enabled': with_crf, 'common_CRF': 'original A7/FoRIS crf_refine and host parameters; excluded from two-step budget',
        'controls': 'FoRIS complete archive and matched MEAN complete control; computed/read only in controls mode and never used by natural fields',
        'dataset_branch': False, 'per_case_component_selection': False,
        'source_sha256': config['source_sha256'], 'basis_sha256': pc.sha(lab.BASIS),
    }
    path = out / 'method.freeze.json'
    if path.exists():
        assert pc.read(path) == definition, 'Frozen B4 method changed'
    else:
        pc.write(path, definition)
    return definition


def produce(cohort, with_crf=False):
    sources = (Path(__file__), Path(workflow.__file__), Path(pc.__file__), Path(lab.__file__),
               Path(pc.__file__).with_name('plan_report.py'), Path(pc.__file__).with_name('repo_methods.py'),
               lab.ROOT / 'src/ics/methods/mean_control.py', lab.ROOT / 'src/ics/m4_crf_cached.py',
               lab.ASSETS / 'third_party/foris_official/models/foris.py',
               lab.ASSETS / 'third_party/foris_official/utils/data.py',
               lab.ASSETS / 'third_party/foris_official/utils/clustering.py',
               lab.ASSETS / 'third_party/foris_official/utils/refinement.py',
               lab.ASSETS / 'a/x1012_A1/borrow_step_effects.json')
    method = {'active': {a: list(v) for a, v in ACTIVE.items()}, 'with_crf': with_crf,
              'dependency_budget': {a: budget(v) for a, v in ACTIVE.items()},
              'main_field': 'independent c32', 'calibration_comparison': ['c probability', 'minmax(c)'],
              'strong_controls': ['complete FoRIS', 'matched complete MEAN']}
    arms = natural_arms(with_crf)
    out, rows, config = workflow.prepare_run('B4', cohort, REQUIRED, arms, method, sources)
    freeze_method(out, config, with_crf)
    if (out / 'sealed.json').exists():
        print({'task': 'B4', 'state': 'ALREADY_SEALED', 'cohort': cohort}, flush=True)
        return
    host, adapter = pc.host_cpu()
    from ics.official_data import load_inputs
    from utils.refinement import crf_refine
    def forbidden(*args, **kwargs):
        raise RuntimeError('B4 natural inference forbids the complete FoRIS pipeline')
    host.predict = host.segment = host._extract_features = forbidden
    assert (host.dino_bg_weight, host.candidate_boost, host.seed_cluster_boost, host.tau) == (.55, .20, .25, .6)
    began, records = time.monotonic(), []
    for i, b in enumerate(rows, 1):
        old = workflow.checked_record(out, b, config)
        if old is not None:
            records.append(old)
            continue
        started = time.monotonic()
        try:
            with pc.inference_only(b):
                fields = workflow.load_fields(b, config, REQUIRED)
                c = torch.from_numpy(fields['c32']).reshape(64, 64)
                assert c.dtype == torch.float32 and torch.isfinite(c).all()
                zero = fields['mask/IND0'].copy()
                preprocessing_start = time.monotonic()
                r, q = lab.debias(lab.feats(b, 'reference')), lab.debias(lab.feats(b, 'query'))
                _, reference_mask, query = load_inputs(b['row'], lab.ASSETS)
                mask = F.interpolate(reference_mask[None, None].float(), (1024, 1024), mode='nearest')
                native_fg = F.interpolate(mask, (64, 64), mode='bilinear', align_corners=False)[0, 0] > .5
                ref_map = r.T.reshape(1, 1, 1024, 64, 64).contiguous()
                q_map = q.T.reshape(1, 1024, 64, 64).contiguous()
                y = lab.coverage64(mask[0, 0]).flatten()
                independent_mu = lab.unit((r * y[:, None]).sum(0))
                host.set_target(query)
                image = host._tgt_image.clone().unsqueeze(0)
                preprocessing_seconds = time.monotonic() - preprocessing_start
                additions, stage_costs = {}, {}
                start = time.monotonic()
                additions['B'], background_info = background_term(r, q, native_fg)
                stage_costs['B'] = time.monotonic() - start
                start = time.monotonic()
                candidate, vote = host._locate_candidates(ref_map, q_map, mask, independent_mu, 1, 64, 64)
                additions['V'] = host.candidate_boost * (vote - .5)
                stage_costs['V'] = time.monotonic() - start
                dense = torch.zeros((64, 64), dtype=torch.bool)
                dense[::2, ::2] = True
                start = time.monotonic()
                prior = host._build_seed_cluster_prior(ref_map, q_map, mask, candidate | dense, 1, 64, 64)
                additions['P'] = host.seed_cluster_boost * (prior - .5)
                stage_costs['P'] = time.monotonic() - start
                for name, sf in (('K_prob', c), ('K_minmax', lab.minmax(c))):
                    start = time.monotonic()
                    additions[name] = host._semantic_cluster_reweight_map(q_map, sf, 1 - sf)
                    stage_costs[name] = time.monotonic() - start
                predictions = {'INDN': fields['mask/INDN'], 'T': fields['mask/T']}
                saved = {'independent_c': c.numpy(), 'vote': vote.numpy(), 'prior': prior.numpy(),
                         **{'addition_' + k: v.numpy() for k, v in additions.items()}}
                candidate_costs = {}
                for arm, active in ACTIVE.items():
                    start = time.monotonic()
                    if not active:
                        regional = fields['ind0_32'].reshape(64, 64).copy()
                        predictions[arm] = zero
                    else:
                        correction = sum((additions[k] for k in active), torch.zeros_like(c))
                        regional = fields['ind0_32'].reshape(64, 64) + workflow.region_mean(correction.numpy(), fields['regions64'])
                        predictions[arm] = pc.pack(pc.readout(regional))
                    saved['field_' + arm] = regional
                    cost = budget(active)
                    cost.update(readout_seconds=time.monotonic() - start,
                                dependency_stage_seconds=sum(stage_costs[k] for k in dependencies(active)),
                                shared_feature_preprocessing_seconds=preprocessing_seconds if active else 0.,
                                latency_scope='measured shared component costs plus per-arm readout; cached c construction/encoding excluded')
                    candidate_costs[arm] = cost
                if with_crf:
                    geometries = None
                    for arm in tuple(predictions):
                        start = time.monotonic()
                        initial = torch.from_numpy(pc.unpack(predictions[arm]))
                        refined = crf_refine(host._crf, host._crf_band_px, host._crf_p_core, image, initial)
                        predictions[arm + '_crf'] = pc.pack(refined.numpy())
                        current = (host._crf.spatial_filter._prepared_geometry, host._crf.bilateral_filter._prepared_geometry)
                        if geometries is None:
                            geometries = current
                        else:
                            assert all(x is y for x, y in zip(geometries, current)), 'CRF geometry reuse changed'
                        cost = dict(candidate_costs[arm]) if arm in candidate_costs else {'FoRIS_stage_count': 0, 'computed_dependencies': []}
                        cost.update(common_CRF=True, CRF_seconds=time.monotonic() - start)
                        candidate_costs[arm + '_crf'] = cost
                assert set(predictions) == set(arms) and adapter.encoder is None and adapter.encoder_calls == 0
            record = workflow.save_record(out, b, config, predictions, saved, started,
                     {'zero_step_bit_exact': True, 'background': background_info, 'candidate_budget_and_costs': candidate_costs,
                      'stage_batch_seconds': stage_costs, 'batch_stage_function_calls': {'B': 1, 'V': 1, 'P': 1, 'K': 2},
                      'FoRIS_score_consumed_by_natural_inference': False, 'complete_FoRIS_pipeline_calls': 0})
            records.append(record)
        finally:
            host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None
        if i <= 3 or i % 50 == 0:
            print({'task': 'B4', 'cohort': cohort, 'completed': i, 'n': len(rows),
                   'seconds': round(time.monotonic() - began, 1)}, flush=True)
    workflow.seal_run(out, rows, records, began)


def controls(cohort):
    """Benchmark-only stage. Never supplies a natural component or score field."""
    import repo_methods as rm
    pc.setup('cpu')
    out = lab.ASSETS / 'a/x1012_B4' / cohort
    seal, config = pc.read(out / 'sealed.json'), pc.read(out / 'config.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED'
    assert pc.sha(out / 'config.json') == seal['config_sha256'] and pc.sha(out / 'manifest.json') == seal['manifest_sha256']
    assert all(pc.sha(p) == h for p, h in config['source_sha256'].items())
    assert pc.sha(lab.BASIS) == pc.read(out / 'method.freeze.json')['basis_sha256']
    rows, frozen = workflow.cohort_rows(cohort), pc.read(out / 'manifest.json')
    assert len(rows) == len(frozen) and all(workflow.same_inputs(a, b) for a, b in zip(rows, frozen))
    assert all('B' in b.get('old_arms', {}) for b in rows), 'Wait for complete FoRIS baseline bindings'
    root = out / 'controls'; (root / 'predictions').mkdir(parents=True, exist_ok=True)
    (root / 'records').mkdir(exist_ok=True)
    records, began = [], time.monotonic()
    for i, b in enumerate(rows, 1):
        name, record_path = pc.key(b), root / 'records' / (pc.key(b) + '.json')
        if record_path.exists():
            record = pc.read(record_path); assert pc.sha(record['path']) == record['sha256']
            records.append(record); continue
        start = time.monotonic()
        with pc.inference_only(b):
            original, receipt = workflow.read_archive(b)
            entry = b.get('old_arms', {}).get('G')
            if entry is not None:
                digest = pc.sha(entry['path'])
                if 'sha256' in entry: assert digest == entry['sha256']
                with np.load(entry['path'], allow_pickle=False) as z: mean = pc.unpack(z[entry['cli_key']])
                mean_source = {'mode': 'complete_MEAN_archive', 'path': entry['path'], 'sha256': digest}
            else:
                entry = b.get('score64')
                assert entry is not None, 'Wait for matched MEAN score baseline: ' + name
                digest = pc.sha(entry['path'])
                if 'sha256' in entry: assert digest == entry['sha256']
                mean = rm.field_mask(rm.mean_field(b)).numpy()
                mean_source = {'mode': 'exact_repository_MEAN_from_cached_features_and_score',
                               'score_path': entry['path'], 'score_sha256': digest}
        pred = root / 'predictions' / (name + '.npz')
        assert not pred.exists(), 'Preserve partial control output'
        np.savez_compressed(pred, FoRIS=pc.pack(original), MEAN=pc.pack(mean))
        record = {'index': name, 'path': str(pred), 'sha256': pc.sha(pred), 'FoRIS': receipt,
                  'MEAN': mean_source, 'seconds': time.monotonic() - start, 'query_GT_reads': 0,
                  'new_encoder_calls': 0, 'natural_components_consume_these_fields': False}
        pc.write(record_path, record); records.append(record)
        if i <= 3 or i % 50 == 0:
            print({'task': 'B4-controls', 'completed': i, 'n': len(rows)}, flush=True)
    pc.write(root / 'sealed.json', {'state': 'ALL_COMPLETE_CONTROLS_SEALED', 'n': len(rows), 'records': records,
             'natural_seal_sha256': pc.sha(out / 'sealed.json'), 'method_freeze_sha256': pc.sha(out / 'method.freeze.json'),
             'seconds': time.monotonic() - began, 'new_encoder_calls': 0, 'query_GT_reads': 0})


def score(cohort):
    import plan_report as pr
    pc.setup('cpu')
    out = lab.ASSETS / 'a/x1012_B4' / cohort
    seal, control_seal = pc.read(out / 'sealed.json'), pc.read(out / 'controls/sealed.json')
    config = pc.read(out / 'config.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED' and control_seal['state'] == 'ALL_COMPLETE_CONTROLS_SEALED'
    assert pc.sha(out / 'config.json') == seal['config_sha256'] and pc.sha(out / 'manifest.json') == seal['manifest_sha256']
    assert pc.sha(out / 'sealed.json') == control_seal['natural_seal_sha256']
    assert pc.sha(out / 'method.freeze.json') == control_seal['method_freeze_sha256']
    assert all(pc.sha(p) == h for p, h in config['source_sha256'].items())
    assert pc.sha(lab.BASIS) == pc.read(out / 'method.freeze.json')['basis_sha256']
    rows = pc.read(out / 'manifest.json')
    indexed, benchmarks = {r['index']: r for r in seal['records']}, {r['index']: r for r in control_seal['records']}
    assert len(rows) == len(indexed) == len(benchmarks) == control_seal['n']
    # Validate every artifact before the first query GT read.
    for b in rows:
        assert workflow.checked_record(out, b, config) == indexed[pc.key(b)]
        receipt = benchmarks[pc.key(b)]
        assert pc.sha(receipt['path']) == receipt['sha256']
    arms = ('FoRIS', 'MEAN') + tuple(config['arms'])
    records = []
    for b in rows:
        with np.load(indexed[pc.key(b)]['prediction_path'], allow_pickle=False) as z:
            predictions = {a: pc.unpack(z[a]) for a in config['arms']}
        with np.load(benchmarks[pc.key(b)]['path'], allow_pickle=False) as z:
            predictions.update({a: pc.unpack(z[a]) for a in ('FoRIS', 'MEAN')})
        records.append(pr.score_record(b, predictions))
    definitions = pc.read(out / 'method.freeze.json')
    budgets = definitions['candidates']
    costs = {}
    cost_arms = tuple(ACTIVE) + (tuple(a + '_crf' for a in ACTIVE) if config['method']['with_crf'] else ())
    for arm in cost_arms:
        values = [r['candidate_budget_and_costs'][arm] for r in seal['records']]
        added = [v['dependency_stage_seconds'] + v['shared_feature_preprocessing_seconds'] + v['readout_seconds'] + v.get('CRF_seconds', 0.) for v in values]
        base_arm = arm.removesuffix('_crf')
        costs[arm] = {'FoRIS_stage_count': budgets[base_arm]['FoRIS_stage_count'],
                      'computed_dependencies': budgets[base_arm]['computed_dependencies'],
                      'median_added_seconds': float(np.median(added)), 'mean_added_seconds': float(np.mean(added)),
                      'common_CRF': arm.endswith('_crf'),
                      'latency_scope': values[0]['latency_scope']}
    label = 'B4' if cohort == 'explore' else 'B4_seed150'
    result = pr.report(label, out, records, arms, [
        '主证据始终独立c；12个固定零步/单项/合法两步组合，跨十域同一算法。全部自然掩码与完整FoRIS/MEAN强控先封存，之后才读取query GT。',
        'B仅用R native前景均值、top20% hard-negative均值正交化与-.55*Q相似度；native FG/BG缺失时中性0。V/P/K直接源函数，不运行完整FoRIS取中间场。',
        'P_only虽不加V boost，仍计算V候选，因此预算V+P=2；BV/BK/VK/VP均2步，B+P/P+K/penalty依赖超过2均排除。',
        'K_prob固定(sf,sbn)=(c,1-c)，K_minmax固定(minmax(c),1-minmax(c))；两者为独立互补适配，与官方分别归一化前景/硬负背景语义不同。',
        '零步IND0逐位复用P0 sealed mask；所有修正先做原K64区域均值，再与保存的IND0场相加，最后原IND读出。公共原CRF可在首次freeze前启用，另记成本且不计两步。',
        'MEAN只作为完整强控，单独controls阶段复用archive或从缓存FoRIS score重入原MEAN；这些数据不进入任何自然组件。成本报告为缓存c以外的新增计算，不伪称含编码的端到端耗时。',
        '这是固定开发候选表；不按域/例选择组件，不视为confirm或自动通过C1。',
    ], extra={'method': definitions, 'candidate_budgets_and_costs': costs,
              'cohort': cohort, 'complete_controls': ['FoRIS', 'MEAN']})
    eligible = {a: {d: s[a]['delta_to_foris_pp'] for d, s in result['summary'].items()}
                for a in config['arms'] if a.removesuffix('_crf') in ACTIVE}
    passing = [a for a, values in eligible.items() if len(values) == 10 and all(v >= -.5 for v in values.values())]
    pc.write(out / 'completion.json', {'state': 'DEVELOPMENT_CANDIDATES_REPORTED', 'n': len(rows),
             'cohort': cohort, 'ten_domain_within_minus_05_candidates': passing, 'candidate_delta_pp': eligible,
             'new_encoder_calls': 0, 'confirm_scored': False, 'per_case_or_domain_component_selection': False,
             'method_freeze_sha256': pc.sha(out / 'method.freeze.json')})
    path = lab.ROOT / 'docs/research/x1012_tasks' / (label + '.md')
    with path.open('a') as f:
        f.write('\n## 实际依赖预算与新增计算\n\n|候选|FoRIS步数|依赖|新增时间中位秒|\n|---|---:|---|---:|\n')
        for arm, cost in costs.items():
            f.write(f"|{arm}|{cost['FoRIS_stage_count']}|{','.join(cost['computed_dependencies']) or 'none'}|{cost['median_added_seconds']:.4f}|\n")
        f.write('\n公共CRF若启用，其逐例时间另存record；新增耗时不包含已缓存c与DINO编码。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'controls', 'score'))
    parser.add_argument('--cohort', choices=('seed150', 'explore'), default='seed150')
    parser.add_argument('--crf', action='store_true', help='enable fixed common CRF when producing/freeze first time')
    args = parser.parse_args()
    if args.mode == 'produce': produce(args.cohort, args.crf)
    elif args.mode == 'controls': controls(args.cohort)
    else: score(args.cohort)
