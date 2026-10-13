"""C1-only GT-free inference from verified cached O/24 inputs."""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn.functional as F
import lab
import plan_common as pc
import b1_kshot as raw_core
import b4_minimal as b4
import a2_exchange as workflow
import a7_generic as a7
import a4_modulation as a4m


def modulated_evidence(c, labels, ind0, indn, sim_max_fp16, back_f_fp16):
    """Exact A4 nine-point quality selection, including its FP32 endpoints."""
    raw = c.numpy()
    novelty = np.asarray(sim_max_fp16, dtype=np.float32)
    novelty = (novelty - novelty.min()) / max(float(novelty.max() - novelty.min()), 1e-6)
    back = np.asarray(back_f_fp16, dtype=np.float64)
    onehot = F.one_hot(labels, 64).float()
    endpoint_masks = {0.: pc.readout(ind0), 1.: pc.readout(indn)}
    values, fields, masks, tokens = [], [], [], []
    for strength in a4m.STRENGTHS:
        v = torch.from_numpy(raw * np.power(novelty, strength))
        field = onehot @ ((onehot.T @ v) / onehot.sum(0).clamp_min(1))
        mask = endpoint_masks[strength] if strength in (0., 1.) else pc.readout(field)
        values.append(a4m.quality(mask, raw, back))
        tokens.append(v); fields.append(field); masks.append(mask)
    chosen = int(np.argmax(values))  # Ascending nine-point grid: lower s wins ties.
    return {'c_mod': tokens[chosen], 'modulated_region': fields[chosen],
            'modulation_mask': masks[chosen], 'sim_max_fp16': np.asarray(sim_max_fp16),
            'back_f_fp16': np.asarray(back_f_fp16),
            'modulation_detail': {'evidence_modulation': 'A4_max_J_continuous',
                                  'selected_s': a4m.STRENGTHS[chosen], 'J_grid': values,
                                  'storage': 'sim_max/back_f FP16; c/ind0/raw-novelty indn FP32',
                                  'ties': 'lower s', 'query_GT_reads': 0}}


def independent_state(binding, counters, modulation=None, rho_gate=None):
    """Minimal P0 formula on CPU: global kernel shift and the same K64 mean."""
    r = lab.debias(raw_core.read_raw(binding['fit']['reference_raw'], counters))
    q = lab.debias(raw_core.read_raw(binding['fit']['query_raw_identity'], counters))
    mask = lab.mask1024(binding['row']['reference_mask_path']).float()
    coverage = lab.coverage64(mask).flatten()
    rho = float(1 - (coverage * (coverage >= .9)).sum() / coverage.sum().clamp_min(1e-9))
    triggered = rho_gate is not None and rho > rho_gate
    mu = .5 * (r.mean(0) + q.mean(0))
    sim = lab.unit(q - mu) @ lab.unit(r - mu).T
    maximum = sim.max()
    kernel = torch.exp((sim - maximum) / .05)
    df, db = kernel @ coverage, kernel @ (1 - coverage)
    c = df / (df + db).clamp_min(1e-30)
    qunit = lab.unit(q)
    centroids = lab.kmeans(qunit, 64, iters=6)
    labels = (qunit @ centroids.T).argmax(1)
    onehot = F.one_hot(labels, 64).float()
    ind0 = onehot @ ((onehot.T @ c) / onehot.sum(0).clamp_min(1))
    state = {'r': r, 'q': q, 'mask': mask, 'coverage': coverage,
             'c': c, 'ind0': ind0, 'regions64': labels,
             'reference_rho': rho, 'reference_rho_gate_triggered': triggered}
    if modulation is not None and not triggered:
        assert modulation == 'A4_max_J_continuous'
        sim_max = sim.max(1).values
        raw_novelty = lab.minmax(sim_max)
        indn = onehot @ ((onehot.T @ (c * raw_novelty)) / onehot.sum(0).clamp_min(1))
        reverse = torch.softmax(sim / .05, dim=0)
        back_f = reverse @ coverage / coverage.sum().clamp_min(1e-6)
        state.update(modulated_evidence(c, labels, ind0, indn,
                                       sim_max.numpy().astype(np.float16), back_f.numpy().astype(np.float16)))
    return state


def build_host(candidate):
    required = candidate.get('dependencies', [])
    needs_host = candidate['common_crf'] or any(k != 'B' for k in required)
    if not needs_host:
        return None
    host, adapter = pc.host_cpu()
    assert adapter.encoder is None and adapter.encoder_calls == 0
    def forbidden(*args, **kwargs):
        raise RuntimeError('C1 forbids complete FoRIS/encoder inference')
    host.predict = host.segment = host._extract_features = forbidden
    return host


def exact_neighbors(q, block=256):
    """P0 exhaustive cosine top20 excluding self, with bounded CPU blocks."""
    unit = lab.unit(q)
    n = len(unit)
    ids = torch.empty((n, 20), dtype=torch.long)
    for start in range(0, n, block):
        stop = min(start + block, n)
        similarity = unit[start:stop] @ unit.T
        similarity[torch.arange(stop - start), torch.arange(start, stop)] -= 1e9
        ids[start:stop] = similarity.topk(20, dim=1).indices
    return ids.numpy().astype(np.uint16)


def b4_prediction(binding, state, candidate, host):
    active = tuple(candidate['active_score_terms'])
    deps = tuple(candidate['dependencies'])
    c = state['c'].reshape(64, 64)
    additions, costs = {}, {}
    mask = state['mask'][None, None]
    ref_map = state['r'].T.reshape(1, 1, 1024, 64, 64).contiguous()
    q_map = state['q'].T.reshape(1, 1024, 64, 64).contiguous()
    if 'P' in deps or candidate['common_crf']:
        from ics.official_data import load_inputs
        _, _, query = load_inputs(binding['row'], lab.ASSETS)
        host.set_target(query)
    for dep in deps:
        start = time.monotonic()
        if dep == 'B':
            native = F.interpolate(mask, (64, 64), mode='bilinear', align_corners=False)[0, 0] > .5
            additions[dep], background = b4.background_term(state['r'], state['q'], native)
        elif dep == 'V':
            prototype = lab.unit((state['r'] * state['coverage'][:, None]).sum(0))
            candidates, vote = host._locate_candidates(ref_map, q_map, mask, prototype, 1, 64, 64)
            additions[dep] = host.candidate_boost * (vote - .5)
        elif dep == 'P':
            dense = torch.zeros((64, 64), dtype=torch.bool); dense[::2, ::2] = True
            prior = host._build_seed_cluster_prior(ref_map, q_map, mask, candidates | dense, 1, 64, 64)
            additions[dep] = host.seed_cluster_boost * (prior - .5)
        elif dep in ('K_prob', 'K_minmax'):
            sf = c if dep == 'K_prob' else lab.minmax(c)
            additions[dep] = host._semantic_cluster_reweight_map(q_map, sf, 1 - sf)
        else:
            raise ValueError('Unknown frozen stage: ' + dep)
        costs[dep] = time.monotonic() - start
    correction = sum((additions[k] for k in active), torch.zeros_like(c))
    primary = state.get('c_mod', state['c']).reshape(64, 64)
    basis = candidate.get('decision_basis', 'region_mean')
    graph_info, extra_fields = None, {}
    if basis == 'reciprocal_graph':
        start = time.monotonic()
        neighbors = exact_neighbors(state['q'])
        decision_field, graph_info = a7.independent_graph(primary + correction, state['q'], neighbors)
        graph_info['neighbor_source'] = 'confirm CPU exact unit(q) top20 excluding self; block256'
        costs['independent_graph'] = time.monotonic() - start
        extra_fields['query_neighbors20'] = neighbors
    else:
        assert basis in ('region_mean', 'K64_region_mean')
        decision_field = state.get('modulated_region', state['ind0']).numpy().reshape(64, 64)
        if active:
            decision_field = decision_field + workflow.region_mean(correction.numpy(), state['regions64'].numpy())
    prediction = pc.readout(decision_field)
    if 'modulation_mask' in state and not active and basis != 'reciprocal_graph':
        prediction = state['modulation_mask']  # Preserve A4's FP32 endpoint mask rule.
    if candidate['common_crf']:
        from utils.refinement import crf_refine
        start = time.monotonic()
        prediction = crf_refine(host._crf, host._crf_band_px, host._crf_p_core,
                               host._tgt_image[None], torch.from_numpy(prediction)).numpy()
        costs['common_CRF'] = time.monotonic() - start
    detail = {'actual_stage_calls': list(deps), 'FoRIS_stage_count': len(deps),
              'stage_seconds': costs, 'decision_basis': basis,
              'complete_FoRIS_calls': 0, 'FoRIS_score_consumed': False}
    if graph_info is not None:
        detail['independent_graph'] = graph_info
    if 'modulation_detail' in state:
        detail['evidence_modulation'] = state['modulation_detail']
        extra_fields.update(c_mod32=state['c_mod'].numpy(), sim_max_fp16=state['sim_max_fp16'],
                            back_f_fp16=state['back_f_fp16'])
    if 'B' in deps:
        detail['background'] = background
    return prediction, {'c32': state['c'].numpy(), 'ind0_32': state['ind0'].numpy(),
                        'regions64': state['regions64'].numpy().astype(np.uint16),
                        'candidate_field': decision_field, **extra_fields}, detail


def infer(binding, candidate, host):
    counters, start = {}, time.monotonic()
    try:
        with raw_core.legal_inference(binding):
            state = independent_state(binding, counters, candidate.get('evidence_modulation'), candidate.get('reference_rho_gate'))
            base_seconds = time.monotonic() - start
            if state['reference_rho_gate_triggered']:
                prediction = pc.readout(state['c'])
                fields = {'c32': state['c'].numpy(), 'candidate_field': state['c'].numpy().reshape(64, 64)}
                detail = {'actual_stage_calls': [], 'FoRIS_stage_count': 0, 'decision_basis': 'raw_T',
                          'complete_FoRIS_calls': 0, 'FoRIS_score_consumed': False, 'stage_seconds': {},
                          'maxJ_graph_FoRIS_CRF_skipped': True}
            else:
                prediction, fields, detail = b4_prediction(binding, state, candidate, host)
            detail.update(reference_rho=state['reference_rho'],
                          reference_rho_gate_triggered=state['reference_rho_gate_triggered'])
        return {'candidate': pc.pack(prediction)}, fields, {
            **counters, **detail, 'base_field_seconds': base_seconds,
            'new_encoder_calls': 0, 'query_GT_reads': 0}
    finally:
        if host is not None:
            host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None
