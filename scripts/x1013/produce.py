"""Natural predictions for T0--T7; query GT is guarded throughout each call."""
from __future__ import annotations

import time
import numpy as np
import torch
import common as c


def fallback(fields):
    return float(fields['reference_rho'][0]) > .9


def candidate(fields):
    return c.readout(c.foreground_field(fields))


def t1(binding, fields):
    import repo_methods as repo
    mean_field = repo.mean_field(binding)
    mean = repo.field_mask(mean_field).numpy()
    mismatch = -1
    if 'G' in binding['old_arms']:
        old = c.rc.lab.arm_cli(binding, 'G').numpy()
        mismatch = int(np.count_nonzero(mean ^ old))
    c.save_npz(c.task_path('T1') / 'fields' / (c.key(binding) + '.npz'),
               mean=mean_field, mean_xor=np.asarray(mismatch, np.int32))
    return {'foris': c.pack(c.rc.lab.arm_cli(binding, 'B').numpy()),
            'foris_noCRF': c.pack(c.readout(repo.foris_score(binding))),
            'mean': c.pack(mean), 'T': c.pack(c.readout(fields['c32']))}


def t2(binding, fields):
    graph, e = c.laplacian(fields), fields['explanation']
    kernel = fields['c32'].reshape(64, 64)
    modulated = fields['c_mod'].reshape(64, 64)
    v, p = fields['addition_V'], fields['addition_P']
    targets = {'A0': kernel, 'A1': modulated, 'A2': modulated + v,
               'A3': modulated + p, 'A4': fields['primary'], 'A5': kernel + (v + p),
               'A6': fields['primary'], 'A7': fields['primary']}
    predictions = {}
    for name, primary in targets.items():
        if fallback(fields):
            field = kernel
        elif name == 'A7':
            field = primary
        else:
            field = c.solve(primary, np.ones_like(e) if name == 'A6' else e, graph)
        predictions[name] = c.pack(c.readout(field))
    if not np.array_equal(predictions['A4'], c.pack(candidate(fields))):
        c.write(c.task_path('T2') / 'gate_failure.json',
                {'index': str(binding['index']), 'gate': 'A4 == T0'})
        raise RuntimeError(f"A4 identity gate failed: {binding['index']}")
    return predictions


def t3(binding, fields):
    keep_fallback = fallback(fields) and int(fields['touched']) >= 64
    return {'VP_no_fallback': c.pack(c.readout(fields['VP_UN'])),
            'G1': c.pack(c.readout(fields['c32'] if keep_fallback else fields['VP_UN']))}


def t4(binding, fields):
    if fallback(fields):
        initial = c.pack(candidate(fields))
        return {name: initial for name in ('F6a', 'F6b', 'F6c')}
    guided = c.solve(fields['primary'], fields['explanation'], c.laplacian(fields), fields['guide'])
    return {'F6a': c.pack(c.readout(guided)), 'F6b': c.pack(c.readout(guided, absolute=True)),
            'F6c': c.pack(c.readout(fields['VP_UN'], absolute=True))}


def cluster_mean(field, labels):
    labels = np.asarray(labels, dtype=np.int64).ravel()
    means = np.bincount(labels, weights=np.asarray(field).ravel()) / np.bincount(labels)
    return means[labels].reshape(64, 64).astype(np.float32)


def t5(binding, fields):
    from utils.clustering import agglomerative_clustering
    q = c.rc.lab.debias(c.rc.lab.feats(binding, 'query'))
    labels = agglomerative_clustering(c.rc.lab.unit(q), tau=.6).numpy()
    c.save_npz(c.task_path('T5') / 'clusters' / (c.key(binding) + '.npz'),
               labels=labels.astype(np.uint16), n=np.asarray(int(labels.max()) + 1, np.int32))
    if fallback(fields):
        initial = c.pack(candidate(fields))
        return {'F1': initial, 'F1b': initial}
    field = fields['VP_UN']
    averaged = cluster_mean(field, labels)
    return {'F1': c.pack(c.readout(averaged)), 'F1b': c.pack(c.readout(.5 * field + .5 * averaged))}


def background(binding):
    """Wrap the reference loader; all other R5 calculations stay unchanged."""
    original = c.rc.lab.mask1024
    reference = c.Path(binding['row']['reference_mask_path']).resolve()
    def inverse(path):
        mask = original(path)
        return ~mask if c.Path(path).resolve() == reference else mask
    c.rc.lab.mask1024 = inverse
    try:
        fields = c.role_fields(binding)
    finally:
        c.rc.lab.mask1024 = original
    # Background always uses the graph path, regardless of its own rho.
    return fields


def symmetric(front, back):
    m_front, m_back = c.readout(front), c.readout(back)
    rewritten = int(m_back.sum()) < int(m_front.sum())
    average = .5 * (c.rc.lab.minmax(torch.from_numpy(c.up(front))).numpy()
                    + 1 - c.rc.lab.minmax(torch.from_numpy(c.up(back))).numpy()) > .5
    return m_front, m_back, (~m_back if rewritten else m_front), average, rewritten


def t6(binding, fields):
    if fallback(fields):
        initial = c.pack(candidate(fields))
        c.save_npz(c.task_path('T6') / 'fields' / (c.key(binding) + '.npz'),
                   skipped=np.asarray(True), rewritten=np.asarray(False))
        return {name: initial for name in ('S-self', 'S-avg', 'S-ref')}
    bg = background(binding)
    m_front, m_back, self_mask, average, rewritten = symmetric(fields['VP_UN'], bg['VP_UN'])
    c.save_npz(c.task_path('T6') / 'fields' / (c.key(binding) + '.npz'), **bg,
               skipped=np.asarray(False), rewritten=np.asarray(rewritten))
    return {'S-self': c.pack(self_mask), 'S-avg': c.pack(average),
            'S-ref': c.pack(~m_back if float(fields['reference_fraction']) > .5 else m_front)}


def crf(binding, mask):
    global CRF_HOST, CRF_ADAPTER
    if 'CRF_HOST' not in globals():
        CRF_HOST, CRF_ADAPTER = c.rc.pc.host_cpu()
        torch.set_num_threads(2)
    from ics.official_data import decoded_rgb
    from utils.refinement import crf_refine
    row = binding['row']
    query = decoded_rgb(c.ASSETS / row['query_path'], row.get('query_crop'))
    CRF_HOST.set_target(query)
    image = CRF_HOST._tgt_image.clone().unsqueeze(0)
    result = crf_refine(CRF_HOST._crf, CRF_HOST._crf_band_px, CRF_HOST._crf_p_core,
                        image, torch.from_numpy(mask)).numpy()
    if CRF_ADAPTER.encoder is not None or CRF_ADAPTER.encoder_calls:
        raise RuntimeError('CRF unexpectedly invoked an encoder')
    CRF_HOST._tgt_image = CRF_HOST._orig_tgt_size = None
    return result


def t7(binding, fields):
    initial = candidate(fields)
    refined = c.pack(crf(binding, initial))
    return {'C1': c.pack(initial) if fallback(fields) else refined, 'C2': refined}


def appendix(binding, fields):
    # Fixed user-requested control, evaluated only after T9; never in the chain.
    if fallback(fields):
        return {'MU_control': c.pack(candidate(fields))}
    with np.load(c.task_path('T1') / 'fields' / (c.key(binding) + '.npz'), allow_pickle=False) as archive:
        mean = c.up(archive['mean']).astype(np.float64)
    independent = c.rc.lab.minmax(torch.from_numpy(c.up(fields['VP_UN']))).numpy().astype(np.float64)
    return {'MU_control': c.pack(.5 * mean + .5 * independent > .5)}


def t0(binding):
    fields = c.role_fields(binding)
    c.save_npz(c.fields_path(binding), **fields)
    return {'candidate': c.pack(c.readout(c.foreground_field(fields)))}


def worker(job):
    task, binding = job
    destination = c.prediction_path(task, binding)
    if destination.exists() and (task != 'T0' or c.fields_path(binding).exists()):
        return {'index': str(binding['index']), 'cached': True}
    started = time.monotonic()
    with c.rc.pc.inference_only(binding):
        if task == 'T0':
            predictions = t0(binding)
        else:
            predictions = globals()[task.lower()](binding, c.load_fields(binding))
        c.save_npz(destination, **predictions)
    return {'index': str(binding['index']), 'seconds': time.monotonic() - started}
