"""A7: independent-field CRF and reciprocal query-graph smoothing.

No FoRIS score or rank correction enters inference. The original CRF function
uses the prebuilt host and reuses image geometry across the three baseline masks.
The default cohort is the full fixed explore manifest; seed150 is a first pass.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import cg
import lab
import plan_common as pc
import a2_exchange as workflow

BASES = ('IND0', 'INDN', 'T')
PRED_ARMS = BASES + ('IND0_crf', 'INDN_crf', 'T_crf', 'IND_graph')
REQUIRED = ('c32', 'ind0_32', 'indn_32', 'mask/IND0', 'mask/INDN',
            'mask/T', 'query_neighbors20', 'ref_stats')


def independent_graph(c, q, neighbors):
    """MEAN's W/A/lambda/CG, using independent c and no reference rank term."""
    q = F.normalize(torch.as_tensor(q, dtype=torch.float32), dim=1)
    n = q.shape[0]
    ids = torch.as_tensor(neighbors, dtype=torch.long)
    assert ids.shape == (n, 20) and int(ids.min()) >= 0 and int(ids.max()) < n
    assert not bool((ids == torch.arange(n)[:, None]).any()), 'Self-neighbor in P0 graph'
    similarities = torch.empty((n, 20), dtype=torch.float32)
    # Reuse the frozen neighbor identities; avoid allocating N x 20 x C at once.
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        similarities[start:stop] = (q[start:stop, None, :] * q[ids[start:stop]]).sum(-1)
    distance = (1 - similarities).clamp_min(0)
    weights = torch.exp(-distance / distance[:, -1:].clamp_min(1e-6)).numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(n), 20), ids.numpy().ravel())), shape=(n, n))
    w = w.multiply(w.T)
    w.data = np.sqrt(w.data)
    degree = np.asarray(w.sum(1)).ravel()
    w = w / max(float(degree.mean()), 1e-8)
    degree = np.asarray(w.sum(1)).ravel()
    # Same minmax and confidence A as source MEAN; y=s deletes its rank correction.
    s = lab.minmax(torch.as_tensor(c, dtype=torch.float32).reshape(-1)).numpy()
    assert s.size == n and np.isfinite(s).all()
    a = .1 + np.abs(2 * s - 1)
    a = (a / a.mean()).astype(np.float64)
    y = s.astype(np.float64)
    H = sparse.diags(a) + 16 * (sparse.diags(degree) - w)
    rhs = a * y
    iterations = [0]
    def callback(_):
        iterations[0] += 1
    z, status = cg(H, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300, callback=callback)
    if status:
        raise RuntimeError('Independent graph CG failure: ' + str(status))
    assert np.isfinite(z).all()
    relative = float(np.linalg.norm(H @ z - rhs) / max(np.linalg.norm(rhs), 1e-12))
    info = {'lambda': 16, 'neighbors': 20, 'neighbor_source': 'P0 query_neighbors20',
            'reciprocal_undirected_edges': int(w.nnz // 2), 'cg_iterations': iterations[0],
            'cg_relative_residual': relative, 'rank_correction': False, 'FoRIS_score_consumed': False,
            'confidence': 's=minmax(c32); A=(.1+abs(2*s-1))/mean(.1+abs(2*s-1))'}
    return z.reshape(64, 64).astype(np.float32), info


def produce(cohort):
    sources = (Path(__file__), Path(workflow.__file__), Path(pc.__file__), Path(lab.__file__),
               lab.ROOT / 'src/ics/methods/mean_control.py', lab.ROOT / 'scripts/cached_dino.py',
               lab.ROOT / 'src/ics/m4_crf_cached.py',
               lab.ASSETS / 'third_party/foris_official/utils/refinement.py')
    method = {'CRF': 'original crf_refine on sealed IND0/INDN/T masks; same FoRIS host parameters; reuse geometry per image',
              'graph': 'P0 reciprocal 20NN weights; lambda16; s=minmax(c32); source confidence A; rhs=A*s; no rank term',
              'graph_readout': 'bilinear1024 -> minmax -> >.5; replaces IND K64 region averaging',
              'baseline': 'FoRIS archives enter only the separately invoked scoring phase'}
    out, rows, config = workflow.prepare_run('A7', cohort, REQUIRED, PRED_ARMS, method, sources)
    if (out / 'sealed.json').exists():
        print('A7 ' + cohort + ' already sealed', flush=True)
        return
    host, adapter = pc.host_cpu()
    from ics.official_data import load_inputs
    from utils.refinement import crf_refine
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
                assert all(fields[k].dtype == np.float32 and fields[k].shape == (4096,)
                           and np.isfinite(fields[k]).all() for k in ('c32', 'ind0_32', 'indn_32'))
                predictions = {a: fields['mask/' + a] for a in BASES}
                q = lab.debias(lab.feats(b, 'query'))
                graph, graph_info = independent_graph(fields['c32'], q, fields['query_neighbors20'])
                predictions['IND_graph'] = pc.pack(pc.readout(graph))
                _, _, query = load_inputs(b['row'], lab.ASSETS)
                host.set_target(query)
                image = host._tgt_image.clone().unsqueeze(0)
                geometries = None
                for a in BASES:
                    initial = torch.from_numpy(pc.unpack(fields['mask/' + a]))
                    refined = crf_refine(host._crf, host._crf_band_px, host._crf_p_core, image, initial)
                    predictions[a + '_crf'] = pc.pack(refined.numpy())
                    current = (host._crf.spatial_filter._prepared_geometry,
                               host._crf.bilateral_filter._prepared_geometry)
                    if geometries is None:
                        geometries = current
                    else:
                        assert all(x is y for x, y in zip(geometries, current)), 'Same-image CRF geometry was rebuilt'
                assert adapter.encoder is None and adapter.encoder_calls == 0
            record = workflow.save_record(out, b, config, predictions, {'independent_graph': graph}, started,
                                          {'graph': graph_info, 'crf_geometry_reused': True,
                                           'FoRIS_score_consumed': False, 'ref_stats': fields['ref_stats'].tolist()})
            records.append(record)
        finally:
            host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None
        if i <= 3 or i % 50 == 0:
            print({'task': 'A7', 'cohort': cohort, 'completed': i, 'n': len(rows),
                   'seconds': round(time.monotonic() - began, 1)}, flush=True)
    workflow.seal_run(out, rows, records, began)


def score(cohort):
    out, result = workflow.score_task('A7', cohort, PRED_ARMS, [
        '独立推理固定完整队列，不消费FoRIS score/MEAN rank修正；只读P0字段、查询缓存特征、参考掩码及RGB。全部预测先封存，再绑定FoRIS基线并读query GT。',
        'IND0/INDN/T基线逐位复用P0封存mask；三者均调用原utils.refinement.crf_refine，参数与FoRIS host相同，同图CRF几何复用。',
        '图平滑以minmax(c32)作独立unary，使用P0的20近邻、互惠几何平均权重、平均度归一化、原confidence A及lambda16；没有参考guide或FoRIS秩修正。',
        '图输出直接作token场读出，替代IND0区域平均；CPU单进程4线程，新增编码为零。',
    ])
    increments = {d: {a + '_CRF_delta_pp': s[a + '_crf']['class_miou'] - s[a]['class_miou'] for a in BASES}
                  for d, s in result['summary'].items()}
    for d, s in result['summary'].items():
        increments[d]['graph_delta_to_IND0_pp'] = s['IND_graph']['class_miou'] - s['IND0']['class_miou']
    macro = {k: float(np.mean([v[k] for v in increments.values()])) for k in next(iter(increments.values()))}
    pc.write(out / 'generic_increments.json', {'per_domain': increments, 'equal_domain_mean': macro})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score'))
    parser.add_argument('--cohort', choices=('seed150', 'explore'), default='explore')
    args = parser.parse_args()
    produce(args.cohort) if args.mode == 'produce' else score(args.cohort)
