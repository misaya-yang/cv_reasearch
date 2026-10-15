"""W32: does the frozen encoder's last-block attention key carry the entity structure that its outputs lack?
Plan and readings: docs/research/x1012_tasks/W32_key_entity_PLAN.md (fixed before this file was run).

Same 416 S episodes, field G, oracles and rule as W15 (imported, not copied).  New query-only groupings:
  kspK  dense affinity max(0, cos) of last-block keys (after k_norm, before RoPE; patch tokens; 16 heads concatenated)
  ospK  the same construction on debiased O24 outputs (control: key vs dense construction)
NJW spectral clustering, K in {4, 8, 16, 32}.  CPU, FP32, one process, 2 threads; no features are written.

    nice -n 19 python scripts/x1012/w32_key_entity.py produce
    python scripts/x1012/w32_key_entity.py report
"""
import gzip, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
sys.path.insert(0, str(ROOT / 'scripts/x1012s'))
import common as c
import w15_query_extent as w15

OUT = c.task_path('W32')
KS = w15.KS
GROUPS = tuple(f'{p}sp{k}' for p in 'ko' for k in KS)


def spectral(x, seed=0):
    """NJW on the dense non-negative cosine affinity of the rows of x (4096 x D)."""
    from scipy.sparse.linalg import eigsh
    from scipy.cluster.vq import kmeans2
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    W = np.maximum(x @ x.T, 0); np.fill_diagonal(W, 0)
    s = 1 / np.sqrt(np.maximum(W.sum(1), 1e-12))
    M = s[:, None] * W * s[None, :]
    vals, vecs = eigsh(M, k=max(KS), which='LA', tol=1e-6, v0=np.ones(len(M)))
    order = np.argsort(-vals); vals, vecs = vals[order], vecs[:, order]
    out = {}
    for k in KS:
        e = vecs[:, :k]
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        _, lab = kmeans2(e, k, minit='++', seed=seed + k, iter=30)
        out[k] = lab
    return out, vals[:10].tolist()


def produce():
    import psutil, torch
    torch.set_num_threads(2)
    c.initialize(); torch.set_num_threads(2)
    import photo_holdout as ph
    task, rt = ph.runtime()
    enc = ph.module('w32_encoder', ph.ASSETS / 'a/x1013_official/frozen/encoder.py').TimmDINOv3(str(ph.WEIGHTS)).eval().requires_grad_(False)
    grab = {}
    enc.m.blocks[-1].attn.k_norm.register_forward_hook(lambda mod, inp, out: grab.__setitem__('k', out))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    rows = [b for b in w15.sample() if str(b['index']) not in done]
    print('todo', len(rows), flush=True)
    with path.open('a') as fh:
        for n, b in enumerate(rows, 1):
            while psutil.cpu_percent(interval=2) > 70:
                time.sleep(30)
            began = time.monotonic()
            # W15's natural mask, oracles and rule (same field G).
            fields = c.load_fields(b)
            gt, valid = c.truth(b)
            g = gt & valid
            with np.load(c.task_path('T1') / 'fields' / (c.key(b) + '.npz'), allow_pickle=False) as z:
                m = c.up(z['mean'].astype(np.float32)).astype(np.float64)
            u = c.rc.lab.minmax(torch.from_numpy(c.up(fields['VP_UN']))).numpy().astype(np.float64)
            nat = (.5 * m + .5 * u > .5) & valid
            v_t, g_t, n_t = w15.tokens(valid), w15.tokens(g), w15.tokens(nat)
            iu = lambda mask: [int((mask & g).sum()), int((mask | g).sum())]
            blow = lambda sel: np.repeat(np.repeat(sel.reshape(64, 64), 16, 0), 16, 1) & valid
            r = {'index': str(b['index']), 'dataset': b['dataset'], 'nat': iu(nat),
                 'natQ': iu(blow((n_t * 2 >= np.maximum(v_t, 1)) & (v_t > 0)))}
            # One CPU forward: O24 outputs and last-block keys of the query.
            _, _, q = rt.load_inputs(b['row'], ph.ASSETS)
            with torch.inference_mode():
                o = enc.get_intermediate_layers(rt.host._transform(q)[None], n=1, reshape=True)[0]
            o = o[0].permute(1, 2, 0).reshape(4096, 1024)
            k = grab.pop('k')[0, :, 5:, :].permute(1, 0, 2).reshape(4096, 1024).numpy().astype(np.float64)
            feats = {'k': k, 'o': c.rc.lab.debias(o).numpy().astype(np.float64)}
            for p, x in feats.items():
                labs, r[p + '_eig10'] = spectral(x)
                for K, lab in labs.items():
                    name = f'{p}sp{K}'
                    r['SO_' + name] = w15.seg_oracle(lab, v_t, g_t)
                    nn_, kk = np.bincount(lab, v_t, 4096), np.bincount(lab, n_t, 4096)
                    r['rule_' + name] = iu(blow(((kk * 2 >= nn_) & (nn_ > 0))[lab]))
            r['seconds'] = round(time.monotonic() - began, 1)
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 10 == 0:
                print(n, len(rows), r['seconds'], flush=True)
    print('done', flush=True)


def report():
    import score
    W15 = {r['index']: r for r in map(json.loads, (c.task_path('W15') / 'episodes.jsonl').read_text().splitlines())}
    E1 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt'))}
    W = {r['index']: r for r in map(json.loads, w15.W14.read_text().splitlines()) if 'G' in r}
    gaps = np.array([w['G']['oracle'] - w['G']['natural'] for w in W.values()])
    cut = np.quantile(gaps, .8)
    R = [dict(E1[r['index']], **r) for r in map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines())]
    same = sum(R_['nat'] == W15[R_['index']]['nat'] and R_['natQ'] == W15[R_['index']]['natQ'] for R_ in R)
    iou = lambda x: 100 * x[0] / max(x[1], 1)
    for e in R:
        w = W[e['index']]['G']
        e['orc'], e['top'] = w['oracle'], w['oracle'] - w['natural'] >= cut
        for g in w15.GROUPS:
            e['SO_' + g] = W15[e['index']]['SO_' + g]
    lines = [f'n {len(R)}  top-gap {sum(e["top"] for e in R)} (gap >= {cut:.1f})  nat/natQ identical to W15: {same}/{len(R)}']
    doms = [d for d in c.DOMAINS if any(e['dataset'] == d for e in R)]
    cols = ['nat', 'natQ', 'orc'] + ['SO_sp' + str(k) for k in KS] + ['SO_' + g for g in GROUPS]
    lines.append('\n## mean episode IoU: oracles vs natural (SO_spK = W15 kNN-graph spectral on outputs)')
    lines.append(f'{"subset":12s}{"n":>5s}' + ''.join(f'{k:>10s}' for k in cols))
    val = lambda e, k: iou(e[k]) if isinstance(e[k], list) else e[k]
    for nm, L in [('all', R), ('top-gap', [e for e in R if e['top']]), ('rest', [e for e in R if not e['top']])] + \
                 [(d[:12], [e for e in R if e['dataset'] == d]) for d in doms]:
        lines.append(f'{nm:12s}{len(L):5d}' + ''.join(f'{np.mean([val(e, k) for e in L]):10.1f}' for k in cols))
    lines.append('\n## rule (segment majority of nat) minus natQ, class mIoU per domain')
    lines.append(f'{"":10s}' + ''.join(f'{d[:6]:>8s}' for d in doms) + '    mean  >=0')
    for g in GROUPS:
        v = [score.miou([e for e in R if e['dataset'] == d], 'rule_' + g) - score.miou([e for e in R if e['dataset'] == d], 'natQ') for d in doms]
        lines.append(f'{g:10s}' + ''.join(f'{x:+8.2f}' for x in v) + f'{np.mean(v):+8.2f}{sum(x >= 0 for x in v):5d}')
    lines.append('\n## median of the top-10 eigenvalues (dense affinity)')
    for p in 'ko':
        lines.append(p + ' ' + ' '.join(f'{x:.3f}' for x in np.median([e[p + '_eig10'] for e in R], axis=0)))
    text = '\n'.join(lines)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    with gzip.open(ROOT / 'evidence/research/w32_key_entity_S.jsonl.gz', 'wt') as f:
        for e in R:
            f.write(json.dumps({k: e[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'nat', 'natQ', 'k_eig10', 'o_eig10')
                                + tuple('SO_' + g for g in GROUPS) + tuple('rule_' + g for g in GROUPS)}) + '\n')


if __name__ == '__main__':
    produce() if sys.argv[1] == 'produce' else report()
