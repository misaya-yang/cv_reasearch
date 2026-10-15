"""W25: choose SH's integration scale per episode with the label-free quality J already used for modulation.  (S)

Motivation: W24 shows every inherited constant of SH trades along one axis (more region integration helps
COCO/ISIC/SUIM, hurts LVIS/PACO/iSAID/Lung/Road); W09 shows the per-episode best lambda is worth +4.4 (oracle, six
values) and that lambda moves the cut, not the ranking; area-based choices failed (W09/W10, C4).  SH's novelty
modulation already chooses its strength per episode by J = P C / (P + C - P C), P = mean kernel transfer c over the
mask's native tokens, C = reference-FG mass captured through the reverse transfer back_f (rc.quality) -- a
label-free choice whose removal costs 1.31.  J has not been tried for the integration scale.
Arms (before the CRF; rho > .9 keeps T): SH (lambda 16); L4, L64 (both solves at that lambda);
  Jsel = the one of {L4, SH, L64} with the largest J;  orc = the best of the three by GT (diagnostic only).
Reading fixed before running: Jsel >= SH + 0.5 in the ten-domain mean with no domain significantly below SH ->
add the CRF and run the one-shot T-half check; otherwise J does not transfer to scale and the route is closed.

    python scripts/x1012/w25_j_scale.py produce [workers]
    python scripts/x1012/w25_j_scale.py report
"""
import gzip, json, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

OUT = c.task_path('W25')
LAMS = (4, 16, 64)


def one(binding):
    import torch
    import torch.nn.functional as F
    from w09_integration_lambda import solve
    from w24_sensitivity import mean_solve_lam
    rc, lab = c.rc, c.rc.lab
    gt, valid = c.truth(binding)
    r_, q, _, y, f = rc.independent(binding, {})
    out = {'index': str(binding['index'])}
    if float(f['reference_rho'][0]) > .9:
        mask = (lab.minmax(lab.up(torch.from_numpy(f['c32']).reshape(64, 64))) > .5).numpy()
        for a in ('SH', 'L4', 'L64', 'Jsel', 'orc'):
            out[a] = c.iu(mask, gt, valid)
        out['pick'] = 16
        return out
    c_mod, _ = rc.modulate(torch.from_numpy(f['c32']), torch.from_numpy(f['regions64'].astype(np.int64)),
                           f['ind0_32'], f['indn_32'], f['sim_max'], f['back_f'])
    L = rc.graph(q, f['query_neighbors20']).tocsr()
    cov = y.numpy()
    qn, rn = F.normalize(q.float(), dim=1), F.normalize(r_.float(), dim=1)
    fi = np.flatnonzero(cov >= .9)
    if not len(fi):
        fi = np.flatnonzero(cov == cov.max())
    guide = (qn @ F.normalize(rn[fi].mean(0), dim=0)).numpy()
    fg = torch.from_numpy(cov >= .5 if (cov >= .5).any() else cov == cov.max())
    mu_fg = F.normalize(rn[fg].mean(0), dim=0)
    bg = rn[~fg]
    if len(bg):
        hard = bg[(bg @ mu_fg).topk(max(1, int(.2 * len(bg)))).indices]
        mu_bg = F.normalize(hard.mean(0), dim=0); mu_bg = mu_bg - (mu_bg @ mu_fg) * mu_fg
        mu_bg = mu_bg / mu_bg.norm().clamp_min(1e-8)
    else:
        mu_bg = torch.zeros_like(mu_fg)
    p = (qn @ mu_fg - .55 * (qn @ mu_bg)).numpy().reshape(64, 64)
    up = lambda z: lab.up(torch.from_numpy(np.asarray(z, np.float32).reshape(64, 64))).numpy().astype(np.float64)
    masks, J = {}, {}
    for lam in LAMS:
        u = lab.minmax(torch.from_numpy(up(solve(c_mod.numpy(), f['log_e'], L, lam)))).numpy()
        m = up(mean_solve_lam(p, guide, L, lam))
        masks[lam] = .5 * m + .5 * u > .5
        J[lam] = rc.quality(masks[lam], f['c32'], f['back_f'].astype(np.float64))
    iou = {lam: c.iu(masks[lam], gt, valid) for lam in LAMS}
    pick = max(LAMS, key=lambda lam: (J[lam], lam == 16))
    best = max(LAMS, key=lambda lam: iou[lam][0] / max(iou[lam][1], 1))
    out.update(SH=iou[16], L4=iou[4], L64=iou[64], Jsel=iou[pick], orc=iou[best], pick=pick,
               J={str(k): v for k, v in J.items()})
    return out


def produce_all(workers):
    import w13_plugin as w13
    OUT.mkdir(parents=True, exist_ok=True)
    rows = w13.subset([b for b in c.rows() if c.half(b) == 'S'], 150)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in rows if str(b['index']) not in done]
    print('sample', len(rows), 'todo', len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool, path.open('a') as fh:
        for r in pool.map(one, todo, chunksize=4):
            fh.write(json.dumps(r) + '\n'); fh.flush()
    print('done', flush=True)


def report():
    import score
    from collections import Counter
    with gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt') as f:
        E1 = {r['index']: r for r in map(json.loads, f)}
    R = [dict(E1[r['index']], **r) for r in map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines())]
    rng = np.random.default_rng(c.SEED)
    names = [d[:6] for d in c.DOMAINS]
    print('n', len(R), f'\n## class mIoU before CRF minus SH (lambda 16)\n{"arm":6s}' + ''.join(f'{d:>8s}' for d in names) + '    mean  >=0  sig+  sig-')
    for a in ('L4', 'L64', 'Jsel', 'orc'):
        v, sp, sn = [], 0, 0
        for d in c.DOMAINS:
            L = [e for e in R if e['dataset'] == d]
            v.append(score.miou(L, a) - score.miou(L, 'SH')); lo, hi = score.paired_ci(L, a, 'SH', rng)
            sp += lo > 0; sn += hi < 0
        print(f'{a:6s}' + ''.join(f'{x:+8.2f}' for x in v) + f'{np.mean(v):+8.2f}{sum(x >= 0 for x in v):5d}{sp:6d}{sn:6d}')
    G = [e for e in R if 'J' in e]
    iou = lambda x: x[0] / max(x[1], 1)
    print('\nJ picks:', dict(Counter(e['pick'] for e in G)), ' oracle picks:',
          dict(Counter(max((4, 16, 64), key=lambda l: iou(e['SH' if l == 16 else f'L{l}'])) for e in G)))
    print('J picks the oracle lambda in', round(float(np.mean([e['Jsel'] == e['orc'] for e in G])), 3), 'of episodes')
    with gzip.open(ROOT / 'evidence/research/w25_j_scale_S.jsonl.gz', 'wt') as f:
        for e in R:
            f.write(json.dumps({k: e[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'SH', 'L4', 'L64', 'Jsel', 'orc', 'pick')}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 3) if sys.argv[1] == 'produce' else report()
