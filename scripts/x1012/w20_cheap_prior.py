"""W20: a cheap stand-in for FoRIS's seed-cluster prior inside the FoRIS-free head Xp.  (S half)

Motivation (W19): Xp+ (M from the one-prototype contrast p, fused with U0) matches MFU in mean and is faster
than FoRIS, but COCO is lower in both halves (Xp+ - MFU: S -1.66, T -2.12); ISIC/iSAID flip sign between
halves (noise at 150/domain).  On S, M0U+ (M from the full FoRIS score) - Xp+ on COCO is +1.30, so the gap
sits in what the FoRIS score adds over p.  FoRIS adds .20 (vote - .5) (NN vote, 0.066 s, no clustering) and
.25 (prior - .5), whose cost is an agglomerative clustering of all 4096 query tokens (0.19 s), plus Part 4
(another clustering, 0.18 s).
Scores for M (all keep everything else of Xp; compared BEFORE the final CRF, which adds ~+0.2 uniformly):
  Xp      p                                               (W19 head)
  XpV     p + V                     V = cached exact FoRIS vote term (cheap)
  XpVP    p + V + P                 P = cached exact FoRIS seed-cluster prior term   (control, not a candidate)
  XpVK    p + V + .25 (prior_K - .5)   FoRIS's seed-prior formula on the existing k-means-64 feature regions
  XpVKc   p + V + .25 (prior_Kc - .5)  same formula on k-means-64 of [feature, .35 colour, .20 position]
                                        (FoRIS's clustering input, spherical k-means instead of agglomerative)
  XpVKc16 same with 16 clusters.  Added after the formula check, before any result: FoRIS's agglomerative
          clustering (tau .6) yields only 5-18 clusters on the checked episodes, so 64 may be too fine.
Reading fixed before running: a candidate (XpVK / XpVKc / XpVKc16) is adopted if on COCO it recovers >= half of the
XpVP - Xp gap and its ten-domain mean is >= Xp's; then it gets the CRF and the T-half check.  If XpVP itself
does not close the COCO gap, the gap is in Part 4 / U's V,P and the prior is not the target.
Sample: the W13 S subset (<= 150 per domain) plus every S-half COCO episode (456).

    python scripts/x1012/w20_cheap_prior.py produce [workers]
    python scripts/x1012/w20_cheap_prior.py report
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

OUT = c.task_path('W20')
ARMS = ('Xp', 'XpV', 'XpVP', 'XpVK', 'XpVKc', 'XpVKc16')


def sample():
    import w13_plugin as w13
    rows = [b for b in c.rows() if c.half(b) == 'S']
    keep = {str(b['index']) for b in w13.subset(rows, 150)} | {str(b['index']) for b in rows if b['dataset'] == 'coco'}
    return [b for b in rows if str(b['index']) in keep]


def seed_prior(x, labels, mu_fg, candidates, k=64):
    """FoRIS _build_seed_cluster_prior arithmetic with given cluster labels (x: unit query features)."""
    import torch
    import torch.nn.functional as F
    labels = torch.as_tensor(labels, dtype=torch.long)
    onehot = F.one_hot(labels, k).float()
    area_all = onehot.sum(0).clamp_min(1.0)
    protos = F.normalize(onehot.T @ x / area_all[:, None], dim=1)
    matched = labels[candidates]
    if matched.numel() == 0:
        return candidates.float().numpy()
    area_w = torch.bincount(matched, minlength=k).float() / area_all
    cross = (onehot.T @ (x @ mu_fg)) / area_all
    ids = torch.unique(matched)
    seed = int(ids[torch.argmax((cross * area_w)[ids])])
    combined = cross * (protos[seed] @ protos.T).clamp_min(0) * area_w
    combined[seed] = combined.max().clamp_min(1e-6)
    combined = (combined - combined.min()) / (combined.max() - combined.min()).clamp_min(1e-6)
    return combined[labels].numpy()


def colour_tokens(binding):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    row = binding['row']
    with Image.open(c.ASSETS / row['query_path']) as opened:
        image = np.asarray(opened.convert('RGB')).copy()
    if row.get('query_crop') is not None:
        x0, y0, x1, y1 = row['query_crop']
        image = image[y0:y1, x0:x1]
    t = c.rc.host.transform(Image.fromarray(image))
    low = F.interpolate(t[None], size=(64, 64), mode='bilinear', align_corners=False)[0]
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    return F.normalize((low * std + mean).clamp(0, 1).reshape(3, -1).T, dim=1)


def one(binding):
    import torch
    import torch.nn.functional as F
    import produce
    from w09_integration_lambda import solve
    from w19_simple_head import heads, mean_graph, mean_solve
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    r = {'index': str(binding['index'])}
    if produce.fallback(fields):
        with np.load(c.prediction_path('T1', binding), allow_pickle=False) as z:
            t_mask = c.unpack(z['T'])
        for a in ARMS:
            r[a] = c.iu(t_mask, gt, valid)
        return r
    mm = lambda f: c.rc.lab.minmax(torch.from_numpy(c.up(np.asarray(f, np.float32)))).numpy().astype(np.float64)
    u0 = mm(solve(fields['c_mod'], fields['log_e'], c.laplacian(fields), 16).reshape(64, 64))
    q, rr, cov, guide, cheap = heads(binding, fields)
    p = cheap['p'].reshape(64, 64)
    V, P = fields['addition_V'], fields['addition_P']
    fg = cov >= .5 if (cov >= .5).any() else cov == cov.max()
    mu_fg = F.normalize(rr[torch.from_numpy(fg)].mean(0), dim=0)
    dense = torch.zeros((64, 64), dtype=torch.bool); dense[::2, ::2] = True
    cand = (torch.from_numpy(V > 0) | dense).reshape(-1)
    prior_k = seed_prior(q, fields['regions64'].astype(np.int64), mu_fg, cand).reshape(64, 64)
    pos = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing='ij'), -1).reshape(-1, 2)
    xc = F.normalize(torch.cat([q, .35 * colour_tokens(binding), .20 * F.normalize(pos, dim=1)], 1), dim=1)
    labels_c = (xc @ c.rc.lab.kmeans(xc, 64, iters=6).T).argmax(1).numpy()
    prior_kc = seed_prior(q, labels_c, mu_fg, cand).reshape(64, 64)
    labels_16 = (xc @ c.rc.lab.kmeans(xc, 16, iters=6).T).argmax(1).numpy()
    prior_16 = seed_prior(q, labels_16, mu_fg, cand, k=16).reshape(64, 64)
    scores = {'Xp': p, 'XpV': p + V, 'XpVP': p + V + P,
              'XpVK': p + V + .25 * (prior_k - .5), 'XpVKc': p + V + .25 * (prior_kc - .5),
              'XpVKc16': p + V + .25 * (prior_16 - .5)}
    L = mean_graph(q)
    for a, s in scores.items():
        m = c.up(mean_solve(s, guide, L)).astype(np.float64)
        r[a] = c.iu(.5 * m + .5 * u0 > .5, gt, valid)
    return r


def produce_all(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sample()
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in rows if str(b['index']) not in done]
    print('sample', len(rows), 'todo', len(todo), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool, path.open('a') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=4), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 200 == 0:
                print(n, len(todo), flush=True)
    print('done', flush=True)


def report():
    import score
    with gzip.open(ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', 'rt') as f:
        E1 = {r['index']: r for r in map(json.loads, f)}
    R = [dict(E1[r['index']], **r) for r in map(json.loads, (OUT / 'episodes.jsonl').read_text().splitlines())]
    with gzip.open(ROOT / 'evidence/research/w13_plugin_S.jsonl.gz', 'rt') as f:
        W13 = {r['index']: r for r in map(json.loads, f)}
    rng = np.random.default_rng(c.SEED)
    print('n', len(R), ' COCO', sum(e['dataset'] == 'coco' for e in R))
    names = [d[:6] for d in c.DOMAINS]
    print(f'\n## class mIoU before CRF minus Xp\n{"arm":7s}' + ''.join(f'{d:>17s}' for d in names) + '     mean  >=0')
    for a in ARMS[1:]:
        v, cells = [], []
        for d in c.DOMAINS:
            L = [e for e in R if e['dataset'] == d]
            x = score.miou(L, a) - score.miou(L, 'Xp'); lo, hi = score.paired_ci(L, a, 'Xp', rng)
            v.append(x); cells.append(f'{x:+6.2f}[{lo:+5.1f},{hi:+5.1f}]')
        print(f'{a:7s}' + ''.join(f'{s:>17s}' for s in cells) + f'{np.mean(v):+9.2f}{sum(x >= 0 for x in v):5d}')
    L = [e for e in R if e['index'] in W13]
    for e in L:
        e['M0U'] = W13[e['index']]['M0U']
    v = [score.miou([e for e in L if e['dataset'] == d], 'M0U') - score.miou([e for e in L if e['dataset'] == d], 'Xp') for d in c.DOMAINS]
    print(f'{"M0U":7s}' + ''.join(f'{x:+17.2f}' for x in v) + f'{np.mean(v):+9.2f}   (W13 sample only: M from the full FoRIS score, U0)')
    print('\nabsolute class mIoU before CRF:', {a: round(float(np.mean([score.miou([e for e in R if e['dataset'] == d], a) for d in c.DOMAINS])), 2) for a in ARMS})
    with gzip.open(ROOT / 'evidence/research/w20_cheap_prior_S.jsonl.gz', 'wt') as f:
        for e in R:
            f.write(json.dumps({k: e[k] for k in ('index', 'dataset', 'fold', 'cls', 'photo', 'gt') + ARMS}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 3) if sys.argv[1] == 'produce' else report()
