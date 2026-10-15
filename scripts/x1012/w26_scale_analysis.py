"""W26-0: can SH's per-episode integration scale be chosen?  Read-only analysis of existing S-half evidence.

Motivation: SH+ is below FoRIS on COCO and ISIC in both halves (W23); W24 shows every constant trades along the
integration-scale axis; W25 closes max-J and leaves a three-lambda oracle of +1.57.  Before planning option C (a
per-episode scale signal) we measure what the existing per-episode records allow:
  1  headroom structure: fixed lambda, best fixed lambda per domain, two- and three-way oracles, concentration;
  2  within-domain rank correlation of label-free quantities (and GT area, diagnostic) with IoU(64) - IoU(16);
  3  capability: a supervised selector trained leave-one-domain-out on those quantities (diagnostic only);
  4  where SH+ loses to FoRIS / MFU on COCO and ISIC (GT-area bins; FN and FP pixels).
No encoder, no new head runs; inputs are w25_j_scale_S, w23_simple_pipeline_{S,T}, x1013_E1_iu, w11_t_ablation and
w25_j_values_S (the J values of W25, None for rho > .9).

    python scripts/x1012/w26_scale_analysis.py
"""
import gzip, json, sys, zlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
import score

A = 1024 * 1024
LAM = {4: 'L4', 16: 'SH', 64: 'L64'}
iou = lambda x: x[0] / max(x[1], 1)
ld = lambda f: [json.loads(l) for l in gzip.open(ROOT / f'evidence/research/{f}.jsonl.gz', 'rt')]


def table(R, arms, ref):
    out = {}
    for d in sorted({e['dataset'] for e in R}):
        L = [e for e in R if e['dataset'] == d]
        out[d] = {a: score.miou(L, a) - score.miou(L, ref) for a in arms}
    return out


def show(title, T):
    arms = next(iter(T.values())).keys()
    print(f'\n## {title}\n{"":10s}' + ''.join(f'{a:>14s}' for a in arms))
    for d, v in T.items():
        print(f'{d[:9]:10s}' + ''.join(f'{x:+14.2f}' for x in v.values()))
    print(f'{"mean":10s}' + ''.join(f'{np.mean([v[a] for v in T.values()]):+14.2f}' for a in arms))


def main():
    from scipy.stats import spearmanr
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    R = ld('w25_j_scale_S')
    J = {r['index']: r['J'] for r in ld('w25_j_values_S')}
    rho = {r['index']: r['rho'] for r in ld('w23_simple_pipeline_S')}
    for e in R:
        a = {l: e[k] for l, k in LAM.items()}
        e['o16_64'] = max(a[16], a[64], key=iou); e['o4_16'] = max(a[4], a[16], key=iou)
    T = table(R, ('L4', 'L64', 'o4_16', 'o16_64', 'orc'), 'SH')
    for v in T.values():
        v['dom'] = max(0., v['L4'], v['L64'])
    show('1 class mIoU before CRF minus SH (lambda 16); dom = best fixed lambda per domain (tuned, diagnostic)', T)

    G = [e for e in R if J[e['index']]]
    gain = np.sort([100 * (iou(e['orc']) - iou(e['SH'])) for e in G])[::-1]
    print(f'non-fallback {len(G)}: mean per-episode oracle gain {gain.mean():.2f}; top 10% hold {gain[:len(G) // 10].sum() / gain.sum():.2f}, '
          f'top 20% {gain[:len(G) // 5].sum() / gain.sum():.2f}; episodes gaining > 5 points {np.mean(gain > 5):.3f}')
    top = sorted(G, key=lambda e: iou(e['SH']) - iou(e['orc']))[:len(G) // 10]
    print(f"top 10%: winner lambda 64 in {sum(iou(e['L64']) >= iou(e['L4']) for e in top)}/{len(top)}; "
          f"SH IoU {100 * np.mean([iou(e['SH']) for e in top]):.1f} vs {100 * np.mean([iou(e['SH']) for e in G if e not in top]):.1f} elsewhere")

    def feats(e, gt=False):
        j, a = J[e['index']], {l: max(e[k][2], 1) / A for l, k in LAM.items()}
        x = [np.log(a[16]), np.log(a[64] / a[16]), np.log(a[4] / a[16]), j['4'], j['16'], j['64'],
             j['64'] - j['16'], j['4'] - j['16'], rho[e['index']]]
        return x + ([np.log(max(e['gt'], 1) / A)] if gt else [])
    names = ('log a16', 'grow64', 'grow4', 'J4', 'J16', 'J64', 'dJ64', 'dJ4', 'rho', 'GT area')
    print('\n## 2 within-domain Spearman with IoU(64) - IoU(16)  (GT area is diagnostic)')
    print(f'{"":10s}' + ''.join(f'{n:>9s}' for n in names))
    for d in sorted({e['dataset'] for e in G}) + ['ALL']:
        L = [e for e in G if d in ('ALL', e['dataset'])]
        if len(L) < 30:
            continue
        X, y = np.array([feats(e, True) for e in L]), [iou(e['L64']) - iou(e['SH']) for e in L]
        print(f'{d[:9]:10s}' + ''.join(f'{spearmanr(X[:, i], y)[0]:+9.2f}' for i in range(len(names))))

    def select(split, gt, model):
        for e in R:
            e['sel'] = e['SH']
        for k in sorted({split(e) for e in G}):
            tr = [e for e in G if split(e) != k]; te = [e for e in G if split(e) == k]
            Xtr, Xte = np.array([feats(e, gt) for e in tr]), np.array([feats(e, gt) for e in te])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            pred = {}
            for l in (4, 64):
                m = Ridge(1.0) if model == 'ridge' else GradientBoostingRegressor(
                    n_estimators=150, max_depth=2, learning_rate=.05, subsample=.8, random_state=0)
                m.fit((Xtr - mu) / sd, [iou(e[LAM[l]]) - iou(e['SH']) for e in tr])
                pred[l] = m.predict((Xte - mu) / sd)
            for i, e in enumerate(te):
                e['sel'] = e[LAM[max((0., 16), (pred[4][i], 4), (pred[64][i], 64))[1]]]
        return {d: v['sel'] for d, v in table(R, ('sel',), 'SH').items()}
    S = {}
    for gt in (False, True):
        for model in ('ridge', 'gbr'):
            S[f"{model}{'+GT' if gt else ''}"] = select(lambda e: e['dataset'], gt, model)
    S['gbr+GT pooled'] = select(lambda e: zlib.crc32(e['photo'].encode()) % 5, True, 'gbr')
    show('3 supervised lambda selector minus SH: leave-one-domain-out (pooled = 5-fold by photo, domains mixed)',
         {d: {k: S[k][d] for k in S} for d in S['ridge']})

    print('\n## 4 SH+ minus FoRIS and minus MFU on COCO / ISIC by GT-area bin: dFN, dFP in % of domain GT pixels; dIoU per episode')
    for h, base in (('S', 'x1013_E1_iu'), ('T', 'w11_t_ablation')):
        B = {r['index']: r for r in ld(base)}
        W = [dict(B[r['index']], **r) for r in ld(f'w23_simple_pipeline_{h}')]
        for d in ('coco', 'isic'):
            L = [e for e in W if e['dataset'] == d]
            total = sum(e['gt'] for e in L)
            for ref in ('foris', 'MFU'):
                cells = []
                for lo, hi in ((0, .03), (.03, .10), (.10, .25), (.25, 1.01)):
                    b = [e for e in L if lo <= e['gt'] / A < hi]
                    fn = lambda a: sum(e['gt'] - e[a][0] for e in b); fp = lambda a: sum(e[a][2] - e[a][0] for e in b)
                    cells.append(f"<{hi:.2f} n{len(b)} FN{100 * (fn('SH+') - fn(ref)) / total:+.1f} FP{100 * (fp('SH+') - fp(ref)) / total:+.1f} "
                                 f"IoU{100 * np.mean([iou(e['SH+']) - iou(e[ref]) for e in b]):+.1f}")
                print(f'{h} {d:4s} - {ref:5s} ' + ' | '.join(cells))


if __name__ == '__main__':
    main()
