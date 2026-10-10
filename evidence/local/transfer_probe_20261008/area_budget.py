"""Label diagnostics from the stored per-episode counts (1024 fine RCG field, levels 0.30-0.70). No encoder, no GPU.

Run from evidence/local/research_20261005:  python3 ../transfer_probe_20261008/area_budget.py
P(level) = U + I - T is the predicted area at a level. Class mIoU pools I and U per class, then averages classes.
"""
import json
import numpy as np

SETS = {'COCO-D': ['rcg2_group_groupD'], 'PASCAL-Part': ['rcg2_grouppascal_part'], 'SUIM': ['rcg2_group_suim'],
        'PACO-Part': [f'rcg2_group_paco_part_f{i}' for i in range(4)], 'LVIS': [f'rcg2_group_lvis_f{i}' for i in range(4)]}
BINS = [(0, .01), (.01, .05), (.05, .2), (.2, 2)]
rng = np.random.RandomState(0)


def miou(I, U, c):
    return 100 * np.mean([I[c == k].sum() / max(U[c == k].sum(), 1) for k in np.unique(c)])


def load(dirs):
    cols = [[] for _ in range(4)]
    for j, d in enumerate(dirs):
        z, rows = np.load(d + '/counts.npz'), json.load(open(d + '/rows.json'))
        for col, v in zip(cols, (z['I_fine'], z['U_fine'], z['truth'], np.array([f"{j}:{r['c']}" for r in rows]))):
            col.append(v)
    IF, UF, T, C = (np.concatenate(v) for v in cols)
    return IF, UF, T.astype(float), np.unique(C, return_inverse=True)[1], np.load(dirs[0] + '/counts.npz')['levels']


for name, dirs in SETS.items():
    IF, UF, T, c, lv = load(dirs)
    n, ar, k5 = len(T), np.arange(len(T)), int(np.argmin(abs(lv - .5)))
    pick = lambda idx: miou(IF[ar, idx], UF[ar, idx], c)
    fixed, base = np.full(n, k5), None
    base = pick(fixed)
    iou, sh, P = IF / np.maximum(UF, 1), T / 1024 ** 2, UF - T[:, None] + IF
    found, miss = iou.max(1) >= .5, iou.max(1) < .2
    print(f"\n== {name}  n={n}  fixed 0.5: {base:.2f}  best level per episode: {pick(iou.argmax(1)) - base:+.2f}")

    # 1. misses by target size, and the value of emptying the mask on missed episodes
    row = ' / '.join(f"{100 * miss[(sh >= a) & (sh < b)].mean():.1f}% of {((sh >= a) & (sh < b)).sum()}" for a, b in BINS)
    I, U = IF[ar, fixed].copy(), UF[ar, fixed].copy()
    over = (P[ar, fixed][miss] / T[miss])
    I[miss], U[miss] = 0, T[miss]
    print(f"  missed episodes by target share <1% / 1-5% / 5-20% / >20%: {row}")
    print(f"  empty mask on missed episodes: {miou(I, U, c) - base:+.2f}   (their predicted/true area, median: {np.median(over):.1f}x)")

    # 2. best pooled level by true-size bin (found episodes)
    row = ' / '.join(f"{lv[(IF[g].sum(0) / np.maximum(UF[g].sum(0), 1)).argmax()]:.2f}" if g.sum() > 5 else '-'
                     for g in (((sh >= a) & (sh < b) & found) for a, b in BINS))
    print(f"  best pooled level by target share, found episodes: {row}")

    # 3. how much knowing the area is worth, and how exact it has to be
    match = lambda target: np.abs(np.log(np.maximum(P, 1)) - np.log(np.maximum(target, 1))[:, None]).argmin(1)
    noisy = [np.mean([pick(match(T * np.exp(s * rng.randn(n)))) - base for _ in range(20)]) for s in (.3, .5, .7, 1.)]
    print(f"  level whose area equals the true area: {pick(match(T)) - base:+.2f};  with log-normal error sd 0.3 / 0.5 / 0.7 / 1.0: "
          + ' / '.join(f"{x:+.2f}" for x in noisy))

    # 4. the fixed cut's own area error, and which tail holds the value
    e = np.log(np.maximum(P[ar, fixed], 1) / T)
    hi, lo = e > np.log(2), e < -np.log(2)
    tail = lambda g: pick(np.where(g, match(T), fixed)) - base
    print(f"  log(area at 0.5 / true area): median {np.median(e):+.2f}, robust sd {1.4826 * np.median(abs(e - np.median(e))):.2f}; "
          f"over 2x too big {100 * hi.mean():.0f}% of episodes, over 2x too small {100 * lo.mean():.0f}%")
    print(f"  true-area level only where too big: {tail(hi):+.2f};  only where too small: {tail(lo):+.2f};  only the rest: {tail(~hi & ~lo):+.2f}")

    # 5. spread of the target's share inside a class (a second draw of the class is what a reference would give)
    res = [np.log(sh[c == k]) - np.log(sh[c == k]).mean() for k in np.unique(c) if (c == k).sum() >= 3]
    if res:
        m = np.mean([len(r) for r in res])
        print(f"  within-class sd of log target share: {np.concatenate(res).std() / np.sqrt(1 - 1 / m):.2f} "
              f"({len(res)} classes with >= 3 episodes, small-sample corrected)")

try:
    from scipy.special import i1, k0, k1
    print('\n== screened-Poisson model of the graph solve on an ideal disk, rho = radius / operator width')
    print('  level that lands on the true edge: ' + ', '.join(f"rho {r}: {r * i1(r) * k0(r) / (1 - r * k1(r)):.2f}" for r in (.5, 1, 1.5, 2, 3, 4, 8)))
except ImportError:
    pass


def rank_corr(a, b):
    r = lambda v: np.argsort(np.argsort(v)).astype(float)
    return np.corrcoef(r(a), r(b))[0, 1]


print('\n== found episodes: rank correlation of the best level with field statistics (background median, inside median, mask share) and true share')
for name, dirs in SETS.items():
    IF, UF, T, c, lv = load(dirs)
    F = np.concatenate([np.load(d + '/counts.npz')['fields'].astype(np.float32) for d in dirs]).reshape(len(T), -1)
    iou, m = IF / np.maximum(UF, 1), F > .5
    ok = (iou.max(1) >= .5) & m.any(1) & (~m).any(1)
    best = lv[iou.argmax(1)]
    bg = np.array([np.median(f[~mm]) if (~mm).any() else 0 for f, mm in zip(F, m)])
    ins = np.array([np.median(f[mm]) if mm.any() else 0 for f, mm in zip(F, m)])
    print(f"  {name:11s} " + ' '.join(f"{rank_corr(best[ok], v[ok]):+.2f}" for v in (bg, ins, m.mean(1), T)))

print('\n== simulation: an independent size estimate used one way only. Shrink to the estimated area when the 0.5 mask exceeds k times the estimate.')
print('   estimate = true area x log-normal error, sd 0.7 / 1.0 / 1.4; gain over fixed 0.5, mean of 20 draws')
for name, dirs in SETS.items():
    IF, UF, T, c, lv = load(dirs)
    n, ar = len(T), np.arange(len(T))
    fixed, P = np.full(n, int(np.argmin(abs(lv - .5)))), UF - T[:, None] + IF
    pick = lambda idx: miou(IF[ar, idx], UF[ar, idx], c)
    match = lambda target: np.abs(np.log(np.maximum(P, 1)) - np.log(np.maximum(target, 1))[:, None]).argmin(1)
    cells = []
    for k in (2, 4):
        gains = []
        for s in (.7, 1., 1.4):
            est = [T * np.exp(s * rng.randn(n)) for _ in range(20)]
            gains.append(np.mean([pick(np.where(P[ar, fixed] > k * e, match(e), fixed)) for e in est]) - pick(fixed))
        cells.append(f"k={k}: " + ' / '.join(f"{g:+.2f}" for g in gains))
    print(f"  {name:11s} " + '   '.join(cells))


def auc(score, label):
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = label.sum(), (~label).sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


print('\n== same-class resampling: the size estimate is the true share of ANOTHER episode of the same class (what a reference of that class would give).')
print('   Unlike the log-normal simulation above this keeps the link between a target being unusually small and being over-covered. Mean of 50 draws.')
for name, dirs in SETS.items():
    IF, UF, T, c, lv = load(dirs)
    n, ar = len(T), np.arange(len(T))
    fixed, P = np.full(n, int(np.argmin(abs(lv - .5)))), UF - T[:, None] + IF
    pick = lambda idx: miou(IF[ar, idx], UF[ar, idx], c)
    match = lambda target: np.abs(np.log(np.maximum(P, 1)) - np.log(np.maximum(target, 1))[:, None]).argmin(1)
    base, iou = pick(fixed), IF / np.maximum(UF, 1)
    members = {k: np.flatnonzero(c == k) for k in np.unique(c)}
    has = np.array([len(members[k]) > 1 for k in c])
    e0 = np.log(np.maximum(P[ar, fixed], 1) / T)
    out = {key: [] for key in ('sd', 'two', 'k2', 'k4', 'auc_big', 'auc_miss', 'fire4', 'pure4')}
    for _ in range(50):
        est = T.copy()
        for i in np.flatnonzero(has):
            m = members[c[i]]
            est[i] = T[rng.choice(m[m != i])]
        ratio = np.log(np.maximum(P[ar, fixed], 1) / est)
        out['sd'].append(np.log(est[has] / T[has]).std())
        out['two'].append(pick(np.where(has, match(est), fixed)) - base)
        for k in (2, 4):
            out[f'k{k}'].append(pick(np.where(has & (ratio > np.log(k)), match(est), fixed)) - base)
        out['auc_big'].append(auc(ratio[has], (e0 > np.log(2))[has]))
        out['auc_miss'].append(auc(ratio[has], (iou.max(1) < .2)[has]))
        fire = has & (ratio > np.log(4))
        out['fire4'].append(fire.mean())
        out['pure4'].append((e0 > np.log(2))[fire].mean() if fire.any() else np.nan)
    f = {k: np.nanmean(v) for k, v in out.items()}
    print(f"  {name:11s} episodes with a same-class partner {100 * has.mean():.0f}%; sd of log(estimate/true) {f['sd']:.2f}; "
          f"area matched to estimate {f['two']:+.2f}; one-way k=2 {f['k2']:+.2f}, k=4 {f['k4']:+.2f} "
          f"(k=4 fires on {100 * f['fire4']:.0f}% of episodes, {100 * f['pure4']:.0f}% of them truly over 2x too big); "
          f"AUC of mask/estimate for 'over 2x too big' {f['auc_big']:.2f}, for 'missed' {f['auc_miss']:.2f}")
