"""The false-alarm floor: complete FoRIS in (target area g, recall r, false-alarm area F) coordinates. Label diagnostic.

Stored per-episode counts only; nothing is encoded, predicted or fitted. For a mask P and truth G in one frame:
I = |P & G|, U = |P | G| = g + F exactly, with g = |G| and F = |P - G|. Class mIoU pools I and U inside a class, so
IoU_c = sum(I) / sum(g + F): recall and false-alarm area are the only two things a mask can change.

Run from the repository root:  python3 evidence/local/false_alarm_floor_20261010/floor_law.py
Inputs: evidence/local/research_20261005/rcg2_group* (five self-drawn packs of about 600, complete FoRIS with CRF and the
RCG fine field, 1024 frame), research_20261005/cut_levels4000 (COCO public 4000, FoRIS score without CRF),
evidence_bench_step3_20261007/B_holdout_actual600_v1 (COCO fresh600, 64 x 64 token truth and FoRIS response).
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

np.seterr(all='ignore')
ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'research_20261005'
N = 1024.0 ** 2
SETS = {'COCO': ['rcg2_groupB', 'rcg2_groupC', 'rcg2_group_groupD'], 'PASCAL-Part': ['rcg2_grouppascal_part'],
        'SUIM': ['rcg2_group_suim'], 'PACO-Part': [f'rcg2_group_paco_part_f{i}' for i in range(4)],
        'LVIS': [f'rcg2_group_lvis_f{i}' for i in range(4)]}
BINS = [(0, .002), (.002, .01), (.01, .05), (.05, .2), (.2, 2)]
NAMES = ['<0.2%', '0.2-1%', '1-5%', '5-20%', '>20%']
COCO80 = ('person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter bench '
          'bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard '
          'sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon '
          'bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table '
          'toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors '
          'teddy_bear hair_drier toothbrush').split()


def miou(I, U, c):
    """The ledger's convention for these packs: pooled I/U per (pack, class), mean over classes."""
    return 100 * np.mean([I[c == k].sum() / max(U[c == k].sum(), 1) for k in np.unique(c)])


def load(dirs):
    out = {k: [] for k in ('g', 'native', 'rcg', 'rcg_fine', 'rcg2', 'IF', 'UF', 'c')}
    for j, d in enumerate(dirs):
        z, rows = np.load(PACK / d / 'counts.npz'), json.load(open(PACK / d / 'rows.json'))
        out['g'].append(z['truth']); out['IF'].append(z['I_fine']); out['UF'].append(z['U_fine'])
        for a in ('native', 'rcg', 'rcg_fine', 'rcg2'):
            out[a].append(z['iu:' + a])
        out['c'].append(np.array([f"{j}:{r['c']}" for r in rows]))
        levels = z['levels']
    out = {k: np.concatenate(v) for k, v in out.items()}
    out['c'] = np.unique(out['c'], return_inverse=True)[1]
    return {k: (v.astype(float) if k != 'c' else v) for k, v in out.items()}, levels


def boot(fn, n, draws=2000, seed=0):
    rs = np.random.RandomState(seed)
    v = np.array([fn(rs.randint(0, n, n)) for _ in range(draws)])
    return np.nanquantile(v, [.025, .975])


D = {name: load(dirs) for name, dirs in SETS.items()}

print('1. Independence of recall, false-alarm area and target area (complete FoRIS, 1024 frame; episodes)')
print('   set          n   FoRIS | rho(F,g) [95%]            rho(r,g) [95%]            rho(F,r) | slope log F on log g | F %image mean / median')
for name, (d, _) in D.items():
    g, I, U = d['g'], d['native'][:, 0], d['native'][:, 1]
    F, r, n = U - g, I / g, len(g)
    a = boot(lambda i: spearmanr(F[i], g[i])[0], n); b = boot(lambda i: spearmanr(r[i], g[i])[0], n)
    print(f"   {name:11s} {n:4d}  {miou(I, U, d['c']):5.2f} | {spearmanr(F, g)[0]:+.2f} [{a[0]:+.2f},{a[1]:+.2f}]"
          f"      {spearmanr(r, g)[0]:+.2f} [{b[0]:+.2f},{b[1]:+.2f}]      {spearmanr(F, r)[0]:+.2f}    | "
          f"{np.polyfit(np.log(g), np.log(F + 256), 1)[0]:+.2f}                 | {100 * F.mean() / N:5.2f} / {100 * np.median(F) / N:4.2f}")

print('\n2. By target share of the image: recall and false-alarm area do not follow the target')
for name, (d, _) in D.items():
    g, I, U = d['g'], d['native'][:, 0], d['native'][:, 1]
    F, r, sh = U - g, I / g, g / N
    print(f"   {name}")
    print('     bin      n    mean IoU  recall mean/median  F %image mean/median  F tokens(median)  pred/true(median)')
    for (a, b), nm in zip(BINS, NAMES):
        m = (sh >= a) & (sh < b)
        if m.sum() < 8:
            continue
        print(f"     {nm:7s} {m.sum():4d}  {100 * (I / U)[m].mean():7.1f}   {r[m].mean():.3f} / {np.median(r[m]):.3f}      "
              f"{100 * F[m].mean() / N:5.2f} / {100 * np.median(F[m]) / N:4.2f}       {np.median(F[m]) / 256:9.1f}      {np.median((I[m] + F[m]) / g[m]):9.2f}")

print('\n3. Class mIoU if one side were removed (label diagnostic, not an attainable gain)')
print('   set          FoRIS |  F=0    FN=0 |  F/2    F/4  | F=0 only where F>g  (episodes with F>g) | top 10% episodes hold of sum F')
for name, (d, _) in D.items():
    g, I, U, c = d['g'], d['native'][:, 0], d['native'][:, 1], d['c']
    F = U - g
    lo = boot(lambda i: miou(I[i], g[i], c[i]), len(g), 500)
    top = np.sort(F)[::-1][:max(1, len(F) // 10)].sum() / F.sum()
    print(f"   {name:11s}  {miou(I, U, c):5.2f} | {miou(I, g, c):5.2f} [{lo[0]:.1f},{lo[1]:.1f}]  {miou(g, U, c):5.2f} | {miou(I, g + F / 2, c):5.2f}  "
          f"{miou(I, g + F / 4, c):5.2f}  | {miou(I, g + np.where(F > g, 0, F), c):5.2f}               ({100 * (F > g).mean():4.1f}%)          | {100 * top:4.1f}%")

print('\n4. Every stored arm in the same coordinates: pooled recall, pooled false-alarm area per unit target area')
for name, (d, _) in D.items():
    g, c = d['g'], d['c']
    print(f"   {name}")
    for arm in ('native', 'rcg', 'rcg_fine', 'rcg2'):
        I, U = d[arm][:, 0], d[arm][:, 1]
        F = U - g
        print(f"     {arm:9s} mIoU {miou(I, U, c):5.2f} | recall {100 * I.sum() / g.sum():4.1f}% | sum F / sum g {F.sum() / g.sum():.3f} | "
              f"F=0 ceiling {miou(I, g, c):5.2f} | episodes F>g {100 * (F > g).mean():4.1f}%")

print('\n5. Independence model: IoU_i = r_j g_i / (g_i + F_j), (r_j, F_j) drawn from the same set regardless of g_i (episode mean IoU)')
rs = np.random.RandomState(0)
print('   set           actual  model | per bin actual/model')
for name, (d, _) in D.items():
    g, I, U = d['g'], d['native'][:, 0], d['native'][:, 1]
    F, r, n = U - g, I / g, len(g)
    M = np.mean([(lambda j: r[j] * g / (g + np.minimum(F[j], N - g)))(rs.randint(0, n, n)) for _ in range(300)], 0)
    s = f"   {name:11s}   {100 * (I / U).mean():5.1f}  {100 * M.mean():5.1f} |"
    for (a, b), nm in zip(BINS, NAMES):
        m = (g / N >= a) & (g / N < b)
        if m.sum() >= 8:
            s += f"  {nm} {100 * (I / U)[m].mean():4.1f}/{100 * M[m].mean():4.1f}"
    print(s)
print('   transplant: targets of the row set meet the (recall, false alarm) pairs of the column set')
print('   ' + ' ' * 12 + ''.join(f"{k:>12s}" for k in D))
for a, (da, _) in D.items():
    g, row = da['g'], f"   {a:12s}"
    for b, (db, _) in D.items():
        Fb, rb = db['native'][:, 1] - db['g'], db['native'][:, 0] / db['g']
        row += f"{100 * np.mean([(lambda j: (rb[j] * g / (g + np.minimum(Fb[j], N - g))).mean())(rs.randint(0, len(Fb), len(g))) for _ in range(200)]):12.1f}"
    print(row)

print('\n6. What one level for all episodes can trade on the stored RCG fine field (levels 0.30-0.70)')
for name, (d, lv) in D.items():
    g, I, U, c = d['g'], d['IF'], d['UF'], d['c']
    F = U - g[:, None]
    k5 = int(np.argmin(abs(lv - .5)))
    half = np.where(F.sum(0) <= F[:, k5].sum() / 2)[0]
    b = (I / np.maximum(U, 1)).argmax(1); e = np.arange(len(g))
    s = f"   {name:11s} at 0.50: mIoU {miou(I[:, k5], U[:, k5], c):5.2f}, recall {100 * I[:, k5].sum() / g.sum():4.1f}%, sum F / sum g {F[:, k5].sum() / g.sum():.3f}"
    if len(half):
        k = half[0]
        s += f" | sum F halves at {lv[k]:.3f}: recall {100 * I[:, k].sum() / g.sum():4.1f}%, mIoU {miou(I[:, k], U[:, k], c):5.2f}"
    s += f" | same recall with half the false alarms would read {miou(I[:, k5], g + F[:, k5] / 2, c):5.2f}"
    s += f" | best level per episode {miou(I[e, b], U[e, b], c):5.2f} (recall {100 * I[e, b].sum() / g.sum():4.1f}%, sum F / sum g {(U[e, b] - g).sum() / g.sum():.3f})"
    print(s)

print('\n7. COCO public 4000, FoRIS score without CRF at 0.5: which classes are hard')
z = np.load(PACK / 'cut_levels4000/counts.npz'); rows = json.load(open(PACK / 'cut_levels4000/rows.json'))
c = np.array([r['c'] for r in rows]); k = int(np.argmin(abs(z['levels'] - .5)))
g, I, U = z['truth'].astype(float), z['I_foris'][:, k].astype(float), z['U_foris'][:, k].astype(float)
F, r = U - g, I / g
per = [(COCO80[i], I[c == i].sum() / U[c == i].sum(), np.median(g[c == i]) / N, np.median(r[c == i]), np.median(F[c == i]) / N, F[c == i].mean() / N) for i in np.unique(c)]
J, G_, R_, Fm = (np.array([p[i] for p in per]) for i in (1, 2, 3, 4))
print(f"   over 80 classes: rho(class IoU, median F) {spearmanr(J, Fm)[0]:+.2f}, rho(class IoU, median g) {spearmanr(J, G_)[0]:+.2f}, "
      f"rho(class IoU, median recall) {spearmanr(J, R_)[0]:+.2f}, rho(median F, median g) {spearmanr(Fm, G_)[0]:+.2f}")
w = [spearmanr(F[c == i], g[c == i])[0] for i in np.unique(c)]
print(f"   inside a class: mean rho(F,g) {np.mean(w):+.2f}, rho(r,g) {np.mean([spearmanr(r[c == i], g[c == i])[0] for i in np.unique(c)]):+.2f}; "
      f"all 4000: rho(F,g) {spearmanr(F, g)[0]:+.2f}, class mIoU {100 * J.mean():.2f}, F=0 ceiling {100 * np.mean([I[c == i].sum() / g[c == i].sum() for i in np.unique(c)]):.2f}")
per.sort(key=lambda p: -p[4])
for tag, part in (('largest', per[:8]), ('smallest', per[-8:])):
    print(f"   {tag} median false alarm: " + '; '.join(f"{p[0]} IoU {100 * p[1]:.0f} g {100 * p[2]:.1f}% F {100 * p[4]:.2f}%" for p in part))

print('\n8. COCO fresh600 on the 64 x 64 grid, FoRIS response min-max > 0.5: where the false alarms are')
z = np.load(ROOT / 'evidence_bench_step3_20261007/B_holdout_actual600_v1/tokens.npz')
rows = json.load(open(ROOT / 'evidence_bench_step3_20261007/B_holdout_actual600_v1/rows.json'))
cls = np.unique([f"{r['fold']}:{r['c']}" for r in rows], return_inverse=True)[1]
G = z['truth'].astype(np.float32).reshape(-1, 64, 64) > .5
S = z['score'].astype(np.float32).reshape(-1, 64, 64)
S = (S - S.min((1, 2), keepdims=True)) / (S.max((1, 2), keepdims=True) - S.min((1, 2), keepdims=True))
P, n, eight = S > .5, len(S), np.ones((3, 3))
dist = np.stack([ndi.distance_transform_edt(~G[i]) if G[i].any() else np.full((64, 64), 99.) for i in range(n)])
read = lambda M: miou((M & G).sum((1, 2)).astype(float), (M | G).sum((1, 2)).astype(float), cls)
fp = P & ~G
near, far = fp & (dist <= 1.5), fp & (dist > 4.5)
attached = np.zeros(n); peak = np.zeros(n); skirt_true = np.zeros(n)
for i in range(n):
    lab, _ = ndi.label(P[i], structure=eight)
    hit = np.unique(lab[P[i] & G[i]]); attached[i] = (np.isin(lab, hit[hit > 0]) & fp[i]).sum()
    high, m = ndi.label(S[i] > .75, structure=eight)
    wrong = [h for h in range(1, m + 1) if G[i][high == h].mean() < .5]
    right = np.unique(lab[np.isin(high, [h for h in range(1, m + 1) if h not in wrong])])
    inpeak = np.isin(high, wrong) & fp[i]
    peak[i] = inpeak.sum(); skirt_true[i] = (fp[i] & ~inpeak & np.isin(lab, right[right > 0])).sum()
tot = fp.sum()
top = S.reshape(n, -1).argmax(1)
print(f"   class mIoU {read(P):.2f}; without any false alarm {read(P & G):.2f}; without any miss {read(P | G):.2f}; "
      f"without false alarms beyond 4.5 tokens of the truth {read(P & ~far):.2f}; without those within 1.5 tokens {read(P & ~near):.2f}")
print(f"   false-alarm mass: within 1.5 tokens of the truth {100 * near.sum() / tot:.1f}%, beyond 4.5 tokens {100 * far.sum() / tot:.1f}%, "
      f"in a predicted component that touches the truth {100 * attached.sum() / tot:.1f}%")
print(f"   by score level: inside a wrong peak (> 0.75, mostly not truth) {100 * peak.sum() / tot:.1f}%, low skirt of a component with a right peak "
      f"{100 * skirt_true.sum() / tot:.1f}%, low region with no right peak {100 * (tot - peak.sum() - skirt_true.sum()) / tot:.1f}%")
print(f"   rho(false-alarm tokens, truth tokens) {spearmanr(fp.sum((1, 2)), G.sum((1, 2)))[0]:+.2f}; the highest token lies on the truth in "
      f"{100 * G.reshape(n, -1)[np.arange(n), top].mean():.1f}% of episodes")
