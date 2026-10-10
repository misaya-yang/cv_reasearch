"""Is the over-drawing of small targets a fixed dilation (a point-spread length of the pipeline)?

Label diagnostic on stored counts, complete FoRIS masks and RCG fine field at 0.5. Areas in 64x64-grid tokens.
Equivalent radius r = sqrt(area / pi); excess radius d = r_mask - r_truth.
Model (exponential edge, one length L):  r_mask = R + L * ln(1 + exp(-R / L)).
Run from evidence/local/research_20261005:  python3 ../transfer_probe_20261009/dilation_law.py
"""
import numpy as np

SETS = {'COCO-D': ['rcg2_group_groupD'], 'PASCAL-Part': ['rcg2_grouppascal_part'], 'SUIM': ['rcg2_group_suim'],
        'PACO-Part': [f'rcg2_group_paco_part_f{i}' for i in range(4)], 'LVIS': [f'rcg2_group_lvis_f{i}' for i in range(4)]}
EDGES = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 4097]
TOK = 1024 ** 2 / 4096
model = lambda R, L: R + L * np.log1p(np.exp(-R / L))


def fit(R, r):
    grid = np.arange(.25, 12.01, .05)
    err = [np.median(np.abs(np.log(model(R, L) / r))) for L in grid]
    return grid[int(np.argmin(err))], min(err)


pool = {'foris': [[], []], 'rcg': [[], []]}
for name, dirs in SETS.items():
    z = [np.load(d + '/counts.npz') for d in dirs]
    F, T = (np.concatenate([x[k] for x in z]).astype(float) for k in ('iu:native', 'truth'))
    IF, UF = (np.concatenate([x[k] for x in z]).astype(float) for k in ('I_fine', 'U_fine'))
    t = T / TOK
    R = np.sqrt(t / np.pi)
    arms = {'foris': ((F[:, 1] - T + F[:, 0]) / TOK, F[:, 0] / T), 'rcg': ((UF[:, 16] - T + IF[:, 16]) / TOK, IF[:, 16] / T)}
    print(f"\n== {name} n={len(T)}")
    for arm, (p, rec) in arms.items():
        ok = (rec >= .5) & (p > 0)
        r = np.sqrt(p / np.pi)
        L, e = fit(R[ok & (t < 640)], r[ok & (t < 640)])
        pool[arm][0].append(R[ok & (t < 640)]); pool[arm][1].append(r[ok & (t < 640)])
        print(f"  {arm:5s}: episodes with recall>=0.5: {100 * ok.mean():.0f}%;  fitted length L = {L:.2f} tokens (median abs log error of the radius {e:.3f})")
        if arm == 'foris':
            print('    truth tokens   n   excess radius d, quartiles 25/50/75   mask/truth median   model at fitted L   | all episodes: mask tokens 10/50/90 percentile')
            for a, b in zip(EDGES[:-1], EDGES[1:]):
                m = (t >= a) & (t < b)
                if (m & ok).sum() < 8:
                    continue
                d = (r - R)[m & ok]
                q = np.percentile(d, [25, 50, 75])
                pm = np.percentile(p[m], [10, 50, 90])
                Rm = np.median(R[m & ok])
                print(f"    {a:5d}-{b:<5d} {(m & ok).sum():4d}   {q[0]:5.2f} / {q[1]:5.2f} / {q[2]:5.2f}      {np.median(p[m & ok] / t[m & ok]):8.2f}        {(model(Rm, L) / Rm) ** 2:8.2f}          | {pm[0]:6.0f} / {pm[1]:6.0f} / {pm[2]:6.0f}")
print()
for arm, (Rs, rs) in pool.items():
    L, e = fit(np.concatenate(Rs), np.concatenate(rs))
    print(f"all five datasets pooled, {arm}: L = {L:.2f} tokens (median abs log error {e:.3f})")
