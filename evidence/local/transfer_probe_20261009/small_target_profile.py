"""Why small targets are over-drawn: is there a floor on the mask size, and does the field peak on the target?

Label diagnostic on stored counts (complete FoRIS mask; RCG fine field at levels 0.30-0.70). Areas in 64x64-grid tokens.
Run from evidence/local/research_20261005:  python3 ../transfer_probe_20261009/small_target_profile.py
"""
import json
import numpy as np

SETS = {'LVIS': [f'rcg2_group_lvis_f{i}' for i in range(4)], 'PACO-Part': [f'rcg2_group_paco_part_f{i}' for i in range(4)],
        'COCO-D': ['rcg2_group_groupD'], 'PASCAL-Part': ['rcg2_grouppascal_part'], 'SUIM': ['rcg2_group_suim']}
BINS, NAMES = [(0, .002), (.002, .01), (.01, .05), (.05, .2), (.2, 2)], ['<0.2%', '0.2-1%', '1-5%', '5-20%', '>20%']
TOK = 1024 ** 2 / 4096

for name, dirs in SETS.items():
    z = [np.load(d + '/counts.npz') for d in dirs]
    F, T = (np.concatenate([x[k] for x in z]).astype(float) for k in ('iu:native', 'truth'))
    IF, UF = (np.concatenate([x[k] for x in z]).astype(float) for k in ('I_fine', 'U_fine'))
    lv = z[0]['levels']
    sh, pf = T / 1024 ** 2, (F[:, 1] - T + F[:, 0]) / TOK
    P = (UF - T[:, None] + IF)
    q = lambda x: ' / '.join(f"{v:.0f}" for v in np.percentile(x, [1, 5, 25, 50]))
    print(f"\n== {name} n={len(T)}")
    print(f"  mask size in tokens, percentiles 1 / 5 / 25 / 50:  FoRIS {q(pf)};  RCG at 0.5 {q(P[:, 16] / TOK)};  truth {q(T / TOK)}")
    print(f"  share of episodes with truth below 10 tokens: {100 * (T / TOK < 10).mean():.0f}%;  FoRIS mask below 10 tokens: {100 * (pf < 10).mean():.0f}%")
    small = sh < .01
    if small.sum() > 20:
        b = np.polyfit(np.log(T[small] / TOK), np.log(np.maximum(pf[small], .1)), 1)
        print(f"  targets below 1%: log(FoRIS mask) = {b[0]:.2f} x log(truth) + {b[1]:.2f}   (slope 1 would mean the mask follows the target size)")
    print('  bin       level 0.30: recall>=.5, precision(med), mask/truth(med) |  level 0.50  |  level 0.70  | best level at the 0.70 cap')
    for (a, c), nm in zip(BINS, NAMES):
        m = (sh >= a) & (sh < c)
        if m.sum() < 15:
            continue
        cells = []
        for k in (0, 16, 32):
            rec, pre = IF[m, k] / T[m], IF[m, k] / np.maximum(P[m, k], 1)
            cells.append(f"{100 * (rec >= .5).mean():3.0f}%, {100 * np.median(pre):3.0f}%, {np.median(P[m, k] / T[m]):5.2f}x")
        cap = ((IF[m] / np.maximum(UF[m], 1)).argmax(1) == len(lv) - 1).mean()
        print(f"  {nm:7s}  " + '  |  '.join(cells) + f"  |  {100 * cap:3.0f}%")
