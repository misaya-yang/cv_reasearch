"""LVIS (and the other four for comparison): complete FoRIS by query-target size. Label diagnostic, stored counts only.

Run from evidence/local/research_20261005:  python3 ../transfer_probe_20261009/lvis_by_size.py
"""
import json
import numpy as np

SETS = {'LVIS': [f'rcg2_group_lvis_f{i}' for i in range(4)], 'PACO-Part': [f'rcg2_group_paco_part_f{i}' for i in range(4)],
        'COCO-D': ['rcg2_group_groupD'], 'PASCAL-Part': ['rcg2_grouppascal_part'], 'SUIM': ['rcg2_group_suim']}
BINS = [(0, .002), (.002, .01), (.01, .05), (.05, .2), (.2, 2)]
NAMES = ['<0.2%', '0.2-1%', '1-5%', '5-20%', '>20%']


def miou(I, U, c):
    return 100 * np.mean([I[c == k].sum() / max(U[c == k].sum(), 1) for k in np.unique(c)])


for name, dirs in SETS.items():
    F, T, C, IF, UF = [], [], [], [], []
    for j, d in enumerate(dirs):
        z, rows = np.load(d + '/counts.npz'), json.load(open(d + '/rows.json'))
        F.append(z['iu:native'].astype(float)); T.append(z['truth'].astype(float))
        IF.append(z['I_fine']); UF.append(z['U_fine'])
        C.append(np.array([f"{j}:{r['c']}" for r in rows]))
    F, T, IF, UF = np.concatenate(F), np.concatenate(T), np.concatenate(IF), np.concatenate(UF)
    c = np.unique(np.concatenate(C), return_inverse=True)[1]
    sh, j = T / 1024 ** 2, F[:, 0] / np.maximum(F[:, 1], 1)
    pred = F[:, 1] - T + F[:, 0]
    best = (IF / np.maximum(UF, 1)).max(1)
    base = miou(F[:, 0], F[:, 1], c)
    print(f"\n== {name} n={len(T)}  FoRIS {base:.2f}")
    print('  bin      n   share  tokens(med)  mean IoU  IoU<0.2  empty  pred/true(med)  >2x  best-level IoU  perfect-bin gain')
    for (a, b), nm in zip(BINS, NAMES):
        m = (sh >= a) & (sh < b)
        if m.sum() < 5:
            continue
        I, U = F[:, 0].copy(), F[:, 1].copy()
        I[m], U[m] = T[m], T[m]
        print(f"  {nm:7s} {m.sum():4d}  {100 * m.mean():4.0f}%  {np.median(sh[m]) * 4096:9.0f}  {100 * j[m].mean():8.1f}  {100 * (j[m] < .2).mean():6.0f}%  "
              f"{100 * (pred[m] == 0).mean():4.0f}%  {np.median(pred[m] / T[m]):12.2f}  {100 * (pred[m] > 2 * T[m]).mean():3.0f}%  {100 * best[m].mean():11.1f}  {miou(I, U, c) - base:+14.2f}")

    # is the target inside the FoRIS mask (too big, zoom into it) or elsewhere (identity miss)?
    rec, k30 = F[:, 0] / np.maximum(T, 1), 0
    rec30 = IF[:, k30] / np.maximum(T, 1)
    print('  bin      recall>=0.5 (FoRIS mask)  recall>=0.5 and IoU<0.5  recall<0.2  recall>=0.5 at field level 0.30   episode-mean IoU if recall>=0.5 cases were perfect')
    for (a, b), nm in zip(BINS, NAMES):
        m = (sh >= a) & (sh < b)
        if m.sum() < 5:
            continue
        print(f"  {nm:7s} {100 * (rec[m] >= .5).mean():10.0f}%  {100 * ((rec[m] >= .5) & (j[m] < .5)).mean():22.0f}%  {100 * (rec[m] < .2).mean():14.0f}%  {100 * (rec30[m] >= .5).mean():16.0f}%")
