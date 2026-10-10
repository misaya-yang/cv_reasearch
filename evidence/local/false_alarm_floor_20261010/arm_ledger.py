"""Every sealed arm of the M4 cohorts in (pooled recall, false-alarm area per unit target area). Label diagnostic.

Reads stored per-episode counts only (original-image frame); nothing is encoded, predicted or fitted.
For a mask P and truth G: I = |P & G|, U = g + F with g = |G|, F = |P - G|, so an arm can only move recall or F.
Score convention of each cohort is kept: DeepGlobe total I/U, the others observed class mean inside a fold, fold mean.

Run from the repository root:  python3 evidence/local/false_alarm_floor_20261010/arm_ledger.py
Inputs: parallel_signal_20261010/01_multi_observation/fusion_v1, 02_task_identity/{paco,lvis,coco,deep},
difficult_region_signal_20261010/candidate_reference_v1, parallel_signal_20261010/04_dataset_differences/mask_profiles.jsonl.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

np.seterr(all='ignore')
ROOT = Path(__file__).resolve().parents[1]
SIGNAL = ROOT / 'parallel_signal_20261010'
lines = lambda path: [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def score(I, U, fold, cls, total):
    if total:
        return 100 * I.sum() / U.sum()
    return 100 * np.mean([np.mean([I[(fold == f) & (cls == c)].sum() / max(U[(fold == f) & (cls == c)].sum(), 1)
                                   for c in np.unique(cls[fold == f])]) for f in np.unique(fold)])


def table(title, rows, arms, total, base):
    """rows: dicts with fold, class_id, g and arm -> (I, F)."""
    fold, cls = (np.array([r[k] for r in rows]) for k in ('fold', 'class_id'))
    g = np.array([r['g'] for r in rows], float)
    F0 = sum(r[base][1] for r in rows)
    print(f"\n{title}  n={len(rows)}")
    print('   arm                               score | recall | sumF/sumg (vs FoRIS) | F=0 ceiling | no-miss | F>g  | rho(F,g) rho(r,g) rho(F,r)')
    for a in arms:
        I, F = (np.array([r[a][k] for r in rows], float) for k in (0, 1))
        print(f"   {a:33s} {score(I, g + F, fold, cls, total):5.2f} | {100 * I.sum() / g.sum():5.1f}% | {F.sum() / g.sum():6.3f}  (x{F.sum() / F0:4.2f})    | "
              f"{score(I, g, fold, cls, total):6.2f}      | {score(g, g + F, fold, cls, total):6.2f}  | {100 * (F > g).mean():4.1f}% | "
              f"{spearmanr(F, g)[0]:+.2f}    {spearmanr(I / g, g)[0]:+.2f}    {spearmanr(F, I / g)[0]:+.2f}")


fusion = lines(SIGNAL / '01_multi_observation/fusion_v1/episodes.jsonl')
for group in ('lvis_original600', 'paco_fast600', 'deep_original100', 'deep_fast100'):
    part = [r for r in fusion if r['group'] == group]
    arms = [a for a in ('foris', 'unfiltered', 'geometry', 'region', 'restore.whole', 'restore.accepted') if a in part[0]['confusion']['original']]
    rows = [dict(fold=r['fold'], class_id=r['class_id'], g=r['confusion']['original']['foris']['tp'] + r['confusion']['original']['foris']['fn'],
                 **{a: (r['confusion']['original'][a]['tp'], r['confusion']['original'][a]['fp']) for a in arms}) for r in part]
    table(f"1. nine local windows: {group}", rows, arms, 'deep' in group, 'foris')

identity = {name: lines(SIGNAL / f'02_task_identity/{name}/episodes.jsonl') for name in ('paco', 'lvis', 'coco', 'deep')}
for name, part in identity.items():
    arms = list(part[0]['iu'])
    rows = [dict(fold=r['fold'], class_id=r['class_id'], g=r['gt_pixels'], **{a: (r['iu'][a][0], r['iu'][a][1] - r['gt_pixels']) for a in arms}) for r in part]
    table(f"2. role competition and presence arms: {name}", rows, arms, name == 'deep', 'foris.crf')

truth = {r['episode_id']: r['gt_pixels'] for name in ('paco', 'deep') for r in identity[name]}
candidate = lines(ROOT / 'difficult_region_signal_20261010/candidate_reference_v1/scored_episodes.jsonl')
for dataset in ('paco_part', 'deepglobe_road'):
    part = [r for r in candidate if r['dataset'] == dataset]
    arms = list(part[0]['iu'])
    rows = [dict(fold=r['fold'], class_id=r['class_id'], g=truth[r['episode_id']],
                 **{a: (r['iu'][a][0], r['iu'][a][1] - truth[r['episode_id']]) for a in arms}) for r in part]
    table(f"3. reference ridge and bounded anchor arms: {dataset}", rows, arms, 'deep' in dataset, 'foris.crf')

profile = {}
for p in lines(SIGNAL / '04_dataset_differences/mask_profiles.jsonl'):
    profile.setdefault(p['cohort'], {})[p['episode_id']] = p
print('\n4. What the nine windows change, by target share of the query image (pooled inside a bin)')
for group, cohort in (('lvis_original600', 'LVIS600'), ('paco_fast600', 'PACO600'), ('deep_original100', 'Deep100')):
    part = [r for r in fusion if r['group'] == group]
    get = lambda arm: np.array([[r['confusion']['original'][arm][k] for k in ('tp', 'fp', 'fn')] for r in part], float)
    f, u, w = get('foris'), get('unfiltered'), get('region')
    g = f[:, 0] + f[:, 2]
    share = g / np.array([profile[cohort][r['episode_id']]['query_input_area'] for r in part], float)
    print(f"   {cohort}")
    for a, b, name in ((0, .01, '<1%'), (.01, .05, '1-5%'), (.05, .2, '5-20%'), (.2, 2, '>20%')):
        m = (share >= a) & (share < b)
        if m.sum() < 8:
            continue
        G = g[m].sum()
        print(f"     {name:6s} n={m.sum():3d} | FoRIS recall {100 * f[m, 0].sum() / G:5.1f}% F/g {f[m, 1].sum() / G:6.3f} IoU {100 * f[m, 0].sum() / (G + f[m, 1].sum()):5.1f}"
              f" | all windows recall {100 * u[m, 0].sum() / G:5.1f}% F/g {u[m, 1].sum() / G:6.3f}"
              f" | with acceptance recall {100 * w[m, 0].sum() / G:5.1f}% F/g {w[m, 1].sum() / G:6.3f} IoU {100 * w[m, 0].sum() / (G + w[m, 1].sum()):5.1f}")

print('\n5. Reference geometry against the query (reference mask only on the left side of each pair)')
print('   cohort            n | target mass in tokens covered >= 90%: reference / query median | rho(reference share, query share) | within a factor 2')
for cohort, rows in profile.items():
    rows = list(rows.values())
    a, b = (np.array([r[k]['fg_fraction'] for r in rows]) for k in ('reference', 'query_GT_diagnostic'))
    pure = [np.median([r[k]['fg_mass_ge90'] for r in rows]) for k in ('reference', 'query_GT_diagnostic')]
    ok = (a > 0) & (b > 0)
    print(f"   {cohort:16s} {len(rows):4d} | {pure[0]:.2f} / {pure[1]:.2f} | {spearmanr(a, b)[0]:+.2f} | {100 * (np.abs(np.log2(a[ok] / b[ok])) < 1).mean():.0f}%")
