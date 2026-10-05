"""Read-only: score the sealed DEV241 run of D against its own native, and split native's errors into the two goals
(delete what is wrong, add what is missed). Query truth is read; nothing here is an inference result.
Writes only into this directory."""
import json, sys, os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
W = Path(__file__).parent; sys.path.insert(0, '/Users/yang/projects/CVPR2027/src')
import numpy as np
from scipy import ndimage as ndi
from ics.experiment import unpack, packet, summarize
RUN, RCG, OUT = W / 'gpu_multilayer_dev241_v1', W / 'rcg241', W; T = np.load(W / 'truth241.npz')
rows = json.loads((RUN / 'manifest.json').read_text()); S = np.ones((3, 3), bool)
arrays, corr, reach, mism = {}, {}, [], []
def comps(mask, other, lo=.1):
    """Pixels of `mask` in connected components whose share inside `other` is below lo, and the rest."""
    lab, n = ndi.label(mask, S)
    if not n: return np.zeros_like(mask), np.zeros_like(mask)
    size = np.bincount(lab.ravel(), minlength=n + 1); hit = np.bincount(lab.ravel(), weights=other.ravel(), minlength=n + 1)
    low = (hit / np.maximum(size, 1) < lo); low[0] = False
    return low[lab], mask & ~low[lab]
for row in rows:
    key = row['key']
    truth, cached = unpack(T['truth/' + key]), unpack(T['native/' + key])
    with np.load(RUN / 'predictions' / (key + '.npz'), allow_pickle=False) as f: m = {k: unpack(f[k]) for k in f.files}
    with np.load(RCG / 'predictions' / (key + '.npz'), allow_pickle=False) as f: m['rcg'] = unpack(f['rcg'])
    nat, D, R = m['native'], m['multilayer'], m['rcg']
    d = int((nat != cached).sum())
    if d: mism.append((key, d))
    # fixed set rules on sealed masks: one operator deletes, another adds (no per-episode choice)
    m['set: delete by D, add by RCG'] = (nat & D) | (R & ~nat)
    m['set: delete by RCG, add by D'] = (nat & R) | (D & ~nat)
    m['set: delete where D and RCG both delete'] = nat & (D | R)
    m['set: delete where D or RCG deletes'] = nat & D & R
    m['set: add where D and RCG both add'] = nat | (D & R)
    m['set: add where D or RCG adds'] = nat | D | R
    m['set: majority of native, D, RCG'] = (nat.astype(int) + D + R) >= 2
    # what each goal is worth (truth)
    wrong, mixed = comps(nat, truth)            # native components with < 10% truth: whole wrong regions
    missed, touched = comps(truth, nat)          # truth components with < 10% found: whole missed objects
    leak = mixed & ~truth; part = touched & ~nat
    m['gt: delete every false pixel'] = nat & truth
    m['gt: delete whole wrong regions'] = nat & ~wrong
    m['gt: delete false pixels attached to the target'] = nat & ~leak
    m['gt: add every missed pixel'] = nat | truth
    m['gt: add whole missed objects'] = nat | missed
    m['gt: complete objects already touched'] = nat | touched
    m['gt: delete wrong regions and add missed objects'] = (nat & ~wrong) | missed
    for arm, x in m.items():
        arrays.setdefault(arm, []).append([int((x & truth).sum()), int((x | truth).sum())])
        add, dele = x & ~nat, nat & ~x
        corr.setdefault(arm, []).append(dict(key=key, c=row['c'], fold=row['fold'], batch=str(row.get('batch', 'unspecified')),
            add_TP=int((add & truth).sum()), delete_FP=int((dele & ~truth).sum()), delete_TP=int((dele & truth).sum()), add_FP=int((add & ~truth).sum())))
    rec = dict(key=key, c=row['c'], fold=row['fold'], batch=str(row.get('batch')), truth=int(truth.sum()), native=int(nat.sum()),
               wrong=int(wrong.sum()), leak=int(leak.sum()), missed=int(missed.sum()), part=int(part.sum()))
    for arm in ('multilayer', 'multilayer.control', 'multilayer.delta.control', 'rcg'):
        x = m[arm]
        rec[arm] = dict(wrong_deleted=int((wrong & ~x).sum()), leak_deleted=int((leak & ~x).sum()), missed_added=int((missed & x).sum()), part_added=int((part & x).sum()))
    reach.append(rec)
arrays = {k: np.asarray(v, np.int64) for k, v in arrays.items()}
rep, draws = summarize(rows, arrays, corr)
tot = {k: sum(r[k] for r in reach) for k in ('truth', 'native', 'wrong', 'leak', 'missed', 'part')}
rep['error_mass'] = tot; rep['native_replay_mismatch'] = mism
rep['reach'] = {a: {k: sum(r[a][k] for r in reach) for k in ('wrong_deleted', 'leak_deleted', 'missed_added', 'part_added')} for a in ('multilayer', 'multilayer.control', 'multilayer.delta.control', 'rcg')}
rep['episodes_with'] = dict(wrong=sum(r['wrong'] > 0 for r in reach), missed=sum(r['missed'] > 0 for r in reach),
                            wrong_over_10pct_of_native=sum(r['wrong'] > .1 * max(r['native'], 1) for r in reach), missed_over_10pct_of_truth=sum(r['missed'] > .1 * max(r['truth'], 1) for r in reach))
(OUT / 'edit_budget.json').write_text(json.dumps(rep, indent=1)); (OUT / 'edit_reach.json').write_text(json.dumps(reach))
print('native replay: %d episodes differ from the cached native, %d pixels in all' % (len(mism), sum(d for _, d in mism)), mism[:6])
print('%-52s %6s %24s %11s | %9s %9s %9s %9s' % ('arm', 'mIoU', 'vs native [95%]', 'up/down/tie', 'add_TP', 'add_FP', 'del_FP', 'del_TP'))
for a in arrays:
    c = rep['contrasts'][a].get('native'); k = rep['corrections_vs_native'][a]
    print('%-52s %6.2f %s | %9d %9d %9d %9d' % (a, rep['scores'][a], '%+6.2f [%+6.2f, %+6.2f] %4d/%d/%d' % (c['gain'], c['ci95'][0], c['ci95'][1], c['up'], c['down'], c['tie']) if c else ' ' * 36, k['add_TP'], k['add_FP'], k['delete_FP'], k['delete_TP']))
for a in ('multilayer',):
    for b in ('multilayer.control', 'multilayer.delta.control', 'rcg'):
        c = rep['contrasts'][a][b]; print('%s minus %s: %+.2f [%+.2f, %+.2f] %d/%d/%d' % (a, b, c['gain'], c['ci95'][0], c['ci95'][1], c['up'], c['down'], c['tie']))
for f in ('folds', 'batchs'):
    print(f, {k: (v['n'], {a: round(v['gain_vs_native'][a], 2) for a in ('multilayer', 'multilayer.control', 'multilayer.delta.control', 'rcg', 'multilayer.delete_only', 'multilayer.add_only')}) for k, v in rep[f].items()})
print('error mass (pixels):', tot, rep['episodes_with']); print('reach:', rep['reach'])
