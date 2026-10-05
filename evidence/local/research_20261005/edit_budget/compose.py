"""A + B + auxiliary, read on sealed masks (truth read; ceilings, not methods). Every arm is an edit of native: its
additions (outside native) and its deletions (inside native) never overlap, so their four pixel counts add.
rho = share of the side effect that an auxiliary removes without touching the benefit:
  additions: I = I0 + add_TP, U = U0 + (1 - rho) * add_FP;  deletions: I = I0 - (1 - rho) * del_TP, U = U0 - del_FP."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/Users/yang/projects/CVPR2027/src')
from ics.experiment import unpack, metric, photo_groups
W = Path(__file__).parent; T = np.load(W / 'truth241.npz')
def load(run, extra=None):
    rows = json.loads((W / run / 'manifest.json').read_text()); out = []
    for r in rows:
        k = r['key']; t = unpack(T['truth/' + k]); nat = unpack(T['native/' + k])
        with np.load(W / run / 'predictions' / (k + '.npz')) as f: m = {a: unpack(f[a]) for a in f.files if a != 'native'}
        if extra:
            with np.load(W / extra / 'predictions' / (k + '.npz')) as f: m['rcg'] = unpack(f['rcg'])
        rec = dict(I0=int((nat & t).sum()), U0=int((nat | t).sum()), FN=int((t & ~nat).sum()), FP=int((nat & ~t).sum()))
        for a, x in m.items():
            add, de = x & ~nat, nat & ~x
            rec[a] = (int((add & t).sum()), int((add & ~t).sum()), int((de & ~t).sum()), int((de & t).sum()))
        rec['_m'] = m; rec['_t'] = t; rec['_n'] = nat; out.append(rec)
    return rows, out
def table(rows, recs, arms, title):
    cls = np.array([r['c'] for r in rows]); g = photo_groups(rows); ng = int(g.max()) + 1
    wts = np.stack([np.bincount(d, minlength=ng) for d in np.random.RandomState(0).randint(ng, size=(2000, ng))])[:, g]
    I0, U0 = (np.array([r[k] for r in recs], float) for k in ('I0', 'U0')); base = np.stack([I0, U0], 1); b = metric(base, cls)
    def gain(I, U, ci=False):
        v = np.stack([I, U], 1); d = metric(v, cls) - b
        if not ci: return '%+6.2f' % d
        s = np.array([metric(v, cls, w) - metric(base, cls, w) for w in wts]); return '%+6.2f [%+5.2f, %+5.2f]' % (d, *np.percentile(s, [2.5, 97.5]))
    FN, FP = sum(r['FN'] for r in recs), sum(r['FP'] for r in recs)
    print('\n== %s: n=%d, native %.2f, missed pixels %.2f M, false pixels %.2f M' % (title, len(rows), b, FN / 1e6, FP / 1e6))
    print('%-34s | as A (adds): recovers | true share | gain now, side effect halved, removed | as B (deletes): removes | false share | gain now, halved, removed' % 'arm')
    C = {}
    for a in arms:
        c = np.array([r[a] for r in recs], float); C[a] = c; aT, aF, dF, dT = c.T
        print('%-34s | %5.1f%% of missed | %5.1f%% | %s %s %s | %5.1f%% of false | %5.1f%% | %s %s %s' % (a, 100 * aT.sum() / FN, 100 * aT.sum() / max(aT.sum() + aF.sum(), 1),
              gain(I0 + aT, U0 + aF), gain(I0 + aT, U0 + .5 * aF), gain(I0 + aT, U0),
              100 * dF.sum() / FP, 100 * dF.sum() / max(dF.sum() + dT.sum(), 1), gain(I0 - dT, U0 - dF), gain(I0 - .5 * dT, U0 - dF), gain(I0, U0 - dF)))
    return C, I0, U0, gain
rows, recs = load('gpu_multilayer_dev241_v1', 'rcg241'); arms = ['rcg', 'multilayer.delta.control', 'multilayer', 'multilayer.control']
C, I0, U0, gain = table(rows, recs, arms, 'DEV241')
# unions over the four arms, with an ideal auxiliary (only true additions kept, only false deletions kept)
uI, uU, aI, aU, dU = [], [], [], [], []
for r in recs:
    m, t, nat = r['_m'], r['_t'], r['_n']; anyadd = np.zeros_like(t); anydel = np.zeros_like(t)
    for a in arms: anyadd |= m[a] & ~nat; anydel |= nat & ~m[a]
    aT, dF = int((anyadd & t).sum()), int((anydel & ~t).sum()); aI.append(aT); dU.append(dF)
I0a, U0a = I0, U0; aI, dU = np.array(aI, float), np.array(dU, float)
FN, FP = sum(r['FN'] for r in recs), sum(r['FP'] for r in recs)
print('union of the four arms, ideal auxiliary: additions recover %.1f%% of missed -> %s; deletions remove %.1f%% of false -> %s; both -> %s' % (100 * aI.sum() / FN, gain(I0 + aI, U0, True), 100 * dU.sum() / FP, gain(I0, U0 - dU, True), gain(I0 + aI, U0 - dU, True)))
A, B = C['multilayer.delta.control'], C['rcg']
for name, (ra, rb) in {'A = transition-only additions, B = RCG deletions, no auxiliary': (0, 0), 'side effects halved': (.5, .5), 'side effects removed': (1, 1), 'only A cleaned': (1, 0), 'only B cleaned': (0, 1)}.items():
    print('%-66s %s' % (name, gain(I0 + A[:, 0] - (1 - rb) * B[:, 3], U0 + (1 - ra) * A[:, 1] - B[:, 2], True)))
print('additivity check (sum of the two single edits vs the arm itself):')
rows5, recs5 = load('gpu_latent_mini50_v1'); table(rows5, recs5, ['rcg', 'latent_native', 'latent_native.control', 'latent_native.two_slot_em.control'], 'mini50')
first = {'%d_%d_%d' % (r['fold'], r['e'], r['c']) for r in json.loads(Path('/Users/yang/projects/CVPR2027/evidence/local/research_20261005/first20.json').read_text())['episodes']}
sel = [r for r, row in zip(recs, rows) if row['key'] in first]; FN20, FP20 = sum(r['FN'] for r in sel), sum(r['FP'] for r in sel)
rep = json.loads((W / 'gpu_second20_v1/report.json').read_text())['corrections_vs_native']
print('\n== first20 (aggregate counts from the sealed report): missed %.2f M, false %.2f M' % (FN20 / 1e6, FP20 / 1e6))
for a, k in rep.items():
    if a == 'native': continue
    print('%-42s adds: recovers %5.1f%% of missed, %5.1f%% true | deletes: removes %5.1f%% of false, %5.1f%% false' % (a, 100 * k['add_TP'] / FN20, 100 * k['add_TP'] / max(k['add_TP'] + k['add_FP'], 1), 100 * k['delete_FP'] / FP20, 100 * k['delete_FP'] / max(k['delete_FP'] + k['delete_TP'], 1)))
