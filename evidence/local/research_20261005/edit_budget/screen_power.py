"""What a 20- or 50-episode screen can resolve: gains of sealed arms on first20, mini50, all 241, and the spread of the
gain over random subsets of DEV241 drawn like mini50 (fold-stratified, without replacement)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/Users/yang/projects/CVPR2027/src')
from ics.experiment import unpack, metric
W = Path(__file__).parent; T = np.load(W / 'truth241.npz'); rows = json.loads((W / 'gpu_multilayer_dev241_v1/manifest.json').read_text())
A = {}
for r in rows:
    k = r['key']; t = unpack(T['truth/' + k])
    with np.load(W / 'gpu_multilayer_dev241_v1/predictions' / (k + '.npz')) as f: m = {a: unpack(f[a]) for a in ('native', 'multilayer', 'multilayer.delta.control')}
    with np.load(W / 'rcg241/predictions' / (k + '.npz')) as f: m['rcg'] = unpack(f['rcg'])
    for a, x in m.items(): A.setdefault(a, []).append([(x & t).sum(), (x | t).sum()])
A = {a: np.asarray(v, np.int64) for a, v in A.items()}; np.savez(W / 'iu241.npz', **A)
cls, fold, keys = np.array([r['c'] for r in rows]), np.array([r['fold'] for r in rows]), [r['key'] for r in rows]
mini = {r['key'] for r in json.loads((W / 'gpu_latent_mini50_v1/manifest.json').read_text())}
first = {'%d_%d_%d' % (r['fold'], r['e'], r['c']) for r in json.loads(Path('/Users/yang/projects/CVPR2027/evidence/local/research_20261005/first20.json').read_text())['episodes']}
gain = lambda a, ix: metric(A[a][ix], cls[ix]) - metric(A['native'][ix], cls[ix])
sel = {'first20': np.array([k in first for k in keys]), 'mini50': np.array([k in mini for k in keys]), 'all 241': np.ones(len(keys), bool)}
sel['241 without mini50'] = ~sel['mini50']
rng = np.random.RandomState(0)
def draw(n):
    ix = np.zeros(len(keys), bool); per = [n // 4 + (i < n % 4) for i in range(4)]
    for f, p in enumerate(per): ix[rng.choice(np.flatnonzero(fold == f), p, replace=False)] = True
    return ix
for a in ('multilayer', 'multilayer.delta.control', 'rcg'):
    print(a, {s: (int(ix.sum()), round(gain(a, ix), 2)) for s, ix in sel.items()})
    for n in (20, 50, 120):
        g = np.array([gain(a, draw(n)) for _ in range(2000)])
        print('   random %3d of 241: gain mean %+.2f, sd %.2f, 2.5%%..97.5%% %+.2f..%+.2f, share of draws >= +2: %.2f, share <= 0: %.2f' % (n, g.mean(), g.std(), *np.percentile(g, [2.5, 97.5]), (g >= 2).mean(), (g <= 0).mean()))
