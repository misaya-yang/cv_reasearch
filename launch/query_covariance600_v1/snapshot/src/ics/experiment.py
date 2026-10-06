"""Label-isolated cache inference and exact RandomState(0) photo-group scoring."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_rows(path):
    path = Path(path)
    if path.suffix == '.csv':
        with path.open() as f:
            rows = list(csv.DictReader(f))
    else:
        d = json.loads(path.read_text())
        rows = d if isinstance(d, list) else d['episodes']
    seen, identities, out = set(), set(), []
    for row in rows:
        r = dict(row)
        for k in ('fold', 'e', 'c'):
            r[k] = int(r[k])
        r['key'] = f"{r['fold']}_{r['e']}_{r['c']}"
        identity = (r['c'], Path(r['support']).name, Path(r['query']).name)
        if r['key'] in seen or identity in identities:
            raise ValueError(f"Duplicate episode: {r['key']}")
        if identity[1] == identity[2]:
            raise ValueError('Reference and query are the same photo')
        seen.add(r['key']); identities.add(identity); out.append(r)
    if not out:
        raise ValueError('Empty cohort')
    return out


def photo_groups(rows):
    parent = list(range(len(rows))); seen = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, r in enumerate(rows):
        for role in ('support', 'query'):
            p = Path(r[role]).name
            if p in seen:
                a, b = find(i), find(seen[p]); parent[max(a,b)] = min(a,b)
            seen[p] = i
    roots = {}; groups = []
    for i in range(len(rows)):
        root = find(i); roots.setdefault(root, len(roots)); groups.append(roots[root])
    return np.array(groups)


def packet(root, row):
    return Path(root) / row.get('packet_export', f"results/extent_v1/run/packets/{row['key']}.npz")


def load_inputs(root, row):
    import torch
    root = Path(root)
    fp = root / row.get('feature_export', f"cache/evidence_v1/feat/{row['key']}.pt")
    if not fp.exists() and fp.suffix == '.pt':
        fp = fp.with_suffix('.npz')
    if fp.suffix == '.npz':
        with np.load(fp, allow_pickle=False) as f:
            q, r = f['q'].copy(), f['r'].copy()
    else:
        f = torch.load(fp, map_location='cpu', weights_only=True)
        q, r = f['q'], f['r']
    with np.load(packet(root, row), allow_pickle=False) as p:
        cov, score = p['cov'].copy(), p['score'].copy()
    # Do not load query truth, native or pre in this function.
    return (q, r, cov, score), {'features': str(fp), 'feature_sha256': sha(fp),
                               'packet_sha256': sha(packet(root,row))}


def render(field):
    import torch
    import torch.nn.functional as F
    a = np.asarray(field, dtype=np.float32)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Field must be finite HxW')
    return F.interpolate(torch.from_numpy(a.copy())[None,None], (1024,1024),
                         mode='bilinear', align_corners=False)[0,0].numpy() > .5


def unpack(a):
    if a.shape != (131072,) or a.dtype != np.uint8:
        raise ValueError('Expected packed 1024 mask')
    return np.unpackbits(a).reshape(1024,1024).astype(bool)


def metric(iu, classes, weights=None):
    ids, cls = np.unique(classes, return_inverse=True)
    w = np.ones(len(cls)) if weights is None else weights
    i = np.bincount(cls, weights=w*iu[:,0], minlength=len(ids))
    u = np.bincount(cls, weights=w*iu[:,1], minlength=len(ids))
    present = np.bincount(cls, weights=w, minlength=len(ids)) > 0
    return float(100 * np.mean((i / np.maximum(u,1))[present]))


def summarize(rows, arrays, corrections):
    cls = np.array([r['c'] for r in rows]); groups = photo_groups(rows)
    g = int(groups.max())+1
    draws = np.random.RandomState(0).randint(g, size=(2000,g))
    weights = np.stack([np.bincount(d,minlength=g) for d in draws])[:,groups]
    scores = {a: metric(v,cls) for a,v in arrays.items()}
    samples = {a: np.array([metric(v,cls,w) for w in weights]) for a,v in arrays.items()}
    result = dict(n=len(rows), classes=len(set(cls)), photo_groups=g,
                  largest_photo_group=int(np.bincount(groups).max()), scores=scores,
                  bootstrap={'draws':2000,'rng':'RandomState(0)','unit':'connected support/query photographs'},
                  exposure='development; not independent confirmation', contrasts={})
    bases = [a for a in arrays if a == 'native' or a == 'rcg' or a.endswith('.control')]
    for arm,v in arrays.items():
        result['contrasts'][arm] = {}
        for base in bases:
            if base == arm: continue
            delta = v[:,0]/np.maximum(v[:,1],1)-arrays[base][:,0]/np.maximum(arrays[base][:,1],1)
            result['contrasts'][arm][base] = {
                'gain':scores[arm]-scores[base],
                'ci95':np.percentile(samples[arm]-samples[base],[2.5,97.5]).tolist(),
                'up':int((delta>1e-12).sum()),'down':int((delta< -1e-12).sum()),
                'tie':int((np.abs(delta)<=1e-12).sum())}
    for field in ('fold','batch'):
        result[field+'s'] = {}
        for label in sorted(set(str(r.get(field,'unspecified')) for r in rows)):
            ix = np.array([str(r.get(field,'unspecified')) == label for r in rows])
            vals = {a:metric(v[ix],cls[ix]) for a,v in arrays.items()}
            result[field+'s'][label] = {'n':int(ix.sum()),'scores':vals,
                'gain_vs_native':{a:s-vals['native'] for a,s in vals.items()}}
    result['corrections_vs_native'] = {a:{k:int(sum(r[k] for r in rec))
        for k in ('add_TP','delete_FP','delete_TP','add_FP')} for a,rec in corrections.items()}
    result['corrections_by_class'] = {a:{str(c):{k:int(sum(r[k] for r in rec if r['c']==c))
        for k in ('add_TP','delete_FP','delete_TP','add_FP')} for c in sorted(set(cls))} for a,rec in corrections.items()}
    result['corrections_by_batch'] = {a:{b:{k:int(sum(r[k] for r in rec if r['batch']==b))
        for k in ('add_TP','delete_FP','delete_TP','add_FP')} for b in sorted(set(r['batch'] for r in rec))} for a,rec in corrections.items()}
    return result, draws


def evaluate(root, run):
    run = Path(run); seal = json.loads((run/'sealed.json').read_text())
    if sha(run/'manifest.json') != seal['manifest_sha256']:
        raise ValueError('Manifest changed after inference')
    rows = json.loads((run/'manifest.json').read_text()); arrays = {'native':[]}; corrections = {}; details=[]
    for row in rows:
        key = row['key']; f = run/'predictions'/f'{key}.npz'
        if sha(f) != seal['predictions'][key]: raise ValueError('Prediction changed')
        if sha(packet(root,row)) != seal['inputs'][key]['packet_sha256']:
            raise ValueError('Evaluation packet changed')
        with np.load(packet(root,row),allow_pickle=False) as p:
            truth, native, pre = (unpack(p[k]) for k in ('truth','native','pre'))
            score = p['score'].astype(np.float32)
        if np.any(render((score-score.min())/max(float(score.max()-score.min()),1e-6)) != pre):
            raise ValueError(f'FoRIS pre renderer mismatch: {key}')
        masks = {'native':native}
        with np.load(f,allow_pickle=False) as p:
            masks.update({a:unpack(p[a]) for a in p.files})
        for arm,m in masks.items():
            iu = [int((m&truth).sum()),int((m|truth).sum())]
            arrays.setdefault(arm,[]).append(iu)
            add, delete = m&~native, native&~m
            corr = dict(key=key,c=row['c'],fold=row['fold'],batch=str(row.get('batch','unspecified')),
                        add_TP=int((add&truth).sum()),delete_FP=int((delete&~truth).sum()),
                        delete_TP=int((delete&truth).sum()),add_FP=int((add&~truth).sum()))
            corrections.setdefault(arm,[]).append(corr)
            details.append(dict(corr,arm=arm,intersection=iu[0],union=iu[1]))
    arrays = {a:np.asarray(v,dtype=np.int64) for a,v in arrays.items()}
    if any(len(v)!=len(rows) for v in arrays.values()): raise ValueError('Unpaired arms')
    result,draws = summarize(rows,arrays,corrections)
    result['prediction_seal_sha256'] = sha(run/'sealed.json')
    (run/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (run/'episode_metrics.json').write_text(json.dumps(details,indent=2)+'\n')
    np.save(run/'bootstrap_photo_draws.npy',draws)
    return result
