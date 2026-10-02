"""Recover immutable photo-role manifest from the exact seed-0 episode algorithm; CPU only."""
import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--meta',required=True); p.add_argument('--select',required=True); p.add_argument('--reliab',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); meta=pickle.loads(Path(a.meta).read_bytes()); sr=json.loads(Path(a.select).read_text())['records']; rr=json.loads(Path(a.reliab).read_text())['records']
    np.random.seed(0); eps=[]
    for _ in sr:
        c=int(np.random.choice([4*v for v in range(20)],1,replace=False)[0]); q=str(np.random.choice(meta[c],1,replace=False)[0])
        while True:
            s=str(np.random.choice(meta[c],1,replace=False)[0])
            if s!=q: break
        eps.append((c,q,s))
    rng=np.random.default_rng(0); records=[]
    canon=lambda n:str(int(re.search(r'(\d+)\.[^.]+$',n).group(1)))
    for e,((c,q,s),sel,rel) in enumerate(zip(eps,sr,rr)):
        candidates=[o for o,(oc,oq,_) in enumerate(eps) if o!=e and oc==c and oq not in (s,q)]
        seen=set(); candidates=[o for o in candidates if not(eps[o][1] in seen or seen.add(eps[o][1]))]
        pool=[int(o) for o in rng.permutation(candidates)[:15]]; names=[s,q]+[eps[o][1] for o in pool]
        assert sel['e']==rel['e']==e and sel['c']==rel['c']==c
        assert len(pool)==len(rel['pool']['qual'])
        nonempty=np.asarray(rel['pool']['area'])!=-9
        assert int(nonempty.sum())+1==len(sel['pseudo']['t'])
        assert np.array_equal(np.asarray(rel['pool']['link'])[nonempty],sel['pseudo']['link'][1:])
        records.append(dict(e=e,c=c,image_ids=[canon(n) for n in names],image_names=names,donor_cache_indices=[2*o+1 for o in pool]))
    out=dict(provenance=dict(seed=0,fold=0,M=15,canonical='numeric COCO photo ID',source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in (a.meta,a.select,a.reliab)}),records=records)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out)); print('validated immutable roles for',len(records),'episodes',flush=True)

if __name__=='__main__': main()
