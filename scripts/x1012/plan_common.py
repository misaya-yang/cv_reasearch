"""Shared cache-only execution/report helpers for the independent-base plan.

No model constructor or encoding fallback exists. Existing lab functions remain
unchanged; callers select CPU/MPS before calling them.
"""
from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
import builtins
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
import numpy as np
import torch
import lab

ROOT, ASSETS = lab.ROOT, lab.ASSETS
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n'); t.replace(p)


def key(b):
    return str(b['index']) if isinstance(b['index'], str) else f"{b['index']:06d}"


def seed_rows(n=150):
    return lab.subset(n) + lab.ext_subset(n)


def setup(device='cpu'):
    lab.DEV=torch.device(device); torch.set_num_threads(4); torch.manual_seed(0)


@contextmanager
def inference_only(binding):
    """Deny the current query annotation except an identical legal R-mask file."""
    from PIL import Image
    r=binding['row']; reference=Path(r['reference_mask_path']).resolve()
    forbidden={Path(r[k]).resolve() for k in ('query_mask_path','query_ignore_mask_path') if k in r}
    forbidden.discard(reference)
    saved=builtins.open,io.open,Image.open
    def wrap(fn):
        def guarded(path,*a,**kw):
            if isinstance(path,(str,bytes,os.PathLike)) and Path(os.fsdecode(path)).resolve() in forbidden:
                raise RuntimeError('Query GT is forbidden in inference')
            return fn(path,*a,**kw)
        return guarded
    builtins.open,io.open,Image.open=map(wrap,saved)
    try:
        with torch.inference_mode():
            yield
    finally:
        builtins.open,io.open,Image.open=saved


class Replay(torch.nn.Module):
    encoder=None
    encoder_calls=0

    def bind(self,binding):
        from p0_manifest import raw_metadata
        self.bound={}
        for name in ('reference_raw','query_raw_identity'):
            r=raw_metadata(binding['fit'][name]['payload_path'])
            self.bound[r['input_sha256']]=r
        self.memo={}

    def get_intermediate_layers(self,inputs,n=1,reshape=True):
        from raw_feature_cache import tensor_hash
        assert n==1 and reshape and inputs.dtype==torch.float32 and inputs.device.type=='cpu'
        maps=[]
        for x in inputs.numpy():
            h=tensor_hash(x)
            if h not in self.memo:
                b=self.bound[h]
                assert sha(b['entry_path'])==b['entry_sha256'] and sha(b['payload_path'])==b['payload_sha256']
                with np.load(b['payload_path'],allow_pickle=False) as z:
                    raw=z['O/24'].copy()
                assert raw.dtype==np.float32 and raw.shape==(4096,1024) and np.isfinite(raw).all()
                assert tensor_hash(raw)==b['tensor_sha256']
                self.memo[h]=torch.from_numpy(raw).reshape(64,64,1024).permute(2,0,1).contiguous()
            maps.append(self.memo[h])
        return [torch.stack(maps)]


def extension(name,path):
    if name in sys.modules:
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.modules[name]=m
    return m


def host_cpu():
    """Original cache_host with existing native binaries; no build/write side effects."""
    import ics.m4_crf as crf
    import ics.m4_crf_cached as prepared_crf
    from cached_dino import cache_host
    native=extension('ics_permutohedral_m4',ASSETS/'runtime/macos/crf/ics_permutohedral_m4.so')
    prepared=extension('ics_permutohedral_m4_prepared_v1',ASSETS/'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so')
    class Layer(torch.nn.Module):
        def __init__(self,bilateral,theta_alpha,theta_beta,theta_gamma,nhwc=True):
            super().__init__(); self.bilateral=bilateral; self.parameters_=(theta_alpha,theta_beta,theta_gamma)
        def forward(self,x,image):
            v=x.detach().float().permute(0,2,3,1).contiguous()
            c=image.detach().float().permute(0,2,3,1).contiguous()
            return native.forward(v,c,self.bilateral,*self.parameters_).permute(0,3,1,2).contiguous()
    module=types.ModuleType('PermutohedralFiltering'); module.PermutohedralLayer=Layer
    sys.modules['PermutohedralFiltering']=module
    sys.path.insert(0,str(ASSETS/'third_party/crf_source/src'))
    install,build=crf.install,prepared_crf.build_backend
    crf.install=lambda *a,**k:(Layer,{'prebuilt':True})
    prepared_crf.build_backend=lambda *a,**k:prepared
    adapter=Replay()
    try:
        host=cache_host(ASSETS,adapter)
    finally:
        crf.install,prepared_crf.build_backend=install,build
    return host,adapter


def pack(pred):
    return np.packbits(np.asarray(pred,dtype=bool))


def unpack(bits,shape=(1024,1024)):
    return np.unpackbits(bits)[:int(np.prod(shape))].reshape(shape).astype(bool)


def readout(field):
    up=lab.up(torch.as_tensor(field,dtype=torch.float32).reshape(64,64))
    return (lab.minmax(up)>.5).cpu().numpy()


def truth(b):
    """Scoring-only interface; callers seal predictions before entering it."""
    gt=lab.mask1024(b['row']['query_mask_path']).numpy()
    ignore=lab.ignore1024(b)
    valid=np.ones((1024,1024),bool) if ignore is None else ~ignore.numpy()
    return gt&valid,valid


def pixel_metrics(pred,gt,valid):
    p=np.asarray(pred,dtype=bool)&valid
    tp=int(np.count_nonzero(p&gt)); fp=int(np.count_nonzero(p&~gt)); fn=int(np.count_nonzero(gt&~p))
    area=int(np.count_nonzero(gt)); predicted=int(np.count_nonzero(p))
    return {'I':tp,'U':tp+fp+fn,'TP':tp,'FP':fp,'FN':fn,'GT':area,'predicted':predicted,
            'IoU':100*tp/max(tp+fp+fn,1),'target_fraction':area/max(int(valid.sum()),1),
            'abs_log2_area_error':float(abs(np.log2(max(predicted,1)/max(area,1)))),
            'zero_predicted_area':predicted==0}


def pooled(rows):
    slots={}
    for r in rows:
        slot=(str(r['fold']),str(r['class_id']))
        old=slots.setdefault(slot,[0,0]); old[0]+=r['I']; old[1]+=r['U']
    return float(np.mean([100*a/max(b,1) for a,b in slots.values()])) if slots else None


def summarize(records,arms,bootstrap=1000,seed=20261012):
    """Plan metric: mean across observed (fold,global-class) pooled slots.

    Resample query photographs as clusters, retaining every episode/arm together.
    Each bootstrap replicate uses the observed class slots present in that draw,
    exactly the plan's observed-class estimator; missing slots are not zeroed.
    """
    output={}; rng=np.random.default_rng(seed)
    for d in lab.ALL_DOMAINS:
        subset=[r for r in records if r['dataset']==d]
        if not subset: continue
        slots=sorted({(str(r['fold']),str(r['class_id'])) for r in subset})
        photos=sorted({r['query_photo'] for r in subset}); pi={p:i for i,p in enumerate(photos)}
        si={s:i for i,s in enumerate(slots)}
        from scipy.sparse import csr_matrix
        photo_indices=[pi[r['query_photo']] for r in subset]
        slot_indices=[si[(str(r['fold']),str(r['class_id']))] for r in subset]
        dimensions=(len(photos),len(slots))
        presence=csr_matrix((np.ones(len(subset)),(photo_indices,slot_indices)),shape=dimensions)
        weights=np.stack([np.bincount(rng.integers(0,len(photos),len(photos)),minlength=len(photos)) for _ in range(bootstrap)])
        bs={}
        represented=(weights@presence)>0
        for a in arms:
            inter=csr_matrix(([r['metrics'][a]['I'] for r in subset],(photo_indices,slot_indices)),shape=dimensions)
            union=csr_matrix(([r['metrics'][a]['U'] for r in subset],(photo_indices,slot_indices)),shape=dimensions)
            values=100*(weights@inter)/np.maximum(weights@union,1)
            bs[a]=(values*represented).sum(1)/represented.sum(1)
        table={}
        for a in arms:
            ms=[r['metrics'][a] for r in subset]
            vals=[dict(m,fold=r['fold'],class_id=r['class_id']) for r,m in zip(subset,ms)]
            bands={label:[m for m in ms if lo<=m['target_fraction']<hi] for label,lo,hi in (('<2%',0,.02),('2-20%',.02,.2),('>20%',.2,1.00001))}
            delta=bs[a]-bs['FoRIS']
            table[a]={'n':len(ms),'photo_n':len(photos),'class_slot_n':len(slots),'class_miou':pooled(vals),
                      'episode_mean_iou':float(np.mean([m['IoU'] for m in ms])),
                      'FN_GT':sum(m['FN'] for m in ms)/max(sum(m['GT'] for m in ms),1),
                      'FP_GT':sum(m['FP'] for m in ms)/max(sum(m['GT'] for m in ms),1),
                      'median_abs_log2_area_error':float(np.median([m['abs_log2_area_error'] for m in ms])),
                      'zero_prediction_n':sum(m['zero_predicted_area'] for m in ms),
                      'size_bands':{k:{'n':len(v),'episode_mean_iou':float(np.mean([m['IoU'] for m in v])) if v else None} for k,v in bands.items()},
                      'delta_to_foris_pp':pooled(vals)-pooled([dict(r['metrics']['FoRIS'],fold=r['fold'],class_id=r['class_id']) for r in subset]),
                      'paired_query_photo_bootstrap95_pp':[float(x) for x in np.quantile(delta,[.025,.975])]}
        output[d]=table
    return output
