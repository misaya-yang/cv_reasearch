"""RSRM SAFR attention-branch fusion adapted to the existing DINOv3-L.

Read post-LayerScale attention increments, with the original final LayerNorm,
without changing the encoder forward. Selection uses reference labels only.
RSRM's published default is ViT-B; this is a ViT-L composition experiment.
"""
from __future__ import annotations
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import cg
from .methods import rcg


def selector(source):
    source = Path(source)
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RSRM')
    names = ['features_refusion','find_best_semantic_layer','fisher_ratio']
    methods = [next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in names]
    # These are the reviewed reference-only source functions. Do not import
    # the upstream constructor (which loads another backbone) or its KMeans.
    isolated = ast.Module(body=[ast.ClassDef(name='SAFRSelector',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    namespace = {'torch':torch,'F':F}
    exec(compile(ast.fix_missing_locations(isolated),str(source),'exec'),namespace)
    receipt = dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        method_ast_sha256=hashlib.sha256(ast.dump(isolated,include_attributes=False).encode()).hexdigest(),methods=names)
    return namespace['SAFRSelector'](),receipt


@contextmanager
def capture(model):
    """Collect normalized attention-only patch maps; hooks return no replacement."""
    got,handles = {},[]
    prefix = model.num_prefix_tokens
    def hook(index,scale=None):
        def save(module,inputs,output):
            if scale is not None:
                output = output * scale
            patches = model.norm(output[:,prefix:])
            n = patches.shape[1]; h = int(n**.5)
            if h*h != n: raise ValueError('Square patch grid required')
            got[index] = patches.transpose(1,2).reshape(patches.shape[0],patches.shape[2],h,h).cpu()
        return save
    try:
        for i,block in enumerate(model.blocks):
            if hasattr(block,'ls1'):
                handles.append(block.ls1.register_forward_hook(hook(i)))
            elif type(block).__name__=='EvaBlock' and hasattr(block,'gamma_1'):
                handles.append(block.attn.register_forward_hook(hook(i,block.gamma_1)))
            else:
                raise ValueError('Unsupported attention increment path: '+type(block).__name__)
        yield got
    finally:
        for handle in handles: handle.remove()


def fuse(branches,last_reference,reference_mask,selection):
    """Original SAFR selection/fusion; explicit baseline fallback if undefined."""
    layers = torch.stack([branches[k] for k in sorted(branches)],dim=1)
    if layers.ndim!=5 or layers.shape[0]!=2: raise ValueError('Paired support/query attention branches required')
    mask = F.interpolate(reference_mask[None,None].float(),layers.shape[-2:],mode='bilinear',align_corners=True)[0,0]>.5
    counts = [int(mask.sum()),int((~mask).sum())]
    if min(counts)<2:
        return None,dict(fallback='reference role has fewer than two patches',role_counts=counts)
    support = layers[:1]; std = layers.std(dim=(2,3,4),keepdim=True)
    if not torch.isfinite(layers).all() or bool((std<=0).any()):
        return None,dict(fallback='nonfinite or constant attention branch',role_counts=counts)
    weights = selection.features_refusion([last_reference[None]], [support], [mask[None].long()])
    if not torch.isfinite(weights).all() or float(weights.sum())<=0:
        return None,dict(fallback='upstream SAFR selection has no finite supported weights',role_counts=counts)
    fused = (layers*weights[:,:,None,None,None]/std).sum(1)
    q,r = (F.normalize(fused[i].flatten(1).T.contiguous(),dim=1) for i in (1,0))
    fg = mask.flatten()
    mu_f,mu_b = (F.normalize(r[v].mean(0),dim=0) for v in (fg,~fg))
    foreground = (q@mu_f).numpy(); difference = (q@(mu_f-mu_b)).numpy()
    return {'fg':foreground,'fg_bg':difference},dict(fallback=None,role_counts=counts,
        selected_layers=(torch.nonzero(weights[0]>0,as_tuple=False).flatten()+1).tolist(),weights=weights[0].tolist(),
        branch_definition='final-LN(post-LayerScale attention increment), no residual/FFN',
        mask_definition='RSRM bilinear align_corners=True >.5',backbone_adaptation='DINOv3-L/24 layers')


def solve_guide(q,cov,score,guide):
    """Historical MEAN graph/constants with only its semantic guide supplied."""
    q=F.normalize(torch.as_tensor(q).float(),dim=1)
    guide=np.asarray(guide,np.float32).reshape(-1)
    if q.shape!=(4096,1024) or guide.shape!=(4096,) or not np.isfinite(guide).all():
        raise ValueError('Invalid guide/cache shapes')
    s=rcg.minmax(score).ravel(); y=(s+.25*(rcg.rank(guide)-rcg.rank(s))).astype(np.float64)
    sim=q@q.T;sim.fill_diagonal_(-2);values,idx=sim.topk(20,dim=1);del sim
    distance=(1-values).clamp_min(0)
    weights=torch.exp(-distance/distance[:,-1:].clamp_min(1e-6)).numpy().ravel()
    w=sparse.csr_matrix((weights,(np.repeat(np.arange(4096),20),idx.numpy().ravel())),shape=(4096,4096))
    w=w.multiply(w.T);w.data=np.sqrt(w.data)
    degree=np.asarray(w.sum(1)).ravel();w=w/max(float(degree.mean()),1e-8);degree=np.asarray(w.sum(1)).ravel()
    a=.1+np.abs(2*s-1);a=(a/a.mean()).astype(np.float64)
    h=sparse.diags(a)+16*(sparse.diags(degree)-w);rhs=a*y;iterations=[0]
    def cb(_):iterations[0]+=1
    z,status=cg(h,rhs,x0=y,rtol=1e-7,atol=1e-9,maxiter=300,callback=cb)
    if status:raise RuntimeError('Guide control CG failure: '+str(status))
    return z.reshape(64,64).astype(np.float32),dict(cg_iterations=iterations[0],
        cg_relative_residual=float(np.linalg.norm(h@z-rhs)/max(np.linalg.norm(rhs),1e-12)),
        graph_undirected_edges=int(w.nnz//2),alpha=.25,lambda_=16)
