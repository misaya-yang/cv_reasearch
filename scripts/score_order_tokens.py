#!/usr/bin/env python3
"""Attribute the loss of the ordering role from the token record of run_order_tokens.py (CPU, reads truth: a diagnostic).

  python scripts/score_order_tokens.py outputs/claude_order_fresh600
"""
import numpy as np, json, sys
from scipy import ndimage
D=sys.argv[1].rstrip('/')+'/'
Z=np.load(D+'tokens.npz'); rows=json.load(open(D+'rows.json')); N=len(rows); cls=np.array([r['c'] for r in rows])
t=Z['truth'].astype(np.float32); F={k:Z[k].astype(np.float32) for k in ('s2','s3','score','rcg','a_fg','a_bg','knn_fg')}; F['nn']=F['a_fg']-F['a_bg']; nbr=Z['nbr'].astype(np.int64)
def miou(I,U): return float(np.mean([I[cls==c].sum()/max(U[cls==c].sum(),1e-9) for c in np.unique(cls)])*100)
def best_cut(f):
    """per episode: the set of tokens above the cut that maximises soft IoU"""
    m=np.zeros((N,4096),bool); I=np.zeros(N); U=np.zeros(N)
    for n in range(N):
        o=np.argsort(-f[n]); ci=np.cumsum(t[n][o]); cu=t[n].sum()+np.cumsum(1-t[n][o]); k=int(np.argmax(ci/np.maximum(cu,1e-9))); m[n,o[:k+1]]=True; I[n]=ci[k]; U[n]=cu[k]
    return m,I,U
def half(f):
    lo=f.min(1,keepdims=True); hi=f.max(1,keepdims=True); m=(f-lo)/np.maximum(hi-lo,1e-9)>.5; return m,(m*t).sum(1),t.sum(1)+(m*(1-t)).sum(1)
print('## ordering role: token-level class mIoU, %d episodes'%N); print('| field | cut at half of its range | best cut per episode | ordering loss |'); print('|---|---:|---:|---:|')
M={}
for k in ('nn','knn_fg','s2','s3','score','rcg'):
    m,I,U=best_cut(F[k]); M[k]=m; _,Ih,Uh=half(F[k]); print('| %s | %.2f | %.2f | %.2f |'%(k,miou(Ih,Uh),miou(I,U),100-miou(I,U)))
for k in ('score','rcg'):
    m=M[k]; obj=t>=.9; bg=t<=.1; mix=~obj&~bg
    miss=obj&~m; false=bg&m
    Rp=F['nn']>0; det_n=np.take_along_axis(m[:,None,:].repeat(1,1),np.zeros((N,1,1),int),1) if False else None
    nd=np.stack([m[n][nbr[n]].mean(1) for n in range(N)]); nt=np.stack([(t[n][nbr[n]]>.5).mean(1) for n in range(N)])
    comp_miss=np.zeros((N,4096),bool); comp_false=np.zeros((N,4096),bool)
    for n in range(N):
        lo,_=ndimage.label((t[n]>.5).reshape(64,64)); hit=np.unique(lo[m[n].reshape(64,64)&(lo>0)]); comp_miss[n]=((lo>0)&~np.isin(lo,hit)).flatten()
        lm,_=ndimage.label(m[n].reshape(64,64)); hit=np.unique(lm[(t[n]>.5).reshape(64,64)&(lm>0)]); comp_false[n]=((lm>0)&~np.isin(lm,hit)).flatten()
    base_I=(m*t).sum(1); base_U=t.sum(1)+(m*(1-t)).sum(1); b=miou(base_I,base_U)
    def fix(sel,kind):
        if kind=='miss': I=base_I+(sel*t).sum(1); U=base_U
        else: I=base_I; U=base_U-(sel*(1-t)).sum(1)
        return miou(I,U)-b
    print('\n## what the best cut of `%s` still gets wrong (best-cut mIoU %.2f); points recovered if that part alone were right'%(k,b))
    tm=(~m*t).sum(); tf=(m*(1-t)).sum()
    print('missed object area %.0f tokens, false area %.0f tokens; in mixed (boundary) tokens: missed %.0f%%, false %.0f%%'%(tm,tf,100*(~m*t*mix).sum()/tm,100*(m*(1-t)*mix).sum()/tf))
    print('| part | share of that error | points |'); print('|---|---:|---:|')
    def line(name,sel,kind,tot): print('| %s | %.0f%% | %+.2f |'%(name,100*((sel*t).sum() if kind=='miss' else (sel*(1-t)).sum())/tot,fix(sel,kind)))
    line('all missed',~m,'miss',tm); line('all false',m,'false',tf)
    line('missed, boundary tokens',~m&mix,'miss',tm); line('false, boundary tokens',m&mix,'false',tf)
    line('missed: whole object region undetected',miss&comp_miss,'miss',tm); line('missed: part of a detected region',miss&~comp_miss,'miss',tm)
    line('false: separate region with no object',false&comp_false,'false',tf); line('false: attached to a detected object',false&~comp_false,'false',tf)
    line('missed: nearest reference token is object',miss&Rp,'miss',tm); line('missed: nearest reference token is background',miss&~Rp,'miss',tm)
    line('false: nearest reference token is object',false&Rp,'false',tf); line('false: nearest reference token is background',false&~Rp,'false',tf)
    line('missed: most feature neighbours detected',miss&(nd>.5),'miss',tm); line('missed: most feature neighbours undetected',miss&(nd<=.5),'miss',tm)
    line('false: most feature neighbours detected',false&(nd>.5),'false',tf); line('false: most feature neighbours undetected',false&(nd<=.5),'false',tf)
    line('missed: most feature neighbours are object (GT)',miss&(nt>.5),'miss',tm); line('missed: most feature neighbours are background (GT)',miss&(nt<=.5),'miss',tm)
    line('false: most feature neighbours are background (GT)',false&(nt<=.5),'false',tf); line('false: most feature neighbours are object (GT)',false&(nt>.5),'false',tf)
    iou=base_I/np.maximum(base_U,1e-9); o=np.argsort(iou); cum=np.cumsum((base_U-base_I)[o])/(base_U-base_I).sum()
    print('episodes with best-cut IoU < 0.2: %d (%.0f%% of all error area); < 0.5: %d (%.0f%%)'%((iou<.2).sum(),100*(base_U-base_I)[iou<.2].sum()/(base_U-base_I).sum(),(iou<.5).sum(),100*(base_U-base_I)[iou<.5].sum()/(base_U-base_I).sum()))
