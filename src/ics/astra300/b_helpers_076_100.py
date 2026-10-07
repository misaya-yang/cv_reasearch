"""Original B common protocol: adjacent Ward candidates and four-fold banks.

No A/C calibration, no query labels. Matchers are explicit partial injective
role-to-observed-point searches with top-four choices and independent nulls.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
import numpy as np
from scipy import sparse
from scipy.special import logsumexp

from .common import Result, artifact, continuous_original, ArtifactUnavailable
from .group_076_150_common import unit,dot,edges4,blocks,spherical


def ward(x,hw,valid,levels=(8,32,128)):
    ids=np.flatnonzero(valid>0);n=len(x)
    if not len(ids):return ()
    parent=np.arange(n);members={int(i):np.array([i],np.int32)for i in ids}
    count={int(i):1 for i in ids};mean={int(i):x[i].copy()for i in ids};adj={int(i):set()for i in ids}
    ii,jj=edges4(hw,valid);heap=[]
    def value(a,b):return count[a]*count[b]/(count[a]+count[b])*np.sum((mean[a]-mean[b])**2)
    for a,b in zip(ii,jj):
        a,b=int(a),int(b);adj[a].add(b);adj[b].add(a);heapq.heappush(heap,(float(value(a,b)),a,b,1,1))
    desired=set(min(int(k),len(ids))for k in levels);regions=[];seen=set()
    def save():
        for a in sorted(members):
            m=members[a];key=m.tobytes()
            if key not in seen:regions.append(m.copy());seen.add(key)
    if len(members)in desired:save()
    while heap and len(members)>min(desired):
        energy,a,b,ca,cb=heapq.heappop(heap)
        if a not in members or b not in members or count[a]!=ca or count[b]!=cb or b not in adj[a]:continue
        if a>b:a,b=b,a
        ca,cb=count[a],count[b]
        merged=np.sort(np.r_[members[a],members[b]])
        mean[a]=(ca*mean[a]+cb*mean[b])/(ca+cb);count[a]=ca+cb;members[a]=merged
        neighbors=(adj[a]|adj[b])-{a,b};adj[a]=neighbors
        members.pop(b);mean.pop(b);count.pop(b);adj.pop(b)
        for neighbor in neighbors:
            adj[neighbor].discard(b);adj[neighbor].add(a)
            lo,hi=sorted((a,neighbor));heapq.heappush(heap,(float(value(lo,hi)),lo,hi,count[lo],count[hi]))
        if len(members)in desired:save()
    return tuple(regions)


def logmean(profile,weights,temperature=.07):
    weights=np.asarray(weights,float);keep=weights>0
    if not keep.any():return np.full(len(profile),np.nan)
    return temperature*(logsumexp(profile[:,keep]/temperature+np.log(weights[keep]),axis=1)-np.log(weights[keep].sum()))


def boundary_excluded_fold(hw,fold):
    b=blocks(hw,2);held=b==fold
    i,j=edges4(hw);excluded=held.copy()
    excluded[j[held[i]]]=True;excluded[i[held[j]]]=True
    return excluded


@dataclass
class Frame:
    ep:object
    x:np.ndarray
    hw:tuple
    valid:np.ndarray
    r:np.ndarray
    coverage:np.ndarray
    weights:np.ndarray
    r_ids:np.ndarray
    profile:np.ndarray
    fg:np.ndarray
    bg:np.ndarray
    fcenters:np.ndarray
    bcenters:np.ndarray
    flabels:np.ndarray
    blabels:np.ndarray
    regions:tuple
    ell:np.ndarray
    role_f:np.ndarray
    role_b:np.ndarray
    top_f:np.ndarray
    top_b:np.ndarray
    geometry_q:np.ndarray
    geometry_r:np.ndarray
    source:bool=False
    fold:int|None=None


class BPrepared:
    def __init__(self,ep):
        self.ep=ep;self.r=unit(ep.r);self.q=unit(ep.q);self.rv=np.asarray(ep.wvalid,float);self.qv=np.asarray(ep.q_valid,float)
        self.c=np.divide(ep.wf,self.rv,out=np.zeros_like(self.rv),where=self.rv>0)
        self.rb=blocks(ep.r_hw,2);self.rr=None
        self.q_regions=ward(self.q,ep.q_hw,self.qv);self.r_regions=ward(self.r,ep.r_hw,self.rv)
    def frame(self,source=False,fold=None,compute_profile=True):
        ids=np.flatnonzero((self.rv>0)&(True if fold is None else ~boundary_excluded_fold(self.ep.r_hw,fold)))
        x,hw,valid=(self.r,self.ep.r_hw,self.rv)if source else(self.q,self.ep.q_hw,self.qv)
        r=self.r[ids];coverage=self.c[ids];weights=self.rv[ids]
        fg=np.flatnonzero(coverage>=.9);bg=np.flatnonzero(coverage<=.1)
        fc,fl=spherical(r[fg],min(8,len(fg)))if len(fg)else(np.empty((0,x.shape[1])),np.empty(0,int))
        bc,bl=spherical(r[bg],min(8,len(bg)))if len(bg)else(np.empty((0,x.shape[1])),np.empty(0,int))
        profile=dot(x,r)if compute_profile else np.empty((len(x),0))
        if compute_profile:
            ell=logmean(profile,coverage*weights)-logmean(profile,(1-coverage)*weights)
            if not np.isfinite(ell).all():ell=np.zeros(len(x))
            tf=np.argsort(-profile[:,fg],axis=1,kind='stable')[:,:4]if len(fg)else np.empty((len(x),0),int)
            tb=np.argsort(-profile[:,bg],axis=1,kind='stable')[:,:4]if len(bg)else np.empty((len(x),0),int)
        else:ell=np.zeros(len(x));tf=tb=np.empty((len(x),0),int)
        y,xpos=np.unravel_index(np.arange(len(x)),hw)
        qgeo=np.column_stack(((y+.5)/hw[0],(xpos+.5)/hw[1]))
        ry,rx=np.unravel_index(ids,self.ep.r_hw)
        rgeo=np.column_stack(((ry+.5)/self.ep.r_hw[0],(rx+.5)/self.ep.r_hw[1]))
        return Frame(self.ep,x,hw,valid,r,coverage,weights,ids,profile,fg,bg,fc,bc,fl,bl,
                     self.r_regions if source else self.q_regions,ell,dot(x,fc),dot(x,bc),tf,tb,qgeo,rgeo,source,fold)
    def structure_active(self):
        for fold in range(4):
            ids=np.flatnonzero((self.rv>0)&~boundary_excluded_fold(self.ep.r_hw,fold))
            held=np.flatnonzero((self.rb==fold)&(self.rv>0))
            if not len(held):return False
            if not np.any(self.c[ids]>=.9)or not np.any(self.c[ids]<=.1):return False
        return bool(np.any((self.rv>0)&(self.c>=.9))and np.any((self.rv>0)&(self.c<=.1)))
    def degenerate(self):
        if self.ep.wf.sum()==0:return np.full(len(self.q),-1.),'empty_reference_mask'
        if not np.any((self.rv>0)&(self.c>=.9)):
            sim=dot(self.q,self.r[self.rv>0]);ell=logmean(sim,self.ep.wf[self.rv>0])-logmean(sim,self.ep.wb[self.rv>0])
            return ell,'no_pure_foreground_soft_complete_coverage_logmean'
        if not np.any((self.rv>0)&(self.c<=.1)):
            fg=np.flatnonzero((self.rv>0)&(self.c>=.9));mean=unit(self.r[fg].mean(axis=0));score=self.q@mean
            held_scores=[];units=0
            for fold in range(4):
                test=fg[self.rb[fg]==fold];train=fg[self.rb[fg]!=fold]
                if len(test)and len(train):held_scores.extend((self.r[test]@unit(self.r[train].mean(axis=0))).tolist());units+=1
            threshold=float(np.quantile(held_scores,.05))if units>=2 else 1.
            return score-threshold,'no_background_single_class_spatial_leaveout_support'
        return None


_CACHE=None
def prepare_b(ep):
    global _CACHE
    if _CACHE is None or _CACHE.ep is not ep:_CACHE=BPrepared(ep)
    return _CACHE


def calibrate(p,fn,mode='threshold'):
    scores=[];coverage=[];weights=[]
    if not p.structure_active():return None
    for fold in range(4):
        f=p.frame(True,fold);s=np.asarray(fn(f),float)
        held=(p.rb==fold)&(p.rv>0)
        if s.shape!=(len(p.r),)or not np.isfinite(s[held]).all():raise ValueError('B source fn must return complete finite field')
        scores.extend(s[held]);coverage.extend(p.c[held]);weights.extend(p.rv[held])
    s=np.asarray(scores);c=np.asarray(coverage);w=np.asarray(weights)
    if mode=='iqr':return float(np.subtract(*np.percentile(s,[75,25])))
    candidates=np.unique(np.quantile(s,[.1,.25,.5,.75,.9]))
    def risk(t):
        prediction=s>t
        return .5*(np.sum(w*c*~prediction)/np.sum(w*c)+np.sum(w*(1-c)*prediction)/np.sum(w*(1-c)))
    return min(candidates,key=lambda t:(risk(t),-t))


def owner(frame,energies,point_fields):
    out=np.zeros(len(frame.x));best=np.full(len(frame.x),np.inf);size=np.full(len(frame.x),np.inf);index=np.full(len(frame.x),np.inf)
    for stable,(c,energy,field)in enumerate(zip(frame.regions,energies,point_fields)):
        field=np.asarray(field,float)
        if field.shape!=(len(c),):raise ValueError('owner fields must align with every candidate member')
        value=float(energy)/max(np.count_nonzero(field),1)
        win=(value<best[c])|((value==best[c])&((len(c)<size[c])|((len(c)==size[c])&(stable<index[c]))))
        ids=c[win];out[ids]=field[win];best[ids]=value;size[ids]=len(c);index[ids]=stable
    return out


def finish_b(ep,raw,method,kind,scale=None,info=None):
    info=dict(info or {},method_id=method,source_group='B supplied Astra300',qgt_access=False,branch=kind)
    if kind=='D':return Result(np.asarray(raw).reshape(ep.q_hw),0.,info=info)
    p0=np.asarray(artifact(ep,'foris_p0'),float)
    mask0=np.asarray(artifact(ep,'foris_mask0'))
    producer=artifact(ep,'foris_producer');renderer=artifact(ep,'foris_renderer')
    if not isinstance(producer,dict)or producer.get('pipeline')!='FoRIS_native_Part1_2_3_4_original_config':
        raise ArtifactUnavailable('B H0 requires original-config actual FoRIS Part1--4; MEAN host is not a substitute')
    if producer.get('source_image_hashes')!=ep.producer.get('source_image_hashes'):
        raise ArtifactUnavailable('FoRIS H0 images do not match this native feature episode')
    if mask0.dtype!=bool or mask0.shape!=tuple(ep.original_shape):raise ValueError('Actual original-size FoRIS mask required')
    info.update(host_producer=producer,host_renderer=renderer)
    if p0.shape!=tuple(ep.q_hw):raise ValueError('B p0 must match physical native patch grid; original M0 stays original-sized')
    if kind=='W':
        a=np.asarray(raw).reshape(ep.q_hw);a=np.where(np.isfinite(a),a,p0)
        p1=continuous_original(ep,a)>.5;old=continuous_original(ep,p0)>.5
        mask=np.where(p1!=old,p1,mask0)
        return Result(a,.5,mask,info)
    u=np.asarray(raw).reshape(ep.q_hw)
    if scale is None or scale<=0:u=np.zeros_like(u)
    else:u=u/scale
    if kind=='Z':u=np.minimum(u,0)
    b=np.log(np.clip(p0,.01,.99)/(1-np.clip(p0,.01,.99)))
    p1=continuous_original(ep,b+u)>0;old=continuous_original(ep,b)>0
    mask=np.where(p1!=old,p1,mask0)
    if kind=='Z':mask&=mask0
    return Result(b+u,0.,mask,info)


def role_neighbors(f,foreground):
    ids=f.fg if foreground else f.bg;label=f.flabels if foreground else f.blabels
    count=len(f.fcenters if foreground else f.bcenters)
    matrix=np.zeros((count,count),bool);lookup={int(f.r_ids[i]):int(role)for i,role in zip(ids,label)}
    ii,jj=edges4(f.ep.r_hw)
    for a,b in zip(ii,jj):
        if int(a)in lookup and int(b)in lookup:
            ra,rb=lookup[int(a)],lookup[int(b)]
            if ra!=rb:matrix[ra,rb]=matrix[rb,ra]=True
    return matrix


def choices(f,c,foreground):
    own=f.role_f if foreground else f.role_b;opposite=f.role_b if foreground else f.role_f
    top=f.top_f if foreground else f.top_b;labels=f.flabels if foreground else f.blabels
    rival=opposite.max(axis=1)if opposite.shape[1]else np.ones(len(f.x))
    options=[]
    for role in range(own.shape[1]):
        eligible=np.array([np.any(labels[top[q]]==role)for q in c])if top.shape[1]else np.zeros(len(c),bool)
        ids=c[eligible];order=np.lexsort((ids,rival[ids]-own[ids,role]))[:4]
        options.append(tuple(ids[order].tolist()))
    return options,rival


def partial_match(f,c,foreground=True,edge_weight=1.,forced=None,allowed_roles=None,width=32):
    own=f.role_f if foreground else f.role_b;options,rival=choices(f,c,foreground);adj=role_neighbors(f,foreground)
    states=[(0.,())]
    active=set(range(len(options)))if allowed_roles is None else set(allowed_roles)
    for role,points in enumerate(options):
        opts=(-1,)+points if role in active else(-1,)
        if forced is not None and role in forced:opts=(forced[role],)
        expanded=[]
        for energy,mapping in states:
            for point in opts:
                if point>=0 and point in mapping:continue
                cost=0 if point<0 else rival[point]-own[point,role]
                if point>=0 and edge_weight:
                    for old,oldpoint in enumerate(mapping):
                        if oldpoint>=0 and adj[old,role]:
                            # Reference relation licenses a deformable local edge, not a fixed pose.
                            cost+=edge_weight*max(np.linalg.norm(f.geometry_q[point]-f.geometry_q[oldpoint])-2/max(f.hw),0)
                expanded.append((energy+float(cost),mapping+(int(point),)))
        expanded.sort(key=lambda state:(state[0],state[1]));states=expanded[:width]
    if not states:return 0.,tuple([-1]*own.shape[1]),np.zeros(len(c))
    energy,mapping=states[0];field=np.zeros(len(c));location={int(q):i for i,q in enumerate(c)}
    for role,point in enumerate(mapping):
        if point>=0 and point in location:
            margin=own[point,role]-rival[point]
            field[location[point]]=margin if foreground else-margin
    return energy,mapping,field
