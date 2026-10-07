"""Shared deterministic legal structures for Pro30 M11--M16, no query GT."""
from __future__ import annotations
import heapq,time
import numpy as np
from scipy import sparse
from .common import unit,artifact,ArtifactUnavailable
_CACHE_EP=None;_CACHE={}


def cached(ep,name,fn):
    global _CACHE_EP,_CACHE
    if _CACHE_EP is not ep:_CACHE_EP=ep;_CACHE={}
    if name not in _CACHE:_CACHE[name]=fn()
    return _CACHE[name]


def edges4(hw,valid=None):
    grid=np.arange(np.prod(hw)).reshape(hw);i=np.r_[grid[:-1].ravel(),grid[:,:-1].ravel()];j=np.r_[grid[1:].ravel(),grid[:,1:].ravel()]
    if valid is not None:keep=(valid[i]>0)&(valid[j]>0);i,j=i[keep],j[keep]
    return i,j


def prototype(ep):
    return cached(ep,'prototype',lambda:np.asarray(ep.q)@(unit(np.sum(ep.r*ep.wf[:,None],axis=0))-unit(np.sum(ep.r*ep.wb[:,None],axis=0)) ))


def weighted_atoms(x,weights,K=32,rounds=5):
    x=np.asarray(x,float);w=np.asarray(weights,float);ids=np.flatnonzero(w>0)
    if not len(ids):return np.empty((0,x.shape[1])),np.empty(0),()
    K=min(K,len(ids));first=int(ids[np.argmax(w[ids])]);chosen=[first];distance=np.full(len(ids),np.inf)
    while len(chosen)<K:
        distance=np.minimum(distance,np.maximum(1-x[ids]@x[chosen[-1]],0));score=w[ids]*distance;score[np.isin(ids,chosen)]=-np.inf
        chosen.append(int(ids[np.argmax(score)]))
    centers=x[chosen].copy()
    for _ in range(rounds):
        labels=np.argmax(x[ids]@centers.T,axis=1)
        for k in range(K):
            selected=ids[labels==k]
            if len(selected):centers[k]=unit(np.sum(x[selected]*w[selected,None],axis=0))
    labels=np.argmax(x[ids]@centers.T,axis=1);groups=[ids[labels==k]for k in range(K)];keep=[k for k,g in enumerate(groups)if len(g)]
    mass=np.array([w[groups[k]].sum()for k in keep]);return centers[keep],mass/mass.sum(),tuple(groups[k]for k in keep)


def ward_tree(x,hw,valid,target=128):
    """FP64 weighted spatial Ward; exactly full binary history after target cut."""
    x=np.asarray(x,np.float64);valid=np.asarray(valid,float);ids=np.flatnonzero(valid>0)
    means={int(i):x[i].copy()for i in ids};mass={int(i):float(valid[i])for i in ids};members={int(i):np.array([i],int)for i in ids};adj={int(i):set()for i in ids};heap=[]
    def cost(a,b):return float(mass[a]*mass[b]/(mass[a]+mass[b])*np.sum((means[a]-means[b])**2))
    i,j=edges4(hw,valid)
    for a,b in zip(i,j):a,b=int(a),int(b);adj[a].add(b);adj[b].add(a);heapq.heappush(heap,(cost(a,b),a,b,mass[a],mass[b]))
    target=min(target,len(ids));nodes=[];node_ids={};leaves=None
    def initialize():
        nonlocal leaves
        for a in sorted(members):nodes.append({'support':members[a].copy(),'children':(),'mass':mass[a],'mean':unit(means[a])});node_ids[a]=len(nodes)-1
        leaves=tuple(range(len(nodes)))
    if len(members)==target:initialize()
    while heap and len(members)>1:
        _,a,b,ma,mb=heapq.heappop(heap)
        if a not in members or b not in members or ma!=mass[a]or mb!=mass[b]or b not in adj[a]:continue
        if a>b:a,b=b,a
        olda,oldb=mass[a],mass[b];support=np.sort(np.r_[members[a],members[b]]);means[a]=(olda*means[a]+oldb*means[b])/(olda+oldb);mass[a]=olda+oldb;members[a]=support
        if leaves is not None:
            nodes.append({'support':support.copy(),'children':(node_ids[a],node_ids[b]),'mass':mass[a],'mean':unit(means[a])});node_ids[a]=len(nodes)-1;node_ids.pop(b)
        neighbors=(adj[a]|adj[b])-{a,b};adj[a]=neighbors
        for mapping in(means,mass,members,adj):mapping.pop(b)
        for neighbor in neighbors:
            adj[neighbor].discard(b);adj[neighbor].add(a);lo,hi=sorted((a,neighbor));heapq.heappush(heap,(cost(lo,hi),lo,hi,mass[lo],mass[hi]))
        if leaves is None and len(members)==target:initialize()
    if leaves is None:initialize()
    return nodes,leaves,tuple(node_ids[a]for a in sorted(members))


def tree(ep,role='q',target=128):
    x,hw,v=(ep.q,ep.q_hw,ep.q_valid)if role=='q'else(ep.r,ep.r_hw,ep.wvalid)
    return cached(ep,('ward',role,target),lambda:ward_tree(x,hw,v,target))


def region_graph(ep,role='q',target=512):
    nodes,leaves,roots=tree(ep,role,target);regions=[nodes[l]['support']for l in leaves];centers=np.array([nodes[l]['mean']for l in leaves]);N=len(regions)
    owner=np.full(len(ep.q if role=='q'else ep.r),-1,int)
    for k,C in enumerate(regions):owner[C]=k
    hw=ep.q_hw if role=='q'else ep.r_hw;valid=ep.q_valid if role=='q'else ep.wvalid;i,j=edges4(hw,valid)
    spatial={tuple(sorted((int(owner[a]),int(owner[b]))))for a,b in zip(i,j)if owner[a]!=owner[b]}
    similarity=centers@centers.T;order=np.argsort(-similarity,axis=1,kind='stable');neighbors=[set(row[row!=k][:min(20,N-1)])for k,row in enumerate(order)]
    pairs=spatial|{(a,b)for a in range(N)for b in neighbors[a]if a<b and a in neighbors[b]}
    edges=np.array(sorted(pairs),int).reshape(-1,2)
    return regions,centers,edges,owner


def host_B(ep):
    """Only the frozen original complete FoRIS+MEAN16 host, never old MEAN."""
    def build():
        from . import common
        if hasattr(common,'host_B'):value=common.host_B(ep)
        elif ep.producer.get('synthetic_contract_fixture'):
            value=artifact(ep,'pro30_host_B')
        else:
            operator=artifact(ep,'pro30_B_operator')
            if not callable(operator):raise ArtifactUnavailable('Actual full original FoRIS+MEAN16 B operator required')
            output=operator(ep,reference=None,query=None,apd='native',fixed_native_gate=True)
            if not isinstance(output,tuple)or len(output)!=2:raise ArtifactUnavailable('Source-bound B operator returns(field,execution identity/cost)')
            field,producer=output
            if not isinstance(producer,dict)or producer.get('complete_FoRIS_MEAN16')is not True or producer.get('native_gate_locked')is not True or producer.get('query_GT_read')is not False or not producer.get('source_sha256'):
                raise ArtifactUnavailable('Complete original B execution identity and no-QGT binding required')
            checkpoint=ep.producer.get('model_assets',ep.producer).get('checkpoint_sha256')
            if not checkpoint or producer.get('checkpoint_sha256')!=checkpoint:raise ArtifactUnavailable('Actual B checkpoint differs from native episode')
            producer=dict(producer,pro30_host='complete_original_FoRIS_plus_MEAN16')
            value={'field':np.asarray(field).reshape(ep.q_hw),'producer':producer}
        if isinstance(value,Result):
            value={'field':value.field,'info':value.info}
        if not isinstance(value,dict):raise ArtifactUnavailable('Pro30 M13/M16 require complete original FoRIS+MEAN16 B field and source provenance')
        field=value.get('field',value.get('b'))
        if field is None:raise ArtifactUnavailable('Complete B native continuous field missing')
        field=np.asarray(field,float)
        if field.shape!=tuple(ep.q_hw)or not np.isfinite(field).all():raise ValueError('Complete B field must match native physical query grid')
        producer=value.get('producer',value.get('info',{}))
        if producer.get('pro30_host')!='complete_original_FoRIS_plus_MEAN16':raise ArtifactUnavailable('Host must explicitly bind complete original FoRIS+MEAN16; MEAN/prototype are not substitutes')
        return field.ravel()-.5,producer
    # Import here avoids coupling common to this owner helper.
    from .common import Result
    return cached(ep,'complete_B_host',build)
