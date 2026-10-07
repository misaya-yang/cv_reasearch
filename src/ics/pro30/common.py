"""Pro30 shared B_R and explicit CPU100 versus continuous-highres rendering.

The Episode/provider is Astra-compatible. M02--M10 preserve the historical
two-threshold CPU100 path; this is deliberately distinct from highres outputs.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
import numpy as np

from ics.astra300.common import (Episode,Result,ArtifactUnavailable,as_episode,
    load_episode,validate,artifact,require_artifact,require_artifacts,
    array_hash,readonly,continuous_original)
from ics.cpu100.common import Result as CPUResult,render as cpu100_render
from ics.cpu100.invariance_support import _weighted_ridge,_huber_objective,_mm

EPS=1e-12
BR_CONSTANTS={'equal_mass_positions_per_role':128,'ridge':.01,'Huber_delta':.5,
    'IRLS_rounds':12,'bias_penalized':False,'soft_target':'2wf/wvalid-1',
    'role_sample_weights':'.5wf/sum(samplewf)+.5wb/sum(samplewb)'}
_BR_CACHE=OrderedDict()
_PCA_CACHE=OrderedDict()
CARD_SHA256={'M01': '1ccf3811cdac760368b197a576547e6b44e73dc0d5a50cd8b998b73a6d64f28e', 'M02': 'aa86db0333729d1bb3cc0d478c272026fed8d22a5886e2131c1ab95689cea5d0', 'M03': '636b6b20b4b6fa491054548df6da78bd109566d29ac81dcb532c75236fc9926d', 'M04': '5cbeb1287e49ebff7ccd2499a81631dc935ff8691cc948672e4868005a0c08a7', 'M05': '12891173082912bb4d1e599966cfccdc765b4668ba7ac2307ac7845636847d10', 'M06': 'f780090f4b75ee16fa29da5890705580d7cd418794b7a3bddb2104f14b062533', 'M07': 'be306acd43e5be0296df155b8db3fac71ced5039f3d373bf0c09852dcda318e5', 'M08': 'f636266085f12ab13d332d691a1f45b7dc4104d7c694591dca3539931212c9f4', 'M09': '6066c0140d4491b08de0776163513692f46ff89309387a6e0f004e84268aa961', 'M10': '6441be407718b6eb3b814203ef3d6d0a4e038d139ee6c471fb391fbdbf52d96d', 'M11': '9bbc7808d1c18d9db449ac1b9bcf0ff94af6f783c4e0c4fb3e8bb852f9f5c2f4', 'M12': 'aa523b714cafcc366df03560825bb3b5c0850197a6d4ac8b4aff3648e88aec8a', 'M13': 'd83f98e05bbf52a02ba9574bd16194d32daf27cad53ad36fca86b46bfaff665e', 'M14': '76dc4bf261d417449512d29631f09f5a94ad0f23342a6a7f691ba81691b4c677', 'M15': 'e3ceaaa7f4104163230d463e32f216882d069241e63dee368ff1908e1687a895', 'M16': 'c781cb00f7767f87e83daebc7fd0ad2ab88a60e55e48d65b6fc3eab655344bb3', 'M17': 'ad49f0ff6c0959fce87fb9ecfde5b54b5d5f27f72fffe7069a7057eda6b1d388', 'M18': '18615e2ab4c8bd3a2a654f4806e7548a7266c04dcca98becd6bd7c92bcac6806', 'M19': 'bfb719c5eaab46880b126cb39d47f59515d835a064acc2759d294cfb0fb8753d', 'M20': 'd41e06dd2d3a73964e026aa782b8bb3e5b80f1819597dac598fc45126feafed4', 'M21': '7c9f5fc542a2d7be9738a386c7b121e56924584b5567011e928652d5ddc06b7d', 'M22': '713dc263a4b37874d950345c3c27d72c0f62fac9815599c73cb38e273516ca2a', 'M23': '8aedee20331dfd30709b53d304cb99962f6b1b0bd0edc8a8548a11ae018189af', 'M24': 'e4e41fcb247ec87ffa7e384fc7d8a1fdd20cb019c174c763b74b6eb68bff2d24', 'M25': '74f5ab29a7f8fa555ef0bbc41d89862d3976214b0b49333d783f4858db795766', 'M26': '053ef3c570a14c2e8970d8052ded0c744149062cf0317b75def9a7c26d30e38a', 'M27': 'e959f88240b6d3d070ac7088e61d5da431a6fdf44999713c3f1dfde097c31e79', 'M28': '9f4a2eb9716b6fbfc7c67507048f5d4b1b9fbca303457d6201e1324ad68d6412', 'M29': '6faed0c2f9db3f0d9826d9997b80eb1b3a183a1194c3dab2b9bcdc910a0a2898', 'M30': '24a94e5fc5b3a58b2a3a3202cc34ef7bc7c8d18cfe58f60b5a97da5077193ade'}


def unit(x):
    x=np.asarray(x,float)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-6)


def blocks(hw):
    yy,xx=np.indices(hw)
    return ((yy>=hw[0]/2)*2+(xx>=hw[1]/2)).ravel().astype(int)


def joint_basis(ep,dimension=16):
    """Unwhitened joint valid-token PCA, preceded by the source class direction."""
    from scipy.linalg import eigh
    validate(ep);start=time.perf_counter()
    key=(ep.source_id,array_hash(ep.r),array_hash(ep.q),array_hash(ep.wvalid),array_hash(ep.q_valid))
    hit=key in _PCA_CACHE
    if hit:
        axes,pca_info=_PCA_CACHE.pop(key);_PCA_CACHE[key]=(axes,pca_info)
    else:
        x=np.r_[ep.r[ep.wvalid>0],ep.q[ep.q_valid>0]];center=x.mean(0);z=x-center
        covariance=z.T@z/max(len(z),1);count=min(31,covariance.shape[0])
        values,vectors=eigh(covariance,subset_by_index=[covariance.shape[0]-count,covariance.shape[0]-1])
        axes=vectors[:,::-1];axes=axes[:,values[::-1]>max(float(values.max()),1e-12)*1e-12]
        pca_info=dict(PCA_population='all physical valid R/Q token observations, unweighted and unwhitened',
            PCA_dimension=axes.shape[1],PCA_seconds=time.perf_counter()-start)
        _PCA_CACHE[key]=(axes,pca_info)
        while len(_PCA_CACHE)>4:_PCA_CACHE.popitem(last=False)
    delta=np.sum(ep.r*ep.wf[:,None],axis=0)/max(ep.wf.sum(),EPS)-np.sum(ep.r*ep.wb[:,None],axis=0)/max(ep.wb.sum(),EPS)
    candidates=[delta]+list(axes[:,:max(0,dimension-1)].T);basis=[]
    for vector in candidates:
        value=vector.copy()
        # Twice-reorthogonalized Gram-Schmidt preserves the first source axis.
        for _ in range(2):
            if basis:value-=np.array(basis).T@(np.array(basis)@value)
        norm=np.linalg.norm(value)
        if norm>1e-10:basis.append(value/norm)
        if len(basis)>=dimension:break
    u=np.array(basis).T if basis else np.empty((ep.r.shape[1],0))
    return u,dict(pca_info,PCA_cache_hit=hit,basis_dimension=u.shape[1],basis_seconds_this_call=time.perf_counter()-start)


def equal_mass_sample(ep):
    """Exactly the historical 128 positions per role, stable native order."""
    ids=[]
    for weights in (ep.wf,ep.wb):
        cumulative=np.cumsum(weights)
        if cumulative[-1]<=0:raise ValueError('B_R two-role fit requires positive source mass')
        targets=(np.arange(128)+.5)*cumulative[-1]/128
        ids.extend(np.searchsorted(cumulative,targets,side='left'))
    ids=np.unique(ids);ids=ids[ep.wvalid[ids]>0]
    target=2*ep.wf[ids]/ep.wvalid[ids]-1
    weights=.5*ep.wf[ids]/ep.wf[ids].sum()+.5*ep.wb[ids]/ep.wb[ids].sum()
    return ids,target,weights


@dataclass
class Readout:
    coefficient:np.ndarray
    bias:float
    info:dict
    @property
    def coef(self):return self.coefficient
    def predict(self,x):return _mm(np.asarray(x),self.coefficient)+self.bias


def fit_br(ep,reference=None,*,huber=True):
    """B_R exact source contract; arbitrary legal auxiliary features supported."""
    validate(ep)
    r=np.asarray(ep.r if reference is None else reference,float)
    if r.ndim!=2 or len(r)!=len(ep.r) or not np.isfinite(r).all():
        raise ValueError('Reference readout descriptors must align with native R positions')
    ids,y,weights=equal_mass_sample(ep)
    key=(ep.source_id,array_hash(r),array_hash(ep.wf),array_hash(ep.wvalid),huber)
    if key in _BR_CACHE:
        fitted=_BR_CACHE.pop(key);_BR_CACHE[key]=fitted
        return Readout(fitted.coefficient,fitted.bias,dict(fitted.info,fit_cache_hit=True,fit_seconds_this_call=0.))
    start=time.perf_counter();x=r[ids]
    coefficient,bias=_weighted_ridge(x,y,weights,strength=.01)
    objectives=[_huber_objective(x,y,weights,coefficient,bias)]
    if huber:
        for _ in range(12):
            residual=_mm(x,coefficient)+bias-y
            robust=weights*np.minimum(1,.5/np.maximum(np.abs(residual),1e-12))
            coefficient,bias=_weighted_ridge(x,y,robust,strength=.01)
            objectives.append(_huber_objective(x,y,weights,coefficient,bias))
    if not np.isfinite(coefficient).all() or not np.isfinite(bias):raise ValueError('Nonfinite Pro30 B_R readout')
    info=dict(contract='Pro30_B_R_exact_CPU100_reference_fit',constants=BR_CONSTANTS,
        fit_sample_ids=ids.tolist(),fit_tokens=len(ids),descriptor_dimension=r.shape[1],
        objective_history=objectives,Huber=huber,optimization_data='current known R labels only',
        fit_cache_hit=False,fit_seconds_this_call=time.perf_counter()-start)
    fitted=Readout(readonly(coefficient),float(bias),info);_BR_CACHE[key]=fitted
    while len(_BR_CACHE)>16:_BR_CACHE.popitem(last=False)
    return fitted


def degenerate_margin(ep):
    """Source §2.1: empty MR or explicit single-class cosine support."""
    validate(ep)
    if ep.wf.sum()<=0:
        return np.full(len(ep.q),-1.),{'degeneration':'empty_reference_target','valid_target_prompt':False}
    if ep.wb.sum()>0:return None
    mean=unit(np.sum(ep.r*ep.wf[:,None],axis=0))
    c=np.divide(ep.wf,ep.wvalid,out=np.zeros_like(ep.wf,float),where=ep.wvalid>0)
    ids=np.flatnonzero((ep.wvalid>0)&(c>=.9));partition=blocks(ep.r_hw)
    values=[];effective=0
    for block in range(4):
        held=ids[partition[ids]==block];source=ids[partition[ids]!=block]
        if len(held) and len(source):
            source_mean=unit(np.sum(ep.r[source]*ep.wf[source,None],axis=0))
            values.extend((ep.r[held]@source_mean).tolist());effective+=1
    cut=float(np.quantile(values,.1)) if effective>=2 else .5
    margin=ep.q@mean-cut;margin=np.asarray(margin);margin[ep.q_valid<=0]=-1.
    return margin,dict(degeneration='full_foreground_reference_single_class_support',
        threshold=cut,threshold_rule='pureFG leave-spatial-block cosine q.10; <2 blocks fixed.5',
        effective_reference_blocks=effective,rule_quality='unverified',negative_evidence_disabled=True)


def br_margin(ep,reference=None,query=None,*,huber=True):
    degeneration=degenerate_margin(ep)
    if degeneration is not None:return degeneration
    fitted=fit_br(ep,reference,huber=huber)
    q=np.asarray(ep.q if query is None else query,float)
    if q.ndim!=2 or len(q)!=len(ep.q) or q.shape[1]!=len(fitted.coefficient) or not np.isfinite(q).all():
        raise ValueError('Query readout descriptors must align with original Q positions and fitted channels')
    score=np.empty(len(q))
    for start in range(0,len(q),256):score[start:start+256]=fitted.predict(q[start:start+256])
    score[ep.q_valid<=0]=-1.
    return score,fitted.info


def finish(ep,margin,method_id,info=None,*,invalid_value=None):
    """M02--M10 exact CPU100 two-threshold output, returned as Astra mask."""
    start=time.perf_counter();field=np.asarray(margin,float).reshape(ep.q_hw).copy()
    if invalid_value is not None:field.ravel()[ep.q_valid<=0]=invalid_value
    if not np.isfinite(field).all():raise ValueError('Complete finite signed field required')
    output=cpu100_render(ep,CPUResult(field,{}))
    metadata=dict(info or {},method_id=method_id,query_GT_read=False,
        renderer='CPU100: physical64 FP32→bilinear1024 >0→binary bilinear original >.5',
        renderer_seconds=time.perf_counter()-start,work_mask_shape=[1024,1024],original_mask_shape=list(ep.original_shape),
        complete_all_instances=True)
    return Result(field,0.,readonly(output['original']),metadata)


def finish_highres(ep,field,method_id,info=None,*,threshold=0.,field_space='native_canvas'):
    """Explicit one-continuous-threshold branch; NEVER compress a fine field64."""
    start=time.perf_counter();field=np.asarray(field,float)
    if field.ndim!=2 or not np.isfinite(field).all():raise ValueError('Finite physical high-resolution field required')
    if field_space=='original':
        if field.shape!=tuple(ep.original_shape):raise ValueError('Original-space field shape differs')
        original=field
    elif field_space=='native_canvas':original=continuous_original(ep,field)
    else:raise ValueError('Highres field must declare native_canvas or original')
    metadata=dict(info or {},method_id=method_id,query_GT_read=False,
        renderer='continuous physical highres→original then single strict threshold; no64 compression',
        renderer_seconds=time.perf_counter()-start,output_field_hw=list(field.shape),complete_all_instances=True)
    if field_space=='original':metadata['field_space']='original'
    return Result(field,threshold,readonly(original>threshold),metadata)


def br_result(ep,method_id='PRO30_B_R',*,huber=True):
    margin,info=br_margin(ep,huber=huber)
    return finish(ep,margin,method_id,info)


def source_contract(mid):
    """Frozen original-card identity for concise callable manifests."""
    from pathlib import Path
    import hashlib
    path=Path(__file__).resolve().parents[3]/'evidence/local/pro30_20261007/source'/('M'+str(mid).zfill(2)+'.md')
    if not path.is_file():return {'original_method':'M'+str(mid).zfill(2),'card_sha256':CARD_SHA256['M'+str(mid).zfill(2)],'full_source_sha256':'56caf05b4b063147b598df85ca67d98f503c7c73440df61c5c8530ac02374729'}
    return {'original_method':'M'+str(mid).zfill(2),'card_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'full_source_sha256':'56caf05b4b063147b598df85ca67d98f503c7c73440df61c5c8530ac02374729'}
