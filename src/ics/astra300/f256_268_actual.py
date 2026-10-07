"""Producer-bound observation providers for the original F256/F268 wrappers."""
from __future__ import annotations
import numpy as np
from . import common
from . import f276_300 as K
from . import f_protocol as P
from . import e_views_246_249 as V
from . import e_helpers_226_250 as H
from . import f285_299_native as N
from . import f280_300_observations as O

EPS=1e-6


def _receipt(c,observations):
    episode,owner,callback,qrole=O.backend(c);assets=c.ep.producer.get('model_assets',c.ep.producer)
    return dict(checkpoint_sha256=assets.get('checkpoint_sha256'),config_sha256=assets.get('config_sha256'),
        source_image_hashes=c.ep.producer.get('source_image_hashes'),real_encoder_execution=True,
        actual_receipts=observations,calibration_query_observed_role=qrole,
        frozen_binding=getattr(owner,'binding',{}))


def three_views(c):
    episode,owner,callback,qrole=O.backend(c)
    if c.ep.producer.get('model_input_side')!=1024:raise common.ArtifactUnavailable('F256 requires actual1024 base/flip/+8px views')
    original=N._known_original_target(c)
    if original is None:raise common.ArtifactUnavailable('F256 transformed R mask needs actual known original reference pixels')
    target,known=original;fields={'r':[None]*3,'q':[None]*3};valid=[None]*3;receipts=[]
    basef=c.c*np.asarray(c.ep.wvalid);basev=np.asarray(c.ep.wvalid)
    fields['r'][0]=V._weighted_margin(c.r,c.r,basef,basev);fields['q'][0]=V._weighted_margin(c.q,c.r,basef,basev);valid[0]=c.valid.copy()
    for index,name in ((1,'horizontal_flip'),(2,'translate_x_8')):
        views={}
        for role in ('r','q'):
            observed=qrole if role=='q' else 'r';canvas=V._base_canvas(episode,observed)
            altered=np.ascontiguousarray(canvas[:,::-1]) if index==1 else V._shift(canvas,dx=8)
            raw,receipt=V.encode(episode,observed,altered);receipts.append(dict(role=role,operation=name,**receipt))
            if index==1:raw=raw[:,::-1]
            views[role]=V._unit(raw).reshape(-1,raw.shape[-1])
        if index==1:
            f,v=basef,basev;rg=c.ep.reference_geometry;qg=c.ep.query_geometry
        else:
            rg=V._phase_geometry(c.ep.reference_geometry,0,8);qg=V._phase_geometry(c.ep.query_geometry,0,8)
            A,rv=V._valid_operator(target.shape,c.rhw,rg)
            v=(A@known.ravel().astype(float))*rv;f=(A@(known&target).ravel().astype(float))*rv
        r=V._weighted_margin(views['r'],views['r'],f,v);q=V._weighted_margin(views['q'],views['r'],f,v)
        if r is None or q is None:raise ArithmeticError('F256 transformed legal bank has no two classes')
        if index==2:
            for role,value,g,shape,hw,baseg in (('r',r,rg,target.shape,c.rhw,c.ep.reference_geometry),('q',q,qg,c.ep.original_shape,c.hw,c.ep.query_geometry)):
                original_field=H.native_to_original(value.reshape(hw),shape,g);A=H.footprint(shape,hw,baseg);fields[role][index]=np.asarray(A@original_field.ravel())
            observed=V._observed_pixels(target.shape,rg);A=H.footprint(target.shape,c.rhw,c.ep.reference_geometry)
            valid[index]=c.valid&(np.asarray(A@observed.ravel().astype(float))>=1-1e-6)
        else:fields['r'][index]=r;fields['q'][index]=q;valid[index]=c.valid.copy()
    return dict(real_encoder_execution=True,native_dtype='float32',view_names=['original','horizontal_flip','translate_x_8'],
        reference_margins=np.stack(fields['r'],1).astype(np.float32),query_margins=np.stack(fields['q'],1).astype(np.float32),
        reference_valid=np.stack(valid,1),encoder_receipt=_receipt(c,receipts),
        training_banks_rebuilt_after_buffered_split=True,registered_views_physical=True,additional_views_at_most=4)


def _pair_response(c,role,a,b,refresh=True,mean_control=False):
    n=len(c.r) if role=='r' else c.n;hw=c.rhw if role=='r' else c.hw
    split=np.zeros(n,int);split[a]=1;split[b]=2;merge=split.copy();merge[b]=1
    if mean_control:
        base=c.r if role=='r' else c.q;merged=base.copy();merged[[a,b]]=K.unit(base[[a,b]].mean(0));split_z=base;merge_z=merged;receipts=[]
    else:
        merged,rM=O.tail(c,role,patch_groups=merge,private_special_tokens=True,refresh_qkv=refresh)
        separated,rS=O.tail(c,role,patch_groups=split,private_special_tokens=True,refresh_qkv=refresh)
        merge_z=K.unit(merged);split_z=K.unit(separated);receipts=[rM,rS]
    direction=K.unit(c.r[c.F].mean(0))-K.unit(c.r[c.B].mean(0))
    response=((merge_z[[a,b]]-split_z[[a,b]])@direction)/.1
    return response,receipts


def split_merge(c,control=None):
    reference=O._role_pairs(c,FGFG=4,FGBG=4)
    if not any(kind=='FGFG' for _,kind in reference) or not any(kind=='FGBG' for _,kind in reference):
        raise ArithmeticError('F268 actual reference has no known FGFG/FGBG paired blocks')
    source=[];target=[];receipts=[]
    for (a,b),kind in reference[:8]:
        response,r=_pair_response(c,'r',a,b,refresh=control!='fixed_QK',mean_control=control=='group_mean');receipts+=r;source.append(response);target.append(kind=='FGFG')
    disagreement=np.zeros(c.n)
    for i in range(len(c.Q)):
        for j in range(i):disagreement+=c.Q[i]!=c.Q[j]
    order=sorted(range(len(c.edges)),key=lambda j:(-float(disagreement[c.edges[j]].sum()),tuple(c.edges[j])))
    query=[];responses=[];used=set()
    for j in order:
        a,b=map(int,c.edges[j])
        if a in used or b in used:continue
        response,r=_pair_response(c,'q',a,b,refresh=control!='fixed_QK',mean_control=control=='group_mean');receipts+=r;query.append((a,b));responses.append(response);used|={a,b}
        if len(query)>=4:break
    episode,owner,callback,qrole=O.backend(c);session=owner.observe(episode,qrole);weights=[]
    for a,b in query:
        values=[]
        for layer in (23,24):
            ab=session.attention[layer].weights(np.array([a+5]),np.array([b+5]))[:,0,0];ba=session.attention[layer].weights(np.array([b+5]),np.array([a+5]))[:,0,0];values.append(np.median((ab+ba)/2))
        weights.append(float(np.mean(values)))
    return dict(real_encoder_execution=True,qkv_recomputed=control not in ('fixed_QK','group_mean'),private_special_tokens=True,tail_blocks=2,native_dtype='float32',
        reference_responses=np.asarray(source,np.float32),reference_merge_role=np.asarray(target,bool),
        query_responses=np.asarray(responses,np.float32).reshape(-1,2),query_edges=np.asarray(query,int).reshape(-1,2),
        ordinary_attention_edge_weights=np.asarray(weights,np.float32),encoder_receipt=_receipt(c,receipts),
        singleton_native_patch_blocks=True,other_partition_identical=True,actual_component_block_count=sum(r['actual_tail_blocks'] for r in receipts),
        maximum_logical_two_block_replays=24,control=control)


def configure_resources(ep,number,extra=None):
    resources=dict(extra or {});resources['bound_episode']=ep
    resources['tail_replay']=common.artifact(ep,'tail_replay')
    if number==256:resources['frozen_encode_rgb']=common.artifact(ep,'frozen_encode_rgb');resources['f256_provider']=three_views
    elif number==268:resources['f268_provider']=split_merge
    else:raise ValueError('Only F256/F268 actual observer providers')
    if ep.reference_mask is not None:resources['reference_training_pixels']=common.readonly(ep.reference_mask)
    return resources
