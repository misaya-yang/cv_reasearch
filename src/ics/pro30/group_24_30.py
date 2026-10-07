"""Original Pro30 M24--M30; actual added resources are mandatory.

No original300 substitution, no synthetic producer acceptance, no encoder load,
no parameter search. Full B is a bound operator, not a supplied score shortcut.
"""
from functools import partial
from dataclasses import replace
import time
import numpy as np
from scipy import sparse
from scipy.special import expit
from ics.methods.direct_dino_features import resize,sample_grid
from ics.cpu100.common import rgb_view
from . import common
from . import operators_24_30 as op
from .eva_24_30 import runtime,F_unit,asset_header

BUDGET_SECONDS=2.
METHODS={};CONTROLS={};CONTRACTS={};REQUIREMENTS={}


def _runtime(ep):
    rt=runtime(ep);rt._pro30_call_start=dict(rt.stats);return rt


def _episode(ep):
    common.validate(ep)
    if ep.q.dtype not in (np.dtype('float32'),np.dtype('float64')) or ep.r.dtype!=ep.q.dtype:
        raise common.ArtifactUnavailable('FP32 native patches or explicit FP64 unit export required; no FP16')
    if (ep.producer.get('kind') not in {'frozen_DINOv3_FP32_native_final_LN_patches','frozen_DINOv3_FP32_native_final_LN_patches_then_unit'}
            or ep.producer.get('FoRIS_Part1_applied') is not False or ep.q_hw!=(64,64) or ep.r_hw!=(64,64)
            or ep.q.shape[1]!=1024 or ep.producer.get('model_input_side')!=1024):
        raise common.ArtifactUnavailable('Original Pro30 internal cards need actual native DINOv3-L/16 1024, not processed or synthetic fixtures')
    mask=ep.reference_mask if ep.reference_mask is not None else common.artifact(ep,'reference_mask')
    if not ep.reference_geometry or not ep.query_geometry:raise common.ArtifactUnavailable('Actual physical resize/pad geometry required')
    if ep.r_rgb is not None and np.asarray(mask).shape!=ep.r_rgb.shape[:2]:raise ValueError('MR and original R RGB dimensions differ')
    if ep.reference_geometry.get('original_hw') is not None and tuple(ep.reference_geometry['original_hw'])!=tuple(np.asarray(mask).shape):
        raise ValueError('MR pixels differ from recorded physical original geometry')
    area,valid=op.exact_footprint(mask,ep.r_hw,ep.reference_geometry)
    g=ep.query_geometry;side=float(g['view_side']);oy,ox=g['padding_top_left'];sh,sw=g['resized_hw']
    if side!=1024 or min(sh,sw)<=0 or min(oy,ox)<0 or oy+sh>side or ox+sw>side:raise ValueError('Recorded native1024 query geometry required')
    edges=np.arange(65)*16
    vy=np.maximum(0,np.minimum(edges[1:],oy+sh)-np.maximum(edges[:-1],oy))/16
    vx=np.maximum(0,np.minimum(edges[1:],ox+sw)-np.maximum(edges[:-1],ox))/16
    ep=replace(common.as_episode(ep),wf=area,wvalid=valid,q_valid=(vy[:,None]*vx[None,:]).ravel())
    return ep,op.source_split(ep.r_hw,area,valid)


def _B(ep,reference=None,query=None,*,apd='native',fixed_native_gate=True):
    """Provider must execute the actual complete original FoRIS + MEAN16."""
    operator=common.artifact(ep,'pro30_B_operator')
    if not callable(operator):raise common.ArtifactUnavailable('A bound complete B operator is required, not a cached score array')
    output=operator(ep,reference=reference,query=query,apd=apd,fixed_native_gate=fixed_native_gate)
    if not isinstance(output,tuple) or len(output)!=2:raise common.ArtifactUnavailable('B operator must return (b64, producer/cost info)')
    field,info=output;field=np.asarray(field,float).reshape(64,64)
    if (not np.isfinite(field).all() or not isinstance(info,dict) or info.get('complete_FoRIS_MEAN16') is not True
            or info.get('query_GT_read') is not False or not info.get('source_sha256') or not info.get('checkpoint_sha256')):
        raise common.ArtifactUnavailable('Full B execution identity, no QGT and fixed source binding required')
    expected=ep.producer.get('model_assets',ep.producer).get('checkpoint_sha256')
    if info['checkpoint_sha256']!=expected:raise common.ArtifactUnavailable('B checkpoint differs from the episode')
    if fixed_native_gate and info.get('native_gate_locked') is not True:raise common.ArtifactUnavailable('Changed encoder B requires the original native APD gate fixed')
    if apd is False and info.get('APD_applied') is not False:raise common.ArtifactUnavailable('This branch explicitly disables APD')
    return field,info


def _end(ep,field,mid,info,start,rt=None,threshold=0.):
    elapsed=time.perf_counter()-start
    metadata=dict(info,source=common.source_contract(int(mid.split('__')[0][-2:])),method_wall_seconds=elapsed,
                  cached_CPU_budget_seconds=BUDGET_SECONDS,cost_gate_pass=elapsed<=BUDGET_SECONDS,
                  authorize_fixed600=False,natural_quality='unknown',original300_count_increment=0,
                  all_instances=True,parameter_search=False)
    if rt is not None:
        metadata['actual_encoder_operations_in_bound_runtime']=dict(rt.stats)
        before=getattr(rt,'_pro30_call_start',{})
        metadata['actual_encoder_operations_this_call']={key:rt.stats[key]-before.get(key,0) for key in rt.stats if isinstance(rt.stats[key],(int,float)) and not isinstance(rt.stats[key],bool)}
        metadata['original_native_input_extraction_cost_is_separate']=True
        metadata['cold_end_to_end_requires_original_input_and_all_declared_extra_operations']=True
    field=np.asarray(field)
    if field.shape==(64,64):
        metadata['source_renderer_parity']='shared exact CPU100 physical64→1024 strict threshold→binary-original'
        return common.finish(ep,field-threshold,mid,metadata)
    # The shared owner supplies the fine branch, without compressing128 to64.
    # Explicit review hold remains for the original card's1024/binary-original
    # wording versus the present shared continuous-original fine branch.
    metadata['source_renderer_parity']='fine branch needs common-owner work1024/binary-original parity review'
    return common.finish_highres(ep,field,mid,metadata,threshold=threshold)


def _degenerate(ep,mid,start):
    out=common.degenerate_margin(ep)
    if out is None:return None
    margin,info=out
    return _end(ep,margin.reshape(64,64),mid,dict(info,inactive=True),start)


def _numpy(session):return session['features'][0].detach().cpu().numpy().copy()


def _same_data_control(ep,mid,callback):
    operator=common.artifact(ep,'pro30_same_budget_controls')
    if not callable(operator):raise common.ArtifactUnavailable('Actual same-budget '+callback+' executor required')
    result=operator(ep,callback)
    if not isinstance(result,common.Result) or result.info.get('actual_same_budget_resources') is not True:
        raise common.ArtifactUnavailable('A supplied score or cheaper proxy cannot stand in for the full same-budget control')
    result.info.update(method_id=mid,counts_as_method=False,source=common.source_contract(int(mid.split('__')[0][-2:])))
    return result


def m26(ep,*,mode='neutral',apd=False,control=None):
    start=time.perf_counter();ep,split=_episode(ep);mid='PRO30_M26'+('' if control is None else '__'+control)
    degenerate=_degenerate(ep,mid,start)
    if degenerate is not None:return degenerate
    if control=='same_cost_native_multiview':return _same_data_control(ep,mid,'M26_native_multiview_measured_equal_cost')
    if mode=='native':
        b,host=_B(ep,apd=apd);return _end(ep,b-.5,mid,dict(host=host,attention_mode=mode,APD=apd,identity_control=control=='prefix_zero_identity'),start)
    rt=_runtime(ep);prefix_rope=None
    if mode=='vit5_prefix':
        prefix_rope,producer=common.require_artifacts(ep,'pro30_independent_prefix_rope','pro30_independent_prefix_rope_producer')
        if producer.get('checkpoint_sha256')!=rt.binding['model_assets']['checkpoint_sha256'] or producer.get('independent_coordinates_and_frequency') is not True:
            raise common.ArtifactUnavailable('Actual declared ViT5-style frozen prefix coordinate/frequency control asset required')
        prefix_rope=rt.torch.as_tensor(np.asarray(prefix_rope),dtype=rt.torch.float32)
    r=_numpy(rt.native('r',mode=mode,prefix_rope=prefix_rope));q=_numpy(rt.native('q',mode=mode,prefix_rope=prefix_rope))
    b,host=_B(ep,r,q,apd=apd)
    return _end(ep,b-.5,mid,dict(host=host,attention_mode=mode,APD=apd,modified_layers=list(range(1,25)),
        exact_3d_QK_scale='1/sqrt(original_head_dim)',prefix_keys_retained=True,phase_claim='coordinate-origin shift only; not image/crop invariance',
        matched_prefix_attenuation_certificates=getattr(rt,'prefix_mass_certificates',None) if mode=='prefix_attenuation' else None),start,rt)


def _torch_loss(rt,features,w,area,valid,ids):
    torch=rt.torch;ids=torch.as_tensor(ids,dtype=torch.bool);area=torch.as_tensor(area,dtype=features.dtype);valid=torch.as_tensor(valid,dtype=features.dtype)
    logits=(features[0]@w)/.07;a=area[ids];b=(valid-area)[ids]
    return (a*torch.nn.functional.softplus(-logits[ids])).sum()/a.sum()+(b*torch.nn.functional.softplus(logits[ids])).sum()/b.sum()


def m24(ep,*,control=None):
    start=time.perf_counter();ep,split=_episode(ep);mid='PRO30_M24'+('' if control is None else '__'+control)
    degenerate=_degenerate(ep,mid,start)
    if degenerate is not None:return degenerate
    b0,base_info=_B(ep)
    if not split.adequate:return _end(ep,b0-.5,mid,dict(host=base_info,inactive=True,fallback='A/C_support_insufficient',split=split.info),start)
    w=op.direction(ep.r,ep.wf,ep.wvalid,split.anchor,normalize=False)
    group_ids=[split.anchor&(split.group==k) for k in (0,1)]
    calibration=[split.calibration&(split.group==k) for k in (2,3)]
    if w is None or any(min(ep.wf[ids].sum(),ep.wb[ids].sum())<=0 for ids in group_ids+calibration):
        return _end(ep,b0-.5,mid,dict(host=base_info,inactive=True,fallback='zero_direction_or_missing_group_role'),start)
    rt=_runtime(ep);torch=rt.torch;wt=torch.as_tensor(w,dtype=torch.float32)
    heads=int(rt.model.blocks[-1].attn.num_heads);gates=torch.ones((4,heads),requires_grad=True)
    rs=rt.native('r',need_gradient=True,gates=gates,head_direction=wt);qs=rt.native('q',head_direction=wt)
    gradients=[]
    for index,ids in enumerate(group_ids):
        loss=_torch_loss(rt,rs['features'],wt,ep.wf,ep.wvalid,ids)
        gradients.append(rt.vjp(loss,gates,retain_graph=index==0).detach().numpy())
    candidate,selected=op.select_heads(gradients)
    if not selected:return _end(ep,b0-.5,mid,dict(host=base_info,inactive=True,fallback='no_consistent_positive_head_gradient',gradient_groups=np.asarray(gradients).tolist()),start,rt)
    if control=='random_heads_mean5':
        fields=[];details=[]
        for seed in range(5):
            ids=np.random.RandomState(seed).choice(4*heads,len(selected),replace=False);g=np.ones((4,heads));g.ravel()[ids]=.8
            r=_numpy(dict(features=rt.suffix(rs['h20'].detach(),rs,gates=torch.as_tensor(g,dtype=torch.float32))))
            q=_numpy(dict(features=rt.suffix(qs['h20'].detach(),qs,gates=torch.as_tensor(g,dtype=torch.float32))))
            b,host=_B(ep,r,q);fields.append(b);details.append(dict(seed=seed,heads=ids.tolist(),host=host))
        return _end(ep,np.mean(fields,axis=0)-.5,mid,dict(diagnostic_only=True,five_fixed_seeds_not_selected_by_QGT=details),start,rt)
    if control=='absolute_gradient':
        chosen=np.argsort(-np.abs(np.asarray(gradients)).mean(axis=0).ravel(),kind='stable')[:len(selected)]
        candidate=np.ones_like(candidate);candidate.ravel()[chosen]=.8
    if control=='uniform_equal_energy':
        # Match actual R+Q AV squared energy, not an unweighted count of gates.
        energy=np.stack([rs['heads'][index]['energy'].numpy()+qs['heads'][index]['energy'].numpy() for index in range(21,25)])
        ratio=float(np.sum(candidate**2*energy)/max(float(energy.sum()),1e-12))
        candidate=np.full_like(candidate,np.sqrt(ratio))
    if control=='output_norm':
        energy=np.stack([rs['heads'][index]['energy'].numpy() for index in range(21,25)])
        chosen=np.argsort(-energy.ravel(),kind='stable')[:len(selected)]
        candidate=np.ones_like(candidate);candidate.ravel()[chosen]=.8
    if control=='readonly_head_linear':
        rmargin=op.dot(ep.r,w)[:,None];qmargin=op.dot(ep.q,w)[:,None]
        rheads=np.stack([rs['heads'][layer]['response'][0,head].numpy() for layer,head in selected],axis=1)
        qheads=np.stack([qs['heads'][layer]['response'][0,head].numpy() for layer,head in selected],axis=1)
        residual,readout=op.residual_readout(np.c_[rmargin,rheads],np.c_[qmargin,qheads],rmargin,qmargin,ep.wf,ep.wvalid,split)
        return _end(ep,b0-.5+residual.reshape(64,64),mid,dict(host=base_info,selected_heads=selected,readout=readout,actual_native_AV_projection_only=True,encoder_unchanged=True),start,rt)
    if control=='gate1':candidate=np.ones_like(candidate)
    rr=rt.suffix(rs['h20'].detach(),rs,gates=torch.as_tensor(candidate,dtype=torch.float32));r=_numpy(dict(features=rr))
    original_losses=[op.balanced_loss(op.dot(ep.r,w)/.07,ep.wf,ep.wvalid,ids) for ids in calibration]
    candidate_losses=[op.balanced_loss(op.dot(r,w)/.07,ep.wf,ep.wvalid,ids) for ids in calibration]
    accept=bool(all(new<=old for new,old in zip(candidate_losses,original_losses)) and np.mean(original_losses)-np.mean(candidate_losses)>=.01)
    # Diagnostic controls execute their explicit intervention; primary alone uses C acceptance.
    execute=accept or control is not None
    info=dict(selected_heads=selected,gate_values=candidate.tolist(),gradient_groups=np.asarray(gradients).tolist(),
              calibration_native_losses=original_losses,calibration_modified_losses=candidate_losses,accepted=accept,control=control,split=split.info,
              direction_A_frozen=True,APD_native_gate_locked=True)
    if not execute:return _end(ep,b0-.5,mid,dict(info,host=base_info,inactive=True,fallback='C_head_intervention_rejected'),start,rt)
    qq=rt.suffix(qs['h20'].detach(),qs,gates=torch.as_tensor(candidate,dtype=torch.float32));q=_numpy(dict(features=qq));b,host=_B(ep,r,q)
    return _end(ep,b-.5,mid,dict(info,host=host,inactive=False),start,rt)


def _attribution(rt,session,w,valid,*,points=(1/6,1/2,5/6)):
    torch=rt.torch;hidden=session['h20'].detach();patch=hidden[:,5:];valid=torch.as_tensor(valid>0)
    baseline=patch[:,valid].mean(dim=1,keepdim=True);delta=patch-baseline;ap=0.;am=0.;ordinary=[]
    for point in points:
        # Independent leaf: path t is never differentiated or multiplied twice.
        leaf=torch.cat((hidden[:,:5],baseline+point*delta),dim=1).detach().requires_grad_(True)
        features=rt.suffix(leaf,session);ordinary.append(features[0].detach().numpy());logit=features[0]@w/.07
        positive=torch.nn.functional.softplus(logit[valid]).mean();negative=torch.nn.functional.softplus(-logit[valid]).mean()
        gp=rt.vjp(positive,leaf,retain_graph=True)[:,5:];gm=rt.vjp(negative,leaf)[:,5:]
        ap=ap+(delta*gp).sum(-1)/len(points);am=am+(delta*gm).sum(-1)/len(points)
    return ap[0].detach().numpy(),am[0].detach().numpy(),ordinary


def m25(ep,*,control=None):
    start=time.perf_counter();ep,split=_episode(ep);mid='PRO30_M25'+('' if control is None else '__'+control)
    degenerate=_degenerate(ep,mid,start)
    if degenerate is not None:return degenerate
    b,host=_B(ep);w=op.direction(ep.r,ep.wf,ep.wvalid,split.anchor) if split.adequate else None
    if w is None:return _end(ep,b-.5,mid,dict(host=host,inactive=True,fallback='A/C_insufficient_or_zero_direction'),start)
    rt=_runtime(ep);wt=rt.torch.as_tensor(w,dtype=rt.torch.float32);descriptors=[]
    for role,features,valid in [('r',ep.r,ep.wvalid),('q',ep.q,ep.q_valid)]:
        session=rt.native(role);ap,am,ordinary=_attribution(rt,session,wt,valid,points=(1.,) if control=='gradient_times_input' else (1/6,1/2,5/6))
        margin=op.dot(features,w);simple=margin[:,None]
        if control=='linear_logit_IG':enhanced=np.c_[margin,ap-am]
        elif control=='even_amplitude_IG':enhanced=np.c_[margin,ap+am]
        elif control=='ordinary_feature_mean':enhanced=np.c_[margin,op.dot(np.mean(ordinary,axis=0),w)]
        else:enhanced=np.c_[margin,ap,am,ap*am]
        descriptors.append((enhanced,simple))
    if control in ('attention_rollout','block20_direct'):return _same_data_control(ep,mid,'M25_'+control+'_actual_same_budget')
    residual,readout=op.residual_readout(descriptors[0][0],descriptors[1][0],descriptors[0][1],descriptors[1][1],ep.wf,ep.wvalid,split)
    return _end(ep,b-.5+residual.reshape(64,64),mid,dict(host=host,readout=readout,control=control,
        integration_points=[1/6,1/2,5/6] if control!='gradient_times_input' else [1.],prefix_fixed=True,independent_path_leaves=True,
        exact_completeness_claim=False,two_directions_not_independent_class_roles=True,descriptor_directions_fixed_from_A=True),start,rt)


def _scaled_canvas(ep,role,side):
    from PIL import Image
    return np.asarray(Image.fromarray(rgb_view(ep,role)).resize((side,side),Image.Resampling.BILINEAR))


def m27(ep,*,control=None):
    start=time.perf_counter();ep,split=_episode(ep);mid='PRO30_M27'+('' if control is None else '__'+control)
    degenerate=_degenerate(ep,mid,start)
    if degenerate is not None:return degenerate
    rt=_runtime(ep)
    # Inspect the actual asset before any of ten added512 forwards, even on fallback.
    token,token_info=rt.trained_mask_token(ep);b,host=_B(ep)
    if not split.adequate:return _end(ep,b-.5,mid,dict(host=host,inactive=True,fallback='A/C_insufficient',mask_asset=token_info),start,rt)
    if control in ('mean_token','token_permutation','masked_uniform_average','unmasked_positions_TTA'):
        return _same_data_control(ep,mid,'M27_'+control+'_same10views_and512geometry')
    yy,xx=np.indices((32,32));group=(yy+2*xx)%4;descriptors=[]
    for role,native in [('r',ep.r),('q',ep.q)]:
        canvas=_scaled_canvas(ep,role,512);ordinary=rt.encode(role,canvas);context=np.zeros_like(ordinary)
        for k in range(4):
            ids=(group==k).ravel();masked=rt.encode(role,canvas,mask=ids,mask_token=token);context[ids]=masked[ids]
        oo=common.unit(resize(ordinary.reshape(32,32,-1),(64,64)).reshape(4096,-1));cc=common.unit(resize(context.reshape(32,32,-1),(64,64)).reshape(4096,-1))
        descriptors.append((native,oo,cc,ordinary,context))
    # 512 direction uses exact mass contributed by A's 16px physical cells only.
    f=(ep.wf*split.anchor).reshape(32,2,32,2).mean((1,3)).ravel();v=(ep.wvalid*split.anchor).reshape(32,2,32,2).mean((1,3)).ravel()
    mask=ep.reference_mask if ep.reference_mask is not None else common.artifact(ep,'reference_mask')
    geometry=dict(ep.reference_geometry,view_side=512,resized_hw=[x/2 for x in ep.reference_geometry['resized_hw']],padding_top_left=[x/2 for x in ep.reference_geometry['padding_top_left']])
    f512,v512=op.exact_footprint(mask,(32,32),geometry)
    if not np.allclose(f512,ep.wf.reshape(32,2,32,2).mean((1,3)).ravel(),atol=1e-10):raise ArithmeticError('512 physical coverage mapping differs from native16px cells')
    w0=op.direction(ep.r,ep.wf,ep.wvalid,split.anchor,normalize=False);wo=op.direction(descriptors[0][3],f,v,np.ones(1024,bool),normalize=False);wc=op.direction(descriptors[0][4],f,v,np.ones(1024,bool),normalize=False)
    if any(w is None for w in (w0,wo,wc)):return _end(ep,b-.5,mid,dict(host=host,inactive=True,fallback='zero_anchor_descriptor_direction',mask_asset=token_info),start,rt)
    fields=[]
    for native,oo,cc,_,_ in descriptors:
        m0,mo,mc=op.dot(native,w0),op.dot(oo,wo),op.dot(cc,wc);cos=np.sum(oo*cc,axis=1)
        fields.append((np.c_[m0,mo,mc,cos,mo*mc,np.abs(mo-mc)],np.c_[m0,mo]))
    if control=='native_plus_unmasked512':fields=[(simple,simple[:,:1]) for enhanced,simple in fields]
    residual,readout=op.residual_readout(fields[0][0],fields[1][0],fields[0][1],fields[1][1],ep.wf,ep.wvalid,split)
    return _end(ep,b-.5+residual.reshape(64,64),mid,dict(host=host,readout=readout,mask_asset=token_info,
        mask_groups='(row+2col)%4, exact one quarter',collect_only_masked_positions=True,context_descriptor_nonconstant=bool(np.ptp(descriptors[1][4])>0),
        reference_512_area_hash=common.array_hash(f512),reference_512_valid_hash=common.array_hash(v512),directions_A_frozen=True),start,rt)


def _quadrants(rt,ep,role):
    from PIL import Image
    image=rgb_view(ep,role);field=np.empty((96,96,1024),np.float32)
    for iy in range(2):
        for ix in range(2):
            crop=image[iy*512:(iy+1)*512,ix*512:(ix+1)*512]
            view=np.asarray(Image.fromarray(crop).resize((768,768),Image.Resampling.BILINEAR))
            feature=rt.encode(role,view).reshape(48,48,1024);field[iy*48:(iy+1)*48,ix*48:(ix+1)*48]=feature
    return common.unit(resize(field,(64,64)).reshape(4096,1024))


def m28(ep,*,control=None):
    start=time.perf_counter();ep,split=_episode(ep);mid='PRO30_M28'+('' if control is None else '__'+control)
    degenerate=_degenerate(ep,mid,start)
    if degenerate is not None:return degenerate
    rt=_runtime(ep);r1=_quadrants(rt,ep,'r');q1=_quadrants(rt,ep,'q');b0,host0=_B(ep);b1,host1=_B(ep,r1,q1)
    losses=[]
    for reference in (ep.r,r1):
        w=op.direction(reference,ep.wf,ep.wvalid,split.anchor,normalize=False) if split.adequate else None
        losses.append(op.balanced_loss(op.dot(reference,w)/.07,ep.wf,ep.wvalid,split.calibration) if w is not None else None)
    extra=max(0.,losses[1]-losses[0]) if all(x is not None for x in losses) else 0.
    yy,xx=np.indices((64,64));distance=np.minimum(np.abs((yy+.5)*16-512),np.abs((xx+.5)*16-512)).ravel();penalty=.05*np.maximum(0,1-distance/32)+extra
    p0=np.clip(b0.ravel(),.01,.99);p1=np.clip(b1.ravel(),.01,.99)
    if control in ('uniform_average','pointwise_max','pointwise_min','single_scale_reference_loss'):
        if control=='uniform_average':field=(b0+b1)/2
        elif control=='pointwise_max':field=np.maximum(b0,b1)
        elif control=='pointwise_min':field=np.minimum(b0,b1)
        else:field=b1 if losses[0] is not None and losses[1]<losses[0] else b0
        return _end(ep,field-.5,mid,dict(hosts=[host0,host1],reference_losses=losses,control=control),start,rt)
    if control=='equal_latency_full_native':return _same_data_control(ep,mid,'M28_native_highres_measured_equal_latency')
    graph,graph_producer=common.require_artifacts(ep,'pro30_native_query_graph','pro30_native_query_graph_producer')
    if (graph_producer.get('query_unit_array_sha256')!=common.array_hash(ep.q)
            or graph_producer.get('mutual_neighbors')!=20 or graph_producer.get('mean_degree_normalized') is not True
            or not graph_producer.get('source_sha256')):
        raise common.ArtifactUnavailable('Original native query graph producer/content binding required')
    graph=sparse.csr_matrix(graph)
    if graph.shape!=(4096,4096) or np.any(graph.data<0) or (graph-graph.T).nnz and np.max(np.abs((graph-graph.T).data))>1e-8:
        raise common.ArtifactUnavailable('Actual original native normalized mutual20NN graph required')
    unary=np.c_[-np.log1p(-p0),-np.log1p(-p1)+penalty,-np.log(p0),-np.log(p1)+penalty]
    initial=(p0>.5).astype(int)*2
    labels,solver=op.expansion_potts(unary,graph,initial,scale_penalty=0. if control=='binary_potts_only' else .02)
    y=labels//2
    return _end(ep,(2*y-1).reshape(64,64),mid,dict(hosts=[host0,host1],solver=solver,reference_losses=losses,
        fine_scale_reference_penalty=extra,scale1_tokens=int(np.count_nonzero(labels%2)),all_four_quadrants=True,
        graph_positions='original whole-image native coordinates',crop_coords_are_not_global_positions=True),start,rt)


def _phase_views(rt,ep):
    image=rgb_view(ep,'q');padded=np.pad(image,((8,8),(8,8),(0,0)),mode='reflect');features=[ep.q]
    for sy,sx in ((8,0),(0,8),(8,8)):
        view=padded[8-sy:8-sy+1024,8-sx:8-sx+1024];features.append(rt.encode('q',view))
    return features


def m29(ep,*,control=None):
    start=time.perf_counter();ep,split=_episode(ep);mid='PRO30_M29'+('' if control is None else '__'+control)
    degenerate=_degenerate(ep,mid,start)
    if degenerate is not None:return degenerate
    b,host=_B(ep)
    if control=='same128_bilinear_B':return _end(ep,resize(b,(128,128))-.5,mid,dict(host=host,control=control,no64_compression=True),start)
    w=op.direction(ep.r,ep.wf,ep.wvalid,split.anchor,normalize=False) if split.adequate else None
    if w is None:return _end(ep,b-.5,mid,dict(host=host,inactive=True,fallback='A/C_insufficient_or_zero_direction'),start)
    beta,calibration=op.area_calibration(op.dot(ep.r,w),ep.wf,ep.wvalid,split)
    if beta is None:return _end(ep,b-.5,mid,dict(host=host,inactive=True,measurement_calibration=calibration),start)
    rt=_runtime(ep);features=_phase_views(rt,ep);observations=[np.clip((op.dot(q,w)-beta[0])/max(beta[1],1e-3),0,1) for q in features]
    g=ep.query_geometry;oy,ox=g['padding_top_left'];sh,sw=g['resized_hw'];box=(oy,ox,oy+sh,ox+sw);phases=[(0,0),(8,0),(0,8),(8,8)]
    operators=[];valids=[]
    for phase in phases:
        a,v=op.phase_operators(phase=phase,valid_box=box);operators.append(a);valids.append(v)
    D,_=op.phase_operators(valid_box=box);initial=resize(np.clip(b,0,1),(128,128))
    if control in ('same_views_fine16','same_views_continuous_residual_transfer'):
        return _same_data_control(ep,mid,'M29_'+control+'_exact_four_phase_features_and_cost')
    if control=='phase_uniform_average':
        yy,xx=np.indices((128,128));ys=(yy+.5)*8;xs=(xx+.5)*8
        mapped=[sample_grid(a.reshape(64,64),(ys+sy)/16-.5,(xs+sx)/16-.5) for a,(sy,sx) in zip(observations,phases)]
        return _end(ep,np.mean(mapped,axis=0)-.5,mid,dict(host=host,measurement_calibration=calibration,control=control,phases=phases),start,rt)
    field,solver=op.inverse_mask(observations,operators,valids,D,np.clip(b,0,1).ravel(),initial,data=control!='B_plus_TV',base_term=control!='data_only',tv_term=control!='data_only')
    return _end(ep,field-.5,mid,dict(host=host,measurement_calibration=calibration,solver=solver,phases=phases,
        observed_phase_feature_hashes=[common.array_hash(q) for q in features],phase_APD=False,
        physical_fine_cell_pixels=8,phase_patch_origin='negative shift: phase8 covers original [-8,8] in first patch',
        no64_compression=True,identity_mainly_from_B=True,linear_area_model_is_unverified=True),start,rt)


def m30(ep,*,control=None):
    start=time.perf_counter();ep,split=_episode(ep);mid='PRO30_M30'+('' if control is None else '__'+control)
    degenerate=_degenerate(ep,mid,start)
    if degenerate is not None:return degenerate
    b,host=_B(ep);w=op.direction(ep.r,ep.wf,ep.wvalid,ep.wvalid>0)
    if w is None:return _end(ep,b-.5,mid,dict(host=host,inactive=True,fallback='zero_full_reference_direction'),start)
    if control in ('direct_perturbation_mean','dense_patch_removal','mean_replacement','variance_noise','same_image_permutation'):
        return _same_data_control(ep,mid,'M30_'+control+'_same8suffix_and_no_external_images')
    rt=_runtime(ep);torch=rt.torch;session=rt.native('q');hidden=session['h20'].detach();patch=hidden[:,5:];valid=torch.as_tensor(ep.q_valid>0)
    baseline=torch.quantile(patch[:,valid],.5,dim=1,keepdim=True);p=torch.as_tensor(np.clip(b.ravel(),.05,.95),dtype=torch.float32)
    v0=torch.logit(p);v=v0.detach().clone().requires_grad_(True);optimizer=torch.optim.Adam([v],lr=.3,betas=(.9,.999),eps=1e-8)
    wt=torch.as_tensor(w,dtype=torch.float32);edges=op.fine_edges((64,64),ep.q_valid>0);edge=torch.as_tensor(edges,dtype=torch.long);history=[];changes=[]
    native_logits=torch.as_tensor(op.dot(ep.q,w)/.07,dtype=torch.float32)
    def energy(mask):
        if control in ('BCE_TV_only','native_static_energy'):
            dp,dm=native_logits,native_logits
        else:
            hp=torch.cat((hidden[:,:5],mask[None,:,None]*patch+(1-mask[None,:,None])*baseline),1)
            hm=torch.cat((hidden[:,:5],(1-mask[None,:,None])*patch+mask[None,:,None]*baseline),1)
            dp=rt.suffix(hp,session)[0]@wt/.07;dm=rt.suffix(hm,session)[0]@wt/.07
        smooth=((mask[edge[:,0]]-mask[edge[:,1]])**2+1e-6).sqrt().mean() if len(edges) else mask.sum()*0
        bce=torch.nn.functional.binary_cross_entropy(mask[valid],p[valid]);data=(mask*torch.nn.functional.softplus(-dp)+(1-mask)*torch.nn.functional.softplus(dm)+.25*torch.nn.functional.softplus(dm))[valid].mean()
        total=(data if control!='BCE_TV_only' else data*0)+.10*smooth+.50*bce
        return total,dict(data=float(data.detach()),TV=float(smooth.detach()),BCE=float(bce.detach()))
    for step in range(1 if control=='one_gradient_step' else 4):
        old=torch.sigmoid(v).detach();optimizer.zero_grad(set_to_none=True);objective,parts=energy(torch.sigmoid(v));gradient=rt.vjp(objective,v,suffix_units=0 if control in ('BCE_TV_only','native_static_energy') else 2);v.grad=gradient;optimizer.step()
        with torch.no_grad():v.copy_(torch.maximum(torch.minimum(v,v0+1.5),v0-1.5).clamp(-6,6))
        change=float(torch.max(torch.abs(torch.sigmoid(v)-old)));history.append(dict(step=step,objective=float(objective.detach()),**parts));changes.append(change)
        if change<1e-3:break
    mask=torch.sigmoid(v).detach().numpy().reshape(64,64);mask.ravel()[ep.q_valid<=0]=0
    return _end(ep,mask-.5,mid,dict(host=host,optimizer='Adam only episode mask logits; all weights remain frozen',objective_history=history,
        maximum_mask_changes=changes,maximum_steps=4,convergence_claim=False,trust_radius=1.5,
        immutable_negative_bound=float(expit(-1.5)),immutable_positive_bound=float(expit(1.5)),prefix_fixed=True,
        baseline='actual valid Q rawH20 per-channel median',control=control),start,rt)


for number,fn in [(24,m24),(25,m25),(26,m26),(27,m27),(28,m28),(29,m29),(30,m30)]:
    mid='PRO30_M'+str(number);METHODS[mid]=fn
    REQUIREMENTS[mid]=['actual native R/Q at1024 and original complete MR/geometry','bound complete_FoRIS_MEAN16 pro30_B_operator','actual bound pro30_encoder_context']
    CONTRACTS[mid]=dict(common.source_contract(number),input_contract='N+X',host='full original FoRIS continuous + MEAN_a.25_l16; original quirks preserved',
        query_GT_read=False,parameter_search=False,cached_budget_seconds=2.,fixed_evaluation='same600, no200screening',quality='unknown',native_ready=False)

for number,names in {
    24:['random_heads_mean5','absolute_gradient','output_norm','uniform_equal_energy','readonly_head_linear','gate1'],
    25:['linear_logit_IG','even_amplitude_IG','ordinary_feature_mean','gradient_times_input','attention_rollout','block20_direct'],
    27:['native_plus_unmasked512','masked_uniform_average','mean_token','token_permutation','unmasked_positions_TTA'],
    28:['uniform_average','pointwise_max','pointwise_min','single_scale_reference_loss','binary_potts_only','equal_latency_full_native'],
    29:['phase_uniform_average','same_views_fine16','same_views_continuous_residual_transfer','same128_bilinear_B','data_only','B_plus_TV'],
    30:['BCE_TV_only','native_static_energy','one_gradient_step','direct_perturbation_mean','dense_patch_removal','mean_replacement','variance_noise','same_image_permutation']}.items():
    for name in names:CONTROLS[f'PRO30_M{number}__{name}']=partial(METHODS[f'PRO30_M{number}'],control=name)
for name,mode,apd in [('native_APD','native','native'),('native_noAPD','native',False),('neutral_APD','neutral','native'),
    ('prefix_attenuation','prefix_attenuation',False),('rope_off','rope_off',False),('vit5_prefix','vit5_prefix',False),
    ('prefix_zero_identity','native',False),('same_cost_native_multiview','native','native')]:
    CONTROLS['PRO30_M26__'+name]=partial(m26,mode=mode,apd=apd,control=name)
for mid in CONTRACTS:CONTRACTS[mid]['controls']=[key for key in CONTROLS if key.startswith(mid+'__')]


def probe_m26_real_software(ep,*,coordinate_check=False):
    """Root-owned actual single R software/cost audit, never automatically run."""
    from .eva_24_30 import audit_observed_qk,parameter_array_hash
    start=time.perf_counter();ep,_=_episode(ep);rt=_runtime(ep)
    hash_start=time.perf_counter();before=parameter_array_hash(rt.model);hash_seconds=time.perf_counter()-hash_start
    rt.native('r',mode='native',audit=True)
    audit=audit_observed_qk(rt.audit)
    coordinate=None
    if coordinate_check:
        fixed=_numpy(rt.native('r',mode='neutral'))
        shifted=_numpy(rt.native('r',mode='neutral_coordinate_shift'))
        coordinate=dict(max_feature_error=float(np.max(np.abs(fixed-shifted))),
            content_and_RGB_unchanged=True,shift='native RoPE patch1 minus patch0, composed identically on all patch Q/K',
            actual_image_translation_checked=False,extra_reference_forwards=2)
    hash_start=time.perf_counter();after=parameter_array_hash(rt.model);hash_seconds+=time.perf_counter()-hash_start
    if before!=after:raise common.ArtifactUnavailable('Actual encoder parameter hash changed in the software audit')
    return dict(scope='one actual native R forward and bounded exact-QK audit; no segmentation-quality result',
        checkpoint_sha256=rt.binding['model_assets']['checkpoint_sha256'],encoder_binding=rt.binding,
        actual_operations=dict(rt.stats),software_certificate=audit,total_seconds=time.perf_counter()-start,
        parameter_array_sha256_before=before,parameter_array_sha256_after=after,parameter_hash_verification_seconds=hash_seconds,
        coordinate_origin_check=coordinate,
        eligible_to_expand600=False,next='run one complete M26 and matched controls with bound B; compare measured cached and total costs to2s separately')
