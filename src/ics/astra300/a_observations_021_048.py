"""A cards requiring actual frozen-model internal or raw final-LN observations.

No architecture construction, downloads or substitute final-feature observations
are performed. The root's source-bound provider supplies the actual operators.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

from .common import ArtifactUnavailable,require_artifact
from .a_helpers_001_050 import (EPS,Frame,b0,fit_rbf,fit_ridge,fps,infer_a,
    nearest_mean_distance,rank_basis,sqdist,threshold,unit)


def _plain(frame):
    return lambda q,ids,role:b0(frame,q),{"mechanism":"A_B0_5NN"}


def _binding(ep,packet=None,raw_cache=False):
    if str(ep.producer.get('kind','')).startswith('synthetic'):
        return {'execution_kind':'synthetic_fixture_not_DINO'}
    try:
        binding=require_artifact(ep,'internal_encoder_binding')
    except ArtifactUnavailable:
        if not raw_cache or not isinstance(packet,dict) or not isinstance(packet.get('producer'),dict):raise
        binding=packet['producer']
        if binding.get('FoRIS_Part1_applied') is not False or binding.get('kind') not in (
            'frozen_DINOv3_FP32_native_final_LN_patches','frozen_DINOv3_FP32_native_final_LN_patches_then_unit'):
            raise ArtifactUnavailable('Actual raw finalLN producer binding required; processed cache is inadmissible')
    else:
        if binding.get('execution_kind')!='real_frozen_cpu_model':
            raise ArtifactUnavailable('Actual frozen CPU internal observation binding required')
    expected=ep.producer.get('model_assets',ep.producer);observed=binding.get('model_assets',binding)
    for key in ('checkpoint_sha256','config_sha256'):
        if key in expected and expected[key]!=observed.get(key):
            raise ArtifactUnavailable('Actual internal/cache model source does not match native episode: '+key)
    return binding


def _packet(ep,name,leading=None):
    packet=require_artifact(ep,name)
    if not isinstance(packet,dict) or not all(role in packet for role in ('q','r')):
        raise ArtifactUnavailable(name+' must supply actual q/r aligned observations')
    binding=_binding(ep,packet,raw_cache=name=='raw_finalLN')
    if leading is not None and list(packet.get('blocks',[]))!=leading:
        raise ArtifactUnavailable(name+' wrong actual one-based block indices')
    out={}
    for role,n in (('q',len(ep.q)),('r',len(ep.r))):
        array=np.asarray(packet[role],float)
        if not np.isfinite(array).all():raise ValueError(name+' contains nonfinite actual observations')
        out[role]=array
    return out,binding


def _head_banks(frame,reference):
    fi=fps(frame.x,frame.fids,64);bi=fps(frame.x,frame.bids,64)
    if not len(fi) or not len(bi):return None
    return reference[fi],reference[bi]


def _head_score(x,f,b):
    return (nearest_mean_distance(x,b)-nearest_mean_distance(x,f))/2


def fit_a022(frame,all_heads=False,concatenated=False):
    packet,binding=_packet(frame.ep,'head_messages_last')
    r,q=packet['r'],packet['q']
    if r.ndim!=3 or q.shape[1:]!=r.shape[1:] or r.shape[0]!=len(frame.x) or q.shape[0]!=len(frame.ep.q) or r.shape[2]!=frame.x.shape[1]:
        raise ArtifactUnavailable('A022 actual post-W_O perhead message must have [N,H,D], including original prefix-value denominator')
    r,q=unit(r),unit(q);models=[];rejected=0
    if concatenated:
        rd=r.reshape(len(r),-1);qd=q.reshape(len(q),-1)
        coef=fit_ridge(frame,rd)
        return lambda x,ids,role:((rd if role=='r' else qd)[ids]@coef[:-1]+coef[-1],{}),{
            'control':'all_actual_head_messages_concatenated_ridge','L2':1.,'dimensions':rd.shape[1],'binding':binding}
    for head in range(r.shape[1]):
        banks=_head_banks(frame,r[:,head])
        if banks is None:continue
        f,b=banks;errors=[];gap=[];source_scores=[]
        for block in range(4):
            held=frame.train&(frame.blocks==block);wf,wb=frame.wf[held],frame.wb[held]
            if wf.sum()<=0 or wb.sum()<=0:continue
            inner=Frame.make(frame.ep,frame.train&(frame.blocks!=block));bb=_head_banks(inner,r[:,head])
            if bb is None:continue
            score=_head_score(r[held,head],*bb)
            positive=score>0
            errors.append(float(.5*(wf[~positive].sum()/wf.sum()+wb[positive].sum()/wb.sum())))
            gap.append(float(wf@score/wf.sum()-wb@score/wb.sum()));source_scores.extend(score.tolist())
        reliable=bool(errors) and max(errors)<.5 and min(gap)>0
        if not reliable and not all_heads:rejected+=1;continue
        scale=max(float(np.quantile(source_scores,.75)-np.quantile(source_scores,.25)) if source_scores else 1.,1e-6)
        center=float(np.median(source_scores)) if source_scores else 0.
        models.append((head,f,b,center,scale))
    if not models:
        predict,info=_plain(frame)
        return predict,dict(info,card_id='A022',degeneration='no_crossblock_reliable_actual_head',observed_heads=r.shape[1])
    def predict(x,ids,role):
        messages=packet[role][ids]
        messages=unit(messages)
        scores=np.array([(_head_score(messages[:,head],f,b)-center)/scale for head,f,b,center,scale in models])
        return np.median(scores,axis=0),{'head_count':len(models),'real_internal_execution':binding.get('execution_kind')=='real_frozen_cpu_model'}
    return predict,{'selected_heads':[m[0] for m in models],'rejected_heads':rejected,'all_heads_control':all_heads,
                    'message_location':'actual A_hV_hW_Oh before bias/LayerScale','binding':binding}


def _rank_channels(values,valid):
    # Rank maps are unlabeled image statistics; invalid physical padding is not
    # allowed to change their empirical population.
    out=np.zeros((values.shape[-1],np.prod(values.shape[:-1])))
    flat=values.reshape(-1,values.shape[-1]);ids=np.flatnonzero(valid)
    for j,row in enumerate(flat):
        out[ids,j]=rankdata(row[ids],method='average')/max(len(ids),1)
    return out


def fit_a026(frame,mode='combined',random_rows=False):
    packet,binding=_packet(frame.ep,'attention_special_to_patch',[12,16,20,24])
    r,q=packet['r'],packet['q']
    if r.ndim!=3 or r.shape[0]!=4 or r.shape[1] not in (1,5) or r.shape[2]!=len(frame.x) or q.shape!=(4,r.shape[1],len(frame.ep.q)):
        raise ArtifactUnavailable('A026 actual4layers CLS[/4register] rows required, not final-feature channels')
    if random_rows:
        rng=np.random.default_rng(0);r=rng.uniform(0,1,r.shape);q=rng.uniform(0,1,q.shape)
    rr=_rank_channels(r,frame.ep.wvalid>0);qq=_rank_channels(q,frame.ep.q_valid>0)
    # Native origin margin, no FoRIS-host score or semantic special-token cosine.
    def margin(x):
        f=unit(np.sum(frame.x*frame.wf[:,None],0));b=unit(np.sum(frame.x*frame.wb[:,None],0))
        return x@(f-b)
    if frame.wf.sum()<=0 or frame.wb.sum()<=0:return _plain(frame)
    rm=margin(frame.x);qm=margin(frame.ep.q)
    rd=rr if mode=='attention' else (rm[:,None] if mode=='margin' else np.c_[rr,rm])
    qd=qq if mode=='attention' else (qm[:,None] if mode=='margin' else np.c_[qq,qm])
    if mode=='complete_profile_RBF':
        basis=frame.x[frame.train]
        fi=fps(frame.x,frame.fids,64);bi=fps(frame.x,frame.bids,64);ids=np.r_[fi,bi]
        support=np.c_[rr[ids],frame.x[ids]@basis.T]
        distance=sqdist(support,support);width=max(float(np.median(distance[np.triu_indices(len(ids),1)])),EPS)
        coef=np.linalg.solve(np.exp(-distance/width)+np.eye(len(ids)),np.r_[np.ones(len(fi)),-np.ones(len(bi))])
        def predict(x,ids,role):
            response=(rr if role=='r' else qq)[ids];out=np.empty(len(x))
            for start in range(0,len(x),128):
                z=np.c_[response[start:start+128],x[start:start+128]@basis.T]
                out[start:start+128]=np.exp(-sqdist(z,support)/width)@coef
            return out,{'complete_native_R_profile_columns':len(basis),'actual_response_channels':rr.shape[1]}
        return predict,{'control':'all_actual_attention_responses_plus_complete_native_affinity_RBF',
            'width2':width,'L2':1.,'binding':binding}
    coef=fit_ridge(frame,rd)
    def predict(x,ids,role):
        z=(rd if role=='r' else qd)[ids]
        return z@coef[:-1]+coef[-1],{'response_dimensions':rr.shape[1],'no_register_degeneration':r.shape[1]==1}
    return predict,{'ridge_L2':1.,'actual_blocks':[12,16,20,24],'channels':r.shape[1],
        'mode':mode,'random_uniform_rows_control':random_rows,'rank_population':'all physically-valid imagepatches','binding':binding}


def fit_a032(frame,transitions=True,reverse=False,profiles=False):
    packet,binding=_packet(frame.ep,'midlayers',[12,16,20,24])
    r,q=packet['r'],packet['q']
    if r.shape!=(4,len(frame.x),frame.x.shape[1]) or q.shape!=(4,len(frame.ep.q),frame.x.shape[1]):
        raise ArtifactUnavailable('A032 needs actual raw post-block four-state paths; cannot repeat finalLN four times')
    r,q=unit(r),unit(q)
    if reverse:r,q=r[::-1],q[::-1]
    if frame.wf.sum()<=0 or frame.wb.sum()<=0:return _plain(frame)
    if profiles:
        anchor_ids=np.r_[fps(frame.x,frame.fids,64),fps(frame.x,frame.bids,64)]
        f=unit(np.sum(frame.x*frame.wf[:,None],0));b=unit(np.sum(frame.x*frame.wb[:,None],0))
        rd=np.c_[np.concatenate([r[k]@r[k,anchor_ids].T for k in range(4)],axis=1),frame.x@(f-b)]
        qd=np.c_[np.concatenate([q[k]@r[k,anchor_ids].T for k in range(4)],axis=1),frame.ep.q@(f-b)]
        model=fit_rbf(frame,rd);support,coef,width=model
        return lambda x,ids,role:(np.exp(-sqdist((rd if role=='r' else qd)[ids],support)/width)@coef,{}),{
            'control':'all_four_actual_layer_role_anchor_affinity_profiles_RBF','profile_columns':rd.shape[1],
            'perrole_anchor_cap':64,'original_native_margin_also_present':True,'binding':binding}
    states={};cutoffs=[]
    for layer in range(4):
        f=unit(np.sum(r[layer]*frame.wf[:,None],0));b=unit(np.sum(r[layer]*frame.wb[:,None],0))
        rm=r[layer]@(f-b);qm=q[layer]@(f-b)
        negative=float(np.quantile(rm[frame.bids],.75)) if len(frame.bids) else 0.
        positive=float(np.quantile(rm[frame.fids],.25)) if len(frame.fids) else 0.
        lo,hi=sorted((negative,positive));cutoffs.append((lo,hi))
        for role,margin in (('r',rm),('q',qm)):
            states.setdefault(role,[]).append(np.where(margin<lo,0,np.where(margin>hi,2,1)))
    rs,qs=np.array(states['r']).T,np.array(states['q']).T
    tables=[]
    for weights in (frame.wf,frame.wb):
        counts=np.ones((3,3,3))
        for k in range(3):np.add.at(counts[k],(rs[:,k],rs[:,k+1]),weights)
        tables.append(counts/counts.sum(2,keepdims=True))
    ff=unit(np.sum(frame.x*frame.wf[:,None],0));bb=unit(np.sum(frame.x*frame.wb[:,None],0))
    def predict(x,ids,role):
        path=(rs if role=='r' else qs)[ids];score=x@(ff-bb)/.07
        if transitions:
            for k in range(3):
                score+=np.log(tables[0][k,path[:,k],path[:,k+1]]/tables[1][k,path[:,k],path[:,k+1]])/3
        return score,{'transition_terms':3 if transitions else 0}
    return predict,{'states':3,'layer_thresholds':cutoffs,'Laplace_pseudocount':1.,
        'layer_order':[24,20,16,12] if reverse else [12,16,20,24],
        'raw_intermediate_states_no_invented_LN':True,'binding':binding}


def _raw(ep):
    packet,binding=_packet(ep,'raw_finalLN')
    for role,native in (('r',ep.r),('q',ep.q)):
        a=packet[role]
        if a.shape!=native.shape or not np.allclose(unit(a),native,atol=2e-5,rtol=2e-5):
            raise ArtifactUnavailable('Actual raw finalLN is not bound to the native unit episode: '+role)
    return packet,binding


def _unmix_endpoints(frame):
    packet,binding=_raw(frame.ep);r=packet['r'];h,w=frame.ep.r_hw
    yy,xx=np.indices((h,w));coords=np.c_[yy.ravel(),xx.ravel()]
    windows=[]
    for index in fps(frame.x,np.flatnonzero(frame.train),64):
        ids=np.flatnonzero(frame.train & (np.max(np.abs(coords-coords[index]),axis=1)<=2))
        c=frame.c[ids]
        if len(ids)<3 or np.ptp(c)<.5:continue
        design=np.c_[c,1-c]
        if np.linalg.cond(design.T@design)>100:continue
        # Actual held-out mixed observations, not synthetic points or an
        # in-sample residual asserting that the chosen endpoints generalize.
        held_residual=[];held_alpha_error=[]
        for j in np.flatnonzero((c>0)&(c<1)):
            keep=np.arange(len(ids))!=j;d=design[keep]
            if len(d)<2 or np.ptp(c[keep])<.5 or np.linalg.cond(d.T@d)>100:continue
            endpoints=np.linalg.lstsq(d,r[ids[keep]],rcond=None)[0]
            predicted=design[j]@endpoints
            held_residual.append(float(np.sum((r[ids[j]]-predicted)**2)))
            delta=endpoints[0]-endpoints[1]
            alpha=float(np.clip((r[ids[j]]-endpoints[1])@delta/max(delta@delta,EPS),0,1))
            held_alpha_error.append(abs(alpha-c[j]))
        if not held_residual or float(np.median(held_alpha_error))>.1:continue
        endpoint=np.linalg.lstsq(design,r[ids],rcond=None)[0]
        windows.append((endpoint[0],endpoint[1],max(float(np.quantile(held_residual,.99)),1e-8),int(index)))
    return packet,binding,windows


def fit_a033(frame,plain_endpoint_classifier=False):
    packet,binding,windows=_unmix_endpoints(frame)
    if not windows:
        predict,info=_plain(frame)
        def fallback(x,ids,role):
            score,activity=predict(x,ids,role)
            return np.clip((score+2)/4,0,1),activity
        return fallback,dict(info,card_id='A033',degeneration='no_well_conditioned_real_soft_coverage_window')
    def predict(x,ids,role):
        raw=packet[role][ids]
        base,activity=b0(frame,x);alpha=np.clip((base+2)/4,0,1)
        total_weight=np.zeros(len(x));alpha_sum=np.zeros(len(x));accepted=np.zeros(len(x),bool)
        for f,b,bound,_ in windows:
            delta=f-b
            a=np.clip((raw-b)@delta/max(delta@delta,EPS),0,1)
            residual=np.sum((raw-b-a[:,None]*delta)**2,1)
            valid=np.ones(len(x),bool) if plain_endpoint_classifier else residual<=bound
            weight=np.where(valid,np.exp(-residual/max(bound,EPS)),0)
            total_weight+=weight;alpha_sum+=weight*a;accepted|=valid
        alpha[accepted]=alpha_sum[accepted]/np.maximum(total_weight[accepted],EPS)
        return alpha,dict(activity,unmix_accepted_points=int(accepted.sum()),points=len(x))
    return predict,{'actual_raw_LN_endpoints':len(windows),'soft_coverage_span_min':.5,
        'condition_number_max':100.,'held_mixed_residual_q':.99,'held_mixed_alpha_median_error_max':.1,
        'residual_weighted_alpha':True,'plain_endpoint_control':plain_endpoint_classifier,'binding':binding}


def fit_a048(frame):
    packet,binding,windows=_unmix_endpoints(frame)
    # Only endpoint pairs independently validated on actual mixed patches are
    # eligible; synthetic points never supply their own validation evidence.
    validated=[]
    raw=packet['r']
    for f,b,bound,index in windows:
        mixed=np.flatnonzero(frame.train&(frame.c>0)&(frame.c<1)&(frame.blocks!=frame.blocks[index]))
        if not len(mixed):continue
        delta=f-b;alpha=np.clip((raw[mixed]-b)@delta/max(delta@delta,EPS),0,1)
        residual=np.sum((raw[mixed]-b-alpha[:,None]*delta)**2,1)
        good=(residual<=bound)&(np.abs(alpha-frame.c[mixed])<=.1)
        if good.mean()>=.8:validated.append((f,b,bound,index))
    if not validated:
        predict,info=_plain(frame)
        def fallback(x,ids,role):
            score,activity=predict(x,ids,role);return np.clip((score+2)/4,0,1),activity
        return fallback,dict(info,card_id='A048',degeneration='no_real_crossblock_mixed_validation_pairs',candidate_pairs=len(windows))
    fractions=np.array([0.,.25,.5,.75,1.])
    regressions=[]
    def spline(value):return np.c_[np.ones(len(value)),value,np.maximum(value[:,None]-fractions[1:-1],0)]
    # Fit a piecewise-linear share readout from the fixed generated knot labels
    # plus actual reference mixed observations. This can differ from projecting
    # alpha directly; it is not an identity interpolation renamed regression.
    for f,b,bound,_ in validated:
        delta=f-b
        actual=np.flatnonzero(frame.train&(frame.c>0)&(frame.c<1))
        coordinate=(raw[actual]-b)@delta/max(delta@delta,EPS)
        residual=np.sum((raw[actual]-b-np.clip(coordinate,0,1)[:,None]*delta)**2,1)
        actual=actual[residual<=bound];coordinate=coordinate[residual<=bound]
        training=np.r_[fractions,coordinate];targets=np.r_[fractions,frame.c[actual]]
        design=spline(training)
        coefficient=np.linalg.solve(design.T@design+1e-4*np.eye(design.shape[1]),design.T@targets)
        regressions.append(coefficient)
    def predict(x,ids,role):
        query=packet[role][ids];values=[];validities=[]
        for (f,b,bound,_),coefficient in zip(validated,regressions):
            # Piecewise-linear regression on the generated line's knot values;
            # orthogonal residual is priced and never becomes a FG label.
            delta=f-b;coordinate=(query-b)@delta/max(delta@delta,EPS)
            estimate=np.clip(spline(coordinate)@coefficient,0,1)
            # Residual is to the geometric constrained line, not to a fitted
            # share's new endpoint. Keep geometry rejection separate from label.
            geometric=np.clip(coordinate,0,1)
            residual=np.sum((query-b-geometric[:,None]*delta)**2,1)
            values.append(estimate);validities.append(residual<=bound)
        values=np.array(values).T;valid=np.array(validities).T
        base,activity=b0(frame,x);out=np.clip((base+2)/4,0,1);accepted=valid.any(1)
        for i in np.flatnonzero(accepted):
            predictions=values[i,valid[i]]
            out[i]=np.clip(float(predictions.mean()-predictions.std()),0,1)
        return out,dict(activity,validated_mixture_points=int(accepted.sum()),points=len(x))
    return predict,{'real_mixed_validated_pairs':len(validated),'fractions':fractions.tolist(),
        'coordinate_residual_tolerance_fraction':.1,'validation_acceptance_min_fraction':.8,'binding':binding}


def fit_soft_raw_ridge(frame):
    packet,binding=_raw(frame.ep)
    ids=np.flatnonzero(frame.train);x=packet['r'][ids]
    design=np.c_[x,np.ones(len(x))];weights=frame.valid[ids]/max(frame.valid.sum(),EPS)
    a=design*np.sqrt(weights[:,None]);y=frame.c[ids]*np.sqrt(weights)
    if len(ids)<a.shape[1]:coef=a.T@np.linalg.solve(a@a.T+np.eye(len(ids)),y)
    else:coef=np.linalg.solve(a.T@a+np.eye(a.shape[1]),a.T@y)
    def predict(x,ids,role):
        raw=packet[role][ids]
        return np.clip(raw@coef[:-1]+coef[-1],0,1),{}
    return predict,{'control':'actualraw_soft_coverage_ridge','L2':1.,'binding':binding}


def _observation_receipt(fn):
    def execute(ep):
        provider=getattr(ep,'provider',None)
        before=dict(getattr(provider,'receipt',{}))
        result=fn(ep)
        after=dict(getattr(provider,'receipt',{}))
        if 'actual_extra_forwards' in after:
            delta=int(after['actual_extra_forwards']-before.get('actual_extra_forwards',0))
            result.info['new_encoder_forwards']=delta
            result.info['internal_observation_receipt_delta']={key:after.get(key,0)-before.get(key,0)
                for key in ('actual_extra_forwards','actual_tail_blocks','forward_wall_seconds','forward_cpu_seconds')}
        else:
            result.info['new_encoder_forwards']=0
            result.info['observation_cost_scope']='supplied source-bound artifacts only; upstream observation production is not zero-cost'
        return result
    return execute


def install(register,requirements,methods,controls,recipes):
    register('A022',fit_a022,(
        'Actual lastblock head AV includingprefix denominator multipliedby its realW_O block; unitnormalizeeachD-message forcosine5NN.',
        'Requireeacheligiblecrossblock headbalancederror<.5 andpositiveFG−BGgap; normalizewithheadreferenceheldmedian/IQR, floored1e-6, aggregate median.',),
        (('all_heads_median',lambda f:fit_a022(f,all_heads=True)),('all_actual_heads_concat_ridge',lambda f:fit_a022(f,concatenated=True))))
    register('A026',fit_a026,(
        'Actual4 layers [12,16,20,24],5specialtoken rows orCLSonly documenteddegen; averageheads actualprovider, image-validpatch meanrank/n.',
        'BalancedridgeL2=1 onresponse+r/q weightedmean prototypecosmargin; empiricalrank usesnoQlabels andexcludesphysicalpadding.',),
        (('original_margin_only',lambda f:fit_a026(f,mode='margin')),('attention_response_only',lambda f:fit_a026(f,mode='attention')),
         ('uniform_random_rows',lambda f:fit_a026(f,random_rows=True)),
         ('all_responses_complete_native_profile_RBF',lambda f:fit_a026(f,mode='complete_profile_RBF'))))
    register('A032',fit_a032,(
        'Actual post-block raw12/16/20/24 states unitnormalized separately; negativeBGq.75/positiveFGq.25 thresholds sorted into uncertaintyinterval.',
        'ClassweightedLaplace1 row-normalized3×3 transitions, sumloglikelihoodratio/3 +originalnativeprototypecosmargin/.07; noinventedintermediateLN.',),
        (('final_margin_only',lambda f:fit_a032(f,transitions=False)),('reverse_actual_layer_order',lambda f:fit_a032(f,reverse=True)),
         ('all_actual_four_layer_profiles_RBF',lambda f:fit_a032(f,profiles=True))))
    # Original source specifies alpha>0.5, not the C_R-selected identity cut.
    for mid,fit,assumptions in (
        ('A033',fit_a033,('Actualraw post-finalLN,notunit orpreLN; radius2patch squarewindows cap64FPS, coverageptp>=.5, 2endpoint leastsquares condition<=100.',
                         'Actualleave-one-mixed-patch fit, medianalphacoverageerror<=.1; heldfeature residualq.99 reject. Residualexpweightedalpha acrosseligiblepairs, originalU(alpha,.5).')),
        ('A048',fit_a048,('UseonlyA033endpointpairswithotherblock realmixedpatchresidual<=bound andalphacoverageerror<=.1 on>=80%; sourceonlyvalidation, noQlabels.',
                         'Fivepredeterminedalpha knots+legitimateactualRmixes traincontinuous linear-spline basis at.25/.5/.75 withridge1e-4; meanpairalpha−std consistency penalty, originalU(alpha,.5).'))):
        recipes[mid]={'implementation_assumption':list(assumptions),'threshold':.5,'calibration':'explicitsourcealphacut; source-onlyresidual/mixedvalidation','renderer':'A_U_original_alpha_strict_.5'}
        methods[mid]=lambda ep,mid=mid,fit=fit,assumptions=assumptions:infer_a(ep,mid,fit,assumptions,fixed_threshold=.5)
    controls['A033__same_endpoint_alpha_without_residual_rejection']=lambda ep:infer_a(ep,'A033__same_endpoint_alpha_without_residual_rejection',lambda f:fit_a033(f,plain_endpoint_classifier=True),('same actualraw endpointpairs; controlonly',),fixed_threshold=.5)
    controls['A048__same_endpoint_direct_unmix']=lambda ep:infer_a(ep,'A048__same_endpoint_direct_unmix',fit_a033,('same actualraw endpoint source; direct constrainedalpha control',),fixed_threshold=.5)
    for mid in ('A033','A048'):
        controls[mid+'__actualraw_softlabel_ridge']=lambda ep,mid=mid:infer_a(ep,mid+'__actualraw_softlabel_ridge',fit_soft_raw_ridge,('actual rawLN andsame completeMR coverage; ordinarysoft regression control',),fixed_threshold=.5)
    for mid,name in (('A022','head_messages_last'),('A026','attention_special_to_patch'),('A032','midlayers'),('A033','raw_finalLN'),('A048','raw_finalLN')):
        requirements[mid]=[name,'source_bound_model_or_raw_cache_binding']
    for mid in ('A022','A026','A032','A033','A048'):
        methods[mid]=_observation_receipt(methods[mid])
        for cid in list(controls):
            if cid.startswith(mid+'__'):controls[cid]=_observation_receipt(controls[cid])
