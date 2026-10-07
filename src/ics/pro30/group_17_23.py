"""Source-defined PRO30 M17--M23. Controls never count as methods.

Frozen card source 56caf05b4b063147b598df85ca67d98f503c7c73440df61c5c8530ac02374729.
No query label, extra reference, downloaded model or altered numerical budget.
"""
from __future__ import annotations
from functools import partial
from dataclasses import replace
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import logsumexp
from scipy.ndimage import binary_dilation
from .common import (Result,validate,unit,readonly,require_artifact,ArtifactUnavailable,
    source_contract,degenerate_margin,finish,finish_highres,br_result,br_margin)
from . import helpers_17_23 as H
from .internal_17_23 import get_context

METHODS={};CONTROLS={'PRO30_B_R':br_result};REQUIREMENTS={};CONTRACTS={}


def _finish(ep,z,mid,info,start,*,highres=False,threshold=0.):
    elapsed=time.perf_counter()-start
    info=dict(info,**source_contract(int(mid.split('__')[0][-2:])),
        quality='unmeasured frozen Pro30 complete method',total_method_wall_seconds_before_renderer=elapsed,
        wall_cost_includes_any_actual_encoder_calls=True,cached_CPU_budget_seconds=2.,cost_gate_pass=elapsed<=2.,
        software_complete=True,query_GT_read=False,authorize_fixed600=False)
    if highres:return finish_highres(ep,z,mid,info,threshold=threshold)
    return finish(ep,np.asarray(z)-threshold,mid,info,invalid_value=-1.)


def _degenerate(ep,mid,start):
    result=degenerate_margin(ep)
    if result is None:return None
    z,info=result;return _finish(ep,z,mid,info,start)


def m17(ep,*,variant='inverse'):
    validate(ep);start=time.perf_counter();mid='PRO30_M17'+('' if variant=='inverse' else '__'+variant)
    # FullMR, not an old downsampled label pack, decides whether the target is
    # empty. A tiny target must not trigger empty-prompt fallback by shrinkage.
    wf,wvalid,canvas=H.reference_exact_areas(ep);area_error=float(np.max(np.abs(wf-ep.wf)))
    ep=replace(ep,wf=wf,wvalid=wvalid)
    degeneration=_degenerate(ep,mid,start)
    if degeneration is not None:return degeneration
    R=H.inverse_regions(ep,'r');Q=H.inverse_regions(ep,'q')
    labels=R['labels'];valid=R['valid'];s=len(R['mass']);fg=np.bincount(labels[valid],weights=canvas[valid],minlength=s);bg=R['mass']-fg
    if np.min(bg)<-1e-9:raise ValueError('Reference exact area labels exceed physical region area')
    bg=np.maximum(bg,0);ur=R['u0'] if variant=='U0_pooling' else R['u'];uq=Q['u0'] if variant=='U0_pooling' else Q['u']
    # Region may span several source-validation groups; never fit such a region.
    yy,xx=np.indices(labels.shape);groups=((np.minimum(yy*4//labels.shape[0],3)+2*np.minimum(xx*4//labels.shape[1],3))%4)
    memberships=np.zeros((s,4),bool)
    for k in range(4):memberships[np.unique(labels[valid&(groups==k)]),k]=True
    def scores(x,source):
        fc,fm,_=H.fps_lloyd(ur,fg*source,4,5);bc,bm,_=H.fps_lloyd(ur,bg*source,8,5)
        return H.role_lse(x,fc,fm)-H.role_lse(x,bc,bm),dict(FG_modes=len(fc),BG_modes=len(bc),FG_mass=fm.tolist(),BG_mass=bm.tolist())
    all_source=np.ones(s,bool);rscore,prototypes=scores(ur,all_source)
    thresholds=[];folds=[]
    for k in range(4):
        train=~memberships[:,k];held=memberships[:,k]&(memberships.sum(1)==1)
        fallback=fg[train].sum()<=0 or bg[train].sum()<=0
        if fallback:train=all_source;score=rscore
        else:score,_=scores(ur,train)
        cut,trainrisk=H.best_cut(score[train],fg[train],bg[train]);thresholds.append(cut)
        folds.append(dict(group=k,train_regions=int(train.sum()),held_regions=int(held.sum()),
            crossing_regions_excluded=int((memberships[:,k]&(memberships.sum(1)>1)).sum()),threshold=cut,train_balanced_error=trainrisk,
            held_diagnostic_balanced_error=H.risk(score[held],fg[held],bg[held],cut) if fg[held].sum()>0 and bg[held].sum()>0 else None,
            low_evidence_fallback=bool(fallback)))
    cut=float(np.median(thresholds));z,prototypes=scores(uq,all_source)
    potts_info={}
    if variant in ('scalar_region_pool','scalar_MEAN_region_pool','RGB_Potts'):
        rz=H.baseline_b(ep)-.5 if variant=='scalar_MEAN_region_pool' else br_margin(ep,query=ep.q)[0]
        z=(Q['A'].T@rz)/Q['D']
        cut=0.
        if variant=='RGB_Potts':
            # Same SLIC adjacency with an actual binary RGB Potts min-cut;
            # continuous graph smoothing is not mislabeled as a Potts solver.
            labels=Q['labels'];pairs=np.r_[np.c_[labels[:-1].ravel(),labels[1:].ravel()],np.c_[labels[:,:-1].ravel(),labels[:,1:].ravel()]]
            pairs=pairs[(pairs[:,0]>=0)&(pairs[:,1]>=0)&(pairs[:,0]!=pairs[:,1])];pairs=np.unique(np.sort(pairs,axis=1),axis=0)
            image,_=H.image_and_valid(ep,'q');image=image.astype(float)/255.;color=np.column_stack([np.bincount(labels[Q['valid']],weights=image[...,channel][Q['valid']],minlength=len(z))/Q['mass'] for channel in range(3)])
            capacity=.1*np.exp(-np.sum((color[pairs[:,0]]-color[pairs[:,1]])**2,axis=1)/.1)
            from ics.astra300.e_float_cut import exact_potts_cut
            selected,potts_info=exact_potts_cut(z,pairs,capacity);z=np.where(selected,1.,-1.)
    field=np.full(Q['labels'].shape,-1.);field[Q['valid']]=z[Q['labels'][Q['valid']]]
    return _finish(ep,field,mid,dict(active=True,reference_inverse=R['info'],query_inverse=Q['info'],folds=folds,
        output_threshold=cut,prototype_info=prototypes,lambda_inverse=.1,temperature=.07,solver='shared sparse LU for all actual raw channels',
        original_pack_reference_coverage_max_disagreement=area_error,reference_fullmask_area_recomputed_before_empty_prompt_test=True,
        control_Potts_solver=potts_info,
        implementation_assumption='classic deterministic Lab SLIC10 iterations, connectivity minarea=.5 mean; role FPS5Lloyd; ties smallest abs cut',
        partition_labels_sha256=H.cached(ep,'M17_Qlabelhash',lambda:__import__('hashlib').sha256(Q['labels'].tobytes()).hexdigest())),start,highres=True,threshold=cut)


def _hyperplanes(ep,source='query'):
    x,v=(ep.q,ep.q_valid) if source=='query' else (ep.r,ep.wvalid)
    centers,mass,assignment=H.fps_lloyd(x,v,32,10)
    if len(centers)<2:return None
    sim=H.mm(centers,centers.T);pairs=set()
    for i in range(len(centers)):
        remaining=np.array([j for j in range(len(centers)) if j!=i]);order=remaining[np.argsort(-sim[i,remaining],kind='stable')]
        for j in np.r_[order[:2],order[-2:]]:
            if np.linalg.norm(centers[i]-centers[j])>1e-10:pairs.add(tuple(sorted((i,int(j)))))
    pairs=sorted(pairs)
    if not pairs:return None
    norm=np.array([np.linalg.norm(centers[a]-centers[b]) for a,b in pairs]);rp=H.mm(ep.r,centers.T);qp=H.mm(ep.q,centers.T)
    R=np.column_stack([(rp[:,a]-rp[:,b])/n for (a,b),n in zip(pairs,norm)]);Q=np.column_stack([(qp[:,a]-qp[:,b])/n for (a,b),n in zip(pairs,norm)])
    R=np.c_[R,-R];Q=np.c_[Q,-Q];directions=np.vstack([(centers[a]-centers[b])/n for (a,b),n in zip(pairs,norm)]);directions=np.r_[directions,-directions]
    return centers,assignment,pairs,directions,R,Q


def m19(ep,*,variant='query_continuous'):
    validate(ep);start=time.perf_counter();mid='PRO30_M19'+('' if variant=='query_continuous' else '__'+variant)
    degeneration=_degenerate(ep,mid,start)
    if degeneration is not None:return degeneration
    bank=H.cached(ep,'M19_'+('reference' if variant=='reference_pairs' else 'query'),lambda:_hyperplanes(ep,'reference' if variant=='reference_pairs' else 'query'))
    if bank is None:return _finish(ep,H.proto(ep),mid,dict(active=False,inactive_reason='fewer two distinct query centers'),start)
    centers,assignment,pairs,directions,R,Q=bank;groups=H.groups4(ep.r_hw)
    four_valid=all(ep.wf[groups==k].sum()>0 and ep.wb[groups==k].sum()>0 for k in range(4))
    losses=np.empty((R.shape[1],4));thresholds=np.empty_like(losses)
    for c in range(R.shape[1]):
        for k in range(4):
            train=(groups!=k)&(ep.wvalid>0) if four_valid else ep.wvalid>0;held=(groups==k)&(ep.wvalid>0) if four_valid else train
            cut,_=H.best_cut(R[train,c],ep.wf[train],ep.wb[train]);thresholds[c,k]=cut
            losses[c,k]=H.risk(R[held,c],ep.wf[held],ep.wb[held],cut)
    chosen=int(np.lexsort((np.arange(R.shape[1]),np.ptp(losses,axis=1),np.mean(losses,axis=1)))[0]);cut,trainrisk=H.best_cut(R[:,chosen],ep.wf,ep.wb)
    z=Q[:,chosen]-cut
    if variant=='nearest_query_center':
        # Same query centers, source named modes: QP02 has its own historical control.
        # This explicit nearest-center control applies selected continuous direction
        # only to center scores and never claims to recreate the historical QP02.
        z=H.mm(centers,directions[chosen])[assignment]-cut
    split=0
    for group in range(len(centers)):
        points=z[assignment==group];split+=bool(np.any(points>0) and np.any(points<=0))
    return _finish(ep,z,mid,dict(active=True,centers=centers.tolist(),unordered_pairs=pairs,candidate_directions=len(directions),
        candidate_reference_label_exposure=False,selected_index=chosen,selected_direction=directions[chosen].tolist(),threshold=cut,
        four_group_risks=losses[chosen].tolist(),four_group_thresholds=thresholds[chosen].tolist(),full_source_balanced_error=trainrisk,
        low_evidence_selection=not four_valid,split_query_cluster_fraction=split/len(centers),
        implementation_assumption='query spherical FPS initialization and10 Lloyd rounds; nearest/farthest measured in query feature-space cosine'),start)


def m18(ep,*,variant='absolute'):
    validate(ep);start=time.perf_counter();mid='PRO30_M18'+('' if variant=='absolute' else '__'+variant)
    degeneration=_degenerate(ep,mid,start)
    if degeneration is not None:return degeneration
    if np.sum(ep.q_valid>0)<2:return _finish(ep,H.proto(ep),mid,dict(active=False,inactive_reason='fewer two valid nodes'),start)
    tree=H.ward_tree(ep);rids=np.arange(len(ep.r));F,fw=H.area_sample(rids,ep.wf);B,bw=H.area_sample(rids,ep.wb)
    def kernel(a,b):return np.exp((np.clip(H.mm(a,b.T),-1,1)-1)/.1)
    def quad(w,K,v):return float(H.mm(H.mm(w,K),v))
    def mmd(x,w,y,v):return quad(w,kernel(x,x),w)+quad(v,kernel(y,y),v)-2*quad(w,kernel(x,y),v)
    rF=ep.r[F];rB=ep.r[B];FF=quad(fw,kernel(rF,rF),fw);BB=quad(bw,kernel(rB,rB),bw)
    refgroups=H.groups4(ep.r_hw);groupbanks=[]
    for k in range(4):
        ids,weights=H.area_sample(rids,ep.wf*(refgroups==k))
        if len(ids):groupbanks.append((ep.r[ids],weights))
    tolerances=[mmd(x,w,y,v) for i,(x,w) in enumerate(groupbanks) for y,v in groupbanks[i+1:]]
    tF=float(np.quantile(tolerances,.9)+.05) if tolerances else .05
    u=H.proto(ep);node_values={};metrics={}
    for node in tree['nodes']:
        members=tree['members'][node];ids,w=H.area_sample(members,ep.q_valid[members]);x=ep.q[ids]
        QQ=quad(w,kernel(x,x),w);QF=quad(w,kernel(x,rF),fw);QB=quad(w,kernel(x,rB),bw)
        DF=QQ+FF-2*QF;DB=QQ+BB-2*QB;v=min(DB-DF,tF-DF)
        if variant=='relative_MMD':v=DB-DF
        elif variant=='pointwise_KDE':v=QF-QB
        elif variant=='mean_variance':
            xm=np.average(x,axis=0,weights=w);fm=np.average(rF,axis=0,weights=fw);bm=np.average(rB,axis=0,weights=bw)
            xv=float(np.sum(w*np.sum((x-xm)**2,axis=1)));fv=float(np.sum(fw*np.sum((rF-fm)**2,axis=1)));bv=float(np.sum(bw*np.sum((rB-bm)**2,axis=1)))
            DF=float(np.sum((xm-fm)**2)+(xv-fv)**2);DB=float(np.sum((xm-bm)**2)+(xv-bv)**2);v=min(DB-DF,.05-DF)
        node_values[node]=v;metrics[node]=dict(QQ=QQ,RR_F=FF,RR_B=BB,QR_F=QF,QR_B=QB,DF=DF,DB=DB,v=v,mass=float(tree['mass'][node]),samples=len(ids))
    values={};choices={}
    # Child indices precede their parent; leaf decisions contain absolute gate.
    for node in tree['nodes']:
        area=tree['mass'][node];v=node_values[node]
        if node in tree['leaves']:
            gain=area*(v+.25*np.average(u[tree['members'][node]],weights=ep.q_valid[tree['members'][node]]))
            values[node]=max(0.,gain);choices[node]='foreground' if gain>0 else 'background'
        else:
            left,right=tree['children'][node];split=values[left]+values[right];fg=area*v
            # Explicit background > split > foreground tie priority.
            if max(split,fg)<=0:values[node]=0.;choices[node]='background'
            elif split>=fg:values[node]=split;choices[node]='split'
            else:values[node]=fg;choices[node]='foreground'
    z=np.full(len(ep.q),-1.);selected=[];todo=list(tree['roots'])
    while todo:
        node=todo.pop()
        if choices[node]=='foreground':z[tree['members'][node]]=1.;selected.append(node)
        elif choices[node]=='split':todo.extend(tree['children'][node])
    if variant=='same_count_prototype':
        count=int(np.sum(z>0));z[:]=-1.;valid=np.flatnonzero(ep.q_valid>0);order=valid[np.argsort(-u[valid],kind='stable')];z[order[:count]]=1.
    return _finish(ep,z,mid,dict(active=True,tree_seconds=tree['seconds'],leaves=len(tree['leaves']),nodes=len(tree['nodes']),
        tF=tF,reference_tolerance_pairs=len(tolerances),low_evidence_tolerance=not bool(tolerances),selected_nodes=selected,
        node_terms={str(k):v for k,v in metrics.items()},absolute_query_query_term_used=variant in ('absolute','same_count_prototype'),
        kernel_temperature=.1,max_area_samples=64,implementation_assumption='|C| is valid token area mass; no node-size threshold selection'),start)


def m20(ep,*,variant='full'):
    validate(ep);start=time.perf_counter();mid='PRO30_M20'+('' if variant=='full' else '__'+variant)
    degeneration=_degenerate(ep,mid,start)
    if degeneration is not None:return degeneration
    if min(np.sum(ep.wvalid>0),np.sum(ep.q_valid>0))<2:return _finish(ep,H.proto(ep),mid,dict(active=False,inactive_reason='fewer two valid graph nodes'),start)
    R=H.cached(ep,'M20_Rspectrum',lambda:H.spectrum(ep.r,ep.wvalid,ep.r_hw));Q=H.cached(ep,'M20_Qspectrum',lambda:H.spectrum(ep.q,ep.q_valid,ep.q_hw))
    ir,vr,Wr,Lr,pr,lr,rinfo=R;iq,vq,Wq,Lq,pq,lq,qinfo=Q;d=ep.r.shape[1];dim=min(64,d)
    projection=np.linalg.qr(np.random.RandomState(0).normal(size=(d,dim)),mode='reduced')[0]
    FR=np.c_[H.mm(ep.r[ir],projection),np.ones(len(ir))];FQ=np.c_[H.mm(ep.q[iq],projection),np.ones(len(iq))]
    AR=H.mm(pr.T,vr[:,None]*FR);AQ=H.mm(pq.T,vq[:,None]*FQ)
    if variant=='descriptor_shuffle':AQ=AQ[:,np.r_[np.random.RandomState(0).permutation(dim),dim]]
    constant_r=AR[:,-1];constant_q=AQ[:,-1];C=np.zeros((len(lq),len(lr)))
    # Because phi0 is exactly constant, constraint fixes column0, including all
    # query rows. More general rounding residual is checked explicitly below.
    C[:,0]=constant_q/constant_r[0];lam=0. if variant=='no_spectral_commutation' else .1
    if variant=='diagonal_C':
        for i in range(1,min(C.shape)):
            C[i,i]=AQ[i]@AR[i]/(AR[i]@AR[i]+lam*(lr[i]-lq[i])**2+.01)
    else:
        G=H.mm(AR[1:],AR[1:].T)
        for i in range(len(lq)):
            matrix=G+np.diag(lam*(lr[1:]-lq[i])**2+.01)
            rhs=H.mm(AR[1:],AQ[i]-C[i,0]*AR[0]);C[i,1:]=np.linalg.solve(matrix,rhs)
    y=2*ep.wf[ir]/vr-1;a=H.mm(pr.T,vr*y);zq=H.mm(pq,H.mm(C,a));before=zq.copy()
    rawL=sparse.diags(np.asarray(Wq.sum(1)).ravel())-Wq
    if variant=='descriptor_ridge':
        normal=H.mm(FR.T,vr[:,None]*FR)+.01*np.eye(FR.shape[1]);coeff=np.linalg.solve(normal,H.mm(FR.T,vr*y));zq=H.mm(FQ,coeff);before=zq.copy()
    elif variant=='ordinary_graph_readout':zq=H.proto(ep)[iq];before=zq.copy()
    zq=splu((sparse.eye(len(iq))+.2*rawL).tocsc()).solve(zq);z=np.full(len(ep.q),-1.);z[iq]=zq
    return _finish(ep,z,mid,dict(active=True,reference_spectrum=rinfo,query_spectrum=qinfo,R_eigenvalues=lr.tolist(),Q_eigenvalues=lq.tolist(),
        map_rank=int(np.linalg.matrix_rank(C)),constant_constraint_residual=float(np.linalg.norm(H.mm(C,constant_r)-constant_q)),
        descriptor_relative_residual=float(np.linalg.norm(H.mm(C,AR)-AQ)/max(np.linalg.norm(AQ),H.EPS)),C=C.tolist(),
        lambda_commutation=lam,ridge=.01,post_smoothing_difference=float(np.linalg.norm(zq-before)),foreground_valid_mass=float(vq[zq>0].sum()),
        graph_constants={'mutual_k':20,'four_neighbor_weight':.05,'cos_temperature':.07,'mass_normalized_to_one':False},
        implementation_assumption='literal symmetric sum_ij smoothing produces I+.2 unnormalized graph Laplacian; eigsh tol1e-8 max10000'),start)


def _groups8(hw):
    yy,xx=np.indices(hw);return (((yy*64//hw[0])//8+2*(xx*64//hw[1]//8))%4).ravel()


def _semantic(x,ep,A):
    xf=unit(np.sum(x*(ep.wf*A)[:,None],axis=0));xb=unit(np.sum(x*(ep.wb*A)[:,None],axis=0));return xf-xb


def _changed_b(ep,rraw,qraw):
    operator=require_artifact(ep,'pro30_B_operator')
    if not callable(operator):raise ArtifactUnavailable('Actual complete B altered-encoder operator required')
    value=operator(ep,reference=unit(rraw),query=unit(qraw),apd='native',fixed_native_gate=True)
    if not isinstance(value,tuple) or len(value)!=2:raise ArtifactUnavailable('Complete B operator returns (b, producer/cost info)')
    value,info=value
    if not isinstance(info,dict) or info.get('complete_FoRIS_MEAN16') is not True or info.get('native_gate_locked') is not True or info.get('query_GT_read') is not False:
        raise ArtifactUnavailable('Changed feature B requires complete fixed original APD-gate pipeline, without QGT')
    value=np.asarray(value,float)
    if value.size!=len(ep.q) or not np.isfinite(value).all():raise ValueError('Altered encoder B must return full original b field')
    return value.ravel(),info


def _gram_drift(base,candidate,valid):
    ids=np.flatnonzero(valid>0);ids=ids[np.unique(np.linspace(0,len(ids)-1,min(256,len(ids))).astype(int))]
    x=base[ids];y=candidate[ids]
    return float(np.mean((H.mm(x,x.T)-H.mm(y,y.T))**2))


def _gram_gate_update(original,candidate,current=None):
    # Original GramLoop gate uses all native patches, blockwise only to reduce
    # memory; not an approximation to a sampled Gram or diagonal.
    x=unit(original[5:]);y=unit(candidate[5:]);drift=np.empty(len(x))
    for start in range(0,len(x),64):
        difference=H.mm(y[start:start+64],y.T)-H.mm(x[start:start+64],x.T)
        drift[start:start+len(difference)]=np.mean(difference*difference,axis=1)
    current=original if current is None else current
    gate=.56*np.exp(-drift/max(float(np.median(drift)),1e-6));output=current.copy();output[5:]+=gate[:,None]*(candidate[5:]-current[5:])
    # Formula supplies patch gates only: preserve prefix states in this control.
    return output,dict(whole_patch_Gram=True,median_drift=float(np.median(drift)),gate_min=float(gate.min()),gate_max=float(gate.max()),prefix_state_preserved=True)


def m21(ep,*,variant='reference_accept'):
    validate(ep);start=time.perf_counter();mid='PRO30_M21'+('' if variant=='reference_accept' else '__'+variant)
    degeneration=_degenerate(ep,mid,start)
    if degeneration is not None:return degeneration
    ctx=get_context(ep);receipt0=dict(ctx.receipt);states={role:ctx.state(ep,role,24,prefix=True) for role in ('r','q')}
    replay={role:ctx.suffix(ep,role,states[role],start_layer=22,normalize=False) for role in ('r','q')}
    risk_rows=[];gram=[];groups=_groups8(ep.r_hw);validfolds=[];representations=[]
    for gamma in (0.,.15,.30):
        feature=unit(ctx.normalize(states['r']+gamma*(replay['r']-states['r']),unit_output=False)[5:]);representations.append(feature)
        losses=[]
        for k in range(4):
            held=(groups==k)&(ep.wvalid>0);halo=binary_dilation(held.reshape(ep.r_hw),np.ones((3,3),bool)).ravel()
            train=(groups!=k)&~halo&(ep.wvalid>0)
            valid=ep.wf[train].sum()>0 and ep.wb[train].sum()>0 and ep.wf[held].sum()>0 and ep.wb[held].sum()>0
            if gamma==0:validfolds.append(bool(valid))
            if valid:
                axis=_semantic(feature,ep,train);logit=H.mm(feature[held],axis)/.07;losses.append(H.balanced_loss(logit,ep.wf[held],ep.wb[held]))
            else:losses.append(np.nan)
        risk_rows.append(losses);gram.append(_gram_drift(representations[0],feature,ep.wvalid))
    risk_rows=np.asarray(risk_rows);valid=np.asarray(validfolds,bool);gamma=0.;selected=0
    if valid.any():
        delta=risk_rows[:,valid]-risk_rows[0,valid];objectives=np.max(delta,axis=1)+.1*np.array(gram);selected=int(np.argmin(objectives))
        if selected and np.all(delta[selected]<=0) and np.mean(delta[selected])<=-.01:gamma=(0.,.15,.30)[selected]
        else:selected=0
    control_gram={}
    if variant in ('fixed_gamma_15','native_replay_average'):gamma=.15 if variant=='fixed_gamma_15' else .5
    if variant in ('same_budget_Gram_gate','original_GramLoop'):
        rounds=2 if variant=='original_GramLoop' else 1;new={}
        for role in ('r','q'):
            current=states[role];records=[]
            for iteration in range(rounds):
                candidate=replay[role] if iteration==0 else ctx.suffix(ep,role,current,start_layer=22,normalize=False)
                current,record=_gram_gate_update(states[role],candidate,current);records.append(record)
            new[role]=ctx.normalize(current,unit_output=False)[5:];control_gram[role]=records
        b,hostinfo=_changed_b(ep,new['r'],new['q'])
    elif gamma==0:b=H.baseline_b(ep);hostinfo={'cached_native_B':True}
    else:
        new={role:ctx.normalize(states[role]+gamma*(replay[role]-states[role]),unit_output=False)[5:] for role in ('r','q')}
        b,hostinfo=_changed_b(ep,new['r'],new['q'])
    risk_record=[[float(x) if np.isfinite(x) else None for x in row] for row in risk_rows]
    return _finish(ep,b-.5,mid,dict(active=bool(gamma>0 or control_gram),gamma=gamma,reference_fold_logloss=risk_record,
        valid_reference_folds=validfolds,reference_Gram_drift=gram,control_Gram_records=control_gram,
        replay_blocks_zero_based=[21,22,23],replayed_full_prefix_state=True,host=hostinfo,
        internal_cost_receipt=dict(ctx.receipt),internal_receipt_before=receipt0,
        implementation_assumption='invalid reference folds omitted; none returns gamma0; GramLoop patch gates leave prefix states unchanged, pending exact official control parity review'),start)


def _trajectory(ep,ctx,A):
    raw={role:[ctx.state(ep,role,k) for k in (8,12,16,20,24)] for role in ('r','q')}
    x={role:[ctx.normalize(value) for value in raw[role]] for role in ('r','q')}
    axes=[_semantic(value,ep,A) for value in x['r']]
    a={role:np.column_stack([H.mm(value,axis) for value,axis in zip(x[role],axes)]) for role in ('r','q')}
    psi={};c={}
    for role in ('r','q'):
        delta=np.diff(a[role],axis=1);psi[role]=np.c_[a[role],delta,delta[:,:-1]*delta[:,1:],np.abs(delta)]
        c[role]=np.c_[np.ones(len(a[role])),a[role][:,-1],a[role][:,-1]**2,np.linalg.norm(raw[role][-1],axis=1)]
    valid=ep.wvalid>0;n=valid.sum();penalty=np.diag([0.,1.,1.,1.]);cr=c['r'][valid]
    B=np.linalg.solve(H.mm(cr.T,cr)/n+penalty,H.mm(cr.T,psi['r'][valid])/n)
    e={role:psi[role]-H.mm(c[role],B) for role in ('r','q')}
    return raw,x,a,psi,c,e,B


def _full_state_control(ep,ctx,layers,*,kernel=False):
    from .common import fit_br,equal_mass_sample
    r=np.concatenate([ctx.normalize(ctx.state(ep,'r',k)) for k in layers],axis=1)/np.sqrt(len(layers))
    q=np.concatenate([ctx.normalize(ctx.state(ep,'q',k)) for k in layers],axis=1)/np.sqrt(len(layers))
    if not kernel:return fit_br(ep,r).predict(q)
    ids,target,weight=equal_mass_sample(ep);KR=np.exp((np.clip(H.mm(r,r[ids].T),-1,1)-1)/.1);KQ=np.exp((np.clip(H.mm(q,r[ids].T),-1,1)-1)/.1)
    # Same sample exposure; actual full-state kernel, no compressed final proxy.
    G=KR[ids];sw=np.sqrt(weight);alpha=np.linalg.solve(sw[:,None]*G*sw[None]+.01*np.eye(len(ids)),sw*target)
    return H.mm(KQ,sw*alpha)


def m22(ep,*,variant='conditional_trajectory'):
    validate(ep);start=time.perf_counter();mid='PRO30_M22'+('' if variant=='conditional_trajectory' else '__'+variant)
    degeneration=_degenerate(ep,mid,start)
    if degeneration is not None:return degeneration
    b=H.baseline_b(ep);A,C,okay=H.anchor_calibration(ep)
    if not okay:return _finish(ep,b-.5,mid,dict(active=False,inactive_reason='anchor/calibration blocks or role area mass below fixed minima'),start)
    ctx=get_context(ep);raw,x,a,psi,c,e,B=_trajectory(ep,ctx,A)
    if variant in ('five_layer_linear','five_layer_kernel','two_endpoints','final_layer'):
        layers=(8,12,16,20,24) if variant.startswith('five') else (8,24) if variant=='two_endpoints' else (24,)
        z=_full_state_control(ep,ctx,layers,kernel=variant.endswith('kernel'))
        return _finish(ep,z,mid,dict(active=True,internal_cost_receipt=dict(ctx.receipt),readout='same B_R reference exposure and same physical renderer',layers=list(layers)),start)
    re=np.c_[c['r'],e['r']];qe=np.c_[c['q'],e['q']]
    if variant=='raw_trajectory':re=np.c_[c['r'],psi['r']];qe=np.c_[c['q'],psi['q']]
    elif variant=='corresponding_regularizer':
        # Input (c,psi) is transformed by the exact fixed B before the same
        # center/MAD/clip and penalty; this proves reparameterization equivalence.
        re=np.c_[c['r'],psi['r']-H.mm(c['r'],B)];qe=np.c_[c['q'],psi['q']-H.mm(c['q'],B)]
    residual,info=H.residual_readout(ep,re,qe,c['r'],c['q'],A,C)
    info.update(actual_layers=[8,12,16,20,24],native_per_layer_finalLN=False,custom_normalization='same native finalLN parameters applied to each true raw state',
        trajectory_projection_B=B.tolist(),unlabeled_projection_uses_C_features=True,C_labels_used_in_descriptors=False,
        conditional_reparameterization_creates_information=False,internal_cost_receipt=dict(ctx.receipt),
        modifiable_native_node_fraction=float(np.mean((b>.3)&(b<.7))),max_residual_magnitude=float(np.max(np.abs(residual))))
    return _finish(ep,H.bounded_fusion(b,residual),mid,info,start)


def _response_descriptors(ep,ctx,A,*,local=False,ordinary=False,margin_only=False):
    raw20={role:ctx.state(ep,role,20,prefix=True) for role in ('r','q')}
    x20={role:unit(raw20[role][5:]) for role in ('r','q')}
    x24={role:ctx.normalize(ctx.state(ep,role,24)) for role in ('r','q')}
    axis20=_semantic(x20['r'],ep,A);axis24=_semantic(x24['r'],ep,A);d=unit(_semantic(raw20['r'][5:],ep,A));w=unit(axis24)
    if min(np.linalg.norm(d),np.linalg.norm(w))<1e-6:return None
    descriptors={};simple={};stability={};directions={}
    for role in ('r','q'):
        valid=ep.wvalid>0 if role=='r' else ep.q_valid>0;n=len(valid);rng=np.random.RandomState(0);codes=rng.choice((-1.,1.),size=(2,n))
        simple[role]=np.c_[H.mm(x20[role],axis20),H.mm(x24[role],axis24)]
        if margin_only:
            descriptors[role]=simple[role]
            continue
        tangent_all=np.zeros_like(raw20[role]);tangent_all[5:]=valid[:,None]*d
        tangents=[tangent_all]
        for code in codes:
            tangent=np.zeros_like(raw20[role]);tangent[5:]=valid[:,None]*code[:,None]*d;tangents.append(tangent)
        directions[role]=tangents
        if ordinary:
            # Exactly three actual suffix evaluations per image, same supplied
            # direction set. This simple state-perturbation average is not JVP.
            outputs=ctx.ordinary_perturbations(ep,role,raw20[role],tangents)
            descriptors[role]=np.mean([unit(value[5:]) for value in outputs],axis=0)
            continue
        responses=[ctx.jvp(ep,role,raw20[role],tangent,local_only=local)[1][5:] for tangent in tangents]
        all_response=responses[0];self1=codes[0,:,None]*responses[1];self2=codes[1,:,None]*responses[2];self_response=.5*(self1+self2)
        dot=np.sum(self_response*all_response,axis=1);ns=np.linalg.norm(self_response,axis=1);na=np.linalg.norm(all_response,axis=1)
        descriptors[role]=np.c_[simple[role],H.mm(self_response,w),H.mm(all_response,w),ns,na,np.linalg.norm(all_response-self_response,axis=1),dot/np.maximum(ns*na,1e-6)]
        stability[role]=dict(two_code_response_relative_difference=float(np.linalg.norm(self1-self2)/max(np.linalg.norm(self_response),H.EPS)),
            code_self_dot_correlation=float(np.corrcoef(H.mm(self1,w),H.mm(self2,w))[0,1]) if np.std(H.mm(self1,w))*np.std(H.mm(self2,w))>0 else None)
    return descriptors,simple,stability,d,w,raw20,directions


def m23(ep,*,variant='semantic_JVP'):
    validate(ep);start=time.perf_counter();mid='PRO30_M23'+('' if variant=='semantic_JVP' else '__'+variant)
    degeneration=_degenerate(ep,mid,start)
    if degeneration is not None:return degeneration
    b=H.baseline_b(ep);A,C,okay=H.anchor_calibration(ep)
    if not okay:return _finish(ep,b-.5,mid,dict(active=False,inactive_reason='anchor/calibration fixed source eligibility'),start)
    ctx=get_context(ep)
    if variant=='full_state_kernel':
        z=_full_state_control(ep,ctx,(20,24),kernel=True)
        return _finish(ep,z,mid,dict(active=True,actual_layers=[20,24],internal_cost_receipt=dict(ctx.receipt)),start)
    margin_only=variant=='two_native_margins'
    response=_response_descriptors(ep,ctx,A,local=variant=='token_MLP_LN_JVP',
        ordinary=variant=='same_budget_state_perturbation_average',margin_only=margin_only)
    if response is None:return _finish(ep,b-.5,mid,dict(active=False,inactive_reason='anchor semantic direction degenerate'),start)
    phi,simple,stability,d,w,raw20,directions=response
    if variant=='same_budget_state_perturbation_average':
        from .common import fit_br
        z=fit_br(ep,phi['r']).predict(phi['q'])
        return _finish(ep,z,mid,dict(active=True,perturbation_epsilon=1e-3,actual_suffix_calls=6,internal_cost_receipt=dict(ctx.receipt)),start)
    if variant=='two_native_margins':phi=simple
    residual,info=H.residual_readout(ep,phi['r'],phi['q'],simple['r'],simple['q'],A,C)
    descriptor_order=(['margin20','margin24'] if margin_only else
        ['margin20','margin24','w_self','w_all','norm_self','norm_all','norm_all_minus_self','cos_self_all'])
    info.update(semantic_direction20=d.tolist(),semantic_output_direction=w.tolist(),directions_frozen_after_anchor=True,
        descriptor_order=descriptor_order,
        self_estimator_codes=0 if margin_only else 2,space_code_seed=None if margin_only else 0,
        code_independence='not used by the two-margin control' if margin_only else 'independent IID Rademacher codes across all native patch IDs; padded tangent zero',
        response_stability=stability,actual_suffix_JVPs=0 if margin_only else 6,
        internal_cost_receipt=dict(ctx.receipt),modifiable_native_node_fraction=float(np.mean((b>.3)&(b<.7))),
        max_residual_magnitude=float(np.max(np.abs(residual))),implementation_assumption='native semantic margins use unit role-mean cosine difference; tangent d unit raw-H20 role-mean difference')
    return _finish(ep,H.bounded_fusion(b,residual),mid,info,start)


for mid,fn in ((17,m17),(18,m18),(19,m19),(20,m20),(21,m21),(22,m22),(23,m23)):
    METHODS['PRO30_M'+str(mid)]=fn

for mid,fn,variants in (
    (17,m17,('U0_pooling','scalar_region_pool','scalar_MEAN_region_pool','RGB_Potts')),
    (18,m18,('relative_MMD','pointwise_KDE','mean_variance','same_count_prototype')),
    (19,m19,('reference_pairs','nearest_query_center')),
    (20,m20,('descriptor_ridge','ordinary_graph_readout','diagonal_C','no_spectral_commutation','descriptor_shuffle')),
    (21,m21,('same_budget_Gram_gate','original_GramLoop','fixed_gamma_15','native_replay_average')),
    (22,m22,('raw_trajectory','corresponding_regularizer','five_layer_linear','five_layer_kernel','two_endpoints','final_layer')),
    (23,m23,('two_native_margins','full_state_kernel','token_MLP_LN_JVP','same_budget_state_perturbation_average'))):
    for variant in variants:CONTROLS['PRO30_M'+str(mid)+'__'+variant]=partial(fn,variant=variant)

def _qp02_control(ep):
    from ics.cpu100.query_partition import qp02
    start=time.perf_counter()
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):result=qp02(ep)
    if not np.isfinite(result.margin).all():raise FloatingPointError('Original QP02 control nonfinite')
    return _finish(ep,result.margin,'PRO30_M19__QP02_original_fixed',dict(result.info,inherited_control=True),start)

CONTROLS['PRO30_M19__QP02_original_fixed']=_qp02_control
REQUIREMENTS.update({
    'PRO30_M17':['r_raw','q_raw','r_rgb','q_rgb','complete reference_mask','physical geometry'],
    'PRO30_M18':['native unit R/Q','complete MR area weights','geometry'],
    'PRO30_M19':['native unit R/Q','complete MR area weights','geometry'],
    'PRO30_M20':['native unit R/Q','complete MR area weights','geometry'],
    'PRO30_M21':['pro30_internal_context','r_raw','q_raw','mean.continuous','pro30_B_operator'],
    'PRO30_M22':['pro30_internal_context','r_raw','q_raw','mean.continuous'],
    'PRO30_M23':['pro30_internal_context','r_raw','q_raw','mean.continuous']})
for mid in METHODS:
    CONTRACTS[mid]=dict(source_contract(int(mid[-2:])),revision='source_fixed_v1',required_fields=REQUIREMENTS[mid],
        input_contract='N+actual_X' if int(mid[-2:])>=21 or mid.endswith('17') else 'N',
        query_GT_read=False,solver_budget_lowered=False,controls=[k for k in CONTROLS if k.startswith(mid+'__')]+['PRO30_B_R'],
        renderer='physical continuous-highres→original strict cut, no64 shrink' if mid.endswith('17') else 'same source CPU100 two-threshold physical renderer',
        validation_protocol='same frozen600 only after actual singlepair cached CPU cost gate; no200screen')
