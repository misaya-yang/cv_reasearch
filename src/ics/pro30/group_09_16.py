"""Original Pro30 M09--M16; implementations are not natural efficacy claims."""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from .common import (Result,artifact,ArtifactUnavailable,br_margin,br_result,fit_br,finish,finish_highres,
                     degenerate_margin,unit,readonly,source_contract)
from .observations_09_16 import prepare_last,source_statistics
METHODS={};REQUIREMENTS={};ASSUMPTIONS={}
class _Controls(dict):
    def __setitem__(self,name,fn):
        def tagged(ep):
            out=fn(ep);underlying=out.info.get('method_id');out.info=dict(out.info,method_id=name,registered_control_id=name,underlying_method_id=underlying,control=True)
            return out
        super().__setitem__(name,tagged)
CONTROLS=_Controls()


def prepare_m09_m10(ep,*,include_message_sources=True):
    """Single honest X-field preparation entry; record its cost separately.

    Call once for the actual single episode, then time the registered methods
    using these shared derived caches. Cold extraction is not a2-second claim.
    """
    started=time.perf_counter();observed=prepare_last(ep)
    fields={'final_register_r':observed['r']['register'],'final_register_q':observed['q']['register'],'observation_receipt':observed['receipt']}
    if include_message_sources:
        for role in('r','q'):
            message,mass,info=source_statistics(ep,role)
            fields['attention_source_statistics_'+role]={'projected_messages':message,'mean_head_mass':mass,'producer':info}
    fields['preparation_seconds_this_call']=time.perf_counter()-started
    fields['separate_actual_after_preparation_method_budget_seconds']=2.
    return fields


def _return(ep,z,mid,info,highres=False):
    info=dict(info,**source_contract(int(mid[-2:])),implementation_assumptions=ASSUMPTIONS.get(mid,[]))
    field=np.asarray(z,float).reshape(ep.q_hw).copy();field.ravel()[ep.q_valid<=0]=-1.
    return finish_highres(ep,field,mid,info)if highres else finish(ep,field,mid,info)


def _weighted_mean(x,w):return np.sum(x*w[:,None],axis=0)/max(float(np.sum(w)),1e-12)


def _standardize(r,q,w):
    mean=_weighted_mean(r,w);sd=np.sqrt(_weighted_mean((r-mean)**2,w));sd=np.maximum(sd,1e-4)
    return (r-mean)/sd,(q-mean)/sd,{'reference_mean':mean.tolist(),'reference_std_floor1e-4':sd.tolist()}


def _source_features(ep,random=False):
    rm,ra,ri=source_statistics(ep,'r',random_groups=random);qm,qa,qi=source_statistics(ep,'q',random_groups=random)
    directions=unit(np.array([_weighted_mean(rm[:,c],ep.wf)-_weighted_mean(rm[:,c],ep.wb)for c in range(4)]))
    rt=np.einsum('ncd,cd->nc',rm,directions,optimize=False);qt=np.einsum('ncd,cd->nc',qm,directions,optimize=False)
    r=np.column_stack((rt,ra,np.linalg.norm(rm,axis=-1)));q=np.column_stack((qt,qa,np.linalg.norm(qm,axis=-1)))
    return r,q,{'reference_source_direction_norms':np.linalg.norm(directions,axis=1).tolist(),'r_source_audit':ri,'q_source_audit':qi}


def infer_m09(ep,control=None):
    mid='PRO30_M09';deg=degenerate_margin(ep)
    if deg is not None:return _return(ep,deg[0],mid,deg[1])
    started=time.perf_counter()
    if control=='final_LN_same12':
        observed=prepare_last(ep);K=min(12,ep.r.shape[1]);P=np.linalg.qr(np.random.default_rng(9009).normal(size=(ep.r.shape[1],K)))[0][:,:K]
        r=np.asarray(ep.r)@P;q=np.asarray(ep.q)@P
        if K<12:r=np.pad(r,((0,0),(0,12-K)));q=np.pad(q,((0,0),(0,12-K)))
        info={'control':'same12 auxiliary channels from final-LN only; identical necessary observation entry is charged but not used','actual_shared_observation':observed['receipt']}
    else:r,q,info=_source_features(ep,random=control=='random_key_groups')
    if control=='mass_only':r=r[:,4:8];q=q[:,4:8]
    elif control in('total_same12','full_total','full_sources'):
        rm,ra,ri=source_statistics(ep,'r');qm,qa,qi=source_statistics(ep,'q');rt=rm.sum(axis=1);qt=qm.sum(axis=1)
        if control=='total_same12':
            # Same12 added channels from total message only: class direction,
            # total norm and ten deterministic unsupervised reference PCs.
            _,_,V=np.linalg.svd(rt[ep.wvalid>0]-_weighted_mean(rt,ep.wvalid),full_matrices=False);K=min(10,len(V));directions=V[:K]
            direction=unit(_weighted_mean(rt,ep.wf)-_weighted_mean(rt,ep.wb))
            r=np.column_stack((rt@direction,np.linalg.norm(rt,axis=1),rt@directions.T));q=np.column_stack((qt@direction,np.linalg.norm(qt,axis=1),qt@directions.T))
            if K<10:r=np.pad(r,((0,0),(0,10-K)));q=np.pad(q,((0,0),(0,10-K)))
        elif control=='full_total':r=rt;q=qt
        else:r=rm.reshape(len(rm),-1);q=qm.reshape(len(qm),-1)
    r,q,standard=_standardize(r,q,ep.wvalid);rr=unit(np.column_stack((ep.r,.5*r)));qq=unit(np.column_stack((ep.q,.5*q)))
    z,fit=br_margin(ep,rr,qq);info.update(reference_standardizer=standard,fit=fit,actual_shared_observation=prepare_last(ep)['receipt'],readout_seconds_including_source_statistics=time.perf_counter()-started,control=control)
    return _return(ep,z,mid,info)


METHODS['PRO30_M09']=infer_m09
REQUIREMENTS['PRO30_M09']=['actual_native_last_attention_QKV_after_QKnorm_RoPE_scale_mask','actual_output_projection_weight','physical_patch_validity','native_RGB_and_frozen_cpu_encoder_if_internal_fields_not_cached']
ASSUMPTIONS['PRO30_M09']=['A partially physical patch is a physical key; wholly padded keys form the separate padding group. All query rows receive full output, with renderer using exact q geometry.',
                           'Reference statistics use all legal valid-area weights; sources are grouped before output projection and exclude shared output bias and LayerScale.']
for name,mode in [('total_same12','total_same12'),('random_key_groups','random_key_groups'),('mass_only','mass_only'),('full_total_concat_linear','full_total'),('full_sources_concat_linear','full_sources'),('final_LN_same12','final_LN_same12')]:
    CONTROLS['PRO30_control_M09_'+name]=(lambda ep,mode=mode:infer_m09(ep,mode))
CONTROLS['PRO30_control_M09_B_R']=br_result


def _register_basis(ep,replace_register=None):
    observed=prepare_last(ep);r=np.asarray(observed['r']['register'],float);q=np.asarray(observed['q']['register'],float)
    if r.ndim!=2 or q.ndim!=2 or r.shape[1]!=ep.r.shape[1]or q.shape[1]!=ep.q.shape[1]:raise ArtifactUnavailable('Real final-LN registers must share actual final-LN patch channels')
    if replace_register=='mean':r=np.repeat(r.mean(axis=0,keepdims=True),len(r),axis=0);q=np.repeat(q.mean(axis=0,keepdims=True),len(q),axis=0)
    if replace_register=='swap':r,q=q,r
    S=np.r_[r,q];_,singular,V=np.linalg.svd(S,full_matrices=False);rank=min(8,int(np.sum(singular>max(S.shape)*np.finfo(float).eps*(singular[0]if len(singular)else 0))))
    return V[:rank].T,singular,observed['receipt']


def infer_m10(ep,control=None):
    mid='PRO30_M10';deg=degenerate_margin(ep)
    if deg is not None:return _return(ep,deg[0],mid,deg[1])
    U,singular,receipt=_register_basis(ep,'mean'if control=='register_means'else'swap'if control=='swap_RQ_registers'else None)
    r,q=np.asarray(ep.r),np.asarray(ep.q);rf=r@U
    mf=_weighted_mean(rf,ep.wf);mb=_weighted_mean(rf,ep.wb)
    vf=_weighted_mean((rf-mf)**2,ep.wf);vb=_weighted_mean((rf-mb)**2,ep.wb)
    eta=(mf-mb)**2/(vf+vb+1e-6);selected=eta<.05;Un=U[:,selected];rank=Un.shape[1]
    if control=='all_register_directions':Un=U;rank=Un.shape[1]
    if control=='append_register_scalars':
        rr,qq,_=_standardize(r@U,q@U,ep.wvalid);z,fit=br_margin(ep,unit(np.c_[r,.5*rr]),unit(np.c_[q,.5*qq]))
        return _return(ep,z,mid,{'control':control,'fit':fit,'eta':eta.tolist(),'shared_observation':receipt})
    if rank==0:
        z,fit=br_margin(ep);return _return(ep,z,mid,{'inactive_reason':'no reference eta below.05','eta':eta.tolist(),'fit':fit,'shared_observation':receipt})
    original_Un=Un.copy();shrink=.5;matched_energy_info={}
    if control in('random_same_rank','random_same_rank_removed_energy'):Un=np.linalg.qr(np.random.default_rng(10).normal(size=(r.shape[1],rank)))[0][:,:rank]
    if control=='random_same_rank_removed_energy':
        source_energy=float(np.sum((r@original_Un)**2*ep.wvalid[:,None])+np.sum((q@original_Un)**2*ep.q_valid[:,None]))
        random_energy=float(np.sum((r@Un)**2*ep.wvalid[:,None])+np.sum((q@Un)**2*ep.q_valid[:,None]))
        shrink=.5*np.sqrt(source_energy/max(random_energy,1e-300))
        matched_energy_info={'removed_vector_energy_main':.25*source_energy,'removed_vector_energy_control':shrink**2*random_energy,'control_shrink_coefficient':shrink,'control_can_invert_random_directions':bool(shrink>1),'energy_definition':'valid-area-weighted squared norm of x-Px, before re-unit'}
    elif control=='PCA_same_rank':
        X=np.r_[r[ep.wvalid>0],q[ep.q_valid>0]];X=X-X.mean(axis=0)
        from scipy.sparse.linalg import svds
        if rank<min(X.shape):_,_,V=svds(X,k=rank,which='LM',tol=1e-6,v0=np.ones(min(X.shape)));Un=V[::-1].T
        else:Un=np.linalg.svd(X,full_matrices=False)[2][:rank].T
    rr=r-shrink*(r@Un)@Un.T;qq=q-shrink*(q@Un)@Un.T
    if control!='no_reunit':rr=unit(rr);qq=unit(qq)
    # This full-rank source-defined metric uses the same exact B_R fit.
    z,fit=br_margin(ep,rr,qq)
    return _return(ep,z,mid,{'register_singular_values':singular.tolist(),'eta':eta.tolist(),'selected_rank':rank,'metric_eigenvalues':[1-shrink,1.],'fit':fit,'shared_observation':receipt,'control':control,'changed_native_margins_vs_BR':'record in scorer/software probe; not a quality assertion',**matched_energy_info})


METHODS['PRO30_M10']=infer_m10
REQUIREMENTS['PRO30_M10']=['actual_final_LN_native_register_r_q_excluding_CLS','native_RGB_and_frozen_cpu_encoder_if_registers_not_cached']
ASSUMPTIONS['PRO30_M10']=['Native final registers use the same actual final LayerNorm as the patch descriptor channel space; CLS is excluded, no invented prefix vector.',
                           'SVD directions below the ordinary shape-scaled floating-point rank tolerance are omitted; no patch/global centering is part of the main metric.']
for name in('random_same_rank','random_same_rank_removed_energy','PCA_same_rank','all_register_directions','no_reunit','append_register_scalars','register_means','swap_RQ_registers'):
    CONTROLS['PRO30_control_M10_'+name]=(lambda ep,name=name:infer_m10(ep,name))
CONTROLS['PRO30_control_M10_B_R']=br_result


def _projection(D,K=32,seed=12):
    from .structures_09_16 import cached
    K=min(K,D)
    if K==D:return np.eye(D)
    return np.linalg.qr(np.random.default_rng(seed).normal(size=(D,K)))[0][:,:K]


def _log_vmf_C(rho,d):
    from scipy.special import ive,gammaln
    rho=np.asarray(rho,float);output=np.empty_like(rho);zero=float(gammaln(d/2)-np.log(2)-d/2*np.log(np.pi))
    small=rho<1e-3;output[small]=zero-rho[small]**2/(2*d)+rho[small]**4/(4*d*d*(d+2))
    value=rho[~small];scaled=ive(d/2-1,value)
    if np.any(scaled<=0)or not np.isfinite(scaled).all():raise ArithmeticError('Stable scaled-Bessel evaluation unavailable for actual vMF parameters')
    output[~small]=(d/2-1)*np.log(value)-d/2*np.log(2*np.pi)-np.log(scaled)-value
    return output


def _vmf_slot_cost(n,S,prior,tau,alpha,N):
    from scipy.special import gammaln
    rho=np.linalg.norm(prior+20*S,axis=1)
    return _log_vmf_C(rho,S.shape[1])-_log_vmf_C(tau,S.shape[1])-(gammaln(n+alpha)-gammaln(alpha))+.5*np.log(N+1)*(n>0)


def _vmf_fit(x,reference,initialization,no_prior=False):
    from scipy.special import gammaln
    from .structures_09_16 import weighted_atoms
    N,d=x.shape;KF=len(reference);K=KF+16;alpha=np.r_[np.full(KF,.5/KF),np.full(16,.5/16)]
    tau=np.r_[np.full(KF,0. if no_prior else 20.),np.zeros(16)];prior=np.zeros((K,d));prior[:KF]=reference*tau[:KF,None]
    bg,_,_=weighted_atoms(x,np.ones(N),min(16,N),5)if N else(np.empty((0,d)),None,None)
    fg=np.argmax(x@reference.T,axis=1)if N else np.empty(0,int);bc=np.argmax(x@bg.T,axis=1)if N else np.empty(0,int)
    if initialization==0:
        shared=(x@reference.T).max(axis=1) >=(x@bg.T).max(axis=1)if N else np.empty(0,bool)
        assignment=np.where(shared,fg,KF+bc)
    elif initialization==1:assignment=fg.copy()
    elif initialization==2:assignment=KF+bc
    else:
        similarity=(x@reference.T).max(axis=1)if N else np.empty(0)
        assignment=np.where(similarity>=np.median(similarity)if N else np.empty(0,bool),fg,KF+bc)
    n=np.bincount(assignment,minlength=K).astype(float);S=np.zeros((K,d))
    if N:np.add.at(S,assignment,x)
    costs=_vmf_slot_cost(n,S,prior,tau,alpha,N);history=[float(gammaln(N+1)+costs.sum())];changes=[]
    for iteration in range(20):
        changed=0
        for i,point in enumerate(x):
            old=int(assignment[i]);removed_n=n[[old]]-1;removed_S=S[[old]]-point
            remove=float(_vmf_slot_cost(removed_n,removed_S,prior[[old]],tau[[old]],alpha[[old]],N)[0]-costs[old])
            prospective=_vmf_slot_cost(n+1,S+point,prior,tau,alpha,N);delta=prospective-costs+remove;delta[old]=0.
            destination=int(np.argmin(delta))
            if delta[destination]<-1e-10:
                n[old]-=1;S[old]-=point;n[destination]+=1;S[destination]+=point;assignment[i]=destination;changed+=1
                at=[old,destination];costs[at]=_vmf_slot_cost(n[at],S[at],prior[at],tau[at],alpha[at],N)
        history.append(float(gammaln(N+1)+costs.sum()));changes.append(changed)
        if history[-1]>history[-2]+1e-8:raise ArithmeticError('Exact vMF coordinate descent increased complete source-defined L')
        if not changed:break
    return {'n':n,'S':S,'alpha':alpha,'prior':prior,'tau':tau,'v':prior+20*S,'N':N,'KF':KF,'L':history[-1],'history':history,'changes':changes,'assignment':assignment,'active_FG':int(np.sum(n[:KF]>0)),'active_BG':int(np.sum(n[KF:]>0))}


def infer_m12(ep,control=None):
    from .structures_09_16 import tree,weighted_atoms,cached,prototype
    mid='PRO30_M12';deg=degenerate_margin(ep)
    if deg is not None:return _return(ep,deg[0],mid,deg[1])
    started=time.perf_counter();P=cached(ep,('orthogonal_projection',12),lambda:_projection(ep.q.shape[1]))
    nodes,leaves,roots=tree(ep);regions=[nodes[k]['support']for k in leaves]
    if len(regions)<2:return _return(ep,prototype(ep),mid,{'inactive_reason':'fewer than2 valid query regions'})
    observed=unit(np.array([nodes[k]['mean']for k in leaves])@P);r=unit(np.asarray(ep.r)@P);q=unit(np.asarray(ep.q)@P)
    reference,masses,groups=weighted_atoms(r,ep.wf,4,5)
    # Entire regions, not own tokens, are excluded by one absolute checker group.
    groupsQ=np.array([((int(np.mean(np.unravel_index(C,ep.q_hw)[0]))//8)%2)*2+(int(np.mean(np.unravel_index(C,ep.q_hw)[1]))//8)%2 for C in regions])
    z=np.full(len(ep.q),-1.);receipts=[]
    for fold in range(4):
        predicted=np.flatnonzero(groupsQ==fold)
        if not len(predicted):continue
        train=np.flatnonzero(groupsQ!=fold)if control!='no_leave_region'else np.arange(len(regions))
        candidates=[_vmf_fit(observed[train],reference,init,control=='no_reference_prior')for init in range(4)]
        best=min(range(4),key=lambda k:(candidates[k]['L'],k));model=candidates[best];rho=np.linalg.norm(model['v'],axis=1);weight=np.log((model['n']+model['alpha'])/(model['N']+1));oldC=_log_vmf_C(rho,q.shape[1]);constant=float(_log_vmf_C(np.array([20.]),q.shape[1])[0])
        ids=np.concatenate([regions[k]for k in predicted]);result=np.empty(len(ids))
        from scipy.special import logsumexp
        for start in range(0,len(ids),256):
            points=q[ids[start:start+256]];updated=np.linalg.norm(model['v'][None]+20*points[:,None],axis=-1)
            logp=constant+oldC[None]-_log_vmf_C(updated,q.shape[1])+weight[None];KF=model['KF']
            result[start:start+256]=logsumexp(logp[:,:KF],axis=1)-logsumexp(logp[:,KF:],axis=1)
        z[ids]=result
        receipts.append({'fold':fold,'train_regions':len(train),'held_regions':len(predicted),'selected_initialization':best,'initial_L':[c['L']for c in candidates],'L_history':model['history'],'assignment_changes':model['changes'],'n':model['n'].tolist(),'S':model['S'].tolist(),'tau':model['tau'].tolist(),'active_FG':model['active_FG'],'active_BG':model['active_BG']})
    z[ep.q_valid<=0]=-1.
    return _return(ep,z,mid,{'orthogonal_dimension':P.shape[1],'regions_as_unit_observations':len(regions),'models':receipts,'cached_postprocess_seconds':time.perf_counter()-started,'control':control})


METHODS['PRO30_M12']=infer_m12
REQUIREMENTS['PRO30_M12']=['native_unit_r_q','complete_reference_area_weights','native_geometry']
ASSUMPTIONS['PRO30_M12']=['The common32-dimensional orthogonal map is fixed Gaussian-QR seed12, identity when original channel count is below32; no variance division.',
                           'Four fixed starts are nearest FG/BG direction, all nearest FG, all private kmeans BG, and source-FG-similarity median split. Empty BG prior slots remain in every predictive density.',
                           'Region-centroid checker groups use8-native-token square blocks modulo2 in each coordinate, with a whole region held out; descent uses only a1e-10 numerical strict-decrease tolerance.']
CONTROLS['PRO30_control_M12_no_reference_prior']=lambda ep:infer_m12(ep,'no_reference_prior')
CONTROLS['PRO30_control_M12_no_leave_region']=lambda ep:infer_m12(ep,'no_leave_region')


def infer_m13(ep,control=None):
    from .structures_09_16 import tree,prototype,host_B,edges4
    mid='PRO30_M13';deg=degenerate_margin(ep)
    if deg is not None:return _return(ep,deg[0],mid,deg[1])
    started=time.perf_counter();h,host=host_B(ep);u=prototype(ep);nodes,leaves,roots=tree(ep);regions=[nodes[k]['support']for k in leaves];i,j=edges4(ep.q_hw,ep.q_valid)
    centers=unit(np.array([np.average(ep.q[C],axis=0,weights=ep.q_valid[C])for C in regions]));mean_h=np.array([np.average(h[C],weights=ep.q_valid[C])for C in regions]);mean_u=np.array([np.average(u[C],weights=ep.q_valid[C])for C in regions]);positive=np.flatnonzero((mean_h>0)&(mean_u>0));negative=np.flatnonzero((mean_h<0)&(mean_u<0))
    # Positive connected pieces are split by the128 regions before selecting donors.
    from scipy.ndimage import label
    cc,count=label((h>0).reshape(ep.q_hw),np.array([[0,1,0],[1,1,1],[0,1,0]]));donors=[]
    for k in range(len(regions)):
        for component in np.unique(cc.ravel()[regions[k]]):
            if component<=0:continue
            C=regions[k][cc.ravel()[regions[k]]==component]
            if len(C)and np.average(h[C],weights=ep.q_valid[C])>0 and np.average(u[C],weights=ep.q_valid[C])>0:
                donors.append((int(k),C,float(np.average(h[C],weights=ep.q_valid[C]))))
    prototypes=[]
    for _,C,score in donors:
        top=C[np.lexsort((C,-h[C]))[:max(1,int(np.ceil(len(C)/4)))]];prototypes.append(unit(np.average(ep.q[top],axis=0,weights=ep.q_valid[top])))
    prototypes=np.asarray(prototypes).reshape(-1,ep.q.shape[1]);background=unit(np.array([np.average(ep.q[C],axis=0,weights=ep.q_valid[C])for C in regions]))
    g=np.full(len(regions),np.nan);audit=[]
    for k,C in enumerate(regions):
        forbidden=np.zeros(len(ep.q),bool);forbidden[C]=True;forbidden[j[np.isin(i,C)]]=True;forbidden[i[np.isin(j,C)]]=True
        eligible=[]
        for d,(region,points,score)in enumerate(donors):
            if control=='allow_self' or not np.any(forbidden[points]):eligible.append(d)
        eligible=sorted(eligible,key=lambda d:(-donors[d][2],donors[d][0],int(donors[d][1].min())))[:8]
        if len(eligible)<2:audit.append({'recipient':k,'positive_donors':eligible,'inactive':'fewer than2 disjoint donors'});continue
        similarity=prototypes[eligible]@centers[k]
        a=float(np.sort(similarity)[-2])if control not in('maximum','mean')else float(similarity.max()if control=='maximum'else similarity.mean())
        bg=[d for d in negative if control=='allow_self' or not np.any(forbidden[regions[d]])]
        b=float(np.max(background[bg]@centers[k]))if bg else float(unit(np.sum(ep.r*ep.wb[:,None],axis=0))@centers[k]);g[k]=a-b
        audit.append({'recipient':k,'positive_donor_piece_indices':eligible,'positive_donor_region_ids':[donors[d][0]for d in eligible],'BG_donor_regions':list(map(int,bg)),'a':a,'b':b,'g':g[k]})
    # Use eligible donor recipients' g values only for the common MAD scale.
    donor_g=[]
    for donor_index,(region,C,score)in enumerate(donors):
        forbidden=np.zeros(len(ep.q),bool);forbidden[C]=True;forbidden[j[np.isin(i,C)]]=True;forbidden[i[np.isin(j,C)]]=True
        eligible=[d for d,(_,D,scoreD)in enumerate(donors)if d!=donor_index and not np.any(forbidden[D])]
        eligible=sorted(eligible,key=lambda d:(-donors[d][2],donors[d][0],int(donors[d][1].min())))[:8]
        if len(eligible)<2:continue
        center=unit(np.average(ep.q[C],axis=0,weights=ep.q_valid[C]));similarity=prototypes[eligible]@center
        a=float(np.sort(similarity)[-2]);bg=[d for d in negative if not np.any(forbidden[regions[d]])]
        b=float(np.max(background[bg]@center))if bg else float(unit(np.sum(ep.r*ep.wb[:,None],axis=0))@center);donor_g.append(a-b)
    observed=np.asarray(donor_g)
    scale=max(float(np.median(np.abs(observed-np.median(observed))))if len(observed)else 0.,.05)
    z=h.copy();active=0
    for k,C in enumerate(regions):
        if not np.isfinite(g[k]):continue
        editable=C[np.abs(h[C])<=.25];z[editable]+=.25*np.clip(g[k]/scale,-1,1);active+=len(editable)
    z[ep.q_valid<=0]=-1.
    return _return(ep,z,mid,{'host':host,'donor_pieces':len(donors),'MAD_floor':scale,'donor_scale_gap_samples':observed.tolist(),'donor_recipient_records':audit,'eligible_modified_tokens':active,'cached_postprocess_seconds_including_first_host':time.perf_counter()-started,'control':control})


METHODS['PRO30_M13']=infer_m13
REQUIREMENTS['PRO30_M13']=['complete_original_FoRIS_plus_MEAN16_host_B','native_unit_r_q','complete_R_mask_area_weights']
ASSUMPTIONS['PRO30_M13']=['Positive H connected pieces are intersected with Ward128 regions; donor prototypes use each piece highest ceil(n/4) tokens, and recipients exclude their entire own support plus one physical neighbor step.',
                           'The uniform scale is MAD of eligible donor-region g values with floor.05; an unavailable donor recipient contributes no scale sample. No independent-votes probability is claimed.']
for name in('allow_self','maximum','mean'):CONTROLS['PRO30_control_M13_'+name]=(lambda ep,name=name:infer_m13(ep,name))


def infer_m15(ep,control=None):
    from .flow_09_16 import optimize
    mid='PRO30_M15';deg=degenerate_margin(ep)
    if deg is not None:return _return(ep,deg[0],mid,deg[1])
    start=time.perf_counter();z,info=optimize(ep,control);info['cached_postprocess_seconds']=time.perf_counter()-start
    return _return(ep,z,mid,info)


METHODS['PRO30_M15']=infer_m15
REQUIREMENTS['PRO30_M15']=['native_unit_r_q','complete_reference_area_weights','exact_native_reference_query_geometry','complete_R_original_mask_for_exact_target_bbox']
ASSUMPTIONS['PRO30_M15']=['Top-four role banks use positive legal foreground/background area weights; mixed reference positions can occur in both lists, but exact duplicates are removed before the reject state.',
                           'Coordinate energy averages both directed local-J constraints per undirected physical edge; every edge touching reject pays .05, including two reject endpoints. Jacobian smoothness counts each physical block-boundary edge.',
                           'The ridge Jacobian surrogate uses all nonreject pairs,1e-6 numerical ridge around identity, Gauss-Seidel block updates and singular clipping; it may increase the original truncated energy, which is recorded. Five prescribed rounds are retained, no invented convergence guarantee.']
for name in('nearest','label_potts','shuffle_coords','same_role'):CONTROLS['PRO30_control_M15_'+name]=(lambda ep,name=name:infer_m15(ep,name))


def infer_m11(ep,control=None):
    from .transport_09_16 import execute
    mid='PRO30_M11';deg=degenerate_margin(ep)
    if deg is not None:return _return(ep,deg[0],mid,deg[1])
    start=time.perf_counter();z,info=execute(ep,control);info['cached_postprocess_seconds']=time.perf_counter()-start
    return _return(ep,z,mid,info)


METHODS['PRO30_M11']=infer_m11
REQUIREMENTS['PRO30_M11']=['native_unit_r_q','complete_reference_area_weights','native_geometry']
ASSUMPTIONS['PRO30_M11']=['The otherwise underspecified three scale medians are role-internal off-diagonal1-cos, its square for relation costs, and the unit null-option indicator cost1; each uses the required1e-6 floor. This interpretation must be reviewed before natural validation.',
                           'Thirty log-domain KL-Dykstra row/column/retained-mass updates are followed only by explicit feasible capping and a convex mixture with the feasible independent b*a baseline if mass fell below.5. Repair magnitude and raw residual are recorded; no solver optimum claim.',
                           'Reference/query atoms use highest-weight initial point then area-weighted farthest selection and five weighted Lloyd rounds. DP reads all actual Ward128-leaf tree nodes, and its final selected labels are emitted as exactly+1/-1.']
CONTROLS['PRO30_control_M11_appearance_OT']=lambda ep:infer_m11(ep,'appearance_only')
CONTROLS['PRO30_control_M11_no_column_capacity']=lambda ep:infer_m11(ep,'no_column_capacity')


def infer_m14(ep,control=None):
    from .partition_09_16 import execute
    mid='PRO30_M14';deg=degenerate_margin(ep)
    if deg is not None:return _return(ep,deg[0],mid,deg[1])
    start=time.perf_counter();z,info=execute(ep,control=='binary_no_partition',control=='unlabeled_partition');info['cached_postprocess_seconds']=time.perf_counter()-start
    return _return(ep,z,mid,info)


METHODS['PRO30_M14']=infer_m14
REQUIREMENTS['PRO30_M14']=['native_unit_r_q','complete_reference_area_weights','native_geometry']
ASSUMPTIONS['PRO30_M14']=['The64-channel orthogonal edge projection uses fixed Gaussian-QR seed14; cosine is a65th observed channel, with an unpenalized intercept in the ridge fit.',
                           'Three role-balanced soft training masses are divided by their actual source class mass; a missing source edge role contributes zero weighted target, no invented BG-instance supervision.',
                           'One greatest-decrease adjacent-group merge is attempted per prescribed round, followed by every active group actual-history split. All comparisons use the exact complete objective; ties retain state.']
CONTROLS['PRO30_control_M14_binary_no_partition']=lambda ep:infer_m14(ep,'binary_no_partition')
CONTROLS['PRO30_control_M14_unlabeled_partition']=lambda ep:infer_m14(ep,'unlabeled_partition')


def infer_m16(ep,control=None):
    from .topology_09_16 import execute
    mid='PRO30_M16';deg=degenerate_margin(ep)
    if deg is not None:return _return(ep,deg[0],mid,deg[1],True)
    start=time.perf_counter();field,info=execute(ep,control);info['postprocess_seconds_including_first_host']=time.perf_counter()-start;info.update(source_contract(16),implementation_assumptions=ASSUMPTIONS[mid])
    return finish_highres(ep,field,mid,info,field_space='original')


METHODS['PRO30_M16']=infer_m16
REQUIREMENTS['PRO30_M16']=['complete_original_FoRIS_plus_MEAN16_host_B','complete_R_original_mask','original_Q_RGB_and_exact_geometry']
ASSUMPTIONS['PRO30_M16']=['Reference/query foreground components are8-connected with4-connected background holes; skeleton is exact deterministic Zhang-Suen thinning, endpoint count is degree1 and coarse branches are connected degree>=3 sets.',
                           'The underspecified longest skeleton path is maximum finite geodesic shortest-path distance. Trees use exact two-sweep diameter; cyclic skeletons stream every vertex shortest path, which may exceed the2-second CPU budget and is never silently approximated.',
                           'Reference hole/endpoint/branch dimensions are portable only if their coarse count agrees at all four erosion scales; the four normalized geodesic-length ratios remain the explicitly supplied deformation set.',
                           'Radius1-token morphology is measured through original-image physical patch spacing via Euclidean EDT; candidate windows contain a whole H-positive component plus its one-token uncertain band, weak components are added only if they contain no H-positive point.']
for name in('unary_only','RGB_only','fixed_no_holes'):CONTROLS['PRO30_control_M16_'+name]=(lambda ep,name=name:infer_m16(ep,name))
