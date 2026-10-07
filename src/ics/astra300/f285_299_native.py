"""Supplied native-feature F286/287/289--293/295/298/299 kernels.

No query label is read. Finite dictionaries, alternatives and search budgets
are the card's explicit approximations, not new numbered methods.
"""
from __future__ import annotations
from itertools import combinations,product
import heapq
import numpy as np
from scipy import ndimage
from scipy.special import expit,logsumexp
from scipy.spatial.distance import cdist
from . import f276_300 as K
from . import f_protocol as P
from . import common

EPS=1e-6


def _scale(c,x):
    return max(float(np.median(np.abs(np.asarray(x)-np.median(x)))),EPS)


def _balanced_ridge(x,target,rank=16):
    x=np.asarray(x,float);target=np.asarray(target,float)
    classes=target>.5
    if not classes.any() or classes.all():return None
    mu=x.mean(0);z=x-mu
    _,singular,basis=np.linalg.svd(z,full_matrices=False)
    basis=basis[singular>EPS][:rank];z=z@basis.T
    weights=np.where(classes,.5/classes.sum(),.5/(~classes).sum())
    A=np.c_[z,np.ones(len(z))]
    beta=np.linalg.solve(A.T@(weights[:,None]*A)+np.diag(np.r_[np.ones(len(basis)),EPS]),A.T@(weights*(2*target-1)))
    return mu,basis,beta


def _predict_ridge(model,x):
    if model is None:return np.zeros(len(x))
    mu,basis,beta=model
    return np.c_[(np.asarray(x)-mu)@basis.T,np.ones(len(x))]@beta


def _experts(r,cov,known,q):
    f=known&(cov>=.9);b=known&(cov<=.1)
    if not f.any() or not b.any():return None
    mean=(q@(K.unit(r[f].mean(0))-K.unit(r[b].mean(0))))/.1
    ridge=_balanced_ridge(r[f|b],cov[f|b],16);rs=_predict_ridge(ridge,q)
    spaces=[]
    for bank in (r[f],r[b]):
        mu,basis=K.pca(bank,16);res=q-mu;spaces.append(np.maximum(np.sum(res*res,1)-np.sum((res@basis.T)**2,1),0))
    return np.c_[mean,rs,spaces[1]-spaces[0]]


def expert_observations(c):
    q=_experts(c.r,c.c,c.valid,c.q)
    source=np.full((len(c.r),3),np.nan);count=0
    for train,held in P.buffered_folds(c.rhw,c.valid):
        values=_experts(c.r,c.c,train,c.r[held])
        if values is not None:source[held]=values;count+=1
    known=np.isfinite(source).all(1)&c.valid
    if q is None or count<2 or not known.any():return None
    scale=np.array([_scale(c,source[known,j]) for j in range(3)])
    return q/scale,source/scale,known,scale


def f286(c,weight=1.,control=False):
    observed=expert_observations(c)
    if observed is None:return K.result(c,score=c.s,fallback='insufficient_source_expert_tasks')
    h,rh,known,scales=observed;target=c.c[known]
    error=np.abs(expit(rh[known])-target[:,None]);best=int(np.argmin(error.mean(0)))
    model=K.ridge(rh[known],error,1.);proxy=np.clip(K.predict(model,h),0,1)
    center=rh[known].mean(0);spread=np.maximum(rh[known].std(0),EPS)
    bank=(rh[known]-center)/spread;nearest=cdist(bank,bank,'sqeuclidean');np.fill_diagonal(nearest,np.inf)
    radius=float(np.quantile(nearest.min(1),.95)) if len(bank)>1 else 0.
    outside=cdist((h-center)/spread,bank,'sqeuclidean').min(1)>radius
    e=np.argmin(proxy,1);e[outside]=best;initial=e.copy();y=c.m0.copy();cap=c.edge_weight
    def objective(labels,expert):
        s=h[np.arange(c.n),expert]
        return c.energy(labels,s=s)+weight*(proxy[np.arange(c.n),expert].sum()+cap@(expert[c.edges[:,0]]!=expert[c.edges[:,1]]))
    if control=='average':score=h.mean(1);return K.result(c,labels=K.cut(c,score),score=score,control='same_three_experts_average')
    if control=='source_best':score=h[:,best];return K.result(c,labels=K.cut(c,score),score=score,control='source_best_expert')
    if control in ('independent','once'):
        score=h[np.arange(c.n),e];return K.result(c,labels=K.cut(c,score),score=score,control='independent_gate_then_one_cut')
    adjacent=[[] for _ in range(c.n)]
    for (a,b),fee in zip(c.edges,cap):adjacent[a].append((int(b),float(fee)));adjacent[b].append((int(a),float(fee)))
    best_y=y.copy();best_e=e.copy();best_cost=objective(y,e);costs=[]
    for iteration in range(10):
        for i in range(c.n):
            if outside[i]:continue
            costs_i=np.logaddexp(0,h[i])-y[i]*h[i]+weight*proxy[i]
            for j,fee in adjacent[i]:costs_i+=weight*fee*(np.arange(3)!=e[j])
            e[i]=int(np.argmin(costs_i))
        score=h[np.arange(c.n),e];y=K.cut(c,score);value=objective(y,e);costs.append(float(value))
        if value<best_cost:best_y,best_e,best_cost=y.copy(),e.copy(),value
        if len(costs)>1 and abs(costs[-1]-costs[-2])<EPS:break
    score=h[np.arange(c.n),best_e]
    return K.result(c,labels=best_y,score=score,mechanism='F286_joint_spatial_expert_qualification',rounds=iteration+1,
            objective=float(best_cost),objective_history=costs,expert_counts=np.bincount(best_e,minlength=3).tolist(),
            source_best_expert=best,source_expert_scales=scales.tolist(),outside_source_radius=int(outside.sum()),
            source_radius_squared=radius,initial_route=initial.tolist(),global_optimum=False)


def _pca_views(c):
    mu,basis=K.pca(c.r[c.valid],16)
    if len(basis)<2:return None
    z=(c.r-mu)@basis.T;q=(c.q-mu)@basis.T
    values=[]
    for offset in (0,1):values.append(K.kernel_field(K.unit(q[:,offset::2]),K.unit(z[c.F,offset::2]),K.unit(z[c.B,offset::2])))
    return np.stack(values,1)


def f287(c,weight=1.,control=False):
    heads=_pca_views(c)
    if heads is None:return K.result(c,score=c.s,fallback='two_PCA_views_unidentifiable')
    if control in ('AND','OR','average'):
        score=heads.mean(1);labels=(heads>0).all(1) if control=='AND' else (heads>0).any(1) if control=='OR' else K.cut(c,score)
        return K.result(c,labels=labels,score=score,control=control)
    edits=[];reject=[]
    for head in (0,1):
        proposal=K.cut(c,heads[:,head]);opponent=heads[:,1-head]
        for p in c.P:
            for direction in (True,False):
                changed=p&(proposal!=c.m0)&(proposal==direction)
                if not changed.any():continue
                border=ndimage.binary_dilation(changed.reshape(c.hw)).ravel()&~changed
                interior=float(np.mean((2*direction-1)*opponent[changed]));outside=float(np.mean((1-2*direction)*opponent[border])) if border.any() else 0.
                edits.append((changed,direction));reject.append(max(0.,-(interior+outside)))
    edits=edits[:128];reject=np.asarray(reject[:128]);n=len(edits)
    if not n:return K.result(c,score=c.s,fallback='no_whole_edit_proposal')
    def compose(active):
        y=c.m0.copy();votes=np.zeros(c.n);mass=np.zeros(c.n)
        for j in np.flatnonzero(active):
            mask,direction=edits[j];votes[mask]+=2*direction-1;mass[mask]+=1
        changed=mass>0;y[changed]=votes[changed]>0
        return y
    score=heads.mean(1)
    def energy(active):
        y=compose(active)
        penalty=0.
        for j in np.flatnonzero(active):
            mask,direction=edits[j]
            penalty+=reject[j]*float(np.mean(y[mask]==direction))
        return c.energy(y,s=score)+weight*penalty
    if control=='independent':state=reject==0;info=dict(solver='whole_edits_independently_authorized')
    else:
        factors=[]
        for a,b in combinations(range(n),2):
            if np.any(edits[a][0]&edits[b][0]):factors.append([a,b])
        state,info=K.finite_search(energy,[np.zeros(n,bool),np.ones(n,bool),reject==0],factors)
    y=compose(state)
    return K.result(c,labels=y,score=score,mechanism='F287_cross_authorized_whole_edit_sets',
                    edits=n,chosen_edits=int(state.sum()),rejected_whole_edit_fees=reject.tolist(),**info)


def mask_statistics(c,y,profile=None):
    profile=c.heads[:,2] if profile is None else np.asarray(profile)
    values=[]
    for role in (y,~y):
        values.extend(np.quantile(profile[role],[.1,.25,.5,.75,.9]).tolist() if role.any() else [0.]*5)
    weakest=[];components,count=ndimage.label(np.asarray(y).reshape(c.hw));blocks=P.quarters(c.hw)
    for j in range(1,count+1):
        mask=components.ravel()==j;parts=[profile[mask&(blocks==k)].mean() for k in range(4) if np.any(mask&(blocks==k))]
        if parts:weakest.append(min(parts))
    values.extend([min(weakest) if weakest else 0.,np.mean(weakest) if weakest else 0.,
                   float(c.edge_weight@(y[c.edges[:,0]]!=y[c.edges[:,1]])),float(y.mean()),float(count),
                   float(np.mean(profile[y])) if y.any() else 0.])
    return np.asarray(values,float)


def quadratic(x):
    x=np.asarray(x,float);parts=[x]
    parts.extend([x[...,i:i+1]*x[...,j:j+1] for i in range(x.shape[-1]) for j in range(i,x.shape[-1])])
    return np.concatenate(parts,axis=-1)


def _known_original_target(c):
    # A source fold cannot recover labels outside its current known domain.
    # Actual original R labels are legal only where c.valid retains the token.
    mask=None
    if c.resources and c.resources.get('reference_training_pixels') is not None:mask=c.resources['reference_training_pixels']
    elif c.ep.reference_mask is not None:mask=c.ep.reference_mask
    if mask is None:return None
    from .e_helpers_226_250 import pixel_tokens
    token=pixel_tokens(np.shape(mask),c.rhw,c.ep.reference_geometry)
    return np.asarray(mask,bool),c.valid[token].reshape(np.shape(mask))


def source_rank_tasks(c):
    from dataclasses import replace
    target_and_valid=_known_original_target(c)
    if target_and_valid is None:return []
    target,known_original=target_and_valid;tasks=[]
    for train,held in P.buffered_folds(c.rhw,c.valid):
        if not train.any() or not held.any() or not np.any(c.F&train) or not np.any(c.B&train):continue
        episode=P._reference_query(c.ep,train);source=P.build_context(episode,resources=c.resources)
        domain=common.U(episode,held.reshape(c.rhw).astype(float),.5)&known_original
        if not domain.any():continue
        stats=[];scores=[]
        for y in source.Q:
            stats.append(mask_statistics(source,y));pred=common.U(episode,y.reshape(c.rhw).astype(float),.5)
            I=np.count_nonzero(pred&target&domain);U=np.count_nonzero((pred|target)&domain);scores.append(I/U if U else 1.)
        tasks.append((np.stack(stats),np.array(scores),source))
    return tasks


def f289(c,weight=1.,control=False):
    tasks=source_rank_tasks(c)
    if len(tasks)<2:return K.result(c,score=c.s,fallback='no_two_original_source_rank_tasks')
    mu=np.concatenate([x for x,_,_ in tasks]).mean(0);std=np.maximum(np.concatenate([x for x,_,_ in tasks]).std(0),EPS)
    def features(x):
        z=(x-mu)/std
        return z if control=='linear' else quadratic(z)
    X=[];T=[];groups=[]
    for group,(x,t,_) in enumerate(tasks):
        z=features(x)
        for a,b in combinations(range(len(t)),2):X.append(z[a]-z[b]);T.append(t[a]-t[b]);groups.append(group)
    if not X:return K.result(c,score=c.s,fallback='no_source_candidate_pair')
    X=np.asarray(X);T=np.asarray(T);groups=np.array(groups)
    losses=[]
    for regularizer in (.25,1.,4.):
        loss=[]
        for group in np.unique(groups):
            train=groups!=group;test=groups==group
            beta=np.linalg.solve(X[train].T@X[train]+regularizer*np.eye(X.shape[1]),X[train].T@T[train])
            loss.append(float(np.mean((X[test]@beta-T[test])**2)))
        losses.append(np.mean(loss))
    reg=(.25,1.,4.)[int(np.argmin(losses))]
    beta=np.linalg.solve(X.T@X+reg*np.eye(X.shape[1]),X.T@T)
    statistics=np.stack([mask_statistics(c,y) for y in c.Q]);ranks=features(statistics)@beta
    if control=='profile':ranks=np.array([-c.energy(y,s=c.heads[:,2]) for y in c.Q])
    if control=='average':ranks=np.array([np.mean(c.heads[y,2])-np.mean(c.heads[~y,2]) if y.any() and (~y).any() else -c.energy(y,s=c.heads[:,2]) for y in c.Q])
    # Lambda0 is handled before this kernel. Nonzero lambda ranks complete
    # candidates with the original E0 plus learned relative source-IoU loss.
    costs=np.array([c.energy(y) for y in c.Q])-weight*ranks/_scale(c,ranks)
    j=min(range(len(costs)),key=lambda i:(costs[i],int(c.Q[i].sum()),i))
    return K.result(c,labels=c.Q[j],mechanism='F289_quadratic_pairwise_whole_mask_rank',selected_candidate=j,
             candidate_costs=costs.tolist(),candidate_ranks=ranks.tolist(),source_tasks=len(tasks),regularizer=reg,
             source_regularizer_losses=list(map(float,losses)),statistic_count=16,expanded_dimensions=len(beta),control=control)


def _local_explanations(c,mask):
    """Re-fit both same-capacity local role mixtures for this current range."""
    halo=ndimage.binary_dilation(mask.reshape(c.hw)).ravel()&~mask
    fids=mask&(c.fsim.max(1)>c.bsim.max(1));bids=halo&(c.bsim.max(1)>c.fsim.max(1))
    if not fids.any() or not bids.any():return None
    f=K.modes(c.q[fids],8)[0];b=K.modes(c.q[bids],8)[0]
    # Known reference modes remain available; local atoms are explanations,
    # never new true-label training data or new classes.
    fs=(c.q@np.r_[c.fm,f].T).max(1);bs=(c.q@np.r_[c.bm,b].T).max(1)
    return (fs-bs)/.1


def f290(c,weight=1.,control=False):
    originals=[p.copy() for p in c.P];masks=[p&c.m0 for p in originals];history=[];actions=0;cache={}
    eligible=np.logical_or.reduce(originals) if originals else np.zeros(c.n,bool)
    def compose(states):
        labels=c.m0.copy();labels[eligible]=False
        for mask in states:labels|=mask
        return labels
    def proposal_fee(j,new):
        nonlocal actions
        key=(j,np.packbits(new).tobytes())
        if key in cache:return cache[key]
        actions+=1
        if not new.any():return 0.
        score=_local_explanations(c,new)
        if score is None:return np.inf
        shell=new&~originals[j];whole=float(np.mean(score[new]))
        shell_adv=float(np.mean(score[shell])) if shell.any() else whole
        advantage=whole if control=='shell_mean' else min(whole,shell_adv)
        boundary=float(c.edge_weight@(new[c.edges[:,0]]!=new[c.edges[:,1]]))
        cache[key]=boundary-advantage
        return cache[key]
    def objective(states):
        labels=compose(states);covered=np.logical_or.reduce(states) if states else np.zeros(c.n,bool)
        # Covered ranges pay their two-sided identity fee; uncovered pixels
        # pay the unchanged complete E0 unary. Every physical label boundary
        # is charged once regardless of overlapping proposal explanations.
        unary=np.logaddexp(0,c.s)-c.s*labels
        energy=float(unary[~covered].sum()+c.edge_weight@(labels[c.edges[:,0]]!=labels[c.edges[:,1]]))
        return energy+weight*sum(proposal_fee(j,p) for j,p in enumerate(states))
    current_cost=objective(masks);initial_cost=current_cost
    for iteration in range(8):
        changed=False
        for j,old in enumerate(masks):
            current=old
            dil=ndimage.binary_dilation(current.reshape(c.hw)).ravel();erode=ndimage.binary_erosion(current.reshape(c.hw)).ravel()
            channel=int(np.argmax(c.bsim[current].mean(0))) if current.any() else 0
            bg=c.bsim[:,channel]>c.fsim.max(1);split=current&~bg
            candidates=[current,np.zeros(c.n,bool),dil,erode,split]
            if control=='static':candidates=[current,np.zeros(c.n,bool)]
            costs=[];states=[]
            for proposed in candidates:
                candidate=list(masks);candidate[j]=proposed
                costs.append(objective(candidate));states.append(candidate)
            winner=min(range(len(costs)),key=lambda k:(costs[k],int(compose(states[k]).sum()),k))
            if winner and costs[winner]<costs[0]-EPS:
                masks=states[winner];current_cost=costs[winner];changed=True
        history.append(float(objective(masks)))
        if not changed:break
    y=compose(masks)
    return K.result(c,labels=y,mechanism='F290_extent_changes_recertify_two_sides',rounds=iteration+1,
             explanation_refits=actions,complete_objective_history=history,initial_complete_objective=float(initial_cost),
             overlap_labels_charged_once=True,control=control,global_optimum=False)


def _distinguishing_vectors(c,candidates):
    pairs=[]
    for a,b in combinations(candidates,2):
        difference=a.astype(float)-b.astype(float);mass=np.count_nonzero(difference)
        if mass:pairs.append(difference@(c.q@c.r.T)/mass)
    return np.stack(pairs,1) if pairs else np.zeros((len(c.r),0))


def witness_select(values,ids,count=8,control=False):
    ids=np.asarray(ids,int)
    if control=='random':return ids[np.argsort((ids*2654435761)%4294967296,kind='stable')[:count]]
    if control=='core':return ids[np.argsort(-np.linalg.norm(values[ids],axis=1),kind='stable')[:count]]
    chosen=[];covered=np.zeros(values.shape[1])
    for _ in range(min(count,len(ids))):
        remaining=ids[~np.isin(ids,chosen)]
        if not len(remaining):break
        if values.shape[1]:
            coverage=np.maximum(covered[None],np.abs(values[remaining]))
            gains=coverage.min(1)
        else:gains=np.zeros(len(remaining))
        j=int(remaining[np.argmax(gains)]);chosen.append(j)
        covered=np.maximum(covered,np.abs(values[j]))
    return np.asarray(chosen,int)


def _subset_scores(c,candidates,control=False):
    values=_distinguishing_vectors(c,candidates)
    selected=np.r_[witness_select(values,np.flatnonzero(c.F),8,control),witness_select(values,np.flatnonzero(c.B),8,control)]
    model=_balanced_ridge(c.r[selected],c.c[selected],16)
    return _predict_ridge(model,c.q),selected


def _round_candidates(c,score):
    masks=[]
    for old in c.Q+[score>0,K.cut(c,score)]:
        if not any(np.array_equal(old,y) for y in masks):masks.append(old.copy())
    return masks[:16]


def f291(c,weight=1.,control=False):
    # The source chooses a round using held original masks, then that fixed
    # round is reproduced on Q. Query confidence never chooses the round.
    target_and_valid=_known_original_target(c);losses=np.zeros(3);folds=0
    if target_and_valid is None:return K.result(c,score=c.s,fallback='source_original_pixels_unavailable')
    target,known=target_and_valid
    for train,held in P.buffered_folds(c.rhw,c.valid):
        if not np.any(c.F&train) or not np.any(c.B&train) or not held.any():continue
        episode=P._reference_query(c.ep,train);source=P.build_context(episode,resources=c.resources);candidates=source.Q
        domain=common.U(episode,held.reshape(c.rhw).astype(float),.5)&known
        if not domain.any():continue
        for iteration in range(3):
            score,selected=_subset_scores(source,candidates,control);mask=common.U(episode,expit(score).reshape(c.rhw),.5)
            I=np.count_nonzero(mask&target&domain);U=np.count_nonzero((mask|target)&domain);losses[iteration]+=1-(I/U if U else 1.)
            candidates=_round_candidates(source,score)
        folds+=1
    if folds<2:return K.result(c,score=c.s,fallback='fewer_than_two_source_round_tasks')
    round_index=0 if control=='once' else int(np.argmin(losses));candidates=c.Q;selected_sets=[]
    for _ in range(round_index+1):score,selected=_subset_scores(c,candidates,control);selected_sets.append(selected.tolist());candidates=_round_candidates(c,score)
    scale=_scale(c,c.source.heads[c.source.oof_valid,0]);score=c.s+weight*score/scale
    return K.result(c,score=score,mechanism='F291_candidate_discriminating_reference_witnesses',
             selected_round=round_index+1,source_round_losses=(losses/folds).tolist(),source_round_folds=folds,
             selected_reference_ids=selected_sets,control=control)


def _validated_modes(c,b):
    candidates=[];blocks=P.quarters(c.rhw);f=c.source.fsim;bscore=c.source.bsim
    bindex=int(np.argmax(c.bm@b));known=c.source.oof_valid&c.valid
    if bindex>=bscore.shape[1]:return None
    assignment=f.argmax(1)
    for subset in [(j,) for j in range(len(c.fm))]+list(combinations(range(len(c.fm)),2)):
        if max(subset)>=f.shape[1]:continue
        fscore=f[:,list(subset)].max(1);negative=bscore[:,bindex];good=[];recall=[]
        for block in range(4):
            truef=c.F&known&(blocks==block);trueb=c.B&known&(blocks==block)
            if not truef.any() or not trueb.any():continue
            if np.any(negative[truef|trueb]<-1.5):continue
            # Rejection must retain every currently validated foreground mode
            # in that source block, rather than sacrifice a known minority.
            valid=True
            for mode in np.unique(assignment[truef]):
                ids=truef&(assignment==mode)
                if np.mean(fscore[ids]>negative[ids])<np.mean(f[ids].max(1)>bscore[ids].max(1))-EPS:valid=False
            reject=float(np.mean(negative[trueb]>fscore[trueb]));good.append(reject);recall.append(valid)
        if len(good)>=2 and all(recall) and min(good)>0:candidates.append((min(good),subset))
    return max(candidates,key=lambda x:(x[0],-len(x[1]),tuple(-j for j in x[1])))[1] if candidates else None


def f292(c,weight=1.,control=False):
    # At most32 actual disputed pieces. Range/negative/model change together.
    pieces=c.P[:32];score=c.s.copy();ownership=K._piece_ownership(pieces,c.hw);rounds=[];bindex={}
    for iteration in range(5):
        current=score>0;updated=score.copy();changed=False;models=0
        for j,p in enumerate(pieces):
            proposed=p|ndimage.binary_dilation((p&current).reshape(c.hw)).ravel()
            ids=proposed&(ownership==j)
            if not ids.any():continue
            b=int(np.argmax(c.bsim[ids].mean(0)))
            if control=='fixed' and j in bindex:b=bindex[j]
            bindex[j]=b;subset=_validated_modes(c,c.bm[b])
            if subset is None:continue
            frows=c.F&np.isin((c.r@c.fm.T).argmax(1),subset);brows=c.B&((c.r@c.bm.T).argmax(1)==b)
            model=_balanced_ridge(c.r[frows|brows],c.c[frows|brows],16)
            if model is None:continue
            new=_predict_ridge(model,c.q);updated[ids]=c.s[ids]+weight*new[ids]/_scale(c,c.source.heads[c.source.oof_valid,0]);models+=1
        labels=K.cut(c,updated);updated[np.logical_or.reduce(pieces) if pieces else np.zeros(c.n,bool)]=np.where(labels,abs(updated),-abs(updated))[np.logical_or.reduce(pieces) if pieces else np.zeros(c.n,bool)]
        changed=not np.array_equal(labels,current);score=updated;rounds.append(models)
        if not changed:break
    union=np.logical_or.reduce(pieces) if pieces else np.zeros(c.n,bool);probability=expit(score);probability[~union]=c.p0[~union]
    return K.result(c,score=np.log(np.clip(probability,EPS,1-EPS)/np.clip(1-probability,EPS,1-EPS)),
             mechanism='F292_counterexample_requires_validated_positive_modes',rounds=len(rounds),heads_per_round=rounds,
             strongest_background_mode_by_piece=bindex,control=control,not_label_probability=True)


def _split_validation(c,mode_ids,v):
    blocks=P.quarters(c.rhw);gains=[]
    for block in range(4):
        held=c.valid&(blocks==block);train=c.valid&~ndimage.binary_dilation(held.reshape(c.rhw),structure=np.ones((3,3))).ravel()
        train_ids=mode_ids&train
        if train_ids.sum()<2 or not np.any(c.F&held) or not np.any(c.B&held):continue
        fm=K.modes(c.r[c.F&train],8)[0];bm=K.modes(c.r[c.B&train],8)[0]
        if not len(fm) or not len(bm):continue
        parent=int(np.argmax(fm@K.unit(c.r[train_ids].mean(0))))
        train_ids=c.F&train&((c.r@fm.T).argmax(1)==parent)
        if train_ids.sum()<2:continue
        threshold=np.median(c.r[train_ids]@v);a=train_ids&(c.r@v<=threshold);b=train_ids&~a
        if not a.any() or not b.any():continue
        new=np.r_[fm[:parent],K.unit(c.r[a].mean(0))[None],K.unit(c.r[b].mean(0))[None],fm[parent+1:]]
        old=(c.r@fm.T).max(1)-(c.r@bm.T).max(1);score=(c.r@new.T).max(1)-(c.r@bm.T).max(1)
        assignment=(c.r@fm.T).argmax(1)
        if any(np.mean(score[c.F&held&(assignment==k)]>0)<np.mean(old[c.F&held&(assignment==k)]>0)-EPS for k in np.unique(assignment[c.F&held])):return False
        gains.append(np.mean(old[c.B&held]>0)-np.mean(score[c.B&held]>0))
    return len(gains)>=2 and np.mean(gains)>EPS


def f293(c,weight=1.,control=False):
    fm=c.fm.copy();accepted=[]
    for iteration in range(4):
        fscore=c.q@fm.T;mode=fscore.argmax(1);positive=c.s>0;bpositive=c.bsim.max(1)>fscore.max(1)
        choices=[]
        for j in range(len(fm)):
            a=positive&(mode==j);b=bpositive&(mode==j)
            if not a.any() or not b.any():continue
            v=K.unit(c.q[a].mean(0)-c.q[b].mean(0));rids=c.F&((c.r@fm.T).argmax(1)==j)
            if rids.sum()<2:continue
            if control=='random':v=K.unit(np.sin(np.arange(c.q.shape[1])+j+1))
            if control not in ('random','unvalidated') and not _split_validation(c,rids,v):continue
            threshold=float(np.median(c.r[rids]@v));left=rids&(c.r@v<=threshold);right=rids&~left
            if left.any() and right.any():choices.append((j,v,left,right))
        if not choices:break
        j,v,left,right=choices[0];fm=np.r_[fm[:j],K.unit(c.r[left].mean(0))[None],K.unit(c.r[right].mean(0))[None],fm[j+1:]];accepted.append(dict(parent=j,direction=v.tolist(),source_sizes=[int(left.sum()),int(right.sum())]))
    if not accepted:return K.result(c,score=c.s,fallback='no_source_validated_collision_split',control=control)
    score=((c.q@fm.T).max(1)-c.bsim.max(1))/.1;y=K.cut(c,c.s+weight*score)
    return K.result(c,labels=y,score=c.s+weight*score,mechanism='F293_collision_driven_true_FG_support_split',accepted_splits=accepted,
             final_modes=len(fm),source_labels_unchanged=True,control=control)


def _questions(c):
    # Pool is fixed at32 before query-dependent tree selection. FullK^4
    # screening is reported and evaluated in source-only loop, not hidden.
    rs=c.source.fsim;rb=c.source.bsim;adj=K._adj(type('A',(),dict(q=c.r,edges=K.neighbors(c.rhw)))())
    pairs=[(i,j) for i,ids in enumerate(adj) for j in ids[1:]]
    first=np.array([a for a,b in pairs],int);neighbor=np.array([b for a,b in pairs],int)
    pool=[];roles=(c.c>.5)[first];known=(c.valid&c.source.oof_valid)[first]&(c.valid&c.source.oof_valid)[neighbor]
    for f1,b1,f2,b2 in product(range(len(c.fm)),range(len(c.bm)),range(len(c.fm)),range(len(c.bm))):
        if max(f1,f2)>=rs.shape[1] or max(b1,b2)>=rb.shape[1]:continue
        question=(rs[first,f1]-rb[first,b1])>(rs[neighbor,f2]-rb[neighbor,b2])
        valid=known&(rs[first,f1]>-1.5)&(rb[first,b1]>-1.5)&(rs[neighbor,f2]>-1.5)&(rb[neighbor,b2]>-1.5)
        if not np.any(valid&question) or not np.any(valid&~question):continue
        p=roles[valid].mean();entropy=lambda x: -x*np.log(max(x,EPS))-(1-x)*np.log(max(1-x,EPS))
        gain=entropy(p)-sum(np.mean(question[valid]==v)*entropy(roles[valid&(question==v)].mean()) for v in (False,True))
        if gain>EPS:pool.append((float(-gain),(f1,b1,f2,b2)))
    pool.sort(key=lambda x:(x[0],x[1]));return [v[1] for v in pool[:32]],len(c.fm)**2*len(c.bm)**2


def f295(c,weight=1.,control=False):
    questions,screened=_questions(c)
    if not questions:return K.result(c,score=c.s,fallback='no_positive_source_question_gain',question_pool_screened=screened)
    def observe(x,hw,source=False):
        fs=c.source.fsim if source else x@c.fm.T;bs=c.source.bsim if source else x@c.bm.T;adj=K._adj(type('A',(),dict(q=x,hw=hw,edges=K.neighbors(hw)))())
        # Average each of the actual9 neighborhoods via separate observations;
        # use nearest four-connected neighbor offsets, no arbitrary rowmean ID.
        pairs=[(i,j) for i,ids in enumerate(adj) for j in ids]
        i=np.array([a for a,b in pairs]);j=np.array([b for a,b in pairs]);columns=[]
        for f1,b1,f2,b2 in questions:columns.append((fs[i,f1]-bs[i,b1])>(fs[j,f2]-bs[j,b2]))
        return i,j,np.stack(columns,1)
    ri,rj,R=observe(c.r,c.rhw,True);qi,qj,Q=observe(c.q,c.hw)
    complete=c.valid&c.source.oof_valid&np.all(c.source.fsim>-1.5,1)&np.all(c.source.bsim>-1.5,1)
    valid=complete[ri]&complete[rj];labels=c.c[ri]>.5
    if not valid.any():return K.result(c,score=c.s,fallback='no_complete_source_question_observation')
    controversial=np.zeros(c.n)
    for a,b in combinations(c.Q,2):controversial+=(a!=b)
    query_mass=controversial[qi];source_mean=labels[valid].mean();nodes=[]
    def fit(ids,qids,depth):
        p=(labels[ids].sum()+1)/(len(ids)+2);leaf=float(np.log(p/(1-p)))
        if depth>=3 or len(ids)<4:return ('leaf',leaf)
        current_mean=labels[ids].mean()
        base=-current_mean*np.log(max(current_mean,EPS))-(1-current_mean)*np.log(max(1-current_mean,EPS));choices=[]
        for k in range(R.shape[1]):
            rsplit=R[ids,k]
            if not rsplit.any() or rsplit.all():continue
            impurity=0.
            for value in (False,True):
                sub=ids[rsplit==value];rate=labels[sub].mean();impurity+=len(sub)/len(ids)*(-rate*np.log(max(rate,EPS))-(1-rate)*np.log(max(1-rate,EPS)))
            gain=base-impurity
            if gain<=EPS:continue
            if control=='source':value=gain
            else:
                mass0=query_mass[qids[~Q[qids,k]]].sum();mass1=query_mass[qids[Q[qids,k]]].sum();value=min(mass0,mass1)
            choices.append((value,gain,-k,k))
        if not choices:return ('leaf',leaf)
        k=max(choices)[3];nodes.append(k)
        return ('node',k,fit(ids[~R[ids,k]],qids[~Q[qids,k]],depth+1),fit(ids[R[ids,k]],qids[Q[qids,k]],depth+1))
    tree=fit(np.flatnonzero(valid),np.arange(len(Q)),0)
    def evaluate(row,node):return node[1] if node[0]=='leaf' else evaluate(row,node[3] if row[node[1]] else node[2])
    result=np.array([evaluate(row,tree) for row in Q]);sumvalue=np.bincount(qi,weights=result,minlength=c.n);mass=np.bincount(qi,minlength=c.n)
    score=c.s+weight*sumvalue/np.maximum(mass,1);y=K.cut(c,score)
    return K.result(c,labels=y,score=score,mechanism='F295_candidate_conditioned_second_order_question_tree',
             question_pool=questions,screened_questions=screened,tree=tree,depth_bound=3,neighborhood_observations=len(Q),control=control)


def _proof_rules(c):
    rF=c.source.fsim;rB=c.source.bsim;known=c.source.oof_valid&c.valid
    if not rF.shape[1] or not rB.shape[1]:return []
    profile=c.source.heads[:,2];rules=[];adj=K._adj(type('A',(),dict(q=c.r,edges=K.neighbors(c.rhw)))())
    for role in (False,True):
        for f1,b1,f2,b2 in product(range(rF.shape[1]),range(rB.shape[1]),range(rF.shape[1]),range(rB.shape[1])):
            if (f1,b1)==(f2,b2):continue
            point=(rF[:,f1]-rB[:,b1]>0)==role
            neighbor=np.array([any(((rF[j,f2]-rB[j,b2])>0)==role for j in ids[1:] if known[j]) for ids in adj])
            trigger=known&point&neighbor&((profile>0)==role)
            if trigger.sum()<2:continue
            correct=np.count_nonzero((c.c>.5)[trigger]==role);fee=-np.log((correct+1)/(trigger.sum()+2))
            if correct/trigger.sum()>.5:rules.append((fee,role,f1,b1,f2,b2,int(trigger.sum())))
    rules.sort(key=lambda v:(v[0],v[1:]));return rules[:16]


def rooted_hypergraph(root,edges,OR=False):
    """Positive-fee AND relaxation; a rule cannot create an unrooted proof."""
    distance=np.asarray(root,float).copy();depend=[[] for _ in distance]
    for k,(a,b,t,fee) in enumerate(edges):
        if fee<=0:raise ValueError('Proof rule fee must be strictly positive')
        depend[a].append(k);depend[b].append(k)
    queue=[(float(v),i) for i,v in enumerate(distance) if np.isfinite(v)];heapq.heapify(queue);settled=np.zeros(len(root),bool)
    while queue:
        value,i=heapq.heappop(queue)
        if value>distance[i]+1e-12 or settled[i]:continue
        settled[i]=True
        for k in depend[i]:
            a,b,t,fee=edges[k]
            if OR:
                candidate=min(distance[a],distance[b])+fee
            else:
                if not settled[a] or not settled[b]:continue
                candidate=max(distance[a],distance[b])+fee
            if candidate<distance[t]-1e-12:distance[t]=candidate;heapq.heappush(queue,(candidate,t))
    return distance,settled


def f298(c,weight=1.,control=False):
    rules=_proof_rules(c)
    if not rules:return K.result(c,score=c.s,fallback='no_validated_positive_fee_two_premise_rule')
    def proof_fields(score,profile,fsim,bsim,hw,known):
        adj=K._adj(type('A',(),dict(q=np.zeros((len(score),1)),edges=K.neighbors(hw)))());fields=[];counts=[]
        for role in (False,True):
            root=np.where(known,np.logaddexp(0,score if not role else -score),np.inf);edges=[]
            for fee,cls,f1,b1,f2,b2,count in rules:
                if cls!=role or f1>=fsim.shape[1] or f2>=fsim.shape[1] or b1>=bsim.shape[1] or b2>=bsim.shape[1]:continue
                p1=known&(((fsim[:,f1]-bsim[:,b1])>0)==role)&(fsim[:,f1]>-1.5)&(bsim[:,b1]>-1.5)
                p2=known&(((fsim[:,f2]-bsim[:,b2])>0)==role)&(fsim[:,f2]>-1.5)&(bsim[:,b2]>-1.5)
                for target,ids in enumerate(adj):
                    left=[j for j in ids[1:] if p1[j]];right=[j for j in ids[1:] if p2[j]]
                    pair=next(((a,b) for a in left for b in right if a!=b),None)
                    if pair is not None and known[target] and ((profile[target]>0)==role):edges.append((*pair,target,weight*fee+EPS))
            d,settled=rooted_hypergraph(root,edges,OR=control=='OR');fields.append(d);counts.append(len(edges))
        difference=np.zeros(len(score));finite=np.isfinite(fields[0])&np.isfinite(fields[1]);difference[finite]=fields[0][finite]-fields[1][finite]
        return difference,counts
    difference,counts=proof_fields(c.s,c.heads[:,2],c.fsim,c.bsim,c.hw,np.ones(c.n,bool))
    # Source threshold uses out-of-block direct root scores. It is frozen for
    # Q, and direct roots remain finite at every valid query point.
    source_field,_=proof_fields(c.source.heads[:,0],c.source.heads[:,2],c.source.fsim,c.source.bsim,c.rhw,c.source.oof_valid&c.valid)
    source=source_field[c.source.oof_valid&c.valid];label=c.c[c.source.oof_valid&c.valid]>.5
    thresholds=np.r_[0.,np.quantile(source,[.1,.25,.5,.75,.9])] if len(source) else np.array([0.])
    losses=[np.mean((source>v)!=label) if len(source) else 1. for v in thresholds];threshold=float(thresholds[int(np.argmin(losses))])
    return K.result(c,score=difference-threshold,mechanism='F298_positive_fee_rooted_AND_proofs',rules=rules,
             directed_hyperedges=counts,source_threshold=threshold,control=control,cycle_cannot_self_root=True)


def _dictionary_predictive_cost(c,y,holdout=True):
    blocks=P.quarters(c.hw);fees=[];parameters=0
    dictionary=[K.pca(c.r[c.F],8),K.pca(c.r[c.B],8)]
    for block in (range(4) if holdout else [-1]):
        held=(blocks==block) if holdout else np.ones(c.n,bool);train=~held if holdout else np.ones(c.n,bool)
        for role,(mu,basis) in ((True,dictionary[0]),(False,dictionary[1])):
            fit=train&(y==role);test=held&(y==role)
            if not test.any():continue
            if fit.sum()<max(2,len(basis)):
                # Missing class has direct fixed-reference reconstruction cost.
                residual=c.q[test]-mu;fees.append(float(np.mean(np.maximum(np.sum(residual*residual,1)-np.sum((residual@basis.T)**2,1),0))));continue
            # One class coefficient vector is fitted only on training blocks;
            # the reference basis never changes to suit query hypotheses.
            beta=np.linalg.lstsq(basis.T,(c.q[fit]-mu).mean(0),rcond=None)[0] if len(basis) else np.empty(0)
            prediction=mu+beta@basis;fees.append(float(np.mean(np.sum((c.q[test]-prediction)**2,1))));parameters+=len(beta)
    direct=float(np.mean(np.where(y,np.logaddexp(0,-c.s),np.logaddexp(0,c.s))))
    return (np.mean(fees) if fees else 0.)+direct+parameters*np.log(max(c.n,2))/max(c.n,1)


def f299(c,weight=1.,control=False):
    if control=='direct':costs=np.array([c.energy(y) for y in c.Q])
    else:costs=np.array([c.energy(y)+weight*_dictionary_predictive_cost(c,y,holdout=control!='no_hold') for y in c.Q])
    j=min(range(len(costs)),key=lambda i:(costs[i],int(c.Q[i].sum()),i))
    return K.result(c,labels=c.Q[j],mechanism='F299_fixed_reference_dictionary_cross_block_hypothesis_prediction',
             selected_candidate=j,candidate_costs=costs.tolist(),query_blocks_are_not_independent_validation=True,
             reference_dictionary_rank=8,control=control)


KERNELS={286:f286,287:f287,289:f289,290:f290,291:f291,292:f292,293:f293,295:f295,298:f298,299:f299}
CONTROLS={286:['independent','average','source_best'],287:['AND','OR','average','independent'],289:['linear','profile','average'],
          290:['static','shell_mean'],291:['random','core','once'],292:['fixed'],293:['random','unvalidated'],295:['source'],298:['OR'],299:['no_hold','direct']}


def f296(c,weight=1.,control=False):
    """Original-pixel RGB variables are exposed only by a source-tested residual."""
    from . import e_helpers_226_250 as H
    from ics.methods.pro_paired_environment import exact_potts_cut
    ep=c.ep
    if ep.r_rgb is None or ep.q_rgb is None:raise common.ArtifactUnavailable('F296 requires actual R/Q original RGB')
    rshape=ep.r_rgb.shape[:2];qshape=ep.q_rgb.shape[:2]
    Ar=H.footprint(rshape,c.rhw,ep.reference_geometry);Aq=H.footprint(qshape,c.hw,ep.query_geometry)
    rphi=H.pixel_phi(ep.r_rgb);qphi=H.pixel_phi(ep.q_rgb)
    # True original pixels, not a token-colour imitation, provide Sobel and
    # local contrast targets. This reduction only trains the semantic predictor.
    rt=np.c_[Ar@rphi[:,-1],Ar@H.texture_phi(ep.r_rgb)[:,4]]
    prediction=np.full_like(rt,np.nan);fits=0
    for train,held in P.buffered_folds(c.rhw,c.valid):
        if train.sum()<3 or not held.any():continue
        design=c.r@c.r[train].T
        model=K.ridge(design[train],rt[train],1.,rank=16)
        prediction[held]=K.predict(model,design[held]);fits+=1
    known=np.isfinite(prediction).all(1)&c.valid
    if fits<2:return K.result(c,score=c.s,fallback='no_source_outblock_RGB_prediction')
    residual=rt-prediction
    boundary=np.zeros(len(c.r),bool)
    for a,b in K.neighbors(c.rhw):
        if c.valid[a] and c.valid[b] and (c.c[a]>.5)!=(c.c[b]>.5):boundary[[a,b]]=True
    direct=c.source.heads[:,2]
    base=_balanced_ridge(direct[known,None],boundary[known].astype(float),1)
    enhanced=_balanced_ridge(np.c_[direct[known],residual[known]],boundary[known].astype(float),3)
    if base is None or enhanced is None:return K.result(c,score=c.s,fallback='source_class_interface_unidentifiable')
    base_loss=np.mean((expit(_predict_ridge(base,direct[known,None]))-boundary[known])**2)
    extra_loss=np.mean((expit(_predict_ridge(enhanced,np.c_[direct[known],residual[known]]))-boundary[known])**2)
    if control is None and extra_loss>=base_loss-EPS:return K.result(c,score=c.s,fallback='source_RGB_residual_no_increment',source_residual_Brier=float(extra_loss),source_base_Brier=float(base_loss))
    final=K.ridge((c.r@c.r[c.valid].T)[c.valid],rt[c.valid],1.,rank=16)
    qpred=K.predict(final,c.q@c.r[c.valid].T)
    mapped=np.stack([H.native_to_original(qpred[:,k].reshape(c.hw),qshape,ep.query_geometry) for k in range(2)],-1).reshape(-1,2)
    actual=np.c_[qphi[:,-1],H.texture_phi(ep.q_rgb)[:,4]];qresidual=actual-mapped
    direct_original=H.native_to_original(c.heads[:,2].reshape(c.hw),qshape,ep.query_geometry).ravel()
    proxy=_predict_ridge(enhanced,np.c_[direct_original,qresidual])
    source_proxy=_predict_ridge(enhanced,np.c_[direct[known],residual[known]])
    nonboundary=~boundary[known]
    threshold=float(np.quantile(source_proxy[nonboundary],.95)) if nonboundary.any() else 0.
    domain=(proxy>threshold).reshape(qshape)
    if control in ('no_gate','GrabCut'):domain=np.ones(qshape,bool)
    baseprob=common.continuous_original(ep,c.p0.reshape(c.hw));base_margin=np.log(np.clip(baseprob,EPS,1-EPS)/np.clip(1-baseprob,EPS,1-EPS)).ravel()
    if control=='direct_head':
        rtokens=H.pixel_tokens(rshape,c.rhw,ep.reference_geometry);rknown=c.valid[rtokens]
        target_and_valid=_known_original_target(c)
        if target_and_valid is None:return K.result(c,score=c.s,fallback='source_original_pixels_unavailable')
        target,known_pixels=target_and_valid
        rprofile=H.native_to_original(c.source.heads[:,2].reshape(c.rhw),rshape,ep.reference_geometry).ravel()
        model=_balanced_ridge(np.c_[rprofile,rphi][rknown&known_pixels.ravel()],target.ravel()[rknown&known_pixels.ravel()].astype(float),16)
        score=_predict_ridge(model,np.c_[direct_original,qphi]);labels,_=exact_potts_cut(score,*H.rgb_edges(ep.q_rgb))
        return dict(original_labels=labels.reshape(qshape),original_unary=score.reshape(qshape),info=dict(control='actual_RGB_plus_complete_profile_rank16_head',source_residual_Brier=float(extra_loss),source_base_Brier=float(base_loss)))
    components,count=ndimage.label(domain);rgb=np.asarray(ep.q_rgb,float).reshape(-1,3)/255.;y=(baseprob>.5).ravel();edge,capacity=H.rgb_edges(ep.q_rgb)
    # Residual gate is a finite, nonnegative RGB boundary participation weight.
    gate=np.clip(expit(proxy),0,1);capacity=capacity*(gate[edge[:,0]]+gate[edge[:,1]])/2
    if control=='GrabCut':capacity=H.rgb_edges(ep.q_rgb)[1]
    previous_cost=np.inf;best=y.copy();hist=[];score=base_margin.copy()
    for iteration in range(3):
        colour=np.zeros(len(y));opened=np.zeros(len(y),bool)
        for component in range(1,count+1):
            ids=(components.ravel()==component);models=[]
            for role in (True,False):
                points=rgb[ids&(y==role)]
                if len(points)<3:break
                mu=points.mean(0);cov=np.cov(points.T,bias=True)+1e-4*np.eye(3);delta=rgb[ids]-mu
                sign,logdet=np.linalg.slogdet(cov)
                models.append(-.5*(np.sum(delta*np.linalg.solve(cov,delta.T).T,1)+logdet))
            if len(models)==2:colour[ids]=models[0]-models[1];opened[ids]=True
        score=base_margin+weight*colour
        labels,certificate=exact_potts_cut(score,edge,capacity)
        labels[~opened]=y[~opened]
        # Actual full objective is rescored after fitting the current models;
        # the finite alternating procedure has no global optimality claim.
        cost=float(np.logaddexp(0,score).sum()-score@labels+capacity@(labels[edge[:,0]]!=labels[edge[:,1]]));hist.append(cost)
        if cost<previous_cost:best=labels.copy();previous_cost=cost
        y=labels
    # The wrapper applies F's original-pixel disagreement against its fixed
    # native-unary rendering, while unexposed pixels retain the actual M0.
    best[~domain.ravel()]=c.M0.ravel()[~domain.ravel()]
    return dict(original_labels=best.reshape(qshape),original_unary=score.reshape(qshape),
       info=dict(mechanism='F296_residual_gated_original_RGB_latent_models',source_prediction_folds=fits,
          source_RGB_predictor_rank=16,source_residual_Brier=float(extra_loss),source_base_Brier=float(base_loss),
          source_gate_threshold=threshold,opened_pixels=int(domain.sum()),RGB_domains=count,
          alternating_rounds=3,objective_history=hist,global_optimum=False,control=control))


KERNELS[296]=f296
CONTROLS[296]=['direct_head','GrabCut','no_gate']
