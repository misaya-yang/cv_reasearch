"""D-family source protocol. No query labels, host fields, or external models.

All fitting is temporary, episode-local. The source holdouts rebuild the entire
method, including its query-derived observations, from the remaining R labels.
The finite optimizer budgets are part of the recipes, not convergence claims.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.special import expit, logsumexp

from .common import EPS, Result, validate, artifact, ArtifactUnavailable

_PAIR_CACHE = {}
DEFAULT_CONFIGS = ((4, 2, .25), (4, 2, 1.), (8, 4, .25), (8, 4, 1.), (16, 4, 1.))


def mm(a, b):
    # Accelerate can issue spurious floating-point status warnings; every result
    # is checked. No invalid numerical result is silently converted to a label.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        result = np.asarray(a) @ np.asarray(b)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite matrix product')
    return result


def pair(ep, which='qr'):
    """Exact original-D cosine, a one-episode cache capped at two FP32 grids."""
    key = (id(ep.q), id(ep.r))
    if _PAIR_CACHE.get('key') != key:
        _PAIR_CACHE.clear(); _PAIR_CACHE.update(key=key, q=ep.q, r=ep.r)
    if which not in _PAIR_CACHE:
        left = ep.q if which == 'qr' else ep.r
        _PAIR_CACHE[which] = np.clip(mm(np.asarray(left, np.float32),
                                              np.asarray(ep.r, np.float32).T), -1., 1.)
    return _PAIR_CACHE[which]


def cov(ep):
    return np.divide(ep.wf, ep.wvalid, out=np.zeros(len(ep.r)), where=ep.wvalid > 0)


def role(ep):
    c = cov(ep); valid = ep.wvalid > 0
    return valid & (c >= .5), valid & (c < .5)


def pure(ep):
    c = cov(ep); f = (ep.wvalid > 0) & (c >= .9); b = (ep.wvalid > 0) & (c <= .1)
    fallback = f.sum()<8 or b.sum()<8
    if f.sum()<8:f=ep.wf>0
    if b.sum()<8:b=ep.wb>0
    return f, b, fallback


def blocks(hw, n=2):
    y, x = np.indices(hw)
    return (np.minimum(y*n//hw[0], n-1)*n + np.minimum(x*n//hw[1], n-1)).ravel()


def folds(ep):
    """Four continuous quadrants, one-patch buffer, deterministic fold merging."""
    assignment = blocks(ep.r_hw); valid = ep.wvalid > 0
    def make(ids):
        held = valid & np.isin(assignment, ids)
        buffer = binary_dilation(held.reshape(ep.r_hw), structure=np.ones((3, 3))).ravel()
        train = valid & ~buffer
        return train, held
    def legal(fold):
        train, held = fold
        return held.any() and ep.wf[train].sum()>EPS and ep.wb[train].sum()>EPS
    single = [make([k]) for k in range(4)]
    if all(legal(s) for s in single):
        return single
    # Adjacent quadrants form two continuous half-image folds. Merging cannot
    # create a missing training class; both orientations are checked explicitly.
    for groups in (((0, 1), (2, 3)), ((0, 2), (1, 3))):
        merged = [make(k) for k in groups]
        if all(legal(s) for s in merged):
            return merged
    return []


def restricted(ep, train):
    # Query RGB/features remain legal; held R mask and all external resources
    # disappear from the fitting object. Only the training coverage is exposed.
    feature_names={'r_pre_final_ln','q_pre_final_ln','r_block_minus6','q_block_minus6',
                   'frozen_encode_rgb','frozen_input_gradient','r_raw','q_raw',
                   'r_pre_last2_patch_state','q_pre_last2_patch_state','tail_replay'}
    kept={k:v for k,v in getattr(ep,'artifacts',{}).items() if k in feature_names}
    def provider(_reduced,name):
        if name not in feature_names:
            raise ArtifactUnavailable('Source fold permits actual feature observations only: '+name)
        return artifact(ep,name)
    return replace(ep, wf=ep.wf*train, wvalid=ep.wvalid*train,
                   reference_mask=None, artifacts=kept, provider=provider)


def nearest_distance(sim, selected):
    if not np.any(selected):
        return np.full(len(sim), np.inf)
    return np.sqrt(np.maximum(0., 2.-2.*np.max(sim[:, selected], axis=1)))


def source_radius(ep, selected, quantile=.95):
    """Leave-spatial-block same-role radius; no self or adjacent-block shortcut."""
    sim = pair(ep, 'rr'); assignment = blocks(ep.r_hw)
    values = []
    for k in range(4):
        test = selected & (assignment == k)
        buffer=binary_dilation((assignment==k).reshape(ep.r_hw),structure=np.ones((3,3))).ravel()
        train = selected & ~buffer
        if test.any() and train.any():
            values.extend(nearest_distance(sim[test], train).tolist())
    return float(np.quantile(values, quantile)) if values else np.nan


def b0_field(ep, x=None):
    f, b = role(ep)
    sim = pair(ep, 'qr') if x is None or x is ep.q else mm(x, ep.r.T)
    if not f.any():
        return np.full(len(sim), -1.)
    if not b.any():
        radius = source_radius(ep, f)
        if not np.isfinite(radius):
            return np.full(len(sim), -1.)
        return radius - nearest_distance(sim, f)
    return np.max(sim[:, f], axis=1) - np.max(sim[:, b], axis=1)


def finish(ep, field, method, **info):
    score = np.asarray(field, float).reshape(ep.q_hw).copy()
    score.ravel()[ep.q_valid <= 0] = -1.
    if not np.isfinite(score).all():
        raise FloatingPointError(method+' produced a nonfinite field')
    return Result(score, 0., info=dict(method=method, **info))


def baseline(ep):
    validate(ep)
    return finish(ep, b0_field(ep), 'D_B0', baseline='full_reference_majority_nearest_role')


def project(ep, dimension=32):
    """Fixed seed-zero auxiliary projection, never a replacement identity cue."""
    d = min(dimension, ep.r.shape[1])
    if d == ep.r.shape[1]:
        return np.asarray(ep.r, float), np.asarray(ep.q, float), np.eye(d)
    a = np.random.default_rng(0).standard_normal((ep.r.shape[1], d))/np.sqrt(d)
    return mm(ep.r, a), mm(ep.q, a), a


def sample_weights(ep):
    fmass, bmass = float(ep.wf.sum()), float(ep.wb.sum())
    if min(fmass, bmass) <= EPS:
        return None
    weights = .5*ep.wf/fmass + .5*ep.wb/bmass
    # A weighted soft target preserves the two balanced role losses even at a
    # partially covered token; plain coverage with these weights does not.
    target = (.5*ep.wf/fmass)/np.maximum(weights, EPS)
    return target, weights


def logistic(x, target, weights, *, l2=None, steps=100, extra=None):
    x = np.asarray(x, float); target = np.asarray(target, float); weights = np.asarray(weights, float)
    active = weights > 0; x=x[active]; target=target[active]; weights=weights[active]
    if not len(x):
        raise ValueError('No legal source fitting mass')
    trace = float(np.sum(weights*np.sum(x*x, axis=1))/max(x.shape[1], 1))
    penalty = .01*max(trace, EPS) if l2 is None else float(l2)
    gram = mm(x.T, weights[:, None]*x)
    lipschitz = .25*(float(np.linalg.eigvalsh(gram)[-1])+weights.sum())+penalty
    if extra is not None:
        xx, yy, ww = extra
        xx=np.asarray(xx, float); yy=np.asarray(yy, float); ww=np.asarray(ww, float)
        lipschitz += .25*(float(np.linalg.eigvalsh(mm(xx.T, ww[:, None]*xx))[-1])+ww.sum())
    rate = 1./max(lipschitz, EPS); w=np.zeros(x.shape[1]); bias=0.
    for _ in range(steps):
        residual = weights*(expit(mm(x, w)+bias)-target)
        grad=mm(x.T, residual)+penalty*w; gb=residual.sum()
        if extra is not None:
            residual2=ww*(expit(mm(xx, w)+bias)-yy)
            grad+=mm(xx.T, residual2); gb+=residual2.sum()
        w-=rate*grad; bias-=rate*gb
    return w, float(bias)


def head(ep, xr=None, *, extra=None, keep=None):
    xr = project(ep)[0] if xr is None else xr
    targets = sample_weights(ep)
    if targets is None:
        return None
    y, weights = targets
    if keep is not None:
        weights=weights*np.asarray(keep)
        # Rebalance after removing a known negative role, never relabel it.
        f=.5*ep.wf*keep/max(float((ep.wf*keep).sum()), EPS)
        b=.5*ep.wb*keep/max(float((ep.wb*keep).sum()), EPS)
        weights=f+b; y=f/np.maximum(weights, EPS)
    return logistic(xr, y, weights, extra=extra)


def predict(model, x):
    if model is None:
        raise ValueError('Absent temporary classifier')
    return mm(x, model[0])+model[1]


def kmeans(x, k, *, spherical=True, weights=None):
    """Observed FPS starts; exactly twenty deterministic reassignment updates."""
    x=np.asarray(x, float)
    if not len(x):
        return np.empty((0, x.shape[1])), np.empty(0, int)
    z=x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS) if spherical else x
    chosen=[0]; distance=np.sum((z-z[0])**2, axis=1)
    for _ in range(1, min(k, len(z))):
        index=int(np.argmax(distance))
        if distance[index] <= EPS*EPS:
            break
        chosen.append(index); distance=np.minimum(distance, np.sum((z-z[index])**2, axis=1))
    centers=z[chosen].copy(); weights=np.ones(len(x)) if weights is None else np.asarray(weights)
    for _ in range(20):
        distances=np.sum(z*z, axis=1)[:, None]+np.sum(centers*centers, axis=1)-2*mm(z, centers.T)
        assignment=np.argmin(distances, axis=1)
        updated=[]
        for j in range(len(centers)):
            here=assignment == j
            center=np.average(z[here], axis=0, weights=weights[here]) if here.any() else centers[j]
            if spherical:
                center=center/max(float(np.linalg.norm(center)), EPS)
            updated.append(center)
        centers=np.asarray(updated)
    assignment=np.argmin(np.sum(z*z,axis=1)[:,None]+np.sum(centers*centers,axis=1)-2*mm(z,centers.T),axis=1)
    return centers, assignment


def source_modes(x,k,*,spherical=True,weights=None):
    """No separately fitted component with fewer than eight physical samples."""
    x=np.asarray(x,float)
    if len(x)<8:return np.empty((0,x.shape[1])),np.empty(0,int)
    count=min(int(k),len(x)//8)
    while count>0:
        centers,assignment=kmeans(x,count,spherical=spherical,weights=weights)
        if min(np.bincount(assignment,minlength=len(centers)))>=8:return centers,assignment
        count-=1
    return np.empty((0,x.shape[1])),np.empty(0,int)


def fps_rows(x, maximum):
    x=np.asarray(x, float)
    if not len(x):
        return x.copy()
    chosen=[0]; nearest=np.sum((x-x[0])**2, axis=1)
    for _ in range(1, min(int(maximum), len(x))):
        new=int(np.argmax(nearest))
        if nearest[new] <= EPS*EPS:
            break
        chosen.append(new); nearest=np.minimum(nearest, np.sum((x-x[new])**2, axis=1))
    return x[chosen].copy()


def risk(ep, score, selected=None, threshold=0.):
    selected = ep.wvalid > 0 if selected is None else selected
    p=np.asarray(score)>threshold
    fm=float(ep.wf[selected].sum()); bm=float(ep.wb[selected].sum())
    if min(fm, bm) <= EPS:
        return np.inf
    return float(ep.wf[selected & ~p].sum()/fm + ep.wb[selected & p].sum()/bm)


def threshold_candidates(score):
    values=np.unique(np.asarray(score, float))
    if not len(values):
        return np.asarray([0.])
    return np.r_[np.nextafter(values[0], -np.inf),
                 values[:-1]+(values[1:]-values[:-1])/2., values[-1], 0.]


def best_threshold(ep, score, selected, base, minimum_threshold=None):
    """Exact finite partitions, balanced missed-FG + false-BG objective."""
    ids=np.flatnonzero(selected); order=ids[np.argsort(score[ids], kind='stable')]
    s=score[order]; fw=ep.wf[order]; bw=ep.wb[order]
    if min(fw.sum(), bw.sum()) <= EPS:
        return 0., np.inf, 0
    prefix_f=np.r_[0., np.cumsum(fw)]; prefix_b=np.r_[0., np.cumsum(bw)]
    cuts=np.r_[0, np.flatnonzero(np.diff(s)>0)+1, len(s)]
    errors=prefix_f[cuts]/fw.sum()+(bw.sum()-prefix_b[cuts])/bw.sum()
    candidates=np.empty(len(cuts)); candidates[0]=np.nextafter(s[0], -np.inf); candidates[-1]=s[-1]
    if len(cuts)>2:
        left=cuts[1:-1]; candidates[1:-1]=s[left-1]+(s[left]-s[left-1])/2.
    if minimum_threshold is not None:
        candidates=np.r_[candidates,float(minimum_threshold)]
        errors=np.r_[errors,risk(ep,score,selected,minimum_threshold)]
        errors[candidates<minimum_threshold]=np.inf
    minimum=float(errors.min()); tied=np.flatnonzero(np.abs(errors-minimum)<1e-12)
    edit=[int(np.count_nonzero((score[selected]>candidates[j]) != (base[selected]>0))) for j in tied]
    winner=int(tied[np.argmin(edit)])
    return float(candidates[winner]), minimum, min(edit)


def calibrate(ep, method, builder, configs=DEFAULT_CONFIGS, minimum_threshold=None):
    """Five source configurations plus zero adaptation, at most six candidates.

    builder(train_episode, configuration) returns a callable scoring arbitrary
    original-D rows and diagnostics. Every fold rebuilds all private state.
    """
    validate(ep); base=b0_field(ep)
    if ep.wf.sum()<=EPS or ep.wb.sum()<=EPS:
        return finish(ep, base, method, branch='single_role_documented_B0')
    spatial=folds(ep)
    if not spatial:
        return finish(ep, base, method, branch='missing_role_after_fold_merge_B0')
    selected=np.zeros(len(ep.r), bool); base_oof=np.zeros(len(ep.r)); trains=[]
    for train, held in spatial:
        reduced=restricted(ep, train); selected|=held
        base_oof[held]=b0_field(reduced, ep.r[held]); trains.append((reduced, held))
    best=(risk(ep, base_oof, selected), 0, -1); chosen=None; summaries=[]
    for index, config in enumerate(configs):
        score=np.zeros(len(ep.r)); valid=True
        for reduced, held in trains:
            scorer, detail=builder(reduced, config)
            if scorer is None:
                valid=False; break
            score[held]=(scorer.source_score(np.flatnonzero(held)) if hasattr(scorer,'source_score')
                         else scorer(ep.r[held]))
        if not valid or not np.isfinite(score[selected]).all():
            summaries.append(dict(config=config, legal=False)); continue
        threshold, error, edits=best_threshold(ep, score, selected, base_oof, minimum_threshold)
        summaries.append(dict(config=config, legal=True, source_error=error, edits=edits, threshold=threshold))
        key=(error, edits, index)
        if key < best:
            best=key; chosen=(config, threshold)
    if chosen is None:
        return finish(ep, base, method, branch='source_selected_zero_adaptation', source_candidates=summaries,
                      source_baseline_error=float(best[0]), source_folds=len(spatial))
    scorer, detail=builder(ep, chosen[0])
    if scorer is None:
        return finish(ep, base, method, branch='full_fit_unavailable_B0', source_candidates=summaries)
    score=scorer(ep.q)-chosen[1]
    return finish(ep, score, method, branch='source_selected_method', configuration=chosen[0],
                  source_threshold=chosen[1], source_error=float(best[0]), source_folds=len(spatial),
                  source_candidates=summaries, temporary_reference_fitting=True, **detail)


def fg_recall_non_decrease(ep, original, changed, mode_id, selected):
    for k in np.unique(mode_id[selected & (ep.wf>0)]):
        here=selected & (mode_id==k) & (ep.wf>0)
        if ep.wf[here & (changed>0)].sum()+1e-12 < ep.wf[here & (original>0)].sum():
            return False
    return True


def foreground_modes(ep, k=8):
    f,b,_=pure(ep); assignment=np.full(len(ep.r), -1, int)
    centers, ids=source_modes(ep.r[f], k,weights=ep.wf[f])
    if len(centers):assignment[f]=ids
    return centers, assignment


def qualified_anchors(ep,positive):
    """Common source-supported, double-agreeing nonprobabilistic anchors."""
    xr,xq,p=project(ep); model=head(ep,xr); f,b=role(ep); selected=f if positive else b
    if model is None or not selected.any():
        return np.empty(0,int)
    source=np.full(len(ep.r),np.nan)
    for train,held in folds(ep):
        fitted=head(restricted(ep,train),xr); source[held]=predict(fitted,xr[held])
    other=b if positive else f
    negatives=source[other&np.isfinite(source)]
    if not len(negatives):
        return np.empty(0,int)
    threshold=max(0.,float(np.max(negatives))) if positive else min(0.,float(np.min(negatives)))
    interval=selected&np.isfinite(source)&((source>threshold) if positive else (source<threshold))
    if interval.sum()<8 or len(np.unique(blocks(ep.r_hw)[interval]))<2:
        return np.empty(0,int)
    radius=source_radius(ep,selected)
    if not np.isfinite(radius):
        return np.empty(0,int)
    raw=b0_field(ep); hs=predict(model,xq); distance=nearest_distance(pair(ep),selected)
    agreement=(raw>0)&(hs>threshold) if positive else (raw<0)&(hs<threshold)
    return np.flatnonzero(agreement&(distance<=radius)&(ep.q_valid>0))


def sparse_cost(x, atoms, penalty, updates=20):
    """Signed L1 coordinates, twenty complete sweeps; nested cost stays valid."""
    x=np.asarray(x, float); atoms=np.asarray(atoms, float)
    if not len(atoms):
        return np.sum(x*x, axis=1)
    gram=mm(atoms, atoms.T); correlations=mm(x, atoms.T); a=np.zeros_like(correlations)
    for _ in range(updates):
        for j in range(len(atoms)):
            partial=correlations[:, j]-mm(a, gram[:, j])+a[:, j]*gram[j,j]
            a[:,j]=np.sign(partial)*np.maximum(np.abs(partial)-penalty/2., 0.)/max(gram[j,j], EPS)
    cost=np.sum(x*x, axis=1)-2*np.sum(a*correlations, axis=1)+np.sum(mm(a, gram)*a, axis=1)
    return np.maximum(cost, 0.)+penalty*np.sum(np.abs(a), axis=1)


def calibrate_field(ep,method,builder,configs=DEFAULT_CONFIGS):
    """C for explicitly query-spatial methods: R is the held-out pseudo query.

    Its RGB/descriptor geometry is supplied, never the held reference mask.
    Query-native algorithms are rebuilt on that pseudo query, not evaluated by
    pretending a region-specific score is a function of features alone.
    """
    validate(ep); spatial=folds(ep); base=b0_field(ep)
    if not spatial:
        return finish(ep,base,method,branch='missing_role_after_fold_merge_B0')
    selected=np.zeros(len(ep.r),bool); base_oof=np.zeros(len(ep.r)); episodes=[]
    for train,held in spatial:
        reduced=restricted(ep,train)
        pseudo=replace(reduced,q=ep.r,q_hw=ep.r_hw,q_valid=ep.wvalid.copy(),q_rgb=ep.r_rgb,
                       query_geometry=ep.reference_geometry,original_shape=ep.r_hw)
        selected|=held; base_oof[held]=b0_field(pseudo)[held]; episodes.append((pseudo,held))
    best=(risk(ep,base_oof,selected),0,-1); chosen=None; summaries=[]
    for j,cfg in enumerate(configs):
        field=np.zeros(len(ep.r)); legal=True
        for pseudo,held in episodes:
            out,info=builder(pseudo,cfg)
            if out is None:
                legal=False;break
            field[held]=np.asarray(out).ravel()[held]
        if not legal:
            summaries.append(dict(config=cfg,legal=False));continue
        threshold,error,edits=best_threshold(ep,field,selected,base_oof)
        summaries.append(dict(config=cfg,legal=True,source_error=error,threshold=threshold,edits=edits))
        if (error,edits,j)<best:
            best=(error,edits,j);chosen=(cfg,threshold)
    if chosen is None:
        return finish(ep,base,method,branch='source_selected_zero_adaptation',source_candidates=summaries)
    field,info=builder(ep,chosen[0])
    if field is None:
        return finish(ep,base,method,branch='full_fit_unavailable_B0',source_candidates=summaries)
    return finish(ep,np.asarray(field).ravel()-chosen[1],method,branch='source_selected_method',
                  configuration=chosen[0],source_error=best[0],source_threshold=chosen[1],
                  source_candidates=summaries,calibration_query='reference_spatial_pseudo_query',**info)
