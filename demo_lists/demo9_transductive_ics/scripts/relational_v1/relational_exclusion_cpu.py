"""Exploratory frozen-feature shared-star exclusion CPU reference.

Same input banks for center, independent edges, shared appearance and shared
appearance/geometry. No encoder, downloads, training, GT, confidence threshold
or quality claim. Evidence is restricted to sampled top8 role-balanced roots.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import numpy as np

MODES=('center_only','independent_edges','shared_U','shared_UR','joint_flip')


class RelationContractError(ValueError):pass


def _bool(a,shape,name):
    a=np.asarray(a)
    if a.shape!=shape or a.dtype!=np.bool_:raise RelationContractError(f'{name}: bool {shape} required')
    return a


def _features(value,name):
    a=np.asarray(value)
    if a.ndim!=3 or min(a.shape)<1 or a.dtype.kind!='f':
        raise RelationContractError(f'{name}: nonempty floating [h,w,D] required')
    a=a.astype(np.float64)
    finite=np.isfinite(a).all(-1)
    with np.errstate(over='ignore',invalid='ignore'):
        norm=np.linalg.norm(np.where(np.isfinite(a),a,0.),axis=-1)
    good=finite&np.isfinite(norm)&(norm>1e-12)
    unit=np.zeros_like(a);unit[good]=a[good]/norm[good,None]
    return unit,good


@dataclass(frozen=True)
class Star:
    root: int
    ids: tuple
    features: np.ndarray
    distances: np.ndarray
    adjacent_foreground: bool=False


def build_stars(features, coordinates, valid, *, source_coverage=None):
    """Fixed radius2, maximin feature companions; stable original-index ties."""
    f,good=_features(features,'features');shape=f.shape[:2]
    xy=np.asarray(coordinates)
    if xy.shape!=shape+(2,) or xy.dtype.kind not in 'fiu':
        raise RelationContractError('coordinates must be numeric [h,w,2]')
    xy=xy.astype(float);good &= np.isfinite(xy).all(-1)&_bool(valid,shape,'valid')
    if source_coverage is not None:
        coverage=np.asarray(source_coverage)
        if coverage.shape!=shape or coverage.dtype.kind not in 'fiu' or not np.isfinite(coverage).all() or np.any((coverage<0)|(coverage>1)):
            raise RelationContractError('source coverage must be finite [h,w] in [0,1]')
        coverage=coverage.reshape(-1)
    else:coverage=None
    h,w=shape;flat=f.reshape(-1,f.shape[-1]);xy=xy.reshape(-1,2);stars={}
    for root in np.flatnonzero(good):
        y,x=divmod(int(root),w)
        candidates=[yy*w+xx for yy in range(max(0,y-2),min(h,y+3))
                    for xx in range(max(0,x-2),min(w,x+3)) if yy*w+xx!=root and good[yy,xx]]
        if len(candidates)<2:continue
        ids=np.asarray(candidates,dtype=int) # C-order already ascending
        d0=1.-np.clip(flat[ids]@flat[root],-1.,1.)
        a=int(ids[np.argmax(d0)])
        keep=ids!=a;remaining=ids[keep]
        rank=np.minimum(d0[keep],1.-np.clip(flat[remaining]@flat[a],-1.,1.))
        b=int(remaining[np.argmax(rank)])
        triple=(int(root),a,b);points=xy[list(triple)]
        distances=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
        if np.any(distances[np.triu_indices(3,1)]<=0):continue
        distances/=distances.max()
        adjacent=bool(coverage is not None and (coverage[a]>0 or coverage[b]>0))
        stars[int(root)]=Star(int(root),triple,flat[list(triple)].copy(),distances,adjacent)
    return stars,good


def _center_cache(source_stars,source_coverage):
    ids=np.asarray(sorted(source_stars),dtype=np.int64)
    if not len(ids):return ids,None,np.zeros(0,bool)
    centers=np.stack([source_stars[int(i)].features[0] for i in ids])
    foreground=np.asarray(source_coverage).reshape(-1)[ids]>0
    return ids,centers,foreground


def select_role_banks(query, source_stars, source_coverage, *, center_cache=None):
    """Exactly equal k=min(8,nF,nB), centered-cosine truncation only.

    One vectorized center matrix-vector product; stable lexicographic ties.
    Main caller constructs/cache-stores the source center matrix only once.
    """
    ids,centers,is_F=_center_cache(source_stars,source_coverage) if center_cache is None else center_cache
    k=min(8,int(is_F.sum()),int((~is_F).sum()))
    if not k:return [],[]
    sim=np.clip(centers@query.features[0],-1.,1.)
    roles=[]
    for mask in (is_F,~is_F):
        selected=np.flatnonzero(mask)
        order=np.lexsort((ids[selected],-sim[selected]))[:k]
        roles.append([source_stars[int(ids[i])] for i in selected[order]])
    return roles[0],roles[1]


def assignment_costs(query, source, *, _cost_matrix=None):
    """U and R evaluated on the IDENTICAL shared bijection, never decoupled."""
    C=1.-np.clip(query.features@source.features.T,-1.,1.) if _cost_matrix is None else _cost_matrix
    out=[]
    for perm in ((0,1,2),(0,2,1)):
        # Identical primitive/order to the relaxed edges prevents spurious
        # strict flips at floating ties, without adding a tolerance cutoff.
        U=float((.5*C[0,0]+C[1,perm[1]])/3.)+float((.5*C[0,0]+C[2,perm[2]])/3.)
        R=float(np.mean([abs(query.distances[i,j]-source.distances[perm[i],perm[j]])
                         for i,j in ((0,1),(0,2),(1,2))]))
        out.append((U,R,perm))
    return out


def compare_same_banks(query, foreground, background):
    """Controls share hypotheses and unary center/companion weights.

    Independent edges split center weight equally: each edge chooses its own
    source star/companion minimizing (0.5*c0+cj)/3. Sum the two minima.
    This relaxes consistency/injectivity without overweighting the center.
    """
    if not foreground or len(foreground)!=len(background) or len(foreground)>8:
        return {'delete':{mode:False for mode in MODES},'status':'abstain_missing_or_unequal_roles'}
    costs=[];summaries=[]
    for role,bank in enumerate((foreground,background)):
        center=[];edges=[[],[]];shared=[]
        for source in bank:
            C=1.-np.clip(query.features@source.features.T,-1.,1.)
            center.append(float(C[0,0]))
            for j in (1,2):edges[j-1].extend((float((.5*C[0,0]+C[j,k])/3.),source.root,k,source.ids[k]) for k in (1,2))
            for U,R,perm in assignment_costs(query,source,_cost_matrix=C):
                costs.append((U,R,role,source.root,perm));shared.append((U,source.root,perm,R))
        winners=[min(e) for e in edges];joint=min(shared)
        summaries.append({'center_only':min(center),'independent_edges':sum(e[0] for e in winners),'shared_U':joint[0],
            'independent_winners':[{'weighted_cost':e[0],'source_root':e[1],'source_companion_slot':e[2],'source_companion_id':e[3]} for e in winners],
            'shared_winner':{'source_root':joint[1],'permutation':joint[2],'U':joint[0],'R':joint[3]}})
    values=np.asarray([(a[0],a[1]) for a in costs]);nondominated=[]
    for i,candidate in enumerate(values):
        dominates=np.all(values<=candidate,axis=1)&np.any(values<candidate,axis=1)
        if not np.any(dominates):nondominated.append(i)
    delete={mode:summaries[1][mode]<summaries[0][mode] for mode in ('center_only','independent_edges','shared_U')}
    delete['shared_UR']=bool(nondominated) and all(costs[i][2]==1 for i in nondominated)
    LF,LB=summaries[0]['independent_edges'],summaries[1]['independent_edges']
    UF,UB=summaries[0]['shared_U'],summaries[1]['shared_U']
    delete['joint_flip']=LF<LB and UB<UF
    return {'delete':delete,'status':'ok','joint_flip_predicate':{'LF':LF,'LB':LB,'UF':UF,'UB':UB,
            'delta_F':UF-LF,'delta_B':UB-LB,'independent_FG_advantage':LB-LF,
            'joint_consistency_advantage':(UF-LF)-(UB-LB),'accepted':delete['joint_flip'],
            'definition':'LF<LB and UB<UF; strict ties retain; no GT'},'role_costs':{'F':summaries[0],'B':summaries[1]},
            'assignments':costs,'pareto_indices':nondominated,
            'pareto_has_F':any(costs[i][2]==0 for i in nondominated),
            'query_ids':query.ids,'sampled_F_stars':[{'root':s.root,'ids':s.ids} for s in foreground],
            'sampled_B_stars':[{'root':s.root,'ids':s.ids,'adjacent_foreground':s.adjacent_foreground} for s in background],
            'sampled_F_roots':[s.root for s in foreground],'sampled_B_roots':[s.root for s in background],
            'sampled_B_adjacent':[s.adjacent_foreground for s in background]}


def relational_exclusion(*,query_features,source_features,query_xy,source_xy,
                          query_valid,source_valid,source_coverage,native_patch_fg):
    q=np.asarray(query_features);s=np.asarray(source_features)
    if q.ndim!=3 or s.ndim!=3 or q.shape[-1]!=s.shape[-1]:raise RelationContractError('source/query channels must match')
    native=_bool(native_patch_fg,q.shape[:2],'native_patch_fg')
    qstars,qgood=build_stars(q,query_xy,query_valid)
    sstars,sgood=build_stars(s,source_xy,source_valid,source_coverage=source_coverage)
    cache=_center_cache(sstars,source_coverage)
    fields={mode:np.zeros(native.shape,bool) for mode in MODES};records=[];evaluated=0;sampled=0
    for root in np.flatnonzero(native):
        if int(root) not in qstars:
            records.append({'query_root':int(root),'status':'abstain_invalid_or_insufficient_constellation'});continue
        query=qstars[int(root)];F,B=select_role_banks(query,sstars,source_coverage,center_cache=cache)
        result=compare_same_banks(query,F,B);evaluated+=1;sampled+=len(F)+len(B)
        for mode in MODES:fields[mode].reshape(-1)[root]=result['delete'][mode]
        records.append({'query_root':int(root),**result})
    D=q.shape[-1]
    return {'deletions':fields,'records':records,'metadata':{
        'status':'exploratory CPU reference; real quality unrun','modes':list(MODES),
        'selected_operator':'joint_flip','selected_rule':'native_FG AND LF<LB AND UB<UF',
        'query_native_FG':int(native.sum()),'query_valid_stars':len(qstars),'source_valid_stars':len(sstars),
        'evaluated_query_stars':evaluated,'sampled_source_stars_total':sampled,
        'hypothesis_scope':'equal-role topk by center cosine; k=min(8,nF,nB); not all source assignments',
        'geometry_role':'same-assignment Pareto veto; no guarantee of query background',
        'cost_estimate':{'center_search_dot_MAC_upper':evaluated*len(sstars)*D,
            'cross_endpoint_dot_MAC_upper':sampled*9*D,
            'assignments_upper':2*sampled,'note':'estimates exclude local-star construction and Python overhead; not timing'},
        'query_GT_used':False,'encoder_calls':0,'quality_gain_verified':False}}


def nearest_lift(field,output_hw):
    a=np.asarray(field)
    if a.dtype!=np.bool_ or a.ndim!=2 or not a.size:raise RelationContractError('deletion field must be nonempty bool2D')
    if len(output_hw)!=2 or any(not isinstance(n,(int,np.integer)) or n<1 for n in output_hw):raise RelationContractError('output shape must be positive integers')
    yy=np.arange(output_hw[0])*a.shape[0]//output_hw[0];xx=np.arange(output_hw[1])*a.shape[1]//output_hw[1]
    return a[yy[:,None],xx[None,:]]


def finalize_deletion(deletion, *,native_pre_final,native_final,finalizer,
                      deletion_lifter=None,mapping_contract=None):
    """Nearest deletion lift -> native pre-mask intersection -> original R ->
    original native final intersection. Final action is deletion-only by construction.
    """
    pre=np.asarray(native_pre_final);saved=np.asarray(native_final)
    for a in (pre,saved):
        if a.ndim!=2 or not a.size or a.dtype!=np.bool_:raise RelationContractError('native masks must be nonempty bool2D')
    if not callable(finalizer):raise RelationContractError('original finalizer callback required')
    field=np.asarray(deletion)
    if field.dtype!=np.bool_ or field.ndim!=2 or not field.size:raise RelationContractError('deletion must be nonempty bool2D')
    if deletion_lifter is not None:
        lifted=np.asarray(deletion_lifter(field.copy(),pre.shape))
        lifted=_bool(lifted,pre.shape,'lifted deletion')
        mapping='caller supplied exact deletion lifter'
    elif field.shape==pre.shape:
        lifted=field.copy();mapping='same-grid identity'
    elif mapping_contract=='full_grid_nearest':
        lifted=nearest_lift(field,pre.shape);mapping='caller-verified full_grid_nearest'
    else:
        raise RelationContractError('different grids require exact deletion_lifter or caller-verified full_grid_nearest mapping_contract')
    candidate=pre&~lifted
    noop=np.array_equal(candidate,pre)
    if noop:final=saved.copy()
    else:
        final=np.asarray(finalizer(candidate.copy()))
        if final.dtype!=np.bool_ or final.shape!=saved.shape:raise RelationContractError('finalizer bool output must match native final grid')
        final=final&saved
    return {'pre_final':candidate,'final_mask':final.copy(),'metadata':{
        'mapping_contract':mapping,'finalizer_calls':int(not noop),'changed_pre_final':int((pre&~candidate).sum()),
        'deleted_final':int((saved&~final).sum()),'final_subset_native':True,
        'original_final_intersection_applied':not noop,'native_noop_reused':noop}}


PACKET_FIELDS={'query_features','source_features','query_xy','source_xy','query_valid','source_valid','source_coverage','native_patch_fg'}
META_FIELDS={'representation_id','source_role_definition','native_patch_mask_definition','feature_provenance','coordinate_system'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',required=True);p.add_argument('--metadata',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(argv);metadata=json.loads(Path(a.metadata).read_text())
    if any(not isinstance(metadata.get(k),str) or not metadata[k].strip() for k in META_FIELDS):
        raise RelationContractError('metadata requires nonempty '+','.join(sorted(META_FIELDS)))
    path=Path(a.out)
    if path.suffix!='.npz':raise RelationContractError('--out must end .npz')
    if path.exists() or path.with_suffix('.json').exists():raise RelationContractError('fresh output paths required')
    with np.load(a.packet,allow_pickle=False) as z:
        if set(z.files)!=PACKET_FIELDS:raise RelationContractError('packet must contain exactly documented label-free method fields; no query GT')
        result=relational_exclusion(**{name:z[name] for name in PACKET_FIELDS})
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**{mode+'_delete':field for mode,field in result['deletions'].items()})
    path.with_suffix('.json').write_text(json.dumps({'input_metadata':metadata,**result['metadata'],
        'output_scope':'query-grid deletion fields ONLY; original finalizer/RGB not supplied or run',
        'records':result['records']},indent=2)+'\n')
    print(json.dumps({'output':str(path),'scope':'query-grid only','counts':{k:int(v.sum()) for k,v in result['deletions'].items()},'quality_gain_verified':False}))

if __name__=='__main__':main()
