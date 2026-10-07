"""M05 finite multi-template/multi-pose Hough and the specified local decoder."""
from __future__ import annotations

from functools import partial
import time
import numpy as np
from scipy import ndimage
from scipy.special import logsumexp

from ics.cpu100.cross_image_matching import _roles
from .common import (EPS,validate,unit,fit_br,br_margin,finish,degenerate_margin,source_contract,ArtifactUnavailable)

_CACHE=None


def _components(ep):
    if ep.reference_mask is None:raise ArtifactUnavailable('M05 requires the actual complete original MR, no coarse-mask substitute')
    mask=np.asarray(ep.reference_mask,bool);labels,count=ndimage.label(mask)
    h,w=mask.shape;g=ep.reference_geometry
    if not g:
        if mask.shape!=ep.r_hw:raise ArtifactUnavailable('M05 original MR physical mapping unavailable')
        py,px=np.arange(h),np.arange(w)
    else:
        sh,sw=g['resized_hw'];oy,ox=g['padding_top_left'];side=g['view_side']
        py=np.minimum(ep.r_hw[0]-1,((oy+(np.arange(h)+.5)*sh/h)*ep.r_hw[0]/side).astype(int))
        px=np.minimum(ep.r_hw[1]-1,((ox+(np.arange(w)+.5)*sw/w)*ep.r_hw[1]/side).astype(int))
    native=py[:,None]*ep.r_hw[1]+px[None];pixelcounts=np.bincount(native.ravel(),minlength=len(ep.r))
    foreground=np.bincount(native[mask],minlength=len(ep.r));components=[]
    for cid in range(1,count+1):
        local=labels==cid;mass=np.bincount(native[local],minlength=len(ep.r))
        share=np.divide(mass,foreground,out=np.zeros(len(ep.r)),where=foreground>0)
        coverage=np.divide(ep.wf*share,ep.wvalid,out=np.zeros(len(ep.r)),where=ep.wvalid>0)
        ids=np.flatnonzero((coverage>=.9)&(ep.wvalid>0))
        area=int(local.sum());allids=np.flatnonzero(mass>0)
        components.append(dict(id=cid,area=area,coverage=coverage,ids=ids,allids=allids))
    components.sort(key=lambda c:(-c['area'],c['id']))
    return components,count,sum(c['area'] for c in components[4:])


def _spatial_fps(ep,ids,cap=16):
    if not len(ids):return ids
    yy,xx=np.unravel_index(ids,ep.r_hw);coordinates=np.c_[yy,xx].astype(float)
    chosen=[0];distance=np.sum((coordinates-coordinates[0])**2,axis=1)
    while len(chosen)<min(cap,len(ids)):
        distance[chosen]=-1;index=int(np.argmax(distance));chosen.append(index)
        distance=np.minimum(distance,np.sum((coordinates-coordinates[index])**2,axis=1))
    return ids[chosen]


def _sample(field,y,x):
    valid=(y>=0)&(x>=0)&(y<=field.shape[0]-1)&(x<=field.shape[1]-1)
    values=ndimage.map_coordinates(field,[y,x],order=1,mode='nearest')
    return values,valid


def _trim80(values,valid):
    ordered=np.sort(np.where(valid,values,np.inf),axis=0);n=valid.sum(0)
    j=np.arange(len(values))[:,None];lo=j/np.maximum(n,1);hi=(j+1)/np.maximum(n,1)
    weights=np.maximum(np.minimum(hi,.9)-np.maximum(lo,.1),0)
    weights=np.where(j<n,weights,0);return np.sum(np.where(weights>0,ordered,0)*weights,axis=0)/.8


def _iou(a,b):
    intersection=max(0,min(a[1],b[1])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[2],b[2]))
    aa=max(0,a[1]-a[0])*max(0,a[3]-a[2]);bb=max(0,b[1]-b[0])*max(0,b[3]-b[2])
    return intersection/max(aa+bb-intersection,EPS)


def _templates(ep):
    global _CACHE
    if _CACHE is not None and _CACHE['ep'] is ep:return _CACHE,True
    start=time.perf_counter();components,count,omitted=_components(ep);f,b,_,_=_roles(ep,32)
    if not len(b):raise ArtifactUnavailable('M05 no numerical background dictionary despite nonzero source mass')
    sB,brinfo=br_margin(ep);rB=fit_br(ep).predict(ep.r)
    mean=np.sum(rB*ep.wvalid)/ep.wvalid.sum();sigma=max(float(np.sqrt(np.sum(ep.wvalid*(rB-mean)**2)/ep.wvalid.sum())),.05)
    background=.07*logsumexp(ep.q@b.T/.07,axis=1)
    templates=[]
    for comp in components[:4]:
        anchorids=_spatial_fps(ep,comp['ids'],16)
        if len(anchorids)<4:continue
        yy,xx=np.unravel_index(comp['allids'],ep.r_hw)
        bbox=np.array([yy.min()-.5,yy.max()+.5,xx.min()-.5,xx.max()+.5],float)
        center=np.array([(bbox[0]+bbox[1])/2,(bbox[2]+bbox[3])/2])
        ay,ax=np.unravel_index(anchorids,ep.r_hw);offset=np.c_[ay,ax]-center
        support=(comp['coverage']>.5).reshape(ep.r_hw);ring=ndimage.binary_dilation(support)&~support
        ry,rx=np.nonzero(ring);ringoffset=np.c_[ry,rx]-center
        raw=ep.q@ep.r[anchorids].T-background[:,None]
        valid=ep.q_valid>0;median=np.median(raw[valid],axis=0);q25,q75=np.quantile(raw[valid],[.25,.75],axis=0);scale=np.maximum(q75-q25,.05)
        v=((raw-median)/scale).T.reshape(len(anchorids),*ep.q_hw)
        meanraw=ep.q@unit(ep.r[anchorids].mean(0))-background
        ml,mh=np.quantile(meanraw[valid],[.25,.75]);mean_feature=((meanraw-np.median(meanraw[valid]))/max(float(mh-ml),.05)).reshape(ep.q_hw)
        templates.append(dict(component_id=comp['id'],anchorids=anchorids,offset=offset,ringoffset=ringoffset,
            bbox=bbox-center[[0,0,1,1]],v=v,meanv=v.mean(0),mean_feature=mean_feature))
    _CACHE=dict(ep=ep,templates=templates,sB=sB,rB=rB,sigmaB=sigma,BR=brinfo,
        info=dict(reference_components=count,template_components=len(templates),non_top4_component_area=omitted,
            non_template_component_area=sum(c['area'] for c in components if c['id'] not in {t['component_id'] for t in templates}),
            template_anchor_counts=[len(t['anchorids']) for t in templates],prepare_seconds=time.perf_counter()-start,
            high_coverage_cut=.9,ring_width_native_cells=1,outer_ring_sampling='mean of max(0,anchor-response mean) over valid transformed ring points'))
    return _CACHE,False


def _pose_candidates(ep,t,tid,variant='main'):
    h,w=ep.q_hw;yy,xx=np.indices(ep.q_hw);centers=np.c_[yy.ravel(),xx.ravel()]
    variants=[]
    scales=(1.,) if variant=='translation_only' else (.5,.75,1.,1.5,2.)
    mirrors=(1.,) if variant=='translation_only' else (1.,-1.)
    angles=(0.,) if variant=='translation_only' else (-30.,0.,30.)
    maps=t['v'];offset=t['offset'].copy()
    if variant=='anchor_position_permutation':offset=offset[np.random.default_rng(0).permutation(len(offset))]
    if variant=='mean_feature':maps=np.broadcast_to(t['mean_feature'],maps.shape)
    outermap=maps.mean(0)
    for scale in scales:
        for mirror in mirrors:
            for angle in angles:
                rad=np.deg2rad(angle);rotation=np.array([[np.cos(rad),np.sin(rad)],[-np.sin(rad),np.cos(rad)]])
                matrix=scale*(rotation@np.diag([1.,mirror]));mapped=offset@matrix.T
                values=[];visible=[]
                for k,delta in enumerate(mapped):
                    y=centers[:,0]+delta[0];x=centers[:,1]+delta[1];v,keep=_sample(maps[k],y,x)
                    physical,_=_sample(ep.q_valid.reshape(ep.q_hw),y,x);keep&=physical>0
                    values.append(v);visible.append(keep)
                values=np.array(values);visible=np.array(visible)
                if variant=='no_common_pose':
                    # Same anchor response bank, but independently find each
                    # witness anywhere within its transformed candidate box.
                    sizey=max(1,int(np.ceil((t['bbox'][1]-t['bbox'][0])*scale)))
                    sizex=max(1,int(np.ceil((t['bbox'][3]-t['bbox'][2])*scale)))
                    values=np.array([ndimage.maximum_filter(v,size=(sizey,sizex),mode='nearest').ravel() for v in maps])
                score=_trim80(values,visible)
                ring=t['ringoffset']@matrix.T;penalty=np.zeros(len(centers));seen=np.zeros(len(centers))
                for delta in ring:
                    v,keep=_sample(outermap,centers[:,0]+delta[0],centers[:,1]+delta[1])
                    penalty+=np.maximum(v,0)*keep;seen+=keep
                score-=.25*penalty/np.maximum(seen,1)
                good=(visible.sum(0)>=.75*len(mapped))&(ep.q_valid>0)
                peak=score.reshape(ep.q_hw)==ndimage.maximum_filter(score.reshape(ep.q_hw),size=3,mode='nearest')
                halfpositive=((values>0)&visible).sum(0)>=.5*len(mapped)
                corners=np.array([[t['bbox'][0],t['bbox'][2]],[t['bbox'][0],t['bbox'][3]],
                    [t['bbox'][1],t['bbox'][2]],[t['bbox'][1],t['bbox'][3]]])@matrix.T
                bb=np.array([corners[:,0].min(),corners[:,0].max(),corners[:,1].min(),corners[:,1].max()])
                # Main uses local peaks. Matched-count controls may need valid
                # non-peaks to reproduce main's cost budget after simplifying
                # the pose family; this finite fill is explicitly recorded.
                eligibility=good&peak.ravel() if variant=='main' else good
                for index in np.flatnonzero(eligibility):
                    variants.append(dict(template=tid,center=centers[index],matrix=matrix,score=float(score[index]),
                        main_gate=bool(score[index]>=1 and halfpositive[index]),bbox=bb+centers[index][[0,0,1,1]],center_id=int(index)))
    return variants


def _select(candidates,count=None):
    ordered=sorted(candidates,key=lambda p:(-p['score'],p['template'],p['center_id'],tuple(p['matrix'].ravel())))
    accepted=[];rejected=[]
    for pose in ordered:
        if count is None and not pose['main_gate']:continue
        if any(p['template']==pose['template'] and _iou(p['bbox'],pose['bbox'])>.5 for p in accepted):
            rejected.append(pose);continue
        accepted.append(pose)
        if len(accepted)>=(16 if count is None else count):break
    if count is not None and len(accepted)<count:
        accepted.extend(rejected[:count-len(accepted)])
    return accepted


def _local_decode(ep,state,poses):
    h,w=ep.q_hw;yy,xx=np.indices(ep.q_hw);coordinates=np.c_[yy.ravel(),xx.ravel()]
    merged=np.full(len(ep.q),-np.inf);covered=np.zeros(len(ep.q),bool);records=[]
    for pose in poses:
        t=state['templates'][pose['template']];locations=t['offset']@pose['matrix'].T+pose['center']
        fgids=[]
        for k,point in enumerate(locations):
            cy,cx=np.rint(point).astype(int);cells=np.array([y*w+x for y in range(max(0,cy-1),min(h,cy+2)) for x in range(max(0,cx-1),min(w,cx+2)) if ep.q_valid[y*w+x]>0],int)
            if len(cells):fgids.append(int(cells[np.argmax(t['v'][k].ravel()[cells])]))
        fgids=np.unique(fgids)
        bb=pose['bbox'];center=np.array([(bb[0]+bb[1])/2,(bb[2]+bb[3])/2]);half=np.array([(bb[1]-bb[0])/2,(bb[3]-bb[2])/2])*1.25
        box=np.r_[center[0]-half[0],center[0]+half[0],center[1]-half[1],center[1]+half[1]]
        inside=(coordinates[:,0]>=box[0])&(coordinates[:,0]<=box[1])&(coordinates[:,1]>=box[2])&(coordinates[:,1]<=box[3])&(ep.q_valid>0)
        expanded=(coordinates[:,0]>=box[0]-1)&(coordinates[:,0]<=box[1]+1)&(coordinates[:,1]>=box[2]-1)&(coordinates[:,1]<=box[3]+1)&(ep.q_valid>0)
        ring=np.flatnonzero(expanded&~inside);order=np.lexsort((ring,state['sB'][ring]));bgids=ring[order[:16]]
        if not len(fgids) or not len(bgids) or not inside.any():
            records.append(dict(disabled='missing_local_FG_BG_or_ROI'));continue
        f,b=ep.q[fgids],ep.q[bgids];ids=np.flatnonzero(inside)
        local=np.max(ep.q[ids]@f.T,axis=1)-np.max(ep.q[ids]@b.T,axis=1)
        reference=np.max(ep.r@f.T,axis=1)-np.max(ep.r@b.T,axis=1)
        mean=np.sum(reference*ep.wvalid)/ep.wvalid.sum();sigma=max(float(np.sqrt(np.sum(ep.wvalid*(reference-mean)**2)/ep.wvalid.sum())),.05)
        value=.5*(local/sigma+state['sB'][ids]/state['sigmaB'])
        merged[ids]=np.maximum(merged[ids],value);covered[ids]=True
        records.append(dict(template=t['component_id'],center=pose['center'].tolist(),score=pose['score'],
            FG_dictionary=len(fgids),BG_dictionary=len(bgids),sigma_local=sigma,ROI_points=len(ids)))
    if not covered.any():return state['sB'].copy(),dict(inactive=True,inactive_reason='all accepted poses lack numerical local decoder',poses=records)
    out=np.where(state['sB']>.5,state['sB'],-1.);out[covered]=merged[covered];out[ep.q_valid<=0]=-1.
    return out,dict(inactive=False,accepted_pose_count=len(poses),local_decoder_records=records,covered_query_points=int(covered.sum()))


def hough(ep,mode='main'):
    validate(ep);start=time.perf_counter();mid='PRO30_M05' if mode=='main' else 'PRO30_M05__'+mode
    degeneration=degenerate_margin(ep)
    if degeneration is not None:return finish(ep,degeneration[0],mid,degeneration[1])
    state,hit=_templates(ep);templates=state['templates']
    candidates=[p for j,t in enumerate(templates) for p in _pose_candidates(ep,t,j)]
    accepted=_select(candidates)
    if not accepted:
        return finish(ep,state['sB'],mid,dict(state['info'],**source_contract(5),inactive=True,inactive_reason='no_geometrically_accepted_pose',
            component_cache_hit=hit,new_encoder_forwards=0,postprocess_seconds=time.perf_counter()-start))
    if mode=='B_R_seed_same_count':
        order=np.lexsort((np.arange(len(ep.q)),-state['sB']));order=order[ep.q_valid[order]>0]
        poses=[]
        for k,p in enumerate(accepted):
            copy=dict(p);center=np.array(divmod(int(order[k%len(order)]),ep.q_hw[1]));shift=center-p['center']
            copy.update(center=center,bbox=p['bbox']+shift[[0,0,1,1]],center_id=int(order[k%len(order)]));poses.append(copy)
    elif mode!='main':
        alternatives=[p for j,t in enumerate(templates) for p in _pose_candidates(ep,t,j,mode)]
        poses=_select(alternatives,len(accepted))
    else:poses=accepted
    margin,decode=_local_decode(ep,state,poses)
    info=dict(source_contract(5),**state['info'],**decode,mode=mode,
        main_accepted_pose_count=len(accepted),control_pose_count=len(poses),component_cache_hit=hit,
        pose_scales=[.5,.75,1.,1.5,2.],pose_mirrors=2,pose_angles=[-30,0,30],query_center_count=int(np.prod(ep.q_hw)),
        candidate_local_peak_count=len(candidates),robust_response='peranchor queryvalid median/IQR floor.05',
        pose_box_convention='axis-aligned transformed reference box; width/height expanded factor1.25',
        postprocess_seconds=time.perf_counter()-start,new_encoder_forwards=0,quality='unmeasured geometry hypothesis')
    return finish(ep,margin,mid,info)


def install(methods,controls,requirements,contracts):
    methods['PRO30_M05']=hough
    for mode in ('no_common_pose','mean_feature','anchor_position_permutation','translation_only','B_R_seed_same_count'):
        controls['PRO30_M05__'+mode]=partial(hough,mode=mode)
    requirements['PRO30_M05']=['actual complete original MR','exact R geometry','native unit R/Q','physicalvalidQ']
    contracts['PRO30_M05']=dict(source_contract(5),input_contract='N',host='B_R',renderer='CPU100 two-threshold',
        constants={'templates_max':4,'spatial_highcoverage_anchors_max':16,'anchors_min':4,'scales':[.5,.75,1,1.5,2],
            'angles':[-30,0,30],'mirrors':2,'IQR_min':.05,'trim_fraction':.8,'outer_penalty':.25,'pose_score_min':1.,
            'visible_fraction_min':.75,'positive_anchor_fraction_min':.5,'NMS_template_IoU':.5,'poses_max':16,
            'box_dimension_expansion':1.25,'local_FG_search':3,'local_BG_max':16,'sigma_min':.05,'margin_mix':.5,'outside_B_R_positive_cut':.5},
        controls=[c for c in controls if c.startswith('PRO30_M05__')]+['PRO30_B_R'],
        implementation_assumption='highcoverage.9; original-pixel component footprint shares actualnativeWF; 1nativecell source/query rings; fractional exact80%trim; axis-alignedNMSbox; expansion dimensionfactor1.25; controls fixedmain acceptedcount using valid nonpeak centers and NMS-rejected fill if needed')
