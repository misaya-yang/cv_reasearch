"""Pro30 M16 complete original-pixel finite topology/contour decoder.

Uses an explicit Zhang--Suen skeleton. 'Longest skeleton path' is implemented
as maximum finite geodesic shortest-path distance, exact and streamed for
cyclic skeletons; this convention is a disclosed source ambiguity.
"""
from collections import OrderedDict
import numpy as np
from scipy import ndimage,sparse
from scipy.sparse.csgraph import connected_components,dijkstra
from .common import artifact,ArtifactUnavailable,array_hash,continuous_original
from .structures_09_16 import host_B,cached
_REF_STATS=OrderedDict()
EIGHT=np.ones((3,3),bool);FOUR=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)


def skeleton(mask):
    x=np.pad(np.asarray(mask,bool),1);changed=True
    while changed:
        changed=False
        for phase in(0,1):
            p=[x[:-2,1:-1],x[:-2,2:],x[1:-1,2:],x[2:,2:],x[2:,1:-1],x[2:,:-2],x[1:-1,:-2],x[:-2,:-2]]
            degree=sum(a.astype(np.uint8)for a in p);transitions=sum((~p[k]&p[(k+1)%8]).astype(np.uint8)for k in range(8))
            condition=(~(p[0]&p[2]&p[4])&~(p[2]&p[4]&p[6]))if phase==0 else(~(p[0]&p[2]&p[6])&~(p[0]&p[4]&p[6]))
            remove=x[1:-1,1:-1]&(degree>=2)&(degree<=6)&(transitions==1)&condition
            if remove.any():x[1:-1,1:-1][remove]=False;changed=True
    return x[1:-1,1:-1]


def _longest_geodesic(skel):
    ids=np.flatnonzero(skel.ravel());N=len(ids)
    if N<2:return 0.
    index=np.full(skel.size,-1,int);index[ids]=np.arange(N);index=index.reshape(skel.shape);i=[];j=[];w=[]
    for dy,dx in((0,1),(1,-1),(1,0),(1,1)):
        y0=max(0,-dy);y1=min(skel.shape[0],skel.shape[0]-dy);x0=max(0,-dx);x1=min(skel.shape[1],skel.shape[1]-dx)
        left=index[y0:y1,x0:x1].ravel();right=index[y0+dy:y1+dy,x0+dx:x1+dx].ravel();valid=(left>=0)&(right>=0);i.extend(left[valid]);j.extend(right[valid]);w.extend([np.hypot(dy,dx)]*valid.sum())
    W=sparse.csr_matrix((np.r_[w,w],(np.r_[i,j],np.r_[j,i])),shape=(N,N));count,label=connected_components(W,directed=False);diameter=0.
    for component in range(count):
        points=np.flatnonzero(label==component);graph=W[points][:,points]
        if graph.nnz//2==len(points)-1:
            far=int(np.argmax(dijkstra(graph,directed=False,indices=0)));value=float(dijkstra(graph,directed=False,indices=far).max());diameter=max(diameter,value)
        else:
            for start in range(0,len(points),32):
                distances=dijkstra(graph,directed=False,indices=np.arange(start,min(start+32,len(points))));finite=distances[np.isfinite(distances)];diameter=max(diameter,float(finite.max())if len(finite)else 0.)
    return diameter


def statistics(mask):
    mask=np.asarray(mask,bool);area=int(mask.sum())
    if not area:return np.zeros(4)
    bg,count=ndimage.label(~np.pad(mask,1),FOUR);outside=bg[0,0];holes=sum(k!=outside for k in range(1,count+1))
    skel=skeleton(mask);degree=ndimage.convolve(skel.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-skel
    endpoint=int(np.sum(skel&(degree==1)));_,branches=ndimage.label(skel&(degree>=3),EIGHT);length=_longest_geodesic(skel)
    return np.array([holes,endpoint,branches,length/np.sqrt(area)],float)


def reference_statistics(mask):
    key=array_hash(mask)
    if key in _REF_STATS:
        value=_REF_STATS.pop(key);_REF_STATS[key]=value;return value
    labels,count=ndimage.label(mask,EIGHT);records=[]
    for component in range(1,count+1):
        yy,xx=np.nonzero(labels==component);y0,y1=yy.min(),yy.max()+1;x0,x1=xx.min(),xx.max()+1;C=labels[y0:y1,x0:x1]==component
        distance=ndimage.distance_transform_edt(np.pad(C,1))[1:-1,1:-1]
        observations=np.array([statistics(C if radius==0 else distance>radius)for radius in(0,1,2,4)])
        stable=np.r_[np.all(observations[:,:3]==observations[0,:3],axis=0),True]
        normalizer=np.maximum(np.mean(observations,axis=0)+1,1.)
        records.append({'component':component,'values':observations,'stable_dimensions':stable,'normalizer':normalizer,'pixel_bbox':[int(y0),int(x0),int(y1),int(x1)]})
    _REF_STATS[key]=records
    while len(_REF_STATS)>16:_REF_STATS.popitem(last=False)
    return records


def _dilate_token(mask,spacing):
    if not mask.any():return np.zeros_like(mask,bool)
    return ndimage.distance_transform_edt(~mask,sampling=spacing)<=1

def _erode_token(mask,spacing):return ndimage.distance_transform_edt(np.pad(mask,1),sampling=spacing)[1:-1,1:-1]>1


def execute(ep,control=None):
    if ep.reference_mask is None or ep.q_rgb is None:raise ArtifactUnavailable('M16 requires exact original reference mask and query RGB, no64-grid topology substitute')
    h,host=host_B(ep);original=continuous_original(ep,h.reshape(ep.q_hw));rgb=np.asarray(ep.q_rgb,float)
    if rgb.shape[:2]!=tuple(ep.original_shape):raise ValueError('M16 RGB and original query geometry differ')
    records=reference_statistics(np.asarray(ep.reference_mask,bool))
    luminance=rgb.mean(axis=2);gradient=np.hypot(ndimage.sobel(luminance,0,mode='reflect'),ndimage.sobel(luminance,1,mode='reflect'));den=float(np.quantile(gradient,.95));g=np.clip(gradient/max(den,1e-12),0,1)
    geometry=ep.query_geometry
    physical_tokens=np.array(geometry['resized_hw'],float)/float(geometry['view_side'])*np.array(ep.q_hw)if geometry else np.array(ep.q_hw,float)
    pixel_per_token=np.array(ep.original_shape,float)/physical_tokens;spacing=1/pixel_per_token
    positive,count=ndimage.label(original>0,EIGHT);weak,wcount=ndimage.label(original>-.1,EIGHT);windows=[]
    for component in range(1,count+1):
        support=positive==component;expanded=_dilate_token(support,spacing)&(np.abs(original)<=.25);expanded|=support
        y,x=np.nonzero(expanded);windows.append((int(y.min()),int(x.min()),int(y.max()+1),int(x.max()+1)))
    for component in range(1,wcount+1):
        support=weak==component
        if np.any(support&(positive>0)):continue
        y,x=np.nonzero(support)
        if len(y):windows.append((int(y.min()),int(x.min()),int(y.max()+1),int(x.max()+1)))
    windows=list(dict.fromkeys(windows));candidates=[];audit=[]
    for window_id,(y0,x0,y1,x1)in enumerate(windows):
        field=original[y0:y1,x0:x1];edge=g[y0:y1,x0:x1];uncertain=np.abs(field)<=.25;empty_cost=float(field.sum());clip=.1*float(np.abs(field).sum())
        for threshold in(-.1,0.,.1):
            base=field>threshold
            masks=[base,_erode_token(_dilate_token(base,spacing),spacing),_dilate_token(_erode_token(base,spacing),spacing)]
            for operation,proposal in enumerate(masks):
                proposal=np.where(uncertain,proposal,base);area=int(proposal.sum());T=statistics(proposal)
                unary=float(-np.sum(field*(2*proposal-1)));boundary=proposal&~ndimage.binary_erosion(proposal,FOUR,border_value=0);rgb_cost=.05*float(np.sum((1-edge)[boundary]))
                topology=0.
                if area and records:
                    distances=[]
                    for record in records:
                        difference=np.minimum(np.abs(T[None]-record['values'])/record['normalizer'][None],1.)*record['stable_dimensions'][None]
                        distances.append(float(difference.sum(axis=1).min()))
                    topology=min(.1*area*min(distances),clip)
                if control=='unary_only':topology=0.;rgb_cost=0.
                elif control=='RGB_only':topology=0.
                elif control=='fixed_no_holes'and area:topology=min(.1*area*min(T[0],1),clip)
                cost=unary+topology+rgb_cost;delta=cost-empty_cost
                row={'window':window_id,'bbox':[y0,x0,y1,x1],'threshold':threshold,'operation':['original','close_radius1token','open_radius1token'][operation],'area':area,'topology_statistics':T.tolist(),'unary':unary,'topology':topology,'RGB_boundary':rgb_cost,'empty_cost_same_window':empty_cost,'delta':delta}
                index=len(audit);audit.append(row)
                if delta<0 and area:
                    yy,xx=np.nonzero(proposal);ids=(yy+y0)*ep.original_shape[1]+xx+x0
                    candidates.append({'audit_id':index,'window':window_id,'ids':np.sort(ids),'delta':delta})
    def conflict(A,B):return A['window']==B['window']or bool(np.intersect1d(A['ids'],B['ids'],assume_unique=True).size)
    if len(windows)<=16:
        states=[(0.,())]
        for window in range(len(windows)):
            options=[i for i,c in enumerate(candidates)if c['window']==window];expanded=states.copy()
            for cost,selected in states:
                for i in options:
                    if any(conflict(candidates[i],candidates[j])for j in selected):continue
                    expanded.append((cost+candidates[i]['delta'],selected+(i,)))
            expanded.sort(key=lambda s:(s[0],s[1]));states=expanded[:32]
        value,selected=states[0]
    else:
        selected=[]
        for i in sorted(range(len(candidates)),key=lambda i:(candidates[i]['delta'],candidates[i]['audit_id'])):
            if not any(conflict(candidates[i],candidates[j])for j in selected):selected.append(i)
        value=sum(candidates[i]['delta']for i in selected)
    mask=np.zeros(ep.original_shape,bool)
    for i in selected:mask.ravel()[candidates[i]['ids']]=True
    metadata={'host':host,'reference_stable_statistics':[dict(record,values=record['values'].tolist(),stable_dimensions=record['stable_dimensions'].tolist(),normalizer=record['normalizer'].tolist())for record in records],'windows':windows,'all9_candidates_each_window':audit,'selected_candidates':[candidates[i]['audit_id']for i in selected],'sum_delta_cost':float(value),'topology_resolution':'exact_original_pixels','skeleton_longest_path_definition':'maximum finite geodesic shortest-path distance, exact streamed cyclic graph diameter','control':control}
    return 2*mask.astype(float)-1,metadata
