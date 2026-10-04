"""Physical-point observations from actual RGB crop re-encoding.

No encoder is implemented here. The caller must encode ``masked_rgb`` and/or
``unmasked_rgb`` as declared by its arm, and return real 16x16 descriptors/CLS.
The whole 1024 image's descriptors/CLS come from its original forward. No old
token pooling is substituted for a regional observation; ``old_pool_bank`` is
an explicitly separate control. Query truth is not an input.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
import sys
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WardTree:
    grid_hw: tuple[int, int]
    children: np.ndarray
    merge_costs: np.ndarray
    minimum_leaf: np.ndarray

    @property
    def n_leaves(self) -> int:
        return self.grid_hw[0] * self.grid_hw[1]

    def cut(self, k: int) -> np.ndarray:
        """One stored tree, stable labels ordered by each region's first cell."""
        n = self.n_leaves
        if not 1 <= int(k) <= n or int(k) != k:
            raise ValueError("Ward cut must be an integer between1 and cell count")
        parent = np.arange(n, dtype=np.int64)
        representatives = np.empty(2*n-1, dtype=np.int64)
        representatives[:n] = np.arange(n, dtype=np.int64)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x
        for index, (left, right) in enumerate(self.children[:n-int(k)]):
            a, b = find(int(representatives[left])), find(int(representatives[right]))
            lo, hi = min(a,b), max(a,b)
            parent[hi] = lo
            representatives[n+index] = lo
        roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
        unique = np.unique(roots)
        return np.searchsorted(unique, roots).reshape(self.grid_hw).astype(np.int64)


def build_spatial_ward_tree(features64: Any, grid_hw=(64,64)) -> WardTree:
    """Exact Ward SSE merge, constrained by4-neighbour region adjacency.

    Caller supplies native raw whole-image features; this function neither
    normalises nor debiases them. Float64 costs, stable minimum-leaf tie breaks,
    and one complete tree serve both16- and64-region cuts. No NxN distances.
    """
    values = np.asarray(features64)
    if values.ndim == 3:
        grid_hw = tuple(int(v) for v in values.shape[:2])
        values = values.reshape(-1, values.shape[-1])
    else:
        grid_hw = tuple(int(v) for v in grid_hw)
    n = grid_hw[0]*grid_hw[1]
    if values.ndim != 2 or values.shape[0] != n or values.shape[1] < 1 or min(grid_hw)<1:
        raise ValueError("Ward expects[H,W,D] or[H*W,D] with explicit grid")
    if not np.isfinite(values).all():
        raise ValueError("Ward input must be finite")
    means = np.empty((2*n-1,values.shape[1]), dtype=np.float64)
    means[:n] = values
    sizes = np.ones(2*n-1, dtype=np.int64)
    minima = np.empty(2*n-1, dtype=np.int64); minima[:n] = np.arange(n)
    active = np.zeros(2*n-1, dtype=bool); active[:n] = True
    adjacency = {i:set() for i in range(n)}
    height, width = grid_hw
    for y in range(height):
        for x in range(width):
            a = y*width+x
            for b in ((a+1,) if x+1<width else ()) + ((a+width,) if y+1<height else ()):
                adjacency[a].add(b); adjacency[b].add(a)
    heap = []
    def push(a,b):
        if minima[a]>minima[b]: a,b=b,a
        delta = means[a]-means[b]
        cost = float((sizes[a]*sizes[b]/(sizes[a]+sizes[b]))*np.dot(delta,delta))
        if not np.isfinite(cost): raise ValueError("Ward cost overflow")
        heapq.heappush(heap,(cost,int(minima[a]),int(minima[b]),a,b))
    for a in range(n):
        for b in sorted(adjacency[a]):
            if a<b: push(a,b)
    children = np.empty((max(0,n-1),2), dtype=np.int64)
    costs = np.empty(max(0,n-1), dtype=np.float64)
    for index in range(n-1):
        while heap:
            cost,_,_,a,b = heapq.heappop(heap)
            if active[a] and active[b] and b in adjacency[a]: break
        else:
            raise RuntimeError("Connected Ward graph exhausted prematurely")
        node = n+index
        children[index] = [a,b]; costs[index] = cost
        sizes[node] = sizes[a]+sizes[b]
        means[node] = means[a]+(means[b]-means[a])*(sizes[b]/sizes[node])
        minima[node] = min(minima[a],minima[b]); active[node] = True
        neighbours = (adjacency[a] | adjacency[b])-{a,b}
        active[a] = active[b] = False
        for other in neighbours:
            adjacency[other].discard(a); adjacency[other].discard(b)
            adjacency[other].add(node)
        del adjacency[a],adjacency[b]
        adjacency[node] = neighbours
        for other in sorted(neighbours,key=lambda j:int(minima[j])): push(node,other)
    return WardTree(grid_hw,children,costs,minima)


@dataclass(frozen=True)
class CropGeometry:
    template_id: int
    kind: str
    label: int | None
    bbox_xyxy: tuple[int,int,int,int]
    resized_hw: tuple[int,int]
    padding_ltrb: tuple[int,int,int,int]
    scale_xy: tuple[float,float]
    retained_pixels: int
    bbox_pixels: int
    empty: bool

    def to_crop_xy(self, points_xy):
        points = np.asarray(points_xy,dtype=np.float64)
        x0,y0,_,_ = self.bbox_xyxy; left,top,_,_ = self.padding_ltrb
        return (points-np.asarray([x0,y0]))*np.asarray(self.scale_xy)+np.asarray([left,top])

    def to_work_xy(self, points_xy):
        points = np.asarray(points_xy,dtype=np.float64)
        x0,y0,_,_ = self.bbox_xyxy; left,top,_,_ = self.padding_ltrb
        return (points-np.asarray([left,top]))/np.asarray(self.scale_xy)+np.asarray([x0,y0])


@dataclass(frozen=True)
class ObservationViews:
    masked_rgb: Any                  # Torch[V,3,256,256], genuine image pixels
    unmasked_rgb: Any                # Same bounding box/scale/padding, includes context
    pure_mask: Any                   # Torch[V,1,256,256], no RGB copied into this field
    maskshape: tuple[dict,...]
    geometries: tuple[CropGeometry,...]
    labels16: np.ndarray
    labels64: np.ndarray
    region_values16: tuple[int,...]
    region_values64: tuple[int,...]
    source_mask: np.ndarray | None
    points_xy: np.ndarray            # Pixel-centre coordinates in original1024 work frame
    cell_ids: np.ndarray
    roles: np.ndarray | None         # TrueFG/FalseBG, Source only
    area: np.ndarray                 # Actual role pixels/cell; Query cell area256
    area_weight: np.ndarray          # Source per-role normalisation; Query sum1
    coverage: np.ndarray
    template_ids: np.ndarray         # Source[4,A], Query[3,4096]
    point_valid: np.ndarray
    metadata: dict

    @property
    def num_views(self): return len(self.geometries)

    @property
    def num_templates(self): return self.num_views+1

    def rgb(self, variant="masked"):
        if variant=="masked": return self.masked_rgb
        if variant=="unmasked": return self.unmasked_rgb
        raise ValueError("Only registered masked/unmasked RGB variants")


def _source_points(mask: np.ndarray):
    """Every cell gets its nearest actual pixel for each nonempty role."""
    points=[]; cells=[]; roles=[]; area=[]; coverage=[]
    totals={True:int(mask.sum()),False:int(mask.size-mask.sum())}
    for y in range(64):
        for x in range(64):
            tile=mask[y*16:(y+1)*16,x*16:(x+1)*16]
            for role in (True,False):
                yy,xx=np.nonzero(tile if role else ~tile)
                count=len(xx)
                if count==0: continue
                distance=(xx+.5-8.)**2+(yy+.5-8.)**2
                i=int(np.argmin(distance)) # np.nonzero row order resolves equal distance
                points.append((x*16+int(xx[i])+.5,y*16+int(yy[i])+.5))
                cells.append(y*64+x);roles.append(role);area.append(count);coverage.append(count/256.)
    roles=np.asarray(roles,dtype=bool);area=np.asarray(area,dtype=np.float64)
    weights=np.asarray([count/totals[bool(role)] for count,role in zip(area,roles)],dtype=np.float64)
    return (np.asarray(points,dtype=np.float64).reshape(-1,2),np.asarray(cells,dtype=np.int64),
            roles,area,weights,np.asarray(coverage,dtype=np.float64))


def _crop(image,mask,template_id,kind,label):
    import torch
    import torch.nn.functional as F
    yy,xx=np.nonzero(mask)
    empty=len(xx)==0
    x0,y0,x1,y1=(0,0,1024,1024) if empty else (int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
    height,width=y1-y0,x1-x0
    scale=256./max(height,width)
    rh=max(1,min(256,int(np.floor(height*scale+.5))))
    rw=max(1,min(256,int(np.floor(width*scale+.5))))
    left,top=(256-rw)//2,(256-rh)//2
    right,bottom=256-rw-left,256-rh-top
    keep=torch.as_tensor(mask[y0:y1,x0:x1],dtype=image.dtype,device=image.device)[None,None]
    raw=image[:,y0:y1,x0:x1][None]
    masked=F.interpolate(raw*keep,size=(rh,rw),mode="bilinear",align_corners=False)
    unmasked=F.interpolate(raw,size=(rh,rw),mode="bilinear",align_corners=False)
    pure=F.interpolate(keep,size=(rh,rw),mode="nearest")
    geometry=CropGeometry(template_id,kind,label,(x0,y0,x1,y1),(rh,rw),(left,top,right,bottom),
        (rw/width,rh/height),int(mask.sum()),height*width,empty)
    shape=dict(template_id=template_id,area_pixels=geometry.retained_pixels,bbox_xyxy=list(geometry.bbox_xyxy),
        bbox_area_pixels=height*width,occupancy=geometry.retained_pixels/(height*width),
        aspect_width_over_height=width/height,resized_hw=[rh,rw],padding_ltrb=[left,top,right,bottom],
        empty=empty,pure_mask_contains_no_RGB=True)
    return (F.pad(masked,(left,right,top,bottom),value=0)[0],
            F.pad(unmasked,(left,right,top,bottom),value=0)[0],
            F.pad(pure,(left,right,top,bottom),value=0)[0],geometry,shape)


def build_views(image_work, labels16, labels64, source_mask=None) -> ObservationViews:
    """80 real regional crops; Source additionally gets globalFG/globalBG.

    Outside each region is normalised RGB0 before resizing; letterbox padding
    is also0. The unmasked control uses exactly the same crop geometry.
    Pure-mask fields are binary shape controls, not a claimed RGB normalisation.
    """
    import torch
    if (not torch.is_tensor(image_work) or tuple(image_work.shape)!=(3,1024,1024) or
            not image_work.is_floating_point() or not torch.isfinite(image_work).all()):
        raise ValueError("Native normalised finite RGB tensor[3,1024,1024] required")
    labels=[];values=[]
    for supplied,count in ((labels16,16),(labels64,64)):
        array=np.asarray(supplied)
        if array.shape!=(64,64) or array.dtype.kind not in "iu" or len(np.unique(array))!=count:
            raise ValueError("Exactly16/64 nonempty regions on native64 grid required")
        labels.append(array.astype(np.int64,copy=True));values.append(tuple(int(v) for v in np.unique(array)))
    for value in values[1]:
        if len(np.unique(labels[0][labels[1]==value]))!=1:
            raise ValueError("Fine64 must refine coarse16 from one stored Ward tree")
    mask=None
    if source_mask is not None:
        mask=np.asarray(source_mask)
        if mask.dtype!=np.bool_ or mask.shape!=(1024,1024):
            raise ValueError("Source role mask must be actual bool[1024,1024]")
        mask=mask.copy()
    masked=[];unmasked=[];pure=[];geometries=[];shapes=[]
    for kind,grid,region_values,start in (("coarse16",labels[0],values[0],1),("fine64",labels[1],values[1],17)):
        for index,value in enumerate(region_values):
            role=np.repeat(np.repeat(grid==value,16,axis=0),16,axis=1)
            a,b,c,g,s=_crop(image_work,role,start+index,kind,value)
            masked.append(a);unmasked.append(b);pure.append(c);geometries.append(g);shapes.append(s)
    if mask is not None:
        for role,identifier,kind in ((mask,81,"SourceFG"),(~mask,82,"SourceBG")):
            a,b,c,g,s=_crop(image_work,role,identifier,kind,None)
            masked.append(a);unmasked.append(b);pure.append(c);geometries.append(g);shapes.append(s)
        points,cells,roles,area,weights,coverage=_source_points(mask)
    else:
        yy,xx=np.indices((64,64))
        points=np.stack([xx.reshape(-1)*16+8.,yy.reshape(-1)*16+8.],axis=1)
        cells=np.arange(4096,dtype=np.int64);roles=None
        area=np.full(4096,256.,dtype=np.float64);weights=np.full(4096,1./4096,dtype=np.float64)
        coverage=np.ones(4096,dtype=np.float64)
    lookup16={v:1+i for i,v in enumerate(values[0])};lookup64={v:17+i for i,v in enumerate(values[1])}
    coarse=np.asarray([lookup16[int(labels[0].reshape(-1)[i])] for i in cells],dtype=np.int64)
    fine=np.asarray([lookup64[int(labels[1].reshape(-1)[i])] for i in cells],dtype=np.int64)
    template_ids=np.stack([np.zeros(len(cells),dtype=np.int64),coarse,fine])
    if roles is not None: template_ids=np.concatenate([template_ids,np.where(roles,81,82)[None]],axis=0)
    valid=np.ones_like(template_ids,dtype=bool)
    for slot in range(1,len(template_ids)):
        for identifier in np.unique(template_ids[slot]):
            indices=np.flatnonzero(template_ids[slot]==identifier)
            geometry=geometries[int(identifier)-1]
            xy=geometry.to_crop_xy(points[indices]);left,top,right,bottom=geometry.padding_ltrb
            valid[slot,indices]=(~geometry.empty)&(xy[:,0]>=left)&(xy[:,0]<256-right)&(xy[:,1]>=top)&(xy[:,1]<256-bottom)
    if not valid.all(): raise RuntimeError("A physical point left its actual crop content rectangle")
    metadata=dict(work_hw=[1024,1024],dense_grid_hw=[64,64],crop_hw=[256,256],crop_grid_hw=[16,16],
        num_RGB_views=len(geometries),num_templates=len(geometries)+1,
        template_order="whole0/coarse1..16/fine17..80/SourceFG81/SourceBG82",
        observation_order=["whole","coarse16","fine64"]+(["own_Source_role"] if mask is not None else []),
        physical_coordinate_convention="pixel centres: Source integer+.5; Query16cell centres8+16i; xy order",
        resampling="RGB bilinear align_cornersFalse; pure-mask nearest; centred integer letterbox",
        masked_and_unmasked_geometry_identical=True,normalised_outside_region_and_padding=0,
        regional_encoder_called=False,old_tokens_not_used_for_crops=True,
        source_role_area_normalisation="actual count / all pixels in same Source role",
        query_GT_input=False)
    return ObservationViews(torch.stack(masked),torch.stack(unmasked),torch.stack(pure),tuple(shapes),tuple(geometries),
        labels[0],labels[1],values[0],values[1],mask,points,cells,roles,area,weights,coverage,template_ids,valid,metadata)


def _sample(tokens,xy,pixel_hw):
    import torch
    import torch.nn.functional as F
    xy=torch.as_tensor(xy,dtype=tokens.dtype,device=tokens.device)
    height,width=pixel_hw
    grid=torch.stack([xy[:,0]*(2./width)-1.,xy[:,1]*(2./height)-1.],dim=-1)[None,None]
    field=tokens.permute(2,0,1)[None]
    return F.grid_sample(field,grid,mode="bilinear",padding_mode="border",align_corners=False)[0,:,0].transpose(0,1)


def sample_descriptors(crop_tokens, whole_tokens64, views: ObservationViews):
    """Real crop[16,16,D] sampled at the SAME work-frame physical points."""
    import torch
    if (not torch.is_tensor(crop_tokens) or not torch.is_tensor(whole_tokens64) or
            crop_tokens.ndim!=4 or tuple(crop_tokens.shape[:3])!=(views.num_views,16,16) or
            whole_tokens64.ndim!=3 or tuple(whole_tokens64.shape[:2])!=(64,64) or
            crop_tokens.shape[-1]!=whole_tokens64.shape[-1] or
            crop_tokens.dtype!=whole_tokens64.dtype or crop_tokens.device!=whole_tokens64.device):
        raise ValueError("Real crop[V,16,16,D] and whole[64,64,D] require identical dtype/device/channel")
    if not torch.isfinite(crop_tokens).all() or not torch.isfinite(whole_tokens64).all():
        raise ValueError("Descriptors must be finite")
    slots,anchors=views.template_ids.shape
    local=torch.empty((slots,anchors,whole_tokens64.shape[-1]),dtype=whole_tokens64.dtype,device=whole_tokens64.device)
    local[0]=_sample(whole_tokens64,views.points_xy,(1024,1024))
    for slot in range(1,slots):
        for identifier in np.unique(views.template_ids[slot]):
            indices=np.flatnonzero(views.template_ids[slot]==identifier)
            xy=views.geometries[int(identifier)-1].to_crop_xy(views.points_xy[indices])
            target=torch.as_tensor(indices,dtype=torch.long,device=local.device)
            local[slot,target]=_sample(crop_tokens[int(identifier)-1],xy,(256,256))
    return dict(local=local,raw=local[0],template_ids=views.template_ids.copy(),points_xy=views.points_xy.copy(),
        cell_ids=views.cell_ids.copy(),roles=None if views.roles is None else views.roles.copy(),area=views.area.copy(),
        area_weight=views.area_weight.copy(),coverage=views.coverage.copy(),valid=views.point_valid.copy(),
        metadata=dict(sampling="bilinear align_cornersFalse at registered physical point",channel_normalisation="none; caller owns it",
            actual_crop_descriptors_required=True,whole_slot_is_original_raw_dense=True,query_GT_used=False))


def build_global_bank(crop_cls, whole_cls, views: ObservationViews):
    """Actual CLS controls, indexed by the SAME IDs as local observations."""
    import torch
    if (crop_cls.ndim!=2 or crop_cls.shape[0]!=views.num_views or whole_cls.ndim!=1 or
            crop_cls.shape[1]!=whole_cls.shape[0] or crop_cls.device!=whole_cls.device or crop_cls.dtype!=whole_cls.dtype):
        raise ValueError("Actual regionalCLS[V,D] and wholeCLS[D] required")
    if not torch.isfinite(crop_cls).all() or not torch.isfinite(whole_cls).all(): raise ValueError("CLS must be finite")
    return torch.cat([whole_cls[None],crop_cls],dim=0)


def old_pool_bank(whole_tokens64, views: ObservationViews):
    """Original-image regional pool control; never labelled a crop re-encode."""
    import torch
    if whole_tokens64.ndim!=3 or tuple(whole_tokens64.shape[:2])!=(64,64):
        raise ValueError("Original raw whole64 tokens required")
    values=[whole_tokens64.mean(dim=(0,1))];valid=[True]
    for labels,region_values in ((views.labels16,views.region_values16),(views.labels64,views.region_values64)):
        for value in region_values:
            select=torch.as_tensor(labels==value,dtype=torch.bool,device=whole_tokens64.device)
            values.append(whole_tokens64[select].mean(dim=0));valid.append(True)
    if views.source_mask is not None:
        coverage=views.source_mask.reshape(64,16,64,16).sum(axis=(1,3)).astype(np.float64)
        for weights in (coverage,256.-coverage):
            tensor=torch.as_tensor(weights,dtype=whole_tokens64.dtype,device=whole_tokens64.device)
            if weights.sum()==0:
                values.append(torch.zeros_like(values[0]));valid.append(False)
            else:
                values.append((whole_tokens64*tensor[...,None]).sum(dim=(0,1))/tensor.sum());valid.append(True)
    result=torch.stack(values)
    if result.shape[0]!=views.num_templates: raise RuntimeError("Pool/CLS/local template contract differs")
    return dict(bank=result,valid=np.asarray(valid,dtype=bool),template_ids=np.arange(views.num_templates),
        metadata=dict(actual_original_tokens=True,regional_reencoding=False,channel_normalisation="none"))


def cpu_selfcheck():
    """Six bounded SERVER-only contracts; not pretrained quality evidence."""
    if not sys.platform.startswith("linux") or not str(Path(__file__).resolve()).startswith("/root/"):
        raise RuntimeError("Numerical fixtures are SERVER-only")
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized(): raise RuntimeError("CPU fixtures must not have CUDA")
    checks=[]
    tree=build_spatial_ward_tree(np.zeros((8,8,2),np.float32))
    a,b=tree.cut(16),tree.cut(64)
    assert len(np.unique(a))==16 and len(np.unique(b))==64 and (tree.merge_costs==0).all()
    assert np.array_equal(tree.cut(16),a)
    checks.append("one4adjacentWardtree_stable_ties_and_two_cuts")
    for value in np.unique(a):
        selected=set(np.flatnonzero(a.reshape(-1)==value).tolist());todo=[min(selected)];seen=set(todo)
        while todo:
            i=todo.pop();y,x=divmod(i,8)
            for j in ([i-1] if x else [])+([i+1] if x<7 else [])+([i-8] if y else [])+([i+8] if y<7 else []):
                if j in selected and j not in seen:seen.add(j);todo.append(j)
        assert seen==selected
    checks.append("Ward_regions_remain_spatially_connected")
    yy,xx=np.indices((64,64));labels16=(yy//16)*4+xx//16;labels64=(yy//8)*8+xx//8
    image=torch.ones((3,1024,1024),dtype=torch.float32)
    q=build_views(image,labels16,labels64)
    assert q.num_views==80 and q.num_templates==81 and q.template_ids.shape==(3,4096)
    assert np.all(q.points_xy==np.stack([xx.reshape(-1)*16+8.,yy.reshape(-1)*16+8.],axis=1))
    assert all(np.allclose(g.to_work_xy(g.to_crop_xy(q.points_xy[:5])),q.points_xy[:5]) for g in q.geometries)
    checks.append("Query80_realRGB_samebbox_controls_and_inverse_coordinates")
    source=np.zeros((1024,1024),dtype=bool);source[:,::2]=True
    s=build_views(image,labels16,labels64,source)
    assert s.num_views==82 and s.template_ids.shape==(4,8192) and s.area.sum()==1024**2
    pix=np.floor(s.points_xy).astype(np.int64)
    assert np.array_equal(source[pix[:,1],pix[:,0]],s.roles)
    assert np.allclose([s.area_weight[s.roles].sum(),s.area_weight[~s.roles].sum()],[1.,1.])
    assert torch.any(s.masked_rgb[-2]!=s.unmasked_rgb[-2]) and set(torch.unique(s.pure_mask).tolist())<={0.,1.}
    checks.append("Source_actualFG_BG_physical_pixels_weights_and_role_views")
    tokens=torch.arange(s.num_views,dtype=torch.float32)[:,None,None,None].expand(-1,16,16,2)
    whole=torch.zeros((64,64,2),dtype=torch.float32)
    observed=sample_descriptors(tokens,whole,s)
    assert observed["local"].shape==(4,8192,2)
    for slot in range(1,4):
        assert torch.allclose(observed["local"][slot,:,0],torch.as_tensor(s.template_ids[slot]-1,dtype=torch.float32),atol=1e-5)
    assert torch.equal(observed["raw"],observed["local"][0])
    checks.append("physical_point_sampler_four_slots_exact_template_mapping")
    cls=build_global_bank(torch.ones((82,2)),torch.zeros(2),s)
    pooled=old_pool_bank(whole,s)
    assert cls.shape==(83,2) and pooled["bank"].shape==(83,2) and pooled["valid"].all()
    assert not torch.cuda.is_initialized()
    checks.append("CLS_and_oldpool_use_same_template_IDs_without_encoder_substitution")
    return dict(state="SERVER_REGIONAL_OBSERVATION_CPU_CONTRACTS_PASSED",checks=len(checks),passed=checks,
        CUDA_initialized=False,encoder_calls=0,pretrained_model_tested=False,task_gain_measured=False)
