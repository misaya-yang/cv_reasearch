"""Float64 CPU reference for a fixed, partial-correspondence tree model.

No model, Query labels, training, top-k or image encoding. All Source foreground
states and all Query grid positions remain. Normalised node masses are posterior
probabilities of THIS specified potential model, not calibrated task confidence.
This is an independent equation implementation, not verified Pro source code.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Any

TEMPERATURE = .07
ETA = .5
RADIUS = 4.
GEODESIC_CUTOFF = 12
RHO = .1
UNIT_TOLERANCE = 2e-6


@dataclass(frozen=True)
class RelationKernel:
    sparseK: Any                 # scipy CSR, symmetric, zero diagonal
    scale: Any                   # float64[M], D's diagonal
    eta: float
    m: int
    coords: Any                  # every Source FG coordinate, row-major
    degenerate: bool
    metadata: dict

    def matvec(self, vector):
        import numpy as np
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != (self.m,) or not np.isfinite(value).all():
            raise ValueError("R@v requires a finite vector for ALL Source states")
        if self.m == 0: return value.copy()
        if self.degenerate: return np.full(self.m, value.sum(), dtype=np.float64)
        weighted = self.scale * value
        # A=.5K+.5(11^T-I). No dense MxM edge/relation matrix.
        return value + self.scale * (self.eta * (self.sparseK @ weighted) +
            (1-self.eta) * (weighted.sum() - weighted))


@dataclass(frozen=True)
class QueryTree:
    shape: tuple[int, int]
    parent: Any                  # int64[N], -1 at each root
    children: tuple[tuple[int, ...], ...]
    levels: tuple[Any, ...]       # int64 arrays, breadth-first depths
    edgew: Any                   # float64[N], weight to parent, root=0
    roots: tuple[int, ...]
    preorder: tuple[int, ...]
    postorder: tuple[int, ...]
    metadata: dict


def _hard_source(mask):
    import numpy as np
    value = np.asarray(mask)
    if value.ndim != 2 or min(value.shape) <= 0:
        raise ValueError("Nonempty original Source patch grid required")
    if value.dtype == np.bool_: return value
    if value.dtype == np.uint8 and np.all((value == 0) | (value == 1)):
        return value.astype(bool)
    raise ValueError("Source role must be an explicit bool/binary-uint8 hard grid")


def _unit_features(features, count, label):
    import numpy as np
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != count or value.shape[1] < 1 or not np.isfinite(value).all():
        raise ValueError(label + " requires all finite unit feature rows")
    if count and np.max(np.abs(np.linalg.norm(value,axis=1)-1)) > UNIT_TOLERANCE:
        raise ValueError(label + " must be caller-normalised unit features; no hidden normalisation")
    return value


def semantic_correspondence(source_features, query_features, source_mask):
    """pi=softmax(q@s/.07), retaining every legal Source FG patch.

    Source features may be grid HxWxC or full HWxC; Query features HWxC.
    The dense NxM probability array is required, not a top-k approximation.
    """
    import numpy as np
    mask = _hard_source(source_mask)
    source = np.asarray(source_features,dtype=np.float64)
    if source.ndim == 3 and source.shape[:2] == mask.shape:
        source = source.reshape(mask.size,source.shape[-1])
    source = _unit_features(source,mask.size,"Source semantic features")
    query = np.asarray(query_features,dtype=np.float64)
    query = _unit_features(query,query.shape[0] if query.ndim==2 else -1,"Query semantic features")
    if source.shape[1] != query.shape[1]: raise ValueError("Source/Query semantic feature channels differ")
    foreground = source[mask.reshape(-1)]
    if not len(foreground): return np.empty((len(query),0),dtype=np.float64)
    probability = query @ foreground.T
    probability /= TEMPERATURE
    probability -= probability.max(axis=1,keepdims=True)
    np.exp(probability,out=probability)
    probability /= probability.sum(axis=1,keepdims=True)
    return probability


def build_relation(source_mask):
    """Four-neighbour FG-only geodesics; sparse radius12 K plus rank1 floor.

    This implementation freezes eta=.5 and A_ab=.5K_ab+.5 for a!=b.
    Positive diagonal scaling balances EVERY off-diagonal row to M-1.
    M<=3 uses the exact all-ones R degeneracy; M=0 explicitly abstains.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    mask = _hard_source(source_mask); height,width = mask.shape
    coords = np.argwhere(mask); m = len(coords)
    index = np.full(mask.shape,-1,dtype=np.int64)
    index[mask] = np.arange(m,dtype=np.int64)
    neighbours=[]
    for y,x in coords:
        adjacent=[]
        for dy,dx in ((-1,0),(0,-1),(0,1),(1,0)):
            yy,xx=int(y)+dy,int(x)+dx
            if 0<=yy<height and 0<=xx<width and index[yy,xx]>=0:
                adjacent.append(int(index[yy,xx]))
        neighbours.append(tuple(adjacent))
    indptr=[0];indices=[];values=[]
    seen=np.zeros(m,dtype=np.int64);distance=np.empty(m,dtype=np.int16)
    for start in range(m):
        generation=start+1;seen[start]=generation;distance[start]=0
        queue=deque([start]);row=[]
        while queue:
            node=queue.popleft();depth=int(distance[node])
            if depth==GEODESIC_CUTOFF:continue
            for other in neighbours[node]:
                if seen[other]==generation:continue
                seen[other]=generation;distance[other]=depth+1;queue.append(other)
                row.append((other,depth+1))
        row.sort()
        for other,steps in row:
            indices.append(other);values.append(float(np.exp(-steps*steps/(2*RADIUS*RADIUS))))
        indptr.append(len(indices))
    kernel=csr_matrix((np.asarray(values,np.float64),np.asarray(indices,np.int32),np.asarray(indptr,np.int64)),shape=(m,m))
    scale=np.ones(m,dtype=np.float64);iterations=0;residual=0.
    if m>=4:
        target=float(m-1)
        for iterations in range(513):
            action=ETA*(kernel@scale)+(1-ETA)*(scale.sum()-scale)
            row_sums=scale*action
            residual=float(np.max(np.abs(row_sums-target))/target)
            if residual<=1e-12:break
            if iterations==512 or not np.isfinite(row_sums).all() or (row_sums<=0).any():
                raise RuntimeError("Symmetric Source relation row balancing failed; no fallback/top-k")
            scale*=np.sqrt(target/row_sums)
    for value in (coords,scale,kernel.data,kernel.indices,kernel.indptr):value.setflags(write=False)
    sparse_bytes=sum(value.nbytes for value in (kernel.data,kernel.indices,kernel.indptr))
    metadata=dict(source_grid_shape=list(mask.shape),source_foreground_states=m,all_Source_FG_retained=True,
        eta=ETA,radius=RADIUS,geodesic_cutoff=GEODESIC_CUTOFF,uniform_floor_definition="A_ab=.5K_ab+.5 for a!=b; A_aa=0",
        directed_kernel_nnz=int(kernel.nnz),kernel_storage_bytes=int(sparse_bytes),scale_storage_bytes=int(scale.nbytes),
        dense_relation_allocated=False,offdiag_row_sum_target=m-1 if m else None,
        kernelrowresid=residual,relative_offdiag_row_sum_max_residual=residual,
        absolute_offdiag_row_sum_max_residual=residual*(m-1) if m else 0.,balance_iterations=iterations,
        degenerate=m<=3,degenerate_relation="R is all ones" if 0<m<=3 else "no Source FG; abstain" if not m else None,
        scale_applies=m>=4,independent_equation_implementation_not_verified_Pro_code=True)
    return RelationKernel(kernel,scale,ETA,m,coords,m<=3,metadata)


def build_query_tree(raw_query_features, shape):
    """Deterministic maximum tree on only four-neighbour raw-feature edges.

    Kruskal ties are (-w,i,j), i<j. Zero-weight edges are retained as neutral
    connections, so a complete rectangular Query grid has N-1 tree edges.
    Unit-vector cosine roundoff is clamped to [0,1], with count recorded.
    """
    import numpy as np
    shape=tuple(int(value) for value in shape)
    if len(shape)!=2 or min(shape)<=0:raise ValueError("Complete positive Query grid dimensions required")
    height,width=shape;n=height*width
    raw=np.asarray(raw_query_features,dtype=np.float64)
    if raw.ndim==3 and raw.shape[:2]==shape:raw=raw.reshape(n,raw.shape[-1])
    raw=_unit_features(raw,n,"Raw Query features")
    edges=[];clipped=0
    for i in range(n):
        y,x=divmod(i,width)
        for j in ((i+1,) if x+1<width else ()) + ((i+width,) if y+1<height else ()):
            cosine=float(np.dot(raw[i],raw[j]));weight=max(cosine,0.)
            if weight>1:clipped+=1;weight=1.
            edges.append((weight,i,j))
    edges.sort(key=lambda value:(-value[0],value[1],value[2]))
    representatives=list(range(n))
    def find(i):
        while representatives[i]!=i:
            representatives[i]=representatives[representatives[i]];i=representatives[i]
        return i
    adjacency=[[] for _ in range(n)];accepted=[]
    for weight,i,j in edges:
        left,right=find(i),find(j)
        if left==right:continue
        representatives[right]=left
        adjacency[i].append((j,weight));adjacency[j].append((i,weight));accepted.append((weight,i,j))
    parent=np.full(n,-2,dtype=np.int64);edgew=np.zeros(n,np.float64);depth=np.zeros(n,np.int64)
    roots=[];preorder=[];children=[[] for _ in range(n)]
    for root in range(n):
        if parent[root]!=-2:continue
        roots.append(root);parent[root]=-1;queue=deque([root])
        while queue:
            node=queue.popleft();preorder.append(node)
            for other,weight in sorted(adjacency[node]):
                if other==parent[node]:continue
                if parent[other]!=-2:raise RuntimeError("Kruskal output contains a cycle")
                parent[other]=node;edgew[other]=weight;depth[other]=depth[node]+1
                children[node].append(other);queue.append(other)
    levels=tuple(np.asarray([i for i in preorder if depth[i]==level],dtype=np.int64) for level in range(int(depth.max())+1))
    for array in (parent,edgew,*levels):array.setflags(write=False)
    metadata=dict(query_grid_shape=list(shape),query_nodes=n,grid_edges=len(edges),tree_edges=len(accepted),
        zero_weight_tree_edges=sum(weight==0 for weight,_,_ in accepted),components=len(roots),
        tie_rule="(-weight,i,j), row-major i<j",nonlocal_edges=0,unit_cosine_roundoff_clamps=clipped,
        zero_edges_retained_as_neutral=True,maximum_depth=int(depth.max()))
    return QueryTree(shape,parent,tuple(tuple(items) for items in children),levels,edgew,tuple(roots),
        tuple(preorder),tuple(reversed(preorder)),metadata)


def _validate_inputs(h,pi,relation,tree):
    import numpy as np
    unary=np.asarray(h,dtype=np.float64);probability=np.asarray(pi,dtype=np.float64)
    if unary.shape!=tree.shape or not np.isfinite(unary).all() or (unary<0).any() or (unary>1).any():
        raise ValueError("h must give finite [0,1] mass on EVERY Query node")
    n=unary.size;m=relation.m
    if probability.shape!=(n,m) or not np.isfinite(probability).all() or (probability<0).any():
        raise ValueError("pi must retain every Query x Source-FG state")
    if m and not np.allclose(probability.sum(1),1,atol=1e-12,rtol=0):
        raise ValueError("Each full pi row must sum to1")
    if len(tree.parent)!=n or len(tree.preorder)!=n or len(set(tree.preorder))!=n:
        raise ValueError("Exact complete Query tree required")
    if not np.isfinite(tree.edgew).all() or (tree.edgew<0).any() or (tree.edgew>1).any():
        raise ValueError("Unit-cosine tree weights must remain in[0,1]")
    positions={node:index for index,node in enumerate(tree.preorder)}
    width=tree.shape[1]
    for child,parent in enumerate(tree.parent):
        if parent<0:
            if parent!=-1 or child not in tree.roots:raise ValueError("Query root identity changed")
            continue
        if (parent>=n or child not in tree.children[int(parent)] or positions[int(parent)]>=positions[child] or
                abs(child//width-int(parent)//width)+abs(child%width-int(parent)%width)!=1):
            raise ValueError("Only actual acyclic four-neighbour Query tree edges are legal")
    return unary,probability


def _normalised_node(log_matched,log_uf,log_bg):
    import numpy as np
    maximum=max(float(log_matched.max()),float(log_uf),float(log_bg))
    if not np.isfinite(maximum):raise RuntimeError("All latent states have zero mass")
    matched=np.exp(log_matched-maximum)
    uf=float(np.exp(log_uf-maximum));bg=float(np.exp(log_bg-maximum))
    normalizer=float(matched.sum())+uf+bg
    return matched/normalizer,uf/normalizer,bg/normalizer


def _relation_action(relation,vector,permutation):
    import numpy as np
    if permutation is None:return relation.matvec(vector)
    original=np.empty_like(vector);original[permutation]=vector
    return relation.matvec(original)[permutation]


def tree_sum_product(h,pi,relation,tree,*,rho=RHO,pair_strength=1.,relation_permutation=None,return_state_probabilities=False):
    """EXACT two-pass tree sum-product for M matched, UF and B states.

    Every incoming message is scaled so UF/B entries equal1. A matched outgoing
    entry is vUF+vB+(1-lambda*w)sum(vM)+lambda*w*(RvM)_a. The common normalisation
    is harmless to exact BP; messages are NEVER iterated two-node odds updates.
    Workspace is two dense NxM log arrays, not a dense matrix per Query edge.
    """
    import numpy as np
    if not 0<=rho<1 or not 0<=pair_strength<=1:raise ValueError("Frozen nonnegative rho/lambda domain required")
    unary,probability=_validate_inputs(h,pi,relation,tree);flat=unary.reshape(-1);n=len(flat);m=relation.m
    permutation=None
    if relation_permutation is not None:
        permutation=np.asarray(relation_permutation,dtype=np.int64)
        if permutation.shape!=(m,) or sorted(permutation.tolist())!=list(range(m)):
            raise ValueError("Source shuffle must be a complete state permutation")
    metadata=dict(rho=float(rho),pair_strength=float(pair_strength),exact_two_pass_tree=True,
        foreground_is_calibrated_task_probability=False,relation_only_shuffled=permutation is not None,
        all_Query_nodes=n,all_Source_FG_states=m,abstained=m==0,
        matched_message_workspace_bytes=2*n*m*8,full_pi_bytes=n*m*8,
        R_matvec_calls=0,source_relation_degenerate=relation.degenerate)
    if not m:
        return dict(foreground=unary.copy(),matched_foreground=np.zeros_like(unary),
            unmatched_foreground=unary.copy(),background=1-unary,matched_states=np.empty((n,0)) if return_state_probabilities else None,
            metadata=dict(metadata,abstention_rule="No Source structure: return supplied h without dropping the case; UF/B bookkeeping is fallback, not this model's BP"))
    exact_uniform=bool(np.all(probability==1./m))
    if relation.degenerate or pair_strength==0 or exact_uniform:
        states=(1-rho)*flat[:,None]*probability if return_state_probabilities else None
        return dict(foreground=unary.copy(),matched_foreground=(1-rho)*unary,
            unmatched_foreground=rho*unary,background=1-unary,matched_states=states,
            metadata=dict(metadata,neutral_uniform_pi=exact_uniform,
                neutral_reason="degenerate_Rones" if relation.degenerate else "lambda0" if pair_strength==0 else "exact_uniform_pi_regular_R"))
    incoming=np.zeros((n,m),dtype=np.float64);up=np.zeros((n,m),dtype=np.float64)
    with np.errstate(divide="ignore"):
        logh=np.log(flat);logbg=np.log1p(-flat);loguf=np.log(rho)+logh if rho else np.full(n,-np.inf)
    def node(i,accumulated):
        with np.errstate(divide="ignore"):
            logmatched=np.log(probability[i])+np.log1p(-rho)+logh[i]+accumulated
        return _normalised_node(logmatched,loguf[i],logbg[i])
    def message(i,accumulated,weight):
        strength=pair_strength*weight
        if strength==0:return np.zeros(m,dtype=np.float64)
        matched,uf,bg=node(i,accumulated)
        transformed=_relation_action(relation,matched,permutation)
        metadata["R_matvec_calls"]+=1
        value=uf+bg+(1-strength)*matched.sum()+strength*transformed
        if not np.isfinite(value).all() or (value<=0).any():
            raise RuntimeError("Nonpositive/nonfinite matched message; no clipping fallback")
        return np.log(value)
    for i in tree.postorder:
        for child in tree.children[i]:incoming[i]+=up[child]
        parent=int(tree.parent[i])
        if parent>=0:
            up[i]=message(i,incoming[i],tree.edgew[i])
    matched_mass=np.empty(n);uf_mass=np.empty(n);bg_mass=np.empty(n)
    states=np.empty((n,m),np.float64) if return_state_probabilities else None
    for i in tree.preorder:
        matched,uf,bg=node(i,incoming[i]);matched_mass[i]=matched.sum();uf_mass[i]=uf;bg_mass[i]=bg
        if states is not None:states[i]=matched
        for child in tree.children[i]:
            incoming[child]+=message(i,incoming[i]-up[child],tree.edgew[child])
    normalization=matched_mass+uf_mass+bg_mass
    # Equal to matched+UF algebraically; this form preserves exact h=0/1
    # certainty because impossible B/F states have exactly zero unary mass.
    foreground=1-bg_mass
    if (np.max(np.abs(normalization-1))>1e-12 or foreground.min()<-1e-12 or foreground.max()>1+1e-12):
        raise RuntimeError("Node probability masses failed sum/nonnegativity contract")
    lower=rho*flat/(rho*flat+1-flat) if rho else np.zeros(n)
    metadata.update(probability_sum_max_residual=float(np.max(np.abs(normalization-1))),
        UF_foreground_lower_bound_min_margin=float(np.min(foreground-lower)),
        lower_bound_definition="rho*h/(rho*h+1-h); not calibrated accuracy",directed_messages=2*(n-len(tree.roots)))
    return dict(foreground=foreground.reshape(tree.shape),matched_foreground=matched_mass.reshape(tree.shape),
        unmatched_foreground=uf_mass.reshape(tree.shape),background=bg_mass.reshape(tree.shape),matched_states=states,metadata=metadata)


def binary_potts(h,tree):
    """Fixed naive control: same-label exp(1*w), different-label1, unary h."""
    import numpy as np
    unary=np.asarray(h,np.float64)
    if unary.shape!=tree.shape or not np.isfinite(unary).all() or (unary<0).any() or (unary>1).any():
        raise ValueError("Same complete h/tree required for binary Potts")
    flat=unary.reshape(-1);n=len(flat)
    with np.errstate(divide="ignore"):logunary=np.stack([np.log1p(-flat),np.log(flat)],axis=1)
    incoming=np.zeros((n,2));up=np.zeros((n,2))
    def message(i,acc,weight):
        logvalue=logunary[i]+acc;maximum=logvalue.max();v=np.exp(logvalue-maximum);v/=v.sum()
        same=np.exp(weight);out=np.log(np.array([same*v[0]+v[1],v[0]+same*v[1]]))
        return out-out[0]
    for i in tree.postorder:
        for child in tree.children[i]:incoming[i]+=up[child]
        parent=int(tree.parent[i])
        if parent>=0:up[i]=message(i,incoming[i],tree.edgew[i])
    result=np.empty(n)
    for i in tree.preorder:
        logvalue=logunary[i]+incoming[i];value=np.exp(logvalue-logvalue.max());result[i]=value[1]/value.sum()
        for child in tree.children[i]:incoming[child]+=message(i,incoming[i]-up[child],tree.edgew[child])
    return result.reshape(tree.shape)


def stable_source_permutation(case_identity,m):
    import numpy as np
    if not isinstance(case_identity,str) or not case_identity:
        raise ValueError("Explicit legal Source/Query identity string required; no GT-derived shuffle seed")
    digest=hashlib.sha256(("PCF_SourceRelationShuffle_v1\0"+case_identity).encode()).hexdigest()
    return np.random.default_rng(int(digest[:16],16)).permutation(m),digest


def estimate_work(query_nodes,relation,*,return_state_probabilities=False):
    """Shape/operation metadata ONLY. It deliberately makes no runtime claim."""
    n=int(query_nodes);m=relation.m;calls=2*max(n-1,0) if m>=4 else 0
    sparse_ops=calls*2*int(relation.sparseK.nnz)
    return dict(query_nodes=n,source_states=m,pi_float64_bytes=n*m*8,
        BP_two_dense_log_work_arrays_bytes=2*n*m*8,optional_matched_state_output_bytes=n*m*8 if return_state_probabilities else 0,
        relation_sparse_plus_scale_bytes=relation.metadata["kernel_storage_bytes"]+relation.metadata["scale_storage_bytes"],
        one_BP_arm_R_matvec_calls=calls,one_BP_arm_sparse_multiply_add_operations=sparse_ops,
        one_BP_arm_rank1_and_diagonal_work_order="O(2*(N-1)*M)",dense_edge_M_squared_allocations=0,
        four_structural_BP_arms_sparse_operations=4*sparse_ops,
        not_a_measured_runtime=True,no_downsampling=True,no_Source_state_truncation=True)


def run_controls(h,pi,source_mask,raw_query_features,case_identity,*,relation=None,tree=None,max_sparse_operations=None):
    """All fixed arms; sequential BP workspaces, common h/pi/tree/features.

    A caller budget failure raises BEFORE inference; never top-k/downsample.
    Binary Potts has exp(w) same-label weight, an explicit implementation choice.
    """
    import numpy as np
    relation=build_relation(source_mask) if relation is None else relation
    source=_hard_source(source_mask)
    if relation.m!=int(source.sum()) or not np.array_equal(relation.coords,np.argwhere(source)):
        raise ValueError("Prebuilt relation must preserve the same ALL-Source-FG row-major state identities")
    tree=build_query_tree(raw_query_features,np.asarray(h).shape) if tree is None else tree
    _validate_inputs(h,pi,relation,tree)
    estimate=estimate_work(np.asarray(h).size,relation)
    if max_sparse_operations is not None and estimate["four_structural_BP_arms_sparse_operations"]>max_sparse_operations:
        raise RuntimeError("Full-state operation budget exceeded: hold preparation; no top-k or resolution fallback")
    permutation,digest=stable_source_permutation(case_identity,relation.m)
    configs=(("PCF",RHO,1.,None),("sourceRelationShuffled",RHO,1.,permutation),
             ("noUF",0.,1.,None),("noUF_lambda081",0.,.81,None))
    predictions={};arms={}
    for name,rho,strength,shuffle in configs:
        result=tree_sum_product(h,pi,relation,tree,rho=rho,pair_strength=strength,relation_permutation=shuffle)
        predictions[name]=result["foreground"];arms[name]=result["metadata"]
    predictions["binaryPotts"]=binary_potts(h,tree);predictions["unary"]=np.asarray(h,np.float64).copy()
    return dict(predictions=predictions,statistics=dict(relation=relation.metadata,query_tree=tree.metadata,arms=arms,
        operation_estimate=estimate,source_relation_shuffle_seed_sha256=digest,
        source_shuffle_definition="R'[i,j]=R[p[i],p[j]], pi unchanged",rho=RHO,pair_strength=1.,
        binary_potts_potential="same-label exp(1*w), different-label1",new_image_encodings=0,
        QueryGT_used=False,training=False,postprocessing=False,calibrated_task_probabilities_claimed=False))
