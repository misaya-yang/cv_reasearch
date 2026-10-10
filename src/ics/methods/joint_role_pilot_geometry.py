"""R1-G: legal reference positions and relative inverse-similarity geometry.

Uses the existing, frozen observation and BP primitives. Only reference labels
are accepted. Original b0, model temperature1, lambda1 and4+4 rounds remain.
This is an experimental complete readout, not an adopted default method.
"""
from dataclasses import dataclass
import time
import numpy as np

from . import joint_reference_bp as bp


def coordinate_scale(hw):
    h, w = map(float, hw)
    if min(h, w) <= 0:
        raise ValueError('Require positive input dimensions')
    return np.array([w, h], dtype=np.float64) / (64 * np.sqrt(w*h))


def make_graph(xy, grid_hw, reference_hw, query_hw):
    h, w = grid_hw
    short = bp.build_grid_graph(xy, grid_hw)
    sr = coordinate_scale(reference_hw)
    sq = coordinate_scale(query_hw)
    diameter = float(np.linalg.norm(sr))
    step = sq * np.array([64/w, 64/h])
    lx, ly = np.ceil(16*diameter/step).astype(int)
    pairs = list(zip(short.u.tolist(), short.v.tolist()))
    short_count = len(pairs)
    seen = set(pairs)
    for i in range(h*w):
        row, col = divmod(i, w)
        for j in (i+int(lx) if col+lx<w else -1,
                  i+int(ly)*w if row+ly<h else -1):
            if j >= 0 and (i, j) not in seen:
                pairs.append((i, j)); seen.add((i, j))
    pair = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    u, v = pair.T
    degree = np.bincount(pair.ravel(), minlength=h*w)
    weight = 1/np.maximum(degree[u], degree[v]).astype(np.float64)
    graph = bp.GridGraph(u, v, xy[v]-xy[u], degree, weight,
        np.column_stack((u, v)).ravel(), np.column_stack((v, u)).ravel(),
        np.arange(2*len(u), dtype=np.int64)^1,
        np.repeat(np.arange(len(u), dtype=np.int64), 2))
    neighbours = [[] for _ in range(h*w)]
    for a, b in pair:
        neighbours[a].append(b); neighbours[b].append(a)
    colour = np.full(h*w, -1, dtype=np.int64)
    for i, adjacent in enumerate(neighbours):
        used = {colour[j] for j in adjacent if colour[j] >= 0}
        c = 0
        while c in used: c += 1
        colour[i] = c
    assert np.all(colour[u] != colour[v])
    return graph, colour, dict(short_edges=short_count, span_edges=len(u)-short_count,
        span_x=int(lx), span_y=int(ly), colours=int(colour.max()+1))


class ReferenceSupport:
    """Exact projection onto the union of reference-role pixel squares."""
    def __init__(self, mask1024, reference_hw):
        mask = np.asarray(mask1024)
        if mask.shape != (1024, 1024) or not np.isin(mask, (0, 1)).all():
            raise ValueError('Require a binary aligned reference mask')
        self.cells = mask.astype(bool).reshape(64,16,64,16).transpose(0,2,1,3).reshape(4096,16,16)
        self.step = coordinate_scale(reference_hw)/16
        self.cache = {}

    def bounds(self, atom):
        atom = int(atom)
        if atom not in self.cache:
            patch, role = divmod(atom, 2)
            ys, xs = np.nonzero(self.cells[patch] == bool(role))
            if not len(xs): raise ValueError('Candidate has empty legal role support')
            px, py = patch%64, patch//64
            lo = np.column_stack((16*px+xs, 16*py+ys))*self.step
            hi = lo+self.step
            # A filled rectangle has precisely the same union/projection.
            if len(xs) == (int(xs.max()-xs.min())+1)*(int(ys.max()-ys.min())+1):
                lo, hi = lo.min(0, keepdims=True), hi.max(0, keepdims=True)
            self.cache[atom] = (lo, hi)
        return self.cache[atom]

    def project(self, points, atom_ids, valid):
        shape = np.shape(points)
        flat = np.asarray(points, dtype=np.float64).reshape(-1,2)
        ids = np.asarray(atom_ids).reshape(-1)
        indexes = np.flatnonzero(np.asarray(valid).reshape(-1))
        order = indexes[np.argsort(ids[indexes], kind='stable')]
        result = flat.copy()
        splits = np.r_[0, np.flatnonzero(np.diff(ids[order]))+1, len(order)]
        for start, end in zip(splits[:-1], splits[1:]):
            group = order[start:end]
            if not len(group): continue
            lo, hi = self.bounds(ids[group[0]])
            if len(lo) == 1:
                result[group] = np.clip(flat[group], lo[0], hi[0])
                continue
            for j in range(0, len(group), 256):
                at = group[j:j+256]
                clipped = np.clip(flat[at,None,:], lo[None], hi[None])
                distance = ((clipped-flat[at,None,:])**2).sum(2)
                # np.nonzero is row-major; argmin gives the declared pixel tie.
                result[at] = clipped[np.arange(len(at)), distance.argmin(1)]
        result[~np.asarray(valid).reshape(-1)] = 0
        return result.reshape(shape)


def pair_distance(positions, valid, transforms, graph):
    e, k = len(graph.u), positions.shape[1]
    raw = np.empty((e,k,k), dtype=np.float64)
    for start in range(0,e,bp._CHUNK):
        sl = slice(start, min(start+bp._CHUNK,e))
        u, v, delta = graph.u[sl], graph.v[sl], graph.delta[sl]
        displacement = positions[v,None,:,:]-positions[u,:,None,:]
        distance = np.zeros(displacement.shape[:-1], dtype=np.float64)
        for nodes in (u,v):
            residual = np.einsum('eij,eabj->eabi',transforms[nodes],displacement)-delta[:,None,None,:]
            distance += (residual*residual).sum(-1)
        distance /= 2*(delta*delta).sum(1)[:,None,None]
        raw[sl] = np.where(valid[u,:,None]&valid[v,None,:],distance,np.inf)
    return raw


def _mm_update(positions, valid, atom_ids, support, transforms, graph, colour, old_raw, edge):
    start = time.monotonic()
    xi = positions.copy(); n,k = valid.shape
    delta2 = (graph.delta*graph.delta).sum(1)
    gamma = edge*(old_raw<=1)*graph.weight[:,None,None]/delta2[:,None,None]
    before = bp._expected_pair_cost(edge,old_raw,graph,1.)
    for c in range(int(colour.max())+1):
        numerator = np.zeros((n,k,2)); denominator = np.zeros((n,k))
        incident = np.flatnonzero((colour[graph.u]==c)|(colour[graph.v]==c))
        for start_edge in range(0,len(incident),bp._CHUNK):
            at = incident[start_edge:start_edge+bp._CHUNK]
            u,v,delta = graph.u[at],graph.v[at],graph.delta[at]
            g = gamma[at]
            scale2 = (transforms[u,:,0]**2).sum(1)+(transforms[v,:,0]**2).sum(1)
            offset = np.einsum('eji,ej->ei',transforms[u]+transforms[v],delta)
            mu,mv = g.sum(2),g.sum(1)
            nu = scale2[:,None,None]*np.einsum('eab,ebd->ead',g,xi[v])-mu[:,:,None]*offset[:,None,:]
            nv = scale2[:,None,None]*np.einsum('eab,ead->ebd',g,xi[u])+mv[:,:,None]*offset[:,None,:]
            for nodes,mass,value in ((u,mu,nu),(v,mv,nv)):
                select = colour[nodes]==c
                np.add.at(numerator,nodes[select],value[select])
                np.add.at(denominator,nodes[select],mass[select]*scale2[select,None])
        selected = (colour==c)[:,None]&valid&(denominator>0)
        target = np.divide(numerator,denominator[:,:,None],out=xi.copy(),where=denominator[:,:,None]>0)
        projected = support.project(target,atom_ids,selected)
        xi[selected] = projected[selected]
    A=np.zeros(n);B=np.zeros(n);D=np.zeros(n)
    for start_edge in range(0,len(graph.u),bp._CHUNK):
        sl=slice(start_edge,min(start_edge+bp._CHUNK,len(graph.u)))
        u,v,delta=graph.u[sl],graph.v[sl],graph.delta[sl]
        displacement=xi[v,None,:,:]-xi[u,:,None,:]
        g=gamma[sl]
        vec=(g[:,:,:,None]*displacement).sum((1,2))
        ea=(vec*delta).sum(1)
        eb=vec[:,0]*delta[:,1]-vec[:,1]*delta[:,0]
        ed=(g*(displacement*displacement).sum(-1)).sum((1,2))
        for nodes in (u,v):
            np.add.at(A,nodes,ea);np.add.at(B,nodes,eb);np.add.at(D,nodes,ed)
    updated, info=bp._fit_moments(A,B,D,transforms,(.125,8.))
    zero=(D>0)&(np.hypot(A,B)<=bp._EPS*D)
    old_scale=np.sqrt(np.linalg.det(transforms))
    updated[zero]=transforms[zero]*(.125/old_scale[zero,None,None])
    new_raw=pair_distance(xi,valid,updated,graph)
    after=bp._expected_pair_cost(edge,new_raw,graph,1.)
    if after>before+1e-9*max(1.,abs(before)):
        raise RuntimeError(f'R1-G fixed-belief MM increased: {before} -> {after}')
    return xi,updated,new_raw,dict(seconds=time.monotonic()-start,
        fixed_belief_pair_before=before,fixed_belief_pair_after=after,
        fixed_belief_MM_nonincrease=True,active_edges_for_MM=int((gamma.sum((1,2))>0).sum()),
        scale_no_data_nodes=len(info['no_data_nodes']),rotation_unidentified_nodes=int(zero.sum()))


@dataclass(frozen=True)
class GeometryResult:
    p_fg: np.ndarray
    node_belief: np.ndarray
    positions: np.ndarray
    transforms: np.ndarray
    diagnostics: dict


def solve(observation, reference_mask1024, reference_hw, query_hw, *, grid_hw=(128,128)):
    start=time.monotonic()
    probability=np.asarray(observation['b0'],dtype=np.float64)
    valid=np.asarray(observation['valid'],dtype=bool)
    role=np.asarray(observation['role'])
    patch=np.asarray(observation['refpatch'])
    if probability.ndim!=2 or valid.shape!=probability.shape or role.shape!=probability.shape:
        raise ValueError('Malformed candidate arrays')
    if not valid.any(1).all() or not np.isfinite(probability).all() or (probability[valid]<=0).any():
        raise ValueError('Require positive finite candidate probabilities')
    probability=np.where(valid,probability,0.)
    if not np.allclose(probability.sum(1),1.,rtol=0,atol=1e-6):
        raise ValueError('Candidate probability must sum to one')
    probability/=probability.sum(1,keepdims=True)
    sr,sq=coordinate_scale(reference_hw),coordinate_scale(query_hw)
    centroids=np.asarray(observation['ref_xy'],dtype=np.float64)*sr
    query_xy=np.asarray(observation['query_xy'],dtype=np.float64)*sq
    support=ReferenceSupport(reference_mask1024,reference_hw)
    atom_ids=patch*2+role
    positions=support.project(centroids,atom_ids,valid)
    projection=np.linalg.norm(positions-centroids,axis=-1)[valid]
    graph,colour,graph_info=make_graph(query_xy,grid_hw,reference_hw,query_hw)
    short=bp.build_grid_graph(query_xy,grid_hw)
    forward,initial=bp._initialize(probability,positions,short,(.125,8.))
    fallback=np.asarray(initial['no_data_nodes']+initial['zero_or_numerical_covariance_nodes'],dtype=int)
    forward[fallback]=np.eye(2)
    transforms=np.linalg.inv(forward)
    log_unary=np.full(probability.shape,-np.inf)
    np.log(probability,out=log_unary,where=valid)
    messages=bp._uniform_messages(valid,graph)
    raw=pair_distance(positions,valid,transforms,graph)
    stages=[];update={}
    for stage in range(2):
        stage_start=time.monotonic()
        potential=bp._pair_log_potential(raw,graph,valid,1.)
        residuals=[]
        for _ in range(4):
            messages,residual=bp._message_round(log_unary,messages,graph,potential,valid,.5)
            residuals.append(residual)
        node,edge=bp._beliefs(log_unary,messages,graph,potential)
        stages.append(dict(seconds=time.monotonic()-stage_start,
            message_probability_residuals=residuals,
            expected_pair_cost=bp._expected_pair_cost(edge,raw,graph,1.)))
        if stage==0:
            positions,transforms,raw,update=_mm_update(positions,valid,atom_ids,support,
                transforms,graph,colour,raw,edge)
    p_fg=(node*(role==1)).sum(1)
    if not np.isfinite(p_fg).all():raise RuntimeError('Nonfinite role result')
    priors=(probability*(role==1)).sum(1)
    incident=np.zeros(len(node))
    np.add.at(incident,graph.u,graph.weight);np.add.at(incident,graph.v,graph.weight)
    consistency=max(float(np.abs(edge.sum(2)-node[graph.u]).max()),
                    float(np.abs(edge.sum(1)-node[graph.v]).max())) if len(edge) else 0.
    return GeometryResult(p_fg,node,positions,transforms,dict(
        method='R1-G; original b0; relative inverse geometry; reference-role pixel support',
        seconds=time.monotonic()-start,graph=graph_info,nodes=len(node),edges=len(graph.u),
        initial_projection_nonzero_fraction=float((projection>0).mean()),
        initial_projection_distance_max=float(projection.max()),
        initialization_identity_fallback_nodes=int(len(fallback)),
        candidate_position_variance_median=float(np.median(initial['candidate_position_variance'])),
        MM=update,stages=stages,edge_node_consistency_max_abs=consistency,
        role_flips_vs_unary=int(((p_fg>.5)!=(priors>.5)).sum()),
        mean_absolute_role_probability_change=float(np.abs(p_fg-priors).mean()),
        maximum_incident_budget=float(incident.max()),
        strong_prior_bound_retained=True,R2_role_logit_cap_enabled=False,
        query_GT_reads=0,encoder_forward=0,
        limitation='Finite BP and fixed-belief geometry MM do not guarantee global convergence or segmentation gains.'))
