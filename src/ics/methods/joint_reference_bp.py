"""CPU FP64 reference for the proposed joint correspondence / role solver.

This faithfully retains the proposal's limited geometry, not a corrected or
adopted segmentation method. No FoRIS, encoder, image loader, feature cache or
query labels are accessed. Inputs are candidate probabilities and coordinates.
Default inference is T=1, lambda=1, synchronous4+4 BP, probability damping1/2,
one joint-active edge-belief Procrustes update, sigmas2/1, scales1/8..8.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

_CHUNK = 256  # Memory batching only; does not change the graph or objective.
_EPS = 1e-12


@dataclass(frozen=True)
class BPConfig:
    lambda_: float = 1.
    temperature: float = 1.
    sigmas: tuple[float,float] = (2.,1.)
    rounds: tuple[int,int] = (4,4)
    damping: float = .5  # Old-message probability mass in the convex mixture.
    scale_bounds: tuple[float,float] = (.125,8.)


@dataclass(frozen=True)
class GridGraph:
    u: np.ndarray
    v: np.ndarray
    delta: np.ndarray
    degree: np.ndarray
    weight: np.ndarray
    directed_src: np.ndarray
    directed_dst: np.ndarray
    reverse_id: np.ndarray
    directed_edge_id: np.ndarray


@dataclass(frozen=True)
class JointBPResult:
    p_fg: np.ndarray
    node_belief: np.ndarray
    edge_belief: np.ndarray
    J: np.ndarray
    graph: GridGraph
    diagnostics: dict


def build_grid_graph(query_xy, grid_hw) -> GridGraph:
    """Row-major nodes, per node right edge then down edge; IDs never reorder."""
    h,w = map(int,grid_hw)
    xy = np.asarray(query_xy,dtype=np.float64)
    if h<1 or w<1 or xy.shape!=(h*w,2) or not np.isfinite(xy).all():
        raise ValueError('Query coordinates must match a finite row-major grid')
    edges=[]
    for i in range(h*w):
        row,col=divmod(i,w)
        if col+1<w:edges.append((i,i+1))
        if row+1<h:edges.append((i,i+w))
    pair=np.asarray(edges,dtype=np.int64).reshape(-1,2)
    u,v=pair[:,0],pair[:,1]
    delta=xy[v]-xy[u]
    if len(delta) and ((delta*delta).sum(1)==0).any():
        raise ValueError('Adjacent query coordinates must be distinct')
    degree=np.bincount(pair.ravel(),minlength=h*w)
    weight=1/np.maximum(degree[u],degree[v]).astype(np.float64)
    directed_src=np.column_stack((u,v)).ravel()
    directed_dst=np.column_stack((v,u)).ravel()
    reverse=np.arange(2*len(u),dtype=np.int64)^1
    return GridGraph(u,v,delta,degree,weight,directed_src,directed_dst,reverse,
        np.repeat(np.arange(len(u),dtype=np.int64),2))


def _logsumexp(value, axis, keepdims=False):
    maximum=np.max(value,axis=axis,keepdims=True)
    safe=np.where(np.isfinite(maximum),maximum,0.)
    total=np.exp(value-safe).sum(axis=axis,keepdims=True)
    logged=np.full(total.shape,-np.inf)
    np.log(total,out=logged,where=total>0)
    result=safe+logged
    return result if keepdims else np.squeeze(result,axis=axis)


def _normalize_logs(value, axis):
    norm=_logsumexp(value,axis=axis,keepdims=True)
    if not np.isfinite(norm).all():
        raise ValueError('A probability row has no finite valid states')
    return value-norm


def _incoming(messages, graph, n, k):
    result=np.zeros((n,k),dtype=np.float64)
    # Invalid padded message entries are -inf, not incoming evidence. The
    # corresponding unary already excludes them, avoiding -inf - (-inf).
    np.add.at(result,graph.directed_dst,np.where(np.isfinite(messages),messages,0.))
    return result


def _pair_geometry(ref_xy, valid, J, graph, sigma):
    e,k=len(graph.u),ref_xy.shape[1]
    raw=np.empty((e,k,k),dtype=np.float64)
    for start in range(0,e,_CHUNK):
        end=min(e,start+_CHUNK);sl=slice(start,end)
        u,v=graph.u[sl],graph.v[sl]
        displacement=ref_xy[v,None,:,:]-ref_xy[u,:,None,:]
        left=np.einsum('eab,eb->ea',J[u],graph.delta[sl])
        right=np.einsum('eab,eb->ea',J[v],graph.delta[sl])
        d=((displacement-left[:,None,None,:])**2).sum(-1)
        d+=((displacement-right[:,None,None,:])**2).sum(-1)
        d/=2*sigma*sigma
        pair_valid=valid[u,:,None]&valid[v,None,:]
        if not np.isfinite(d[pair_valid]).all():
            raise ValueError('Geometry exceeded finite FP64 arithmetic')
        raw[sl]=np.where(pair_valid,d,np.inf)
    return raw


def _pair_log_potential(raw, graph, valid, lambda_):
    potential=-lambda_*graph.weight[:,None,None]*np.minimum(raw,1.)
    pair_valid=valid[graph.u,:,None]&valid[graph.v,None,:]
    return np.where(pair_valid,potential,-np.inf)


def _uniform_messages(valid, graph):
    target=valid[graph.directed_dst]
    count=target.sum(1)
    return np.where(target,-np.log(count[:,None]),-np.inf)


def _message_round(log_unary, messages, graph, log_potential, valid, damping):
    """One synchronous sum-product round; damping is a probability mixture."""
    if not len(graph.u):return messages.copy(),0.
    n,k=log_unary.shape
    total=_incoming(messages,graph,n,k)
    reverse=np.where(np.isfinite(messages[graph.reverse_id]),messages[graph.reverse_id],0.)
    cavity=log_unary[graph.directed_src]+total[graph.directed_src]-reverse
    raw=np.empty_like(messages)
    for start in range(0,len(graph.u),_CHUNK):
        end=min(len(graph.u),start+_CHUNK)
        raw[2*start:2*end:2]=_logsumexp(cavity[2*start:2*end:2,:,None]+log_potential[start:end],axis=1)
        raw[2*start+1:2*end:2]=_logsumexp(cavity[2*start+1:2*end:2,None,:]+log_potential[start:end],axis=2)
    raw=_normalize_logs(raw,axis=1)
    if damping==0:updated=raw
    elif damping==1:updated=messages.copy()
    else:updated=np.logaddexp(messages+np.log(damping),raw+np.log1p(-damping))
    updated=_normalize_logs(updated,axis=1)
    residual=float(np.max(np.abs(np.exp(updated)-np.exp(messages))))
    return updated,residual


def _beliefs(log_unary, messages, graph, log_potential):
    n,k=log_unary.shape
    incoming=_incoming(messages,graph,n,k)
    node=np.exp(_normalize_logs(log_unary+incoming,axis=1))
    node/=node.sum(1,keepdims=True)
    edge=np.empty((len(graph.u),k,k),dtype=np.float64)
    for start in range(0,len(graph.u),_CHUNK):
        end=min(len(graph.u),start+_CHUNK);sl=slice(start,end)
        u,v=graph.u[sl],graph.v[sl]
        msg_vu=np.where(np.isfinite(messages[2*start+1:2*end:2]),messages[2*start+1:2*end:2],0.)
        msg_uv=np.where(np.isfinite(messages[2*start:2*end:2]),messages[2*start:2*end:2],0.)
        left=log_unary[u]+incoming[u]-msg_vu
        right=log_unary[v]+incoming[v]-msg_uv
        value=left[:,:,None]+right[:,None,:]+log_potential[sl]
        normalized=np.exp(_normalize_logs(value,axis=(1,2)))
        edge[sl]=normalized/normalized.sum((1,2),keepdims=True)
    return node,edge


def _fit_moments(A,B,D,old_J,scale_bounds):
    n=len(D);low,high=scale_bounds
    result=old_J.copy();has_data=D>0
    norm=np.hypot(A,B)
    raw_scale=np.divide(norm,D,out=np.zeros(n),where=has_data)
    scale=np.clip(raw_scale,low,high)
    theta=np.arctan2(B,A)
    co,si=np.cos(theta)*scale,np.sin(theta)*scale
    result[has_data,0,0]=co[has_data];result[has_data,1,1]=co[has_data]
    result[has_data,0,1]=-si[has_data];result[has_data,1,0]=si[has_data]
    return result,dict(no_data_nodes=np.flatnonzero(~has_data).tolist(),
        zero_or_numerical_covariance_nodes=np.flatnonzero(has_data&(norm<=_EPS*D)).tolist(),
        clipped_to_min_nodes=np.flatnonzero(has_data&(raw_scale<=low)).tolist(),
        clipped_to_max_nodes=np.flatnonzero(has_data&(raw_scale>=high)).tolist(),
        raw_scale=raw_scale.tolist(),moment_mass=D.tolist(),
        convention='D=0 preserves previousJ; covariance0 uses theta0/minscale, not an identified scale')


def _initialize(probability, ref_xy, graph, scale_bounds):
    n=len(probability)
    mean=np.einsum('nk,nkd->nd',probability,ref_xy)
    displacement=mean[graph.v]-mean[graph.u]
    edge_A=graph.weight*(displacement*graph.delta).sum(1)
    edge_B=graph.weight*(displacement[:,1]*graph.delta[:,0]-displacement[:,0]*graph.delta[:,1])
    edge_D=graph.weight*(graph.delta*graph.delta).sum(1)
    A=np.zeros(n);B=np.zeros(n);D=np.zeros(n)
    for nodes in (graph.u,graph.v):
        np.add.at(A,nodes,edge_A);np.add.at(B,nodes,edge_B);np.add.at(D,nodes,edge_D)
    initial=np.tile(np.eye(2),(n,1,1))
    J,info=_fit_moments(A,B,D,initial,scale_bounds)
    variance=np.einsum('nk,nk->n',probability,((ref_xy-mean[:,None,:])**2).sum(-1))
    info.update(candidate_expected_xy=mean.tolist(),candidate_position_variance=variance.tolist(),
        zero_expected_edge_displacement_count=int(((displacement*displacement).sum(1)<=_EPS**2).sum()),
        warning='Candidate mean can erase modes. Variance is dispersion, not a certified multimodality or scale test. Isolated initialization identity is a declared fallback.')
    return J,info


def _expected_pair_cost(edge, raw, graph, lambda_):
    return float(lambda_*np.dot(graph.weight,(edge*np.minimum(raw,1.)).sum((1,2))))


def _update_J(ref_xy, valid, J, graph, raw, edge, sigma, config):
    """Fixed edge snapshot, old joint-active pairs, weighted MM Procrustes."""
    if config.lambda_==0:
        return J.copy(),dict(skipped='lambda_zero_J_is_unidentified',expected_pair_before=0.,expected_pair_after=0.)
    n=len(J);A=np.zeros(n);B=np.zeros(n);D=np.zeros(n)
    active_mass=np.zeros(len(graph.u))
    for start in range(0,len(graph.u),_CHUNK):
        end=min(len(graph.u),start+_CHUNK);sl=slice(start,end)
        u,v=graph.u[sl],graph.v[sl]
        gamma=graph.weight[sl,None,None]*edge[sl]*(raw[sl]<1.)
        mass=gamma.sum((1,2));active_mass[sl]=mass
        reference_delta=ref_xy[v,None,:,:]-ref_xy[u,:,None,:]
        vector=(gamma[:,:,:,None]*reference_delta).sum((1,2))
        delta=graph.delta[sl]
        edge_A=(vector*delta).sum(1)
        edge_B=vector[:,1]*delta[:,0]-vector[:,0]*delta[:,1]
        edge_D=mass*(delta*delta).sum(1)
        for nodes in (u,v):
            np.add.at(A,nodes,edge_A);np.add.at(B,nodes,edge_B);np.add.at(D,nodes,edge_D)
    updated,info=_fit_moments(A,B,D,J,config.scale_bounds)
    new_raw=_pair_geometry(ref_xy,valid,updated,graph,sigma)
    before=_expected_pair_cost(edge,raw,graph,config.lambda_)
    after=_expected_pair_cost(edge,new_raw,graph,config.lambda_)
    if after>before+1e-10*max(1.,abs(before)):
        raise RuntimeError('Fixed-edge-belief MM pair objective increased')
    info.update(expected_pair_before=before,expected_pair_after=after,
        fixed_edge_belief_MM_nonincrease=True,weighted_active_mass=float(active_mass.sum()),
        gamma='edge_weight * actual_edge_belief * (old_double_endpoint_d < 1)',
        warning='MM nonincrease is only fixed edge belief and fixed sigma, not monotone BP/hardMAP/global optimization')
    return updated,info


def _entropy(probability):
    logs=np.zeros_like(probability)
    np.log(probability,out=logs,where=probability>0)
    return -(probability*logs).sum(axis=tuple(range(1,probability.ndim)))


def _stage_info(node,edge,log_unary,raw,graph,config,residuals):
    if len(graph.u):
        consistency=max(float(np.max(np.abs(edge.sum(2)-node[graph.u]))),
            float(np.max(np.abs(edge.sum(1)-node[graph.v]))))
    else:consistency=0.
    product=np.zeros_like(node)
    np.multiply(node,log_unary,out=product,where=np.isfinite(log_unary))
    expected_unary=-float(product.sum())
    expected_pair=_expected_pair_cost(edge,raw,graph,config.lambda_)
    bethe_entropy=float(_entropy(edge).sum()-np.dot(graph.degree-1,_entropy(node)))
    return dict(message_probability_residuals=residuals,edge_node_consistency_max_abs=consistency,
        expected_unary_energy=expected_unary,expected_pair_energy=expected_pair,
        local_belief_Bethe_free_energy=expected_unary+expected_pair-bethe_entropy,
        active_pair_probability_mean=float((edge*(raw<1.)).sum((1,2)).mean()) if len(edge) else None,
        weighted_active_mass=float(np.dot(graph.weight,(edge*(raw<1.)).sum((1,2)))))


def solve(unary_probability, valid_mask, role, ref_xy, query_xy, grid_hw,
          config: BPConfig | None = None) -> JointBPResult:
    """Consume b0 positive probabilities, never unary costs, and return beliefs.

    Arrays: b0/valid/role[N,K], reference statexy[N,K,2], queryxy[N,2].
    Coordinates are passed through in reference patch units. CPU FP64 is an
    explicit reference implementation; no float32/MPS substitution is made.
    Valid probabilities are normalized per node (only an energy constant).
    Stage2 warm-starts the probability-damped messages from stage1.
    """
    start=time.monotonic();config=BPConfig() if config is None else config
    if not isinstance(config,BPConfig):raise TypeError('config must be BPConfig')
    if config.temperature!=1.:
        raise ValueError('This faithful reference fixes model temperatureT=1')
    if not np.isfinite(config.lambda_) or config.lambda_<0 or len(config.sigmas)!=2 or len(config.rounds)!=2:
        raise ValueError('Require nonnegativelambda and exactly two stages')
    if any(not np.isfinite(s) or s<=0 for s in config.sigmas) or any(int(r)!=r or r<0 for r in config.rounds):
        raise ValueError('Invalid fixed sigma/round configuration')
    if not 0<=config.damping<=1 or not 0<config.scale_bounds[0]<=config.scale_bounds[1] or not np.isfinite(config.scale_bounds).all():
        raise ValueError('Invalid damping/scale bounds')
    probability=np.asarray(unary_probability,dtype=np.float64)
    valid=np.asarray(valid_mask,dtype=bool)
    roles=np.asarray(role)
    positions=np.asarray(ref_xy,dtype=np.float64)
    if probability.ndim!=2 or not probability.shape[1] or valid.shape!=probability.shape or roles.shape!=probability.shape or positions.shape!=(*probability.shape,2):
        raise ValueError('Candidate shapes differ')
    if not np.isfinite(probability).all() or (probability<0).any() or not np.isfinite(positions).all():
        raise ValueError('Require finite probabilities and reference coordinates')
    if not valid.any(1).all() or (probability[valid]<=0).any() or not np.isin(roles[valid],(0,1)).all():
        raise ValueError('Each node needs positive valid probabilities and binary roles')
    probability=np.where(valid,probability,0.)
    input_mass=probability.sum(1)
    if (probability[valid]>1+_EPS).any() or not np.allclose(input_mass,1.,rtol=0.,atol=1e-6):
        raise ValueError('unary_probability must be normalized b0 probabilities, not unary costs/weights')
    probability=probability/input_mass[:,None]
    log_unary=np.full(probability.shape,-np.inf)
    np.log(probability,out=log_unary,where=valid)
    graph=build_grid_graph(query_xy,grid_hw)
    if len(graph.degree)!=len(probability):raise ValueError('Candidates do not match grid size')
    J,initial_info=_initialize(probability,positions,graph,config.scale_bounds)
    messages=_uniform_messages(valid,graph)
    stages=[];update_info=None
    for stage,(sigma,rounds) in enumerate(zip(config.sigmas,config.rounds)):
        stage_start=time.monotonic()
        raw=_pair_geometry(positions,valid,J,graph,sigma)
        log_potential=_pair_log_potential(raw,graph,valid,config.lambda_)
        residuals=[]
        for _ in range(rounds):
            messages,residual=_message_round(log_unary,messages,graph,log_potential,valid,config.damping)
            residuals.append(residual)
        node,edge=_beliefs(log_unary,messages,graph,log_potential)
        info=_stage_info(node,edge,log_unary,raw,graph,config,residuals)
        info.update(stage=stage,sigma=sigma,rounds=rounds,seconds=time.monotonic()-stage_start)
        stages.append(info)
        if stage==0:
            J,update_info=_update_J(positions,valid,J,graph,raw,edge,sigma,config)
    p_fg=(node*(roles==1)).sum(1)
    mean_displacement=np.zeros(len(graph.u));norm_mean_displacement=np.zeros(len(graph.u));same_position=np.zeros(len(graph.u))
    for start_edge in range(0,len(graph.u),_CHUNK):
        end=min(len(graph.u),start_edge+_CHUNK);sl=slice(start_edge,end)
        displacement=positions[graph.v[sl],None,:,:]-positions[graph.u[sl],:,None,:]
        norm=np.linalg.norm(displacement,axis=-1)
        mean_displacement[sl]=(edge[sl]*norm).sum((1,2))
        mean_vector=(edge[sl,:,:,None]*displacement).sum((1,2))
        norm_mean_displacement[sl]=np.linalg.norm(mean_vector,axis=1)
        same_position[sl]=(edge[sl]*(norm==0)).sum((1,2))
    scale=np.sqrt(np.linalg.det(J))
    incident=np.zeros(len(node))
    np.add.at(incident,graph.u,graph.weight);np.add.at(incident,graph.v,graph.weight)
    diagnostics=dict(implementation='CPU FP64 faithful reference, not adopted default',
        config=dict(lambda_=config.lambda_,temperature=config.temperature,sigmas=config.sigmas,
            rounds=config.rounds,damping=config.damping,scale_bounds=config.scale_bounds),
        input_probability_row_mass_min=float(input_mass.min()),input_probability_row_mass_max=float(input_mass.max()),
        input_semantics='b0 positive probabilities; per-node normalization removes only energy constants',
        node_count=len(node),candidate_slots=node.shape[1],edge_count=len(graph.u),
        edge_ID_order='row-major nodes, right then down; directed2e=u->v,2e+1=v->u',
        initialization=initial_info,J_update=update_info,stages=stages,
        final_scale=scale.tolist(),scale_at_min_count=int((scale<=config.scale_bounds[0]+_EPS).sum()),
        scale_at_max_count=int((scale>=config.scale_bounds[1]-_EPS).sum()),
        mean_reference_displacement=float(mean_displacement.mean()) if len(edge) else None,
        mean_norm_of_expected_reference_displacement=float(norm_mean_displacement.mean()) if len(edge) else None,
        mean_same_reference_position_edge_mass=float(same_position.mean()) if len(edge) else None,
        maximum_incident_edge_weight_budget=float(incident.max()),
        role_odds_log_change_bound_per_node=(config.lambda_*incident).tolist(),
        model_forward=0,feature_reads=0,image_reads=0,query_GT_reads=0,
        seconds=time.monotonic()-start,
        limitations='Fixed finite BP is approximate; no calibration or global optimality. Cheap min-scale correspondence collapse and bounded role-odds correction are retained. Position variance is not evidence of identifiable scale.')
    return JointBPResult(p_fg,node,edge,J,graph,diagnostics)
