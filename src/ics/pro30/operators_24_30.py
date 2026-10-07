"""Fixed numerical operators for original Pro30 M24--M30.

No encoder, labels other than the current reference, or natural experiment here.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.ndimage import binary_dilation
from scipy.special import expit
from . import common


def dot(a,b):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        c=np.asarray(a)@np.asarray(b)
    if not np.isfinite(c).all():raise ArithmeticError('Nonfinite actual dot')
    return c


def axis_overlap(output_intervals, n, pixel_size=1.):
    """Rows average over their actual intersection with [0,n*pixel_size]."""
    rows=[];cols=[];data=[];valid=[]
    for row,(lo,hi) in enumerate(np.asarray(output_intervals,float)):
        width=hi-lo
        if width<=0:raise ValueError('Positive physical interval required')
        a,b=max(0.,lo),min(n*pixel_size,hi);mass=max(0.,b-a)
        valid.append(mass/width)
        if mass<=0:continue
        ids=np.arange(max(0,int(np.floor(a/pixel_size))), min(n,int(np.ceil(b/pixel_size))))
        weights=np.maximum(0.,np.minimum((ids+1)*pixel_size,b)-np.maximum(ids*pixel_size,a))/mass
        rows.extend([row]*len(ids));cols.extend(ids);data.extend(weights)
    return sparse.csr_matrix((data,(rows,cols)),shape=(len(valid),n)),np.asarray(valid)


def exact_footprint(mask,hw,geometry):
    """Original pixels, not a nearest-neighbor downsample of MR."""
    mask=np.asarray(mask)
    if mask.ndim!=2 or not np.isin(mask,[0,1,255]).all():raise ValueError('Complete binary original MR required')
    side=float(geometry['view_side']); sh,sw=map(float,geometry['resized_hw']);oy,ox=map(float,geometry['padding_top_left'])
    if min(sh,sw)<=0 or min(oy,ox)<0 or oy+sh>side or ox+sw>side:raise ValueError('Physical resize/pad geometry required')
    intervals=[]
    for length,grid,offset,resized in zip(mask.shape,hw,(oy,ox),(sh,sw)):
        edges=(np.arange(grid+1)*side/grid-offset)*length/resized
        intervals.append(np.c_[edges[:-1],edges[1:]])
    ay,vy=axis_overlap(intervals[0],mask.shape[0]);ax,vx=axis_overlap(intervals[1],mask.shape[1])
    coverage=np.asarray(ax@(ay@(mask!=0).astype(float)).T).T.ravel()
    valid=(vy[:,None]*vx[None,:]).ravel()
    return coverage*valid,valid


@dataclass
class Split:
    anchor:np.ndarray
    calibration:np.ndarray
    group:np.ndarray
    adequate:bool
    info:dict


def source_split(hw,area,valid):
    yy,xx=np.indices(hw); br,bc=yy//8,xx//8
    group=((br+2*bc)%4).ravel()
    c=np.isin(group,[2,3]).reshape(hw)
    # Full 8-neighbor one-token buffer; includes corners, no C-label construction.
    a=(~c)&(~binary_dilation(c,structure=np.ones((3,3),bool)))
    a=a.ravel()&(valid>0);c=c.ravel()&(valid>0)
    bid=(br*(int(bc.max())+1)+bc).ravel()
    def support(ids):
        return dict(valid_blocks=len(np.unique(bid[ids])),fg_area=float(np.sum(area[ids])),bg_area=float(np.sum((valid-area)[ids])))
    sa,sc=support(a),support(c)
    adequate=all(x['valid_blocks']>=2 and min(x['fg_area'],x['bg_area'])>=8 for x in (sa,sc))
    return Split(a,c,group,adequate,dict(anchor=sa,calibration=sc,buffer_tokens=int(np.count_nonzero((group<2)&~a)),adequate=adequate))


def direction(reference,area,valid,ids,*,normalize=True):
    f=np.asarray(area)*ids;b=(np.asarray(valid)-area)*ids
    if f.sum()<=0 or b.sum()<=0:return None
    mf=common.unit(np.sum(reference*f[:,None],axis=0));mb=common.unit(np.sum(reference*b[:,None],axis=0))
    d=mf-mb
    return (common.unit(d) if normalize else d) if np.linalg.norm(d)>1e-8 else None


def margin_direction(reference,query,area,valid,ids):
    w=direction(reference,area,valid,ids)
    if w is None:return None
    return dot(query,w)


def balanced_loss(logit,area,valid,ids):
    area=np.asarray(area)*ids;background=(np.asarray(valid)-np.asarray(area))*ids
    if min(area.sum(),background.sum())<=0:return float('inf')
    return float(np.sum(area*np.logaddexp(0,-logit))/area.sum()+np.sum(background*np.logaddexp(0,logit))/background.sum())


@dataclass
class Logistic:
    beta:np.ndarray
    mean:np.ndarray
    scale:np.ndarray
    info:dict
    def predict(self,x):
        z=np.clip((np.asarray(x)-self.mean)/self.scale,-5,5)
        return dot(z,self.beta[:-1])+self.beta[-1]


def fit_logistic(x,area,valid,ids,*,stats=None):
    x=np.asarray(x,float)
    if stats is None:
        median=np.median(x[valid>0],axis=0);mad=np.maximum(np.median(np.abs(x[valid>0]-median),axis=0),1e-3)
    else:median,mad=stats
    z=np.clip((x-median)/mad,-5,5);design=np.c_[z[ids],np.ones(int(np.count_nonzero(ids)))];a=area[ids];b=(valid-area)[ids]
    if min(a.sum(),b.sum())<=0:raise ValueError('Two reference roles needed for D_R')
    fw=a/a.sum();bw=b/b.sum();mass=fw+bw;beta=np.zeros(design.shape[1]);penalty=np.r_[np.ones(z.shape[1]),0.]
    def objective(coef):
        l=dot(design,coef)
        return float(np.sum(fw*np.logaddexp(0,-l)+bw*np.logaddexp(0,l))+.5*np.sum(penalty*coef**2))
    history=[objective(beta)];converged=False;stop='maximum_iterations'
    for iteration in range(25):
        p=expit(dot(design,beta));gradient=dot(design.T,mass*p-fw)+penalty*beta
        if np.linalg.norm(gradient)<1e-5:converged=True;stop='gradient';break
        hessian=dot(design.T,(mass*p*(1-p))[:,None]*design)+np.diag(penalty)+1e-6*np.eye(len(beta))
        step=np.linalg.solve(hessian,gradient);eta=1.;accepted=False
        for backtrack in range(20):
            proposal=beta-eta*step;value=objective(proposal)
            if value<=history[-1]-1e-4*eta*dot(gradient,step):accepted=True;break
            eta*=.5
        if not accepted:stop='Armijo_no_acceptable_step';break
        beta=proposal;history.append(value)
    return Logistic(beta,median,mad,dict(converged=converged,stop=stop,iterations=len(history)-1,objective=history,
                    optimization='current R only; bias unpenalized; L2 .5; Newton25/Armijo20'))


def residual_readout(r_enhanced,q_enhanced,r_simple,q_simple,area,valid,split):
    if not split.adequate:return np.zeros(len(q_enhanced)),dict(enabled=False,fallback='A/C_roles_or_blocks_insufficient',split=split.info)
    e=fit_logistic(r_enhanced,area,valid,split.anchor);s=fit_logistic(r_simple,area,valid,split.anchor)
    le=balanced_loss(e.predict(r_enhanced),area,valid,split.calibration);ls=balanced_loss(s.predict(r_simple),area,valid,split.calibration)
    enabled=ls-le>=.01
    info=dict(enabled=enabled,calibration_enhanced_loss=le,calibration_simple_loss=ls,minimum_improvement=.01,
              anchor_enhanced_solver=e.info,anchor_simple_solver=s.info,split=split.info,descriptors_frozen_at_anchor=True)
    if not enabled:return np.zeros(len(q_enhanced)),info
    e=fit_logistic(r_enhanced,area,valid,valid>0,stats=(e.mean,e.scale));s=fit_logistic(r_simple,area,valid,valid>0,stats=(s.mean,s.scale))
    info.update(full_reference_enhanced_solver=e.info,full_reference_simple_solver=s.info)
    return .20*np.tanh((e.predict(q_enhanced)-s.predict(q_simple))/2),info


def select_heads(gradients):
    gradients=np.asarray(gradients,float)
    if gradients.ndim!=3 or gradients.shape[0]!=2:raise ValueError('Two anchor-group gradients [2,4,H] required')
    candidates=np.argwhere(np.all(gradients>0,axis=0))
    ids=sorted((tuple(int(v) for v in x) for x in candidates),key=lambda t:(-float(gradients[:,t[0],t[1]].min()),t[0],t[1]))[:4]
    gates=np.ones(gradients.shape[1:]);
    for i,h in ids:gates[i,h]=.8
    return gates,[(i+21,h) for i,h in ids]


def area_calibration(margin,area,valid,split):
    target=np.divide(area,valid,out=np.zeros_like(area,float),where=valid>0)
    def fit(ids):
        t=target[ids];y=margin[ids];w=valid[ids];w=w/w.sum();tm=np.sum(w*t);ym=np.sum(w*y)
        slope=max(0.,float(np.sum(w*(t-tm)*(y-ym))/(np.sum(w*(t-tm)**2)+1e-3)))
        return float(ym-slope*tm),slope
    if not split.adequate:return None,dict(enabled=False,fallback='A/C_roles_or_blocks_insufficient')
    intercept,slope=fit(split.anchor);ids=split.calibration;v=valid[ids]
    mse=float(np.average((margin[ids]-intercept-slope*target[ids])**2,weights=v))
    constant=float(np.average(margin[split.anchor],weights=valid[split.anchor]));constant_mse=float(np.average((margin[ids]-constant)**2,weights=v))
    info=dict(anchor_beta=[intercept,slope],calibration_mse=mse,calibration_constant_mse=constant_mse,calibration_target='actual patch FG fraction',area_is_not_fraction=True)
    if slope<.02 or mse>=constant_mse:return None,dict(info,enabled=False,fallback='area_measurement_calibration_rejected')
    beta=fit(valid>0)
    return (beta if beta[1]>=.02 else None),dict(info,full_reference_beta=list(beta),enabled=beta[1]>=.02,fallback=None if beta[1]>=.02 else 'full_reference_area_slope_below_floor')


def phase_operators(hw=(128,128),native=(64,64),phase=(0,0),valid_box=(0.,0.,1024.,1024.)):
    """Exact fine-cell area averages; phase +8 uses original interval [-8,8]."""
    fine_h,fine_w=hw;ny,nx=native;oy,ox,eh,ew=valid_box;side=1024.
    axes=[];valids=[]
    for fine,n,shift,lo,hi in zip(hw,native,phase,(oy,ox),(eh,ew)):
        edges=np.arange(n+1)*side/n-shift;intervals=np.c_[np.maximum(edges[:-1],lo),np.minimum(edges[1:],hi)]
        # axis_overlap expects positive nominal width; zero support uses empty row.
        row=[];col=[];data=[];valid=[]
        for i,(a,b) in enumerate(intervals):
            width=max(0.,b-a);valid.append(width/(side/n))
            if width<=0:continue
            ids=np.arange(max(0,int(np.floor(a/(side/fine)))),min(fine,int(np.ceil(b/(side/fine)))))
            weights=np.maximum(0.,np.minimum((ids+1)*side/fine,b)-np.maximum(ids*side/fine,a))/width
            row.extend([i]*len(ids));col.extend(ids);data.extend(weights)
        axes.append(sparse.csr_matrix((data,(row,col)),shape=(n,fine)));valids.append(np.asarray(valid))
    return sparse.kron(axes[0],axes[1],format='csr'),(valids[0][:,None]*valids[1][None,:]).ravel()


def fine_edges(hw,active=None):
    ids=np.arange(np.prod(hw)).reshape(hw);edges=np.r_[np.c_[ids[:,:-1].ravel(),ids[:,1:].ravel()],np.c_[ids[:-1].ravel(),ids[1:].ravel()]]
    if active is not None:edges=edges[np.asarray(active)[edges].all(axis=1)]
    return edges


def tv(value,edges,epsilon=1e-3):
    if not len(edges):return 0.,np.zeros_like(value)
    diff=value[edges[:,0]]-value[edges[:,1]];norm=np.sqrt(diff**2+epsilon**2);flux=diff/norm/len(edges)
    gradient=np.bincount(edges[:,0],weights=flux,minlength=len(value))-np.bincount(edges[:,1],weights=flux,minlength=len(value))
    return float(norm.mean()),gradient


def inverse_mask(observations,operators,valids,D,base,initial,hw=(128,128),*,data=True,base_term=True):
    """Original M29 convex objective, scaled PG residual and actual certificates."""
    if len(observations)!=4 or len(operators)!=4 or len(valids)!=4:raise ValueError('Four real phases required')
    n=len(base);active=np.asarray(D.sum(axis=0)).ravel()>0;edges=fine_edges(hw,active)
    counts=[int(np.count_nonzero(v>0)) for v in valids]
    base_valid=np.asarray(D.sum(axis=1)).ravel()>0
    n=int(np.count_nonzero(base_valid))
    if not n or not len(edges) or min(counts)<=0:raise ValueError('Actual physical support required')
    def objective_gradient(u):
        energy=0.;gradient=np.zeros(len(u))
        if data:
            for a,op,w,count in zip(observations,operators,valids,counts):
                residual=op@u-a;energy+=.25*np.sum(w*residual**2)/count;gradient+=.5*(op.T@(w*residual))/count
        if base_term:
            residual=(D@u-base)*base_valid;energy+=.25*np.sum(residual**2)/n;gradient+=.5*(D.T@residual)/n
        smooth,g=tv(u,edges);return float(energy+.02*smooth),gradient+.02*g
    infnorm=lambda x:float(np.asarray(np.abs(x).sum(axis=1)).max())
    L=(sum(.5*infnorm(op.T@sparse.diags(w)@op)/count for op,w,count in zip(operators,valids,counts)) if data else 0.)
    L+=(.5*infnorm(D.T@D)/n if base_term else 0.)+.16/(len(edges)*1e-3)
    eta=1/max(L,1e-8);u=np.clip(np.asarray(initial,float).ravel(),0,1);u[~active]=0
    initial_energy,initial_gradient=objective_gradient(u);scale=max(float(np.max(np.abs(initial_gradient))),1e-8);history=[initial_energy];stable=0;stop='maximum_iterations';backtracks=0;residual=None
    for iteration in range(100):
        energy,gradient=objective_gradient(u);accepted=False
        for _ in range(12):
            proposal=np.clip(u-eta*gradient,0,1);proposal[~active]=0;step=proposal-u;next_energy,_=objective_gradient(proposal)
            if next_energy<=energy+dot(gradient,step)+dot(step,step)/(2*eta)+1e-12:accepted=True;break
            eta*=.5;backtracks+=1
        if not accepted:stop='backtracking_failed_kept_last_accepted';break
        residual=float(np.max(np.abs(step))/(eta*scale));u=proposal;history.append(next_energy)
        relative=abs(energy-next_energy)/max(abs(energy),1e-12);stable=stable+1 if relative<1e-6 else 0
        if residual<1e-3:stop='normalized_projected_gradient';break
        if stable>=3:stop='three_relative_objective_changes';break
    return u.reshape(hw),dict(iterations=len(history)-1,stop=stop,objective_history=history,initial_Lipschitz_bound=L,
        final_step=eta,normalized_projected_gradient_residual=residual,backtracks=backtracks,convex_descent_certificate=bool(np.all(np.diff(history)<=1e-10)),
        data_term_enabled=data,base_term_enabled=base_term,invalid_fine_cells_excluded_from_TV=True,quality_guarantee=False)


def expansion_potts(unary,graph,initial,*,rounds=10,scale_penalty=.02):
    """Float mincut alpha-expansion for the four prescribed metric states."""
    import networkx as nx
    unary=np.asarray(unary,float);labels=np.asarray(initial,int).copy();n=len(labels)
    edge=sparse.triu(graph,k=1).tocoo();u,v,w=edge.row,edge.col,edge.data
    states=np.array([[0,0],[0,1],[1,0],[1,1]])
    metric=.10*(states[:,None,0]!=states[None,:,0])+scale_penalty*(states[:,None,1]!=states[None,:,1])
    def energy(lab):return float(unary[np.arange(n),lab].sum()+np.sum(w*metric[lab[u],lab[v]]))
    history=[energy(labels)]
    for _ in range(rounds):
        before=history[-1]
        for alpha in range(4):
            old=labels.copy();cost0=unary[np.arange(n),old].copy();cost1=unary[:,alpha].copy()
            e00=w*metric[old[u],old[v]];e01=w*metric[old[u],alpha];e10=w*metric[alpha,old[v]];e11=np.zeros(len(w))
            capacity=(e01+e10-e00-e11)/2
            if np.any(capacity< -1e-10):raise ArithmeticError('Metric expansion is not submodular')
            capacity=np.maximum(capacity,0)
            np.add.at(cost1,u,e10-e00-capacity);np.add.at(cost1,v,e01-e00-capacity)
            offset=np.minimum(cost0,cost1);cost0-=offset;cost1-=offset
            network=nx.DiGraph();source=n;sink=n+1;network.add_nodes_from(range(n+2))
            for i in range(n):network.add_edge(source,i,capacity=float(cost1[i]));network.add_edge(i,sink,capacity=float(cost0[i]))
            for a,b,c in zip(u,v,capacity):
                if c>0:network.add_edge(int(a),int(b),capacity=float(c));network.add_edge(int(b),int(a),capacity=float(c))
            _,partition=nx.minimum_cut(network,source,sink,flow_func=nx.algorithms.flow.preflow_push)
            proposal=old.copy();switch=np.array([i in partition[1] for i in range(n)]);proposal[switch]=alpha;new=energy(proposal)
            if new<history[-1]-1e-12:labels=proposal;history.append(new)
        if history[-1]>=before-1e-12:break
    return labels,dict(energy_history=history,maximum_rounds=rounds,actual_float_capacity_mincut=True,alpha_order=[0,1,2,3],global_optimality_claim=False)
