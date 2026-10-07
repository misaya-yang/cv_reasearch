"""Pro30 M11 original finite partial relational transport and feasible caps."""
import numpy as np
from scipy.special import logsumexp
from .structures_09_16 import weighted_atoms,tree,cached,prototype


def role_scales(reference):
    distance=np.maximum(1-reference@reference.T,0);off=distance[~np.eye(len(reference),dtype=bool)]
    appearance=max(float(np.median(off))if len(off)else 0.,1e-6)
    relation=max(float(np.median(off**2))if len(off)else 0.,1e-6)
    return appearance,relation,1.,distance


def capped_entropy(cost,b,a,epsilon=.03,no_columns=False):
    logT=-cost/epsilon;qr=np.zeros(len(b));qc=np.zeros(len(a));qm=0.;tiny=1e-300
    logb=np.log(np.maximum(b,tiny));loga=np.log(np.maximum(2*a,tiny))
    for _ in range(30):
        before=logT+qr[:,None];difference=np.maximum(logsumexp(before,axis=1)-logb,0);logT=before-difference[:,None];qr=difference
        if not no_columns:
            before=logT+qc[None];difference=np.maximum(logsumexp(before,axis=0)-loga,0);logT=before-difference[None];qc=difference
        before=logT+qm;shift=max(np.log(.5)-float(logsumexp(before)),0.);logT=before+shift;qm=-shift
    T=np.exp(logT);raw=T.copy()
    T*=np.minimum(1,b/np.maximum(T.sum(axis=1),tiny))[:,None]
    if not no_columns:T*=np.minimum(1,2*a/np.maximum(T.sum(axis=0),tiny))[None]
    mass=T.sum()
    if mass<.5:
        coefficient=(.5-mass)/max(1-mass,tiny);T=(1-coefficient)*T+coefficient*np.outer(b,a)
    residual=max(float(np.maximum(T.sum(axis=1)-b,0).max()),float(np.maximum(T.sum(axis=0)-2*a,0).max())if not no_columns else 0.,max(.5-float(T.sum()),0.))
    if not np.isfinite(T).all()or residual>1e-8:raise ArithmeticError('M11 finite capped OT failed feasible row/column/mass constraints')
    return T,{'raw30update_max_violation':max(float(np.maximum(raw.sum(axis=1)-b,0).max()),float(np.maximum(raw.sum(axis=0)-2*a,0).max())if not no_columns else 0.),'final_feasibility_residual':residual,'numerical_feasibility_repair_L1':float(np.sum(np.abs(T-raw)))}


def transport(query,b,reference,a,appearance_only=False,no_columns=False):
    app_scale,rel_scale,reject_scale,Dr=role_scales(reference);Dq=np.maximum(1-query@query.T,0);app=np.maximum(1-query@reference.T,0)/app_scale
    T=np.outer(b,a);history=[];lambda_=.0 if appearance_only else .25;receipt=None
    def relation(T):
        row=T.sum(axis=1);col=T.sum(axis=0)
        return max(float(row@(Dq**2)@row+col@(Dr**2)@col-2*np.sum((Dq@T@Dr.T)*T)),0.)/rel_scale
    for _ in range(10):
        row=T.sum(axis=1);col=T.sum(axis=0)
        gradient=2*((Dq**2@row)[:,None]+(Dr**2@col)[None]-2*Dq@T@Dr.T)/rel_scale
        T,receipt=capped_entropy(app+lambda_*gradient-.6/reject_scale,b,a,no_columns=no_columns)
        entropy=.03*np.sum(T*(np.log(np.maximum(T,1e-300))-1));appearance=float(np.sum(app*T));relationship=lambda_*relation(T);reject=.6*(1-T.sum())/reject_scale
        history.append(float(appearance+relationship+reject+entropy))
    mass=float(T.sum());appearance=float(np.sum(app*T));relationship=lambda_*relation(T);reject=.6*(1-mass)/reject_scale
    return (appearance+relationship+reject)/mass,{'T':T.tolist(),'matched_mass':mass,'rejected_mass':1-mass,'appearance':appearance,'relationship':relationship,'reject':reject,'entropy_omitted_from_final_cost':True,'linearization_objective_history':history,'scales':[app_scale,rel_scale,reject_scale],**receipt}


def execute(ep,control=None):
    F=cached(ep,'M11_reference_F_atoms',lambda:weighted_atoms(ep.r,ep.wf,32,5));B=cached(ep,'M11_reference_B_atoms',lambda:weighted_atoms(ep.r,ep.wb,32,5));nodes,leaves,roots=tree(ep);u=prototype(ep)
    if len(leaves)<2:return u,{'inactive_reason':'fewer than2 valid query nodes'}
    threshold_samples=[]
    from .common import blocks
    partition=blocks(ep.r_hw)
    if all(np.any((partition==fold)&(ep.wf>0))for fold in range(4)):
        for fold in range(4):
            held=ep.wf*(partition==fold);bank=ep.wf*(partition!=fold);q,b,_=weighted_atoms(ep.r,held,32,5);r,a,_=weighted_atoms(ep.r,bank,32,5)
            d,receipt=transport(q,b,r,a,control=='appearance_only',control=='no_column_capacity');threshold_samples.append(d)
        tF=float(np.quantile(threshold_samples,.9))
    else:tF=.6
    values=[];audits=[]
    for k,node in enumerate(nodes):
        C=node['support'];weights=np.zeros(len(ep.q));weights[C]=ep.q_valid[C];q,b,_=weighted_atoms(ep.q,weights,32,5)
        foreground,fr=transport(q,b,F[0],F[1],control=='appearance_only',control=='no_column_capacity');background,br=transport(q,b,B[0],B[1],control=='appearance_only',control=='no_column_capacity')
        value=min(background-foreground,tF-foreground);values.append(value);audits.append({'node':k,'native_support_count':len(C),'v':value,'F_cost':foreground,'B_cost':background,'F_transport':fr,'B_transport':br})
    gain=np.zeros(len(nodes));choice=[]
    for k,node in enumerate(nodes):
        C=node['support'];area=len(C)
        if not node['children']:
            foreground=area*(values[k]+.25*np.average(u[C],weights=ep.q_valid[C]));gain[k]=max(0.,foreground);choice.append('FG'if foreground>0 else'BG')
        else:
            left,right=node['children'];options=[(0.,'BG'),(gain[left]+gain[right],'SPLIT'),(area*values[k],'FG')]
            best=max(range(3),key=lambda n:(options[n][0],-n));gain[k]=options[best][0];choice.append(options[best][1])
    z=np.full(len(ep.q),-1.)
    def trace(k):
        if choice[k]=='FG':z[nodes[k]['support']]=1
        elif choice[k]=='SPLIT':
            for child in nodes[k]['children']:trace(child)
    for root in roots:trace(root)
    return z,{'nodes':len(nodes),'reference_F_heldout_costs':threshold_samples,'tF':tF,'DP_choice':choice,'DP_gain':gain.tolist(),'node_costs':audits,'control':control}
