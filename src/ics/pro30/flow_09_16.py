"""Source-defined Pro30 M15 partial local coordinate field, no forced match."""
from __future__ import annotations
import time
import numpy as np
from .common import unit
from .structures_09_16 import cached,edges4,prototype


def source_positions(ep):
    def positions(hw,geometry):
        yy,xx=np.indices(hw);coordinate=np.column_stack(((yy.ravel()+.5)/hw[0],(xx.ravel()+.5)/hw[1]))
        if geometry:
            view=float(geometry['view_side']);shape=np.array(geometry['resized_hw'],float);offset=np.array(geometry['padding_top_left'],float)
            coordinate=(coordinate*view-offset)/shape
        return coordinate
    r=positions(ep.r_hw,ep.reference_geometry);q=positions(ep.q_hw,ep.query_geometry)
    if ep.reference_mask is not None:
        y,x=np.nonzero(ep.reference_mask);shape=np.array(ep.reference_mask.shape,float)
        low=np.array([y.min(),x.min()])/shape;high=np.array([y.max()+1,x.max()+1])/shape
    else:
        support=np.flatnonzero(ep.wf>0);y,x=np.unravel_index(support,ep.r_hw);low=np.array([y.min()/ep.r_hw[0],x.min()/ep.r_hw[1]]);high=np.array([(y.max()+1)/ep.r_hw[0],(x.max()+1)/ep.r_hw[1]])
        if ep.reference_geometry:
            view=float(ep.reference_geometry['view_side']);shape=np.array(ep.reference_geometry['resized_hw'],float);offset=np.array(ep.reference_geometry['padding_top_left'],float);low=(low*view-offset)/shape;high=(high*view-offset)/shape
    return (r-low)/np.maximum(high-low,1e-12),q


def top_candidates(ep):
    def build():
        foreground=np.flatnonzero(ep.wf>0);background=np.flatnonzero(ep.wb>0);candidates=np.full((len(ep.q),9),-1,int);unary=np.full((len(ep.q),9),np.inf);unary[:,-1]=.6
        r=np.asarray(ep.r,np.float32);q=np.asarray(ep.q,np.float32)
        for start in range(0,len(q),256):
            stop=min(start+256,len(q));score=q[start:stop]@r.T
            for row,values in enumerate(score):
                ids=[]
                for bank in(foreground,background):
                    k=min(4,len(bank))
                    if k:
                        # Exact score plus stable source-row tie, no query GT.
                        order=np.lexsort((bank,-values[bank]))[:k];ids.extend(bank[order])
                ids=list(dict.fromkeys(map(int,ids)));candidates[start+row,:len(ids)]=ids;unary[start+row,:len(ids)]=1-values[ids]
        unary[ep.q_valid<=0]=np.inf;unary[ep.q_valid<=0,-1]=.6
        return candidates,unary
    return cached(ep,'M15_top4_per_soft_source_role',build)


def optimize(ep,mode=None):
    candidates,unary=top_candidates(ep);positions,coordinate=source_positions(ep)
    if mode=='shuffle_coords':positions=positions[np.random.default_rng(15).permutation(len(positions))]
    coverage=np.divide(ep.wf,ep.wvalid,out=np.zeros_like(ep.wf,float),where=ep.wvalid>0);i,j=edges4(ep.q_hw,ep.q_valid);dv=coordinate[j]-coordinate[i]
    yy,xx=np.indices(ep.q_hw);block=((yy//8)*int(np.ceil(ep.q_hw[1]/8))+xx//8).ravel();B=int(block.max()+1)
    neighbors=[[]for _ in ep.q]
    for edge,(a,b)in enumerate(zip(i,j)):neighbors[int(a)].append((int(b),edge,1));neighbors[int(b)].append((int(a),edge,-1))
    source_margin=prototype(ep);initial=np.argmin(unary,axis=1)
    if mode=='nearest':
        assignment=candidates[np.arange(len(ep.q)),initial];z=source_margin.copy();matched=assignment>=0;z[matched]=2*coverage[assignment[matched]]-1;z[ep.q_valid<=0]=-1
        return z,{'control':'same8 source candidates plus .6 reject, no coordinate interaction','reject_fraction':float(np.mean((~matched)[ep.q_valid>0]))}
    block_edges={}
    for a,b in zip(block[i],block[j]):
        if a!=b:key=tuple(sorted((int(a),int(b))));block_edges[key]=block_edges.get(key,0)+1
    def pair(a,b,edge,J):
        ra,rb=candidates[a,assignment[a]],candidates[b,assignment[b]]
        if ra<0 or rb<0:return .05
        if mode=='label_potts':return .2*float((coverage[ra]>.5)!=(coverage[rb]>.5))
        if mode=='same_role'and(coverage[ra]>.5)!=(coverage[rb]>.5):return 0.
        dp=positions[rb]-positions[ra];delta=coordinate[b]-coordinate[a]
        return .1*(min(float(np.sum((dp-J[block[a]]@delta)**2)),.1)+min(float(np.sum((dp-J[block[b]]@delta)**2)),.1))
    position_pad=np.r_[positions,np.zeros((1,2))];coverage_pad=np.r_[coverage,0.]
    def energy(J):
        selected=candidates[np.arange(len(ep.q)),assignment];valid_pair=(selected[i]>=0)&(selected[j]>=0)
        value=float(unary[np.flatnonzero(ep.q_valid>0),assignment[ep.q_valid>0]].sum())
        if mode=='label_potts':
            costs=.2*((coverage_pad[selected[i]]>.5)!=(coverage_pad[selected[j]]>.5)).astype(float)
        else:
            delta=position_pad[selected[j]]-position_pad[selected[i]]
            predicted_a=np.einsum('nij,nj->ni',J[block[i]],dv);predicted_b=np.einsum('nij,nj->ni',J[block[j]],dv)
            costs=.1*(np.minimum(np.sum((delta-predicted_a)**2,axis=1),.1)+np.minimum(np.sum((delta-predicted_b)**2,axis=1),.1))
            if mode=='same_role':costs[(coverage_pad[selected[i]]>.5)!=(coverage_pad[selected[j]]>.5)]=0.
        costs[~valid_pair]=.05;value+=float(costs.sum())
        if mode!='label_potts':value+=.05*sum(count*np.sum((J[a]-J[b])**2)for(a,b),count in block_edges.items())
        return float(value)
    results=[]
    for reflection in(False,True):
        J=np.tile(np.diag([1.,-1. if reflection else 1.]),(B,1,1));assignment=initial.copy();history=[energy(J)];changes=[]
        for iteration in range(5):
            changed=0
            for a in np.flatnonzero(ep.q_valid>0):
                old=int(assignment[a]);around=np.array([b for b,e,sign in neighbors[a]],int)
                cost=unary[a].copy();references=candidates[a]
                if len(around):
                    other=candidates[around,assignment[around]];delta=coordinate[around]-coordinate[a]
                    difference=position_pad[other][None]-position_pad[references][:,None]
                    if mode=='label_potts':
                        interactions=.2*((coverage_pad[references,None]>.5)!=(coverage_pad[other][None]>.5)).astype(float)
                    else:
                        left=delta@J[block[a]].T;right=np.einsum('nij,nj->ni',J[block[around]],delta)
                        interactions=.1*(np.minimum(np.sum((difference-left[None])**2,axis=-1),.1)+np.minimum(np.sum((difference-right[None])**2,axis=-1),.1))
                        if mode=='same_role':interactions[(coverage_pad[references,None]>.5)!=(coverage_pad[other][None]>.5)]=0.
                    interactions[(references[:,None]<0)|(other[None]<0)]=.05;cost+=interactions.sum(axis=1)
                assignment[a]=int(np.argmin(cost));changed+=int(assignment[a]!=old)
            if mode!='label_potts':
                # A block ridge fit uses all current non-reject directed pairs;
                # it is a fixed surrogate update, no monotonicity claim after SVD.
                selected=candidates[np.arange(len(ep.q)),assignment];active=(selected[i]>=0)&(selected[j]>=0)
                if mode=='same_role':active&=(coverage_pad[selected[i]]>.5)==(coverage_pad[selected[j]]>.5)
                dp=position_pad[selected[j]]-position_pad[selected[i]]
                for b in range(B):
                    left=active&(block[i]==b);right=active&(block[j]==b)
                    X=np.r_[dv[left],-dv[right]];Y=np.r_[dp[left],-dp[right]]
                    matrix=np.eye(2)*1e-6;rhs=np.eye(2)*1e-6
                    if len(X):
                        matrix+=.1*(X.T@X);rhs+=.1*(Y.T@X)
                    for(a,c),count in block_edges.items():
                        if b not in(a,c):continue
                        other=c if a==b else a;matrix+=.05*count*np.eye(2);rhs+=.05*count*J[other]
                    fitted=np.linalg.solve(matrix.T,rhs.T).T;U,s,V=np.linalg.svd(fitted);J[b]=(U*np.clip(s,.25,4))@V
            history.append(energy(J));changes.append(changed)
        mapped=candidates[np.arange(len(ep.q)),assignment];results.append((history[-1],reflection,mapped.copy(),J.copy(),history,changes))
    E,reflection,mapped,J,history,changes=min(results,key=lambda value:(value[0],value[1]));z=source_margin.copy();matched=mapped>=0;z[matched]=2*coverage[mapped[matched]]-1;z[ep.q_valid<=0]=-1
    return z,{'final_complete_energy':E,'chosen_initialization':'horizontal_flip'if reflection else'identity','all_initialization_energies':[r[0]for r in results],'energy_history':history,'assignment_changes':changes,'source_position_ids':mapped.tolist(),'reject_fraction':float(np.mean((~matched)[ep.q_valid>0])),'Jacobian_blocks':J.tolist(),'Jacobian_singular_value_minmax':[float(np.linalg.svd(J,compute_uv=False).min()),float(np.linalg.svd(J,compute_uv=False).max())],'surrogate_J_update_can_increase_original_truncated_energy':True,'control':mode}
