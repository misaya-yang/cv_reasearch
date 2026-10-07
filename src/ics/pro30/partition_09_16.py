"""Pro30 M14 exact stated finite partition/role moves and actual history splits."""
import numpy as np
from .structures_09_16 import cached,region_graph,prototype


def edge_features(x,edges,P):
    output=[]
    for start in range(0,len(edges),256):
        a,b=edges[start:start+256].T;left=x[a];right=x[b]
        feature=np.c_[np.abs(left-right),left*right,(left+right)/2]
        output.append(np.c_[feature@P,np.einsum('nd,nd->n',left,right)])
    return np.concatenate(output)if output else np.empty((0,P.shape[1]+1))


def data(ep):
    def build():
        R,rx,re,ro=region_graph(ep,'r',512);Q,qx,qe,qo=region_graph(ep,'q',512)
        D=rx.shape[1];K=min(64,3*D);P=np.linalg.qr(np.random.default_rng(14).normal(size=(3*D,K)))[0][:,:K]
        x=edge_features(rx,re,P);y=edge_features(qx,qe,P);coverage=np.array([ep.wf[C].sum()/ep.wvalid[C].sum()for C in R]);a,b=re.T
        soft=np.c_[coverage[a]*coverage[b],coverage[a]*(1-coverage[b])+(1-coverage[a])*coverage[b],(1-coverage[a])*(1-coverage[b])]
        role_mass=soft.sum(axis=0);normalized=np.divide(soft,role_mass,out=np.zeros_like(soft),where=role_mass>0)/3
        weight=normalized.sum(axis=1);X=np.c_[x,np.ones(len(x))];Y=normalized;gram=X.T@(X*weight[:,None]);ridge=max(float(np.trace(gram[:-1,:-1]))/max(len(gram)-1,1),1e-12);penalty=np.eye(len(gram))*ridge;penalty[-1,-1]=0
        coef=np.linalg.solve(gram+penalty,X.T@Y);logits=np.c_[y,np.ones(len(y))]@coef;logits-=logits.max(axis=1,keepdims=True);probability=np.exp(logits);probability/=probability.sum(axis=1,keepdims=True)
        c=np.clip(np.log((probability[:,0]+probability[:,2]+1e-8)/(probability[:,1]+1e-8)),-3,3);inside=np.c_[-np.log(probability[:,2]+1e-8),-np.log(probability[:,0]+1e-8)]
        area=np.array([ep.q_valid[C].sum()for C in Q]);normalized_area=area/area.sum()*512;u=prototype(ep);mu=np.array([np.average(u[C],weights=ep.q_valid[C])for C in Q]);edge_scale=512/max(len(qe),1)
        return Q,qe,normalized_area*mu,.25*edge_scale*c,.1*edge_scale*inside,{'R_regions':len(R),'Q_regions':len(Q),'R_edge_role_mass':role_mass.tolist(),'ridge':ridge,'feature_projection_dimension':K,'query_edges':len(qe),'source_BG_instance_labels_used':False}
    return cached(ep,'M14_three_state_edge_data',build)


def execute(ep,binary=False,unlabeled=False):
    Q,qx,qe,qo=region_graph(ep,'q',512);R,rx,re,ro=region_graph(ep,'r',512)
    if len(Q)<2 or len(R)<2:return prototype(ep),{'inactive_reason':'fewer than2 legal reference/query regions'}
    regions,edges,U,cut,inside,info=data(ep);N=len(regions)
    if N<2:return prototype(ep),dict(info,inactive_reason='fewer than2 legal query regions')
    a,b=edges.T
    if binary:
        labels=U>0
        neighbors=[[]for _ in range(N)]
        for e,(i,j)in enumerate(edges):neighbors[int(i)].append((int(j),e));neighbors[int(j)].append((int(i),e))
        for iteration in range(20):
            changed=0
            for i in range(N):
                costs=[]
                for role in(0,1):
                    cost=-U[i]*(2*role-1)
                    for j,e in neighbors[i]:cost+=inside[e,role]if labels[j]==role else cut[e]
                    costs.append(cost)
                new=int(costs[1]<costs[0]);changed+=int(new!=labels[i]);labels[i]=new
            if not changed:break
        margin=np.zeros(N)
        for i in range(N):
            bg=-U[i]*-1;fg=-U[i]
            for j,e in neighbors[i]:bg+=inside[e,0]if not labels[j]else cut[e];fg+=inside[e,1]if labels[j]else cut[e]
            margin[i]=bg-fg
        z=np.full(len(ep.q),-1.)
        for C,score in zip(regions,margin):z[C]=score
        return z,dict(info,control='same unary and three-state edge observations, binary roles without partition',iterations=iteration+1)
    nodes=[{'members':np.array([i],int),'U':float(U[i]),'inside':np.zeros(2),'children':()}for i in range(N)]
    active=set(range(N));labels={i:int(U[i]>0)for i in active};owner=np.arange(N);history=[];moves=[]
    def role_cost(node,role):return -node['U']*(2*role-1)+node['inside'][role]
    def full_energy():
        same=owner[a]==owner[b];value=sum(role_cost(nodes[i],labels[i])for i in active)+cut[~same].sum()+.01*len(active)
        return float(value)
    history.append(full_energy())
    for iteration in range(20):
        changed=False
        for i in sorted(active):
            if unlabeled:continue
            costs=[role_cost(nodes[i],role)for role in(0,1)];best=0 if costs[0]<costs[1]else 1 if costs[1]<costs[0]else labels[i]
            if costs[best]<costs[labels[i]]-1e-10:labels[i]=best;changed=True;moves.append({'kind':'role','group':i,'role':best})
        cross={}
        for e,(left,right)in enumerate(zip(owner[a],owner[b])):
            if left==right:continue
            key=tuple(sorted((int(left),int(right))));value=cross.setdefault(key,[0.,np.zeros(2)]);value[0]+=cut[e];value[1]+=inside[e]
        proposals=[]
        for (left,right),(cut_sum,inside_sum)in cross.items():
            combined={'members':np.sort(np.r_[nodes[left]['members'],nodes[right]['members']]),'U':nodes[left]['U']+nodes[right]['U'],'inside':nodes[left]['inside']+nodes[right]['inside']+inside_sum,'children':(left,right),'split_cut':cut_sum}
            for role in(0,1):
                delta=role_cost(combined,role)-role_cost(nodes[left],labels[left])-role_cost(nodes[right],labels[right])-cut_sum-.01
                if unlabeled:delta=-cut_sum-.01
                if delta<-1e-10:proposals.append((float(delta),left,right,role,combined))
        if proposals:
            delta,left,right,role,new=min(proposals,key=lambda x:x[:4]);index=len(nodes);nodes.append(new);active.remove(left);active.remove(right);active.add(index);labels[index]=role;owner[new['members']]=index;changed=True;moves.append({'kind':'merge','actual_children':[left,right],'new_group':index,'role':role,'delta':delta})
        for i in sorted(list(active)):
            node=nodes[i]
            if not node['children']:continue
            left,right=node['children'];best=None
            for l,r in((0,0),(0,1),(1,0),(1,1)):
                delta=node['split_cut']+.01 if unlabeled else role_cost(nodes[left],l)+role_cost(nodes[right],r)+node['split_cut']+.01-role_cost(node,labels[i])
                if delta<-1e-10 and(best is None or(delta,l,r)<best):best=(float(delta),l,r)
            if best is not None:
                delta,l,r=best;active.remove(i);active.update((left,right));labels[left]=l;labels[right]=r;owner[nodes[left]['members']]=left;owner[nodes[right]['members']]=right;changed=True;moves.append({'kind':'split_actual_history','old_group':i,'children':[left,right],'roles':[l,r],'delta':delta})
        current=full_energy();history.append(current)
        if not unlabeled and current>history[-2]+1e-7:raise ArithmeticError('M14 exact finite move increased full partition/role energy')
        if not changed:break
    z=np.full(len(ep.q),-1.);groups=[]
    for i in sorted(active):
        score=role_cost(nodes[i],0)-role_cost(nodes[i],1)
        for r in nodes[i]['members']:z[regions[r]]=score
        groups.append({'id':i,'member_region_ids':nodes[i]['members'].tolist(),'role':int(score>0),'conditional_margin':float(score),'actual_children':list(nodes[i]['children'])})
    return z,dict(info,complete_energy_history=history,partition_moves=moves,final_groups=groups,control='unlabeled_partition_then_role'if unlabeled else None)
