"""Exact upper-level-set event tree shared by original C133/C136.

Plateaus enter jointly. Only the first binary merge on a multiple-child
plateau has its newly added bridge; subsequent binary unions have no invented
bridge potential. Every physical vertex occurs once in a bridge leaf sequence.
"""
import numpy as np
from .group_076_150_common import edges4


def max_tree(ctx):
    n=len(ctx.x);parent=np.arange(n);active=np.zeros(n,bool);support={};rootnode={};nodes=[]
    i,j=edges4(ctx.hw,ctx.valid);adj=[[]for _ in range(n)]
    for a,b in zip(i,j):adj[int(a)].append(int(b));adj[int(b)].append(int(a))
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=int(parent[a])
        return a
    def append(children,bridge,event=False):
        parts=[nodes[c]['support']for c in children]+[bridge]
        ids=np.sort(np.concatenate(parts))if parts else np.empty(0,int)
        nodes.append({'support':ids,'children':tuple(children),'bridge':np.asarray(bridge,int),'event':event})
        return len(nodes)-1
    order=np.flatnonzero(ctx.valid>0);order=order[np.argsort(-ctx.u[order],kind='stable')];start=0
    while start<len(order):
        stop=start+1
        while stop<len(order)and ctx.u[order[stop]]==ctx.u[order[start]]:stop+=1
        new=order[start:stop];old={}
        for a in new:
            for b in adj[int(a)]:
                if active[b]:r=find(b);old[r]=(support[r].copy(),rootnode[r])
        for a in new:active[a]=True;support[int(a)]=np.array([a],int)
        for a in new:
            for b in adj[int(a)]:
                if not active[b]:continue
                left,right=find(int(a)),find(b)
                if left==right:continue
                if left>right:left,right=right,left
                support[left]=np.sort(np.r_[support.pop(left),support.pop(right)]);parent[right]=left
        groups={}
        for a in new:groups.setdefault(find(int(a)),[]).append(int(a))
        for r,bridge in sorted(groups.items()):
            children=sorted((node for ids,node in old.values()if find(int(ids[0]))==r),key=lambda c:int(nodes[c]['support'].min()))
            bridge=np.array(sorted(bridge),int)
            if len(children)<2:node=append(children,bridge)
            else:
                node=append(children[:2],bridge,True)
                for child in children[2:]:node=append((node,child),np.empty(0,int),False)
            rootnode[r]=node
        start=stop
    roots=sorted((rootnode[r]for r in support),key=lambda c:int(nodes[c]['support'].min()))
    return nodes,roots


def bridge_states(ctx,ids):
    """Energy, S0-edit count and exact labels for BG/FG/MIX bridge states."""
    ids=np.asarray(ids,int);u=ctx.u[ids];target=u>0
    labels=[np.zeros(len(ids),bool),np.ones(len(ids),bool),None]
    if len(ids)>=2:
        mixed=target.copy()
        if mixed.all()or not mixed.any():
            cost=np.abs(u);at=min(range(len(ids)),key=lambda k:(cost[k],int(ids[k])if mixed.all()else-int(ids[k])))
            mixed[at]=~mixed[at]
        labels[2]=mixed
    output=[]
    for y in labels:
        output.append((np.inf,0,None)if y is None else(float(-u@y),int(np.sum(y!=target)),y))
    if not len(ids):output=[(0.,0,np.empty(0,bool)),(np.inf,0,None),(np.inf,0,None)]
    return output


def compatible(states,nonempty):
    roles=[state for state,keep in zip(states,nonempty)if keep]
    if not roles:return 0
    if all(r==0 for r in roles):return 0
    if all(r==1 for r in roles):return 1
    return 2


def decode(ctx,nodes,roots,event_costs=None):
    event_costs={}if event_costs is None else event_costs
    tables=[];pointers=[];bridges=[]
    def recover(index,state):
        node=nodes[index];choice=pointers[index][state]
        if choice is None:return {}
        selected={int(i):bool(y)for i,y in zip(node['bridge'],bridges[index][choice[-1]][2])}
        for child,s in zip(node['children'],choice[:-1]):selected.update(recover(child,s))
        return selected
    for index,node in enumerate(nodes):
        bridge=bridge_states(ctx,node['bridge']);bridges.append(bridge);table=[(np.inf,0)]*3;pointer=[None]*3
        dimensions=[range(3)]*(len(node['children'])+1)
        import itertools
        for states in itertools.product(*dimensions):
            parts=[tables[c][s]for c,s in zip(node['children'],states[:-1])]+[(bridge[states[-1]][0],bridge[states[-1]][1])]
            if not all(np.isfinite(a)for a,b in parts):continue
            state=compatible(states,[True]*len(node['children'])+[len(node['bridge'])>0])
            energy=sum(a for a,b in parts);edit=sum(b for a,b in parts)
            if index in event_costs:energy+=event_costs[index][states]
            key=(energy,edit)
            if key<table[state]:table[state]=key;pointer[state]=states
            elif key==table[state]and pointer[state]is not None:
                def rows(choice):
                    y={int(i):bool(v)for i,v in zip(node['bridge'],bridge[choice[-1]][2])}
                    for c,s in zip(node['children'],choice[:-1]):y.update(recover(c,s))
                    return tuple(y[int(i)]for i in node['support'])
                if rows(states)<rows(pointer[state]):pointer[state]=states
        tables.append(table);pointers.append(pointer)
    field=np.zeros(len(ctx.x),bool)
    for root in roots:
        selected=min(range(3),key=lambda s:(*tables[root][s],tuple(recover(root,s).get(int(i),False)for i in nodes[root]['support'])))
        for i,y in recover(root,selected).items():field[i]=y
    return field,{'nodes':len(nodes),'roots':len(roots),'events':len(event_costs),'energy':sum(min(tables[r])[0]for r in roots)}
