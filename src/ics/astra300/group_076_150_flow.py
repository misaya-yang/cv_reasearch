"""Small exact-capacity CPU graph primitives; no label-derived graph partition."""
from __future__ import annotations
from collections import deque
import heapq
import numpy as np


class Dinic:
    def __init__(self,n):
        self.g=[[] for _ in range(n)]
    def add(self,u,v,c):
        if c<0 or not np.isfinite(c):raise ValueError('finite nonnegative capacity required')
        a=[int(v),len(self.g[v]),float(c)];b=[int(u),len(self.g[u]),0.]
        self.g[u].append(a);self.g[v].append(b)
    def solve(self,s,t):
        n=len(self.g);flow=0.
        # Iterative augmenting paths avoid Python recursion depth on long grids.
        while True:
            level=[-1]*n;level[s]=0;queue=deque([s])
            while queue:
                u=queue.popleft()
                for v,_,cap in self.g[u]:
                    if cap>1e-12 and level[v]<0:level[v]=level[u]+1;queue.append(v)
            if level[t]<0:break
            pointer=[0]*n
            while True:
                nodes=[s];path=[];bottleneck=[float('inf')]
                while nodes and nodes[-1]!=t:
                    u=nodes[-1]
                    while pointer[u]<len(self.g[u]):
                        edge=self.g[u][pointer[u]]
                        if edge[2]>1e-12 and level[edge[0]]==level[u]+1:break
                        pointer[u]+=1
                    if pointer[u]==len(self.g[u]):
                        level[u]=-1;nodes.pop();bottleneck.pop()
                        if path:
                            previous,index=path.pop();pointer[previous]=index+1
                    else:
                        e=self.g[u][pointer[u]];path.append((u,pointer[u]));nodes.append(e[0]);bottleneck.append(min(bottleneck[-1],e[2]))
                if not nodes:break
                amount=bottleneck[-1];flow+=amount
                for u,index in path:
                    edge=self.g[u][index];edge[2]-=amount;self.g[edge[0]][edge[1]][2]+=amount
        reachable=np.zeros(n,bool);reachable[s]=True;queue=deque([s])
        while queue:
            u=queue.popleft()
            for v,_,cap in self.g[u]:
                if cap>1e-12 and not reachable[v]:reachable[v]=True;queue.append(v)
        return flow,reachable


def cut(unary,W,source_weights=None,sink_weights=None):
    unary=np.asarray(unary,float);n=len(unary);source=n;sink=n+1;solver=Dinic(n+2)
    # Source side=FG: positive u gives capacity source→i, negative i→sink.
    sp=np.maximum(unary,0);bp=np.maximum(-unary,0)
    if source_weights is not None:sp=sp+np.asarray(source_weights,float)
    if sink_weights is not None:bp=bp+np.asarray(sink_weights,float)
    for i in range(n):
        if sp[i]>0:solver.add(source,i,sp[i])
        if bp[i]>0:solver.add(i,sink,bp[i])
    coo=W.tocoo()
    for i,j,w in zip(coo.row,coo.col,coo.data):
        if i!=j and w>0:solver.add(int(i),int(j),float(w))
    value,reachable=solver.solve(source,sink)
    return reachable[:n],float(value-np.maximum(unary,0).sum()),solver


def persistent(solver,source,sink,n):
    forward=np.zeros(len(solver.g),bool);forward[source]=True;queue=deque([source])
    reverse=[[] for _ in solver.g]
    for i,edges in enumerate(solver.g):
        for j,_,capacity in edges:
            if capacity>1e-12:reverse[j].append(i)
    while queue:
        i=queue.popleft()
        for j,_,capacity in solver.g[i]:
            if capacity>1e-12 and not forward[j]:forward[j]=True;queue.append(j)
    backward=np.zeros(len(solver.g),bool);backward[sink]=True;queue=deque([sink])
    while queue:
        i=queue.popleft()
        for j in reverse[i]:
            if not backward[j]:backward[j]=True;queue.append(j)
    return forward[:n],backward[:n],~(forward[:n]|backward[:n])


class MinCostFlow:
    def __init__(self,n):self.g=[[] for _ in range(n)]
    def add(self,u,v,capacity,cost):
        a=[int(v),len(self.g[v]),int(capacity),float(cost)]
        b=[int(u),len(self.g[u]),0,-float(cost)]
        self.g[u].append(a);self.g[v].append(b)
    def solve(self,source,sink,amount):
        n=len(self.g);potential=np.zeros(n);flow=0;cost=0.
        while flow<amount:
            distance=np.full(n,np.inf);distance[source]=0;previous=[None]*n;queue=[(0.,source)]
            while queue:
                d,u=heapq.heappop(queue)
                if d!=distance[u]:continue
                for k,(v,_,capacity,weight) in enumerate(self.g[u]):
                    if capacity<=0:continue
                    value=d+weight+potential[u]-potential[v]
                    if value<distance[v]-1e-12:
                        distance[v]=value;previous[v]=(u,k);heapq.heappush(queue,(value,v))
            if not np.isfinite(distance[sink]):break
            known=np.isfinite(distance);potential[known]+=distance[known]
            v=sink
            while v!=source:
                u,k=previous[v];edge=self.g[u][k];edge[2]-=1;self.g[v][edge[1]][2]+=1;cost+=edge[3];v=u
            flow+=1
        return flow,cost
