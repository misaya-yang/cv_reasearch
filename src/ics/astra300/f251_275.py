"""Astra source cards F251--F275: independent, bounded CPU kernels.

The public adapter is below the kernels. Features remain native FP32; no query
labels, downloads, model construction, or historical result caches are used.
Resource-dependent cards refuse missing real observations instead of simulating
them. The finite coordinate solver reports trials and is never called exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
import heapq
import math

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment, lsq_linear
from scipy.special import expit, logsumexp

EPS = 1e-6


def _unit(a):
    a = np.asarray(a, np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), EPS)


def _mad(a):
    a = np.asarray(a, np.float32)
    return max(float(np.median(np.abs(a - np.median(a)))) if a.size else 0., EPS)


def _components(mask, hw, minimum=1):
    labels, count = ndimage.label(np.asarray(mask, bool).reshape(hw))
    return [np.flatnonzero(labels.ravel() == i) for i in range(1, count + 1)
            if np.count_nonzero(labels == i) >= minimum]


def _adjacency(n, edges, costs=None):
    result = [[] for _ in range(n)]
    if costs is None:
        costs = np.ones(len(edges), np.float32)
    for (a, b), cost in zip(edges, costs):
        result[int(a)].append((int(b), float(cost)))
        result[int(b)].append((int(a), float(cost)))
    for row in result:
        row.sort()
    return result


def _shortest(adjacency, starts, targets=None, forbidden_edges=frozenset(), bottleneck=False):
    n = len(adjacency)
    distance = np.full(n, np.inf)
    previous = np.full(n, -1, int)
    queue = []
    for i in sorted(set(map(int, starts))):
        distance[i] = 0.
        heapq.heappush(queue, (0., i))
    target_set = None if targets is None else set(map(int, targets))
    finish = None
    while queue:
        dist, a = heapq.heappop(queue)
        if dist != distance[a]:
            continue
        if target_set is not None and a in target_set:
            finish = a
            break
        for b, cost in adjacency[a]:
            if tuple(sorted((a, b))) in forbidden_edges:
                continue
            new = max(dist, cost) if bottleneck else dist + cost
            if new < distance[b] - 1e-12:
                distance[b], previous[b] = new, a
                heapq.heappush(queue, (new, b))
    path = []
    if finish is not None:
        path = [finish]
        while previous[path[-1]] >= 0:
            path.append(int(previous[path[-1]]))
        path.reverse()
    return distance, path


def _peaks(s, hw):
    field = np.asarray(s).reshape(hw)
    maximum = ndimage.maximum_filter(field, footprint=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), mode="constant", cval=-np.inf)
    # Plateaus contribute their row-first point, not N identical facilities.
    return np.array([int(c[0]) for c in _components((field > 0) & (field == maximum), hw)], int)


def finite_search(objective, starts, domains, pairs=(), rounds=10, tie_key=None):
    """Source S: complete-objective single and local pair trials, <=10 rounds."""
    domains = [tuple(d) for d in domains]
    supplied=sorted(set(tuple(sorted(map(int,pair))) for pair in pairs if pair[0]!=pair[1]))
    paired={};pairs=[]
    for a,b in supplied:
        if paired.get(a,0)<4:pairs.append((a,b));paired[a]=paired.get(a,0)+1
    trials = 0
    outputs = []
    for start in starts:
        x = np.asarray(start, int).copy()
        value = float(objective(x)); trials += 1
        trace = [value]
        if not np.isfinite(value):
            continue
        for _ in range(rounds):
            changed = False
            for i, domain in enumerate(domains):
                for state in domain:
                    if state == x[i]:
                        continue
                    candidate = x.copy(); candidate[i] = state
                    proposed = float(objective(candidate)); trials += 1
                    if proposed < value - 1e-9:
                        x, value, changed = candidate, proposed, True
            for a, b in pairs:
                for sa, sb in product(domains[a], domains[b]):
                    if sa == x[a] or sb == x[b]:
                        continue
                    candidate = x.copy(); candidate[a], candidate[b] = sa, sb
                    proposed = float(objective(candidate)); trials += 1
                    if proposed < value - 1e-9:
                        x, value, changed = candidate, proposed, True
            trace.append(value)
            if not changed:
                break
        tie=(int(np.count_nonzero(x)),tuple(x)) if tie_key is None else tie_key(x)
        outputs.append(((value,*tie),x,trace))
    if not outputs:
        raise ArithmeticError("No finite feasible S initialization")
    best = min(outputs, key=lambda z:z[0])
    return best[1], dict(energy=best[0][0], objective_trace=best[2], objective_trials=trials,
                         starts=len(starts), rounds=len(best[2])-1, exact=False)


def _y_search(c, extra, branch=(), pairs=()):
    starts = [c.m0.astype(int), np.zeros(c.n, int), np.ones(c.n, int)] + [np.asarray(y, int) for y in branch]
    return finite_search(lambda y: c.energy(y) + float(extra(y)), starts,
                         [(0, 1)] * c.n, list(map(tuple, c.edges)) + list(pairs))


def _cut(c, s=None, fixed=None, extra_edges=None):
    from ics.methods.pro_paired_environment import exact_potts_cut
    logits = c.s.copy() if s is None else np.asarray(s, np.float32).copy()
    edges, capacity = c.edges, c.capacity
    if fixed is not None:
        ids, values = fixed
        bound = float(np.abs(logits).sum() + 2 * np.abs(capacity).sum() + 1)
        logits[np.asarray(ids, int)] = (2*np.asarray(values, int)-1)*bound
    if extra_edges is not None:
        ee, ec = extra_edges
        edges = np.concatenate((edges, np.asarray(ee, int).reshape(-1, 2)))
        capacity = np.concatenate((capacity, np.asarray(ec, np.float32)))
    y, certificate = exact_potts_cut(logits, edges, capacity)
    return y.astype(bool), certificate


def _scale(c, values):
    """Use source-only scale supplied by the adapter, never query MAD."""
    return np.asarray(values, np.float32) / c.factor_scale


def _f251(c, weight, control=None):
    h_f = c.fsim.max(1)
    pieces = []
    for mode in range(c.bsim.shape[1]):
        advantage = c.bsim[:, mode] - h_f
        for ids in _components(advantage > 0, c.hw):
            pieces.append((float(advantage[ids].sum()), mode, ids, advantage))
    pieces.sort(key=lambda p: (-p[0], p[1], tuple(p[2])))
    discarded=max(0,len(pieces)-8)
    pieces = pieces[:8]
    factors = []
    for _, mode, ids, advantage in pieces:
        support = np.zeros(c.n, bool); support[ids] = True
        selected = support[c.edges[:, 0]] & support[c.edges[:, 1]]
        pair = np.minimum(advantage[c.edges[selected, 0]], advantage[c.edges[selected, 1]])
        factors.append((ids, c.edges[selected], _scale(c, pair), float(_scale(c, [1-advantage[ids].mean()])[0])))
    k = len(factors)
    def objective(x):
        y, z = x[:c.n], x[c.n:]
        factor = sum(z[j]*(fee + float(np.sum(cost*y[e[:, 0]]*y[e[:, 1]])))
                     for j, (_, e, cost, fee) in enumerate(factors))
        return c.energy(y) + weight*factor
    starts = [np.r_[y, np.ones(k, int)] for y in (c.m0, np.zeros(c.n), np.ones(c.n))]
    starts += [np.r_[c.m0, np.zeros(k, int)]]
    domains = [(0, 1)]*(c.n+k)
    if control == "fixed_z":
        domains[c.n:] = [(1,)]*k
        starts = starts[:3]
    pairs = list(map(tuple, c.edges))
    for j, (ids, _, _, _) in enumerate(factors):
        pairs += [(int(i), c.n+j) for i in ids[:4]]
    x, info = finite_search(objective, starts, domains, pairs,tie_key=lambda v:(int(v[:c.n].sum()),tuple(v[:c.n]),tuple(v[c.n:])))
    info.update(bg_pieces=k, closed_gate_fraction=float(np.mean(x[c.n:] == 0)) if k else 0.,
                discarded_support_components=discarded)
    return dict(y=x[:c.n], info=info)


def _f252(c, weight, control=None):
    adds, deletes = [], []
    for piece in c.pieces:
        a, d = piece[~c.m0[piece]], piece[c.m0[piece]]
        if len(a): adds.append(a)
        if len(d): deletes.append(d)
    coords = np.array(np.unravel_index(np.arange(c.n), c.hw)).T
    transactions = []
    for a in adds:
        mode = int(c.bsim[a].mean(0).argmax())
        matches = [d for d in deletes if c.bsim[d].mean(0).argmax() == mode]
        matches.sort(key=lambda d: (float(np.sum((coords[a].mean(0)-coords[d].mean(0))**2)), tuple(d)))
        for d in matches[:4]:
            gain_a = max(0., float((c.fsim[a].max(1)-c.bsim[a, mode]).mean()))
            gain_d = max(0., float((c.bsim[d, mode]-c.fsim[d].max(1)).mean()))
            transactions.append((a, d, float(_scale(c, [min(gain_a, gain_d)])[0])))
    if control == "shuffle_pair" and transactions:
        ds = [item[1] for item in transactions][::-1]
        transactions = [(a, ds[j], gain) for j, (a, _, gain) in enumerate(transactions)]
    def extra(y):
        if control == "independent":
            return -weight*sum(gain*(np.mean(y[a])+np.mean(1-y[d]))/2 for a, d, gain in transactions)
        return -weight*sum(gain*bool(np.all(y[a]) and not np.any(y[d])) for a, d, gain in transactions)
    pairs = [(int(a[0]), int(d[0])) for a, d, _ in transactions]
    y, info = _y_search(c, extra, pairs=pairs)
    info.update(transactions=len(transactions), realized_transactions=sum(bool(np.all(y[a]) and not np.any(y[d])) for a,d,_ in transactions))
    return dict(y=y, info=info)


def _f253(c, weight, control=None):
    source = c.source
    state = ((source.heads > 0)*np.array([1, 2, 4])).sum(1)
    qstate = ((c.heads > 0)*np.array([1, 2, 4])).sum(1)
    merged = np.array([(source.coverage[state == v].sum()+1)/(np.count_nonzero(state == v)+2) for v in range(8)])
    tables = []
    for block in range(4):
        table = merged.copy()
        for v in range(8):
            ids = (source.blocks == block) & source.oof_valid & (state == v)
            if ids.any(): table[v] = (source.coverage[ids].sum()+1)/(ids.sum()+2)
        tables.append(table)
    probabilities = np.array(tables, np.float32)[:, qstate]
    if control == "independent": probabilities[:] = probabilities.mean(0)
    rng = np.random.default_rng(0)
    scores = np.zeros(len(c.candidates), np.float32)
    for draw in range(256):
        sample = rng.random(c.n) < probabilities[draw % 4]
        for j, y in enumerate(c.candidates):
            union = np.count_nonzero(y | sample)
            scores[j] += (np.count_nonzero(y & sample)/union if union else 1.)/256
    # Weight-zero is the complete source E0 control; other weights blend the
    # fixed model-scenario criterion and baseline criterion in candidate space.
    base = np.array([c.energy(y) for y in c.candidates])
    loss = -scores if weight>0 else (base-base.min())/max(c.n, 1)
    j = min(range(len(loss)), key=lambda i: (float(loss[i]), int(c.candidates[i].sum()), i))
    return dict(y=c.candidates[j], info=dict(candidate_scores=scores.tolist(), selected_candidate=j,
                                           model_scenario_draws=256, random_seed=0, real_query_labels_used=False))


def _f254(c, weight, control=None):
    roots = _peaks(c.s, c.hw)
    length = np.maximum(0., 1-np.sum(c.q[c.edges[:, 0]]*c.q[c.edges[:, 1]], 1))
    length += np.maximum(0., -c.s[c.edges[:, 0]]) + np.maximum(0., -c.s[c.edges[:, 1]])
    adjacency = _adjacency(c.n, c.edges, length)
    root_distance = [_shortest(adjacency, [root])[0] for root in roots]
    paths, per_piece = [], [];path_calls=len(roots)
    for ids in c.pieces:
        order = sorted(range(len(roots)), key=lambda j: (float(root_distance[j][ids].min()), j))[:4]
        options = []
        for r in order:
            _, first = _shortest(adjacency, ids, [roots[r]])
            forbidden = frozenset(tuple(sorted(e)) for e in zip(first[:-1], first[1:]))
            _, second = _shortest(adjacency, ids, [roots[r]], forbidden_edges=forbidden)
            path_calls+=2
            for path in (first, second if control != "shortest_only" else []):
                if path:
                    options.append(len(paths)); paths.append((r, np.array(path, int)))
        per_piece.append(options)
    nr, npth = len(roots), len(paths)
    def objective(x):
        y, z, opened = x[:c.n], x[c.n:c.n+nr], x[c.n+nr:]
        extra = float(z.sum())
        for j, (r, path) in enumerate(paths):
            if opened[j]:
                if not z[r] or not np.all(y[path]): return np.inf
                extra += float(np.maximum(0, -c.s[path]).sum())/c.factor_scale
        for ids, options in zip(c.pieces, per_piece):
            if np.all(y[ids]) and not any(opened[j] for j in options): extra += 1.
        return c.energy(y) + weight*extra
    starts = [np.r_[y, np.zeros(nr+npth, int)] for y in (c.m0, np.zeros(c.n), np.ones(c.n))]
    if paths:
        y = c.m0.copy(); z = np.zeros(nr,int); x = np.zeros(npth,int)
        for j,(r,path) in enumerate(paths):
            if np.all(y[path]): z[r]=1; x[j]=1
        starts.append(np.r_[y,z,x])
    pairs = list(map(tuple,c.edges)) + [(c.n+r,c.n+nr+j) for j,(r,_) in enumerate(paths)]
    x, info = finite_search(objective, starts, [(0,1)]*(c.n+nr+npth), pairs,tie_key=lambda v:(int(v[:c.n].sum()),tuple(v[:c.n]),tuple(v[c.n:])))
    info.update(roots=nr, paths=npth, opened_roots=int(x[c.n:c.n+nr].sum()), opened_paths=int(x[c.n+nr:].sum()),
                path_generation_calls=path_calls)
    return dict(y=x[:c.n],info=info)


def _boundary_segments(m0, hw):
    # Trace directed cell edges, including holes and image-border contours.
    # Every contour is a closed digital boundary; four edges define a segment.
    m = m0.reshape(hw)
    signed = ndimage.distance_transform_edt(m)-ndimage.distance_transform_edt(~m)
    gradient = np.stack(np.gradient(signed),-1)
    boundary=[]
    for y,x in np.argwhere(m):
        site=int(y*hw[1]+x)
        if y==0 or not m[y-1,x]:boundary.append(((int(y),int(x)),(int(y),int(x+1)),site))
        if x==hw[1]-1 or not m[y,x+1]:boundary.append(((int(y),int(x+1)),(int(y+1),int(x+1)),site))
        if y==hw[0]-1 or not m[y+1,x]:boundary.append(((int(y+1),int(x+1)),(int(y+1),int(x)),site))
        if x==0 or not m[y,x-1]:boundary.append(((int(y+1),int(x)),(int(y),int(x)),site))
    outgoing={}
    for j,(a,_,_) in enumerate(boundary):outgoing.setdefault(a,[]).append(j)
    unused=set(range(len(boundary)));segments=[]
    while unused:
        start=min(unused,key=lambda j:boundary[j]);current=start;contour=[]
        while current in unused:
            unused.remove(current);contour.append(boundary[current][2])
            a,b,_=boundary[current];choices=[j for j in outgoing.get(b,[]) if j in unused]
            if not choices:break
            dy,dx=np.subtract(b,a)
            # Right-turn tie preserves separate components at diagonal contacts.
            def turn(j):
                ey,ex=np.subtract(boundary[j][1],boundary[j][0]);cross=dx*ey-dy*ex
                return (-int(cross),boundary[j])
            current=min(choices,key=turn)
        chain=[]
        for offset in range(0,len(contour),4):
            sites=np.unique(contour[offset:offset+4]).astype(int)
            chain.append(len(segments));segments.append((sites,gradient.reshape(-1,2)[sites]))
        yield chain,segments


def chain_dp(unary, pair, closed=False):
    """Exact finite 5-state chain/cycle DP with deterministic state ties."""
    unary=np.asarray(unary); n,k=unary.shape
    best=None
    for first in (range(k) if closed else [None]):
        cost=unary[0].copy()
        if first is not None: cost[np.arange(k)!=first]=np.inf
        back=[]
        for j in range(1,n):
            values=cost[:,None]+pair[j-1]
            arg=np.argmin(values,axis=0); cost=unary[j]+values[arg,np.arange(k)]; back.append(arg)
        if closed: cost=cost+pair[-1][:,first]
        last=int(np.argmin(cost)); states=[last]
        for arg in back[::-1]: states.append(int(arg[states[-1]]))
        states=np.array(states[::-1]); candidate=(float(cost[last]),tuple(states),states)
        if best is None or candidate[:2]<best[:2]:best=candidate
    return best[2],best[0]


def _f255(c,weight,control=None):
    chains=[]; segments=[]
    for chain,all_segments in _boundary_segments(c.m0,c.hw):
        chains.append(chain); segments=all_segments
    if not segments:return dict(y=c.m0,info=dict(boundary_segments=0))
    coords=np.array(np.unravel_index(np.arange(c.n),c.hw)).T
    variants=[]; edit_sets=[]
    for sites,normal in segments:
        norm=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),EPS);normal=normal/norm
        options=[];edits=[]
        for shift in range(-2,3):
            y=c.m0.copy()
            # Each normal sweep includes all crossed sites, so inward moves
            # delete and outward moves add; no displacement-only endpoint edit.
            swept=[]
            for step in (range(1,abs(shift)+1) if shift>0 else range(abs(shift))):
                xy=np.rint(coords[sites]-np.sign(shift)*step*normal).astype(int)
                valid=np.all((xy>=0)&(xy<np.array(c.hw)),1)
                swept.extend(np.ravel_multi_index(xy[valid].T,c.hw).tolist())
            swept=np.unique(swept).astype(int)
            y[swept]=shift>0
            options.append(y);edits.append(np.flatnonzero(y!=c.m0))
        variants.append(options);edit_sets.append(edits)
    source_pairs=c.source.role_pairs
    def pair_fee(a,sa,b,sb):
        ea,eb=edit_sets[a][sa],edit_sets[b][sb]
        if len(np.intersect1d(ea,eb)) and np.any(variants[a][sa][np.intersect1d(ea,eb)]!=variants[b][sb][np.intersect1d(ea,eb)]):return np.inf
        if control=="unary_dp" or not len(ea) or not len(eb) or not len(source_pairs):return 0.
        va=c.q[ea].mean(0);vb=c.q[eb].mean(0)
        # Both segments must be explained by the *same* source FG->BG pair.
        direct=(source_pairs[:,0]@va+source_pairs[:,1]@vb)/.1
        reverse=(source_pairs[:,1]@va+source_pairs[:,0]@vb)/.1
        return -weight*float(logsumexp(direct)-logsumexp(reverse))/c.factor_scale
    def merge(states):
        y=c.m0.copy();written={}
        for j,state in enumerate(states):
            for i in edit_sets[j][state]:
                value=int(variants[j][state][i])
                if i in written and written[i]!=value:return None
                written[i]=value;y[i]=value
        return y
    def objective(states):
        y=merge(states)
        if y is None:return np.inf
        fee=0.
        for chain in chains:
            if len(chain)>1:fee+=sum(pair_fee(a,states[a],b,states[b]) for a,b in zip(chain,chain[1:]+chain[:1]))
        return c.energy(y)+fee
    dp=np.full(len(segments),2,int)
    for chain in chains:
        unary=np.array([[c.energy(variants[j][state])-c.energy(c.m0) for state in range(5)] for j in chain])
        pairs=np.array([[[pair_fee(a,sa,b,sb) for sb in range(5)] for sa in range(5)] for a,b in zip(chain,chain[1:]+chain[:1])])
        states,_=chain_dp(unary,pairs,closed=len(chain)>1);dp[chain]=states
    # DP supplies a branch; global S re-scores E0 and all cross-chain edits.
    candidate=dp if merge(dp) is not None else np.full(len(segments),2)
    near=[(a,b) for a,b in combinations(range(len(segments)),2) if any(len(np.intersect1d(x,y)) for x in edit_sets[a] for y in edit_sets[b])]
    x,info=finite_search(objective,[np.full(len(segments),2),candidate], [range(5)]*len(segments),near,tie_key=lambda v:(int(merge(v).sum()),tuple(merge(v)),tuple(v)))
    info.update(boundary_segments=len(segments),chains=len(chains),cross_chain_conflicts=len(near),displacements=(x-2).tolist())
    return dict(y=merge(x),info=info)


class ResourceUnavailable(RuntimeError):
    """A card's true encoder observation is absent; never a synthetic fallback."""


def _verify_observation(c,observation):
    producer=getattr(getattr(c,"ep",None),"producer",{})
    expected=producer.get("model_assets",producer).get("checkpoint_sha256")
    receipt=observation.get("encoder_receipt",{})
    if not expected or receipt.get("checkpoint_sha256")!=expected or receipt.get("real_encoder_execution") is not True:
        raise ResourceUnavailable("A producer-bound real encoder receipt with the episode checkpoint hash is required")
    images=producer.get("source_image_hashes")
    if images is not None and receipt.get("source_image_hashes")!=images:
        raise ResourceUnavailable("Real observation source images differ from the native episode")


def _f256(c,weight,control=None,resources=None):
    resources=resources or {}
    observation=resources.get("f256_three_views")
    if observation is None:
        raise ResourceUnavailable("F256 requires real R/Q original, horizontal-flip and +8px-translation native DINO views with registered masks and coordinates (6 encoder forwards)")
    if observation.get("real_encoder_execution") is not True or observation.get("native_dtype")!="float32":
        raise ResourceUnavailable("F256 rejects fake callback, old phase cache or non-FP32 observations")
    _verify_observation(c,observation)
    if observation.get("view_names") != ["original","horizontal_flip","translate_x_8"]:
        raise ValueError("F256 needs exactly the source three true views")
    rmargin=np.asarray(observation["reference_margins"],np.float32)
    qmargin=np.asarray(observation["query_margins"],np.float32)
    valid=np.asarray(observation["reference_valid"],bool)
    if rmargin.shape!=(len(c.r),3) or qmargin.shape!=(c.n,3) or valid.shape!=rmargin.shape:
        raise ValueError("F256 backprojected three-view margin shapes disagree")
    good=valid.all(1)&c.source.train_valid
    densities=[]
    for role in (c.source.coverage>=.9,c.source.coverage<=.1):
        ids=good&role
        if ids.sum()<3:raise ArithmeticError("F256 insufficient true-view class observations")
        # Give spatial blocks equal mass instead of treating every token as an
        # independent repeat. Within-block scatter remains in the covariance.
        blocks=np.unique(c.source.blocks[ids]);means=[];covs=[]
        for b in blocks:
            rows=rmargin[ids&(c.source.blocks==b)]
            means.append(rows.mean(0));covs.append((rows-rows.mean(0)).T@(rows-rows.mean(0))/max(len(rows),1))
        mu=np.mean(means,0);cov=np.mean(covs,0)+np.cov(np.asarray(means).T,bias=True) if len(means)>1 else covs[0]
        cov=.5*np.diag(np.diag(cov))+.5*cov
        if control=="diagonal":cov=np.diag(np.diag(cov))
        cov=cov+EPS*np.eye(3,dtype=np.float32)
        sign,logdet=np.linalg.slogdet(cov)
        if sign<=0:raise ArithmeticError("F256 singular covariance")
        diff=qmargin-mu
        densities.append(-.5*(np.sum(diff*np.linalg.solve(cov,diff.T).T,1)+logdet+3*np.log(2*np.pi)))
    margin=densities[0]-densities[1]
    if control=="mean":margin=qmargin.mean(1)
    if control=="median":margin=np.median(qmargin,axis=1)
    return dict(margin=(1-weight)*c.s+weight*margin,info=dict(real_encoder_execution=True,new_encoder_forwards=6,views=observation["view_names"],class_prior="equal",class_covariance="0.5 diagonal plus 0.5 full"))


def _f257(c,weight,control=None):
    roots=_peaks(c.s,c.hw)
    length=np.maximum(0.,1-np.sum(c.q[c.edges[:,0]]*c.q[c.edges[:,1]],1))
    length+=np.maximum(0,-c.s[c.edges[:,0]])+np.maximum(0,-c.s[c.edges[:,1]])
    graph=_adjacency(c.n,c.edges,length)
    distance=[];path_cost=[]
    for root in roots:
        dist,_=_shortest(graph,[root]);distance.append(dist)
        costs=np.zeros(c.n,np.float32)
        for i in range(c.n):
            _,path=_shortest(graph,[root],[i])
            costs[i]=np.maximum(0,-c.s[path]).mean() if path else np.inf
        path_cost.append(costs)
    nr=len(roots)
    distance=np.asarray(distance).T if nr else np.empty((c.n,0))
    path_cost=np.asarray(path_cost).T if nr else np.empty((c.n,0))
    permitted=[np.argsort(distance[i],kind="stable")[:4]+1 for i in range(c.n)]
    domains=[(-1,0,*map(int,p)) for p in permitted]+[(0,1)]*nr
    def objective(x):
        a,z=x[:c.n],x[c.n:];y=a>=0
        if any(a[i]>0 and not z[a[i]-1] for i in range(c.n)):return np.inf
        extra=float(z.sum())+np.count_nonzero(a==0)
        for i in np.flatnonzero(a>0):extra+=float(path_cost[i,a[i]-1])/c.factor_scale
        return c.energy(y)+weight*extra
    starts=[np.r_[np.where(y,0,-1),np.zeros(nr,int)] for y in (c.m0,np.zeros(c.n,bool),np.ones(c.n,bool))]
    if nr:
        assignment=np.where(c.m0,1+np.argmin(distance,axis=1),-1)
        starts.append(np.r_[assignment,np.ones(nr,int)])
    if control=="no_open_fee":
        original=objective
        objective=lambda x:original(x)-weight*float(x[c.n:].sum())
    pairs=list(map(tuple,c.edges))+[(i,c.n+int(r)-1) for i,p in enumerate(permitted) for r in p[:4]]
    x,info=finite_search(objective,starts,domains,pairs,tie_key=lambda v:(int(np.count_nonzero(v[:c.n]>=0)),tuple(v[:c.n]>=0),tuple(v)))
    info.update(root_facilities=nr,opened_facilities=int(x[c.n:].sum()),independent_fg_points=int(np.count_nonzero(x[:c.n]==0)),dijkstra_calls=int(nr*(c.n+1)))
    return dict(y=x[:c.n]>=0,info=info)


def _f258(c,weight,control=None):
    atoms=[np.asarray(ids,int) for ids in c.pieces]+[np.array([i]) for i in range(c.n)]
    npiece=len(c.pieces)
    trials=0;outputs=[]
    def score(selected):
        y=np.zeros(c.n,bool);cover=np.zeros(c.n,int);fees=0.
        for atom,label in selected:
            ids=atoms[atom];cover[ids]+=1;y[ids]=label;fees+=np.log1p(len(ids))
        if np.any(cover!=1):return np.inf,None
        return c.energy(y)+weight*fees,y
    for initial in (c.m0,np.zeros(c.n,bool),np.ones(c.n,bool)):
        selected=[(npiece+i,int(initial[i])) for i in range(c.n)]
        value,_=score(selected);trials+=1;trace=[float(value)]
        for _ in range(1 if control=="greedy_cover" else 10):
            changed=False
            for atom,ids in enumerate(atoms):
                for label in (0,1):
                    intersect=[j for j,(a,_) in enumerate(selected) if np.intersect1d(atoms[a],ids).size]
                    removed=np.unique(np.concatenate([atoms[selected[j][0]] for j in intersect])) if intersect else np.empty(0,int)
                    holes=np.setdiff1d(removed,ids)
                    candidate=[item for j,item in enumerate(selected) if j not in intersect]+[(atom,label)]
                    # Hole labels inherit the current complete field, and are
                    # subsequently tried as singleton variables by the sweep.
                    _,previous=score(selected)
                    candidate += [(npiece+int(i),int(previous[i])) for i in holes]
                    proposed,_=score(candidate);trials+=1
                    if proposed<value-1e-9:selected,value,changed=candidate,proposed,True
            trace.append(float(value))
            if not changed:break
        _,y=score(selected);outputs.append((float(value),int(y.sum()),tuple(y),y,selected,trace))
    best=min(outputs,key=lambda x:x[:3])
    return dict(y=best[3],info=dict(energy=best[0],selected_explanations=len(best[4]),selected_multi_point=sum(len(atoms[a])>1 for a,_ in best[4]),exact_cover=True,exact_optimizer=False,objective_trials=trials,objective_trace=best[5]))


def _source_triples(c):
    src=c.source;fs,bs=src.fsim,src.bsim
    triples=[]
    for a,b,other in product(range(fs.shape[1]),range(bs.shape[1]),range(fs.shape[1])):
        if a==other:continue
        false=src.train_valid&(src.coverage<=.1)&(np.argmax(bs,1)==b)&(fs[:,a]>bs[:,b])
        true=src.train_valid&(src.coverage>=.9)&(np.argmax(fs,1)==a)
        if false.sum()<8 or true.sum()<8:continue
        # Another true F mode must discriminate that exact confusing B mode.
        correction=fs[:,other]-bs[:,b]
        if np.median(correction[true])>0 and np.median(correction[false])<0:
            triples.append((float(np.median(correction[true])-np.median(correction[false])),a,b,other))
    triples.sort(key=lambda t:(-t[0],t[1:]))
    return [t[1:] for t in triples[:8]]


def _f259(c,weight,control=None):
    triples=_source_triples(c);k=len(triples)
    def objective(x):
        y,g=x[:c.n],x[c.n:];extra=0.
        for j,(a,b,other) in enumerate(triples):
            advantage=np.maximum(0,c.bsim[:,b]-c.fsim.max(1))
            support=advantage>0
            if g[j]:extra+=1-float(np.minimum(1,advantage[support])@(1-y[support]))/max(int(support.sum()),1)
            for ids in c.pieces:
                active=(c.fsim[ids,a]-c.bsim[ids].max(1))>0
                witness=(c.fsim[ids,other]-c.bsim[ids].max(1))>0
                if g[j] and np.any(active&y[ids].astype(bool)) and not np.any(witness&y[ids].astype(bool)):
                    extra+=float(np.maximum(0,c.fsim[ids,a]-c.bsim[ids].max(1))[active].sum())/c.factor_scale
        return c.energy(y)+weight*extra
    starts=[np.r_[y,np.zeros(k,int)] for y in (c.m0,np.zeros(c.n),np.ones(c.n))]
    starts += [np.r_[c.m0,np.ones(k,int)]]
    domains=[(0,1)]*(c.n+k)
    if control=="all_on":domains[c.n:]=[(1,)]*k;starts=[np.r_[y,np.ones(k,int)] for y in (c.m0,np.zeros(c.n),np.ones(c.n))]
    if control=="all_off":domains[c.n:]=[(0,)]*k;starts=starts[:3]
    pairs=list(map(tuple,c.edges))
    for j,(a,_,_) in enumerate(triples):pairs += [(int(i),c.n+j) for i in _peaks(c.fsim[:,a]-c.bsim.max(1),c.hw)[:4]]
    x,info=finite_search(objective,starts,domains,pairs,tie_key=lambda v:(int(v[:c.n].sum()),tuple(v[:c.n]),tuple(v[c.n:])))
    info.update(source_confusion_triples=[list(t) for t in triples],scene_states=x[c.n:].tolist())
    return dict(y=x[:c.n],info=info)


def _negative_bands(c):
    negative=c.bsim.max(1)-c.fsim.max(1)
    graph=_adjacency(c.n,c.edges)
    active=set(np.flatnonzero(negative>0).tolist())
    visited=set();bands=[]
    # Split support chains at graph degree !=2; cycles begin at their first site.
    degree={i:sum(j in active for j,_ in graph[i]) for i in active}
    endpoints=sorted(i for i in active if degree[i]!=2)+sorted(active)
    for start in endpoints:
        for nxt,_ in graph[start]:
            edge=tuple(sorted((start,nxt)))
            if nxt not in active or edge in visited:continue
            path=[start];previous=start;current=nxt;visited.add(edge)
            while True:
                path.append(current)
                if degree[current]!=2:break
                following=[j for j,_ in graph[current] if j in active and j!=previous]
                if not following:break
                other=following[0];edge=tuple(sorted((current,other)))
                if edge in visited:break
                visited.add(edge);previous,current=current,other
            bands.append(np.unique(path))
    bands.sort(key=lambda ids:(-float(negative[ids].sum()),tuple(ids)))
    return bands[:32],negative


def _f260(c,weight,control=None):
    bands,negative=_negative_bands(c);roots=_peaks(c.s,c.hw);nb,nr=len(bands),len(roots)
    cut_trials=0;cache={}
    def condition(x):
        nonlocal cut_trials
        key=tuple(x)
        if key in cache:return cache[key]
        fixed={}
        for j,band in enumerate(bands):
            if x[j]:fixed.update({int(i):0 for i in band})
        for j,root in enumerate(roots):
            if x[nb+j]:
                if root in fixed:return np.inf,None
                fixed[int(root)]=1
        ids=np.array(sorted(fixed),int);values=np.array([fixed[i] for i in ids],int)
        y,_=_cut(c,fixed=(ids,values));cut_trials+=1
        fee=-sum(float(negative[ids].sum()) for j,ids in enumerate(bands) if x[j])
        fee-=sum(float(c.s[root]) for j,root in enumerate(roots) if x[nb+j])
        for component in _components(y,c.hw):
            if not any(x[nb+j] and root in component for j,root in enumerate(roots)):
                fee+=float(np.sum(np.maximum(0,c.s[component]))) # independent evidence payment
        value=c.energy(y)+weight*fee/c.factor_scale
        cache[key]=(value,y);return value,y
    if nb+nr==0:return dict(y=_cut(c)[0],info=dict(bands=0,roots=0,conditional_cuts=1))
    starts=[np.zeros(nb+nr,int),np.r_[np.zeros(nb,int),np.ones(nr,int)]]
    starts += [np.r_[np.ones(nb,int),np.zeros(nr,int)]]
    pairs=[(j,nb+k) for j,band in enumerate(bands) for k,root in enumerate(roots) if root in band][:128]
    domains=[(0,1)]*(nb+nr)
    if control=="fixed_bands":domains[:nb]=[(1,)]*nb;starts=[np.r_[np.ones(nb,int),np.zeros(nr,int)]]
    x,info=finite_search(lambda x:condition(x)[0],starts,domains,pairs,tie_key=lambda v:(int(condition(v)[1].sum()),tuple(condition(v)[1]),tuple(v)))
    info.update(bands=nb,roots=nr,conditional_cuts=cut_trials,chosen_bands=x[:nb].tolist(),chosen_roots=x[nb:].tolist())
    return dict(y=condition(x)[1],info=info)


def _prototype_for_roles(c,coverage):
    if coverage.sum()<=0:return np.full(c.n,-100.,np.float32)
    if (c.source.valid-coverage).sum()<=0:
        bank=c.r[coverage>0]
        if len(bank)<2:return np.full(c.n,-100.,np.float32)
        similarity=bank@bank.T;np.fill_diagonal(similarity,-np.inf)
        threshold=float(np.quantile(similarity.max(1),.05))
        return ((c.q@bank.T).max(1)-threshold)/.1
    f=_unit(np.sum(c.r*coverage[:,None],0));b=_unit(np.sum(c.r*(c.source.valid-coverage)[:,None],0))
    return (c.q@f-c.q@b)/.1


def _augment_disagreement(c,candidates,maximum=64):
    unique=[];keys=set()
    def append(y):
        key=np.packbits(np.asarray(y,bool)).tobytes()
        if key not in keys and len(unique)<maximum:keys.add(key);unique.append(np.asarray(y,bool))
    for y in candidates:append(y)
    initial=list(unique)
    for a,b in combinations(initial,2):
        for ids in _components(a!=b,c.hw):
            y=a.copy();y[ids]=~y[ids];append(y)
    return unique


def _f261(c,weight,control=None):
    coverage=c.source.coverage*c.source.valid
    pure_f=c.source.coverage>=.9;pure_b=c.source.coverage<=.1
    fields=[];solutions=[]
    for threshold in (.1,.3,.5,.7,.9):
        roles=coverage.copy();mixed=~(pure_f|pure_b)&c.source.train_valid
        roles[mixed]=(c.source.coverage[mixed]>threshold)*c.source.valid[mixed]
        s=_prototype_for_roles(c,roles);fields.append(s);solutions.append(_cut(c,s=s)[0])
    candidates=_augment_disagreement(c,[c.m0,*solutions])
    energy=np.array([[c.energy(y,s=s) for y in candidates] for s in fields])
    optimum=np.array([c.energy(y,s=s) for y,s in zip(solutions,fields)])
    regret=(energy-optimum[:,None])/c.factor_scale
    criterion=regret.max(0) if control!="mean_field" else regret.mean(0)
    if weight<=0:criterion=np.array([c.energy(y) for y in candidates])
    j=min(range(len(candidates)),key=lambda i:(float(criterion[i]),i!=0,int(candidates[i].sum()),i))
    return dict(y=candidates[j],info=dict(scenarios=5,candidates=len(candidates),selected_candidate=j,maximum_regret=float(regret[:,j].max()),query_area_estimated=False))


def _local_cut(c,ids,s):
    outside=np.setdiff1d(np.arange(c.n),ids)
    logits=c.s.copy();logits[ids]=s[ids]
    return _cut(c,s=logits,fixed=(outside,c.m0[outside]))[0]


def _f262(c,weight,control=None):
    choices=[];local_cuts=0
    for piece in c.pieces:
        mask=np.zeros(c.n,bool);mask[piece]=True
        region=np.flatnonzero(ndimage.binary_dilation(mask.reshape(c.hw)).ravel())
        costs=[];ys=[]
        foreground_center=_unit(np.sum(c.r*(c.source.coverage*c.source.valid)[:,None],axis=0))
        for role,center in [(1,foreground_center)]+[(0,m) for m in c.b]:
            residual=1-c.q@center
            # Same movable ring, same two-label degrees of freedom for F/B;
            # the alternative retains the original complete baseline loss.
            alternative=np.minimum(1-c.fsim.max(1),1-c.bsim.max(1))
            score=(alternative-residual)/.1 if role else (residual-alternative)/.1
            y=c.m0.copy() if control=="fixed_boundary" else _local_cut(c,region,score)
            if control=="fixed_boundary":y[piece]=role
            local_cuts+=1
            cost=float(np.sum(np.where(y[region]==role,residual[region],alternative[region])))
            cost+=float(c.capacity[np.any(np.isin(c.edges,region),1)]@(y[c.edges[np.any(np.isin(c.edges,region),1),0]]!=y[c.edges[np.any(np.isin(c.edges,region),1),1]]))
            costs.append(cost);ys.append(y)
        if not costs:continue
        nf=1;fi=0;bi=1+int(np.argmin(costs[1:]))
        gain=(costs[bi]-costs[fi])/c.factor_scale
        choices.append((region,[c.m0[region],ys[fi][region],ys[bi][region]],gain))
    def merge(x):
        y=c.m0.copy();seen={}
        for j,(ids,options,_) in enumerate(choices):
            if x[j]==0:continue
            for i,label in zip(ids,options[x[j]]):
                if i in seen and seen[i]!=label:return None
                seen[int(i)]=bool(label);y[i]=label
        return y
    def objective(x):
        y=merge(x)
        if y is None:return np.inf
        return c.energy(y)-weight*sum(gain*(x[j]==1)-gain*(x[j]==2) for j,(_,_,gain) in enumerate(choices))
    if not choices:return dict(y=c.m0,info=dict(local_cuts=local_cuts,explanation_pieces=0))
    independent=np.array([1 if gain>0 else 2 for _,_,gain in choices])
    if merge(independent) is None:independent=np.zeros(len(choices),int)
    pairs=[(a,b) for a,b in combinations(range(len(choices)),2) if np.intersect1d(choices[a][0],choices[b][0]).size]
    x,info=finite_search(objective,[np.zeros(len(choices),int),independent],[range(3)]*len(choices),pairs,tie_key=lambda v:(int(merge(v).sum()),tuple(merge(v)),tuple(v)))
    info.update(local_cuts=local_cuts,explanation_pieces=len(choices),fg_bg_equal_boundary_freedom=True)
    return dict(y=merge(x),info=info)


def _ridge(x,target,l2=1.):
    x=np.asarray(x,np.float32);target=np.asarray(target,np.float32)
    design=np.c_[np.ones(len(x),np.float32),x]
    penalty=np.eye(design.shape[1],dtype=np.float32)*l2;penalty[0,0]=0
    return np.linalg.solve(design.T@design+penalty+EPS*np.eye(design.shape[1]),design.T@target)


def _f263(c,weight,control=None):
    src=c.source;valid=src.oof_valid
    predicted=src.heads[:,0]>0;truth=src.coverage>=.5
    # State order TP,TN,FP,FN. Each query point has exactly its two M0 states.
    states=np.where(predicted,np.where(truth,0,2),np.where(truth,3,1))
    counts=np.bincount(states[valid],minlength=4)
    x=np.c_[src.heads[:,0],src.fsim.max(1),src.bsim.max(1)]
    response=np.eye(4,dtype=np.float32)[states]
    coefficients=_ridge(x[valid],response[valid])
    query=np.c_[np.ones(c.n),c.s,c.fsim.max(1),c.bsim.max(1)]@coefficients
    unary=-np.log(np.maximum(expit(query),EPS))
    pair_counts=np.ones((4,4),np.float32)
    for a,b in src.edges:
        if valid[a] and valid[b] and src.blocks[a]==src.blocks[b]:
            pair_counts[states[a],states[b]]+=1;pair_counts[states[b],states[a]]+=1
    joint=pair_counts/pair_counts.sum();marginal=joint.sum(1)
    pair=-np.log(np.maximum(joint/(marginal[:,None]*marginal[None,:]),EPS))
    enabled=bool(np.all(counts>=8)) and control!="no_interaction"
    def state_for(y):return np.where(c.m0,np.where(y,0,2),np.where(y,3,1))
    def extra(y):
        st=state_for(y);value=float(unary[np.arange(c.n),st].sum())
        if enabled:value+=float(pair[st[c.edges[:,0]],st[c.edges[:,1]]].sum())
        return weight*value/c.factor_scale
    y,info=_y_search(c,extra,branch=[query[:,0]+query[:,3]>query[:,1]+query[:,2]])
    info.update(source_error_state_counts=counts.tolist(),interaction_enabled=enabled,dirichlet_pseudocount=1,source_pairs=int(pair_counts.sum()-16))
    return dict(y=y,info=info)


def _pca_models(r,coverage,valid,rank=16):
    models=[]
    for role in (coverage>=.9,coverage<=.1):
        x=r[role&valid]
        if not len(x):raise ArithmeticError("PCA residual head lacks pure role")
        mean=x.mean(0);_,_,vh=np.linalg.svd(x-mean,full_matrices=False)
        basis=vh[:min(rank,max(len(x)-1,0))]
        models.append((mean,basis))
    return models


def _pca_margin(q,models):
    residual=[]
    for mean,basis in models:
        delta=q-mean;residual.append(np.sum(delta*delta,1)-np.sum((delta@basis.T)**2,1))
    return (residual[1]-residual[0])/.1


def _f264(c,weight,control=None):
    src=c.source;valid=src.train_valid
    ridge=_ridge(c.r[valid],src.coverage[valid]*2-1)
    fields=[c.s,np.c_[np.ones(c.n),c.q]@ridge,_pca_margin(c.q,_pca_models(c.r,src.coverage,valid))]
    fold_losses=[]
    for block in range(4):
        train=src.fold_train[block];test=src.fold_test[block]
        if not train.any() or not test.any():continue
        heads=[]
        cover=src.coverage*src.valid*train
        f=_unit(np.sum(c.r*cover[:,None],0));b=_unit(np.sum(c.r*(src.valid*train-cover)[:,None],0))
        heads.append((c.r@f-c.r@b)/.1)
        coeff=_ridge(c.r[train],src.coverage[train]*2-1);heads.append(np.c_[np.ones(len(c.r)),c.r]@coeff)
        try:heads.append(_pca_margin(c.r,_pca_models(c.r,src.coverage,train)))
        except ArithmeticError:heads.append(heads[0])
        fold_losses.append([float((np.logaddexp(0,h[test])-src.coverage[test]*h[test]).mean()) for h in heads])
    losses=np.asarray(fold_losses,np.float32)
    if len(losses)<2:retained=[0];mean_loss=np.zeros(3)
    else:
        mean_loss=losses.mean(0);spread=losses.std(0);retained=np.flatnonzero(mean_loss-mean_loss.min()<=spread+EPS).tolist()
    source_best=min(retained,key=lambda j:(float(mean_loss[j]),j))
    if control=="source_best":return dict(y=_cut(c,s=fields[source_best])[0],info=dict(retained_heads=retained,source_best=source_best))
    solutions=[_cut(c,s=fields[j])[0] for j in retained]
    candidates=_augment_disagreement(c,[solutions[retained.index(source_best)],*c.candidates,*solutions])
    regret=np.array([[c.energy(y,s=fields[j])-c.energy(opt,s=fields[j]) for y in candidates] for j,opt in zip(retained,solutions)])/c.factor_scale
    criterion=regret.max(0) if control!="mean_regret" else regret.mean(0)
    if weight<=0:criterion=np.array([c.energy(y) for y in candidates])
    chosen=min(range(len(candidates)),key=lambda i:(float(criterion[i]),i!=0,int(candidates[i].sum()),i))
    return dict(y=candidates[chosen],info=dict(retained_heads=retained,source_best=source_best,source_fold_losses=losses.tolist(),candidate_count=len(candidates),selected_candidate=chosen,pca_rank=16))


def _piece_features(margin,ids,hw):
    coords=np.array(np.unravel_index(ids,hw)).T;mid=(coords.min(0)+coords.max(0)+1)/2
    quarter=(coords[:,0]>=mid[0])*2+(coords[:,1]>=mid[1])
    values=[float(np.mean(margin[ids][quarter==j])) for j in range(4) if np.any(quarter==j)]
    return [float(margin[ids].mean()),min(values)]


def _f265(c,weight,control=None):
    src=c.source
    def examples(valid):
        features=[];labels=[]
        foreground=_components((src.coverage>=.9)&valid,src.hw)
        for ids in foreground:
            features.append(_piece_features(src.heads[:,0],ids,src.hw));labels.append(1.)
            # BFS-prefix within one BG component preserves connectedness.
            graph=_adjacency(len(c.r),src.edges)
            for region in _components((src.coverage<=.1)&valid,src.hw):
                if len(region)<len(ids):continue
                allowed=set(map(int,region));queue=[int(region[0])];seen=set(queue)
                for point in queue:
                    for nxt,_ in graph[point]:
                        if nxt in allowed and nxt not in seen:seen.add(nxt);queue.append(nxt)
                    if len(queue)>=len(ids):break
                selected=np.array(queue[:len(ids)],int)
                features.append(_piece_features(src.heads[:,0],selected,src.hw));labels.append(0.);break
        return np.asarray(features,np.float32),np.asarray(labels,np.float32)
    features,labels=examples(src.train_valid&src.oof_valid)
    if len(features)<2 or len(set(labels))<2:return dict(y=c.m0,info=dict(unidentifiable_quality_head=True))
    coeff=_ridge(np.array(features),np.array(labels)*2-1)
    held_scores=[];held_labels=[]
    for train,test in zip(src.fold_train,src.fold_test):
        tx,ty=examples(train&src.train_valid&src.oof_valid);vx,vy=examples(test&src.train_valid&src.oof_valid)
        if len(tx)<2 or len(set(ty))<2 or not len(vx):continue
        fit=_ridge(tx,ty*2-1);held_scores.extend(np.c_[np.ones(len(vx)),vx]@fit);held_labels.extend(vy)
    threshold=0.
    if held_scores:
        hs=np.asarray(held_scores);hy=np.asarray(held_labels)
        candidates=np.unique(np.r_[0.,hs,np.nextafter(hs.min(),-np.inf)])
        threshold=min(candidates,key=lambda t:(float(np.mean((hs>t)!=(hy>.5))),abs(float(t)),float(t)))
    score=np.c_[np.ones(len(c.pieces)),np.array([_piece_features(c.s,ids,c.hw) for ids in c.pieces])]@coeff
    gain=(score-threshold)/c.factor_scale
    def labels_for(z):
        union=np.zeros(c.n,bool)
        for j,ids in enumerate(c.pieces):
            if z[j]:union[ids]=True
        # Existing support can survive a selected explanation; an added shell
        # needs its own positive margin instead of inheriting the piece label.
        return union&(c.m0|(c.s>0))
    def objective(z):
        count=np.zeros(c.n,int)
        for j,ids in enumerate(c.pieces):
            if z[j]:count[ids]+=1
        y=labels_for(z)
        return c.energy(y)+weight*(float(np.maximum(count-1,0).sum())+float(z.sum())-float(z@gain))
    j=len(c.pieces)
    if not j:return dict(y=np.zeros(c.n,bool),info=dict(selected_pieces=0))
    starts=[np.zeros(j,int),(gain>1).astype(int),np.ones(j,int)]
    if control=="force_one":
        orig=objective;objective=lambda z:orig(z) if z.any() else np.inf
        starts=[np.eye(j,dtype=int)[int(gain.argmax())],np.ones(j,int)]
    pairs=[(a,b) for a,b in combinations(range(j),2) if np.intersect1d(c.pieces[a],c.pieces[b]).size]
    z,info=finite_search(objective,starts,[(0,1)]*j,pairs,tie_key=lambda v:(int(labels_for(v).sum()),tuple(labels_for(v)),tuple(v)))
    info.update(selected_pieces=int(z.sum()),empty_scene=not bool(z.any()),source_quality_examples=len(labels),quality_leave_block_examples=len(held_scores),source_quality_threshold=float(threshold),query_zero_distribution_known=False)
    return dict(y=labels_for(z),info=info)


def _profile_density(q,r,coverage,valid):
    sim=q@r.T/.1
    output=[]
    for roles in (coverage,1-coverage):
        mass=np.maximum(roles*valid,0)
        if mass.sum()<=0:raise ArithmeticError("No profile role density")
        output.append(logsumexp(sim+np.where(mass>0,np.log(np.maximum(mass,EPS)),-np.inf)[None,:],axis=1)-np.log(mass.sum()))
    return output


def _f266(c,weight,control=None):
    src=c.source
    tf=float(np.quantile(src.fsim[(src.coverage>=.9)&src.oof_valid].max(1),.05))
    tb=float(np.quantile(src.bsim[(src.coverage<=.1)&src.oof_valid].max(1),.05))
    fw=c.fsim.max(1)>tf;bw=c.bsim.max(1)>tb
    unknown=~(fw|bw);pieces=_components(unknown,c.hw)
    # Preserve all unknown points if more than the finite state budget: group
    # complete components in row order; never select just the largest region.
    original_count=len(pieces)
    if len(pieces)>64:pieces=[np.concatenate(pieces[j::64]) for j in range(64)]
    if not pieces:return dict(y=c.m0,info=dict(unknown_components=0))
    lf,lb=_profile_density(c.q,c.r,src.coverage,src.valid*src.train_valid)
    graph=_adjacency(c.n,c.edges,np.maximum(0,-np.minimum(c.s[c.edges[:,0]],c.s[c.edges[:,1]])))
    options=[];path_fees=[];path_calls=0
    for ids in pieces:
        split=_local_cut(c,ids,c.s)[ids]
        _,pf=_shortest(graph,ids,np.flatnonzero(fw),bottleneck=True)
        # Background identity uses the reverse sign in its own bottleneck graph.
        bggraph=_adjacency(c.n,c.edges,np.maximum(0,np.maximum(c.s[c.edges[:,0]],c.s[c.edges[:,1]])))
        _,pb=_shortest(bggraph,ids,np.flatnonzero(bw),bottleneck=True);path_calls+=2
        if not pf and not pb:options.append([c.m0[ids]]*3);path_fees.append([0.,0.,0.]);continue
        options.append([np.ones(len(ids),bool),np.zeros(len(ids),bool),split])
        ff=float(np.maximum(0,-c.s[pf]).sum()) if pf else np.inf
        bf=float(np.maximum(0,c.s[pb]).sum()) if pb else np.inf
        split_fee=(ff if split.any() else 0.)+(bf if (~split).any() else 0.)
        path_fees.append([ff,bf,split_fee])
    def labels(x):
        y=c.m0.copy()
        for j,ids in enumerate(pieces):y[ids]=options[j][x[j]]
        return y
    def objective(x):
        y=labels(x);value=c.energy(y)
        for j,ids in enumerate(pieces):
            value+=weight*float(-np.where(y[ids],lf[ids],lb[ids]).sum())/c.factor_scale
            if control!="no_path":value+=weight*path_fees[j][x[j]]/c.factor_scale
        return value
    starts=[np.full(len(pieces),2),np.zeros(len(pieces),int),np.ones(len(pieces),int)]
    starts=[x for x in starts if np.isfinite(objective(x))]
    if not starts:return dict(y=c.m0,info=dict(isolated_unknown=True,unknown_components=original_count))
    pairs=[(a,b) for a,b in combinations(range(len(pieces)),2) if np.any(np.isin(c.edges[:,0],pieces[a])&np.isin(c.edges[:,1],pieces[b])) or np.any(np.isin(c.edges[:,1],pieces[a])&np.isin(c.edges[:,0],pieces[b]))]
    x,info=finite_search(objective,starts,[range(3)]*len(pieces),pairs,tie_key=lambda v:(int(labels(v).sum()),tuple(labels(v)),tuple(v)))
    info.update(unknown_components=original_count,unknown_state_groups=len(pieces),bottleneck_path_calls=path_calls,unknown_points=int(unknown.sum()),unresolved_output_points=0)
    return dict(y=labels(x),info=info)


def _f267(c,weight,control=None):
    src=c.source
    tf=float(np.quantile(src.fsim[src.oof_valid].max(1),.75));tb=float(np.quantile(src.bsim[src.oof_valid].max(1),.75))
    collision=(c.fsim.max(1)>tf)&(c.bsim.max(1)>tb)
    pieces=_components(collision,c.hw)
    if not pieces:return dict(y=c.m0,info=dict(collision_components=0,module_active=False))
    modes=list(product(range(len(c.f)),range(len(c.b))))
    options=[];costs=[];cuts=0
    for ids in pieces:
        ys=[];fees=[]
        fa=int(c.fsim[ids].mean(0).argmax());ba=int(c.bsim[ids].mean(0).argmax())
        for a,b in modes:
            field=(c.fsim[:,a]-c.bsim[:,b])/.1
            y=_local_cut(c,ids,field);cuts+=1
            ys.append(y[ids]);fees.append(float(np.logaddexp(0,field[ids]).sum()-field[ids]@y[ids])+float(a!=fa or b!=ba))
        options.append(ys);costs.append(fees)
    def labels(x):
        y=c.m0.copy()
        for j,ids in enumerate(pieces):y[ids]=options[j][x[j]]
        return y
    def objective(x):return c.energy(labels(x))+weight*sum(costs[j][x[j]] for j in range(len(x)))/c.factor_scale
    independent=np.array([np.argmin(v) for v in costs],int)
    if control=="fixed_pair":return dict(y=labels(independent),info=dict(local_cuts=cuts,collision_components=len(pieces),fixed_pairs=True))
    pairs=[(a,b) for a,b in combinations(range(len(pieces)),2) if np.any(np.isin(c.edges[:,0],pieces[a])&np.isin(c.edges[:,1],pieces[b]))]
    x,info=finite_search(objective,[independent,np.zeros(len(pieces),int)],[range(len(modes))]*len(pieces),pairs,tie_key=lambda v:(int(labels(v).sum()),tuple(labels(v)),tuple(v)))
    info.update(local_cuts=cuts,collision_components=len(pieces),selected_mode_pairs=[list(modes[j]) for j in x],module_active=True)
    return dict(y=labels(x),info=info)


def _f268(c,weight,control=None,resources=None):
    observation=(resources or {}).get("f268_tail_observations")
    if observation is None:
        raise ResourceUnavailable("F268 requires saved pre-last-two-block states and real frozen QKV recomputation for source 8 pairs and query <=4 split/merge pairs, private special tokens per group (24 two-block tail replays)")
    required={"real_encoder_execution":True,"qkv_recomputed":True,"private_special_tokens":True,"tail_blocks":2,"native_dtype":"float32"}
    if any(observation.get(k)!=v for k,v in required.items()):
        raise ResourceUnavailable("F268 refuses fixed-QK, pooled endpoints or fake tail replay as its main arm")
    _verify_observation(c,observation)
    ref=np.asarray(observation["reference_responses"],np.float32)
    target=np.asarray(observation["reference_merge_role"],bool)
    query=np.asarray(observation["query_responses"],np.float32)
    edges=np.asarray(observation["query_edges"],int).reshape(-1,2)
    if ref.ndim!=2 or ref.shape[1]!=2 or len(ref)>8 or target.shape!=(len(ref),) or query.shape!=(len(edges),2) or len(edges)>4:
        raise ValueError("F268 true-response pair budget/shape violated")
    if not target.any() or target.all():raise ArithmeticError("F268 source response classes unidentifiable")
    distance=((query[:,None]-ref[None])**2).sum(2)
    bandwidth=float(np.median(((ref[:,None]-ref[None])**2).sum(2))) or 1.
    logkernel=-distance/max(bandwidth,EPS)
    logf=logsumexp(logkernel[:,target],axis=1)-np.log(target.sum())
    logb=logsumexp(logkernel[:,~target],axis=1)-np.log((~target).sum())
    # -log(1-p_merge) is a nonnegative disagreement fee; this keeps the
    # stipulated shared binary cut submodular without clipping signed edges.
    edgefee=np.logaddexp(0,logf-logb)/c.factor_scale
    if control=="ordinary_attention":edgefee=np.asarray(observation["ordinary_attention_edge_weights"],np.float32)
    if np.any(edgefee<0):raise ValueError("F268 cut needs nonnegative learned edge fees")
    y,certificate=_cut(c,extra_edges=(edges,weight*edgefee))
    info=dict(exact_cut_certificate=certificate)
    info.update(**required,tail_replays=2*(len(ref)+len(query)),extra_transformer_blocks=4*(len(ref)+len(query)),query_pairs=len(query),reference_pairs=len(ref),learned_edge_fees=edgefee.tolist())
    return dict(y=y,info=info)


def _neighborhood(hw):
    n=np.prod(hw);coords=np.array(np.unravel_index(np.arange(n),hw)).T
    result=[]
    for y,x in coords:
        result.append([int(yy*hw[1]+xx) for yy in range(max(0,y-1),min(hw[0],y+2)) for xx in range(max(0,x-1),min(hw[1],x+2)) if (yy,xx)!=(y,x)])
    return result


def _f269(c,weight,control=None):
    src=c.source;sf=np.argmax(src.fsim,1);sb=np.argmax(src.bsim,1)
    source_attribute=np.c_[sf,sb+len(c.f)]
    source_neighborhood=_neighborhood(src.hw)
    query_attribute=np.c_[np.argmax(c.fsim,1),np.argmax(c.bsim,1)+len(c.f)]
    query_neighborhood=_neighborhood(c.hw)
    attributes=[(0,j) for j in range(len(c.f))]+[(1,j+len(c.f)) for j in range(len(c.b))]
    rules=[]
    for (r1,a),(r2,b) in combinations(attributes,2):
        correct=np.zeros(4,int);wrong=np.zeros(4,int);hit=np.zeros(len(c.r),bool)
        for i,neighbors in enumerate(source_neighborhood):
            if not src.oof_valid[i]:continue
            aa=[j for j in neighbors if src.oof_valid[j] and src.blocks[j]==src.blocks[i] and source_attribute[j,r1]==a and (src.coverage[j]>=.5)==(r1==0)]
            bb=[j for j in neighbors if src.oof_valid[j] and src.blocks[j]==src.blocks[i] and source_attribute[j,r2]==b and (src.coverage[j]>=.5)==(r2==0)]
            hit[i]=any(j!=k for j in aa for k in bb)
        for conclusion in (0,1):
            for block in range(4):
                ids=hit&(src.blocks==block)
                correct[block]=np.count_nonzero(ids&((src.coverage>=.5)==conclusion))
                wrong[block]=np.count_nonzero(ids&((src.coverage>=.5)!=conclusion))
            if np.count_nonzero(correct>wrong)>=2 and correct.sum()>=8 and wrong.sum()>=8:
                probability=(correct.sum()+1)/(correct.sum()+wrong.sum()+2)
                rules.append((probability,r1,a,r2,b,conclusion,-np.log(probability)))
    rules.sort(key=lambda z:(-z[0],z[1:6]));rules=rules[:16]
    factors=[]
    for i,neighbors in enumerate(query_neighborhood):
        for _,r1,a,r2,b,conclusion,fee in rules:
            aa=[j for j in neighbors if query_attribute[j,r1]==a]
            bb=[j for j in neighbors if query_attribute[j,r2]==b]
            witnesses=[(j,k) for j in aa for k in bb if j!=k]
            if witnesses:factors.append((i,witnesses,1-r1,1-r2,conclusion,fee))
    def extra(y):
        value=0.
        for i,witnesses,first,second,conclusion,fee in factors:
            trigger=any(y[j]==first and y[k]==second for j,k in witnesses)
            if control=="unary":trigger=True
            if trigger and y[i]!=conclusion:value+=fee
        return weight*value/c.factor_scale
    pairs=[]
    for i,witnesses,_,_,_,_ in factors:
        for j,k in witnesses[:4]:pairs += [(i,j),(i,k),(j,k)]
    y,info=_y_search(c,extra,pairs=pairs)
    info.update(source_rules=len(rules),high_order_factors=len(factors),premises_require_distinct_sites=True,rule_variable_labels=control!="unary")
    return dict(y=y,info=info)


def _conflict_cores(c):
    # A minimal conflicting core is a pair of adjacent opposite signed mode
    # witnesses linked by a same-class soft clause, or a singleton possessing
    # opposing top-two F/B witnesses. Grow only while the contradiction remains.
    ftop=np.sort(c.fsim,axis=1)[:,-min(2,len(c.f)):]
    btop=np.sort(c.bsim,axis=1)[:,-min(2,len(c.b)):]
    positive=ftop[:,-1]-btop[:,0]
    negative=btop[:,-1]-ftop[:,0]
    candidates=[];subset_trials=0
    for i in np.flatnonzero((positive>0)&(negative>0)):
        candidates.append((float(min(positive[i],negative[i])),np.array([i],int)))
    for (a,b),fee in zip(c.edges,c.capacity):
        conflict=min(max(positive[a],0),max(negative[b],0),fee)+min(max(negative[a],0),max(positive[b],0),fee)
        if conflict>0:candidates.append((float(conflict),np.array([a,b],int)))
    # The remaining minimal inconsistent subsets of these equality/unit
    # clauses are induced paths between opposing single-sided witnesses.
    # Their internal sites must have no unit witness (otherwise a strict
    # subpath is already contradictory). Enumerate every such <=8-site path.
    graph=_adjacency(c.n,c.edges,c.capacity)
    fg_only=(positive>0)&(negative<=0);bg_only=(negative>0)&(positive<=0)
    neutral=(positive<=0)&(negative<=0)
    seen_paths=set()
    for root in np.flatnonzero(fg_only):
        stack=[([int(root)],0.)]
        while stack:
            path,edge_mass=stack.pop();subset_trials+=1
            if len(path)>=8:continue
            for nxt,fee in reversed(graph[path[-1]]):
                if fee<=0 or nxt in path:continue
                if any(nxt==j for p in path[:-1] for j,_ in graph[p]):continue # chord makes a smaller unsat subset
                extended=path+[nxt]
                if bg_only[nxt]:
                    key=tuple(sorted(extended))
                    if key not in seen_paths:
                        seen_paths.add(key)
                        score=float(positive[root]+negative[nxt]+edge_mass+fee)
                        candidates.append((score,np.array(extended,int)))
                elif neutral[nxt]:stack.append((extended,edge_mass+fee))
    candidates.sort(key=lambda x:(-x[0],tuple(x[1])))
    used=set();cores=[]
    for conflict,ids in candidates:
        if not any(int(i) in used for i in ids):cores.append((ids,conflict));used.update(map(int,ids))
    return cores,positive,negative,subset_trials


def _f270(c,weight,control=None):
    cores,positive,negative,subset_trials=_conflict_cores(c);trials=0
    if not cores:return dict(y=c.m0,info=dict(conflict_cores=0,conflict_subset_trials=subset_trials))
    options=[];costs=[];modes=[]
    for ids,_ in cores:
        choices=list(product((0,1),repeat=len(ids)));table=[];pairs=[]
        for bits in choices:
            labels=np.array(bits)
            abandoned=np.sum(np.where(labels,np.maximum(negative[ids],0),np.maximum(positive[ids],0)))
            best=None
            for a,b in product(range(len(c.f)),range(len(c.b))):
                logits=(c.fsim[ids,a]-c.bsim[ids,b])/.1
                data=float(np.logaddexp(0,logits).sum()-logits@labels);trials+=1
                item=(data+abandoned,a,b)
                if best is None or item<best:best=item
            if control=="independent_modes":
                data=0.
                for i,label in zip(ids,labels):
                    logits=(c.fsim[i,:,None]-c.bsim[i,None,:])/.1
                    data+=float(np.min(np.logaddexp(0,logits)-logits*label))
                best=(data+abandoned,-1,-1)
            table.append(float(best[0]));pairs.append([best[1],best[2]])
        options.append(np.asarray(choices,bool));costs.append(table);modes.append(pairs)
    def labels(x):
        y=c.m0.copy()
        for j,(ids,_) in enumerate(cores):y[ids]=options[j][x[j]]
        return y
    def objective(x):return c.energy(labels(x))+weight*sum(costs[j][state] for j,state in enumerate(x))/c.factor_scale
    starts=[]
    for seed in (c.m0,np.zeros(c.n,bool),np.ones(c.n,bool)):
        starts.append(np.array([next(k for k,bits in enumerate(options[j]) if np.array_equal(bits,seed[ids])) for j,(ids,_) in enumerate(cores)]))
    membership=np.full(c.n,-1,int)
    for j,(ids,_) in enumerate(cores):membership[ids]=j
    adjacent=sorted(set(tuple(sorted((int(membership[a]),int(membership[b])))) for a,b in c.edges if membership[a]>=0 and membership[b]>=0 and membership[a]!=membership[b]))
    x,info=finite_search(objective,starts,[range(len(table)) for table in options],adjacent,tie_key=lambda v:(int(labels(v).sum()),tuple(labels(v)),tuple(v)))
    info.update(conflict_cores=len(cores),core_sizes=[len(ids) for ids,_ in cores],conflict_subset_trials=subset_trials,enumerated_label_mode_trials=trials,selected_mode_pairs=[modes[j][state] for j,state in enumerate(x)],max_core_sites=8,actual_core_sites=max(len(ids) for ids,_ in cores),exact_each_mode_label_table=True)
    return dict(y=labels(x),info=info)


def _atlas_fit(c):
    src=c.source;atlases=[]
    for role,centers in ((1,c.f),(0,c.b)):
        valid=src.train_valid&((src.coverage>=.9) if role else (src.coverage<=.1))
        assignment=np.argmax(c.r@centers.T,1)
        for k,center in enumerate(centers):
            ids=np.flatnonzero(valid&(assignment==k))
            if not len(ids):continue
            delta=c.r[ids]-center
            _,_,vh=np.linalg.svd(delta,full_matrices=False);basis=vh[:min(4,len(ids),c.q.shape[1])]
            coordinate=delta@basis.T;bound=np.maximum(np.max(np.abs(coordinate),axis=0),EPS)
            atlases.append(dict(role=role,center=center,basis=basis,bound=bound,ids=ids))
    transitions={}
    for a,b in combinations(range(len(atlases)),2):
        aa,bb=atlases[a],atlases[b]
        if aa["role"]!=bb["role"]:continue
        # Shared native support: nearest-four source neighborhoods overlap.
        sima=c.r@aa["center"];simb=c.r@bb["center"]
        threshold_a=np.min(sima[aa["ids"]]);threshold_b=np.min(simb[bb["ids"]])
        shared=src.train_valid&(sima>=threshold_a)&(simb>=threshold_b)&((src.coverage>=.9) if aa["role"] else (src.coverage<=.1))
        if shared.sum()<max(2,aa["basis"].shape[0],bb["basis"].shape[0]):continue
        xa=(c.r[shared]-aa["center"])@aa["basis"].T;xb=(c.r[shared]-bb["center"])@bb["basis"].T
        transitions[(a,b)]=np.linalg.lstsq(xa,xb,rcond=EPS)[0]
        transitions[(b,a)]=np.linalg.lstsq(xb,xa,rcond=EPS)[0]
    return atlases,transitions


def _f271(c,weight,control=None):
    atlases,transitions=_atlas_fit(c);k=len(atlases)
    if not k:raise ArithmeticError("No pure-role atlas")
    coordinates=[];residual=[]
    for atlas in atlases:
        delta=c.q-atlas["center"]
        t=np.clip(delta@atlas["basis"].T,-atlas["bound"],atlas["bound"])
        coordinates.append(t);residual.append(np.sum((delta-t@atlas["basis"])**2,1))
    residual=np.asarray(residual).T;roles=np.array([a["role"] for a in atlases])
    negative=np.maximum(0,c.bsim.max(1)-c.fsim.max(1))
    coordinate_updates=0
    def objective(x):
        y=roles[x];value=c.energy(y)+weight*float(residual[np.arange(c.n),x].sum())/c.factor_scale
        for a,b in c.edges:
            ka,kb=int(x[a]),int(x[b])
            if roles[ka]==roles[kb]==1 and ka!=kb and (ka,kb) in transitions and control!="no_transition":
                error=coordinates[kb][b]-coordinates[ka][a]@transitions[(ka,kb)]
                value+=weight*(float(error@error)+float(max(negative[a],negative[b])))/c.factor_scale
        return value
    starts=[residual.argmin(1)]
    for y in (c.m0,np.zeros(c.n,bool),np.ones(c.n,bool)):
        assignment=np.array([min(np.flatnonzero(roles==label),key=lambda j:(float(residual[i,j]),j)) for i,label in enumerate(y.astype(int))]);starts.append(assignment)
    x,info=finite_search(objective,starts,[range(k)]*c.n,list(map(tuple,c.edges)),rounds=1,tie_key=lambda v:(int(roles[v].sum()),tuple(roles[v]),tuple(v)))
    # Conditional bounded least squares includes all active coordinate
    # transition terms; commit an update only if the complete objective falls.
    trace=[info["energy"]]
    for _ in range(9):
        old=objective(x);changed=False
        for i in range(c.n):
            ki=int(x[i]);atlas=atlases[ki];rank=len(atlas["basis"])
            if not rank:continue
            matrices=[atlas["basis"].T];targets=[c.q[i]-atlas["center"]]
            for a,b in c.edges:
                if i not in (a,b):continue
                j=int(b if a==i else a);kj=int(x[j])
                if roles[ki]==roles[kj]==1 and ki!=kj and control!="no_transition":
                    if (ki,kj) in transitions:matrices.append(transitions[(ki,kj)].T);targets.append(coordinates[kj][j])
                    if (kj,ki) in transitions:matrices.append(np.eye(rank));targets.append(coordinates[kj][j]@transitions[(kj,ki)])
            solved=lsq_linear(np.vstack(matrices),np.concatenate(targets),bounds=(-atlas["bound"],atlas["bound"]),method="bvls")
            if not solved.success or not np.isfinite(solved.x).all():raise ArithmeticError("F271 bounded coordinate solve failed")
            proposal=solved.x
            previous=coordinates[ki][i].copy();coordinates[ki][i]=proposal
            before=residual[i,ki];residual[i,ki]=np.sum((c.q[i]-atlas["center"]-proposal@atlas["basis"])**2)
            now=objective(x);coordinate_updates+=1
            if now<old-1e-9:old=now;changed=True
            else:coordinates[ki][i]=previous;residual[i,ki]=before
        x,newinfo=finite_search(objective,[x],[range(k)]*c.n,list(map(tuple,c.edges)),rounds=1,tie_key=lambda v:(int(roles[v].sum()),tuple(roles[v]),tuple(v)))
        info["objective_trials"]+=newinfo["objective_trials"];trace.append(objective(x))
        if not changed and trace[-1]>=trace[-2]-1e-9:break
    info.update(energy=float(objective(x)),atlases=k,defined_directed_transitions=len(transitions),bounded_ls_trials=coordinate_updates,alternating_objective_trace=trace,pca_rank=4)
    return dict(y=roles[x],info=info)


def _project_support(delta,center,bg,radius):
    """Euclidean projection into the source-radius ball and all B halfspaces."""
    from scipy.optimize import minimize
    delta=np.asarray(delta,np.float64);center=np.asarray(center,np.float64);bg=np.asarray(bg,np.float64)
    normals=bg-center
    bounds=.5*np.sum(normals*normals,1)
    if radius<=EPS:return np.zeros_like(delta)
    constraints=[dict(type="ineq",fun=lambda x:radius*radius-float(x@x),jac=lambda x:-2*x),
                 dict(type="ineq",fun=lambda x:bounds-normals@x,jac=lambda x:-normals)]
    solution=minimize(lambda x:.5*np.sum((x-delta)**2),np.zeros_like(delta,dtype=np.float64),jac=lambda x:x-delta,
                      constraints=constraints,method="SLSQP",options=dict(maxiter=50,ftol=1e-7))
    if not solution.success or np.linalg.norm(solution.x)>radius+1e-5 or np.any(normals@solution.x>bounds+1e-5):
        raise ArithmeticError("F272 support projection failed: "+str(solution.message))
    return np.asarray(solution.x,np.float32)


def _f272(c,weight,control=None):
    src=c.source;k=len(c.f)
    if not k or not len(c.b):raise ArithmeticError("Discriminative support needs both pure roles")
    source_mode=np.argmax(c.r@c.f.T,1)
    radius=[]
    for mode in range(k):
        means=[]
        for b in range(4):
            ids=src.fold_train[b]&(src.coverage>=.9)&(source_mode==mode)
            if ids.any():means.append(np.linalg.norm(c.r[ids].mean(0)-c.f[mode]))
        radius.append(float(np.quantile(means,.95)) if len(means)>=2 else 0.)
    radius=np.array(radius,np.float32)
    pieces=c.pieces
    delta=np.zeros((len(pieces),k,c.q.shape[1]),np.float32)
    matching_trials=projection_trials=0
    def conditional(y,shifts):
        nonlocal matching_trials
        residual=np.full(c.n,np.inf,np.float32);match_cost=0.;matches=[]
        for j,ids in enumerate(pieces):
            center=c.f+shifts[j]
            costs=np.sum((center[:,None]-c.q[ids][None])**2,2)
            permitted=(c.fsim[ids].T-c.bsim[ids].max(1)[None,:]>0)&y[ids][None,:].astype(bool)
            witness=np.where(permitted,costs,np.inf)
            # K private null columns make null possible for every source mode;
            # Hungarian permits each real point once and each mode once.
            null=np.ones((k,k),np.float32);matrix=np.c_[witness,null]
            row,col=linear_sum_assignment(matrix);matching_trials+=1
            match_cost+=float(matrix[row,col].sum());matches.append([(int(a),int(ids[b])) for a,b in zip(row,col) if b<len(ids)])
            local=costs.min(0)
            residual[ids]=np.minimum(residual[ids],local) # overlap pays once
        residual[~np.isfinite(residual)]=1-c.fsim.max(1)[~np.isfinite(residual)]
        bg=1-c.bsim.max(1)
        value=c.energy(y)+weight*(float(np.where(y,residual,bg).sum())+match_cost)/c.factor_scale
        return value,matches
    outputs=[]
    for initial in (c.m0,np.zeros(c.n,bool),np.ones(c.n,bool)):
        y=initial.copy();shifts=delta.copy();trace=[conditional(y,shifts)[0]];allinfo=dict(objective_trials=0)
        for _ in range(10):
            _,matches=conditional(y,shifts)
            proposed=shifts.copy()
            if control!="fixed_support":
                for j,pairs in enumerate(matches):
                    for mode,point in pairs:
                        proposed[j,mode]=_project_support(c.q[point]-c.f[mode],c.f[mode],c.b,radius[mode]);projection_trials+=1
            if conditional(y,proposed)[0]<conditional(y,shifts)[0]-1e-9:shifts=proposed
            objective=lambda x:conditional(x.astype(bool),shifts)[0]
            y,info=finite_search(objective,[y],[(0,1)]*c.n,list(map(tuple,c.edges)),rounds=1);y=y.astype(bool)
            allinfo["objective_trials"]+=info["objective_trials"];trace.append(conditional(y,shifts)[0])
            if trace[-1]>=trace[-2]-1e-9:break
        outputs.append((trace[-1],int(y.sum()),tuple(y),y,shifts,trace,allinfo))
    best=min(outputs,key=lambda v:v[:3]);info=best[-1]
    info.update(energy=best[0],objective_trace=best[-2],matching_trials=matching_trials,projected_ls_trials=projection_trials,
                source_displacement_radii=radius.tolist(),nonzero_support_shifts=int(np.count_nonzero(np.linalg.norm(best[4],axis=2)>EPS)),capacity="one strong witness per mode per piece; null permitted",overlap_cost_counted_once=True)
    return dict(y=best[3],info=info)


def _shared_private_dictionary(c):
    src=c.source;valid=src.train_valid
    x=c.r[valid];mean=x.mean(0);_,_,vh=np.linalg.svd(x-mean,full_matrices=False)
    f=c.r[valid&(src.coverage>=.9)];b=c.r[valid&(src.coverage<=.1)]
    shared=[]
    for direction in vh:
        if np.var(f@direction)>EPS and np.var(b@direction)>EPS:
            shared.append(direction)
            if len(shared)==4:break
    shared=np.array(shared,np.float32).reshape(-1,c.q.shape[1])
    private=[]
    for rows in (f,b):
        residual=rows-(rows@shared.T)@shared if len(shared) else rows.copy()
        nonzero=np.linalg.norm(residual,axis=1)>EPS;residual=residual[nonzero]
        if not len(residual):private.append(np.empty((0,c.q.shape[1]),np.float32));continue
        selected=[0]
        while len(selected)<min(8,len(residual)):
            distance=np.min(np.sum((residual[:,None]-residual[selected][None])**2,2),1);distance[selected]=-1
            selected.append(int(np.argmax(distance)))
        private.append(_unit(residual[selected]))
    return shared,private


def nonnegative_l1(q,dictionary,l1=1.,rounds=50):
    """Fixed <=50 cyclic coordinate NNLS sweeps with explicit L1 cost."""
    if not len(dictionary):return np.empty((len(q),0),np.float32),np.sum(q*q,1)
    gram=dictionary@dictionary.T;rhs=q@dictionary.T
    alpha=np.zeros_like(rhs,np.float32)
    for _ in range(rounds):
        old=alpha.copy()
        for j in range(len(dictionary)):
            alpha[:,j]=np.maximum(0,(rhs[:,j]-(alpha@gram[:,j]-alpha[:,j]*gram[j,j])-l1/2)/max(float(gram[j,j]),EPS))
        if np.max(np.abs(alpha-old))<EPS:break
    loss=np.sum(q*q,1)-2*np.sum(alpha*rhs,1)+np.sum((alpha@gram)*alpha,1)+l1*alpha.sum(1)
    return alpha,np.maximum(loss,0)


def _f273(c,weight,control=None):
    shared,private=_shared_private_dictionary(c)
    dictionaries=[np.r_[shared,p] for p in private]
    coefficients=[];loss=[];ratio=[]
    for dictionary in dictionaries:
        alpha,error=nonnegative_l1(c.q,dictionary,l1=1.)
        coefficients.append(alpha);loss.append(error)
        private_mass=alpha[:,len(shared):].sum(1)
        ratio.append(private_mass/np.maximum(alpha.sum(1),EPS))
    loss=np.array(loss).T;ratio=np.array(ratio).T
    active=(ratio.max(1)>EPS)
    nnls_trials=2*c.n
    def objective(y):
        role=1-y.astype(int) # dictionary order F, B
        value=c.energy(y)+weight*float(loss[np.arange(c.n),role][active].sum())/c.factor_scale
        if control!="independent":
            for a,b in c.edges:
                if y[a]==y[b] and active[a] and active[b]:value+=weight*float((ratio[a,role[a]]-ratio[b,role[b]])**2)/c.factor_scale
        return value
    domains=[(0,1) if active[i] else (int(c.m0[i]),) for i in range(c.n)]
    starts=[]
    for seed in (c.m0,np.zeros(c.n,bool),np.ones(c.n,bool)):
        initial=seed.copy();initial[~active]=c.m0[~active];starts.append(initial)
    y,info=finite_search(objective,starts,domains,list(map(tuple,c.edges)),rounds=1)
    # Re-solve coefficients against the active same-class responsibility
    # neighbors; accept only descent of the full objective. NNLS bound stays50.
    trace=[objective(y)]
    for _ in range(9):
        changed=False
        for i in range(c.n):
            if not active[i]:continue
            role=1-int(y[i]);dictionary=dictionaries[role];alpha=coefficients[role][i].copy()
            gram=dictionary@dictionary.T;rhs=dictionary@c.q[i]
            neighbors=[int(b if a==i else a) for a,b in c.edges if i in (a,b) and y[a]==y[b] and active[b if a==i else a]]
            if control=="independent" or not neighbors:continue
            before=objective(y)
            # Coordinate conditional quadratic+ratio optimization, exact in
            # each scalar via bounded minimization, not a frozen NNLS shortcut.
            from scipy.optimize import minimize_scalar
            def local(candidate):
                total=max(float(candidate.sum()),EPS);pr=float(candidate[len(shared):].sum()/total)
                reconstruction=float(c.q[i]@c.q[i]-2*candidate@rhs+candidate@gram@candidate+candidate.sum())
                return reconstruction+sum((pr-ratio[j,role])**2 for j in neighbors)
            for _ in range(50):
                old=alpha.copy()
                for j in range(len(alpha)):
                    def scalar(value):
                        trial=alpha.copy();trial[j]=value;return local(trial)
                    proposal=minimize_scalar(scalar,bounds=(0,2),method="bounded",options=dict(xatol=EPS))
                    if not proposal.success or not np.isfinite(proposal.x):raise ArithmeticError("F273 conditional coefficient solve failed")
                    alpha[j]=proposal.x
                if np.max(np.abs(alpha-old))<EPS:break
            nnls_trials+=1
            oldalpha=coefficients[role][i].copy();olderror=loss[i,role];oldratio=ratio[i,role]
            coefficients[role][i]=alpha
            loss[i,role]=float(c.q[i]@c.q[i]-2*alpha@rhs+alpha@gram@alpha+alpha.sum())
            ratio[i,role]=alpha[len(shared):].sum()/max(float(alpha.sum()),EPS)
            if objective(y)<before-1e-9:changed=True
            else:coefficients[role][i]=oldalpha;loss[i,role]=olderror;ratio[i,role]=oldratio
        y,nextinfo=finite_search(objective,[y],domains,list(map(tuple,c.edges)),rounds=1)
        info["objective_trials"]+=nextinfo["objective_trials"];trace.append(objective(y))
        if not changed and trace[-1]>=trace[-2]-1e-9:break
    # A zero-private point is fixed back to the complete baseline, as required.
    y[~active]=c.m0[~active]
    info.update(energy=float(objective(y)),shared_directions=len(shared),private_atoms=[len(p) for p in private],nnls_point_solves=nnls_trials,l1_fee=1.,zero_private_points=int((~active).sum()),alternating_objective_trace=trace)
    return dict(y=y,info=info)


def _representatives(q,maximum=32):
    if len(q)<=maximum:return q.copy()
    selected=[0]
    while len(selected)<maximum:
        distance=1-(q@q[selected].T).max(1);distance[selected]=-np.inf
        selected.append(int(distance.argmax()))
    return q[selected]


def _f274(c,weight,control=None):
    src=c.source;f=c.r[(src.coverage>=.9)&src.train_valid];b=c.r[(src.coverage<=.1)&src.train_valid]
    if not len(f) or not len(b):raise ArithmeticError("Background hold-block likelihood needs pure roles")
    coords=np.array(np.unravel_index(np.arange(c.n),c.hw)).T
    blocks=(coords[:,0]>=c.hw[0]/2)*2+(coords[:,1]>=c.hw[1]/2)
    negative=c.bsim.max(1)-c.fsim.max(1)>0
    lf=logsumexp(c.q@f.T/.1,axis=1)-np.log(len(f))
    scores=[];builds=0
    fixed=np.logical_and.reduce([~y for y in c.candidates])&negative if c.candidates else negative
    for hypothesis in c.candidates:
        value=0.
        for block in range(4):
            evaluate=blocks==block
            eligible=(~hypothesis)&negative
            if control=="fixed_library":eligible=fixed
            if control!="self_fit":eligible&=blocks!=block
            if control=="reference_only":eligible[:]=False
            representatives=_representatives(c.q[eligible],32)
            library=np.r_[b,representatives];builds+=1
            lb=logsumexp(c.q[evaluate]@library.T/.1,axis=1)-np.log(len(library))
            value+=float(-np.where(hypothesis[evaluate],lf[evaluate],lb).sum())
        boundary=float(c.capacity@(hypothesis[c.edges[:,0]]!=hypothesis[c.edges[:,1]]))
        scores.append(weight*value/c.factor_scale+boundary+(1-weight)*c.energy(hypothesis))
    selected=min(range(len(scores)),key=lambda j:(float(scores[j]),int(c.candidates[j].sum()),j))
    return dict(y=c.candidates[selected],info=dict(candidate_scores=list(map(float,scores)),selected_candidate=selected,background_library_builds=builds,max_query_bg_representatives=32,leave_query_block_out=control!="self_fit",reference_bg_retained=True,equal_class_prior=True))


def _f275(c,weight,control=None):
    # Edge coarse blocks can contain <4 children; enumerate their actual binary
    # states without padding children or treating padding as negative labels.
    grid=np.arange(c.n).reshape(c.hw);blocks=[]
    for yy in range(0,c.hw[0],2):
        for xx in range(0,c.hw[1],2):blocks.append(grid[yy:yy+2,xx:xx+2].ravel())
    roles=[(None,-1)]+[(1,j) for j in range(len(c.f))]+[(0,j) for j in range(len(c.b))]
    states=[];fees=[]
    for ids in blocks:
        options=[];costs=[]
        for bits in product((0,1),repeat=len(ids)):
            labels=np.array(bits,bool)
            for role,mode in roles:
                if role==1 and (not labels.any() or not np.any((c.fsim[ids].max(1)-c.bsim[ids].max(1)>0)&labels)):continue
                selected=ids[labels] if role==1 else ids[~labels] if role==0 else np.array([],int)
                if role is not None and not len(selected):continue
                center=c.f[mode] if role==1 else c.b[mode] if role==0 else None
                matching=0. if role is None else max(0.,1-float(_unit(c.q[selected].mean(0))@center))/.1
                # Fine negative evidence is paid point by point. A parent F
                # role neither forces a strong-B child nor authorizes it.
                negative=float(np.maximum(0,c.bsim[ids].max(1)-c.fsim[ids].max(1))[labels].sum())/.1
                options.append((labels,role,mode));costs.append(matching+negative)
        states.append(options);fees.append(costs)
    membership=np.empty(c.n,int)
    for j,ids in enumerate(blocks):membership[ids]=j
    coarse_edges=sorted(set(tuple(sorted((int(membership[a]),int(membership[b])))) for a,b in c.edges if membership[a]!=membership[b]))
    def labels(x):
        y=np.zeros(c.n,bool)
        for j,ids in enumerate(blocks):y[ids]=states[j][x[j]][0]
        return y
    def objective(x):
        y=labels(x);extra=sum(fees[j][x[j]] for j in range(len(x)))
        if control!="no_correspondence":
            for a,b in coarse_edges:
                _,ra,ma=states[a][x[a]];_,rb,mb=states[b][x[b]]
                if ra is not None and rb==ra:extra+=float(ma!=mb)
        return c.energy(y)+weight*extra/c.factor_scale
    starts=[]
    for y in (c.m0,np.zeros(c.n,bool),np.ones(c.n,bool)):
        starts.append(np.array([min((k for k,(bits,_,_) in enumerate(options) if np.array_equal(bits,y[ids])),key=lambda k:(fees[j][k],k)) for j,(ids,options) in enumerate(zip(blocks,states))]))
    x,info=finite_search(objective,starts,[range(len(options)) for options in states],coarse_edges,tie_key=lambda v:(int(labels(v).sum()),tuple(labels(v)),tuple(v)))
    info.update(coarse_blocks=len(blocks),max_states_per_block=max(map(len,states)),fine_labels_jointly_optimized=True,coarse_matching_recomputed_from_current_children=True)
    return dict(y=labels(x),info=info)


CONTROLS={251:"fixed_z",252:"independent",253:"independent",254:"shortest_only",255:"unary_dp",
          256:"diagonal",257:"no_open_fee",258:"greedy_cover",259:"all_off",260:"fixed_bands",
          261:"mean_field",262:"fixed_boundary",263:"no_interaction",264:"source_best",
          265:"force_one",266:"no_path",267:"fixed_pair",268:"ordinary_attention",
          269:"unary",270:"independent_modes",271:"no_transition",272:"fixed_support",273:"independent",
          274:"reference_only",275:"no_correspondence"}


def _reference_scale(c,number):
    """MAD of the card's source factor primitives, using training roles only."""
    src=c.source;valid=src.train_valid
    fs=(c.r@c.f.T)[valid] if len(c.f) else np.empty((valid.sum(),0))
    bs=(c.r@c.b.T)[valid] if len(c.b) else np.empty((valid.sum(),0))
    if not len(fs) or not fs.shape[1] or not bs.shape[1]:return EPS
    margin=fs.max(1)-bs.max(1)
    if number==251:
        advantage=bs-fs.max(1)[:,None];samples=np.r_[np.maximum(advantage,0).ravel(),1-advantage.mean(0)]
    elif number in (252,255,259,260,272):samples=np.r_[np.maximum(margin,0),np.maximum(-margin,0)]
    elif number in (253,258,265,269):samples=np.r_[0.,1.,np.maximum(margin,0)]
    elif number in (261,263,264,267,275):samples=np.logaddexp(0,margin/.1)
    elif number in (262,271):samples=np.r_[1-fs.max(1),1-bs.max(1)]
    elif number in (266,274):
        lf,lb=_profile_density(c.r[valid],c.r,src.coverage,src.valid*valid)
        samples=np.r_[-lf,-lb]
    elif number==257:samples=np.maximum(0,-src.heads[src.oof_valid,0])
    elif number==273:samples=np.r_[1-fs.max(1),1-bs.max(1),np.maximum(margin,0)**2]
    else:samples=src.heads[src.oof_valid,0]
    return _mad(samples)


def run(number,ep,*,weight=None,control=None,resources=None):
    """Complete legal wrapper; an explicit weight is a reported fixed trial."""
    from . import common
    from .f_protocol import build_context,calibrate_weight,exact_e0,kernel_result,WEIGHTS
    if number not in range(251,276):raise ValueError("This module owns F251--F275")
    c=build_context(ep,resources=resources)
    pure_required=number not in (253,254,255,257,258,261)
    if c.trace["empty_reference_fg"] or c.trace["single_class"] or (pure_required and any(c.trace["missing_pure_roles"])):
        reason="empty_reference_fg" if c.trace["empty_reference_fg"] else "single_class_negative_factors_disabled" if c.trace["single_class"] else "pure_role_factor_disabled"
        return common.Result(c.p0.reshape(c.hw),.5,info=dict(c.trace,method=f"F{number:03d}",quality="unknown",fallback=reason,query_GT_read=False,selected_weight=0.))
    if number in (256,268):
        name="f256_three_views" if number==256 else "f268_tail_observations"
        if resources is None:
            resources={name:common.artifact(ep,name)}
        # Verify the true observation before treating source calibration as an
        # implementation. Label-derived source folds must be separately rebuilt.
        try:getattr(__import__(__name__,fromlist=["x"]),f"_f{number}")(c,1.,control,resources)
        except ResourceUnavailable as error:raise common.ArtifactUnavailable(str(error)) from error
        if weight is None:
            raise common.ArtifactUnavailable(f"F{number} additionally requires per-buffered-source-fold real observations rebuilt without held labels; native observations cannot substitute")
    def kernel(context,w):
        context.factor_scale=_reference_scale(context,number)
        if w==0:
            y,certificate=exact_e0(context)
            return dict(y=y,info=dict(primary_factor_disabled=True,exact_E0_certificate=certificate))
        function=globals()[f"_f{number}"]
        if number in (256,268):return function(context,w,control,resources)
        if number in (253,263,264,265,266,267,269,272) and context.trace.get("source_oof_fit_calls",0)<2:
            return dict(probability=context.p0,info=dict(fallback="fewer_than_two_source_OOF_fits",primary_factor_disabled=True))
        try:return function(context,w,control)
        except (ArithmeticError,np.linalg.LinAlgError) as error:
            return dict(probability=context.p0,info=dict(fallback="nonidentifiable_or_numerical_failure",fallback_detail=str(error),primary_factor_disabled=True))
    if weight is None:
        weight,calibration=calibrate_weight(ep,kernel,resources=resources)
    else:
        if float(weight) not in WEIGHTS:raise ValueError("Primary weight must be in {0,.25,1}")
        calibration=dict(calibration="explicit_fixed_weight_trial",selected_weight=float(weight),query_GT_read=False)
    out=kernel(c,float(weight));out.setdefault("info",{}).update(selected_weight=float(weight),calibration=calibration,
            control=control,normalization="source-only MAD of named card factor primitives",source_factor_MAD=c.factor_scale,
            implementation_assumptions_file="evidence/local/astra300_20261007/group_226_300/f251_275/assumptions.json")
    return kernel_result(ep,c,out,method=f"F{number:03d}")


def _public(number):
    def method(ep,**kwargs):return run(number,ep,**kwargs)
    method.__name__=f"F{number:03d}"
    return method


METHODS={f"F{number:03d}":_public(number) for number in range(251,276)}
globals().update(METHODS)
