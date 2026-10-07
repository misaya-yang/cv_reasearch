"""Frozen Astra cards F276--F300: NumPy/SciPy kernels, never query labels.

The public adapter at the end uses astra300.common for shared candidates,
calibration, E0 and the single original-resolution readout. Resource kernels
require observed final-LN vectors/real image views; missing resources are errors,
not an imitation using native final descriptors. See the companion assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations, product
import heapq

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment, nnls
from scipy.special import expit, logsumexp

EPS = 1e-6


def unit(a):
    a = np.asarray(a, float)
    z = np.linalg.norm(a, axis=-1, keepdims=True)
    return np.divide(a, z, out=np.zeros_like(a), where=z > EPS)


def mad(a):
    a = np.asarray(a, float)
    return np.maximum(np.median(np.abs(a - np.median(a, axis=0)), axis=0), EPS)


def ridge(x, y, alpha=1., rank=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mu = x.mean(0)
    xc = x - mu
    u, s, v = np.linalg.svd(xc, full_matrices=False)
    k = len(s) if rank is None else min(int(rank), len(s))
    coef = (v[:k].T * (s[:k] / (s[:k] ** 2 + alpha))) @ (u[:, :k].T @ (y - y.mean(0)))
    return mu, y.mean(0), coef


def predict(model, x):
    mu, intercept, coef = model
    return (np.asarray(x) - mu) @ coef + intercept


def pca(x, rank=16):
    center = x.mean(0)
    _, s, v = np.linalg.svd(x - center, full_matrices=False)
    keep = s > EPS
    return center, v[keep][:rank]


def density(x, bank, width=.1):
    if not len(bank):
        raise ValueError('A class density needs observed reference samples')
    return logsumexp(unit(x) @ unit(bank).T / width, axis=1) - np.log(len(bank))


def kernel_field(x, rf, rb):
    return density(x, rf) - density(x, rb)


def modes(x, maximum=8):
    """Specified row-first/farthest spherical initialization, 20 Lloyd steps."""
    x = unit(x)
    if not len(x):
        return np.empty((0, x.shape[1])), np.empty(0, int)
    seeds = [0]
    near = x @ x[0]
    for _ in range(1, min(maximum, len(x))):
        i = int(np.argmin(near))
        if near[i] >= 1 - EPS:
            break
        seeds.append(i)
        near = np.maximum(near, x @ x[i])
    centers = x[seeds]
    for _ in range(20):
        labels = (x @ centers.T).argmax(1)
        live = np.unique(labels)
        centers = unit(np.stack([x[labels == k].mean(0) for k in live]))
    labels = (x @ centers.T).argmax(1)
    live = np.unique(labels)
    return centers[live], np.searchsorted(live, labels)


def geometric_median(x):
    if not len(x):
        return None
    center = np.median(x, axis=0)
    for _ in range(50):
        d = np.linalg.norm(x - center, axis=1)
        if np.any(d < EPS):
            return x[int(np.argmin(d))].copy()
        w = 1 / d
        moved = np.sum(w[:, None] * x, axis=0) / w.sum()
        if np.linalg.norm(moved - center) <= EPS:
            return moved
        center = moved
    return center


def neighbors(hw):
    a = np.arange(np.prod(hw)).reshape(hw)
    return np.concatenate((np.c_[a[:-1].ravel(), a[1:].ravel()],
                           np.c_[a[:, :-1].ravel(), a[:, 1:].ravel()]))


def quarters(hw):
    yy, xx = np.indices(hw)
    return ((yy >= hw[0] / 2) * 2 + (xx >= hw[1] / 2)).ravel()


def buffered_folds(hw, valid):
    groups = quarters(hw)
    out = []
    for k in range(4):
        hold = (groups == k) & valid
        forbidden = ndimage.binary_dilation(hold.reshape(hw), structure=np.ones((3, 3))).ravel()
        train = valid & ~forbidden
        if hold.any() and train.any():
            out.append((train, hold))
    return out


@dataclass
class Context:
    q: np.ndarray
    r: np.ndarray
    c: np.ndarray
    valid: np.ndarray
    hw: tuple
    rhw: tuple
    s: np.ndarray
    p0: np.ndarray
    fm: np.ndarray
    bm: np.ndarray
    edges: np.ndarray
    edge_weight: np.ndarray
    Q: list
    P: list
    ep: object = None
    resources: object = None
    trace: dict = field(default_factory=dict)

    @property
    def pure(self):
        return self.valid & ((self.c >= .9) | (self.c <= .1))

    @property
    def F(self):
        return self.valid & (self.c >= .9)

    @property
    def B(self):
        return self.valid & (self.c <= .1)

    def energy(self, y, s=None, edges=None, weights=None):
        s = self.s if s is None else np.asarray(s).ravel()
        edges = self.edges if edges is None else edges
        weights = self.edge_weight if weights is None else weights
        y = np.asarray(y, bool).ravel()
        return float(np.logaddexp(0, s).sum() - s @ y + np.sum(weights * (y[edges[:, 0]] != y[edges[:, 1]])))


@dataclass
class KernelResult:
    probability: np.ndarray | None = None
    labels: np.ndarray | None = None
    unary: np.ndarray | None = None
    info: dict = field(default_factory=dict)


def result(c, labels=None, score=None, **info):
    info.update(quality='unknown', query_GT_read=False)
    return KernelResult(probability=None if score is None else expit(score),
                        labels=None if labels is None else np.asarray(labels, bool),
                        unary=c.s if score is None else np.asarray(score), info=info)


def finite_search(objective, starts, factors, rounds=10):
    """S: full objective, deterministic singles and factor-neighbor doubles."""
    evaluations, accepted = 0, 0
    winner = None
    best = np.inf
    pairs = set()
    for group in factors:
        group = sorted(set(map(int, group)))
        for a in group:
            for b in [v for v in group if v > a][:4]:
                pairs.add((a, b))
    for start in starts:
        state = np.asarray(start, bool).copy()
        cost = float(objective(state)); evaluations += 1
        for _ in range(rounds):
            changed = False
            for flip in [(i,) for i in range(len(state))] + sorted(pairs):
                test = state.copy(); test[list(flip)] ^= True
                candidate = float(objective(test)); evaluations += 1
                if candidate < cost - 1e-12:
                    state, cost = test, candidate
                    accepted += 1; changed = True
            if not changed:
                break
        key = (cost, int(state.sum()), tuple(state))
        previous = (best, int(winner.sum()), tuple(winner)) if winner is not None else (np.inf, np.inf, ())
        if key < previous:
            winner, best = state, cost
    return winner, dict(objective=float(best), objective_evaluations=evaluations,
                       accepted_moves=accepted, solver='S finite descent; not global optimum')


def search_labels(c, objective, auxiliary=0, starts=None, rounds=10, factors=None):
    n = len(c.q)
    starts = [c.s > 0, np.zeros(n, bool), np.ones(n, bool)] if starts is None else starts
    seeds = [np.r_[v, np.zeros(auxiliary, bool)] for v in starts]
    group = [list(edge) for edge in c.edges] + [list(np.flatnonzero(p)) for p in c.P]
    if auxiliary:
        group += [list(range(n, n + auxiliary))]
    if factors is not None:
        group += factors
    return finite_search(objective, seeds, group, rounds)


def cut(c, score, weights=None, edges=None):
    # Float capacities: do not silently quantize the claimed exact objective.
    from ics.methods.pro_paired_environment import exact_potts_cut
    y, certificate = exact_potts_cut(np.asarray(score).ravel(), c.edges if edges is None else edges,
                                    c.edge_weight if weights is None else np.asarray(weights))
    c.trace['cut_calls'] = c.trace.get('cut_calls', 0) + 1
    return y


def _pure_roles(c):
    if not c.F.any() or not c.B.any():
        raise ValueError('Pure F/B roles unavailable: disable this role factor through shared common')


def _heads(c):
    mean = c.s
    nearest = (np.max(c.q @ c.r[c.F].T, axis=1) - np.max(c.q @ c.r[c.B].T, axis=1)) / .1
    kernel = kernel_field(c.q, c.r[c.F], c.r[c.B])
    return np.c_[mean, nearest, kernel]


def _adj(c):
    return [np.r_[i, c.edges[c.edges[:, 0] == i, 1], c.edges[c.edges[:, 1] == i, 0]]
            for i in range(len(c.q))]


def _piece_ownership(pieces, hw):
    ids = np.full(np.prod(hw), -1, int)
    yy, xx = np.indices(hw); xy = np.c_[yy.ravel(), xx.ravel()]
    best = np.full(len(ids), np.inf)
    for j, piece in enumerate(pieces):
        if not piece.any():
            continue
        center = xy[piece].mean(0)
        d = ((xy - center) ** 2).sum(1)
        chosen = piece & (d < best)
        ids[chosen], best[chosen] = j, d[chosen]
    return ids


def _bg_chain(c, a, b):
    """A visible B-positive chain must cross the corridor between two pieces."""
    if not a.any() or not b.any():
        return False
    bg = (c.s < 0).reshape(c.hw)
    components, count = ndimage.label(bg)
    near_a = ndimage.binary_dilation(a.reshape(c.hw), iterations=1)
    near_b = ndimage.binary_dilation(b.reshape(c.hw), iterations=1)
    shared = np.intersect1d(np.unique(components[near_a]), np.unique(components[near_b]))
    return bool(np.any(shared > 0))


def _mode_tree(c):
    k = len(c.fm)
    labels = (c.r @ c.fm.T).argmax(1)
    affinity = np.zeros((k, k))
    for i, j in neighbors(c.rhw):
        if c.F[i] and c.F[j] and labels[i] != labels[j]:
            affinity[labels[i], labels[j]] += 1
            affinity[labels[j], labels[i]] += 1
    parent = np.full(k, -1, int); live = {0}
    while len(live) < k:
        choices = [(float(-affinity[a, b]), float(1 - c.fm[a] @ c.fm[b]), a, b)
                   for a in sorted(live) for b in range(k) if b not in live]
        _, _, a, b = min(choices); parent[b] = a; live.add(b)
    return parent


def tree_gap_match(c, y, gap=True):
    """Exact tree DP with null states and explicit matched-neighbor chain ports.

    Pairwise recurrence assigns a query proposal to each mode. A missing mode
    may choose a pair of directly adjacent *matched* mode ports, with the pair
    required to be joined by a B-positive chain; its deletion cost is discounted
    only for that port state. Port enumeration is counted, not claimed linear.
    """
    parent = _mode_tree(c); k = len(parent)
    pieces = [p & y for p in c.P]
    # Each proposal keeps all its observed points, with no invented null FG.
    unary = np.array([[float(np.mean(1 - c.q[p] @ f)) if p.any() else np.inf for p in pieces]
                      for f in c.fm])
    domains = [list(np.argsort(row, kind='stable')[:min(8, len(pieces))]) + [-1] for row in unary]
    children = [[j for j in range(k) if parent[j] == i] for i in range(k)]
    chain = {}
    def joined(a, b):
        key = tuple(sorted((a, b)))
        if key not in chain:
            chain[key] = _bg_chain(c, pieces[a], pieces[b])
        return chain[key]
    evaluations = 0
    @lru_cache(None)
    def dp(node, assigned, ancestor):
        nonlocal evaluations
        evaluations += 1
        base = 1. if assigned < 0 else unary[node, assigned]
        choice = None; best = np.inf
        # Node-star factors create a tree junction recurrence (finite states).
        for assignments in product(*(domains[j] for j in children[node])):
            cost = base + sum(dp(j, slot, assigned)[0] for j, slot in zip(children[node], assignments))
            if assigned < 0 and gap:
                ports = [v for v in (ancestor,) + assignments if v >= 0]
                if any(joined(a, b) for a, b in combinations(ports, 2)):
                    cost -= .75  # Fixed gap fee .25 instead of deletion fee 1.
            if assigned >= 0:
                for slot in assignments:
                    if slot >= 0:
                        cost += .25 * (slot == assigned)  # graph-edit collision, not global capacity
            if cost < best:
                best, choice = cost, assignments
        return best, choice
    roots = [(dp(0, a, -1)[0], a) for a in domains[0]]
    best, root = min(roots)
    assignment = np.full(k, -1, int)
    def recover(node, a, ancestor):
        assignment[node] = a
        for j, b in zip(children[node], dp(node, a, ancestor)[1]):
            recover(j, b, a)
    recover(0, root, -1)
    return float(best), assignment, evaluations


def f276(c, weight=1., control=False):
    _pure_roles(c); n = len(c.q); calls = 0
    def objective(state):
        nonlocal calls
        y = state[:n]
        fee, matched, trials = tree_gap_match(c, y, not control); calls += trials
        supported = np.zeros(n, bool)
        for slot in matched[matched >= 0]:
            supported |= c.P[slot] & y
        # Unmatched positive points have an explicit independent F cost.
        independent = np.sum(y & ~supported) * .25
        return c.energy(y) + weight * (fee + independent)
    state, info = search_labels(c, objective)
    return result(c, labels=state[:n], matching_state_evaluations=calls,
                  mechanism='F276_visible_gap_tree_match', control='uniform_deletion' if control else None, **info)


def _mutual_shift(q, r):
    if not len(q) or not len(r):
        return None
    sim = q @ r.T
    qr, rq = sim.argmax(1), sim.argmax(0)
    qi = np.flatnonzero(rq[qr] == np.arange(len(q)))
    return geometric_median(q[qi] - r[qr[qi]])


def _domain_shift(c, y, remove=False):
    rf = c.r[c.F]
    if remove:
        labels = (rf @ c.fm.T).argmax(1)
        biggest = int(np.argmax(np.bincount(labels, minlength=len(c.fm))))
        rf = rf[labels != biggest]
    df = _mutual_shift(c.q[y], rf)
    db = _mutual_shift(c.q[~y], c.r[c.B])
    if df is None or db is None or df @ db <= 0:
        return np.zeros(c.q.shape[1])
    direction = unit(unit(df) + unit(db))
    blocks = quarters(c.rhw)
    shifts = []
    for a, b in combinations(range(4), 2):
        if (c.valid & (blocks == a)).any() and (c.valid & (blocks == b)).any():
            shifts.append(np.linalg.norm(c.r[c.valid & (blocks == a)].mean(0) - c.r[c.valid & (blocks == b)].mean(0)))
    cap = float(np.median(shifts)) if shifts else 0.
    return direction * min(cap, max(0., float((df @ direction + db @ direction) / 2)))


def f277(c, weight=1., control=False):
    _pure_roles(c); n = len(c.q)
    def objective(y):
        shifts = [_domain_shift(c, y)] if control else [_domain_shift(c, y), _domain_shift(c, y, True)]
        fees = []
        for delta in shifts:
            moved = unit(c.q - delta)
            df = 1 - np.max(moved @ c.fm.T, axis=1)
            db = 1 - np.max(moved @ c.bm.T, axis=1)
            fees.append(np.where(y, df, db))
        # Renormalization is essential: this is not a constant logit shift.
        fit = np.max(fees, axis=0).sum()
        alignment = np.sum(np.linalg.norm(unit(c.q[~y] - shifts[0]) - c.q[~y], axis=1))
        return c.energy(y) + weight * (fit + alignment)
    state, info = search_labels(c, objective, starts=c.Q)
    return result(c, labels=state[:n], mechanism='F277_leave_dominant_double_side_renormalized', **info)


def f278(c, weight=1., control=False):
    _pure_roles(c); n, l = len(c.q), len(c.P); h = _heads(c)
    risk = np.array([[np.logaddexp(0, -h[p, j]).mean() for j in range(3)] for p in c.P])
    conflicts = np.array([np.logaddexp(0, h[p]).mean() for p in c.P])
    def objective(state):
        y, z = state[:n], state[n:]
        cover = np.zeros(n, bool)
        for j in np.flatnonzero(z):
            cover |= c.P[j]
        active = risk[z]
        worst = 0. if not len(active) else (float(np.mean(active)) if control else float(np.max(active)))
        conflict = sum(float(np.mean(np.maximum(-c.s[p & y], 0))) if (p & y).any() else 0. for p in np.asarray(c.P)[z])
        return c.energy(y) + weight * (worst + conflict + .25 * z.sum() + .25 * np.sum(y & ~cover))
    state, info = search_labels(c, objective, auxiliary=l)
    return result(c, labels=state[:n], active_pieces=int(state[n:].sum()),
                  mechanism='F278_instance_worst_head_open_close', **info)


def private_dictionary(c):
    _, axes = pca(c.r[c.valid], 4)
    axes = np.array([a for a in axes if np.var(c.r[c.F] @ a) > EPS and np.var(c.r[c.B] @ a) > EPS])
    if axes.size == 0:
        axes = np.empty((0, c.q.shape[1]))
    residual = c.r - (c.r @ axes.T) @ axes
    # Row-first/farthest chooses real residual atoms; no learned rotation.
    def atoms(mask):
        x = unit(residual[mask]); live = np.linalg.norm(x, axis=1) > EPS; x = x[live]
        if not len(x):
            return x
        selected = [0]; near = x @ x[0]
        for _ in range(1, min(8, len(x))):
            j = int(np.argmin(near))
            if near[j] >= 1 - EPS:
                break
            selected.append(j); near = np.maximum(near, x @ x[j])
        return x[selected]
    return axes, atoms(c.F), atoms(c.B)


def _nonnegative_fit(x, dictionary, l1=.01):
    if not len(dictionary):
        return np.zeros((len(x), 0)), np.sum(x * x, axis=1)
    # Quadratic residual + fixed L1, using augmented NNLS coordinate descent.
    gram, target = dictionary @ dictionary.T, x @ dictionary.T
    coef = np.zeros_like(target)
    for _ in range(50):
        for k in range(len(dictionary)):
            coef[:, k] = np.maximum(0, coef[:, k] + (target[:, k] - coef @ gram[:, k] - l1 / 2) / max(gram[k, k], EPS))
    errors = np.sum((x - coef @ dictionary) ** 2, axis=1) + l1 * coef.sum(1)
    return coef, errors


def f279(c, weight=1., control=False):
    _pure_roles(c); shared, fa, ba = private_dictionary(c)
    df, db = np.r_[shared, fa], np.r_[shared, ba]
    cf, ef = _nonnegative_fit(c.q, df); cb, eb = _nonnegative_fit(c.q, db)
    evidence = eb - ef
    eligible = []
    blocks = quarters(c.hw)
    for p in c.P:
        witnesses = [p & (blocks == k) & (evidence > 0) for k in range(4)]
        # Distinct, diagonally separated 2x2 regions; copying a token cannot pass.
        good = (witnesses[0].any() and witnesses[3].any()) or (witnesses[1].any() and witnesses[2].any())
        if control:
            good = any(w.any() for w in witnesses)
        eligible.append(bool(good and len(shared)))
    n, l = len(c.q), len(c.P)
    ownership = _piece_ownership(c.P, c.hw)
    last_delta = np.zeros((l, len(shared)))
    def objective(state):
        y, z = state[:n], state[n:]
        if np.any(z & ~np.asarray(eligible)):
            return np.inf
        cost = np.where(y, ef, eb).copy()
        for j in np.flatnonzero(z):
            p = ownership == j
            if not p.any():
                continue
            # Shared coefficient offset is fitted jointly using the current labels.
            chosen = np.where(y[p, None], cf[p, :len(shared)], cb[p, :len(shared)])
            delta = np.maximum(0, np.mean(c.q[p] @ shared.T - chosen, axis=0))
            delta = np.clip(delta, 0, 1)
            last_delta[j] = delta
            moved = c.q[p] - delta @ shared
            _, ff = _nonnegative_fit(moved, df); _, bb = _nonnegative_fit(moved, db)
            cost[p] = np.where(y[p], ff, bb)
        return c.energy(y) + weight * (cost.sum() + .25 * z.sum())
    state, info = search_labels(c, objective, auxiliary=l)
    objective(state)
    return result(c, labels=state[:n], mechanism='F279_private_code_local_shared_birth',
                  active_pieces=int(state[n:].sum()), shared_rank=len(shared),
                  private_atoms=[len(fa), len(ba)], delta_norms=np.linalg.norm(last_delta, axis=1).tolist(), **info)


def _neighborhood_modes(x, hw, fm, bm, selected=None, pieces=None):
    response = np.c_[x @ fm.T, x @ bm.T]
    ids = np.arange(len(x)).reshape(hw); out = np.zeros_like(response)
    groups = [np.r_[ids[max(0, y - 1):y + 2, max(0, z - 1):z + 2].ravel()]
              for y in range(hw[0]) for z in range(hw[1])]
    for i, neighborhood in enumerate(groups):
        if selected is not None:
            neighborhood = neighborhood[selected[neighborhood]]
        if pieces is not None and pieces[i] >= 0:
            neighborhood = neighborhood[pieces[neighborhood] == pieces[i]]
        elif pieces is not None:
            neighborhood = np.empty(0, int)
        if len(neighborhood):
            out[i] = response[neighborhood].mean(0)
    return out


def f281(c, weight=1., control=False):
    _pure_roles(c)
    profile_r, profile_q = c.r @ c.r.T, c.q @ c.r.T
    neighborhood = _neighborhood_modes(c.r, c.rhw, c.fm, c.bm)
    residual = np.zeros_like(neighborhood); observed = np.zeros(len(c.r), bool)
    for train, hold in buffered_folds(c.rhw, c.pure):
        model = ridge(profile_r[train], neighborhood[train], rank=16)
        residual[hold] = neighborhood[hold] - predict(model, profile_r[hold]); observed |= hold
    if observed.sum() < 2 or len(np.unique(c.c[observed] >= .9)) < 2:
        return result(c, score=c.s, mechanism='F281', fallback='insufficient buffered residual labels')
    label_head = ridge(residual[observed], (c.c[observed] >= .9).astype(float))
    profile_head = ridge(profile_r[c.pure], neighborhood[c.pure], rank=16)
    center_prediction = predict(profile_head, profile_q)
    ownership = _piece_ownership(c.P, c.hw)
    fixed = _neighborhood_modes(c.q, c.hw, c.fm, c.bm, pieces=ownership) - center_prediction
    def objective(y):
        r = fixed if control else (_neighborhood_modes(c.q, c.hw, c.fm, c.bm, selected=y, pieces=ownership) - center_prediction)
        signal = (predict(label_head, r) - .5) * 2
        signal[ownership < 0] = 0
        return c.energy(y) - weight * float(y @ signal)
    state, info = search_labels(c, objective)
    return result(c, labels=state, mechanism='F281_conditional_complete_profile_residual',
                  rank=16, source_out_of_block_samples=int(observed.sum()), **info)


def _bg_partition(c):
    eligible = (np.max(c.q @ c.bm.T, axis=1) > np.max(c.q @ c.fm.T, axis=1))
    centers, labels = modes(c.q[eligible], 8)
    groups = []
    for j in range(len(centers)):
        mask = np.zeros(len(c.q), bool); mask[np.flatnonzero(eligible)[labels == j]] = True; groups.append(mask)
    return centers, groups


def _exclusive_match(c, y, forbidden):
    total = 0.; matched = []
    for p in c.P:
        ids = np.flatnonzero(p & y & ~forbidden)
        costs = 1 - c.fm @ c.q[ids].T
        padded = np.c_[costs, np.eye(len(c.fm)) * 0 + 1.]
        a, b = linear_sum_assignment(padded)
        total += float(padded[a, b].sum())
        matched.extend(ids[b[b < len(ids)]].tolist())
    return total, matched


def f283(c, weight=1., control=False):
    _pure_roles(c); centers, groups = _bg_partition(c); n = len(c.q)
    def objective(state):
        y, z = state[:n], state[n:]
        forbidden = np.zeros(n, bool)
        bgcost = 0.
        for j in np.flatnonzero(z):
            forbidden |= groups[j]
            bgcost += float(np.sum(1 - c.q[groups[j]] @ centers[j])) + .25
        matchcost, witnesses = _exclusive_match(c, y, np.zeros(n, bool) if control else forbidden)
        return c.energy(y) + weight * (matchcost + bgcost)
    state, info = search_labels(c, objective, auxiliary=len(groups))
    return result(c, labels=state[:n], mechanism='F283_part_capacity_cluster_exclusion',
                  bg_clusters=len(groups), active_bg_clusters=int(state[n:].sum()), **info)


def _dual_loops(c, maximum=32):
    """Shortest cycles through seed-adjacent dual edges; exterior closes border chains."""
    h, w = c.hw
    if min(h, w) < 2:
        return []
    exterior = (h - 1) * (w - 1)
    # A dual edge crosses a primal adjacency, with affinity rather than distance cost.
    graph = [[] for _ in range(exterior + 1)]; coordinates = []
    xy = np.c_[np.indices(c.hw)[0].ravel(), np.indices(c.hw)[1].ravel()]
    for e, (a, b) in enumerate(c.edges):
        ya, xa = xy[a]; yb, xb = xy[b]
        if ya == yb:
            x = min(xa, xb)
            u = (ya - 1) * (w - 1) + x if ya > 0 else exterior
            v = ya * (w - 1) + x if ya < h - 1 else exterior
        else:
            y = min(ya, yb)
            u = y * (w - 1) + xa - 1 if xa > 0 else exterior
            v = y * (w - 1) + xa if xa < w - 1 else exterior
        if u == v:
            continue
        affinity = c.edge_weight[e] / .25
        cost = .1 + affinity
        index = len(coordinates); coordinates.append((u, v, a, b, cost))
        graph[u].append((v, cost, index)); graph[v].append((u, cost, index))
    roots = np.argsort(-c.s, kind='stable')[:16]
    loops = []; seen = set()
    for root in roots:
        if c.s[root] <= 0:
            continue
        root_edges = sorted([j for j, e in enumerate(coordinates) if root in e[2:4]],
                            key=lambda j: (coordinates[j][-1], j))[:2]
        for removed in root_edges:
            u, v, *_ = coordinates[removed]
            dist = np.full(len(graph), np.inf); dist[u] = 0; previous = {}; queue = [(0., u)]
            while queue:
                d, a = heapq.heappop(queue)
                if d != dist[a]:
                    continue
                if a == v:
                    break
                for b, cost, index in graph[a]:
                    if index == removed:
                        continue
                    if d + cost < dist[b]:
                        dist[b] = d + cost; previous[b] = (a, index); heapq.heappush(queue, (dist[b], b))
            if not np.isfinite(dist[v]):
                continue
            fence = [removed]; node = v
            while node != u:
                node, e = previous[node]; fence.append(e)
            key = tuple(sorted(fence))
            if key in seen:
                continue
            seen.add(key)
            blocked = {tuple(sorted(coordinates[e][2:4])) for e in fence}
            # The primal component containing the root is the enclosed range.
            region = np.zeros(h * w, bool); region[root] = True; queue = [int(root)]
            primal = [[] for _ in range(h * w)]
            for a, b in c.edges:
                if tuple(sorted((a, b))) not in blocked:
                    primal[a].append(b); primal[b].append(a)
            for a in queue:
                for b in primal[a]:
                    if not region[b]:
                        region[b] = True; queue.append(int(b))
            if region.all():
                continue
            loops.append((region, np.array(fence), sum(coordinates[e][-1] for e in fence)))
            if len(loops) >= maximum:
                return loops
    return loops


def f284(c, weight=1., control=False):
    _pure_roles(c); loops = _dual_loops(c); n, l = len(c.q), len(loops)
    if not loops:
        return result(c, score=c.s, mechanism='F284', fallback='no positive root with enclosing dual cycle')
    profile = c.q @ c.r.T
    fees = []
    cores = []
    for p, fence, perimeter in loops:
        cores.append(np.flatnonzero(p & (c.s > 0)))
        inside = profile[p].mean(0); outside = profile[~p].mean(0)
        direction = np.mean(inside[c.F] - outside[c.F]) - np.mean(inside[c.B] - outside[c.B])
        fees.append(perimeter / max(1, len(fence)) - direction)
    union = np.logical_or.reduce([row[0] for row in loops])
    def objective(state):
        y, z = state[:n], state[n:]
        if np.any(y[~union] != (c.s > 0)[~union]):
            return np.inf
        for j in np.flatnonzero(z):
            if not len(cores[j]) or not np.any(y[cores[j]]):
                return np.inf
            if control and not y[cores[j][0]]:
                return np.inf
        protected = np.zeros(n, bool)
        for j in np.flatnonzero(z):
            protected[cores[j]] = True
        independent = .25 * np.sum(y & (c.s > 0) & ~protected)
        return c.energy(y) + weight * (float(np.asarray(fees)[z].sum()) + independent)
    seeds = [np.where(union, v, c.s > 0) for v in (c.s > 0, np.zeros(n, bool), np.ones(n, bool))]
    state, info = search_labels(c, objective, auxiliary=l, starts=seeds)
    return result(c, labels=state[:n], mechanism='F284_retractable_identity_core_dual_fence',
                  loops=l, active_loops=int(state[n:].sum()), **info)
