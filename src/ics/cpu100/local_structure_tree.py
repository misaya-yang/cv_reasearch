"""Restricted feature merge-tree alignment, not a general tree edit solver."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .common import Episode, Result, neighbors, unit, validate


@dataclass
class FeatureTree:
    left: np.ndarray
    right: np.ndarray
    mean: np.ndarray
    mass: np.ndarray
    height: np.ndarray
    minimum: np.ndarray
    original: np.ndarray
    roots: list[int]


def build_tree(features, hw, support, valid):
    """Exact single-linkage on induced four-neighbor support, deterministic ties."""
    selected = np.flatnonzero(support)
    n = len(selected)
    if not n:
        return FeatureTree(np.empty(0, int), np.empty(0, int), np.empty((0, features.shape[1])),
                           np.empty(0), np.empty(0), np.empty(0, int), selected, [])
    inverse = np.full(len(features), -1, int)
    inverse[selected] = np.arange(n)
    a, b = neighbors(hw)
    active = support[a] & support[b]
    a, b = a[active], b[active]
    costs = np.maximum(1 - np.einsum("nd,nd->n", features[a], features[b]), 0)
    order = np.lexsort((b, a, costs))
    parent = np.arange(n)
    root_node = np.arange(n)
    left, right = [-1] * n, [-1] * n
    means, mass = [v.copy() for v in features[selected]], valid[selected].tolist()
    height, minimum = [0.] * n, selected.tolist()

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for edge in order:
        aa, bb = find(int(inverse[a[edge]])), find(int(inverse[b[edge]]))
        if aa == bb:
            continue
        l, r = int(root_node[aa]), int(root_node[bb])
        if minimum[l] > minimum[r]:
            l, r = r, l
        total = mass[l] + mass[r]
        left.append(l)
        right.append(r)
        means.append((means[l] * mass[l] + means[r] * mass[r]) / total)
        mass.append(total)
        height.append(float(costs[edge]))
        minimum.append(min(minimum[l], minimum[r]))
        # A representative's choice has no effect on the already stored tree.
        parent[bb] = aa
        root_node[aa] = len(left) - 1
    roots = sorted({int(root_node[find(i)]) for i in range(n)}, key=lambda i: minimum[i])
    return FeatureTree(np.asarray(left), np.asarray(right), np.asarray(means),
                       np.asarray(mass), np.asarray(height), np.asarray(minimum), selected, roots)


def compress(tree, root, cap=8):
    terminals = [root]
    while len(terminals) < cap:
        options = [i for i in terminals if tree.left[i] >= 0]
        if not options:
            break
        chosen = min(options, key=lambda i: (-tree.mass[i], tree.minimum[i]))
        terminals.remove(chosen)
        terminals.extend((int(tree.left[chosen]), int(tree.right[chosen])))
    stop = set(terminals)
    nodes, left, right = [], [], []

    def visit(i):
        if i in stop:
            nodes.append(i)
            left.append(-1)
            right.append(-1)
            return len(nodes) - 1
        l, r = visit(int(tree.left[i])), visit(int(tree.right[i]))
        nodes.append(i)
        left.append(l)
        right.append(r)
        return len(nodes) - 1

    visit(root)
    counts = np.ones(len(nodes), dtype=int)
    for i, (l, r) in enumerate(zip(left, right)):
        if l >= 0:
            counts[i] += counts[l] + counts[r]
    return np.asarray(nodes), np.asarray(left), np.asarray(right), counts


def restricted_alignment(cost, a, b, delta=.25):
    """Raw recursion first, root-only normalization; table is scalar DP."""
    an, al, ar, ac = a
    bn, bl, br, bc = b
    table = np.empty((len(an), len(bn)))
    for i in range(len(an)):
        for j in range(len(bn)):
            value = cost[i, j]
            if al[i] < 0 and bl[j] < 0:
                table[i, j] = value
            elif al[i] < 0:
                table[i, j] = value + delta * (bc[j] - 1)
            elif bl[j] < 0:
                table[i, j] = value + delta * (ac[i] - 1)
            else:
                l, r, s, t = al[i], ar[i], bl[j], br[j]
                table[i, j] = min(value + table[l, s] + table[r, t],
                                  value + table[l, t] + table[r, s],
                                  table[l, j] + delta * (1 + ac[r]),
                                  table[r, j] + delta * (1 + ac[l]),
                                  table[i, s] + delta * (1 + bc[t]),
                                  table[i, t] + delta * (1 + bc[s]))
    return float(table[-1, -1] / (ac[-1] + bc[-1]))


def leaf_bag(cost, a, b):
    """Bidirectional nearest leaf cost, same compressed leaf representation."""
    leaves_a = np.flatnonzero(a[1] < 0)
    leaves_b = np.flatnonzero(b[1] < 0)
    sub = cost[np.ix_(leaves_a, leaves_b)]
    return float((np.min(sub, axis=0).mean() + np.min(sub, axis=1).mean()) / 2)


def rewire_terminal_observations(raw, mass, height, compact):
    """Keep leaf observations/bag and topology, alter nested association.

    Raw mean and mass travel together, so recomputing ancestor means preserves
    the original root mean. The tree shape, node counts and leaf bag are held.
    """
    _, left, right, _ = compact
    terminal = np.flatnonzero(left < 0)
    order = np.random.default_rng(0).permutation(len(terminal))
    new_raw, new_mass, new_height = raw.copy(), mass.copy(), height.copy()
    new_raw[terminal] = raw[terminal[order]]
    new_mass[terminal] = mass[terminal[order]]
    new_height[terminal] = height[terminal[order]]
    for i, (l, r) in enumerate(zip(left, right)):
        if l >= 0:
            new_mass[i] = new_mass[l] + new_mass[r]
            new_raw[i] = (new_raw[l] * new_mass[l] + new_raw[r] * new_mass[r]) / new_mass[i]
    return unit(new_raw), new_height


def decode(tree, evidence):
    best, state = np.zeros(len(tree.left)), np.zeros(len(tree.left), int)
    for i in range(len(tree.left)):
        l, r = tree.left[i], tree.right[i]
        if l >= 0 and best[l] + best[r] > best[i]:
            best[i], state[i] = best[l] + best[r], 1
        whole = tree.mass[i] * evidence[i]
        if whole > best[i]:
            best[i], state[i] = whole, 2
    selected = np.zeros(len(tree.original), bool)
    pending = list(tree.roots)
    while pending:
        i = pending.pop()
        if state[i] == 2:
            stack = [i]
            while stack:
                node = stack.pop()
                if tree.left[node] < 0:
                    selected[node] = True
                else:
                    stack.extend((int(tree.left[node]), int(tree.right[node])))
        elif state[i] == 1:
            pending.extend((int(tree.left[i]), int(tree.right[i])))
    return selected, best


def tree_alignment(ep: Episode, arm="tree") -> Result:
    started = time.perf_counter()
    validate(ep)
    margin = np.full(len(ep.q), -1.)
    if ep.wf.sum() == 0 or ep.wb.sum() == 0 or not np.any(ep.q_valid > 0):
        if ep.wf.sum() > 0 and ep.wb.sum() == 0:
            margin[ep.q_valid > 0] = 1
        return Result(margin.reshape(ep.q_hw), dict(method_id="local_004", arm=arm,
            inactive_reason="empty_original_reference_class_or_query_support",
            wall_seconds=time.perf_counter() - started, natural_segmentation_gain="unmeasured"))
    q, r = unit(ep.q), unit(ep.r)
    fraction = np.divide(ep.wf, ep.wvalid, out=np.zeros_like(ep.wf), where=ep.wvalid > 0)
    masks = [(ep.wvalid > 0) & (fraction >= .5), (ep.wvalid > 0) & (fraction < .5)]
    if not all(np.any(s) for s in masks):
        if np.any(masks[0]):
            margin[ep.q_valid > 0] = 1
        return Result(margin.reshape(ep.q_hw), dict(method_id="local_004", arm=arm,
            inactive_reason="original_MR_nonempty_but_majority_grid_class_lost",
            foreground_weight=float(ep.wf.sum()), background_weight=float(ep.wb.sum()),
            majority_foreground_tokens=int(masks[0].sum()), majority_background_tokens=int(masks[1].sum()),
            wall_seconds=time.perf_counter() - started, natural_segmentation_gain="unmeasured"))
    query = build_tree(q, ep.q_hw, ep.q_valid > 0, ep.q_valid)
    templates, class_ids, omitted = [], [], []
    for class_id, support in enumerate(masks):
        reference = build_tree(r, ep.r_hw, support, ep.wvalid)
        roots = sorted(reference.roots, key=lambda i: (-reference.mass[i], reference.minimum[i]))
        omitted.append(max(0, len(roots) - 2))
        for root in roots[:2]:
            compact = compress(reference, root)
            n = compact[0]
            templates.append((reference.mean[n], reference.mass[n], reference.height[n], compact))
            class_ids.append(class_id)
    query_centroids = unit(query.mean)
    cached = []
    for raw, mass, height, compact in templates:
        if arm == "rewired":
            b_feature, b_height = rewire_terminal_observations(raw, mass, height, compact)
        else:
            b_feature, b_height = unit(raw), height
        cosine = np.empty((len(query.left), len(raw)))
        for start in range(0, len(query.left), 64):
            cosine[start:start + 64] = np.einsum("nd,kd->nk", query_centroids[start:start + 64],
                                               b_feature, optimize=False)
        cost = np.maximum(1 - cosine, 0)
        cost += np.abs(query.height[:, None] - b_height[None])
        cached.append((cost, np.maximum(1 - cosine[:, -1], 0), compact))
    evidence = np.zeros(len(query.left))
    states = 0
    for i in range(len(query.left)):
        compact = compress(query, i)
        nodes = compact[0]
        values = []
        for full_cost, centroid_cost, original in cached:
            b = original
            # Local compressed indices must address this compact feature matrix.
            local_b = (np.arange(len(b[0])), b[1], b[2], b[3])
            local_a = (np.arange(len(nodes)), compact[1], compact[2], compact[3])
            cost = full_cost[nodes]
            if arm == "centroid":
                value = float(centroid_cost[i])
            elif arm == "leaf_bag":
                value = leaf_bag(cost, local_a, local_b)
            else:
                value = restricted_alignment(cost, local_a, local_b)
                states += len(nodes) * len(original[0])
            values.append(value)
        values = np.asarray(values)
        classes = np.asarray(class_ids)
        evidence[i] = np.min(values[classes == 1]) - np.min(values[classes == 0])
        if abs(evidence[i]) <= 1e-12:
            evidence[i] = 0
    chosen, score = decode(query, evidence)
    margin[query.original[chosen]] = 1
    return Result(margin.reshape(ep.q_hw), dict(method_id="local_004", arm=arm,
        valid_query_tokens=len(query.original), query_nodes=len(query.left), query_roots=len(query.roots),
        reference_templates=len(templates), omitted_reference_components=omitted,
        node_evidence_roundoff_tolerance=1e-12,
        maximum_compressed_terminals=8, restricted_alignment_states=states,
        decoder_optimum=float(sum(score[root] for root in query.roots)),
        query_ground_truth_used=False, new_encoder_forwards=0,
        natural_segmentation_gain="unmeasured", wall_seconds=time.perf_counter() - started))
