"""FP64 Torch implementation of the frozen full-state PCF tree model.

No image encoder, Query labels, training, top-k, alternate tree or dense MxM
relation is used. This is an equation implementation, not verified Pro code.
CPU/GPU parity and actual GPU runtime require their separate recorded checks.
"""
from __future__ import annotations

import math
from typing import Any


def _torch():
    import torch
    return torch


def _tensor(value, device, *, dtype=None):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype or torch.float64)
    # CPU reference arrays are read-only; copy without exposing writable aliases.
    return torch.tensor(value, device=device, dtype=dtype or torch.float64)


def _device_for(value, device):
    torch = _torch()
    return torch.device(device if device is not None else
                        value.device if isinstance(value, torch.Tensor) else "cpu")


def _validated_tree(tree, n):
    """Validate the supplied complete tree; keep edges, sort reduction slots only."""
    parent = tuple(int(x) for x in tree.parent)
    if len(parent) != n or len(tree.children) != n:
        raise ValueError("A complete Query tree and every child list are required")
    children = tuple(tuple(sorted(int(x) for x in row)) for row in tree.children)
    roots = tuple(i for i, p in enumerate(parent) if p == -1)
    if not roots or roots != tuple(int(x) for x in tree.roots):
        raise ValueError("Explicit root IDs must match parent=-1")
    seen_children = set()
    for i, row in enumerate(children):
        if len(row) != len(set(row)):
            raise ValueError("Duplicate child IDs are not a tree")
        for child in row:
            if not 0 <= child < n or child == i or parent[child] != i or child in seen_children:
                raise ValueError("Invalid or repeated parent/child edge")
            seen_children.add(child)
    if seen_children != set(range(n)).difference(roots):
        raise ValueError("Every nonroot must appear exactly once as a child")
    depth = [-1] * n
    frontier = list(roots)
    for r in roots:
        depth[r] = 0
    levels = []
    while frontier:
        levels.append(tuple(sorted(frontier)))
        next_frontier = []
        for i in frontier:
            for child in children[i]:
                if depth[child] >= 0:
                    raise ValueError("Tree cycle or repeated node")
                depth[child] = depth[i] + 1
                next_frontier.append(child)
        frontier = next_frontier
    if min(depth) < 0:
        raise ValueError("Tree cycle or unreachable Query node")
    provided = tuple(tuple(sorted(int(x) for x in level)) for level in tree.levels)
    if provided != tuple(levels):
        raise ValueError("Supplied tree levels disagree with actual parents")
    if sorted(int(x) for x in tree.preorder) != list(range(n)) or sorted(int(x) for x in tree.postorder) != list(range(n)):
        raise ValueError("Pre/post traversals must retain every Query node")
    width = max((len(row) for row in children), default=0)
    slots = tuple(row + (-1,) * (width-len(row)) for row in children)
    return parent, children, tuple(levels), slots, roots


class TorchRelation:
    """Same sparse K + diagonal scaling + rank-one action as CPU RelationKernel."""

    def __init__(self, relation, device, permutation=None):
        torch = _torch()
        self.m = int(relation.m)
        self.eta = float(relation.eta)
        self.degenerate = bool(relation.degenerate)
        if self.m < 0 or self.eta != .5:
            raise ValueError("Frozen Source relation requires m>=0 and eta=.5")
        if self.m <= 3 and not self.degenerate:
            raise ValueError("M<=3 must explicitly use the all-ones degeneracy")
        kernel = relation.sparseK
        if kernel.shape != (self.m, self.m) or not getattr(kernel, "has_sorted_indices", False):
            raise ValueError("Sorted full-state Source CSR is required")
        if kernel.format != "csr" or not kernel.has_canonical_format:
            raise ValueError("Canonical CSR with no duplicated state pairs is required")
        if kernel.nnz and (not math.isfinite(float(kernel.data.min())) or
                           not math.isfinite(float(kernel.data.max())) or
                           float(kernel.data.min()) < 0 or float(kernel.data.max()) > 1):
            raise ValueError("Source K requires finite nonnegative geodesic weights <=1")
        if kernel.diagonal().any():
            raise ValueError("Source K must have zero diagonal")
        if (kernel != kernel.T).nnz:
            raise ValueError("Source K must be symmetric")
        # With K=0 the balanced uniform off-diagonal floor is exactly ones.
        # This is structural neutrality, not a near-ones numerical threshold.
        self.is_structurally_ones = self.degenerate or kernel.nnz == 0
        self.scale = _tensor(relation.scale, device)
        if self.scale.shape != (self.m,) or not bool(torch.isfinite(self.scale).all()) or not bool((self.scale > 0).all()):
            raise ValueError("Finite positive full-state D is required")
        self.kernel = torch.sparse_csr_tensor(
            _tensor(kernel.indptr, device, dtype=torch.int64),
            _tensor(kernel.indices, device, dtype=torch.int64),
            _tensor(kernel.data, device), size=(self.m, self.m),
            dtype=torch.float64, device=device)
        self.permutation = None
        self.inverse = None
        if permutation is not None:
            p = _tensor(permutation, device, dtype=torch.int64)
            original = _tensor(permutation, device)
            if p.shape != (self.m,) or not bool(torch.equal(original, p.to(torch.float64))) or not bool(torch.equal(p.sort().values, torch.arange(self.m, device=device))):
                raise ValueError("R shuffle requires a complete integer state permutation")
            self.permutation = p
            self.inverse = p.argsort()
        if self.m >= 4 and not self.degenerate:
            sums = self._original_action(torch.ones((1, self.m), dtype=torch.float64, device=device))
            if not bool(torch.isfinite(sums).all()) or float((sums-self.m).abs().max()) > max(2e-10, self.m*3e-12):
                raise ValueError("Source R must be balanced with total row sum M")

    def _original_action(self, values):
        torch = _torch()
        if self.degenerate:
            return values.sum(dim=1, keepdim=True).expand_as(values)
        weighted = values * self.scale
        # CSR sparse-dense multiply is the sole sparse backend operation. No
        # scatter-add of parent messages and no dense state-pair allocation.
        sparse = torch.sparse.mm(self.kernel, weighted.T.contiguous()).T
        remainder = weighted.sum(dim=1, keepdim=True) - weighted
        return values + self.scale * (self.eta*sparse + (1-self.eta)*remainder)

    def action(self, values):
        if values.ndim != 2 or values.shape[1] != self.m:
            raise ValueError("Batch vectors must retain every Source state")
        if self.permutation is None:
            return self._original_action(values)
        # R'[a,b]=R[p[a],p[b]]; ONLY the relation is shuffled, never pi.
        return self._original_action(values[:, self.inverse])[:, self.permutation]


def semantic_correspondence_torch(source_features, query_features, source_mask,
                                  *, device=None, query_batch=128):
    """All-state softmax(q@s/.07), caller-unit features, no hidden normalization."""
    torch = _torch()
    device = _device_for(query_features, device)
    role = _tensor(source_mask, device, dtype=torch.bool)
    original_role = _tensor(source_mask, device)
    if role.ndim != 2 or min(role.shape) <= 0 or not bool(((original_role == 0) | (original_role == 1)).all()):
        raise ValueError("Explicit binary full Source grid is required")
    source = _tensor(source_features, device)
    query = _tensor(query_features, device)
    if source.ndim == 3 and tuple(source.shape[:2]) == tuple(role.shape):
        source = source.reshape(role.numel(), source.shape[-1])
    if source.ndim != 2 or source.shape[0] != role.numel() or query.ndim != 2 or query.shape[1] != source.shape[1] or source.shape[1] < 1:
        raise ValueError("All Source/Query semantic rows in common coordinates are required")
    if not isinstance(query_batch, int) or query_batch < 1:
        raise ValueError("Positive query batching required; batching is not truncation")
    for values in (source, query):
        if not bool(torch.isfinite(values).all()) or (len(values) and float((values.norm(dim=1)-1).abs().max()) > 2e-6):
            raise ValueError("Caller-normalized finite unit semantic features required")
    selected = source[role.reshape(-1)]
    output = torch.empty((len(query), len(selected)), dtype=torch.float64, device=device)
    if not len(selected):
        return output
    for start in range(0, len(query), query_batch):
        logits = query[start:start+query_batch] @ selected.T / .07
        # Mirror the NumPy reference's operation order, not a precision-changing
        # autocast/flash path. Only Query rows are batched; all states remain.
        logits = logits - logits.max(dim=1, keepdim=True).values
        values = logits.exp()
        output[start:start+query_batch] = values / values.sum(dim=1, keepdim=True)
    return output


def _chunks(ids, count):
    for start in range(0, len(ids), count):
        yield ids[start:start+count]


def _inputs(h, pi, relation, tree, device):
    torch = _torch()
    unary = _tensor(h, device)
    probability = _tensor(pi, device)
    if tuple(unary.shape) != tuple(tree.shape) or not bool(torch.isfinite(unary).all()) or not bool(((unary >= 0) & (unary <= 1)).all()):
        raise ValueError("Finite [0,1] h on every Query position is required")
    n = unary.numel()
    m = int(relation.m)
    if probability.shape != (n, m) or not bool(torch.isfinite(probability).all()) or not bool((probability >= 0).all()):
        raise ValueError("Finite nonnegative full Query x Source-FG pi is required")
    if m and float((probability.sum(dim=1)-1).abs().max()) > 1e-12:
        raise ValueError("Every complete pi row must sum to1")
    parent, children, levels, slots, roots = _validated_tree(tree, n)
    weights = _tensor(tree.edgew, device)
    if weights.shape != (n,) or not bool(torch.isfinite(weights).all()) or not bool(((weights >= 0) & (weights <= 1)).all()):
        raise ValueError("All supplied tree edge weights must lie in [0,1]")
    if any(float(weights[root]) != 0 for root in roots):
        raise ValueError("Root parent weights must be0")
    return unary, probability, parent, children, levels, slots, roots, weights


def tree_sum_product_torch(h, pi, relation, tree, *, rho=.1,
                           pair_strength=1., relation_permutation=None,
                           return_state_probabilities=False, device=None,
                           node_batch=64):
    """Exact frozen two-pass tree model, in FP64 on the requested Torch device.

    Each outgoing UF/B message equals1. Children are gathered and added in
    ascending node-ID slots, never scattered with atomic parent reductions.
    The sparse CSR backend itself still needs actual GPU parity/repeat checks.
    No inferred calibrated uncertainty or empirical segmentation gain is claimed.
    """
    torch = _torch()
    if not math.isfinite(float(rho)) or not 0 <= rho < 1 or not math.isfinite(float(pair_strength)) or not 0 <= pair_strength <= 1:
        raise ValueError("Frozen model requires 0<=rho<1 and 0<=lambda<=1")
    if not isinstance(node_batch, int) or node_batch < 1:
        raise ValueError("Positive depth-node batch required")
    device = _device_for(h, device)
    with torch.no_grad():
        unary, pi, parent, children, levels, slots, roots, edgew = _inputs(h, pi, relation, tree, device)
        flat = unary.reshape(-1)
        n, m = pi.shape
        operator = TorchRelation(relation, device, relation_permutation)
        metadata = dict(rho=float(rho), pair_strength=float(pair_strength),
                        backend="Torch FP64 depth-batched sparseCSR+rank1",
                        device=str(device), exact_two_pass_tree=True,
                        foreground_is_calibrated_task_probability=False,
                        relation_only_shuffled=relation_permutation is not None,
                        all_Query_nodes=n, all_Source_FG_states=m,
                        source_relation_degenerate=bool(relation.degenerate),
                        abstained=m == 0, node_batch=node_batch,
                        fixed_child_reduction="ascending child node IDs; no scatter-add",
                        dense_edge_M_squared_allocations=0,
                        matched_message_workspace_bytes=2*n*m*8,
                        full_pi_bytes=n*m*8, R_matvec_calls=0,
                        GPU_parity_certified=False, measured_runtime_seconds=None,
                        model_neutral=False, native_identity_required=False)
        if not m:
            metadata.update(model_neutral=True, native_identity_required=True,
                            abstention_rule="No Source FG: supplied h retained; UF/B are fallback bookkeeping, not model BP")
            return dict(foreground=unary.clone(), matched_foreground=torch.zeros_like(unary),
                        unmatched_foreground=unary.clone(), background=1-unary,
                        matched_states=pi.clone() if return_state_probabilities else None,
                        metadata=metadata)
        exact_uniform = bool((pi == 1./m).all())
        if operator.is_structurally_ones or pair_strength == 0 or exact_uniform:
            reason = "Rones" if operator.is_structurally_ones else "lambda0" if pair_strength == 0 else "uniform_pi_balanced_R"
            metadata.update(model_neutral=True, native_identity_required=True, neutral_reason=reason)
            return dict(foreground=unary.clone(), matched_foreground=(1-rho)*unary,
                        unmatched_foreground=rho*unary, background=1-unary,
                        matched_states=(1-rho)*flat[:, None]*pi if return_state_probabilities else None,
                        metadata=metadata)
        logh, logbg = flat.log(), torch.log1p(-flat)
        loguf = logh + math.log(rho) if rho else torch.full_like(flat, -torch.inf)
        logcoefficient = math.log1p(-rho)
        parent_ids = _tensor(parent, device, dtype=torch.int64)
        child_slots = _tensor(slots, device, dtype=torch.int64).reshape(n, len(slots[0]))
        # QueryTree weights are CPU metadata. Zero edges send exactly1 and
        # need no sparse action. Upload IDs once, not once per CUDA depth loop.
        positive_levels = tuple(_tensor(tuple(i for i in level if float(tree.edgew[i]) > 0),
                                        device, dtype=torch.int64) for level in levels)
        up = torch.zeros((n, m), dtype=torch.float64, device=device)
        down = torch.zeros_like(up)

        def child_sum(ids, excluded=None):
            result = torch.zeros((len(ids), m), dtype=torch.float64, device=device)
            for slot in range(child_slots.shape[1]):
                child = child_slots[ids, slot]
                valid = child >= 0
                if excluded is not None:
                    valid = valid & (child != excluded)
                result = result + torch.where(valid[:, None], up[child.clamp_min(0)], 0.)
            return result

        def node(ids, accumulated):
            logits = pi[ids].log() + logcoefficient + logh[ids, None] + accumulated
            # All labels use ONE scale. Separate matched normalization would
            # silently change the specified UF and background competition.
            lognormalizer = torch.logaddexp(torch.logsumexp(logits, dim=1),
                                           torch.logaddexp(loguf[ids], logbg[ids]))
            return ((logits-lognormalizer[:, None]).exp(),
                    (loguf[ids]-lognormalizer).exp(),
                    (logbg[ids]-lognormalizer).exp())

        def message(ids, accumulated, weights):
            matched, uf, bg = node(ids, accumulated)
            transformed = operator.action(matched)
            metadata["R_matvec_calls"] += len(ids)
            strength = pair_strength*weights[:, None]
            value = uf[:, None]+bg[:, None]+(1-strength)*matched.sum(dim=1, keepdim=True)+strength*transformed
            # Caller omits zero-strength edges; their stored logs stay exactly0.
            # Phase-boundary finiteness checks avoid a CUDA sync per tree depth.
            # No clipping fallback, epsilon floor or altered unary is permitted.
            return value.log()

        # Upward pass: depth batches cannot depend on other nodes in that depth.
        for level in reversed(positive_levels[1:]):
            for ids in _chunks(level, node_batch):
                up[ids] = message(ids, child_sum(ids), edgew[ids])
        if not bool(torch.isfinite(up).all()):
            raise RuntimeError("Nonfinite upward log message; no clipping fallback")
        # Downward pass: parent contexts exclude this child's upward message.
        for level in positive_levels[1:]:
            for ids in _chunks(level, node_batch):
                parents = parent_ids[ids]
                accumulated = down[parents] + child_sum(parents, excluded=ids)
                down[ids] = message(parents, accumulated, edgew[ids])
        if not bool(torch.isfinite(down).all()):
            raise RuntimeError("Nonfinite downward log message; no clipping fallback")
        matched_mass = torch.empty(n, dtype=torch.float64, device=device)
        uf_mass, bg_mass = torch.empty_like(matched_mass), torch.empty_like(matched_mass)
        states = torch.empty_like(pi) if return_state_probabilities else None
        all_ids = torch.arange(n, device=device)
        for ids in _chunks(all_ids, node_batch):
            matched, uf, bg = node(ids, down[ids]+child_sum(ids))
            matched_mass[ids], uf_mass[ids], bg_mass[ids] = matched.sum(dim=1), uf, bg
            if states is not None:
                states[ids] = matched
        total_foreground = matched_mass+uf_mass
        residual = float((total_foreground+bg_mass-1).abs().max())
        foreground = 1-bg_mass
        if residual > 1e-12 or not bool(torch.isfinite(foreground).all()) or float(foreground.min()) < -1e-12 or float(foreground.max()) > 1+1e-12:
            raise RuntimeError("Node probability masses failed contract")
        # foreground=1-background is algebraically matched+UF and gives exact
        # endpoints without a mask threshold or a posterior clipping repair.
        lower = rho*flat/(rho*flat+1-flat) if rho else torch.zeros_like(flat)
        metadata.update(probability_sum_max_residual=residual,
                        UF_foreground_lower_bound_min_margin=float((foreground-lower).min()),
                        lower_bound_definition="rho*h/(rho*h+1-h); not calibrated accuracy",
                        directed_messages=2*(n-len(roots)), tree_depth=len(levels)-1)
        return dict(foreground=foreground.reshape(tree.shape),
                    matched_foreground=matched_mass.reshape(tree.shape),
                    unmatched_foreground=uf_mass.reshape(tree.shape),
                    background=bg_mass.reshape(tree.shape), matched_states=states,
                    metadata=metadata)


def binary_potts_torch(h, tree, *, device=None, node_batch=64):
    """Naive control: exp(w) same-label, different-label1; exact same h/tree."""
    torch = _torch()
    device = _device_for(h, device)
    unary = _tensor(h, device)
    if tuple(unary.shape) != tuple(tree.shape) or not bool(torch.isfinite(unary).all()) or not bool(((unary >= 0) & (unary <= 1)).all()):
        raise ValueError("Finite same h and full Query tree required")
    if not isinstance(node_batch, int) or node_batch < 1:
        raise ValueError("Positive node_batch required")
    flat = unary.reshape(-1)
    parent, children, levels, slots, roots = _validated_tree(tree, len(flat))
    weights = _tensor(tree.edgew, device)
    if weights.shape != flat.shape or not bool(torch.isfinite(weights).all()) or not bool(((weights >= 0) & (weights <= 1)).all()) or any(float(weights[r]) != 0 for r in roots):
        raise ValueError("Tree weights must be finite [0,1], root0")
    ids_parent = _tensor(parent, device, dtype=torch.int64)
    child_slots = _tensor(slots, device, dtype=torch.int64).reshape(len(flat), len(slots[0]))
    logs = torch.stack((torch.log1p(-flat), flat.log()), dim=1)
    up = torch.zeros_like(logs)
    down = torch.zeros_like(logs)

    def child_sum(ids, excluded=None):
        total = torch.zeros((len(ids), 2), device=device, dtype=torch.float64)
        for slot in range(child_slots.shape[1]):
            child = child_slots[ids, slot]
            valid = child >= 0
            if excluded is not None:
                valid = valid & (child != excluded)
            total = total + torch.where(valid[:, None], up[child.clamp_min(0)], 0.)
        return total

    def message(ids, accumulated, w):
        values = (logs[ids]+accumulated).softmax(dim=1)
        same = w.exp()
        output = torch.stack((same*values[:, 0]+values[:, 1], values[:, 0]+same*values[:, 1]), dim=1).log()
        return output-output[:, :1]

    with torch.no_grad():
        for level in reversed(levels[1:]):
            for chunk in _chunks(level, node_batch):
                ids = _tensor(chunk, device, dtype=torch.int64)
                up[ids] = message(ids, child_sum(ids), weights[ids])
        for level in levels[1:]:
            for chunk in _chunks(level, node_batch):
                ids = _tensor(chunk, device, dtype=torch.int64)
                parents = ids_parent[ids]
                down[ids] = message(parents, down[parents]+child_sum(parents, ids), weights[ids])
        result = torch.empty_like(flat)
        for chunk in _chunks(tuple(range(len(flat))), node_batch):
            ids = _tensor(chunk, device, dtype=torch.int64)
            result[ids] = (logs[ids]+down[ids]+child_sum(ids)).softmax(dim=1)[:, 1]
    return result.reshape(tree.shape)


def shape_memory_audit(query_nodes, source_states, kernel_nnz, *,
                       node_batch=64, max_children=4,
                       return_state_probabilities=False):
    """Conservative shape arithmetic only, no encoder/allocation/runtime probe."""
    n, m, nnz = int(query_nodes), int(source_states), int(kernel_nnz)
    if min(n, m, nnz) < 0 or node_batch < 1 or max_children < 0:
        raise ValueError("Nonnegative shape counts and positive batch required")
    batch = min(n, node_batch)
    pi = n*m*8
    logs = 2*pi
    output = pi if return_state_probabilities else 0
    sparse = 16*nnz+8*(m+1)+8*m
    tree = 8*n*(max_children+5)
    # Chunk intermediates are bounded; allocator/cuSPARSE workspace, encoder
    # sharing and input feature tensors are deliberately separate unknowns.
    temporary_bound = 24*batch*m*8+32*batch*8
    return dict(query_nodes=n, all_Source_states=m, source_kernel_nnz=nnz,
                pi_float64_bytes=pi, two_log_messages_bytes=logs,
                optional_matched_state_output_bytes=output,
                CSR64_plus_scale_bytes=sparse, tree_index_scalar_bytes=tree,
                chunk_temporary_conservative_bytes=temporary_bound,
                array_accounting_upper_bytes=pi+logs+output+sparse+tree+temporary_bound,
                dense_edge_M_squared_allocations=0,
                excludes_encoder_input_features_and_allocator_workspace=True,
                actual_GPU_pilot_required=True, measured_runtime_seconds=None,
                no_Source_topk=True, no_Query_downsample=True,
                parity_target_FP64_max_abs_error=1e-8,
                FP32_equivalence_certified=False)
