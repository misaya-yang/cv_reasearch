"""Actual native query mutual20NN graph requested by Pro30 M28.

This retains mean_graph.py's original Torch topk and FP32 weight equations;
it does not reuse B's Part1 graph as the native-unit-query graph.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import time
import numpy as np
from scipy import sparse

from ics.astra300.common import ArtifactUnavailable, array_hash


def native_query_graph(ep):
    import torch
    import torch.nn.functional as F
    from ics.methods import mean_graph
    start = time.monotonic()
    if ep.q_hw != (64, 64) or ep.q.shape != (4096, 1024) or np.any(ep.q_valid != 1):
        raise ArtifactUnavailable('M28 requires the complete native physical64×64 query graph')
    with torch.inference_mode():
        q = F.normalize(torch.tensor(np.array(ep.q, copy=True), dtype=torch.float32), dim=1)
        similarity = q @ q.T
        similarity.fill_diagonal_(-2)
        values, indices = similarity.topk(mean_graph.CONFIG['query_k'], dim=1)
        distance = (1-values).clamp_min(0)
        weights = torch.exp(-distance / distance[:, -1:].clamp_min(1e-6)).numpy().ravel()
    graph = sparse.csr_matrix((weights, (np.repeat(np.arange(4096), 20), indices.numpy().ravel())), shape=(4096, 4096))
    graph = graph.multiply(graph.T); graph.data = np.sqrt(graph.data)
    degree = np.asarray(graph.sum(1)).ravel()
    graph = (graph / max(float(degree.mean()), 1e-8)).tocsr()
    source = Path(mean_graph.__file__)
    producer = dict(query_unit_array_sha256=array_hash(ep.q), mutual_neighbors=20,
        mean_degree_normalized=True, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        actual_graph_stage='native unit query; original Torch mutual20NN/local bandwidth/sqrt/mean degree',
        physical_hw=[64, 64], wall_seconds=time.monotonic()-start, actual_new_encoder_forwards=0,
        query_GT_read=False, undirected_edges=graph.nnz//2)
    for value in (graph.data, graph.indices, graph.indptr):
        value.setflags(write=False)
    return graph, producer
