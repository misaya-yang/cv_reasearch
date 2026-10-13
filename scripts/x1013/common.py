"""Cache-only inputs and exact frozen candidate primitives for the 2026-10-13 plan.

Prediction functions never open query labels or baseline outputs. Evaluation is
an explicit, later phase, after every prediction of a task has been saved.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS')
for name in THREAD_VARS:
    os.environ[name] = '2'

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.special import expit
import torch

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / 'cv_data'
OUT = ASSETS / 'a'
REPORT = ROOT / 'docs/research/x1013_tasks/REPORT.md'
EVIDENCE = ROOT / 'evidence/research/x1013_iu.jsonl.gz'
RUNTIME = OUT / 'x1013/w02'
sys.path[:0] = [str(ROOT / 'scripts/x1012'), str(ROOT / 'src'), str(ROOT / 'scripts')]
import r5_candidate as rc
import w02_scale_eval as w02

DOMAINS = w02.DOMS
DOMAIN_NAMES = ('COCO', 'LVIS', 'PASCAL', 'PACO', 'SUIM', 'iSAID',
                'ISIC', 'Lung', 'Fundus', 'DeepGlobe')
SEED = 20261013


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def rows():
    return w02.rows_all(0)


def key(binding):
    return str(binding['index']) if isinstance(binding['index'], str) else f"{binding['index']:06d}"


def photo(binding):
    row = binding['row']
    return str(row.get('query_photo_id') or row['query_path'])


def half(binding):
    return 'S' if int(hashlib.sha256(photo(binding).encode()).hexdigest()[:8], 16) % 2 == 0 else 'T'


def task_path(task):
    return OUT / ('x1013_' + task)


def fields_path(binding):
    return task_path('T0') / 'fields' / (key(binding) + '.npz')


def prediction_path(task, binding):
    return task_path(task) / 'predictions' / (key(binding) + '.npz')


def load_fields(binding):
    with np.load(fields_path(binding), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def initialize():
    # Reuse W02's existing dependency runtime, without copying or editing it.
    rc.initialize(str(RUNTIME), 'cpu', 2, 'midrange', False)
    for name in THREAD_VARS:
        os.environ[name] = '2'
    torch.set_num_threads(2)
    torch.manual_seed(0)


def pack(mask):
    return np.packbits(np.asarray(mask, dtype=bool))


def unpack(bits):
    return np.unpackbits(bits)[:1024 * 1024].reshape(1024, 1024).astype(bool)


def up(field):
    return rc.lab.up(torch.as_tensor(field, dtype=torch.float32).reshape(64, 64)).numpy()


def readout(field, absolute=False):
    if absolute:
        return up(field) > .5
    return rc.pc.readout(field)


def foreground_field(fields):
    return fields['c32'].reshape(64, 64) if float(fields['reference_rho'][0]) > .9 else fields['VP_UN']


def laplacian(fields):
    return sparse.csr_matrix((fields['L_data'], fields['L_indices'], fields['L_indptr']),
                             shape=(4096, 4096))


def solve(primary, explanation, graph, guide=None):
    if guide is None:
        return rc.solve(primary, explanation, graph)[0]
    from ics.methods.rcg import minmax, rank
    s = minmax(primary).ravel()
    y = (s + .25 * (rank(guide) - rank(s))).astype(np.float64)
    a = (.1 + np.abs(2 * s - 1)) * explanation
    a = (a / a.mean()).astype(np.float64)
    matrix = sparse.diags(a) + 16 * graph
    rhs = a * y
    result, status = cg(matrix, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300)
    residual = np.linalg.norm(matrix @ result - rhs) / max(np.linalg.norm(rhs), 1e-12)
    if status or not np.isfinite(result).all() or residual >= 2e-7:
        raise RuntimeError(f'Guided CG failed: status={status}, residual={residual}')
    return result.reshape(64, 64).astype(np.float32)


def role_fields(binding):
    """Exact R5 quantities for the original role, also computed for fallback cases."""
    r, q, mask, coverage, fields = rc.independent(binding, {})
    c_mod, selection = rc.modulate(torch.from_numpy(fields['c32']),
        torch.from_numpy(fields['regions64'].astype(np.int64)),
        fields['ind0_32'], fields['indn_32'], fields['sim_max'], fields['back_f'])
    addition_v, addition_p = rc.vp_additions(binding, r, q, mask, coverage)
    # This grouping is part of the frozen R1 identity.
    primary = c_mod.numpy().reshape(64, 64) + (addition_v + addition_p)
    log_e = fields['log_e'].astype(np.float64)
    explanation = expit(log_e - np.median(log_e)).astype(np.float32)
    graph = rc.graph(q, fields['query_neighbors20']).tocsr()
    field = rc.solve(primary, explanation, graph)[0]
    pure = coverage >= .9
    if not bool(pure.any()):
        pure = coverage == coverage.max()
    guide = (q @ rc.lab.unit(r[pure].mean(0))).numpy()
    fields.update(c_mod=c_mod.numpy(), addition_V=addition_v, addition_P=addition_p,
                  primary=primary, explanation=explanation, VP_UN=field,
                  selected_s=np.asarray(selection['selected_s'], np.float32),
                  half=np.asarray(half(binding)), guide=guide,
                  touched=np.asarray(int((coverage > 0).sum()), np.int32),
                  reference_fraction=np.asarray(float(mask.mean()), np.float32),
                  L_data=graph.data, L_indices=graph.indices, L_indptr=graph.indptr)
    return fields


def metadata(binding, fields):
    row = binding['row']
    return {'index': str(binding['index']), 'dataset': binding['dataset'],
            'split': half(binding), 'source_split': binding['_split'],
            'fold': row['fold'], 'cls': row['global_class_id'], 'photo': photo(binding),
            'rho': float(fields['reference_rho'][0])}


def truth(binding):
    return rc.pc.truth(binding)


def iu(mask, gt, valid):
    mask = mask & valid
    return [int(np.count_nonzero(mask & gt)), int(np.count_nonzero(mask | gt)),
            int(np.count_nonzero(mask))]
