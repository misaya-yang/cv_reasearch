#!/usr/bin/env python3
"""Bounded fixed-route inverse diagnostic; no training, download, recursion or GT tuning.

The host is INSID3. B describes only its frozen pooled backward route, NOT the
full nonlinear host. GT projections below are restricted interventions, not an
oracle bound for all possible methods. Ordinary transpose label transfer is a
control. All deployable outputs freeze before the independent evaluator sees GT.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import torch

MU = 0.1
MAX_EPISODES = 40
MAX_SECONDS = 1500


def label_atoms(labels, initial, donors):
    """Intersect feature clusters with frozen P1; exact no-op patch reconstruction."""
    maps, starts, labels0, areas = {}, {}, [], []
    count = 0
    for j in donors:
        keys, inverse, counts = torch.unique(labels[j] * 2 + initial[j].long(),
                                             sorted=True, return_inverse=True, return_counts=True)
        maps[j] = inverse + count
        starts[j] = (count, count + len(keys))
        labels0.append((keys % 2).float() * 2 - 1)
        areas.append(counts.float())
        count += len(keys)
    if not donors:
        device = labels.device
        return torch.empty(0, device=device), maps, starts, torch.empty(0, device=device)
    return torch.cat(labels0), maps, starts, torch.cat(areas)


def masks_from_margins(margins, initial, maps):
    """Zero margin preserves original label; zero-route columns stay untouched."""
    return {j: torch.where(margins[ix] == 0, initial[j], margins[ix] > 0)
            for j, ix in maps.items()}


def sample_gold_patches(gold, h, max_patches=512):
    """Deterministic 8x8-tile round-robin, balanced FG/BG, from ONLY support GT.

    V sums to one: each present class has equal total weight. Actual damping is
    .1 * trace(Bw @ Bw.T) / n_support_samples, fixed relative to the Gram
    scale; neither its coefficient nor weights use query/donor truth.
    """
    truth = gold.detach().cpu().bool().flatten().tolist()
    selected_by_class = []
    for cls in (False, True):
        tiles = {}
        for p, value in enumerate(truth):
            if value != cls:
                continue
            r, c = divmod(p, h)
            tile = (r * 8 // h, c * 8 // h)
            tiles.setdefault(tile, []).append(p)
        # Evenly dispersed classes first; within a tile use stable spatial order.
        ordered = [p for level in range(max([len(v) for v in tiles.values()] + [0]))
                   for tile in sorted(tiles) for p in tiles[tile][level:level + 1]]
        selected_by_class.append(ordered)
    cap = min(max_patches, len(truth))
    counts = [min(len(v), cap // 2) for v in selected_by_class]
    left = cap - sum(counts)
    for cls in (0, 1):
        add = min(left, len(selected_by_class[cls]) - counts[cls])
        counts[cls] += add
        left -= add
    ps = torch.tensor(sorted(selected_by_class[0][:counts[0]] + selected_by_class[1][:counts[1]]),
                      dtype=torch.long, device=gold.device)
    y = gold[ps].float() * 2 - 1
    v = torch.ones_like(y)
    present = int(bool(counts[0])) + int(bool(counts[1]))
    for sign in (-1, 1):
        take = y == sign
        if take.any():
            v[take] = 1.0 / (present * int(take.sum()))
    return ps, y, v


def route_matrix(similarities, atom_indices, n_atoms, k=5):
    """Top-k donor nearest patches per support patch; duplicate atoms accumulate."""
    values = torch.stack(similarities)
    atoms = torch.stack(atom_indices)
    top = values.topk(min(k, len(similarities)), dim=0).indices
    routes = atoms.gather(0, top).T
    B = torch.zeros((values.shape[1], n_atoms), device=values.device, dtype=values.dtype)
    B.scatter_add_(1, routes, torch.full_like(routes, 1.0 / top.shape[0], dtype=B.dtype))
    return B


def effective_mu(gram, coefficient=MU):
    """Fixed relative damping; invariant to a common rescaling of V."""
    return coefficient * torch.trace(gram) / gram.shape[0]


def ridge_delta(B, z0, y, v, mu=MU):
    root = v.sqrt()
    Bw = root[:, None] * B
    residual = root * y - Bw @ z0
    gram = Bw @ Bw.T
    damping = effective_mu(gram, mu)
    dual = torch.linalg.solve(gram + damping * torch.eye(len(y), device=B.device, dtype=B.dtype), residual)
    return Bw.T @ dual, Bw, gram, residual


def transpose_vote(B, y, v, z0):
    mass = B.T @ v
    score = (B.T @ (v * y)) / mass.clamp_min(1e-12)
    return torch.where(mass > 0, score, z0)


def error_projections(Bw, gram, error, mu=MU):
    """Fixed pseudoinverse cutoff and actual-mu shrinkage, using at most 512 rows."""
    eigenvalues, U = torch.linalg.eigh(gram)
    # Gram eigenvalues below 1e-6 of the largest are not counted as observable.
    cutoff = max(float(eigenvalues.max()) * 1e-6, 1e-8)
    good = eigenvalues > cutoff
    signal = Bw @ error
    coordinates = U.T @ signal
    inverse = torch.where(good, eigenvalues.clamp_min(cutoff).reciprocal(), torch.zeros_like(eigenvalues))
    observable = Bw.T @ (U @ (inverse * coordinates))
    shrunk = Bw.T @ (U @ (coordinates / (eigenvalues.clamp_min(0) + mu)))
    return observable, shrunk, signal, dict(rank=int(good.sum()), gram_eigen_cutoff=cutoff,
        largest_gram_eigenvalue=float(eigenvalues.max()),
        smallest_retained_gram_eigenvalue=float(eigenvalues[good].min()) if good.any() else None)


def energy_fraction(projected, error, weight=None):
    weight = torch.ones_like(error) if weight is None else weight
    return float((weight * projected.square()).sum() / (weight * error.square()).sum().clamp_min(1e-12))


def host_predict(s, masks, gold):
    donors = [j for j in sorted(masks) if masks[j].any()]
    return s.predict(1, [0] + donors, [gold] + [masks[j] for j in donors], backward='pooled', k=5)


def repair(s, p1):
    """Inference boundary: this routine reads gt64[0] only; query excluded."""
    donors = list(range(2, s.n))
    gold = s.gt64[0]
    predictions = {'1shot': p1[1],
        'native_naive': s.predict(1, [0] + donors, [gold] + [p1[j] for j in donors], backward='majority'),
        'pooled_p1': host_predict(s, {j: p1[j] for j in donors}, gold)}
    if not donors:
        predictions['inverse_route'] = p1[1].clone()
        return predictions, None
    z0, maps, starts, area = label_atoms(s.lab, p1, donors)
    assert all(torch.equal((z0[maps[j]] > 0), p1[j]) for j in donors), 'atom no-op mismatch'
    ps, y, v = sample_gold_patches(gold, s.h)
    similarities, atom_indices = [], []
    for j in donors:
        similarity, ix = s.nnv(0, j)
        similarities.append(similarity[ps])
        atom_indices.append(maps[j][ix[ps]])
    B = route_matrix(similarities, atom_indices, len(z0))
    assert torch.allclose(B.sum(1), torch.ones(len(ps), device=B.device), atol=1e-6)
    delta, Bw, gram, residual = ridge_delta(B, z0, y, v)
    visited = B.sum(0) > 0
    assert torch.equal(delta[~visited], torch.zeros_like(delta[~visited])), 'zero route changed'
    corrected = masks_from_margins(z0 + delta, p1, maps)
    predictions['inverse_route'] = host_predict(s, corrected, gold)
    voted = masks_from_margins(transpose_vote(B, y, v, z0), p1, maps)
    predictions['transpose_vote_same_B'] = host_predict(s, voted, gold)
    # Scalar round-trip control, same frozen P1 and no query in donor selection.
    rt = {j: s.iou(s.predict(0, [j], [p1[j]]), gold) if p1[j].any() else 0.0 for j in donors}
    chosen = sorted(donors, key=lambda j: (-rt[j], s.names[j]))[:(len(donors) + 1) // 2]
    predictions['scalar_roundtrip_tophalf'] = host_predict(s, {j: p1[j] for j in chosen}, gold)
    info = dict(z0=z0, maps=maps, starts=starts, area=area, B=B, Bw=Bw, gram=gram,
                y=y, v=v, residual=residual, delta=delta, visited=visited, corrected=corrected,
                actual_mu=effective_mu(gram),
                roundtrip=rt, selected=chosen, support_patches=ps)
    return predictions, info


def evaluate_diagnostics(s, p1, info, real_gt):
    """Independent evaluator: donor truth arrives ONLY after deployable repair."""
    donors = list(range(2, s.n))
    predictions = {'true_donor_masks': host_predict(s, {j: real_gt[j] for j in donors}, s.gt64[0])}
    if info is None:
        return predictions, {'no_donors': True}
    z0, maps, area = info['z0'], info['maps'], info['area']
    positive_count = torch.zeros_like(z0)
    for j, ix in maps.items():
        positive_count.scatter_add_(0, ix, real_gt[j].float())
    zgt = 2 * positive_count / area - 1
    error = zgt - z0
    observed, shrunk, signal, spectral = error_projections(info['Bw'], info['gram'], error, mu=info['actual_mu'])
    mismatch = info['v'].sqrt() * info['y'] - info['Bw'] @ zgt
    predictions['diagnostic_gt_atom_labels'] = host_predict(s, masks_from_margins(zgt, p1, maps), s.gt64[0])
    predictions['diagnostic_GT_projection'] = host_predict(s, masks_from_margins(z0 + observed, p1, maps), s.gt64[0])
    predictions['diagnostic_GT_ridge_projection'] = host_predict(s, masks_from_margins(z0 + shrunk, p1, maps), s.gt64[0])
    oracle_visited = {j: torch.where(info['visited'][ix], real_gt[j], p1[j]) for j, ix in maps.items()}
    predictions['diagnostic_GT_visited_atoms_only'] = host_predict(s, oracle_visited, s.gt64[0])
    # Downstream route-use weighting: query is read only by this evaluator.
    query_values, query_atoms = [], []
    for j, ix in maps.items():
        values, patch = s.nnv(1, j)
        query_values.append(values)
        query_atoms.append(ix[patch])
    top = torch.stack(query_values).topk(min(5, len(donors)), dim=0).indices
    route_atoms = torch.stack(query_atoms).gather(0, top).flatten()
    frequency = torch.bincount(route_atoms, minlength=len(z0)).float() / top.shape[0]
    strata = {'all': torch.ones_like(z0, dtype=torch.bool),
              'FP': (z0 > 0) & (zgt < 0), 'FN': (z0 < 0) & (zgt > 0),
              'visited': info['visited'], 'unvisited': ~info['visited']}
    eta = {}
    for name, take in strata.items():
        eta[name] = dict(atoms=int(take.sum()), error_energy=float(error[take].square().sum()),
            eta=energy_fraction(observed[take], error[take]),
            eta_ridge=energy_fraction(shrunk[take], error[take]),
            eta_area=energy_fraction(observed[take], error[take], area[take]),
            eta_ridge_area=energy_fraction(shrunk[take], error[take], area[take]),
            eta_query_usage=energy_fraction(observed[take], error[take], frequency[take]),
            eta_ridge_query_usage=energy_fraction(shrunk[take], error[take], frequency[take]))
    return predictions, dict(atoms=len(z0), support_samples=len(info['y']),
        v_sum=float(info['v'].sum()), v_mean=float(info['v'].mean()),
        actual_mu=float(info['actual_mu']), gram_trace=float(torch.trace(info['gram'])),
        gram_mean_diagonal=float(torch.trace(info['gram']) / info['gram'].shape[0]),
        routed_atoms=int(info['visited'].sum()), routed_area_fraction=float(area[info['visited']].sum() / area.sum()),
        changed_atoms=int(((z0 + info['delta'] > 0) != (z0 > 0)).sum()),
        changed_donor_patches=sum(int((info['corrected'][j] != p1[j]).sum()) for j in donors),
        row_sum_max_error=float((info['B'].sum(1) - 1).abs().max()),
        zero_route_delta_max=float(info['delta'][~info['visited']].abs().max()) if (~info['visited']).any() else 0.0,
        support_residual_before=float(info['residual'].square().sum()),
        support_residual_after=float((info['residual'] - info['Bw'] @ info['delta']).square().sum()),
        signal_energy=float(signal.square().sum()), model_mismatch_energy=float(mismatch.square().sum()),
        signal_to_mismatch=float(signal.square().sum() / mismatch.square().sum().clamp_min(1e-12)),
        eta=eta, spectral=spectral, scalar_roundtrip=info['roundtrip'], selected_donor_indices=info['selected'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--file', required=True)
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prepared-root', default='/root/autodl-tmp/demo9')
    ap.add_argument('--M', type=int, default=15)
    ap.add_argument('--pool-seed', type=int, default=0)
    ap.add_argument('--seconds', type=int, default=MAX_SECONDS)
    a = ap.parse_args()
    if not 1 <= a.limit <= MAX_EPISODES or not 1 <= a.seconds <= MAX_SECONDS or a.M < 0:
        ap.error('limit must be 1..40, seconds 1..1500, M >= 0')
    sys.path.insert(0, str(Path(a.prepared_root) / 'scripts'))
    import _paths  # Resolves the prepared host without modifying it.
    import numpy as np
    from tics import ImageSet, one_shot
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC', '.3')))
    start = time.monotonic()
    def deadline(_signum, _frame):
        raise TimeoutError('fixed probe wall-clock budget reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(a.seconds)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(state='RUNNING', args=vars(a), records=[], mu_coefficient=MU,
        objective_scale='V sum=1, FG/BG balanced; actual_mu=0.1*trace(Bw@Bw.T)/n_support_samples',
        interpretation='Single-fold bounded mechanism diagnostic; GT projections are not full-model oracle bounds.',
        eta_interpretation='Global unweighted eta is projection energy fraction; weighted and stratum ratios may exceed one and are not probabilities.',
        omitted_controls=['feature-classifier label inversion', 'generic graph propagation', 'FoRIS host'],
        safety='Inference receives ONLY support truth; P1 frozen; query excluded from donor fitting; no GT tuning.',
        source_sha256={str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    def write():
        acc = {}
        for r in report['records']:
            for key, (i, u) in r['iu'].items():
                row = acc.setdefault(key, {}).setdefault(r['c'], [0.0, 0.0])
                row[0] += i
                row[1] += u
        report['miou'] = {k: 100 * float(np.mean([i / max(u, 1) for i, u in rows.values()])) for k, rows in acc.items()}
        report['episodes'] = len(report['records'])
        report['elapsed_s'] = time.monotonic() - start
        tmp = out.with_suffix(out.suffix + '.tmp')
        tmp.write_text(json.dumps(report))
        tmp.replace(out)
    try:
        D = torch.load(a.file, weights_only=False, map_location='cpu')
        cls, names = np.array(D['cls']), D['names']
        n = len(names) // 2
        report['fold'] = D.get('fold')
        report['feature_space'] = 'Cached debiased fq; same FP32 normalization as prepared ImageSet'
        report['source_sha256'][str(Path(a.prepared_root) / 'tics/imageset.py')] = hashlib.sha256((Path(a.prepared_root) / 'tics/imageset.py').read_bytes()).hexdigest()
        rng = np.random.default_rng(a.pool_seed)
        write()
        for e in range(min(a.limit, n)):
            episode_start = time.monotonic()
            c, ref, q = int(cls[2 * e]), 2 * e, 2 * e + 1
            candidates = [2 * o + 1 for o in range(n) if o != e and cls[2 * o] == c and names[2 * o + 1] not in (names[ref], names[q])]
            seen = set()
            candidates = [x for x in candidates if not (names[x] in seen or seen.add(names[x]))]
            pool = [int(x) for x in rng.permutation(candidates)[:a.M]]
            idx = [ref, q] + pool
            legal_gt = torch.zeros_like(D['gt64'][idx])
            legal_gt[0] = D['gt64'][ref]
            legal_bits = np.zeros_like(D['gt_bits'][idx])
            legal_bits[0] = D['gt_bits'][ref]
            s = ImageSet(dict(c=c, names=[names[x] for x in idx], fq=D['fq'][idx], lab=D['lab'][idx],
                Po=[D['Po'][x] for x in idx], gt64=legal_gt, gt_bits=legal_bits, S=D['S']))
            with torch.no_grad():
                p1 = one_shot(s, list(range(1, s.n)))
                predictions, info = repair(s, p1)
                # Freeze legal outputs before the independent diagnostic sees truth.
                predictions = {k: v.clone() for k, v in predictions.items()}
                real = D['gt64'][idx].to(s.fd.device).bool().reshape(s.n, -1)
                diagnostic_predictions, diag = evaluate_diagnostics(s, p1, info, real)
                predictions.update(diagnostic_predictions)
                truth = torch.from_numpy(np.unpackbits(D['gt_bits'][q])).to(s.fd.device).bool().reshape(s.S, s.S)
                iu = {}
                for key, m in predictions.items():
                    full = s.up(m)
                    iu[key] = [float((full & truth).sum()), float((full | truth).sum())]
            report['records'].append(dict(e=e, c=c, support=names[ref], query=names[q], donor_ids=[names[x] for x in pool],
                pool=len(pool), episode_s=time.monotonic() - episode_start, iu=iu, diagnostic=diag))
            del s, info, predictions, diagnostic_predictions, real, p1
            write()
            print(json.dumps(dict(episode=e, elapsed_s=round(report['elapsed_s'], 2), miou=report['miou'])), flush=True)
        report['state'] = 'COMPLETED'
    except TimeoutError as ex:
        report.update(state='BUDGET_STOP', error=str(ex))
    except BaseException as ex:
        report.update(state='ERROR', error=repr(ex))
        write()
        raise
    finally:
        signal.alarm(0)
        write()


if __name__ == '__main__':
    main()
