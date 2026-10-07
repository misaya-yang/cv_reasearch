"""Episode-local reference supervision, with no query-label fitting.

These methods optimize temporary classifiers using the known reference mask.
DINO remains frozen; no external model weights, datasets or labels are used.
DR11 was withdrawn as a duplicate of the existing degree-two kernel method.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .common import Result, unit


def _finish(*args, **kwargs):
    from .decision_risk import _finish as finish
    return finish(*args, **kwargs)


def _degenerate(*args, **kwargs):
    from .decision_risk import _degenerate as degenerate
    return degenerate(*args, **kwargs)


def _pure_reference(*args, **kwargs):
    from .decision_risk import _pure_reference as pure
    return pure(*args, **kwargs)


def hull_margin(*args, **kwargs):
    from .decision_risk import hull_margin as solve
    return solve(*args, **kwargs)


def _pure_data(ep):
    fi, bi = _pure_reference(ep)
    if not len(fi) or not len(bi):
        return None
    ids = np.r_[fi, bi]
    y = np.r_[np.ones(len(fi)), -np.ones(len(bi))]
    weights = .5 * np.r_[ep.wf[fi] / ep.wf[fi].sum(), ep.wb[bi] / ep.wb[bi].sum()]
    return ep.r[ids], y, weights, ids


def _start(ep, method):
    fallback = _degenerate(ep, method)
    if fallback:
        return fallback, None
    data = _pure_data(ep)
    if data is None:
        from .common import prototype_margin
        return _finish(ep, prototype_margin(ep), method, state="no_pure_reference_class_fallback"), None
    return None, data


def _anchors(x, weights, cap=16):
    chosen = [int(np.argmax(weights))]
    while len(chosen) < min(cap, len(x)):
        distance = 1 - (x @ x[chosen].T).max(axis=1)
        distance[chosen] = -1
        # Coincident/symmetric vectors should not reorder anchor coordinates
        # when a BLAS implementation changes the last rounding bit.
        j = int(np.flatnonzero(distance >= distance.max() - 1e-12)[0])
        if distance[j] < 1e-12:
            break
        chosen.append(j)
    return x[chosen]


def _nearest_margin(q, x, y):
    return (q @ x[y > 0].T).max(axis=1) - (q @ x[y < 0].T).max(axis=1)


def pure_nearest(ep):
    fallback, data = _start(ep, "DR_control_pure_nearest")
    if fallback:
        return fallback
    x, y, a, _ = data
    return _finish(ep, _nearest_margin(ep.q, x, y), "DR_control_pure_nearest")


def _stump_bank(z):
    z = np.round(z, 12)
    bank, columns = [], []
    for col in range(z.shape[1]):
        values = np.unique(np.quantile(z[:, col], np.linspace(0, 1, 17)))
        cuts = (values[:-1] + values[1:]) / 2
        for cut in cuts:
            for polarity in (1., -1.):
                bank.append((col, float(cut), polarity))
                columns.append(polarity * np.where(z[:, col] > cut, 1., -1.))
    return bank, np.asarray(columns).T


def boosted_prototype_stumps(ep):
    fallback, data = _start(ep, "DR09")
    if fallback:
        return fallback
    x, y, initial, _ = data
    anchors = _anchors(x, initial)
    bank, h = _stump_bank(x @ anchors.T)
    if not bank:
        return _finish(ep, _nearest_margin(ep.q, x, y), "DR09", state="empty_weak_bank_pure_NN_fallback")
    a, chosen, alphas = initial.copy(), [], []
    for _ in range(40):
        errors = a @ (h != y[:, None])
        j, error = int(np.argmin(errors)), float(np.min(errors))
        if error >= .5 - 1e-12:
            break
        alpha = .5 * np.log((1 - max(error, 1e-8)) / max(error, 1e-8))
        chosen.append(j)
        alphas.append(float(alpha))
        a *= np.exp(-alpha * y * h[:, j])
        a /= a.sum()
        if error < 1e-8:
            break
    if not chosen:
        return _finish(ep, _nearest_margin(ep.q, x, y), "DR09", state="weak_bank_uninformative_fallback")
    zq, margin = np.round(ep.q @ anchors.T, 12), np.zeros(len(ep.q))
    for j, alpha in zip(chosen, alphas):
        col, cut, polarity = bank[j]
        margin += alpha * polarity * np.where(zq[:, col] > cut, 1., -1.)
    return _finish(ep, margin, "DR09", rounds=len(chosen), anchors=len(anchors),
                   reference_final_weights=a.tolist(), episode_parameter_optimization=True,
                   external_training=False, DINO_backbone_frozen=True)


def best_prototype_stump(ep):
    fallback, data = _start(ep, "DR_control_best_stump")
    if fallback:
        return fallback
    x, y, a, _ = data
    anchors = _anchors(x, a)
    bank, h = _stump_bank(x @ anchors.T)
    if not bank:
        return _finish(ep, _nearest_margin(ep.q, x, y), "DR_control_best_stump", state="no_stump")
    col, cut, polarity = bank[int(np.argmin(a @ (h != y[:, None])))]
    return _finish(ep, polarity * np.where(np.round(ep.q @ anchors[col], 12) > cut, 1., -1.), "DR_control_best_stump")


def _bootstrap_members(ep, prototype=False):
    fallback, data = _start(ep, "DR_control_bootstrap_prototype" if prototype else "DR10")
    if fallback:
        return fallback
    x, y, a, _ = data
    fids, bids = np.flatnonzero(y > 0), np.flatnonzero(y < 0)
    rng, members, gaps = np.random.default_rng(0), [], []
    for _ in range(16):
        fi = rng.choice(fids, 16, replace=True, p=a[fids] / a[fids].sum())
        bi = rng.choice(bids, 16, replace=True, p=a[bids] / a[bids].sum())
        if prototype:
            margin = ep.q @ (unit(x[fi].mean(0)) - unit(x[bi].mean(0)))
        else:
            sub = replace(ep, r=x[np.r_[fi, bi]], wf=np.r_[np.ones(16), np.zeros(16)],
                          wvalid=np.ones(32), r_hw=(1, 32), reference_geometry={})
            result = hull_margin(sub)
            margin = result.margin.ravel()
            gaps.append(result.info.get("fw_gap"))
        members.append(np.sign(margin))
    members = np.asarray(members)
    method = "DR_control_bootstrap_prototype" if prototype else "DR10"
    return _finish(ep, members.mean(0), method, bootstrap_members=16, seed=0,
                   member_FW_gaps=gaps, independent_member_probability_claim=False,
                   disagreeing_query_tokens=int(np.count_nonzero(np.ptp(members, axis=0) > 0)))


def bootstrap_margin(ep):
    return _bootstrap_members(ep)


def bootstrap_prototype(ep):
    return _bootstrap_members(ep, prototype=True)


def _relu_value_grad(z, y, a, params):
    w, b, v, c = params
    pre = z @ w + b
    hidden = np.maximum(pre, 0)
    logits = hidden @ v + c
    loss = float(a @ np.logaddexp(0, -y * logits) + .0005 * (np.sum(w * w) + v @ v))
    coeff = -a * y / (1 + np.exp(np.clip(y * logits, -50, 50)))
    gh = coeff[:, None] * v[None, :] * (pre > 0)
    grads = (z.T @ gh + .001 * w, gh.sum(0), hidden.T @ coeff + .001 * v, coeff.sum())
    return loss, grads, logits


def reference_relu(ep):
    fallback, data = _start(ep, "DR12")
    if fallback:
        return fallback
    x, y, a, _ = data
    anchors = _anchors(x, a)
    z = x @ anchors.T
    rng = np.random.default_rng(0)
    params = (rng.normal(scale=np.sqrt(2 / (len(anchors) + 16)), size=(len(anchors), 16)),
              np.zeros(16), rng.normal(scale=np.sqrt(2 / 17), size=16), 0.)
    for _ in range(200):
        loss, grads, _ = _relu_value_grad(z, y, a, params)
        params = tuple(p - .05 * g for p, g in zip(params, grads))
    loss, grads, train = _relu_value_grad(z, y, a, params)
    w, b, v, c = params
    margin = np.maximum((ep.q @ anchors.T) @ w + b, 0) @ v + c
    return _finish(ep, margin, "DR12", anchors=len(anchors), hidden_units=16, steps=200,
                   final_reference_loss=loss, final_gradient_norm=float(np.sqrt(sum(np.sum(g * g) for g in grads))),
                   weighted_reference_error=float(a @ ((train > 0) != (y > 0))),
                   episode_parameter_optimization=True, no_parameter_fitting_claim=False,
                   external_training=False, DINO_backbone_frozen=True)


def anchor_linear_ridge(ep):
    """Same anchor input as ReLU/stumps; retired linear family, count zero."""
    fallback, data = _start(ep, "DR_control_anchor_ridge")
    if fallback:
        return fallback
    x, y, a, _ = data
    anchors = _anchors(x, a)
    z = np.c_[x @ anchors.T, np.ones(len(x))]
    coeff = np.linalg.solve(z.T @ (a[:, None] * z) + .1 * np.eye(z.shape[1]), z.T @ (a * y))
    return _finish(ep, np.c_[ep.q @ anchors.T, np.ones(len(ep.q))] @ coeff,
                   "DR_control_anchor_ridge")


def anchor_linear_logistic(ep):
    fallback, data = _start(ep, "DR_control_anchor_logistic")
    if fallback:
        return fallback
    x, y, a, _ = data
    anchors = _anchors(x, a)
    z = x @ anchors.T
    w, bias = np.zeros(len(anchors)), 0.
    for _ in range(200):
        logits = z @ w + bias
        coeff = -a * y / (1 + np.exp(np.clip(y * logits, -50, 50)))
        w -= .05 * (z.T @ coeff + .001 * w)
        bias -= .05 * coeff.sum()
    return _finish(ep, (ep.q @ anchors.T) @ w + bias, "DR_control_anchor_logistic", steps=200)


def anchor_lda(ep):
    fallback, data = _start(ep, "DR_control_anchor_LDA")
    if fallback:
        return fallback
    x, y, a, _ = data
    anchors = _anchors(x, a)
    z = x @ anchors.T
    f = np.average(z[y > 0], axis=0, weights=a[y > 0])
    b = np.average(z[y < 0], axis=0, weights=a[y < 0])
    residual = z - np.where((y > 0)[:, None], f[None, :], b[None, :])
    covariance = residual.T @ (a[:, None] * residual)
    direction = np.linalg.solve(covariance + .05 * np.eye(len(anchors)), f - b)
    return _finish(ep, (ep.q @ anchors.T) @ direction - .5 * float((f + b) @ direction),
                   "DR_control_anchor_LDA")


def edited_reference(ep):
    fallback, data = _start(ep, "DR13")
    if fallback:
        return fallback
    x, y, a, ids = data
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    neighbor = np.argsort(-sim, axis=1, kind="stable")[:, :min(3, len(x) - 1)]
    vote = np.sum(y[neighbor] * a[neighbor], axis=1)
    keep = (vote * y >= 0)
    if not np.any(keep & (y > 0)) or not np.any(keep & (y < 0)):
        keep[:] = True
        state = "editing_removed_class_unfiltered_fallback"
    else:
        state = "reference_role_neighbor_edit"
    return _finish(ep, _nearest_margin(ep.q, x[keep], y[keep]), "DR13", state=state,
                   retained_native_indices=ids[keep].tolist(), dropped_count=int(np.count_nonzero(~keep)))


def condensed_reference(ep):
    fallback, data = _start(ep, "DR14")
    if fallback:
        return fallback
    x, y, a, ids = data
    selected = []
    for sign in (1, -1):
        ci = np.flatnonzero(y == sign)
        mean = np.average(x[ci], axis=0, weights=a[ci])
        selected.append(int(ci[np.argmax(x[ci] @ mean)]))
    order = np.argsort(ids, kind="stable")
    sim = x @ x.T
    for _ in range(len(x)):
        changed = False
        for i in order:
            if y[selected[int(np.argmax(sim[i, selected]))]] != y[i] and i not in selected:
                selected.append(int(i))
                changed = True
        if not changed:
            break
    labels = y[np.asarray(selected)[np.argmax(sim[:, selected], axis=1)]]
    return _finish(ep, _nearest_margin(ep.q, x[selected], y[selected]), "DR14",
                   training_samples=len(x), selected_supports=len(selected),
                   support_native_indices=ids[selected].tolist(),
                   residual_reference_errors=int(np.count_nonzero(labels != y)),
                   query_equivalence_to_full_NN_claim=False)


def _kmeans(x, a, count=2):
    centers = [np.average(x, axis=0, weights=a)]
    while len(centers) < min(count, len(x)):
        d = np.sum((x[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2).min(1)
        j = int(np.argmax(d))
        if d[j] < 1e-12:
            break
        centers.append(x[j].copy())
    centers = np.asarray(centers)
    for _ in range(12):
        labels = np.argmin(np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
        for i in range(len(centers)):
            keep = labels == i
            if keep.any():
                centers[i] = np.average(x[keep], axis=0, weights=a[keep])
    return centers


def _vq_data(data):
    x, y, a, _ = data
    f, b = _kmeans(x[y > 0], a[y > 0]), _kmeans(x[y < 0], a[y < 0])
    return x, y, a, np.vstack((f, b)), np.r_[np.ones(len(f)), -np.ones(len(b))]


def _vq_value_grad(x, y, a, centers, roles):
    d = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    same = y[:, None] == roles[None, :]
    pos = np.argmin(np.where(same, d, np.inf), axis=1)
    neg = np.argmin(np.where(~same, d, np.inf), axis=1)
    row = np.arange(len(x))
    dp, dn = d[row, pos], d[row, neg]
    s = dp + dn + 1e-6
    loss = float(a @ ((dp - dn) / s))
    # epsilon contributes to both derivatives of the smoothed denominator.
    gp = a[:, None] * (2 * (2 * dn + 1e-6) / s ** 2)[:, None] * (centers[pos] - x)
    gn = -a[:, None] * (2 * (2 * dp + 1e-6) / s ** 2)[:, None] * (centers[neg] - x)
    grad = np.zeros_like(centers)
    np.add.at(grad, pos, gp)
    np.add.at(grad, neg, gn)
    return loss, grad


def _vq_margin(q, centers, roles):
    distance = np.sum(q * q, axis=1)[:, None] + np.sum(centers * centers, axis=1)[None, :] - 2 * q @ centers.T
    return distance[:, roles < 0].min(1) - distance[:, roles > 0].min(1)


def risk_prototypes(ep):
    fallback, data = _start(ep, "DR15")
    if fallback:
        return fallback
    x, y, a, centers, roles = _vq_data(data)
    for _ in range(200):
        loss, grad = _vq_value_grad(x, y, a, centers, roles)
        grad *= min(1., 5 / max(float(np.linalg.norm(grad)), 1e-12))
        centers -= .05 * grad
        centers /= np.maximum(1., np.linalg.norm(centers, axis=1) / 2)[:, None]
    loss, grad = _vq_value_grad(x, y, a, centers, roles)
    return _finish(ep, _vq_margin(ep.q, centers, roles), "DR15", steps=200,
                   final_reference_risk=loss, final_gradient_norm=float(np.linalg.norm(grad)),
                   prototype_norms=np.linalg.norm(centers, axis=1).tolist(),
                   episode_parameter_optimization=True, external_training=False)


def class_kmeans(ep):
    fallback, data = _start(ep, "DR_control_two_class_kmeans")
    if fallback:
        return fallback
    x, y, a, centers, roles = _vq_data(data)
    return _finish(ep, _vq_margin(ep.q, centers, roles), "DR_control_two_class_kmeans")


def _metric_targets(z, y):
    d = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
    same = y[:, None] == y[None, :]
    np.fill_diagonal(same, False)
    count = min(2, int(np.min(same.sum(1))))
    return np.argsort(np.where(same, d, np.inf), axis=1, kind="stable")[:, :count]


def _metric_value_grad(z, y, a, targets, matrix):
    transformed = z @ matrix.T
    distance = np.sum((transformed[:, None, :] - transformed[None, :, :]) ** 2, axis=2)
    wrong = y[:, None] != y[None, :]
    impostor = np.argmin(np.where(wrong, distance, np.inf), axis=1)
    grad = .002 * matrix
    loss = .001 * float(np.sum(matrix * matrix))
    for i in range(len(z)):
        for t in targets[i]:
            weight = a[i] / targets.shape[1]
            delta, other = z[i] - z[t], z[i] - z[impostor[i]]
            pull = np.outer(delta, delta)
            dpos = distance[i, t]
            hinge = 1 + dpos - distance[i, impostor[i]]
            loss += weight * (dpos + max(0., hinge))
            grad += 2 * weight * (matrix @ pull)
            if hinge > 0:
                grad += 2 * weight * (matrix @ (pull - np.outer(other, other)))
    return float(loss), grad


def _metric_margin(qz, z, y, matrix):
    qq, rr = qz @ matrix.T, z @ matrix.T
    distance = np.sum(qq * qq, axis=1)[:, None] + np.sum(rr * rr, axis=1)[None, :] - 2 * qq @ rr.T
    return distance[:, y < 0].min(1) - distance[:, y > 0].min(1)


def role_metric(ep):
    fallback, data = _start(ep, "DR16")
    if fallback:
        return fallback
    x, y, a, _ = data
    if min(np.count_nonzero(y > 0), np.count_nonzero(y < 0)) < 2:
        return _finish(ep, _nearest_margin(ep.q, x, y), "DR16", state="no_same_role_target_pair_fallback")
    anchors = _anchors(x, a)
    z = x @ anchors.T
    targets = _metric_targets(z, y)
    matrix = np.eye(len(anchors))[:min(8, len(anchors))].copy()
    for _ in range(200):
        loss, grad = _metric_value_grad(z, y, a, targets, matrix)
        grad *= min(1., 5 / max(float(np.linalg.norm(grad)), 1e-12))
        matrix -= .02 * grad
    loss, grad = _metric_value_grad(z, y, a, targets, matrix)
    return _finish(ep, _metric_margin(ep.q @ anchors.T, z, y, matrix), "DR16", steps=200,
                   final_reference_metric_loss=loss, gradient_norm=float(np.linalg.norm(grad)),
                   learned_metric_rank=min(8, len(anchors)), anchors=len(anchors),
                   blind_whitening_used=False, episode_parameter_optimization=True,
                   external_training=False, query_semantic_preservation_guarantee=False)


def anchor_metric_identity(ep):
    fallback, data = _start(ep, "DR_control_anchor_metric_identity")
    if fallback:
        return fallback
    x, y, a, _ = data
    anchors = _anchors(x, a)
    return _finish(ep, _metric_margin(ep.q @ anchors.T, x @ anchors.T, y, np.eye(len(anchors))),
                   "DR_control_anchor_metric_identity")


METHODS = {"DR09": boosted_prototype_stumps, "DR10": bootstrap_margin,
           "DR12": reference_relu, "DR13": edited_reference,
           "DR14": condensed_reference, "DR15": risk_prototypes, "DR16": role_metric}
CONTROLS = {"DR_control_pure_nearest": pure_nearest,
            "DR_control_best_stump": best_prototype_stump,
            "DR_control_bootstrap_prototype": bootstrap_prototype,
            "DR_control_anchor_ridge": anchor_linear_ridge,
            "DR_control_anchor_logistic": anchor_linear_logistic,
            "DR_control_anchor_LDA": anchor_lda,
            "DR_control_two_class_kmeans": class_kmeans,
            "DR_control_anchor_metric_identity": anchor_metric_identity}
