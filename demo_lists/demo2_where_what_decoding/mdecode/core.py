"""Metric-optimal class-affine decoding for dataset-level confusion-matrix metrics.

For a metric that depends on the per-class true-positive mass TP_c, predicted mass P_c and ground-truth
mass G_c of the WHOLE evaluated set, the first-order-optimal decision for a pixel with posterior p is

    y = argmax_c  alpha_c * p_c - beta_c ,

with (alpha, beta) the partial derivatives of the metric at the solution (see THEORY.md, Theorem 1).
argmax is the special case alpha = 1, beta = 0, which is optimal for pixel accuracy only.

Everything here is label-free unless a function says otherwise: statistics are expectations under the
model's own probabilities.
"""
import torch

METRICS = ("miou", "mdice", "macc", "fwiou")


def _safe(x, eps):
    return x.clamp(min=eps)


def gains(metric, TP, P, G, eps=1e-12):
    """(alpha, beta) such that moving one unit of pixel mass with posterior p_c into class c changes the
    metric by alpha_c * p_c - beta_c (up to the common 1/C factor). TP, P, G share any common scale."""
    if metric == "miou":      # J = TP / U,  U = G + P - TP
        U = _safe(G + P - TP, eps); J = TP / U
        return (1 + J) / U, J / U
    if metric == "mdice":     # D = 2 TP / (G + P)
        S = _safe(G + P, eps); D = 2 * TP / S
        return 2 / S, D / S
    if metric == "macc":      # recall = TP / G : the "maximum-likelihood" rule p_c / prior_c
        return 1 / _safe(G, eps), torch.zeros_like(G)
    if metric == "fwiou":     # sum_c G_c * J_c
        U = _safe(G + P - TP, eps); J = TP / U
        return G * (1 + J) / U, G * J / U
    raise ValueError(metric)


def value(metric, TP, P, G, eps=1e-12):
    if metric == "miou":
        return (TP / _safe(G + P - TP, eps)).mean()
    if metric == "mdice":
        return (2 * TP / _safe(G + P, eps)).mean()
    if metric == "macc":
        return (TP / _safe(G, eps)).mean()
    if metric == "fwiou":
        return (G * TP / _safe(G + P - TP, eps)).sum() / _safe(G.sum(), eps)
    raise ValueError(metric)


def decode(p, alpha, beta, dim=0):
    """p: probabilities with the class axis at `dim`. Returns integer labels."""
    shape = [1] * p.dim(); shape[dim] = -1
    return (alpha.view(shape) * p - beta.view(shape)).argmax(dim)


class SoftStats:
    """Expected TP / P / G of a set of decisions under the model's own probabilities (no labels)."""

    def __init__(self, num_classes, device="cpu"):
        z = lambda: torch.zeros(num_classes, dtype=torch.float64, device=device)
        self.TP, self.P, self.G, self.n, self.C = z(), z(), z(), 0, num_classes

    def add(self, p, pred):
        """p: (C, ...) probabilities; pred: (...) labels chosen for the same pixels."""
        p = p.reshape(self.C, -1); pred = pred.reshape(-1)
        self.P += torch.bincount(pred, minlength=self.C).double()
        self.G += p.sum(1).double()
        self.TP.scatter_add_(0, pred, p.gather(0, pred[None])[0].double())
        self.n += pred.numel()

    def fractions(self, smooth=1e-7):
        return self.TP / self.n, self.P / self.n, self.G / self.n + smooth


def fit_labelfree(sweep, num_classes, metric="miou", iters=4, damping=0.5, device="cpu", smooth=1e-7):
    """Transductive fixed point.  `sweep(alpha, beta)` must return a SoftStats for the decisions obtained
    with decode(p, alpha, beta) over the whole evaluated set (one pass over the data).
    Returns (alpha, beta, history); the returned parameters are the ones with the highest EXPECTED metric
    seen so far (safeguard: never worse than argmax in expectation)."""
    alpha = torch.ones(num_classes, device=device); beta = torch.zeros(num_classes, device=device)
    best, hist, la, lb = None, [], None, None
    for k in range(iters + 1):
        s = sweep(alpha, beta); TP, P, G = s.fractions(smooth)
        psi = float(value(metric, TP, P, G))
        hist.append(psi)
        if best is None or psi > best[0]:
            best = (psi, alpha.clone(), beta.clone())
        if k == iters:
            break
        a, b = gains(metric, TP, P, G, eps=smooth)
        a, b = a.float().to(device), b.float().to(device)
        scale = a.max()                                   # argmax is invariant to a common positive scale
        a, b = a / scale, b / scale
        if la is None:
            la, lb = a, b
        else:                                             # damp in the (log alpha, beta/alpha) parametrisation
            thr = (1 - damping) * (lb / la) + damping * (b / a)
            la = torch.exp((1 - damping) * la.log() + damping * a.log()); lb = thr * la
        alpha, beta = la, lb
    return best[1], best[2], hist


def fit_labeled(sweep_confusion, num_classes, metric="miou", iters=4, damping=0.5, device="cpu", smooth=1e-7):
    """Same fixed point, but statistics come from a labelled calibration set.
    `sweep_confusion(alpha, beta)` returns the (C, C) confusion matrix [true, predicted] of that set."""
    alpha = torch.ones(num_classes, device=device); beta = torch.zeros(num_classes, device=device)
    best, hist, la, lb = None, [], None, None
    for k in range(iters + 1):
        conf = sweep_confusion(alpha, beta).double(); n = conf.sum()
        TP, P, G = conf.diag() / n, conf.sum(0) / n, conf.sum(1) / n + smooth
        val = float(value(metric, TP, P, G)); hist.append(val)
        if best is None or val > best[0]:
            best = (val, alpha.clone(), beta.clone())
        if k == iters:
            break
        a, b = gains(metric, TP, P, G, eps=smooth)
        a, b = a.float().to(device), b.float().to(device)
        scale = a.max(); a, b = a / scale, b / scale
        if la is None:
            la, lb = a, b
        else:
            thr = (1 - damping) * (lb / la) + damping * (b / a)
            la = torch.exp((1 - damping) * la.log() + damping * a.log()); lb = thr * la
        alpha, beta = la, lb
    return best[1], best[2], hist
