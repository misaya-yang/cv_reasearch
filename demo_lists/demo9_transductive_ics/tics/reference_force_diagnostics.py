"""Read-only support-force diagnostics for the existing metric objective.

No query features, labels, training, gain changes or new metric are accepted.
Original FP32 design and witnesses stay fixed; FP64 is diagnostic arithmetic.
"""
import math
import torch
import torch.nn.functional as F
from .reference_metric import difference_design, preference, protected_step


def force_matrix(features, triples, mass, margins, config):
    f = features.double()
    dp = f[triples[:, 0]] - f[triples[:, 1]]
    dn = f[triples[:, 0]] - f[triples[:, 2]]
    probability = ((config.margin - margins.double()) / config.temperature).sigmoid()
    coefficients = mass.double() * probability
    c = dn.T @ (coefficients[:, None] * dn) - dp.T @ (coefficients[:, None] * dp)
    return (c + c.T) / 2, coefficients


def geometry(c, directions):
    u = directions.double()
    b = u.T @ c @ u
    center = torch.trace(c) / len(c)
    bc = b - center * (u.T @ u)
    cf = c - center * torch.eye(len(c), dtype=c.dtype, device=c.device)
    diagonal = b.diag()
    inside = float(b.square().sum())
    cross = float((c @ u).square().sum()) - inside
    total = float(c.square().sum())
    return dict(total_energy=total, centered_energy=float(cf.square().sum()),
                diagonal_energy=float(diagonal.square().sum()),
                centered_diagonal_energy=float(bc.diag().square().sum()),
                projected_energy=inside,
                projected_offdiagonal_energy=float((b - torch.diag(diagonal)).square().sum()),
                one_sided_cross_energy=cross,
                outside_block_energy=total - inside - 2 * cross,
                gram_max_defect=float((u.T @ u - torch.eye(u.shape[1], device=u.device)).abs().max()),
                decomposition_scope='Exact only for orthonormal U; original numerical U is preserved',
                projected_force=b.cpu().tolist())


def response(features, coverage, grid, triples, mass, directions, config, c):
    margins, original_design = difference_design(features, directions, triples)
    a, m, p = original_design.double(), margins.double(), mass.double()
    t, lam = config.temperature, config.regularization
    sig0 = ((config.margin - m) / t).sigmoid()
    g0 = -a.T @ (p * sig0) / t
    preferred64 = preference(a, m, p, config)
    preferred32 = preference(original_design, margins, mass, config)
    applied32, alpha32 = protected_step(preferred32, original_design, margins, config)
    r = preferred64 - 1
    z = (config.margin - m - a @ r) / t
    sig = z.sigmoid()
    grad = lam * r - a.T @ (p * sig) / t
    h0 = lam * torch.eye(len(r), device=a.device, dtype=a.dtype) + a.T @ ((p * sig0 * (1 - sig0) / t ** 2)[:, None] * a)
    hs = lam * torch.eye(len(r), device=a.device, dtype=a.dtype) + a.T @ ((p * sig * (1 - sig) / t ** 2)[:, None] * a)
    unit_bound = float(g0.norm()) / lam
    slope = original_design @ (preferred32 - 1)
    good = margins >= config.margin
    return dict(triplets=len(triples), pure_fg=int((coverage >= .9).sum()),
                pure_bg=int((coverage <= .1).sum()),
                reliable_fraction=float(good.float().mean()),
                weighted_logistic_force=float((p * sig0).sum()),
                g0_norm=float(g0.norm()), force_rms=float(g0.norm()) / math.sqrt(len(r)),
                lambda_value=lam, temperature=t,
                strong_convexity_norm_bound=unit_bound,
                preferred64_norm=float(r.norm()), preferred64_rms=float(r.square().mean().sqrt()),
                bound_usage=float(r.norm()) / unit_bound if unit_bound else None,
                preferred32_rms=float((preferred32 - 1).square().mean().sqrt()),
                applied32_rms=float((applied32 - 1).square().mean().sqrt()),
                alpha=float(alpha32), preferred32_weights=preferred32.cpu().tolist(),
                applied32_weights=applied32.cpu().tolist(),
                guard_active_at_full_step=int((good & (slope < -(1 - config.retain_fraction) * margins.clamp_min(0))).sum()),
                spectral_active_at_full_step=int(((preferred32 < config.minimum) | (preferred32 > config.maximum)).sum()),
                stationarity64_l2=float(grad.norm()), solution_error_bound=float(grad.norm()) / lam,
                hessian0_eigenvalues=torch.linalg.eigvalsh(h0).cpu().tolist(),
                hessian_solution_eigenvalues=torch.linalg.eigvalsh(hs).cpu().tolist(),
                force_vs_original_design_max_difference=float((g0 + torch.diag(directions.double().T @ c @ directions.double()) / t).abs().max()),
                force_projection=geometry(c, directions),
                grid=list(grid), no_query_access=True)


def self_check():
    from .reference_metric import MetricConfig
    torch.manual_seed(2049)
    checks = []
    for index in range(10):
        f = F.normalize(torch.randn(12, 6, dtype=torch.float64), dim=-1)
        u = torch.linalg.qr(torch.randn(6, 3, dtype=torch.float64), mode='reduced')[0]
        triples = torch.tensor([[0, 1, 6], [2, 3, 8], [6, 7, 0], [8, 9, 2]])
        mass = torch.full((4,), .25, dtype=torch.float64)
        config = MetricConfig(rank=3)
        m, a = difference_design(f, u, triples)
        c, coefficients = force_matrix(f, triples, mass, m, config)
        g0 = -a.T @ coefficients / config.temperature
        assert torch.allclose(g0, -torch.diag(u.T @ c @ u) / config.temperature, atol=1e-10, rtol=1e-10)
        rr = response(f, torch.tensor([1.] * 6 + [0.] * 6), (3, 4), triples, mass, u, config, c)
        assert rr['preferred64_norm'] <= rr['strong_convexity_norm_bound'] + 1e-9
        assert rr['stationarity64_l2'] < 1e-8
        geo = rr['force_projection']
        assert geo['outside_block_energy'] >= -1e-10
        assert min(rr['hessian_solution_eigenvalues']) >= config.regularization - 1e-10
        checks.append(dict(index=index, force_error=rr['force_vs_original_design_max_difference'],
                           stationarity=rr['stationarity64_l2']))
    return dict(state='CPU_FORCE_DIAGNOSTICS_PASSED', cases=checks, real_DINO=False,
                query_GT_access=False, CUDA_calls=False, training=False)
