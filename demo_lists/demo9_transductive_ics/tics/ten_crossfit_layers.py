"""Two independent single-pair inference proposals; no encoder training or GT Q.

M07 compares spatially held-out query BG modes to the same unheld modes.
M08 compares one shared three-layer correspondence to late fusion and to its
mathematically equivalent concatenated descriptor. No operator novelty claim.
"""
import math
import sys

import torch
import torch.nn.functional as F


def unit(x):
    return F.normalize(x, dim=-1, eps=1e-8)


def minmax(x):
    z = x - x.min()
    return z / z.max().clamp_min(1e-6)


def modes(x, maximum=4, iterations=12):
    """Deterministic spherical FPS/k-means; mixture masses sum to one."""
    if not len(x):
        return x, x.new_empty((0,))
    x = unit(x.float())
    chosen = [0]
    closest = (1 - x @ x[0]).clamp_min(0)
    for _ in range(1, min(maximum, len(x))):
        if float(closest.max()) < 1e-6:
            break
        j = int(closest.argmax())
        chosen.append(j)
        closest = torch.minimum(closest, (1 - x @ x[j]).clamp_min(0))
    c = x[chosen].clone()
    for _ in range(iterations):
        labels = (x @ c.T).argmax(1)
        sums = torch.zeros_like(c).index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=len(c))
        occupied = counts > 0
        c[occupied] = unit(sums[occupied])
    labels = (x @ c.T).argmax(1)
    counts = torch.bincount(labels, minlength=len(c)).float()
    keep = counts > 0
    return c[keep], counts[keep] / counts[keep].sum()


def reference_roles(ctx):
    fg = ctx.reference_coverage.flatten() > .5
    if not bool(fg.any()):
        fg = ctx.reference_coverage.flatten() > 0
    return fg, ~fg


def public_bg_score(ctx, original, kw):
    """Reconstruct the original native orthogonalized BG score, not sbn."""
    module = sys.modules[ctx.host.__class__.__module__]
    n, h, w = kw['n_refs'], kw['h'], kw['w']
    ds = torch.stack([module.downsample_mask(kw['ref_masks'][s:s+1], h, w)
                      for s in range(n)], 0)
    stats = ctx.host._reference_contrastive_prototypes(
        kw['fmaps_norm'][:, :n], ds, n, mu_fg_per_reference=True)
    if stats is None:
        return None
    mu_f, mu_b, _ = stats
    orth = mu_b - (mu_b * mu_f).sum() * mu_f
    mu_b = orth / orth.norm().clamp_min(1e-8)
    target = (kw['fmaps_norm'][:, n] if ctx.host.use_raw_target_scoring
              else original[4])
    return torch.einsum('bchw,c->bhw', target, mu_b).squeeze(0)


def query_background(ctx, mu_fg, *, held_out):
    h, w = ctx.grid_hw
    q = ctx.query_tokens.float()
    native = F.interpolate(ctx.native['mask'][None, None].float(),
                           (h, w), mode='area')[0, 0] > .5
    score = ctx.native['score'].reshape(h, w)
    excluded = ~native
    if not bool(excluded.any()):
        return q.new_zeros(h, w), torch.zeros_like(excluded), []
    # Relative exclusion evidence, not calibrated posterior or a BG guarantee.
    safe = excluded & (score <= score[excluded].median())
    yy, xx = torch.meshgrid(torch.arange(h, device=q.device),
                            torch.arange(w, device=q.device), indexing='ij')
    result = q.new_zeros(h, w)
    active = torch.zeros_like(safe)
    audits = []
    tau = max(float(ctx.host.cluster_logsumexp_temp), 1e-4)
    mf = unit(mu_fg.reshape(1, -1).float())[0]
    for by in range(2):
        for bx in range(2):
            y0, y1 = by*h//2, (by+1)*h//2
            x0, x1 = bx*w//2, (bx+1)*w//2
            target = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
            forbidden = ((yy >= max(0, y0-1)) & (yy < min(h, y1+1)) &
                         (xx >= max(0, x0-1)) & (xx < min(w, x1+1)))
            eligible = safe & ~forbidden if held_out else safe.clone()
            c, masses = modes(q[eligible.flatten()], maximum=4)
            if len(c):
                orth = c - (c @ mf)[:, None] * mf
                keep = orth.norm(dim=1) > 1e-6
                c, masses = unit(orth[keep]), masses[keep]
                if len(c):
                    masses = masses / masses.sum()
                    z = q[target.flatten()] @ c.T
                    result[target] = tau * torch.logsumexp(
                        z/tau + masses.log()[None], dim=1)
                    active[target] = True
            audits.append(dict(block=[by, bx], candidates=int(eligible.sum()),
                               modes=len(c), excluded_target_halo=held_out,
                               halo_leak=int((eligible & forbidden).sum()) if held_out else None))
    return result, active, audits


def run_crossfit(ctx, held_out=True):
    audit = dict(method='m07', held_out=held_out, tiles=[2, 2], halo_cells=1,
                 modes_per_block=4, iterations=12, mix=.5,
                 selection='native-excluded lower-half score; not GT BG',
                 independence_claim=False, encoder_training=False)

    def change(original_result, ctx, **kw):
        old, sf, sbn, mu_fg, denoised = original_result
        bq, active, blocks = query_background(ctx, mu_fg, held_out=held_out)
        audit['blocks'] = blocks
        audit['active_cells'] = int(active.sum())
        if not bool(active.any()):
            audit['state'] = 'ABSTAIN_NO_LEGAL_QUERY_BACKGROUND'
            return original_result
        bs = public_bg_score(ctx, original_result, kw)
        if bs is None:
            audit['state'] = 'ABSTAIN_MISSING_REFERENCE_ROLES'
            return original_result
        mix = bs.clone()
        mix[active] = .5 * (bs[active] + bq[active])
        weight = float(ctx.host.dino_bg_weight)
        score = old + weight * (bs - mix)
        audit['state'] = 'ACTIVE'
        audit['background_weight'] = weight
        return score, sf, minmax(mix), mu_fg, denoised

    out = ctx.replay(part2_override=change)
    out['audit']['m07'] = audit
    return out


def layered_response(ctx, mode='shared', chunk=256):
    """LME on the same FG/BG banks and three unit-normalized layer maps."""
    fg, bg = reference_roles(ctx)
    if not bool(fg.any()) or not bool(bg.any()):
        return None
    maps = [ctx.features[k] for k in (6, 12, 'final')]
    pairs = [(unit(m[0].flatten(1).T.float()),
              unit(m[1].flatten(1).T.float())) for m in maps]
    tau = max(float(ctx.host.cluster_logsumexp_temp), 1e-4)
    nf, nb = int(fg.sum()), int(bg.sum())
    f, b = [], []
    concat_error = 0.
    for lo in range(0, len(pairs[0][1]), chunk):
        hi = min(lo+chunk, len(pairs[0][1]))
        cosines = [q[lo:hi] @ r.T for r, q in pairs]
        average = torch.stack(cosines).mean(0)
        cr = torch.cat([r for r, _ in pairs], dim=1) / math.sqrt(3)
        cq = torch.cat([q[lo:hi] for _, q in pairs], dim=1) / math.sqrt(3)
        direct = cq @ cr.T
        concat_error = max(concat_error, float((direct-average).abs().max()))
        if mode == 'late':
            fs = [(tau * (torch.logsumexp(c[:, fg]/tau, 1)-math.log(nf)))
                  for c in cosines]
            bs = [(tau * (torch.logsumexp(c[:, bg]/tau, 1)-math.log(nb)))
                  for c in cosines]
            f.append(torch.stack(fs).mean(0)); b.append(torch.stack(bs).mean(0))
        else:
            c = direct if mode == 'concat' else average
            f.append(tau * (torch.logsumexp(c[:, fg]/tau, 1)-math.log(nf)))
            b.append(tau * (torch.logsumexp(c[:, bg]/tau, 1)-math.log(nb)))
    return torch.cat(f).reshape(ctx.grid_hw), torch.cat(b).reshape(ctx.grid_hw), dict(
        layers=[6, 12, 'final_public_Part1'], per_layer='unit patch descriptors',
        weights=[1/3]*3, tau=tau, fg_points=nf, bg_points=nb,
        concat_dot_max_error=concat_error,
        equivalent_concat_operator=True, mode=mode,
        mathematical_novelty_claim=False)


def run_layers(ctx, mode='shared'):
    responses = layered_response(ctx, mode)
    if responses is None:
        return {**ctx.native, 'audit': {**ctx.native.get('audit', {}),
                 'm08': {'state': 'ABSTAIN_MISSING_REFERENCE_ROLE'}}}
    f, b, audit = responses
    def change(original_result, ctx, **kw):
        _, _, _, mu_fg, denoised = original_result
        return f-b, minmax(f), minmax(b), mu_fg, denoised
    out = ctx.replay(part2_override=change)
    out['audit']['m08'] = audit
    return out


METHODS = [
    dict(id='m07', name='spatially_crossfit_query_background',
         function=lambda ctx: run_crossfit(ctx, True),
         naive=lambda ctx: run_crossfit(ctx, False),
         card=[
             'Assumption: query-excluded modes outside a target spatial block provide useful negative evidence beyond self-confirming modes.',
             'Prediction: zero halo leakage, finite common-cosine scores; positive full-output paired advantage over native and no-holdout is required, no pp forecast.',
             'Match: retain only if removal/repair ledger shows task gain beyond the same-mode no-holdout control on all folds.',
             'Mismatch: equal/worse than no-holdout refutes useful crossfitting; foreground-like negative modes demonstrate correlated-error failure.']),
    dict(id='m08', name='shared_cross_layer_correspondence',
         function=lambda ctx: run_layers(ctx, 'shared'),
         naive=lambda ctx: run_layers(ctx, 'late'),
         controls={'concat': lambda ctx: run_layers(ctx, 'concat')},
         card=[
             'Assumption: a common reference point across layers provides useful evidence beyond separate per-layer matches.',
             'Prediction: concatenated and averaged-dot operators agree to float32 rounding; task gain over native and late fusion is required, no claim of gain over equivalent concat.',
             'Match: evidence supports shared multi-layer readout, not a new mathematical operator; foreground and background opportunity are matched.',
             'Mismatch: no task advantage rejects this equal-weight layer combination; concat disagreement is a numerical/interface defect to repair, not novelty.'])]


def cpu_selfcheck():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    c, p = modes(unit(torch.randn(16, 8)))
    assert len(c) <= 4 and abs(float(p.sum())-1) < 1e-6
    assert torch.allclose(c.norm(dim=1), torch.ones(len(c)), atol=1e-6)
    constant, masses = modes(torch.ones(10, 8))
    assert len(constant) == 1 and float(masses[0]) == 1
    rs = [unit(torch.randn(7, 8)) for _ in range(3)]
    qs = [unit(torch.randn(11, 8)) for _ in range(3)]
    avg = sum(q @ r.T for q, r in zip(qs, rs))/3
    direct = (torch.cat(qs, 1)/math.sqrt(3)) @ (torch.cat(rs, 1)/math.sqrt(3)).T
    assert torch.allclose(avg, direct, atol=2e-6, rtol=2e-6)
    mf = unit(torch.randn(1, 8))[0]
    x = unit(torch.randn(4, 8)); orth = unit(x-(x@mf)[:, None]*mf)
    assert float((orth @ mf).abs().max()) < 2e-6
    assert torch.isfinite(minmax(torch.ones(4, 4))).all()
    return dict(state='CPU_CROSSFIT_LAYERS_CHECKED', cases=5,
                CUDA_initialized=torch.cuda.is_initialized(),
                training=False, query_GT_read=False)


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    p = argparse.ArgumentParser(); p.add_argument('--cpu-check', required=True)
    a = p.parse_args(); result = cpu_selfcheck()
    Path(a.cpu_check).parent.mkdir(parents=True, exist_ok=True)
    Path(a.cpu_check).write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
