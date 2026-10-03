"""Where to stop: choosing the target's extent along the reference-induced chain of level sets.

A complete solver such as FoRIS ends with a membership score on the patch grid and cuts it at a constant, the
midpoint of the score range. The functions here keep that score and replace only the cut. Each statistic is
computed for every level of the normalised score, and a rule picks one level. Three sources of evidence:

  boundary    the query's own structure: mean feature affinity of adjacent patches across the region boundary
  round_trip  the reference's exclusion: the region is sent back to the reference and compared with its mask
  signature   the reference's boundary type: adjacent query pairs matched to reference inside/outside pairs

`contrast` uses the score field alone and is the same-information control. The zoom helpers re-observe a located
target at its own scale. Nothing here reads query labels.
"""
import math

import torch
import torch.nn.functional as F

LEVELS = tuple(round(0.2 + 0.0125 * k, 4) for k in range(57))  # 0.20 ... 0.90
RULES = dict(contrast=("contrast", "max"), boundary=("boundary", "min"), round_trip=("round_trip", "max"),
             signature=("signature", "max"), boundary_inverted=("boundary", "max"))


def normalise(score):
    s = score - score.min()
    return s / s.max().clamp_min(1e-6)


def binarise(score, t, hw):
    """FoRIS's `_binarize_response` with the threshold exposed; t = 0.5 reproduces it bit for bit."""
    return F.interpolate(normalise(score)[None, None], size=hw, mode="bilinear", align_corners=False)[0, 0] > t


def level_sets(sn, levels=LEVELS):
    return sn[None] > torch.tensor(levels, device=sn.device, dtype=sn.dtype)[:, None, None]


def dilate(m, it=1):
    """4-neighbour dilation of [L, h, w] boolean maps."""
    x = m.float()
    for _ in range(it):
        p = F.pad(x, (1, 1, 1, 1))
        x = torch.stack([p[:, 1:-1, 1:-1], p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]]).amax(0)
    return x > 0


def score_contrast(sn, R, ring=2):
    """Mean score inside the region minus mean score in a ring around it."""
    out = dilate(R, ring) & ~R
    inside = (sn[None] * R).sum((1, 2)) / R.sum((1, 2)).clamp_min(1)
    around = (sn[None] * out).sum((1, 2)) / out.sum((1, 2)).clamp_min(1)
    return torch.where(out.flatten(1).any(1), inside - around, torch.full_like(inside, -math.inf))


def neighbour_affinity(f):
    """Cosine of each patch with its right and lower neighbour; f is [h, w, C], L2-normalised."""
    return (f[:, :-1] * f[:, 1:]).sum(-1), (f[:-1] * f[1:]).sum(-1)


def boundary_affinity(R, aff_r, aff_d):
    """Mean affinity over the adjacent pairs that the region boundary separates."""
    cr, cd = R[:, :, :-1] ^ R[:, :, 1:], R[:, :-1] ^ R[:, 1:]
    n = cr.sum((1, 2)) + cd.sum((1, 2))
    v = ((aff_r[None] * cr).sum((1, 2)) + (aff_d[None] * cd).sum((1, 2))) / n.clamp_min(1)
    return torch.where(n > 0, v, torch.full_like(v, math.inf))


def round_trip(R, back_idx, cov):
    """IoU on the reference between its mask and the label sent back from each level.

    back_idx[a] is the query patch nearest to reference patch a; cov[a] is the mask coverage of patch a.
    """
    b = R.flatten(1)[:, back_idx].float()
    c = cov.flatten()[None]
    return (b * c).sum(1) / (b + c - b * c).sum(1).clamp_min(1e-6)


def _oriented_pairs(h, w, device):
    idx = torch.arange(h * w, device=device).view(h, w)
    r0, r1, d0, d1 = idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1), idx[:-1].reshape(-1), idx[1:].reshape(-1)
    return dict(right=(r0, r1), left=(r1, r0), down=(d0, d1), up=(d1, d0))


def _cap(ix, cap):
    return ix if len(ix) <= cap else ix[torch.arange(cap, device=ix.device) * len(ix) // cap]


def edge_signature(gq, gs, cov, cap=4096, chunk=2048):
    """For every ordered adjacent query pair (i inside, j outside): how much better it is explained by a
    reference pair that crosses the mask boundary than by a reference pair on one side of it.

    gq, gs are [h, w, C] L2-normalised cross-image features, cov the [h, w] reference mask coverage.
    Returns maps keyed right/left ([h, w-1]) and down/up ([h-1, w]), or None when the mask has no boundary.
    """
    h, w = cov.shape
    sim = gq.flatten(0, 1) @ gs.flatten(0, 1).T
    pairs = _oriented_pairs(h, w, cov.device)
    a = torch.cat([p[0] for p in pairs.values()])
    b = torch.cat([p[1] for p in pairs.values()])
    fg = cov.flatten() >= 0.5
    cross, same = _cap(torch.nonzero(fg[a] & ~fg[b]).flatten(), cap), _cap(torch.nonzero(fg[a] == fg[b]).flatten(), cap)
    if len(cross) == 0 or len(same) == 0:
        return None

    def best(i, j, sel):
        out = []
        for k in range(0, len(i), chunk):
            out.append((sim[i[k:k + chunk]][:, a[sel]] + sim[j[k:k + chunk]][:, b[sel]]).amax(1) / 2)
        return torch.cat(out)

    shape = dict(right=(h, w - 1), left=(h, w - 1), down=(h - 1, w), up=(h - 1, w))
    return {k: (best(i, j, cross) - best(i, j, same)).view(shape[k]) for k, (i, j) in pairs.items()}


def signature_score(R, sig):
    """Mean boundary-type evidence over the adjacent pairs that the region boundary separates."""
    lin, rin = R[:, :, :-1] & ~R[:, :, 1:], ~R[:, :, :-1] & R[:, :, 1:]
    tin, bin_ = R[:, :-1] & ~R[:, 1:], ~R[:, :-1] & R[:, 1:]
    n = lin.sum((1, 2)) + rin.sum((1, 2)) + tin.sum((1, 2)) + bin_.sum((1, 2))
    v = ((sig["right"][None] * lin).sum((1, 2)) + (sig["left"][None] * rin).sum((1, 2))
         + (sig["down"][None] * tin).sum((1, 2)) + (sig["up"][None] * bin_).sum((1, 2))) / n.clamp_min(1)
    return torch.where(n > 0, v, torch.full_like(v, -math.inf))


def level_statistics(sn, aff_r, aff_d, back_idx, cov, sig, levels=LEVELS, min_area=4, max_frac=0.9):
    R = level_sets(sn, levels)
    area = R.sum((1, 2))
    return dict(area=area, valid=(area >= min_area) & (area <= max_frac * sn.numel()),
                contrast=score_contrast(sn, R), boundary=boundary_affinity(R, aff_r, aff_d),
                round_trip=round_trip(R, back_idx, cov),
                signature=signature_score(R, sig) if sig is not None else torch.full((len(levels),), -math.inf,
                                                                                    device=sn.device))


def choose(stats, rule, levels=LEVELS):
    """Index of the chosen level, or None when the rule has nothing to decide with (the caller keeps 0.5)."""
    key, mode = RULES[rule]
    v = stats[key].clone().float()
    ok = stats["valid"] & torch.isfinite(v)
    if not ok.any():
        return None
    v[~ok] = -math.inf if mode == "max" else math.inf
    top = v.max() if mode == "max" else v.min()
    tie = torch.nonzero((v - top).abs() <= 1e-6).flatten()
    t = torch.tensor(levels, device=v.device)
    return int(tie[(t[tie] - 0.5).abs().argmin()])


def zoom_box(mask, margin=1.5, max_frac=0.75, min_side=256):
    """Square crop around the mask's bounding box, side = margin x its longer side, kept inside the frame.

    Returns (x0, y0, x1, y1), or None when the mask is empty or the crop would not enlarge the view by 1/max_frac.
    """
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if len(ys) == 0:
        return None
    H, W = mask.shape
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    side = max(min_side, int(math.ceil(margin * max(y1 - y0, x1 - x0))))
    if side >= max_frac * min(H, W):
        return None
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx = int(min(max(round(cx - side / 2), 0), W - side))
    by = int(min(max(round(cy - side / 2), 0), H - side))
    return bx, by, bx + side, by + side


def to_original(box, frame_hw, orig_wh):
    """A box in the (anisotropically resized) model frame as an integer box in the original image, plus the
    model-frame box those integer coordinates correspond to."""
    (H, W), (ow, oh) = frame_hw, orig_wh
    x0, y0, x1, y1 = box
    ox0, oy0 = int(math.floor(x0 * ow / W)), int(math.floor(y0 * oh / H))
    ox1, oy1 = min(ow, max(ox0 + 2, int(math.ceil(x1 * ow / W)))), min(oh, max(oy0 + 2, int(math.ceil(y1 * oh / H))))
    back = (int(round(ox0 * W / ow)), int(round(oy0 * H / oh)), int(round(ox1 * W / ow)), int(round(oy1 * H / oh)))
    return (ox0, oy0, ox1, oy1), back


def paste(pred, box, frame_hw):
    """Place a crop-frame prediction back into an empty model frame."""
    x0, y0, x1, y1 = box
    out = torch.zeros(frame_hw, dtype=torch.bool, device=pred.device)
    out[y0:y1, x0:x1] = F.interpolate(pred[None, None].float(), size=(y1 - y0, x1 - x0), mode="bilinear",
                                      align_corners=False)[0, 0] > 0.5
    return out
