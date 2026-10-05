"""Boundary of a coarse mask from finer tokens of the same image (shared by the CPU reading and the GPU pipeline).
A band token's mixing share alpha is its position between the means of its nearest sure-object and sure-surround tokens in the
image's own feature space; core = 1, far = 0, band = alpha; the soft field is cut at 0.5."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree

G = 128  # fine grid; sample k is centred at pixel 8k + off of the 1024 view (off 0: stride-8 grid, off 4: the 2048 view)
_P = np.stack(np.mgrid[0:G, 0:G], 0).reshape(2, -1).T.astype(np.float32)


def to_px(v, off, size=1024):
    c = (torch.arange(size) + .5 - off) / 8
    g = (c / (G - 1) * 2 - 1).clamp(-1, 1)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    return F.grid_sample(torch.as_tensor(v, dtype=torch.float32).view(1, 1, G, G), torch.stack([gx, gy], -1)[None], mode="bilinear", align_corners=True)[0, 0]


def from64(s, off):
    """FoRIS's 64 x 64 field (sample i centred at 16i + 8) read at the fine grid's centres."""
    c = (8 * torch.arange(G) + off - 8) / 16
    g = (c / 63 * 2 - 1).clamp(-1, 1)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    return F.grid_sample(s.float().cpu().view(1, 1, 64, 64), torch.stack([gx, gy], -1)[None], mode="bilinear", align_corners=True)[0, 0]


def alpha(q, core, far, unk, k=8):
    """q: [G*G, C] unit tokens (CPU). Returns the soft G x G field, or None when a side has no sure token."""
    out = core.astype(np.float32).ravel().copy()
    ui = np.nonzero(unk.ravel())[0]
    if len(ui) == 0 or core.sum() == 0 or far.sum() == 0:
        return None
    means = []
    for src in (core, far):
        si = np.nonzero(src.ravel())[0]
        kk = min(k, len(si))
        _, nn = cKDTree(_P[si]).query(_P[ui], k=kk)
        means.append(q[torch.from_numpy(si[nn.reshape(len(ui), kk)])].mean(1))
    g, b = means
    f = q[torch.from_numpy(ui)]
    out[ui] = (((f - b) * (g - b)).sum(-1) / ((g - b) ** 2).sum(-1).clamp_min(1e-6)).clamp(0, 1).numpy()
    return out.reshape(G, G)


def trimap(m, band):
    st = np.ones((3, 3), bool)
    core = ndimage.binary_erosion(m, st, iterations=band)
    far = ~ndimage.binary_dilation(m, st, iterations=band)
    lab, n = ndimage.label(m, st)
    if n:  # a component with no core keeps its deepest half
        din = ndimage.distance_transform_cdt(m, metric="chessboard")
        mx = np.asarray(ndimage.maximum(din, lab, range(1, n + 1)))
        for c in np.nonzero(mx <= band)[0] + 1:
            core |= (lab == c) & (din >= max(1, .5 * mx[c - 1]))
    return core, far, ~core & ~far


def refine(q, score64, off, band, mix):
    """q: [G*G, C] unit tokens (CPU float); score64: FoRIS's score on its 64 x 64 grid, already in 0..1.
    Returns the refined mask at 1024 x 1024 (bool)."""
    sc = from64(score64, off)
    m = (sc > .5).numpy()
    core, far, unk = trimap(m, band)
    a = alpha(q, core, far, unk)
    if a is None:
        a = m.astype(np.float32)
    if mix:
        a = np.where(unk, (a + sc.numpy()) / 2, a)
    return to_px(a, off) > .5


def fine_tokens(encode, deb, q, grid):
    """encode: [B,3,H,W] -> [B,C,h,w] unit tokens; deb: debias on [1,1,C,h,w] or identity; q: [1,3,1024,1024].
    grid 's8': four encodings shifted by half a patch, interleaved; 'z2': the view enlarged twice. Returns [G*G, C] on q's device."""
    if grid == "z2":
        x = deb(encode(F.interpolate(q, scale_factor=2, mode="bicubic", align_corners=False))[None])[0, 0]
        return x.flatten(-2).transpose(-1, -2)
    first = encode(q)
    h, w = first.shape[-2:]
    sh = q.shape[-1] // w // 2
    out = torch.zeros(first.shape[1], 2 * h, 2 * w, device=q.device)
    for dy in (0, 1):
        for dx in (0, 1):
            x = first if not (dy or dx) else encode(F.pad(q, (sh * dx, 0, sh * dy, 0), mode="replicate")[..., :q.shape[-2], :q.shape[-1]])
            out[:, (1 - dy)::2, (1 - dx)::2] = deb(x[None])[0, 0]
    return out.flatten(-2).transpose(-1, -2)
