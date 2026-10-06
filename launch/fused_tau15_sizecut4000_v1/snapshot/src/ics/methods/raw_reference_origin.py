"""Two explicit readouts of raw frozen DINO tokens, before FoRIS processing.

The primary is cosine nearest-reference-token hard-label transfer. The secondary
is cosine foreground-prototype minus background-prototype, with threshold zero.
Neither uses the FoRIS mask/score, a positional projection, clustering or CRF.
"""
import numpy as np
import torch
import torch.nn.functional as F

CONFIG = dict(primary="model.raw_nn", secondary="model.raw_mean", reference_label="coverage>=0.5",
              token_features="encoder final norm, before positional projection", cosine_l2_normalization=True,
              nearest_ties="first reference token in row-major order", prototype="unit(mean(unit(reference tokens)))",
              secondary_threshold=0.0, renderer="hard token mask -> bilinear1024 align_corners=False -> >0.5",
              model_forwards=0, query_gt_in_inference=False)


@torch.inference_mode()
def predict(q, r, cov, *, device="cpu", chunk=512):
    q = torch.as_tensor(q, device=device, dtype=torch.float32)
    r = torch.as_tensor(r, device=device, dtype=torch.float32)
    c = torch.as_tensor(cov, device=device, dtype=torch.float32)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1] or c.ndim != 2 or len(q) != c.numel() or len(r) != c.numel():
        raise ValueError("Expected aligned query/reference token grids and reference coverage")
    if chunk < 1 or not all(bool(torch.isfinite(x).all()) for x in (q, r, c)):
        raise ValueError("Nonfinite inputs or invalid chunk size")
    if bool(((c < 0) | (c > 1)).any()) or bool((q.norm(dim=1) == 0).any()) or bool((r.norm(dim=1) == 0).any()):
        raise ValueError("Invalid coverage or zero-norm raw tokens")
    fg = c.flatten() >= .5
    if not bool(fg.any()) or bool(fg.all()):
        raise ValueError("Both reference foreground and background are required")
    q, r = F.normalize(q, dim=1), F.normalize(r, dim=1)
    labels, margins, ties = [], [], 0
    for start in range(0, len(q), chunk):
        sim = q[start:start + chunk] @ r.T
        labels.append(fg[sim.argmax(1)])
        margin = sim[:, fg].max(1).values - sim[:, ~fg].max(1).values
        margins.append(margin); ties += int((margin == 0).sum())
    nn, margin = torch.cat(labels), torch.cat(margins)
    prototype_margin = q @ (F.normalize(r[fg].mean(0), dim=0) - F.normalize(r[~fg].mean(0), dim=0))
    token_masks = {"model.raw_nn": nn, "model.raw_mean": prototype_margin > 0}
    fields = {"model.raw_nn": margin, "model.raw_mean": prototype_margin}
    # Fields are signed ranking margins, not replacements for hard NN labels at ties.
    masks = {k: v.reshape(c.shape).cpu().numpy() for k, v in token_masks.items()}
    values = {k: v.reshape(c.shape).cpu().numpy().astype(np.float32) for k, v in fields.items()}
    return masks, values, dict(reference_fg_tokens=int(fg.sum()), cross_role_cosine_ties=ties,
                              field_semantics="signed cosine margins; hard masks are authoritative", config=CONFIG)
