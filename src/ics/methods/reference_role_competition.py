"""Compete against the actual best foreground and background reference roles.

Correction to the first spatial-pair hypothesis: a query token must outrank its
strongest reference BG competitor, not merely the BG next to its selected FG.
No labels, class names, thresholds fitted to scores or model calls occur here.
"""
import torch
from .anchored_task_readout import RoleBank, _fp32_cpu

@torch.inference_mode()
def competitive_role_field(q_unit, bank: RoleBank):
    if not bank.valid:
        raise ValueError('Invalid reference bank: '+str(bank.reason))
    q=_fp32_cpu(q_unit,'unit query O24')
    if q.ndim!=2 or q.shape[1]!=bank.foreground.shape[1]:
        raise ValueError('Role dimensions differ')
    if q.shape[0] and (q.norm(dim=1)-1).abs().max()>1e-3:
        raise ValueError('Require normalized actual query tokens')
    fg,fg_ids=(q@bank.foreground.T).max(1)
    bg,bg_ids=(q@bank.background.T).max(1)
    h=torch.tanh((fg-bg)/.07)
    return h,dict(fg=fg,bg=bg,fg_ids=fg_ids,bg_ids=bg_ids)
