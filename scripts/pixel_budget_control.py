"""Exact expected-I/U pixel control inside available add/delete domains.

Probabilities are constant on16x16token cells. Sorting8192weighted action groups
solves each cardinality-constrained fractional-programming inner problem exactly;
Dinkelbach updates certify the unrestricted pixel alternative to structured operators.
"""


def optimize(probability, origin, addition_domain, deletion_domain, budget, iterations=64):
    import torch
    p=probability.reshape(64,16,64,16).mean((1,3)).reshape(-1).double()
    def cells(mask):
        return mask.reshape(64,16,64,16).permute(0,2,1,3).reshape(4096,256)
    adds,deletes=cells(addition_domain),cells(deletion_domain)
    availability=torch.cat([adds.sum(1),deletes.sum(1)])
    initial_inter=(cells(origin).sum(1)*p).sum()
    initial_union=p.sum()*256+origin.sum()-initial_inter
    ratio=initial_inter/initial_union.clamp_min(1e-12)
    chosen=torch.zeros_like(availability)
    residual=0.
    for step in range(iterations):
        gain=p-ratio*(1-p)
        values=torch.cat([gain,-gain])
        order=torch.argsort(values,descending=True,stable=True)
        capacity=availability[order]
        capacity=torch.where(values[order]>0,capacity,0)
        remaining=budget-capacity.cumsum(0)+capacity
        selected=torch.minimum(capacity,remaining.clamp_min(0))
        chosen=torch.zeros_like(availability);chosen[order]=selected
        ca,cd=chosen[:4096],chosen[4096:]
        intersection=initial_inter+((ca-cd)*p).sum()
        union=initial_union+((ca-cd)*(1-p)).sum()
        residual=float(intersection-ratio*union)
        updated=intersection/union.clamp_min(1e-12)
        if residual <= 1e-8:
            ratio=updated;break
        if updated+1e-12 < ratio:raise ValueError('Fractional objective decreased')
        ratio=updated
    else:raise ValueError('Pixel-control certificate did not converge')
    selected_add=adds & (adds.cumsum(1)<=chosen[:4096,None])
    selected_delete=deletes & (deletes.cumsum(1)<=chosen[4096:,None])
    def pixels(cell):
        return cell.reshape(64,64,16,16).permute(0,2,1,3).reshape(-1)
    output=(origin|pixels(selected_add))&~pixels(selected_delete)
    if int((output^origin).sum())>budget:raise ValueError('Pixel budget exceeded')
    return output,dict(expected_iou=float(ratio),residual=residual,iterations=step+1,changed=int(chosen.sum()),
                       certificate='exact expected-count ratio inside unrestricted available pixel/budget family; not true-IoU dominance')
