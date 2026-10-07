"""One fixed post-RCG candidate: joint 3x3 appearance/mask-template transfer.

New quantity: labels of the entire matched reference neighborhood, reconciled
over overlapping query neighborhoods. Matching reads the joint arrangement of
nine DINO vectors, not only a center vector or its nearest-reference label.
No graph solve, threshold estimation, reference-only fitting, or reverse match.
"""
import torch


OFFSETS = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)]
ORIENTATIONS = [(mirror, turns) for mirror in (False, True) for turns in range(4)]


def rotate(y, x, mirror, turns):
    if mirror:
        x = -x
    for _ in range(turns):
        y, x = x, -y
    return y, x


def shifted_indices(side, dy, dx, device):
    y, x = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing='ij')
    return ((y + dy).clamp(0, side-1)*side + (x + dx).clamp(0, side-1)).flatten()


@torch.inference_mode()
def predict(q, r, cov, base, side=64):
    """Return candidate plus two fixed mechanism controls; all masks are B subsets."""
    n = side*side
    assert q.shape[0] == r.shape[0] == cov.numel() == base.numel() == n
    device=q.device
    sim=q.float() @ r.float().T
    qi=[shifted_indices(side, y, x, device) for y, x in OFFSETS]
    # Replicated borders are declared; identical for candidate and controls.
    best_values=torch.full((n,4),-float('inf'),device=device)
    best_ids=torch.zeros((n,4),dtype=torch.long,device=device)
    for oi,(mirror,turns) in enumerate(ORIENTATIONS):
        scores=torch.zeros_like(sim)
        for oq,(dy,dx) in enumerate(OFFSETS):
            ry,rx=rotate(dy,dx,mirror,turns)
            ri=shifted_indices(side,ry,rx,device)
            scores += sim[qi[oq][:,None],ri[None,:]]/9
        vals,ids=scores.topk(4,dim=1)
        vals=torch.cat((best_values,vals),dim=1)
        ids=torch.cat((best_ids,ids+oi*n),dim=1)
        best_values,which=vals.topk(4,dim=1)
        best_ids=torch.gather(ids,1,which)
    ri=best_ids%n
    ori=best_ids//n
    positive=cov.flatten()>=.5
    if not positive.any():
        positive=cov.flatten()>=cov.max().clamp_min(1e-6)
    labels=positive.float()
    votes=torch.zeros(n,device=device)
    shuffled=torch.zeros_like(votes)
    counts=torch.zeros_like(votes)
    # Center-only uses exactly the same matched templates and four votes.
    center=labels[ri].mean(1)
    noncenter=[i for i,o in enumerate(OFFSETS) if o != (0,0)]
    permutation={i:noncenter[(j+1)%len(noncenter)] for j,i in enumerate(noncenter)}
    permutation[4]=4
    for oq,(dy,dx) in enumerate(OFFSETS):
        target=qi[oq]
        local=torch.zeros((n,4),device=device)
        permuted=torch.zeros_like(local)
        for oi,(mirror,turns) in enumerate(ORIENTATIONS):
            ry,rx=rotate(dy,dx,mirror,turns)
            iy=ri//side;ix=ri%side
            src=((iy+ry).clamp(0,side-1)*side+(ix+rx).clamp(0,side-1))
            py,px=rotate(*OFFSETS[permutation[oq]],mirror,turns)
            src_perm=((iy+py).clamp(0,side-1)*side+(ix+px).clamp(0,side-1))
            choose=ori==oi
            local += labels[src]*choose
            permuted += labels[src_perm]*choose
        votes.scatter_add_(0,target,local.mean(1))
        shuffled.scatter_add_(0,target,permuted.mean(1))
        counts.scatter_add_(0,target,torch.ones(n,device=device))
    probability=votes/counts.clamp_min(1)
    shuffled=shuffled/counts.clamp_min(1)
    b=base.flatten().bool()
    return {'joint_mask':b & (probability>=.5),
            'center_control':b & (center>=.5),
            'permuted_control':b & (shuffled>=.5)}
