"""INSID3's pipeline, step by step, with every intermediate quantity returned (same operations as models/insid3.py)."""
import torch, torch.nn.functional as F
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import downsample_mask
from utils.refinement import upsample_mask


def protos(X, lab, K):
    """Mean feature per cluster, L2-normalised. X (N, C), lab (N,)."""
    oh = F.one_hot(lab, K).float(); return F.normalize(oh.T @ X.float(), dim=1), oh.sum(0)


@torch.no_grad()
def dissect(model, ref_img, ref_mask, tgt_img, labels=None):
    """ref_img/tgt_img (1,3,S,S) normalised; ref_mask (1,S,S) bool. One shot. Returns a dict of intermediates at feature resolution."""
    imgs = torch.cat([ref_img, tgt_img], 0).unsqueeze(0)
    f = F.normalize(model._extract_features(imgs).float(), p=2, dim=2); _, _, C, h, w = f.shape
    fd = model._debias_features(f); m = downsample_mask(ref_mask.unsqueeze(1), h, w)                    # (h, w) bool
    R, T, Rd, Td = (x.reshape(C, -1).T for x in (f[0, 0], f[0, 1], fd[0, 0], fd[0, 1]))                # (N, C) each
    mf = m.reshape(-1); proto = F.normalize(Rd[mf].mean(0), dim=0)                                      # reference prototype (debiased)
    sim_fwd = (Td @ proto).reshape(h, w)                                                                # target patch vs prototype
    S = Td @ Rd.T                                                                                       # (Nt, Nr) debiased cross-image similarity
    nn = S.argmax(1); back = mf[nn].reshape(h, w)                                                       # nearest reference patch is foreground
    fwd = sim_fwd > 0
    if fwd.sum() == 0: fwd = sim_fwd > float(torch.quantile(sim_fwd, 0.9))
    cand = fwd & back
    if labels is None: labels = agglomerative_clustering(T, model.tau)
    lab = labels.reshape(h, w); K = int(lab.max()) + 1
    Pd, area = protos(Td, labels, K); Po, _ = protos(T, labels, K)
    out = dict(h=h, w=w, K=K, lab=lab, cand=cand, back=back, sim_fwd=sim_fwd, proto=proto, Pd=Pd, Po=Po, area=area, m=m, R=R, T=T, Rd=Rd, Td=Td, S=S)
    # seed and aggregation
    cnt = torch.bincount(labels[cand.reshape(-1)], minlength=K).float(); aw = cnt / area; matched = cnt > 0
    if cand.sum() == 0:
        out.update(seed=-1, combined=torch.zeros(K, device=T.device), pred=cand, cross=torch.zeros(K, device=T.device), intra=torch.zeros(K, device=T.device), aw=aw); return out
    cs = Pd @ proto; seed = int(torch.where(matched, cs, torch.full_like(cs, -9)).argmax())
    intra = Po @ Po[seed]; cross = (F.one_hot(labels, K).float().T @ sim_fwd.reshape(-1)) / area
    aw2 = aw.clone(); aw2[seed] = 1.0; combined = cross * intra * aw2
    out.update(seed=seed, combined=combined, pred=(combined > model.merge_threshold)[lab], cross=cross, intra=intra, aw=aw)
    return out


def up(mask, size): return upsample_mask(mask, size, size)
