"""INSID3 on cached features for a small set of images of one concept.

Image 0 is the labelled reference; images 1.. are unlabeled (the episode's query first, then the pool, then distractors).
predict() repeats models/insid3.py::predict_mask from cached features and cluster labels (checked against the released
code by scripts/cache_episodes.py), for any number of references with true or predicted masks, and adds one option:

  backward="majority"  INSID3: one vote per reference image (is the nearest patch in that image foreground?), half needed
  backward="pooled"    the k references whose nearest patch is most similar vote (optionally weighted by `weights`);
                       with `known`, only patches with a trusted label are searched in each reference
"""
import math
import numpy as np, torch, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def cluster_protos(X, lab, K):
    oh = F.one_hot(lab, K).float(); return F.normalize(oh.T @ X, dim=1), oh.sum(0)


class ImageSet:
    def __init__(self, d, merge=0.2):
        """d: dict with fq (n, P, C') float16 debiased features (coordinates in the non-positional subspace), lab (n, P) cluster
        labels, Po list of (K_i, C) cluster prototypes of the original features, gt64 (n, h, h) bool, gt_bits packed (n, S*S/8), S."""
        self.c = d["c"]; self.names = d["names"]
        self.fd = F.normalize(d["fq"].to(DEV).float(), dim=-1); self.n, self.P, self.C = self.fd.shape; self.h = int(math.sqrt(self.P))
        self.lab = d["lab"].to(DEV).long(); self.K = [int(l.max()) + 1 for l in self.lab]; self.Pd, self.Po, self.area = [], [], []
        for i in range(self.n):
            a, ar = cluster_protos(self.fd[i], self.lab[i], self.K[i]); self.Pd.append(a); self.area.append(ar); self.Po.append(d["Po"][i].to(DEV).float())
        self.gt64 = d["gt64"].to(DEV).reshape(self.n, -1); self.S = d["S"]
        self.gt = torch.from_numpy(np.unpackbits(d["gt_bits"], axis=1)).to(DEV).bool().reshape(self.n, self.S, self.S)
        self.merge = merge; self._nn = {}

    def nnv(self, t, j):
        """(similarity, index) of the nearest patch of image j for every patch of image t"""
        if (t, j) not in self._nn: self._nn[(t, j)] = (self.fd[t] @ self.fd[j].T).max(1)
        return self._nn[(t, j)]

    def up(self, m):
        return F.interpolate(m.reshape(1, 1, self.h, self.h).float(), size=(self.S, self.S), mode="bilinear", align_corners=False)[0, 0] > 0.5

    def iu(self, m, t):
        """intersection and union with the true mask of image t at full resolution (the benchmark's scoring)"""
        p = self.up(m); g = self.gt[t]; return float((p & g).sum()), float((p | g).sum())

    @staticmethod
    def iou(p, q):
        return float((p & q).sum()) / max(float((p | q).sum()), 1.0)

    @torch.no_grad()
    def predict(self, t, refs, masks, backward="majority", k=5, weights=None, known=None):
        """Mask (P,) bool of image t given reference images `refs` with patch masks `masks` (list of (P,) bool)."""
        keep = [i for i, m in enumerate(masks) if m.any()]
        if not keep: return torch.zeros(self.P, dtype=torch.bool, device=DEV)
        refs = [refs[i] for i in keep]; masks = [masks[i] for i in keep]
        if known is not None: known = [known[i] for i in keep]
        w = torch.ones(len(refs), device=DEV) if weights is None else torch.tensor([float(weights[i]) for i in keep], device=DEV).clamp_min(1e-3)
        proto = F.normalize(sum(wi * self.fd[j][m].mean(0) for wi, j, m in zip(w, refs, masks)) / w.sum(), dim=0)
        sim_fwd = self.fd[t] @ proto; fwd = sim_fwd > 0
        if fwd.sum() == 0: fwd = sim_fwd > float(torch.quantile(sim_fwd, 0.9))
        if backward == "majority":
            votes = sum(m[self.nnv(t, j)[1]].float() for j, m in zip(refs, masks)); back = votes >= math.ceil(len(refs) / 2)
        else:
            Vs, Ls = [], []
            for i, (j, m) in enumerate(zip(refs, masks)):
                if known is None or known[i].all(): v, ix = self.nnv(t, j)
                else: v, ix = (self.fd[t] @ self.fd[j].T).masked_fill(~known[i][None], -2.0).max(1)
                Vs.append(v); Ls.append(m[ix].float())
            V = torch.stack(Vs); L = torch.stack(Ls); kk = min(k, len(refs)); top = V.topk(kk, dim=0).indices      # (kk, P)
            W = w[:, None].expand_as(V).gather(0, top); back = (L.gather(0, top) * W).sum(0) / W.sum(0) > (0.5 if kk > 1 else 0.0)
        cand = fwd & back
        if cand.sum() == 0: return cand
        lab, K, Pd, Po, area = self.lab[t], self.K[t], self.Pd[t], self.Po[t], self.area[t]
        cnt = torch.bincount(lab[cand], minlength=K).float(); aw = cnt / area
        cs = Pd @ proto; seed = int(torch.where(cnt > 0, cs, torch.full_like(cs, -9)).argmax())
        aw[seed] = 1.0; combined = ((F.one_hot(lab, K).float().T @ sim_fwd) / area) * (Po @ Po[seed]) * aw
        return (combined > self.merge)[lab]
