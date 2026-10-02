"""INSID3 on cached features: any number of (possibly pseudo-labelled) references, no encoder calls.

A ClassSet holds, for one concept, image 0 = the labelled reference and images 1..N = unlabeled images of the concept.
predict() repeats models/insid3.py::predict_mask (prototype averaged over references, forward test, backward majority vote,
seed, aggregation) from the cached features and cluster labels.
"""
import math, numpy as np, torch, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def cluster_protos(X, lab, K):
    oh = F.one_hot(lab, K).float(); return F.normalize(oh.T @ X, dim=1), oh.sum(0)


class ClassSet:
    def __init__(self, path, U=None, tau=0.6, merge=0.2):
        d = torch.load(path, weights_only=False) if isinstance(path, str) else path; self.c = d["c"]; self.names = d["names"]
        self.lab = d["lab"].to(DEV).long(); self.K = [int(l.max()) + 1 for l in self.lab]
        self.Pd, self.Po, self.area = [], [], []
        if "fq" in d:        # compact format: debiased features in an orthonormal basis of the non-positional subspace + stored cluster prototypes
            self.fd = F.normalize(d["fq"].to(DEV).float(), dim=-1); self.n, self.P, self.C = self.fd.shape
            for i in range(self.n):
                a, ar = cluster_protos(self.fd[i], self.lab[i], self.K[i]); self.Pd.append(a); self.area.append(ar); self.Po.append(d["Po"][i].to(DEV).float())
        else:
            self.f = d["fn"].to(DEV).float(); self.n, self.P, self.C = self.f.shape
            self.fd = F.normalize(self.f - (self.f @ U) @ U.T, dim=-1)
            for i in range(self.n):
                a, ar = cluster_protos(self.fd[i], self.lab[i], self.K[i]); b, _ = cluster_protos(self.f[i], self.lab[i], self.K[i])
                self.Pd.append(a); self.Po.append(b); self.area.append(ar)
        self.h = int(math.sqrt(self.P)); self.n_real = d.get("n_real", self.n)   # images n_real.. are distractors (do not contain the class)
        self.gt64 = d["gt64"].to(DEV).reshape(self.n, -1)                      # (n, P) bool, as INSID3 downsamples reference masks
        self.gt = torch.from_numpy(np.unpackbits(d["gt_bits"], axis=1)).to(DEV).bool().reshape(self.n, d["S"], d["S"]); self.S = d["S"]
        self.merge = merge; self._nn = {}

    def nn(self, t, j):
        """index of the nearest patch of image j for every patch of image t (debiased features)"""
        return self.nnv(t, j)[1]

    def nnv(self, t, j):
        """(similarity, index) of the nearest patch of image j for every patch of image t (debiased features)"""
        if (t, j) not in self._nn: self._nn[(t, j)] = (self.fd[t] @ self.fd[j].T).max(1)
        return self._nn[(t, j)]

    def up(self, m):
        return F.interpolate(m.reshape(1, 1, self.h, self.h).float(), size=(self.S, self.S), mode="bilinear", align_corners=False)[0, 0] > 0.5

    def iu(self, m, t):
        p = self.up(m); g = self.gt[t]; return float((p & g).sum()), float((p | g).sum())

    def iou64(self, m, t):
        g = self.gt64[t]; return float((m & g).sum()) / max(float((m | g).sum()), 1.0)

    @torch.no_grad()
    def predict(self, t, refs, masks, weights=None, need=None, veto=None, return_parts=False, backward="majority", k=1, bonus0=0.0, delta=0.0, margin=0.0, known=None, vote_weights=None):
        """INSID3 prediction for image t from references `refs` (image indices) with patch masks `masks` (list of (P,) bool).
        weights: optional per-reference prototype weights; also used by the legacy majority branch, not pooled voting.
        vote_weights: opt-in pooled-only 1D vector, one finite nonnegative value per original reference.
        Empty-mask references are filtered together with their weights; zero selected vote mass uses unweighted voting.
        need:    threshold override for the legacy majority backward test only.
        veto:    optional (P,) bool; target patches marked True can never be candidates."""
        vote_w = None
        if vote_weights is not None:
            if backward != "pooled": raise ValueError("vote_weights requires backward='pooled'")
            if len(refs) != len(masks): raise ValueError("refs and masks must have the same length when vote_weights is set")
            vote_w = torch.as_tensor(vote_weights, device=DEV, dtype=torch.float64)
            if vote_w.ndim != 1 or vote_w.numel() != len(refs):
                raise ValueError("vote_weights must be a 1D vector with one value per original reference")
            if not torch.isfinite(vote_w).all() or (vote_w < 0).any():
                raise ValueError("vote_weights must contain only finite nonnegative values")
        keep = [i for i, m in enumerate(masks) if m.any()]
        if not keep: return torch.zeros(self.P, dtype=torch.bool, device=DEV)
        refs = [refs[i] for i in keep]; masks = [masks[i] for i in keep]
        if vote_w is not None: vote_w = vote_w[keep]
        if known is not None: known = [known[i] for i in keep]
        w = torch.ones(len(refs), device=DEV) if weights is None else torch.tensor([weights[i] for i in keep], device=DEV, dtype=torch.float)
        proto = F.normalize(sum(wi * self.fd[j][m].mean(0) for wi, j, m in zip(w, refs, masks)) / w.sum(), dim=0)
        sim_fwd = self.fd[t] @ proto; fwd = sim_fwd > 0
        if fwd.sum() == 0: fwd = sim_fwd > float(torch.quantile(sim_fwd, 0.9))
        if backward == "majority":            # INSID3: one vote per reference image (is the nearest patch in that image foreground?)
            votes = sum(wi * m[self.nn(t, j)].float() for wi, j, m in zip(w, refs, masks))
            thr = math.ceil(len(refs) / 2) * float(w.sum()) / len(refs) if need is None else need
            back = votes >= thr - 1e-6
        elif backward == "pu":                # positive-unlabeled: pseudo masks are trusted as foreground, their complement only weakly as background
            sfg = torch.full((self.P,), -1.0, device=DEV); sbg = torch.full((self.P,), -1.0, device=DEV)
            for j, m in zip(refs, masks):
                S = self.fd[t] @ self.fd[j].T
                sfg = torch.maximum(sfg, S[:, m].max(1).values)
                if (~m).any(): sbg = torch.maximum(sbg, S[:, ~m].max(1).values - (0.0 if j == 0 else delta))
            votes = sfg - sbg; back = votes > margin
        elif backward == "tri":               # pooled over patches with a trusted label only (known[i]: (P,) bool per reference)
            Vs, Ls = [], []
            for j, m, kn in zip(refs, masks, known):
                S = self.fd[t] @ self.fd[j].T
                if not kn.all(): S = S.masked_fill(~kn[None], -2.0)
                v, ix = S.max(1); Vs.append(v); Ls.append(m[ix].float())
            V = torch.stack(Vs); Lb = torch.stack(Ls); kk = min(k, len(refs)); top = V.topk(kk, dim=0).indices
            votes = Lb.gather(0, top).mean(0); back = votes > 0.5 if kk > 1 else votes > 0
        else:                                 # pooled: the k references whose nearest patch is most similar decide (k=1: nearest patch over all references)
            V = torch.stack([self.nnv(t, j)[0] for j in refs]); Lb = torch.stack([m[self.nn(t, j)] for j, m in zip(refs, masks)]).float()   # (R, P)
            if bonus0 and refs[0] == 0: V = V.clone(); V[0] += bonus0
            kk = min(k, len(refs)); top = V.topk(kk, dim=0).indices
            if vote_w is None:
                votes = Lb.gather(0, top).mean(0); back = votes > 0.5 if kk > 1 else votes > 0
            else:
                selected_labels = Lb.gather(0, top)
                selected_w = vote_w[:, None].expand_as(Lb).gather(0, top)
                # Positive rescaling preserves the weighted mean and avoids overflow in its sum.
                scale = selected_w.max(0).values
                selected_w = selected_w / torch.where(scale > 0, scale, torch.ones_like(scale))
                total = selected_w.sum(0)
                weighted = (selected_w * selected_labels).sum(0) / torch.where(total > 0, total, torch.ones_like(total))
                votes = torch.where(total > 0, weighted, selected_labels.mean(0))
                back = votes > 0.5 if kk > 1 else votes > 0
        cand = fwd & back
        if veto is not None: cand = cand & ~veto
        if cand.sum() == 0: return cand
        lab, K, Pd, Po, area = self.lab[t], self.K[t], self.Pd[t], self.Po[t], self.area[t]
        cnt = torch.bincount(lab[cand], minlength=K).float(); aw = cnt / area; matched = cnt > 0
        cs = Pd @ proto; seed = int(torch.where(matched, cs, torch.full_like(cs, -9)).argmax())
        intra = Po @ Po[seed]; cross = (F.one_hot(lab, K).float().T @ sim_fwd) / area
        aw[seed] = 1.0; combined = cross * intra * aw
        pred = (combined > self.merge)[lab]
        if return_parts: return pred, dict(cand=cand, seed=seed, combined=combined, sim_fwd=sim_fwd, votes=votes)
        return pred

    @torch.no_grad()
    def transfer(self, t, refs, labels, k=3, T=0.05, w0=1.0):
        """Soft label transfer to image t: every patch averages the labels of its nearest patch in each of the k references
        whose nearest patch is most similar (weights softmax(similarity / T); the labelled reference 0 gets weight * w0).
        labels: list of (P,) float tensors in [-1, 1] (+1 foreground, -1 background). Returns (P,) scores in [-1, 1]."""
        V = torch.stack([self.nnv(t, j)[0] for j in refs]); L = torch.stack([l[self.nn(t, j)] for j, l in zip(refs, labels)])   # (R, P)
        kk = min(k, len(refs)); top = V.topk(kk, dim=0)
        w = torch.softmax(top.values / T, dim=0)
        if w0 != 1.0 and refs[0] == 0:
            w = w * torch.where(top.indices == 0, w0, 1.0); w = w / w.sum(0, keepdim=True)
        return (w * L.gather(0, top.indices)).sum(0)

    def cluster_mean(self, t, score):
        """average a per-patch score inside INSID3's clusters of image t"""
        lab, K = self.lab[t], self.K[t]
        return ((F.one_hot(lab, K).float().T @ score) / self.area[t])[lab]
