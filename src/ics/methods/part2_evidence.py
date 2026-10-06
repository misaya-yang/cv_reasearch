"""One Part2 background intervention; every remaining stage is source FoRIS.

The input is the retained post-Part1 cache, including its FP16 rounding. No
encoder, positional projection, query annotation or custom finalizer is used.
"""
from __future__ import annotations

import hashlib
import inspect
import time

import torch
import torch.nn.functional as F

ARMS = ("native_cache", "bg_prototype_lse", "bg_prototype_max")
CONFIG = dict(hard_bg_fraction=0.2, cluster_tau=0.6, temperature=0.07,
              bg_weight=0.55, orthogonal_zero_threshold=1e-8,
              query_labels_in_inference=False, encoder_forwards=0)


def orthogonal_prototypes(prototypes, mu_fg, native_mu_bg):
    """Drop degenerate residuals and fall back to the native orthogonal mean."""
    orth = prototypes - (prototypes @ mu_fg)[:, None] * mu_fg[None]
    norms = orth.norm(dim=1)
    keep = norms > CONFIG["orthogonal_zero_threshold"]
    if bool(keep.any()):
        return orth[keep] / norms[keep, None], int((~keep).sum()), False
    orth_mean = native_mu_bg - (native_mu_bg * mu_fg).sum() * mu_fg
    return (orth_mean / orth_mean.norm().clamp_min(1e-8))[None], len(prototypes), True


def background_response(target, prototypes, aggregation, temperature=0.07):
    sim = torch.einsum("bchw,kc->bkhw", target, prototypes)
    if aggregation == "lse":
        return (temperature * torch.logsumexp(sim / temperature, dim=1)).squeeze(0)
    if aggregation == "max":
        return sim.max(dim=1).values.squeeze(0)
    raise ValueError(f"Unknown background aggregation: {aggregation}")


def intervention_class(native_class, native_module):
    """Subclass the inspected stage-2 signature; capture native FG unchanged."""
    expected = ("self", "ref_feats", "tgt_feat", "ref_masks", "n_refs", "h", "w", "tgt_feat_raw")
    if tuple(inspect.signature(native_class._part2_stage2_contrastive_score).parameters) != expected:
        raise RuntimeError("Native Part2 stage2 signature changed; inspect source before replay")

    class Part2EvidenceFoRIS(native_class):
        background_aggregation = "native"
        _capture_part2 = False

        def _reference_contrastive_prototypes(self, *args, **kwargs):
            stats = super()._reference_contrastive_prototypes(*args, **kwargs)
            if self._capture_part2:
                self._native_part2_stats = stats
            return stats

        def _part2_stage2_contrastive_score(self, ref_feats, tgt_feat, ref_masks,
                                          n_refs, h, w, tgt_feat_raw):
            self._capture_part2 = True
            try:
                original = super()._part2_stage2_contrastive_score(
                    ref_feats, tgt_feat, ref_masks, n_refs, h, w, tgt_feat_raw)
            finally:
                self._capture_part2 = False
            if original is None or self.background_aggregation == "native":
                self.part2_receipt = dict(aggregation="native")
                return original
            _, sf, _, mu_fg = original
            _, mu_bg, fg_protos = self._native_part2_stats
            masks = torch.stack([native_module.downsample_mask(ref_masks[s:s + 1], h, w)
                                 for s in range(n_refs)])
            bg_cols = [ref_feats[0, s][:, ~masks[s]] for s in range(n_refs)]
            bg = torch.cat(bg_cols, dim=1)
            if bg.shape[1]:
                # Exactly the source top-k, before prototype normalization or clustering.
                similarity = torch.einsum("cn,c->n", bg, mu_fg)
                count = max(1, int(CONFIG["hard_bg_fraction"] * similarity.numel()))
                hard = bg[:, torch.topk(similarity, k=count).indices]
                x = F.normalize(hard.transpose(0, 1), p=2, dim=1)
                labels = native_module.agglomerative_clustering(x, tau=self.tau)
                clusters = int(labels.max().item()) + 1
                protos = native_module.compute_cluster_prototypes(x, labels, K=clusters)
            else:
                count, clusters = 0, 0
                protos = torch.empty((0, mu_fg.numel()), dtype=mu_fg.dtype, device=mu_fg.device)
            protos, dropped, fallback = orthogonal_prototypes(protos, mu_fg, mu_bg)
            score_target = tgt_feat_raw if self.use_raw_target_scoring and tgt_feat_raw is not None else tgt_feat
            cluster_target = tgt_feat_raw if self.use_raw_target_clustering and tgt_feat_raw is not None else tgt_feat
            # Identical native FG prototypes and LSE expression; sf and mu_fg are returned verbatim.
            sim_fg = torch.einsum("bchw,kc->bkhw", cluster_target, fg_protos)
            temperature = max(1e-4, self.cluster_logsumexp_temp)
            fg = (temperature * torch.logsumexp(sim_fg / temperature, dim=1)).squeeze(0)
            sb = background_response(score_target, protos, self.background_aggregation, temperature)
            sbn = sb - sb.min()
            sbn = sbn / sbn.max().clamp_min(1e-6)
            self.part2_receipt = dict(aggregation=self.background_aggregation, hard_bg_tokens=count,
                                     bg_clusters=clusters, orthogonal_prototypes=len(protos),
                                     dropped_prototypes=dropped, native_fallback=fallback,
                                     bg_prototypes_sha256=hashlib.sha256(protos.detach().cpu().numpy().tobytes()).hexdigest(),
                                     fg_and_mu_preserved=True)
            return fg - float(self.dino_bg_weight) * sb, sf, sbn, mu_fg

    return Part2EvidenceFoRIS


@torch.no_grad()
def prefinal_tail(host, fmaps, aggregation):
    """Direct CPU Part2 -> Part3 -> Part4 -> native binarize, before CRF.

    The released CRF extension requires CUDA. Its separate finalize phase must
    complete before these predictions may be scored as a complete method.
    """
    host.background_aggregation = aggregation
    masks = host._ref_masks.unsqueeze(1)
    n_refs, h, w = host._ref_images.shape[0], fmaps.shape[-2], fmaps.shape[-1]
    times = {}

    def timed(name, function, *args, **kwargs):
        start = time.monotonic()
        value = function(*args, **kwargs)
        times[name] = time.monotonic() - start
        return value

    part2 = timed("part2", host._part2_background_suppression,
                  fmaps_norm=fmaps, ref_masks=masks, n_refs=n_refs, h=h, w=w)
    if part2 is None:
        raise RuntimeError("No foreground tokens in reference mask")
    s2, sf, sbn, mu_fg, denoised = part2
    s3, cand, prior = timed("part3", host._part3_clustering, s2, sf=sf, mu_fg=mu_fg,
                           ref_feats_raw=fmaps[:, :n_refs], tgt_feat_raw=fmaps[:, n_refs],
                           ref_masks=masks, n_refs=n_refs, h=h, w=w)
    score = timed("part4", host._part4_semantic_consistency_correction, s3,
                  sf=sf, sbn=sbn, cand_soft=cand, seed_prior=prior, tgt_feat=denoised)
    pre = timed("binarize", host._binarize_response, score, target_hw=host._tgt_image.shape[-2:])
    fields = dict(s2=s2, sf=sf, sbn=sbn, mu_fg=mu_fg, s3=s3, cand_soft=cand,
                  seed_prior=prior, score=score, gated_target_norm=denoised.norm(dim=1)[0])
    if any(not torch.isfinite(value).all() for value in fields.values()):
        raise ValueError("Non-finite native/intervention stage output")
    return pre, fields, dict(host.part2_receipt), times
