"""SAM2 high-res skip conditioned surrogate with image-only static projection."""
import math
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from takeover_sam2_patch_check import tiles
from takeover_compact_phase_fallback import build_patch_cache
from takeover_head_fallback import grid


def static_features(high_res):
    t0, t1 = tiles(high_res[0], 4), tiles(high_res[1], 2)
    return torch.cat((t1.flatten(1), t0.flatten(1)), -1), t0, t1


def build_cache(decoder, high_res, student):
    assert decoder.use_high_res_features and decoder.num_mask_tokens == 4
    static, t0, t1 = static_features(high_res)
    # The folded bias and static768 projection are evaluated once per image.
    hidden = F.linear(static, student[0].weight[:, 256:], student[0].bias)
    return SimpleNamespace(static_hidden=hidden, t0=t0, t1=t1,
                           phase=build_patch_cache(decoder))


def phase_patches(decoder, phase, states, skip0, skip1):
    pre = (states @ phase.conv_matrix + phase.conv_bias).reshape(-1, 4, 64)
    pre = pre + skip1.permute(0, 2, 3, 1).reshape(-1, 4, 64)
    norm = decoder.output_upscaling[1]
    mean = pre.mean(-1, keepdim=True)
    variance = (pre-mean).square().mean(-1, keepdim=True)
    child = (pre-mean)/torch.sqrt(variance+norm.eps)
    child = decoder.output_upscaling[2](child*norm.weight+norm.bias)
    second = child @ phase.second_matrix + phase.second_bias
    raster = second.reshape(-1, 2, 2, 2, 2, 32).permute(0, 5, 1, 3, 2, 4).reshape(-1, 32, 16)
    return decoder.output_upscaling[4](raster+skip0.flatten(2))


def head(decoder, student, cache, x, hyper, center, basis, budget):
    p = len(x)
    hidden = F.linear(x, student[0].weight[:, :256])+cache.static_hidden[None]
    coeff = student[2](student[1](hidden))
    weights = torch.einsum('pmc,ucr->pmur', hyper, basis.reshape(16, 32, 64)).reshape(p, 64, 64)
    bias = torch.einsum('pmc,uc->pmu', hyper, center.reshape(16, 32)).reshape(p, 64)
    ph = (torch.bmm(coeff, weights.transpose(1, 2))+bias[:, None]).reshape(p, 4096, 4, 16)
    if budget:
        margin = (ph.abs()/hyper.norm(dim=-1)[:, None, :, None].clamp_min(1e-6)).amin((-2, -1))
        idx = margin.topk(math.ceil(4096*budget), dim=-1, largest=False).indices
        batch = torch.arange(p, device=x.device)[:, None]
        patches = phase_patches(decoder, cache.phase, x[batch, idx].reshape(-1, 256),
                                cache.t0[idx].reshape(-1, 32, 4, 4),
                                cache.t1[idx].reshape(-1, 64, 2, 2))
        h = hyper[:, None].expand(-1, idx.shape[1], -1, -1).reshape(-1, 4, 32)
        ph[batch, idx] = torch.bmm(h, patches).reshape(p, idx.shape[1], 4, 16)
    return grid(ph)


def dense_phase_head(decoder, cache, x, hyper):
    p = len(x)
    pre = (x @ cache.phase.conv_matrix+cache.phase.conv_bias).reshape(p,4096,4,64)
    pre = pre+cache.t1.permute(0,2,3,1).reshape(4096,4,64)[None]
    norm = decoder.output_upscaling[1]
    mean = pre.mean(-1,keepdim=True)
    variance = (pre-mean).square().mean(-1,keepdim=True)
    child = decoder.output_upscaling[2]((pre-mean)/torch.sqrt(variance+norm.eps)*norm.weight+norm.bias)
    second = child @ cache.phase.second_matrix+cache.phase.second_bias
    raster = second.reshape(p,4096,2,2,2,2,32).permute(0,1,6,2,4,3,5).reshape(p,4096,32,16)
    feature = decoder.output_upscaling[4](raster+cache.t0.flatten(2)[None])
    return grid(torch.einsum('pmc,pncu->pnmu',hyper,feature))


def decode(decoder, student, cache, image, pe, sparse, dense, center, basis, budget, mode='compact'):
    # Official transformer, token0/multimask, IoU, mask-token and object-score
    # interfaces retained. The full original upscaler is absent in this path.
    offset = int(decoder.pred_obj_scores)
    learned = [decoder.iou_token.weight, decoder.mask_tokens.weight]
    if offset:
        learned.insert(0, decoder.obj_score_token.weight)
    tokens = torch.cat((torch.cat(learned)[None].expand(len(sparse), -1, -1), sparse), 1)
    src = torch.repeat_interleave(image, len(sparse), dim=0)+dense
    pos = torch.repeat_interleave(pe, len(sparse), dim=0)
    hs, x = decoder.transformer(src, pos, tokens)
    mask_tokens = hs[:, offset+1:offset+5]
    hyper = torch.stack([mlp(mask_tokens[:, i]) for i, mlp in enumerate(decoder.output_hypernetworks_mlps)], 1)
    scores = decoder.iou_prediction_head(hs[:, offset])
    obj = decoder.pred_obj_score_head(hs[:, 0]) if offset else scores.new_ones(len(sparse), 1)*10
    masks = dense_phase_head(decoder,cache,x,hyper) if mode=='dense_phase' else head(decoder, student, cache, x, hyper, center, basis, budget)
    return masks, scores, mask_tokens, obj


def load_student(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    assert state['normalization_folded']
    student = nn.Sequential(nn.Linear(1024, 64), nn.GELU(), nn.Linear(64, 64)).to(device).eval()
    student.load_state_dict(state['state'])
    return student, state['feature_center'], state['basis']


def dynamic_choice(decoder, masks, scores):
    stable = decoder._get_stability_scores(masks[:, :1])[:, 0] >= decoder.dynamic_multimask_stability_thresh
    return torch.where(stable, torch.zeros_like(stable, dtype=torch.long), scores[:, 1:].argmax(-1)+1)


def quality_all(masks, truth, valid=None):
    from takeover_rank_features import quality
    # Existing quality helper drops its first mask. Keep native SAM2 token0 too.
    return quality(torch.cat((torch.zeros_like(masks[:, :1]), masks), 1), truth, valid)
