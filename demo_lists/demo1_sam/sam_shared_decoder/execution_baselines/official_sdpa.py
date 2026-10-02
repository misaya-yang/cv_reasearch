"""Supplementary uncached official computation with SDPA attention only.

Pinned official sources/weights are unchanged. The untouched explicit official
decoder remains the numerical reference. This adapter is a fifth measured arm,
so the original's applicable SDPA optimization is not withheld from baselines.
"""
import torch

from baselines import plain_attention


def official_sdpa_predict(model, image, pe, dense, sparse):
    b = sparse.shape[0]
    learned = torch.cat((model.iou_token.weight, model.mask_tokens.weight), 0)
    tokens = torch.cat((learned[None].expand(b, -1, -1), sparse), 1)
    # Retain the original's image replication/addition and dense projections.
    x = (torch.repeat_interleave(image, b, dim=0) + dense.expand(b, -1, -1, -1))
    height, width = x.shape[-2:]
    x = x.flatten(2).transpose(1, 2)
    position = torch.repeat_interleave(pe, b, dim=0).flatten(2).transpose(1, 2)
    q = tokens
    for block in model.transformer.layers:
        if block.skip_first_layer_pe:
            q = plain_attention(block.self_attn, q, q, q, "sdpa")
        else:
            q = q + plain_attention(block.self_attn, q + tokens, q + tokens, q, "sdpa")
        q = block.norm1(q)
        q = block.norm2(q + plain_attention(block.cross_attn_token_to_image,
                                            q + tokens, x + position, x, "sdpa"))
        q = block.norm3(q + block.mlp(q))
        x = block.norm4(x + plain_attention(block.cross_attn_image_to_token,
                                            x + position, q + tokens, q, "sdpa"))
    q = model.transformer.norm_final_attn(q + plain_attention(
        model.transformer.final_attn_token_to_image, q + tokens, x + position, x, "sdpa"))
    upscaled = model.output_upscaling(x.transpose(1, 2).reshape(b, -1, height, width))
    hyper = torch.stack([mlp(q[:, i+1]) for i, mlp in enumerate(model.output_hypernetworks_mlps)], 1)
    masks = (hyper @ upscaled.flatten(2)).reshape(b, model.num_mask_tokens, *upscaled.shape[-2:])
    return masks, model.iou_prediction_head(q[:, 0])
