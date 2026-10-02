"""Give projection-merged dense/factor routes the identical shared phase head."""
from dataclasses import dataclass
import torch

from baselines import core, heads, plain_attention, unheads, read_attention, write_attention
from factor_candidate import FactorState
from projection_merge import MergeCache, projected_read, factor_write
from prototype import phase_outputs


@dataclass
class MergedPhaseCache:
    first: object
    groups: tuple
    method: str
    plans: object
    second_matrix: torch.Tensor
    second_bias: torch.Tensor

    @classmethod
    def build(cls, model, image, dense, position, method, plans):
        merged = MergeCache.build(model, image, dense, position, method=method, plans=plans)
        conv = model.output_upscaling[3]
        assert conv.kernel_size == conv.stride == (2, 2)
        assert conv.padding == conv.output_padding == (0, 0)
        assert conv.dilation == (1, 1) and conv.groups == 1
        matrix = conv.weight.permute(0, 2, 3, 1).reshape(conv.in_channels, -1)
        return cls(merged.first, merged.groups, merged.method, merged.plans, matrix, conv.bias.repeat(4))


def merged_phase_predict(model, cache, sparse, backend="explicit"):
    """Exact same transformer schedules as merge; only the common head changes."""
    b = sparse.shape[0]
    learned = torch.cat((model.iou_token.weight, model.mask_tokens.weight), 0)
    tokens = torch.cat((learned[None].expand(b, -1, -1), sparse), 1)
    factor = cache.method == "factor"
    state = FactorState(cache, b) if factor else None
    x = None if factor else cache.first.base.expand(b, -1, -1)
    q, values = tokens, {}
    for i, block in enumerate(model.transformer.layers):
        if block.skip_first_layer_pe:
            q = plain_attention(block.self_attn, q, q, q, backend)
        else:
            q = q + plain_attention(block.self_attn, q + tokens, q + tokens, q, backend)
        q = block.norm1(q)
        if i == 0:
            k, v = cache.first.first_k, cache.first.first_v
        else:
            k, v = values["k"], values["v"]
        q = block.norm2(q + projected_read(block.cross_attn_token_to_image, q + tokens, k, v, backend))
        q = block.norm3(q + block.mlp(q))
        write = block.cross_attn_image_to_token
        if factor:
            write_q = cache.first.first_q if i == 0 else values["q"]
            state.update(*factor_write(write, q, tokens, write_q), block.norm4)
            projected = cache.groups[i].factor_project(state)
        else:
            if i == 0:
                update = write_attention(write, q + tokens, q, x, cache.first, i, cache.plans[1], backend)
            else:
                update = write.out_proj(unheads(core(heads(values["q"], write.num_heads),
                    heads(write.k_proj(q + tokens), write.num_heads), heads(write.v_proj(q), write.num_heads), backend)))
            x = block.norm4(x + update)
            projected = cache.groups[i].dense_project(x)
        values = dict(zip(cache.groups[i].names, projected))
    q = model.transformer.norm_final_attn(q + projected_read(model.transformer.final_attn_token_to_image,
        q + tokens, values["k"], values["v"], backend))
    return phase_outputs(model, cache, values["conv"], q)
