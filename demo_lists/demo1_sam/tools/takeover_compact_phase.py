"""Same dense-associated cached transformer as the strongest phase control."""
from prototype import plain_attention,dense_read,dense_write,shape_plan
from takeover_compact_head_infer import head
from takeover_compact_phase_fallback import head as phase_head
import torch


def predict(model,cache,sparse,student,center,basis,budget,plans,phase_fallback=False):
    b=len(sparse);learned=torch.cat((model.iou_token.weight,model.mask_tokens.weight),0)
    tokens=torch.cat((learned[None].expand(b,-1,-1),sparse),1)
    q=tokens;x=cache.first.base.expand(b,-1,-1)
    for i,block in enumerate(model.transformer.layers):
        if block.skip_first_layer_pe:q=plain_attention(block.self_attn,q,q,q,'explicit')
        else:q=q+plain_attention(block.self_attn,q+tokens,q+tokens,q,'explicit')
        q=block.norm1(q)
        q=block.norm2(q+dense_read(block.cross_attn_token_to_image,q+tokens,x,cache.first,i,plans[2*i],'explicit'))
        q=block.norm3(q+block.mlp(q))
        x=block.norm4(x+dense_write(block.cross_attn_image_to_token,q+tokens,q,x,cache.first,i,plans[2*i+1],'explicit'))
    q=model.transformer.norm_final_attn(q+dense_read(model.transformer.final_attn_token_to_image,q+tokens,x,cache.first,len(model.transformer.layers),plans[-1],'explicit'))
    hyper=torch.stack([mlp(q[:,i+1]) for i,mlp in enumerate(model.output_hypernetworks_mlps)],1)
    masks=phase_head(model,cache,student,x,hyper,center,basis,budget) if phase_fallback else head(model,student,x,hyper,center,basis,budget)
    return masks,model.iou_prediction_head(q[:,0])
