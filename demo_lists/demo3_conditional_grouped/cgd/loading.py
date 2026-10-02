import torch
from src.models.transformer.pixnerd_t2i_heavydecoder import PixNerDiT

MODEL_KWARGS=dict(in_channels=3,patch_size=16,num_groups=24,hidden_size=1536,txt_embed_dim=2048,txt_max_length=128,num_text_blocks=4,decoder_hidden_size=64,num_encoder_blocks=16,num_decoder_blocks=2)

def load_pretrained(path):
    # mmap avoids materializing a second host copy; no unsafe pickle fallback.
    blob=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    state=blob['state_dict'] if 'state_dict' in blob else blob
    prefix='ema_denoiser.'
    weights={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    if not weights:raise ValueError('Expected official EMA denoiser keys; inspect checkpoint before changing prefix')
    with torch.device('meta'):base=PixNerDiT(**MODEL_KWARGS)
    base.load_state_dict(weights,strict=True,assign=True)
    return base
