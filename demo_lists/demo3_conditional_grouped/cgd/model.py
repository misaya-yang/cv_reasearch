import contextlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .butterfly import butterfly,group_indices

MODES=('original','fixed','conditional','group32','neighborhood','neighborhood_gated','neighborhood_static_gate','neighborhood_time_gate','neighborhood_full_stem','dynamic','full_stem')

class StaticContextGate(nn.Module):
    """Unconditional channel gain: controls whether new gain parameters alone help."""
    def __init__(self,dim):
        super().__init__();self.bias=nn.Parameter(torch.zeros(dim))
    def forward(self,s):return self.bias.expand(s.shape[0],-1)

class FullStem(nn.Module):
    """Exact nearest-phase old stem plus zero-initialized new pixel columns."""
    def __init__(self,base,side):
        super().__init__();old=base.patch_size
        if side%old:raise ValueError('Integer decoder/encoder ratio required')
        ratio=side//old
        keep=torch.tensor([c*side*side+(i*ratio)*side+j*ratio for c in range(base.in_channels) for i in range(old) for j in range(old)],device='cpu')
        flag=torch.ones(base.in_channels*side*side,dtype=torch.bool,device='cpu');flag[keep]=False
        self.register_buffer('keep',keep);self.register_buffer('extra',torch.arange(len(flag),device='cpu')[flag])
        self.extra_proj=nn.Linear(len(self.extra),base.hidden_size,bias=False)
        nn.init.zeros_(self.extra_proj.weight)
    def forward(self,x,old_embedder):return old_embedder(x.index_select(-1,self.keep))+self.extra_proj(x.index_select(-1,self.extra))

class GroupedPixNerd(nn.Module):
    """Explicit heavy-decoder adapter. No debug hooks or detached stem gradients."""
    def __init__(self,base,side=32,mode='conditional',angle_std=0.,checkpoint_encoder=False,butterfly_backend='eager',gate_placement='aligned',gate_envelope='none',cfg_gate='independent'):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.base=base;self.side=side;self.p=side*side;self.mode=mode;self.checkpoint_encoder=checkpoint_encoder
        if butterfly_backend not in ('eager','triton'):raise ValueError(butterfly_backend)
        self.butterfly_backend=butterfly_backend
        if gate_placement not in ('aligned','shifted','image_mean'):raise ValueError(gate_placement)
        if gate_placement!='aligned' and mode!='neighborhood_gated':raise ValueError('Placement ablations require the semantic gate')
        self.gate_placement=gate_placement
        if gate_envelope not in ('none','sigma','reliability'):raise ValueError(gate_envelope)
        if gate_envelope!='none' and mode!='neighborhood_gated':raise ValueError('Envelope diagnostic requires semantic gate')
        self.gate_envelope=gate_envelope
        if cfg_gate not in ('independent','shared_uncond'):raise ValueError(cfg_gate)
        if cfg_gate!='independent' and mode!='neighborhood_gated':raise ValueError('CFG routing requires semantic gate')
        self.cfg_gate=cfg_gate
        # Replicated edge pixels are correlated repeats: correct local-noise variance.
        offsets=torch.arange(-2,4);coords=(torch.arange(side)[:,None]+offsets).clamp(0,side-1)
        counts=F.one_hot(coords,num_classes=side).float().sum(1)/6
        q=counts.square().sum(1)
        self.register_buffer('variance_noise_factor',(1-q[:,None]*q[None,:]).reshape(1,side*side,1))
        base.requires_grad_(False)
        for m in [base.x_embedder,base.final_layer,*base.blocks[base.num_encoder_blocks:]]:m.requires_grad_(True)
        local=mode in ('neighborhood','neighborhood_gated','neighborhood_static_gate','neighborhood_time_gate','neighborhood_full_stem')
        self.group=32 if mode=='group32' else (1 if mode in ('original','full_stem') or local else 2)
        extra=35 if local else self.group-1
        self.partner=nn.Linear(3*extra,base.decoder_hidden_size,bias=False) if extra else None
        if self.partner is not None:nn.init.zeros_(self.partner.weight)
        self.angle_head=None
        if mode in ('conditional','fixed'):
            self.angle_head=nn.Linear(base.hidden_size,int(math.log2(self.p))*self.p//2)
            nn.init.zeros_(self.angle_head.weight);nn.init.normal_(self.angle_head.bias,std=angle_std)
        self.filter_head=nn.Linear(base.hidden_size,36) if mode=='dynamic' else None
        self.filter_proj=nn.Linear(3,base.decoder_hidden_size,bias=False) if mode=='dynamic' else None
        if self.filter_head is not None:
            nn.init.zeros_(self.filter_head.weight);nn.init.zeros_(self.filter_head.bias)
        self.full_stem=FullStem(base,side) if mode in ('full_stem','neighborhood_full_stem') else None
        self.context_gate=None
        if mode in ('neighborhood_gated','neighborhood_time_gate'):
            self.context_gate=nn.Sequential(nn.RMSNorm(base.hidden_size,eps=1e-6,elementwise_affine=False),nn.Linear(base.hidden_size,16,bias=False),nn.SiLU(),nn.Linear(16,base.decoder_hidden_size))
            nn.init.zeros_(self.context_gate[-1].weight);nn.init.zeros_(self.context_gate[-1].bias)
        if mode=='neighborhood_static_gate':self.context_gate=StaticContextGate(base.decoder_hidden_size)
        if side%base.patch_size:raise ValueError('Integer patch scaling required')
        base.decoder_patch_scaling_h=side/base.patch_size;base.decoder_patch_scaling_w=side/base.patch_size
        if self.partner is not None and not local:self.register_buffer('indices',group_indices(self.p,self.group,'cpu'))

    def semantic(self,x,t,y,raw):
        b=self.base;B,_,H,W=x.shape;eh,ew=H*b.patch_size//self.side,W*b.patch_size//self.side
        if H%self.side or W%self.side:raise ValueError('Image dimensions must be multiples of decoder patch')
        with contextlib.nullcontext() if self.full_stem is not None else torch.no_grad():
            xpos=b.fetch_pos(eh//b.patch_size,ew//b.patch_size,x.device)
            tt=b.t_embedder(t.view(-1)).view(B,-1,b.hidden_size)
            y=b.y_embedder(y).view(B,-1,b.hidden_size)+b.y_pos_embedding.to(y.dtype)
            condition=F.silu(tt)
            for block in b.text_refine_blocks:y=block(y,condition)
            if self.full_stem is not None:s=self.full_stem(raw,b.s_embedder)
            else:
                enc=F.interpolate(x,(eh,ew))
                enc=F.unfold(enc,kernel_size=b.patch_size,stride=b.patch_size).transpose(1,2)
                s=b.s_embedder(enc)
            for block in b.blocks[:b.num_encoder_blocks]:
                s=checkpoint(block,s,y,condition,xpos,use_reentrant=False) if self.checkpoint_encoder and self.full_stem is not None and self.training else block(s,y,condition,xpos)
            return F.silu(tt+s),tt

    def neighborhood(self,rgb,k):
        # Fixed same-patch neighborhood with replicate edges; even k6 has offsets -2..3.
        image=rgb.transpose(1,2).reshape(-1,3,self.side,self.side)
        lo=(k-1)//2;hi=k-1-lo
        patches=F.unfold(F.pad(image,(lo,hi,lo,hi),mode='replicate'),k)
        return patches.reshape(-1,3,k*k,self.p).permute(0,3,2,1)

    def forward(self,x,t,y):
        transform=butterfly
        if self.butterfly_backend=='triton':
            from .cuda_butterfly import butterfly as transform
        b=self.base;B,_,H,W=x.shape
        raw=F.unfold(x,kernel_size=self.side,stride=self.side).transpose(1,2)
        ss,tt=self.semantic(x,t,y,raw);L=ss.shape[1];s=ss.reshape(B*L,-1)
        rgb=raw.reshape(B*L,3,self.p).transpose(1,2)
        angles=None
        if self.angle_head is not None:
            if self.mode=='conditional':angles=self.angle_head(s).reshape(B*L,-1,self.p//2)
            else:
                # Time-only basis is identical across patches: compute once/image.
                angles=self.angle_head(tt[:,0,:]).reshape(B,1,-1,self.p//2).expand(-1,L,-1,-1).reshape(B*L,-1,self.p//2)
            rgb=transform(rgb,angles)
        if self.filter_head is not None:
            kernel=self.filter_head(s)[:,None,:,None]
            filtered=(self.neighborhood(rgb,6)*kernel).sum(-2)
        field=b.x_embedder(rgb,self.side,self.side)
        if self.filter_proj is not None:
            # Preserve direct raw RGB: a velocity head must retain its noise term.
            field=field+self.filter_proj(filtered)
        if self.partner is not None:
            if self.mode in ('neighborhood','neighborhood_gated','neighborhood_static_gate','neighborhood_time_gate','neighborhood_full_stem'):
                neighbors=self.neighborhood(rgb,6)
                own=2*6+2
                extra=torch.cat((neighbors[...,:own,:],neighbors[...,own+1:,:]),dim=-2)
            else:extra=rgb[:,self.indices[:,1:],:]
            context=self.partner(extra.reshape(B*L,self.p,-1))
            if self.context_gate is not None:
                if self.mode=='neighborhood_time_gate':gain=self.context_gate(tt[:,0,:])[:,None,:].expand(-1,L,-1).reshape(B*L,-1)
                elif self.cfg_gate=='shared_uncond':
                    if B%2:raise ValueError('Official CFG batch must contain unconditioned then conditioned halves')
                    image_s=s.reshape(B,L,-1)[:B//2].reshape(B//2*L,-1)
                    image_gain=self.context_gate(image_s)
                    gain=torch.cat((image_gain,image_gain),dim=0)
                else:gain=self.context_gate(s)
                if self.gate_placement!='aligned':
                    gain=gain.reshape(B,L,-1)
                    if self.gate_placement=='shifted':gain=gain.roll(L//2,dims=1)
                    else:gain=gain.mean(1,keepdim=True).expand(-1,L,-1)
                    gain=gain.reshape(B*L,-1)
                if self.gate_envelope=='sigma':
                    gain=(gain.reshape(B,L,-1)*(1-t).reshape(B,1,1)).reshape(B*L,-1)
                if self.gate_envelope=='reliability':
                    variance=neighbors.var(dim=-2,unbiased=False).mean(-1,keepdim=True)/self.variance_noise_factor
                    time=t.reshape(B,1,1,1).expand(-1,L,-1,-1).reshape(B*L,1,1)
                    rho=((1-time).square()/(variance+1e-6)).clamp(0,1)
                    mask=(1-time)+time*rho
                    context=context*(1+gain[:,None,:]*mask)
                else:context=context*(1+gain[:,None,:])
            field=field+context
        for block in b.blocks[b.num_encoder_blocks:]:field=block(field,s)
        out=b.final_layer(field)
        if angles is not None:out=transform(out,angles,inverse=True)
        out=out.transpose(1,2).reshape(B,L,-1).transpose(1,2).contiguous()
        return F.fold(out,(H,W),kernel_size=self.side,stride=self.side)
