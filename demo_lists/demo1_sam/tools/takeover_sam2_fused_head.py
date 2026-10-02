"""Stream the original two local upsampling stages directly into all mask logits."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _fused(X,W1,B1,LNW,LNB,W2,B2,T0,T1,H,Y,
           EPS:tl.constexpr,BN:tl.constexpr,ROUND:tl.constexpr):
    block=tl.program_id(0);prompt=tl.program_id(1)
    dtype = X.dtype.element_ty
    parents=block*BN+tl.arange(0,BN)
    k=tl.arange(0,64);columns=tl.arange(0,256)
    acc=tl.zeros((BN,256),tl.float32)
    # A full256x256 FP16 weight tile exceeded this GPU's shared-memory limit.
    # Keep original K256 contraction and FP32 accumulation in four K64 tiles.
    for start in range(0,256,64):
        kk=k+start
        x=tl.load(X+(prompt*4096+parents[:,None])*256+kk[None,:])
        w=tl.load(W1+kk[:,None]*256+columns[None,:])
        acc=tl.dot(x,w,acc,input_precision='ieee')
    first=acc+tl.load(B1+columns)[None,:]
    first=first.to(dtype)
    skip=tl.load(T1+parents[:,None]*256+(columns[None,:]%64)*4+columns[None,:]//64)
    first=(first.to(tl.float32)+skip.to(tl.float32)).to(dtype)
    pre=tl.reshape(first,(BN*4,64)).to(tl.float32)
    mean=tl.sum(pre,1)/64
    if ROUND:mean=mean.to(dtype).to(tl.float32)
    delta=pre-mean[:,None]
    if ROUND:delta=delta.to(dtype).to(tl.float32)
    squared=delta*delta
    if ROUND:squared=squared.to(dtype).to(tl.float32)
    variance=tl.sum(squared,1)/64
    if ROUND:variance=variance.to(dtype).to(tl.float32)
    v=variance+EPS
    if ROUND:v=v.to(dtype).to(tl.float32)
    denom=tl.sqrt(v)
    if ROUND:denom=denom.to(dtype).to(tl.float32)
    normalized=delta/denom[:,None]
    if ROUND:normalized=normalized.to(dtype).to(tl.float32)
    channels=tl.arange(0,64)
    scaled=normalized*tl.load(LNW+channels)[None,:].to(tl.float32)
    if ROUND:scaled=scaled.to(dtype).to(tl.float32)
    ln=scaled+tl.load(LNB+channels)[None,:].to(tl.float32)
    if ROUND:ln=ln.to(dtype).to(tl.float32)
    child=(0.5*ln*(1+libdevice.erf(ln*0.7071067811865476))).to(dtype)
    c2=tl.arange(0,128)
    w2=tl.load(W2+channels[:,None]*128+c2[None,:])
    second=(tl.dot(child,w2,input_precision='ieee')+tl.load(B2+c2)[None,:]).to(dtype)
    rows=tl.arange(0,BN*4)
    parent2=block*BN+rows//4;phase1=rows%4;phase2=c2//32
    raster=((phase1[:,None]//2)*2+phase2[None,:]//2)*4+(phase1[:,None]%2)*2+phase2[None,:]%2
    skip0=tl.load(T0+parent2[:,None]*512+(c2[None,:]%32)*16+raster)
    second=(second.to(tl.float32)+skip0.to(tl.float32)).to(dtype).to(tl.float32)
    feature=(0.5*second*(1+libdevice.erf(second*0.7071067811865476))).to(dtype)
    feature=tl.reshape(feature,(BN*16,32))
    hc=tl.arange(0,32);m=tl.arange(0,16)
    hyper=tl.load(H+prompt*128+m[None,:]*32+hc[:,None],m[None,:]<4,0)
    logits=tl.dot(feature,hyper,input_precision='ieee').to(dtype)
    px=tl.arange(0,BN*16);parent=block*BN+px//16;ph1=(px%16)//4;ph2=px%4
    row=(parent//64)*4+(ph1//2)*2+ph2//2
    col=(parent%64)*4+(ph1%2)*2+ph2%2
    tl.store(Y+(prompt*4+m[None,:])*65536+row[:,None]*256+col[:,None],logits,m[None,:]<4)


def fused_head(decoder,cache,x,hyper,parents=16,warps=4,round_intermediates=True):
    assert x.shape[1:]==(4096,256) and x.is_contiguous() and hyper.is_contiguous()
    assert hyper.shape==(len(x),4,32) and x.dtype in (torch.float16,torch.float32)
    norm=decoder.output_upscaling[1]
    y=torch.empty((len(x),4,256,256),device=x.device,dtype=x.dtype)
    _fused[(4096//parents,len(x))](x,cache.phase.conv_matrix,cache.phase.conv_bias,norm.weight,norm.bias,
          cache.phase.second_matrix,cache.phase.second_bias,cache.t0,cache.t1,hyper,y,
          EPS=norm.eps,BN=parents,ROUND=round_intermediates,num_warps=warps,num_stages=1,enable_fp_fusion=False)
    return y
