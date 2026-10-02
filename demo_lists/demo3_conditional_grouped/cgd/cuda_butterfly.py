"""Exact-real butterfly, fused per-patch CUDA; reversible first-order backward."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _forward(X,A,Y,S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,P:tl.constexpr,L:tl.constexpr,INV:tl.constexpr):
 m=tl.program_id(0);i=tl.arange(0,P)
 x0=tl.load(X+m*S0+i*S1);x1=tl.load(X+m*S0+i*S1+S2);x2=tl.load(X+m*S0+i*S1+2*S2)
 for n in range(L):
  level=L-1-n if INV else n;stride=1<<level;pair=i^stride;first=(i&stride)==0
  j=(i//(2*stride))*stride+i%stride
  angle=tl.load(A+m*A0+level*A1+j*A2)
  c=libdevice.cos(angle);s=libdevice.sin(angle)
  if INV:s=-s
  q0=tl.gather(x0,pair,0);q1=tl.gather(x1,pair,0);q2=tl.gather(x2,pair,0)
  sign=tl.where(first,-1.,1.)
  x0=c*x0+sign*s*q0;x1=c*x1+sign*s*q1;x2=c*x2+sign*s*q2
 tl.store(Y+m*P*3+i*3,x0);tl.store(Y+m*P*3+i*3+1,x1);tl.store(Y+m*P*3+i*3+2,x2)

@triton.jit
def _backward(Y,A,G,DX,DA,G0:tl.constexpr,G1:tl.constexpr,G2:tl.constexpr,A0:tl.constexpr,A1:tl.constexpr,A2:tl.constexpr,P:tl.constexpr,L:tl.constexpr,INV:tl.constexpr):
 m=tl.program_id(0);i=tl.arange(0,P)
 x0=tl.load(Y+m*P*3+i*3);x1=tl.load(Y+m*P*3+i*3+1);x2=tl.load(Y+m*P*3+i*3+2)
 g0=tl.load(G+m*G0+i*G1);g1=tl.load(G+m*G0+i*G1+G2);g2=tl.load(G+m*G0+i*G1+2*G2)
 for n in range(L):
  level=n if INV else L-1-n;stride=1<<level;pair=i^stride;first=(i&stride)==0
  j=(i//(2*stride))*stride+i%stride
  angle=tl.load(A+m*A0+level*A1+j*A2);c=libdevice.cos(angle);s=libdevice.sin(angle)
  direction=-1. if INV else 1.
  s=s*direction
  q0=tl.gather(x0,pair,0);q1=tl.gather(x1,pair,0);q2=tl.gather(x2,pair,0)
  h0=tl.gather(g0,pair,0);h1=tl.gather(g1,pair,0);h2=tl.gather(g2,pair,0)
  da=direction*((-g0*q0+h0*x0)+(-g1*q1+h1*x1)+(-g2*q2+h2*x2))
  tl.store(DA+m*L*(P//2)+level*(P//2)+j,da,mask=first)
  sign=tl.where(first,1.,-1.)
  # Recover previous layer's values and adjoints using the same inverse.
  x0=c*x0+sign*s*q0;x1=c*x1+sign*s*q1;x2=c*x2+sign*s*q2
  g0=c*g0+sign*s*h0;g1=c*g1+sign*s*h1;g2=c*g2+sign*s*h2
 tl.store(DX+m*P*3+i*3,g0);tl.store(DX+m*P*3+i*3+1,g1);tl.store(DX+m*P*3+i*3+2,g2)

class _Butterfly(torch.autograd.Function):
 @staticmethod
 def forward(ctx,rgb,angles,inverse):
  if not rgb.is_cuda or rgb.dtype!=torch.float32 or angles.dtype!=torch.float32:raise ValueError('CUDA FP32 only; use eager reference otherwise')
  if rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('Expected [patches,P,3]')
  m,p,_=rgb.shape;l=p.bit_length()-1
  if p!=1<<l or p>1024 or angles.shape!=(m,l,p//2):raise ValueError('Invalid butterfly shape')
  out=torch.empty((m,p,3),device=rgb.device,dtype=rgb.dtype)
  _forward[(m,)](rgb,angles,out,*rgb.stride(),*angles.stride(),P=p,L=l,INV=inverse,num_warps=8,enable_fp_fusion=False)
  ctx.save_for_backward(out,angles);ctx.inverse=inverse
  return out
 @staticmethod
 @torch.autograd.function.once_differentiable
 def backward(ctx,grad):
  out,angles=ctx.saved_tensors;m,p,_=out.shape;l=p.bit_length()-1
  dx=torch.empty_like(out);da=torch.empty((m,l,p//2),device=out.device,dtype=out.dtype)
  _backward[(m,)](out,angles,grad,dx,da,*grad.stride(),*angles.stride(),P=p,L=l,INV=ctx.inverse,num_warps=8,enable_fp_fusion=False)
  return dx,da,None

def butterfly(rgb,angles,inverse=False):return _Butterfly.apply(rgb,angles,inverse)
