from gpu_guard import require_gpu_unpaused
require_gpu_unpaused(__file__)
import pathlib,sys,json,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from cgd.butterfly import butterfly as ref
from cgd.cuda_butterfly import butterfly as cuda

torch.manual_seed(2027);torch.set_num_threads(2);rows=[]
for p in (4,16,64,256,1024):
 for inverse in (False,True):
  # Planar RGB and broadcast angles exercise actual decoder layouts.
  x=torch.randn(2,3,p,device='cuda').transpose(1,2).detach().requires_grad_();a=(torch.randn(1,p.bit_length()-1,p//2,device='cuda')*.2).expand(2,-1,-1).detach().requires_grad_()
  w=torch.randn_like(x)
  u=ref(x,a,inverse);gr=torch.autograd.grad((u.tanh()*w).sum(),(x,a))
  v=cuda(x,a,inverse);gc=torch.autograd.grad((v.tanh()*w).sum(),(x,a))
  torch.testing.assert_close(u,v,rtol=2e-5,atol=1e-5)
  for g,h in zip(gr,gc):torch.testing.assert_close(g,h,rtol=2e-4,atol=2e-5)
  rows.append({'P':p,'inverse':inverse,'forward_max':(u-v).abs().max().item(),'input_gradient_max':(gr[0]-gc[0]).abs().max().item(),'angle_gradient_max':(gr[1]-gc[1]).abs().max().item()})
# Both uses of the same angles must contribute to the gradient.
x=torch.randn(8,1024,3,device='cuda',requires_grad=True);a=(torch.randn(8,10,512,device='cuda')*.1).requires_grad_()
y=ref(ref(x,a).tanh(),a,True);g=torch.autograd.grad(y.square().mean(),(x,a))
z=cuda(cuda(x,a).tanh(),a,True);h=torch.autograd.grad(z.square().mean(),(x,a))
for q,r in zip(g,h):torch.testing.assert_close(q,r,rtol=5e-4,atol=1e-7)
result={'status':'PASS_KERNEL_FP32_CURRENT_TESTS','rows':rows,'scope':'First-order CUDA implementation equivalence only; no model-quality or end-to-end speed claim. Inverse reconstruction is floating-point, not bitwise.'}
(ROOT/'results/cuda_butterfly_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
