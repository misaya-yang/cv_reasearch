from gpu_guard import require_gpu_unpaused
require_gpu_unpaused(__file__)
import json,pathlib,sys,time,statistics
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'assets/source/PixNerd')]
import torch
from PIL import Image
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize
from cgd.loading import load_pretrained
from cgd.model import GroupedPixNerd

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
d=json.loads((ROOT/'assets/data/coco_pilot_v1/manifest.json').read_text());row=next(r for r in d['rows'] if r['split']=='val');dr=ROOT/'assets/data/coco_pilot_v1';pre=Compose([Resize(1024),CenterCrop(1024),ToTensor(),Normalize((.5,)*3,(.5,)*3)])
x=pre(Image.open(dr/row['path']).convert('RGB'))[None].cuda();y=torch.load(dr/'text_cache'/(str(row['image_id'])+'.pt'),weights_only=True)['embedding'][None].cuda().float();g=torch.Generator().manual_seed(2027);noise=torch.randn(x.shape,generator=g).cuda();t=torch.tensor([.5],device='cuda');xt=t[:,None,None,None]*x+(1-t[:,None,None,None])*noise;target=x-noise
base=load_pretrained(ROOT/'assets/checkpoints/pixnerd/model.ckpt').cuda().float();m=GroupedPixNerd(base,mode='conditional',side=32).cuda();state=torch.load(ROOT/'results/pilot_v1/conditional/resume.pt',weights_only=True,map_location='cpu')
with torch.no_grad():
 for n,p in m.named_parameters():
  if n in state['trainable']:p.copy_(state['trainable'][n].to(p))
del state
# Check actual pretrained+trained complete outputs and angle-head gradients.
m.butterfly_backend='eager';out=m(xt,t,y);loss=(out-target).square().mean();loss.backward();refout=out.detach();refgrad=m.angle_head.weight.grad.clone();m.zero_grad(set_to_none=True);del out,loss
m.butterfly_backend='triton';out=m(xt,t,y);loss=(out-target).square().mean();loss.backward();delta=(out-refout).abs().max().item();gd=(m.angle_head.weight.grad-refgrad).abs().max().item()
torch.testing.assert_close(out,refout,rtol=1e-5,atol=1e-5);torch.testing.assert_close(m.angle_head.weight.grad,refgrad,rtol=1e-3,atol=1e-7);m.zero_grad(set_to_none=True);del out,loss,refout,refgrad
bench={}
for backend in ('eager','triton'):
 m.butterfly_backend=backend
 def step():m.zero_grad(set_to_none=True);(m(xt,t,y)-target).square().mean().backward()
 for _ in range(3):step()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
 for _ in range(10):
  torch.cuda.synchronize();start=time.perf_counter();step();torch.cuda.synchronize();times.append(time.perf_counter()-start)
 bench[backend]={'forward_backward_seconds':times,'median_seconds':statistics.median(times),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
report={'status':'PASS_CURRENT_PRETRAINED_TRAINED_FIXTURE','max_velocity_abs':delta,'angle_gradient_abs':gd,'precision':'FP32','tf32':False,'bench':bench,'scope':'Complete semantic+pixel forward/backward; excludes AdamW/data/text encoding. Same weights/input/target, shared GPU. No isolated-throughput or quality gain claim.'};(ROOT/'results/model_backend_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
