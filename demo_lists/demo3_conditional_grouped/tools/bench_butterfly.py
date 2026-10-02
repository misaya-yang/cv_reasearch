from gpu_guard import require_gpu_unpaused
require_gpu_unpaused(__file__)
import pathlib,sys,json,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from cgd.butterfly import butterfly as ref
from cgd.cuda_butterfly import butterfly as cuda

torch.set_num_threads(2);torch.manual_seed(2027);result={}
for name,fn in [('eager',ref),('triton',cuda)]:
 x=torch.randn(1024,1024,3,device='cuda',requires_grad=True);a=(torch.randn(1024,10,512,device='cuda')*.1).requires_grad_()
 def step():
  z=fn(fn(x,a).tanh(),a,True);z.square().mean().backward();x.grad=None;a.grad=None
 for _ in range(4):step()
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
 for _ in range(12):
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();step();end.record();end.synchronize();times.append(start.elapsed_time(end))
 result[name]={'milliseconds':times,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
 del x,a;torch.cuda.empty_cache()
report={'scope':'Operator-only analyze/tanh/synthesize/backward, 1024 patches of P1024; shared GPU; excludes model/semantic stream. Not an end-to-end speed claim.','results':result}
(ROOT/'results/butterfly_operator_bench.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
