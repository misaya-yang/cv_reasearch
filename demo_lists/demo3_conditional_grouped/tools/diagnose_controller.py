from gpu_guard import require_gpu_unpaused
require_gpu_unpaused(__file__)
import pathlib,json,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'assets/source/PixNerd')]
import torch
from PIL import Image
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize
from cgd.loading import load_pretrained
from cgd.model import GroupedPixNerd

torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
r=ROOT/'assets/data/coco_pilot_v1';d=json.loads((r/'manifest.json').read_text());rows=[v for v in d['rows'] if v['split']=='val'][:4];pre=Compose([Resize(1024),CenterCrop(1024),ToTensor(),Normalize((.5,)*3,(.5,)*3)]);out=[]
for mode in ('conditional','dynamic'):
 m=GroupedPixNerd(load_pretrained(ROOT/'assets/checkpoints/pixnerd/model.ckpt').cuda().float(),mode=mode).cuda();state=torch.load(ROOT/f'results/pilot_v1/{mode}/resume.pt',weights_only=True,map_location='cpu')
 with torch.no_grad():
  for n,p in m.named_parameters():
   if n in state['trainable']:p.copy_(state['trainable'][n].to(p))
 del state
 with torch.no_grad():
  for row in rows:
   x=pre(Image.open(r/row['path']).convert('RGB'))[None].cuda();y=torch.load(r/'text_cache'/(str(row['image_id'])+'.pt'),weights_only=True)['embedding'][None].cuda().float();g=torch.Generator().manual_seed(2027+row['image_id']);noise=torch.randn(x.shape,generator=g).cuda()
   for tv in (.2,.5,.8):
    t=torch.tensor([tv],device='cuda');xt=tv*x+(1-tv)*noise;raw=torch.nn.functional.unfold(xt,32,stride=32).transpose(1,2);s,tt=m.semantic(xt,t,y,raw);s=s.reshape(-1,m.base.hidden_size);z=m.angle_head(s) if mode=='conditional' else m.filter_head(s)
    out.append({'mode':mode,'image_id':row['image_id'],'t':tv,'semantic_rms':s.square().mean().sqrt().item(),'controller_rms':z.square().mean().sqrt().item(),'controller_max':z.abs().max().item()})
 del m;torch.cuda.empty_cache()
(ROOT/'results/controller_diagnostics.json').write_text(json.dumps({'scope':'Four development-diagnostic images; no parameter selection/quality claim.','rows':out},indent=2));print(json.dumps(out,indent=2))
