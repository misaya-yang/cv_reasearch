"""Pretrained 1024 resource probe / paired short training. No automatic branch escalation."""
from gpu_guard import require_gpu_unpaused
require_gpu_unpaused(__file__)
import argparse,hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'assets/source/PixNerd')]
import torch
torch.set_num_threads(2)
from PIL import Image
from torchvision.transforms import Resize,CenterCrop,ToTensor,Normalize
from cgd.loading import load_pretrained
from cgd.model import GroupedPixNerd,MODES

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--phase',choices=['resource','train'],default='resource');p.add_argument('--steps',type=int,default=64);p.add_argument('--batch',type=int,default=1);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--gate-lr',type=float);p.add_argument('--seed',type=int,default=2027);p.add_argument('--checkpoint',type=pathlib.Path,default=ROOT/'assets/checkpoints/pixnerd/model.ckpt');p.add_argument('--data',type=pathlib.Path,default=ROOT/'assets/data/coco_pilot_v1/manifest.json');p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--resume',type=pathlib.Path);p.add_argument('--warmstart',type=pathlib.Path);p.add_argument('--time-sampling',choices=['native','stratified'],default='native');p.add_argument('--butterfly-backend',choices=['eager','triton'],default='eager');a=p.parse_args()
 if a.resume and a.warmstart:p.error('Choose exact resume or a declared fresh-optimizer warmstart')
 adapter_source_sha256=hashlib.sha256((ROOT/'cgd/model.py').read_bytes()).hexdigest()
 torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; do not run this on the no-card host')
 a.out.mkdir(parents=True,exist_ok=True);d=json.loads(a.data.read_text());base=load_pretrained(a.checkpoint).cuda().float();base.decoder_patch_scaling_h=2;base.decoder_patch_scaling_w=2
 pre=__import__('torchvision').transforms.Compose([Resize(1024),CenterCrop(1024),ToTensor(),Normalize((.5,)*3,(.5,)*3)])
 def example(row):
  with Image.open(a.data.parent/row['path']) as im:img=pre(im.convert('RGB'))
  c=torch.load(a.data.parent/'text_cache'/(str(row['image_id'])+'.pt'),weights_only=True)
  if c['caption_sha256']!=hashlib.sha256(row['caption'].encode()).hexdigest():raise ValueError('Text cache caption mismatch')
  return img,c['embedding'].float()
 def fixture(rows,seed,fixed_t=None):
  pairs=[example(r) for r in rows];x=torch.stack([z[0] for z in pairs]).cuda();y=torch.stack([z[1] for z in pairs]).cuda();g=torch.Generator().manual_seed(seed)
  t=torch.randn(len(rows),generator=g).sigmoid();t=t/(t+(1-t)*4.) if fixed_t is None else torch.full_like(t,fixed_t)
  if fixed_t is None and a.phase=='train' and a.time_sampling=='stratified':
   # Independent time RNG leaves the paired noise stream unchanged.
   gt=torch.Generator().manual_seed(seed+9017);u=torch.rand(len(rows),generator=gt)
   band=(torch.arange(len(rows))+seed-a.seed)%3;t=.05+.3*band+.3*u
  noise=torch.randn(x.shape,generator=g).cuda();t=t.cuda();xt=t[:,None,None,None]*x+(1-t[:,None,None,None])*noise
  return xt,t,y,x-noise
 rows=[r for r in d['rows'] if r['split']=='train'];val=[r for r in d['rows'] if r['split']=='val'];xt,t,y,target=fixture(rows[:a.batch],a.seed)
 with torch.no_grad():reference=base(xt,t,y)
 model=GroupedPixNerd(base,mode=a.mode,side=32,checkpoint_encoder=True,butterfly_backend=a.butterfly_backend).cuda()
 with torch.no_grad():initial=model(xt,t,y)
 numerical={'max_abs':(initial-reference).abs().max().item(),'rtol':1e-5,'atol':1e-5}
 try:torch.testing.assert_close(initial,reference,rtol=1e-5,atol=1e-5);numerical['status']='PASS_CURRENT_FIXTURE'
 except AssertionError:numerical['status']='FAIL';(a.out/'numerical.json').write_text(json.dumps(numerical,indent=2));raise
 del reference,initial
 if a.resume or a.warmstart:
  state=torch.load(a.resume or a.warmstart,weights_only=True,map_location='cpu');train_keys={n for n,p in model.named_parameters() if p.requires_grad}
  old_keys=set(state['trainable'])
  if a.resume and old_keys!=train_keys:raise ValueError('Resume parameter contract mismatch')
  if a.warmstart and (not old_keys<=train_keys or any(not n.startswith(('context_gate.','full_stem.extra_proj.')) for n in train_keys-old_keys)):raise ValueError('Unexpected warmstart key mismatch')
  with torch.no_grad():
   for n,p in model.named_parameters():
    if n in old_keys:p.copy_(state['trainable'][n].to(p))
  if a.warmstart and train_keys-old_keys:
   with torch.no_grad():
    current=model(xt,t,y);gate,stem=model.context_gate,model.full_stem
    try:
     model.context_gate=None;model.full_stem=None;prior=model(xt,t,y)
    finally:model.context_gate=gate;model.full_stem=stem
    torch.testing.assert_close(current,prior,rtol=1e-5,atol=1e-5)
    numerical['warmstart_prior_max_abs']=(current-prior).abs().max().item()
    numerical['warmstart_status']='PASS_CURRENT_PRETRAINED_TRAINED_FIXTURE'
    del current,prior
 elif model.angle_head is not None and a.phase=='train':
  g=torch.Generator(device='cuda').manual_seed(a.seed)
  with torch.no_grad():model.angle_head.bias.copy_(torch.randn(model.angle_head.bias.shape,device='cuda',generator=g)*.001)
 if a.gate_lr is not None and (model.context_gate is None or a.resume):raise ValueError('Gate LR diagnostic requires gated mode and a fresh optimizer')
 params=[p for p in model.parameters() if p.requires_grad]
 if a.gate_lr is not None:
  gate=list(model.context_gate.parameters());gate_ids={id(p) for p in gate};params=[{'params':[p for p in params if id(p) not in gate_ids],'lr':a.lr},{'params':gate,'lr':a.gate_lr}]
 opt=torch.optim.AdamW(params,lr=a.lr,betas=(.9,.999),weight_decay=0.)
 if a.resume:opt.load_state_dict(state['optimizer'])
 times=[];losses=[];sample_times=[];start_step=state.get('step',0) if a.resume or a.warmstart else 0;count=4 if a.phase=='resource' else a.steps
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
 training_wall_start=time.perf_counter()
 for step in range(start_step,start_step+count):
  if a.phase=='train':
   ids=[rows[(step*a.batch+i)%len(rows)] for i in range(a.batch)];xt,t,y,target=fixture(ids,a.seed+step)
   sample_times.extend(t.detach().cpu().tolist())
  opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
  loss=(model(xt,t,y)-target).square().mean();loss.backward();opt.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
  times.append(elapsed);losses.append(loss.item())
  if step%8==0:print(json.dumps({'mode':a.mode,'step':step,'loss':loss.item(),'update_seconds':elapsed}),flush=True)
 training_wall_seconds=time.perf_counter()-training_wall_start
 trainable={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad} if a.phase=='train' else {}
 if a.phase=='train':torch.save({'mode':a.mode,'step':start_step+count,'trainable':trainable,'optimizer':opt.state_dict()},a.out/'resume.pt')
 report={'phase':a.phase,'mode':a.mode,'numerical':numerical,'torch':torch.__version__,'precision':'FP32','tf32':False,'image_side':1024,'semantic_tokens':1024,'patch_positions':1024,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'first_update_seconds':times[0],'warmed_update_seconds':times[1:],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'losses':losses,'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'timing_scope':'Complete semantic+pixel forward/backward/AdamW GPU update; excludes image IO/text-cache IO. Shared GPU, not isolated performance evidence.','seed':a.seed,'evaluation_seed':2027}
 report.update(parameter_group_lrs=[g['lr'] for g in opt.param_groups],training_loop_wall_seconds=training_wall_seconds,training_times=sample_times,time_sampling=a.time_sampling,butterfly_backend=a.butterfly_backend,optimizer_restarted=bool(a.warmstart),initial_state=str(a.resume or a.warmstart or a.checkpoint),adapter_source_sha256=adapter_source_sha256)
 if a.phase=='train':
  model.eval();scores=[]
  with torch.no_grad():
   for index,row in enumerate(val):
    for j,tv in enumerate((.2,.5,.8)):
     xx,tt,yy,tgt=fixture([row],2027+row['image_id']*3+j,tv);mse=(model(xx,tt,yy)-tgt).square().mean().item();scores.append({'image_id':row['image_id'],'t':tv,'tau':(1-tv)/tv,'velocity_mse':mse})
    if index%16==15:print(json.dumps({'mode':a.mode,'stage':'validation','images_done':index+1,'images_total':len(val)}),flush=True)
  report['heldout_paired_scores']=scores
 (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('losses','heldout_paired_scores')},indent=2),flush=True)
if __name__=='__main__':main()
