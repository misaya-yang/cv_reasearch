"""Paired rollout inspection with the vendor AdamLM solver, not a quality benchmark."""
from gpu_guard import require_gpu_unpaused
require_gpu_unpaused(__file__)
import argparse,gc,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'assets/source/PixNerd')]
import torch
from PIL import Image
from cgd.loading import load_pretrained
from cgd.model import GroupedPixNerd
from src.diffusion.flow_matching.adam_sampling import AdamLMSampler
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.base.guidance import simple_guidance_fn

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--images',type=int,default=4);p.add_argument('--only',choices=['pretrained','neighborhood','neighborhood_gated']);p.add_argument('--pair',action='store_true');p.add_argument('--guidance',type=float,default=4.);p.add_argument('--indices',type=int,nargs='+');p.add_argument('--cfg-gate',choices=['independent','shared_uncond'],default='independent');p.add_argument('--manifest',type=pathlib.Path,default=ROOT/'assets/data/coco_pilot_v1/manifest.json');p.add_argument('--static-adapter',type=pathlib.Path,default=ROOT/'results/context_gate_v1/neighborhood/resume.pt');p.add_argument('--gate-adapter',type=pathlib.Path,default=ROOT/'results/context_gate_v1/neighborhood_gated/resume.pt');a=p.parse_args()
 if a.only and a.pair:p.error('Choose a single arm or the matched pair')
 if (a.out/'report.json').exists():raise ValueError('Refuse to overwrite completed generation')
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 a.out.mkdir(parents=True,exist_ok=True);dr=a.manifest.parent
 rows=[r for r in json.loads(a.manifest.read_text())['rows'] if r['split']=='val'][:a.images]
 if len(rows)!=a.images:raise ValueError('Insufficient predeclared captions')
 selected=list(enumerate(rows))
 if a.indices is not None:
  if len(a.indices)!=len(set(a.indices)) or any(i<0 or i>=a.images for i in a.indices):raise ValueError('Invalid subset indices')
  selected=[(i,row) for i,row in selected if i in a.indices]
 empty=ROOT/'assets/data/coco_pilot_v1/text_cache/empty_caption.pt'
 if not empty.exists():
  from transformers import Qwen3Model,Qwen2Tokenizer
  weights=ROOT/'assets/checkpoints/qwen3';tok=Qwen2Tokenizer.from_pretrained(weights,local_files_only=True,padding_side='right')
  q=Qwen3Model.from_pretrained(weights,local_files_only=True,torch_dtype=torch.bfloat16).eval().cuda()
  tokens=tok([''],truncation=True,max_length=128,padding='max_length',return_tensors='pt')
  with torch.no_grad():embedding=q(**{k:v.cuda() for k,v in tokens.items()}).last_hidden_state[0].cpu().clone()
  torch.save({'embedding':embedding,'caption':''},empty);del q;gc.collect();torch.cuda.empty_cache()
 uncond=torch.load(empty,weights_only=True)['embedding'][None].cuda().float()
 sampler=AdamLMSampler(scheduler=LinearScheduler(),guidance_fn=simple_guidance_fn,num_steps=25,guidance=a.guidance,timeshift=3.,order=2)
 arms=[('pretrained','original',None),('neighborhood','neighborhood',a.static_adapter),('neighborhood_gated','neighborhood_gated',a.gate_adapter)]
 if a.pair:arms=[arm for arm in arms if arm[0]!='pretrained']
 if a.only:arms=[arm for arm in arms if arm[0]==a.only]
 report={'precision':'FP32','tf32':False,'solver':'Unmodified vendor AdamLMSampler._impl_sampling; bypass BF16 forward decorator uniformly','steps':25,'cfg':a.guidance,'cfg_gate':a.cfg_gate,'selected_indices':[i for i,row in selected],'timeshift':3.,'order':2,'image_side':1024,'status':'PAIRED_VISUAL_DIAGNOSTIC_ONLY','rows':[],'warning':'Pretrained vs finetuned comparison includes training. Neighbor vs gate uses matched training. No FID/GenEval/SOTA claim.'}
 for name,mode,checkpoint in arms:
  base=load_pretrained(ROOT/'assets/checkpoints/pixnerd/model.ckpt').cuda().float();model=GroupedPixNerd(base,mode=mode,side=32,cfg_gate=a.cfg_gate if mode=='neighborhood_gated' else 'independent').cuda().eval()
  if checkpoint:
   state=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
   if set(state['trainable'])!={n for n,v in model.named_parameters() if v.requires_grad}:raise ValueError('Adapter contract mismatch')
   with torch.no_grad():
    for n,v in model.named_parameters():
     if n in state['trainable']:v.copy_(state['trainable'][n].to(v))
   del state
  for i,row in selected:
   condition=torch.load(dr/'text_cache'/(str(row['image_id'])+'.pt'),weights_only=True)['embedding'][None].cuda().float()
   noise=torch.randn((1,3,1024,1024),generator=torch.Generator().manual_seed(31000+i)).cuda()
   torch.cuda.synchronize();start=time.perf_counter()
   with torch.no_grad():trajectory,_=sampler._impl_sampling(model,noise,condition,uncond);output=trajectory[-1]
   torch.cuda.synchronize();elapsed=time.perf_counter()-start
   if not torch.isfinite(output).all():raise RuntimeError(f'Nonfinite rollout: {name} {i}')
   pixels=((output[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
   filename=f'{name}_{i}_{row["image_id"]}.png';Image.fromarray(pixels).save(a.out/filename)
   report['rows'].append({'arm':name,'adapter':str(checkpoint) if checkpoint else 'official EMA pretrained','image_id':row['image_id'],'caption':row['caption'],'seed':31000+i,'file':filename,'seconds':elapsed,'output_rms':output.square().mean().sqrt().item(),'clipped_fraction':(output.abs()>1).float().mean().item()})
   print(json.dumps(report['rows'][-1]),flush=True);del trajectory,output
  del model,base;gc.collect();torch.cuda.empty_cache()
 (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
