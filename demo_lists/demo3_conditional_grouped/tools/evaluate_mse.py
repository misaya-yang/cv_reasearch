"""Read-only paired evaluation on predeclared image IDs, no optimizer or checkpoint writes."""
from gpu_guard import require_gpu_unpaused
require_gpu_unpaused(__file__)
import argparse,hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'assets/source/PixNerd')]
import torch
from PIL import Image
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize
from cgd.loading import load_pretrained
from cgd.model import GroupedPixNerd,MODES
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--adapter',type=pathlib.Path,required=True);p.add_argument('--data',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--native-crop',action='store_true');p.add_argument('--save-reconstructions',type=int,default=0);p.add_argument('--gate-envelope',choices=['none','sigma','reliability'],default='none');p.add_argument('--gate-placement',choices=['aligned','shifted','image_mean'],default='aligned');a=p.parse_args()
 if a.out.exists():raise ValueError('Refuse to overwrite previous evaluation')
 source_sha256=hashlib.sha256((ROOT/'cgd/model.py').read_bytes()).hexdigest()
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rows=[r for r in json.loads(a.data.read_text())['rows'] if r['split']=='val'];assert rows
 state=torch.load(a.adapter,weights_only=True,map_location='cpu',mmap=True);m=GroupedPixNerd(load_pretrained(ROOT/'assets/checkpoints/pixnerd/model.ckpt').cuda().float(),mode=a.mode,side=32,gate_placement=a.gate_placement,gate_envelope=a.gate_envelope).cuda().eval()
 keys={n for n,v in m.named_parameters() if v.requires_grad}
 if keys!=set(state['trainable']):raise ValueError('Adapter contract mismatch')
 with torch.no_grad():
  for n,v in m.named_parameters():
   if n in keys:v.copy_(state['trainable'][n].to(v))
 step=state['step'];del state;pre=Compose(([CenterCrop(1024)] if a.native_crop else [Resize(1024),CenterCrop(1024)])+[ToTensor(),Normalize((.5,)*3,(.5,)*3)]);scores=[];torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 with torch.no_grad():
  for index,row in enumerate(rows):
   with Image.open(a.data.parent/row['path']) as im:
    if a.native_crop and min(im.size)<1024:raise ValueError('Native crop requires both dimensions >=1024')
    x=pre(im.convert('RGB'))[None].cuda()
   cache=torch.load(a.data.parent/'text_cache'/(str(row['image_id'])+'.pt'),weights_only=True)
   if cache['caption_sha256']!=hashlib.sha256(row['caption'].encode()).hexdigest():raise ValueError('Caption cache mismatch')
   y=cache['embedding'][None].cuda().float()
   def save_image(tensor,name):
    folder=a.out.parent/(a.out.stem+'_images');folder.mkdir(parents=True,exist_ok=True);pixels=((tensor[0].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy();Image.fromarray(pixels).save(folder/name)
   if index<a.save_reconstructions:save_image(x,str(row['image_id'])+'_target.png')
   for j,tv in enumerate((.2,.5,.8)):
    g=torch.Generator().manual_seed(2027+row['image_id']*3+j);torch.randn(1,generator=g);noise=torch.randn(x.shape,generator=g).cuda();t=torch.tensor([tv],device='cuda');target=x-noise;pred=m(tv*x+(1-tv)*noise,t,y);error=pred-target
    if index<a.save_reconstructions:save_image(tv*x+(1-tv)*noise+(1-tv)*pred,str(row['image_id'])+f'_t{tv}.png')
    if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite output')
    dx=error[:,:,:,1:]-error[:,:,:,:-1];dy=error[:,:,1:,:]-error[:,:,:-1,:]
    scores.append({'image_id':row['image_id'],'t':tv,'velocity_mse':error.square().mean().item(),'error_gradient_mse':.5*(dx.square().mean()+dy.square().mean()).item()})
   if index%16==15:print(json.dumps({'mode':a.mode,'images_done':index+1}),flush=True)
 torch.cuda.synchronize();report={'phase':'evaluation_only','mode':a.mode,'adapter':str(a.adapter),'step':step,'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'source_sha256':source_sha256,'gate_placement':a.gate_placement,'gate_envelope':a.gate_envelope,'native_crop':a.native_crop,'save_reconstructions':a.save_reconstructions,'precision':'FP32','tf32':False,'evaluation_seed':2027,'heldout_paired_scores':scores,'wall_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'limits':'Read-only velocity/error-gradient diagnostic. Native-crop photos use identical empty text conditioning. Posthoc interventions reuse scored images. No generation/perceptual/SOTA claim.'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
