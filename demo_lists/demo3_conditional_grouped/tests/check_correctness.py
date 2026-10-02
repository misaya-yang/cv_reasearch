import copy,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'assets/source/PixNerd')]
import torch
from cgd.butterfly import butterfly,group_indices
from cgd.model import GroupedPixNerd,MODES
from src.models.transformer.pixnerd_t2i_heavydecoder import PixNerDiT

torch.set_num_threads(2);torch.manual_seed(2027);rows=[]
for p in (4,16,64,1024):
 x=torch.randn(2,3,p,3,dtype=torch.float64);theta=torch.randn(2,3,p.bit_length()-1,p//2,dtype=torch.float64)*.2
 z=butterfly(x,theta);back=butterfly(z,theta,True)
 inv=(back-x).abs().max().item();norm=(z.square().sum((-2,-1))-x.square().sum((-2,-1))).abs().max().item()
 assert inv<1e-12 and norm<1e-10
 for c in range(3):assert torch.equal(z[...,c],butterfly(x[...,c:c+1].expand_as(x),theta)[...,0])
 rows.append({'check':'orthogonal','P':p,'inverse_max':inv,'norm_error':norm})
x=torch.randn(1,4,3,dtype=torch.double,requires_grad=True);theta=torch.randn(1,2,2,dtype=torch.double,requires_grad=True)*.1
assert torch.autograd.gradcheck(lambda a,t:butterfly(butterfly(a,t).tanh(),t,True),(x,theta),eps=1e-6,atol=1e-5)
rows.append({'check':'forward_inverse_angle_gradcheck','pass':True})
for g in (2,32):assert torch.equal(group_indices(64,g,'cpu')[:,0],torch.arange(64))
base=PixNerDiT(in_channels=3,patch_size=2,hidden_size=16,num_groups=2,decoder_hidden_size=8,num_encoder_blocks=1,num_decoder_blocks=1,num_text_blocks=1,txt_embed_dim=8,txt_max_length=4)
# A nonzero field is essential: official random initialization zeroes the final head.
torch.nn.init.normal_(base.final_layer.linear.weight,std=.15);torch.nn.init.normal_(base.final_layer.linear.bias,std=.02)
base.decoder_patch_scaling_h=4;base.decoder_patch_scaling_w=4
x=torch.randn(2,3,16,16);t=torch.tensor([.2,.8]);y=torch.randn(2,4,8)
with torch.no_grad():ref=base(x,t,y)
for mode in MODES:
 model=GroupedPixNerd(copy.deepcopy(base),side=8,mode=mode)
 out=model(x,t,y);err=(out-ref).abs().max().item();assert err<2e-6,(mode,err)
 model.zero_grad();out.square().mean().backward()
 if model.angle_head is not None:
  grad=model.angle_head.weight.grad.abs().max().item();assert grad>0 and torch.isfinite(model.angle_head.weight.grad).all()
 elif model.full_stem is not None:
  grad=model.full_stem.extra_proj.weight.grad.abs().max().item();assert grad>0
 else:grad=None
 assert all(p.grad is None for p in model.base.blocks[:base.num_encoder_blocks].parameters())
 rows.append({'check':'official_tiny_heavy_interface','mode':mode,'identity_max':err,'new_path_gradient_max':grad})
result={'status':'PASS_CPU_RANDOM_MODULE_ONLY','torch':torch.__version__,'checkpoint_loaded':False,'rows':rows,'limits':'No pretrained, GPU, quality, throughput or SOTA claim.'}
(ROOT/'results').mkdir(exist_ok=True);(ROOT/'results/cpu_correctness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
