"""m05/m10 observation interventions for a shared, frozen public-FoRIS context.
No encoder/model constructor, query labels, training, download or retained NxN.
New views are owned/counted by ctx.new_view; naive controls share every view.
"""
import math
from itertools import combinations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CARD05=[
 'Assumption: mutual reference-FG correspondences contain at least three geometrically consistent witnesses of query target scale/location, independently of native mask area.',
 'Prediction: correspondence-ROI mask improves original IoU over native and compute-matched native-box paired zoom; correspondence geometric residual falls within one native patch.',
 'Match: new observation placement/scale adds task evidence; cropping alone is controlled, not claimed novel.',
 'Mismatch: fewer than three inliers abstains; unreliable deformation/multiinstance correspondence or no advantage over native-box zoom rejects this fixed alignment, not repaired with queryGT.'
]
CARD10=[
 'Assumption: reference spatial-heldout Part1 FG/BG cosine-gap can monotonically predict patch coverage and transfer across images/phases; this is not whole-FoRIS score calibration.',
 'Prediction: the four shared phase observations yield better original IoU than their identically calibrated/anchored shift average; improvements concern boundary/holes, not identity errors.',
 'Match: regularized reconstruction uses interleaved sampling information beyond averaging; reference-only calibration is task inference, not independent calibration or a posterior.',
 'Mismatch: insufficient heldout FG/BG/range abstains; context dominates physical coverage or inverse fails to beat average, stop; box-mean Nyquist nullspace prevents an unconditional unique-superresolution claim.'
]


def _native(ctx):return ctx.native['mask'].bool()
def _score(ctx):return ctx.native['score'].float()
def _normal(x):
 x=x-x.min();return x/x.max().clamp_min(1e-6)
def _hw(ctx):return tuple(int(x) for x in ctx.work_hw)
def _tensor(x,device=None):return torch.as_tensor(x,device=device)
def _unit(x):return F.normalize(x.float(),dim=-1)
def _memo(ctx,key):
 if not hasattr(ctx,'_ten_observation_lowband'):ctx._ten_observation_lowband={}
 return ctx._ten_observation_lowband,key


def _budget_snapshot(ctx):
 b=getattr(ctx,'budget',None)
 if b is None:return None
 return dict(max_extra_b2=getattr(b,'max_extra_b2',None),extra_b2_calls=getattr(b,'extra_b2_calls',None),actual_b2_calls=getattr(b,'actual_b2_calls',None),native_b2_calls=getattr(b,'native_b2_calls',None),cached_view_keys=sorted(str(x) for x in getattr(b,'cache',{})))
def _budget_ledger(before,ctx,maximum):
 after=_budget_snapshot(ctx)
 delta=lambda key:None if before is None or after is None or before[key] is None or after[key] is None else after[key]-before[key]
 return dict(before=before,after=after,extra_requested_delta=delta('extra_b2_calls'),actual_n1B2_reported_delta=delta('actual_b2_calls'),max_extra_B2_this_method_and_naive=maximum,shared_global_cap8=True,actual_counter_semantics='root-driver forward counter; a zero uninstrumented tiny-factory counter is not proof of zero encoder calls')


def robust_similarity(source_xy,query_xy,patch=16):
 """Fixed pair-vote bank + one-patch consensus; no query mask/area input."""
 s=np.asarray(source_xy,float);q=np.asarray(query_xy,float)
 if len(s)<3 or s.shape!=q.shape:return None
 pairs=list(combinations(range(len(s)),2))
 if len(pairs)>96:pairs=[pairs[i] for i in np.linspace(0,len(pairs)-1,96).astype(int)]
 candidates=[]
 for a,b in pairs:
  ds=s[b]-s[a];dq=q[b]-q[a]
  if np.linalg.norm(ds)<patch or np.linalg.norm(dq)<patch:continue
  z=complex(*dq)/complex(*ds);A=np.array([[z.real,-z.imag],[z.imag,z.real]])
  t=q[a]-A@s[a];err=np.linalg.norm(s@A.T+t-q,axis=1);inside=err<=patch
  candidates.append((int(inside.sum()),-float(np.median(err[inside])) if inside.any() else -math.inf,a,b,A,t,inside))
 if not candidates:return None
 best=max(candidates,key=lambda x:x[:4]);inside=best[-1]
 if int(inside.sum())<3:return None
 ss=s[inside];qq=q[inside];sc=ss.mean(0);qc=qq.mean(0)
 den=np.square(ss-sc).sum()
 if den<=0:return None
 zs=(ss-sc)[:,0]+1j*(ss-sc)[:,1];zq=(qq-qc)[:,0]+1j*(qq-qc)[:,1]
 z=np.sum(zq*np.conj(zs))/den;A=np.array([[z.real,-z.imag],[z.imag,z.real]]);t=qc-A@sc
 scale=float(abs(z));err=np.linalg.norm(s@A.T+t-q,axis=1)
 if not np.isfinite(A).all() or scale<=0:return None
 return dict(A=A,t=t,scale=scale,inliers=inside,residual_median=float(np.median(err[inside])),
             hypothesis_count=len(candidates),angle=float(np.angle(z)))


def mutual_FG_matches(ctx,max_matches=256):
 r=_unit(ctx.ref_tokens);q=_unit(ctx.query_tokens);cov=ctx.reference_coverage.flatten().to(r.device)
 n=len(q);bestq=torch.empty(n,device=q.device,dtype=torch.long);bestv=torch.empty(n,device=q.device)
 bestr=torch.full((len(r),),-math.inf,device=r.device);bestback=torch.zeros(len(r),device=r.device,dtype=torch.long)
 margin=torch.empty(n,device=q.device);fg=cov>0
 if not fg.any() or not (~fg).any():return None
 for start in range(0,n,128):
  sim=q[start:start+128]@r.T;v,ix=sim.max(1);bestq[start:start+len(v)]=ix;bestv[start:start+len(v)]=v
  margin[start:start+len(v)]=sim[:,fg].max(1).values-sim[:,~fg].max(1).values
  rv,ri=sim.max(0);take=rv>bestr;bestr[take]=rv[take];bestback[take]=ri[take]+start
 i=torch.arange(n,device=q.device);valid=(bestback[bestq]==i)&fg[bestq]&(margin>0)
 ids=i[valid]
 if len(ids)<3:return None
 order=torch.argsort(margin[ids],descending=True,stable=True)[:max_matches];ids=ids[order];a=bestq[ids]
 H,W=_hw(ctx);h,w=ctx.reference_coverage.shape
 xy=lambda j:torch.stack(((j%w+.5)*(W/w),(j//w+.5)*(H/h)),1)
 return dict(source_xy=xy(a).cpu().numpy(),query_xy=xy(ids).cpu().numpy(),source_ids=a.cpu().numpy(),query_ids=ids.cpu().numpy(),margin=margin[ids].cpu().numpy())


def bbox(mask):
 a=np.asarray(mask,dtype=bool);yx=np.argwhere(a)
 if not len(yx):return None
 y0,x0=yx.min(0);y1,x1=yx.max(0)+1;return np.array([x0,y0,x1,y1],float)
def square_box(box,hw,margin=1.25,min_side=64):
 H,W=hw;cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
 side=min(max((box[2]-box[0])*margin,(box[3]-box[1])*margin,min_side),H,W)
 x0=min(max(cx-side/2,0),W-side);y0=min(max(cy-side/2,0),H-side)
 return np.array([x0,y0,x0+side,y0+side])
def original_box(box,pil,hw):
 H,W=hw;x0,y0,x1,y1=box
 return (max(0,int(math.floor(x0*pil.width/W))),max(0,int(math.floor(y0*pil.height/H))),
         min(pil.width,int(math.ceil(x1*pil.width/W))),min(pil.height,int(math.ceil(y1*pil.height/H))))
def work_box_from_original(box,pil,hw):
 H,W=hw;return np.array([box[0]*W/pil.width,box[1]*H/pil.height,box[2]*W/pil.width,box[3]*H/pil.height])
def crop_view(ctx,sbox,qbox):
 hw=_hw(ctx);sb=original_box(sbox,ctx.support_rgb,hw);qb=original_box(qbox,ctx.query_rgb,hw)
 mask=ctx.reference_mask_original
 mask=_tensor(mask).bool()[sb[1]:sb[3],sb[0]:sb[2]]
 if not mask.any() or sb[2]<=sb[0] or qb[2]<=qb[0] or sb[3]<=sb[1] or qb[3]<=qb[1]:return None,None
 if sb==(0,0,ctx.support_rgb.width,ctx.support_rgb.height) and qb==(0,0,ctx.query_rgb.width,ctx.query_rgb.height):return ctx,work_box_from_original(qb,ctx.query_rgb,hw)
 key='m05.S'+','.join(map(str,sb))+'.Q'+','.join(map(str,qb))
 nv=ctx.new_view(ctx.support_rgb.crop(sb),mask,ctx.query_rgb.crop(qb),key,
                 metadata=dict(method='m05',coordinate_contract='full original-image rectangular crops, actual integer bounds recorded; public square resize',source_original_box=sb,query_original_box=qb))
 return nv,work_box_from_original(qb,ctx.query_rgb,hw)
def compose_ROI(ctx,nv,box):
 H,W=_hw(ctx);x0,y0,x1,y1=[int(round(x)) for x in box];x0=max(0,x0);y0=max(0,y0);x1=min(W,x1);y1=min(H,y1)
 if x1<=x0 or y1<=y0:return _native(ctx).clone()
 if nv is ctx and (x0,y0,x1,y1)==(0,0,W,H):return _native(ctx).clone()
 pre=ctx.native['stages']['pre_refinement_mask'].bool().clone()
 local=nv.native['stages']['pre_refinement_mask'].float()
 pre[y0:y1,x0:x1]=F.interpolate(local[None,None],(y1-y0,x1-x0),mode='bilinear',align_corners=False)[0,0]>.5
 final=ctx.finalize_work_mask(pre).bool();outside=torch.ones((H,W),device=final.device,dtype=torch.bool);outside[y0:y1,x0:x1]=False
 final[outside]=_native(ctx)[outside];return final


def run_m05(ctx):
 cache,key=_memo(ctx,'m05')
 if key in cache:return cache[key]
 before=_budget_snapshot(ctx);base=_native(ctx);matches=mutual_FG_matches(ctx);hw=_hw(ctx)
 if matches is None:out=dict(mask=base,score=_score(ctx),stages={},audit=dict(abstain='fewer than3 legal mutual FG correspondences',new_views=0),controls={'native_box_zoom':base});out['audit'].setdefault('encoder_budget_ledger',_budget_ledger(before,ctx,2 if key=='m05' else 3));cache[key]=out;return out
 fit=robust_similarity(matches['source_xy'],matches['query_xy'],patch=hw[0]/ctx.reference_coverage.shape[0])
 if fit is None:out=dict(mask=base,score=_score(ctx),stages={},audit=dict(abstain='no3-point one-patch geometric consensus',new_views=0),controls={'native_box_zoom':base});out['audit'].setdefault('encoder_budget_ledger',_budget_ledger(before,ctx,2 if key=='m05' else 3));cache[key]=out;return out
 refbox=bbox(ctx.reference_mask.cpu().numpy())
 if refbox is None:raise ValueError('reference mask empty')
 sbox=square_box(refbox,hw,min_side=min(64,*hw));corners=np.array([[sbox[0],sbox[1]],[sbox[0],sbox[3]],[sbox[2],sbox[1]],[sbox[2],sbox[3]]])
 transformed=corners@fit['A'].T+fit['t'];b=np.r_[transformed.min(0),transformed.max(0)]
 qbox=square_box(b,hw,margin=1.,min_side=min(64,*hw))
 nv,actual=crop_view(ctx,sbox,qbox)
 if nv is None:out=dict(mask=base,score=_score(ctx),stages={},audit=dict(abstain='projected existing-asset crop invalid',new_views=0),controls={'native_box_zoom':base});out['audit'].setdefault('encoder_budget_ledger',_budget_ledger(before,ctx,2 if key=='m05' else 3));cache[key]=out;return out
 main=compose_ROI(ctx,nv,actual);native_box=bbox(base.cpu().numpy())
 naive=base
 if native_box is not None:
  qnaive=square_box(native_box,hw,min_side=min(64,*hw));nn,nb=crop_view(ctx,sbox,qnaive)
  if nn is not None:naive=compose_ROI(ctx,nn,nb)
 audit=dict(method='correspondence ROI observation',correspondence_count=len(matches['query_ids']),inlier_count=int(fit['inliers'].sum()),scale=fit['scale'],angle=fit['angle'],residual_median=fit['residual_median'],source_box=sbox.tolist(),query_box=qbox.tolist(),actual_query_box=actual.tolist(),no_native_area_for_main_scale=True,source_crop_same_for_naive=True,outside_ROI_preserved_exact=True,max_extra_B2=2,known_failure='mutual correspondences/rigid transform may be wrong under pose, reference multiinstance extent or ambiguous background; clipping reduces scale alignment',not_novel_by_itself=True)
 audit['encoder_budget_ledger']=_budget_ledger(before,ctx,2)
 out=dict(mask=main,score=_score(ctx),stages={},audit=audit,controls={'native_box_zoom':naive});out['audit'].setdefault('encoder_budget_ledger',_budget_ledger(before,ctx,2 if key=='m05' else 3));cache[key]=out;return out

def naive_m05(ctx):
 before=_budget_snapshot(ctx);z=run_m05(ctx);return dict(mask=z['controls']['native_box_zoom'],score=z['score'],stages={},audit=dict(compute='same two paired views as m05; cache/memo reused',control='nativebox query ROI, same source crop',encoder_budget_ledger=_budget_ledger(before,ctx,2)))


def shift_work(image,dx,dy):
 a=np.asarray(image);H,W=a.shape[:2];y=np.minimum(np.arange(H)+dy,H-1);x=np.minimum(np.arange(W)+dx,W-1)
 return a[y[:,None],x[None,:]].copy()
def valid_phase(hw,grid,dx,dy,device):
 H,W=hw;h,w=grid;y=(torch.arange(h,device=device)+1)*(H/h)+dy;x=(torch.arange(w,device=device)+1)*(W/w)+dx
 return (y[:,None]<=H)&(x[None,:]<=W)
def fit_iso(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);order=np.argsort(x,kind='stable');xx=x[order];yy=y[order]
 u,idx,count=np.unique(xx,return_index=True,return_counts=True);sums=np.add.reduceat(yy,idx);blocks=[]
 for j,(tot,n) in enumerate(zip(sums,count)):
  blocks.append([j,j,float(tot),int(n)])
  while len(blocks)>1 and blocks[-2][2]/blocks[-2][3]>blocks[-1][2]/blocks[-1][3]:
   b=blocks.pop();a=blocks.pop();blocks.append([a[0],b[1],a[2]+b[2],a[3]+b[3]])
 means=np.empty(len(u))
 for a,b,tot,n in blocks:means[a:b+1]=tot/n
 return u,means

def calibrated_observation(ctx,valid):
 r=_unit(ctx.ref_tokens);q=_unit(ctx.query_tokens);cov=ctx.reference_coverage.to(r.device).flatten();h,w=ctx.reference_coverage.shape
 usable=valid.flatten().to(r.device);fg=(cov>0)&usable;bg=(cov==0)&usable
 if not fg.any() or not bg.any():return None
 response=torch.full((len(r),),float('nan'),device=r.device);tile=min(8,max(1,min(h,w)//2));ys=torch.arange(h,device=r.device)[:,None];xs=torch.arange(w,device=r.device)[None,:]
 for y0 in range(0,h,tile):
  for x0 in range(0,w,tile):
   target=(ys>=y0)&(ys<y0+tile)&(xs>=x0)&(xs<x0+tile);exclude=(ys>=y0-2)&(ys<y0+tile+2)&(xs>=x0-2)&(xs<x0+tile+2)
   bank=usable&~exclude.flatten();Fbank=fg&bank;Bbank=bg&bank
   ids=torch.nonzero(target.flatten()&usable).flatten()
   if not len(ids) or not Fbank.any() or not Bbank.any():continue
   sim=r[ids]@r.T;response[ids]=sim[:,Fbank].max(1).values-sim[:,Bbank].max(1).values
 good=torch.isfinite(response);xx=response[good].cpu().numpy();yy=cov[good].cpu().numpy()
 if len(xx)<4 or not (yy>0).any() or not (yy==0).any() or float(yy.max()-yy.min())<1e-6:return None
 knots,means=fit_iso(xx,yy)
 if float(means.max()-means.min())<1e-6:return None
 query=[]
 for start in range(0,len(q),128):
  sim=q[start:start+128]@r.T;gap=sim[:,fg].max(1).values-sim[:,bg].max(1).values
  query.append(torch.as_tensor(np.interp(gap.cpu().numpy(),knots,means),device=q.device,dtype=torch.float32))
 out=torch.cat(query).reshape(h,w)
 return out,dict(reference_calibration='spatial-block+2patch-halo excluded from BOTH FG/BG banks; no self NN',point_response='native Part1 maxFGcos-maxBGcos, NOTwholeFoRIS score',source_valid_calibration_cells=int(good.sum()),source_coverage_range=[float(yy.min()),float(yy.max())],source_fit_MSE=float(np.mean((np.interp(xx,knots,means)-yy)**2)),source_constant_MSE=float(np.var(yy)),calibration_is_task_internal_not_independent=True,probability_posterior_claim=False)


def patch_mean(m,phase):
 dy,dx=phase;H,W=m.shape;h=(H-dy)//2;w=(W-dx)//2
 out=m.new_zeros(H//2,W//2);out[:h,:w]=F.avg_pool2d(m[dy:dy+2*h,dx:dx+2*w][None,None],2,2)[0,0];return out
def patch_adjoint(v,shape,phase):
 dy,dx=phase;H,W=shape;h=(H-dy)//2;w=(W-dx)//2;out=v.new_zeros(shape)
 out[dy:dy+2*h,dx:dx+2*w]=v[:h,:w].repeat_interleave(2,0).repeat_interleave(2,1)/4;return out

def inverse_coverage(obs,weights,phases,rgb,m0,core,lam=.02,eta=.1,steps=200):
 den=sum(patch_adjoint(w,m0.shape,p) for w,p in zip(weights,phases)).clamp_min(1e-6)
 avg=sum(patch_adjoint(w*c,m0.shape,p) for w,c,p in zip(weights,obs,phases))/den
 naive=((avg*den+eta*core*m0)/(den+eta*core)).clamp(0,1);m=naive.clone()
 wx=torch.exp(-20*(rgb[:,1:]-rgb[:,:-1]).square().sum(-1));wy=torch.exp(-20*(rgb[1:]-rgb[:-1]).square().sum(-1))
 for _ in range(steps):
  grad=2*sum(patch_adjoint(w*(patch_mean(m,p)-c),m.shape,p) for w,c,p in zip(weights,obs,phases))
  dx=m[:,1:]-m[:,:-1];dy=m[1:]-m[:-1]
  gx=wx*dx/(dx.square()+.01**2).sqrt();gy=wy*dy/(dy.square()+.01**2).sqrt()
  tv=torch.zeros_like(m);tv[:,:-1]-=gx;tv[:,1:]+=gx;tv[:-1]-=gy;tv[1:]+=gy
  grad+=lam*tv+2*eta*core*(m-m0);new=(m-.02*grad).clamp(0,1)
  if float((new-m).abs().max())<1e-5:m=new;break
  m=new
 return m,naive


def run_m10(ctx):
 cache,key=_memo(ctx,'m10')
 if key in cache:return cache[key]
 before=_budget_snapshot(ctx);hw=_hw(ctx);H,W=hw;base=_native(ctx);h,w=ctx.reference_coverage.shape
 if H//h!=16 or W//w!=16 or H%8 or W%8:raise ValueError('m10 requires actual16px full patch grid and8px latent alignment')
 phases_px=[(0,0),(8,0),(0,8),(8,8)];obs=[];weights=[];audits=[]
 base_valid=valid_phase(hw,(h,w),0,0,base.device);cal=calibrated_observation(ctx,base_valid)
 if cal is None:out=dict(mask=base,score=_score(ctx),stages={},audit=dict(abstain='spatial-heldout reference coverage not identifiable',extra_B2=0),controls={'phase_average':base});out['audit'].setdefault('encoder_budget_ledger',_budget_ledger(before,ctx,2 if key=='m05' else 3));cache[key]=out;return out
 obs.append(cal[0]);weights.append(base_valid.float());audits.append({**cal[1],'phase_xy':[0,0],'valid_complete_patch_count':int(base_valid.sum()),'grid_hw':[h,w],'work_hw':[H,W]})
 swork=ctx.support_rgb.convert('RGB').resize((W,H),Image.Resampling.BILINEAR);qwork=ctx.query_rgb.convert('RGB').resize((W,H),Image.Resampling.BILINEAR)
 source_mask=ctx.reference_mask.cpu().numpy()
 for dx,dy in phases_px[1:]:
  sm=torch.from_numpy(shift_work(source_mask,dx,dy))
  nv=ctx.new_view(Image.fromarray(shift_work(swork,dx,dy)),sm,Image.fromarray(shift_work(qwork,dx,dy)),'m10.phase.%d.%d'%(dx,dy),metadata=dict(method='m10',coordinate_contract='canonical working RGB shifted left/up; replicated RGB/mask padding; only complete physical16x16 patches valid',physical_origin_offset_xy=[dx,dy],source_query_same_shift=True))
  valid=valid_phase(hw,(h,w),dx,dy,nv.ref_tokens.device);c=calibrated_observation(nv,valid)
  if c is None:obs.append(obs[0].new_zeros(h,w));weights.append(obs[0].new_zeros(h,w));audits.append(dict(abstain='phase calibration unavailable',offset=[dx,dy]))
  else:obs.append(c[0]);weights.append(valid.float());audits.append({**c[1],'phase_xy':[dx,dy],'valid_complete_patch_count':int(valid.sum()),'grid_hw':[h,w],'work_hw':[H,W]})
  del nv
 phases=[(dy//8,dx//8) for dx,dy in phases_px];sn=_normal(_score(ctx));m0=F.interpolate(base[None,None].float().to(obs[0].device),(H//8,W//8),mode='area')[0,0]
 core=F.interpolate(((sn>=.8)|(sn<=.2))[None,None].float().to(m0.device),m0.shape,mode='nearest')[0,0]
 rgb=torch.as_tensor(np.asarray(qwork).copy(),device=m0.device).float()/255
 rgb=F.avg_pool2d(rgb.permute(2,0,1)[None],8,8)[0].permute(1,2,0)
 m,naive=inverse_coverage(obs,weights,phases,rgb,m0,core)
 make=lambda z:ctx.finalize_work_mask(F.interpolate(z[None,None],hw,mode='nearest')[0,0]>.5).bool()
 final=make(m);control=make(naive)
 audit=dict(method='4phase reference-calibrated coverage inverse',extra_B2=3,phases_xy=phases_px,phase_calibration=audits,lambda_TV=.02,eta_core=.1,recipe='fixed, no queryGT selection; HuberTV epsilon.01 projected gradient200 max',patch16_latent8=True,source_query_physical_validity_explicit=True,no_full_tokens_or_NxN_retained=True,calibration_field_NOTwholeFoRIS=True,posterior_claim=False,known_failure='cross-image coverage calibration/context effects can fail; 2x2 means have alternating-stripe/checkerboard nullspace; 8px latent threshold cannot recover arbitrary sub8px rods; priors do not certify identity',not_novel_by_itself=True)
 audit['encoder_budget_ledger']=_budget_ledger(before,ctx,3)
 out=dict(mask=final,score=F.avg_pool2d(m[None,None],2,2)[0,0],stages={'phase_average_coverage':F.avg_pool2d(naive[None,None],2,2)[0,0]},audit=audit,controls={'phase_average':control});out['audit'].setdefault('encoder_budget_ledger',_budget_ledger(before,ctx,2 if key=='m05' else 3));cache[key]=out;return out

def naive_m10(ctx):
 before=_budget_snapshot(ctx);z=run_m10(ctx);return dict(mask=z['controls']['phase_average'],score=z['stages'].get('phase_average_coverage',z['score']),stages={},audit=dict(control='same four phase fields, same reference calibration and core anchors, adjoint-weighted average',same_encoder_budget=True,encoder_budget_ledger=_budget_ledger(before,ctx,3)))

METHODS=[dict(id='m05',function=run_m05,naive=naive_m05,card=CARD05,card4lines=CARD05),dict(id='m10',function=run_m10,naive=naive_m10,card=CARD10,card4lines=CARD10)]
