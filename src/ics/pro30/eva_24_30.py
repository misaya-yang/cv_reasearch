"""Bound current timm Eva operators for Pro30, without model construction.

Actual model, tensors, weights and native RoPE come from an explicit existing
InternalEncoder. Missing assets raise, never synthetic replacement on deployment.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import threading
import time
import types
import numpy as np
from . import common

_LOCK=threading.RLock()


def phase_neutral_attention(q,k,v,q_rotated,k_rotated,prefix,*,mask=None,causal=False):
    """Exact original four-interaction formula through one 3d SDPA."""
    import torch
    import torch.nn.functional as F
    if not 0<prefix<q.shape[-2] or q.shape!=k.shape or q.shape!=q_rotated.shape or k.shape!=k_rotated.shape:
        raise ValueError('Aligned post-QKnorm native Q/K and prefix IDs required')
    qg,qp=q[...,:prefix,:],q[...,prefix:,:];kg,kp=k[...,:prefix,:],k[...,prefix:,:]
    qnew=torch.cat((torch.cat((torch.zeros_like(qg),qg,qg),-1),torch.cat((q_rotated[...,prefix:,:],qp,torch.zeros_like(qp)),-1)),-2)
    knew=torch.cat((torch.cat((torch.zeros_like(kg),kg,torch.zeros_like(kg)),-1),torch.cat((k_rotated[...,prefix:,:],torch.zeros_like(kp),kp),-1)),-2)
    # Explicit scale is 1/sqrt(d), NEVER SDPA's 1/sqrt(3d).
    return F.scaled_dot_product_attention(qnew,knew,v,attn_mask=mask,is_causal=causal,dropout_p=0.,scale=q.shape[-1]**-.5)


def explicit_phase_logits(q,k,q_rotated,k_rotated,prefix):
    logits=q@k.transpose(-1,-2)
    logits[...,prefix:,prefix:]=q_rotated[...,prefix:,:]@k_rotated[...,prefix:,:].transpose(-1,-2)
    return logits*q.shape[-1]**-.5


def audit_observed_qk(observation,*,chunk=32):
    """Bounded real-QK certificate; all native keys, tiled query rows."""
    import torch
    import torch.nn.functional as F
    q,k,v=(observation[name] for name in ('q','k','v'))
    qr,kr=observation['q_rotated'],observation['k_rotated'];prefix=observation['prefix'];n=q.shape[-2];d=q.shape[-1]
    qnew=torch.cat((torch.cat((torch.zeros_like(q[...,:prefix,:]),q[...,:prefix,:],q[...,:prefix,:]),-1),torch.cat((qr[...,prefix:,:],q[...,prefix:,:],torch.zeros_like(q[...,prefix:,:])),-1)),-2)
    knew=torch.cat((torch.cat((torch.zeros_like(k[...,:prefix,:]),k[...,:prefix,:],torch.zeros_like(k[...,:prefix,:])),-1),torch.cat((kr[...,prefix:,:],torch.zeros_like(k[...,prefix:,:]),k[...,prefix:,:]),-1)),-2)
    logits_error=0.;output_error=0.;start=time.perf_counter()
    with torch.no_grad():
        for first in range(0,n,chunk):
            last=min(n,first+chunk);direct=q[...,first:last,:]@k.transpose(-1,-2)
            pfirst=max(first,prefix)
            if last>pfirst:direct[...,pfirst-first:,prefix:]=qr[...,pfirst:last,:]@kr[...,prefix:,:].transpose(-1,-2)
            direct=direct*d**-.5;expanded=(qnew[...,first:last,:]@knew.transpose(-1,-2))*d**-.5
            logits_error=max(logits_error,float((direct-expanded).abs().max()))
            expected=direct.softmax(-1)@v
            actual=F.scaled_dot_product_attention(qnew[...,first:last,:],knew,v,dropout_p=0.,scale=d**-.5)
            output_error=max(output_error,float((actual-expected).abs().max()))
    return dict(prefix_tokens=prefix,head_dimension=d,token_count=n,tile_rows=chunk,all_keys_in_softmax=True,
                four_interaction_max_logit_error=logits_error,SDPA_max_output_error=output_error,
                explicit_scale=d**-.5,default_3d_scale_is_wrong=True,operator_audit_seconds=time.perf_counter()-start)


def matched_prefix_factor(q,k,prefix,target,*,chunk=64):
    """Match actual total cross-prefix softmax mass using ONLY attenuation.

    The target is an observed same-role/same-layer neutral-encoder mass, not a
    guessed .8. If attenuation cannot match it, this control is unavailable.
    """
    import torch
    ratios=[];n=q.shape[-2]
    with torch.no_grad():
        for first in range(0,n,chunk):
            last=min(n,first+chunk);logits=(q[...,first:last,:]@k.transpose(-1,-2))*q.shape[-1]**-.5
            for row in range(first,last):
                value=logits[...,row-first,:]
                cross=value[...,prefix:] if row<prefix else value[...,:prefix]
                same=value[...,:prefix] if row<prefix else value[...,prefix:]
                ratios.append(torch.logsumexp(cross,-1)-torch.logsumexp(same,-1))
        ratio=torch.stack(ratios,-1).reshape(-1)
        mass=lambda logfactor:float(torch.sigmoid(ratio+logfactor).sum())
        upper=mass(0.);lower=mass(np.log(1e-8));tolerance=1e-5*max(1.,target)
        if target>upper+tolerance or target<lower-tolerance:
            raise common.ArtifactUnavailable('Observed neutral prefix mass cannot be matched by positive attenuation alone')
        lo,hi=np.log(1e-8),0.
        for _ in range(40):
            mid=(lo+hi)/2
            if mass(mid)>target:hi=mid
            else:lo=mid
        factor=float(np.exp((lo+hi)/2))
        return factor,dict(native_cross_prefix_mass=upper,target_cross_prefix_mass=float(target),matched_cross_prefix_mass=mass(np.log(factor)),factor=factor,tolerance=tolerance)


def asset_header(path):
    """Read only safetensors JSON header; no tensor load or external fetch."""
    path=Path(path)
    if not path.is_file():return dict(state='unavailable',reason='checkpoint_absent',path=str(path))
    with path.open('rb') as stream:
        size=int.from_bytes(stream.read(8),'little')
        if size<=0 or size>100_000_000:raise ValueError('Invalid safetensors header')
        header=json.loads(stream.read(size))
    keys=[key for key in header if key!='__metadata__']
    mask_keys=[key for key in keys if key.split('.')[-1]=='mask_token']
    return dict(state='header_observed',path=str(path),mask_token_keys=mask_keys,
                mask_token_descriptors={k:header[k] for k in mask_keys},tensor_count=len(keys),trained_token_supported=bool(mask_keys),
                training_provenance_verified=False)


class EvaRuntime:
    """Explicit adapter around an already loaded real frozen CPU encoder."""
    def __init__(self, internal, ep):
        self.internal=internal;self.model=internal.model;self.torch=internal.torch
        self.binding=dict(internal.binding);self.check=internal.forward.check_frozen
        from ics.cpu100.encoder import assert_matches_producer
        assert_matches_producer(ep.producer,self.binding)
        if self.binding.get('execution_kind')!='real_frozen_cpu_model' or len(self.model.blocks)!=24 or self.model.num_prefix_tokens!=5:
            raise common.ArtifactUnavailable('Actual frozen DINOv3-L24 five-prefix CPU Eva required')
        if type(self.model).__module__!='timm.models.eva' or type(self.model).__name__!='Eva':
            raise common.ArtifactUnavailable('Current actual timm Eva class is required')
        if any(m.training for m in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()):
            raise common.ArtifactUnavailable('All deployed model weights must be frozen and eval')
        self.check();self.stats=dict(full_forwards=0,suffix_forwards=0,VJPs=0,suffix_backward_units=0,mask_only_backward_calls=0,full_forward_seconds=0.,suffix_forward_seconds=0.,VJP_seconds=0.,device='cpu',actual_model_execution=True)
        self.sessions={};self.audit={};self.ep=ep

    @contextmanager
    def attention(self,*,mode='native',gates=None,prefix_rope=None,audit=False,head_direction=None):
        torch=self.torch;import torch.nn.functional as F
        old=[]
        def replacement(index,attn):
            # Bind the REAL source callable, including the producer's half layout.
            apply_rope=attn.forward.__func__.__globals__.get('apply_rot_embed_cat')
            if not callable(apply_rope):raise common.ArtifactUnavailable('Native Eva apply_rot_embed_cat not bound')
            if attn.qkv is None or attn.q_bias is not None or attn.v_bias is not None or attn.gate is not None or not isinstance(attn.norm,torch.nn.Identity):
                raise common.ArtifactUnavailable('Pinned bias-free fused QKV/no inner norm/no gate attention contract changed')
            def forward(module,x,rope=None,attn_mask=None,is_causal=False):
                batch,n,dim=x.shape
                q,k,v=module.qkv(x).reshape(batch,n,3,module.num_heads,-1).permute(2,0,3,1,4).unbind(0)
                q,k=module.q_norm(q),module.k_norm(k);prefix=int(module.num_prefix_tokens)
                if prefix!=5:raise common.ArtifactUnavailable('Native five-prefix IDs changed')
                qr,kr=q,k
                if rope is not None and mode!='rope_off':
                    qr=torch.cat((q[...,:prefix,:],apply_rope(q[...,prefix:,:],rope,half=module.rotate_half)),-2).type_as(v)
                    kr=torch.cat((k[...,:prefix,:],apply_rope(k[...,prefix:,:],rope,half=module.rotate_half)),-2).type_as(v)
                    if mode=='vit5_prefix':
                        if prefix_rope is None:raise common.ArtifactUnavailable('Explicit independently positioned prefix RoPE control asset required')
                        qr=torch.cat((apply_rope(q[...,:prefix,:],prefix_rope,half=module.rotate_half),qr[...,prefix:,:]),-2)
                        kr=torch.cat((apply_rope(k[...,:prefix,:],prefix_rope,half=module.rotate_half),kr[...,prefix:,:]),-2)
                if mode=='neutral':output=phase_neutral_attention(q,k,v,qr,kr,prefix,mask=attn_mask,causal=is_causal)
                else:
                    mask=attn_mask
                    if mode=='prefix_attenuation':
                        if attn_mask is not None or is_causal:raise common.ArtifactUnavailable('The fixed original unmasked Eva prefix-mass control requires its original full key set')
                        if not hasattr(self,'prefix_mass_targets'):raise common.ArtifactUnavailable('Actual neutral per-layer prefix masses required for the matched attenuation control')
                        factor,certificate=matched_prefix_factor(qr,kr,prefix,float(self.prefix_mass_targets[index-1]))
                        self.prefix_mass_certificates[index]=certificate
                        ids=torch.arange(n,device=q.device);cross=(ids[:,None]<prefix)^(ids[None,:]<prefix)
                        bias=torch.where(cross,x.new_tensor(np.log(factor)),x.new_tensor(0.))
                        if mask is not None:
                            bias=bias.masked_fill(~mask,-float('inf')) if mask.dtype==torch.bool else bias+mask
                        mask=bias
                    output=F.scaled_dot_product_attention(qr,kr,v,attn_mask=mask,is_causal=is_causal,dropout_p=0.,scale=module.scale)
                if audit and index==1:
                    # Only first-layer token-bounded audit; real raw tensors are retained.
                    self.audit=dict(q=q.detach().clone(),k=k.detach().clone(),v=v.detach().clone(),q_rotated=qr.detach().clone(),k_rotated=kr.detach().clone(),prefix=prefix,scale=float(module.scale),mode=mode)
                if head_direction is not None and index>=21:
                    # Read actual AV before concat/proj/bias. Each head's scalar
                    # response is its actual projection contribution along w.
                    projected=module.proj.weight.T@head_direction
                    projected=projected.reshape(module.num_heads,-1)
                    response=(output[:,:,prefix:,:]*projected[None,:,None,:]).sum(-1)
                    self._head_observations[index]=dict(response=response.detach().clone(),energy=(output[:,:,prefix:,:]**2).mean(dim=(0,2,3)).detach().clone())
                if gates is not None and index>=21:
                    output=output*gates[index-21][None,:,None,None]
                output=output.transpose(1,2).reshape(batch,n,dim)
                return module.proj_drop(module.proj(module.norm(output)))
            return forward
        with _LOCK:
            try:
                for index,block in enumerate(self.model.blocks,1):
                    attn=block.attn;old.append((attn,attn.forward));attn.forward=types.MethodType(replacement(index,attn),attn)
                yield
            finally:
                for attn,forward in old:attn.forward=forward

    def native(self,role,*,need_gradient=False,mode='native',audit=False,prefix_rope=None,gates=None,head_direction=None):
        # Gradient graph starts at a detached H20 leaf; first20 are not differentiated.
        key=(role,need_gradient,mode)
        if key in self.sessions and not audit and gates is None and head_direction is None:return self.sessions[key]
        from ics.cpu100.common import rgb_view
        image=rgb_view(self.ep,role);torch=self.torch;captured={};hooks=[];self._head_observations={}
        def h20(module,args,output):
            leaf=output.detach().clone();leaf.requires_grad_(need_gradient);captured['h20']=leaf
            return leaf
        def save_rope(index):
            def save(module,args,kwargs):
                rope=kwargs.get('rope',args[1] if len(args)>1 else None)
                captured.setdefault('ropes',{})[index]=None if rope is None else rope.detach().clone()
            return save
        self.check();start=time.perf_counter()
        if mode=='prefix_attenuation':
            target,producer=common.require_artifacts(self.ep,'pro30_neutral_prefix_mass_targets','pro30_neutral_prefix_mass_producer')
            target=np.asarray(target,float)
            if (target.shape!=(2,24) or not np.isfinite(target).all() or np.any(target<0)
                    or producer.get('checkpoint_sha256')!=self.binding['model_assets']['checkpoint_sha256']
                    or producer.get('target_array_sha256')!=common.array_hash(np.asarray(common.artifact(self.ep,'pro30_neutral_prefix_mass_targets')))
                    or producer.get('query_unit_array_sha256')!=common.array_hash(self.ep.q)
                    or producer.get('reference_unit_array_sha256')!=common.array_hash(self.ep.r)
                    or producer.get('real_neutral_encoder_observation') is not True or producer.get('query_GT_read') is not False):
                raise common.ArtifactUnavailable('Actual neutral all24-layer R/Q prefix mass producer/content binding required')
            self.prefix_mass_targets=target[0 if role=='r' else 1];self.prefix_mass_certificates={}
        hooks.append(self.model.blocks[19].register_forward_hook(h20))
        for index in range(21,25):hooks.append(self.model.blocks[index-1].register_forward_pre_hook(save_rope(index),with_kwargs=True))
        try:
            scope=torch.enable_grad() if need_gradient else torch.no_grad()
            with scope,torch.autocast('cpu',enabled=False),self.attention(mode=mode,audit=audit,prefix_rope=prefix_rope,gates=gates,head_direction=head_direction):
                out=self.model.forward_features(self.internal._normalized(self.internal._input(image)))
            captured['raw']=out;captured['features']=F_unit(out[:,5:]);captured['role']=role;captured['heads']=dict(self._head_observations)
        finally:
            for hook in hooks:hook.remove()
            self.stats['full_forwards']+=1;self.stats['full_forward_seconds']+=time.perf_counter()-start
        if mode=='native':
            actual=captured['features'][0].detach().numpy();expected=self.ep.r if role=='r' else self.ep.q
            if actual.shape!=expected.shape or not np.allclose(actual,expected,atol=2e-5,rtol=2e-5):
                raise common.ArtifactUnavailable('Native continuation input differs from sealed ep features')
        self.check();self.sessions[key]=captured;return captured

    def suffix(self,hidden,session,*,gates=None):
        torch=self.torch;self.check();start=time.perf_counter();x=hidden
        with torch.enable_grad(),torch.autocast('cpu',enabled=False),self.attention(gates=gates):
            for index in range(21,25):x=self.model.blocks[index-1](x,rope=session['ropes'][index])
            x=self.model.norm(x);x=F_unit(x[:,5:])
        self.stats['suffix_forwards']+=1;self.stats['suffix_forward_seconds']+=time.perf_counter()-start;self.check();return x

    def vjp(self,value,leaf,*,retain_graph=False,suffix_units=1):
        start=time.perf_counter();gradient=self.torch.autograd.grad(value,leaf,retain_graph=retain_graph,only_inputs=True)[0]
        self.stats['VJPs']+=1;self.stats['suffix_backward_units']+=suffix_units
        self.stats['mask_only_backward_calls']+=int(suffix_units==0)
        self.stats['VJP_seconds']+=time.perf_counter()-start;return gradient

    def encode(self,role,image,*,mask=None,mask_token=None):
        torch=self.torch;hooks=[]
        if mask is not None:
            if mask_token is None:raise common.ArtifactUnavailable('Actual trained mask-token required')
            selected=torch.as_tensor(mask,dtype=torch.bool,device='cpu').reshape(1,-1)
            token=torch.as_tensor(mask_token,dtype=torch.float32,device='cpu').reshape(1,1,-1)
            def replace(module,args,output):
                if output.ndim!=3 or output.shape[1]!=selected.shape[1]:raise common.ArtifactUnavailable('Current native patch_embed mask insertion layout differs')
                return torch.where(selected[...,None],token,output)
            hooks.append(self.model.patch_embed.register_forward_hook(replace))
        self.check();start=time.perf_counter()
        try:
            with torch.no_grad(),torch.autocast('cpu',enabled=False):
                raw=self.model.forward_features(self.internal._normalized(self.internal._input(image)))
            result=F_unit(raw[:,5:])[0].detach().numpy().copy()
        finally:
            for hook in hooks:hook.remove()
            self.stats['full_forwards']+=1;self.stats['full_forward_seconds']+=time.perf_counter()-start
        self.check();return result

    def trained_mask_token(self,ep):
        token,producer=common.require_artifacts(ep,'pro30_trained_mask_token','pro30_mask_token_producer')
        token=np.asarray(token)
        checkpoint=self.binding['model_assets']['checkpoint_sha256']
        if (not isinstance(producer,dict) or producer.get('checkpoint_sha256')!=checkpoint or producer.get('trained_mask_token') is not True
                or producer.get('token_array_sha256')!=common.array_hash(token) or not producer.get('actual_checkpoint_key')):
            raise common.ArtifactUnavailable('Same-checkpoint learned mask token content and training provenance required')
        if token.dtype!=np.float32 or token.size!=1024 or not np.isfinite(token).all():raise ValueError('Actual FP32 1024-dimensional trained mask token required')
        checkpoint_path=producer.get('actual_checkpoint_path')
        if checkpoint_path is None:raise common.ArtifactUnavailable('Actual same-checkpoint header is required to verify the token key')
        header=asset_header(checkpoint_path)
        if producer['actual_checkpoint_key'] not in header.get('mask_token_keys',[]):raise common.ArtifactUnavailable('Current checkpoint contains no declared trained mask-token asset')
        # Read the exact stored tensor; producer assertions cannot stand in for bytes.
        from safetensors import safe_open
        with safe_open(checkpoint_path,framework='np') as stored:actual=stored.get_tensor(producer['actual_checkpoint_key'])
        if not np.array_equal(actual.reshape(-1),token.reshape(-1)):raise common.ArtifactUnavailable('Supplied mask token differs from actual checkpoint tensor')
        h=hashlib.sha256()
        with Path(checkpoint_path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
        if h.hexdigest()!=checkpoint:raise common.ArtifactUnavailable('Mask token is not from the same deployed checkpoint')
        return token,dict(producer,header=header,actual_tensor_content_checked=True)


def F_unit(tensor):
    import torch.nn.functional as F
    return F.normalize(tensor,dim=-1)


def runtime(ep):
    try:bound=common.artifact(ep,'pro30_encoder_context')
    except common.ArtifactUnavailable:
        bound=common.artifact(ep,'pro30_internal_context')
        if hasattr(bound,'internal'):bound=bound.internal
    if isinstance(bound,EvaRuntime):
        if (bound.ep.source_id!=ep.source_id or common.array_hash(bound.ep.q)!=common.array_hash(ep.q)
                or common.array_hash(bound.ep.r)!=common.array_hash(ep.r)):
            raise common.ArtifactUnavailable('Runtime episode binding differs')
        return bound
    return EvaRuntime(bound,ep)
