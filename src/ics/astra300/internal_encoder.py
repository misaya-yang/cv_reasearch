"""Actual CPU frozen-model hooks and two-block replay, with lazy observations.

Native blocks execute unchanged. SDPA is observed through a scoped wrapper;
normalization, RoPE, bias, LayerScale, residuals and MLP stay in the model.
No checkpoint is downloaded and no final descriptor is substituted for a state.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
from pathlib import Path
import threading
import time

import numpy as np

from .common import ArtifactUnavailable, array_hash, readonly, unit

_LOCK = threading.RLock()


def _np(tensor):
    return readonly(tensor.detach().cpu().numpy())


def _tensor(torch, value):
    return torch.from_numpy(np.array(value, copy=True)).to(dtype=torch.float32, device='cpu')


@dataclass
class AttentionObservation:
    """Observed tensors include all prefix values and native softmax denominator."""
    q: object
    k: object
    v: object
    head_output: object
    q_pre_rope: object
    k_pre_rope: object
    projection_weight: object
    scale: float
    prefix: int
    block: int
    mask: object = None
    causal: bool = False

    def weights(self, rows=None, columns=None, *, chunk=64):
        """True attention; selecting patch columns never renormalizes the subset."""
        import torch
        n = self.q.shape[-2]
        rows = np.arange(n) if rows is None else np.asarray(rows, int)
        columns = np.arange(self.k.shape[-2]) if columns is None else np.asarray(columns, int)
        if np.any(rows < 0) or np.any(rows >= n) or np.any(columns < 0) or np.any(columns >= self.k.shape[-2]):
            raise ValueError('Attention row/column IDs outside native sequence')
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(rows), chunk):
                selected = torch.as_tensor(rows[start:start+chunk], dtype=torch.long)
                logits = (self.q.index_select(-2, selected) @ self.k.transpose(-2,-1)) * self.scale
                if self.mask is not None:
                    mask = self.mask
                    if mask.ndim >= 2 and mask.shape[-2] != 1:
                        mask = mask.index_select(-2, selected)
                    logits = logits.masked_fill(~mask, -float('inf')) if mask.dtype == torch.bool else logits + mask
                if self.causal:
                    keys = torch.arange(self.k.shape[-2]); allowed = keys[None,:] <= selected[:,None]
                    logits = logits.masked_fill(~allowed, -float('inf'))
                probability = logits.softmax(-1)
                outputs.append(_np(probability.index_select(-1, torch.as_tensor(columns, dtype=torch.long)))[0])
        return readonly(np.concatenate(outputs, axis=1))

    def patch_attention(self):
        ids = np.arange(self.prefix, self.q.shape[-2])
        return self.weights(ids, ids)

    def special_to_patch(self, *, mean_heads=True):
        value = self.weights(np.arange(self.prefix), np.arange(self.prefix, self.q.shape[-2]))
        return readonly(value.mean(0) if mean_heads else value)

    def patch_head_output(self):
        return _np(self.head_output[0,:,self.prefix:])

    def head_messages(self):
        """A_h V_h W_Oh, before output bias and block LayerScale, [N,H,D]."""
        import torch
        h, dh = self.head_output.shape[1], self.head_output.shape[-1]
        with torch.inference_mode():
            parts = [self.head_output[0,j,self.prefix:] @ self.projection_weight[:,j*dh:(j+1)*dh].T for j in range(h)]
            return _np(torch.stack(parts, dim=1))

    def edge_actions(self, edges):
        """Exact alpha_i,j W_h(v_j-o_i) for directed patch edges, [H,E,D]."""
        import torch
        edges = np.asarray(edges, int).reshape(-1,2)
        rows = edges[:,0] + self.prefix; columns = edges[:,1] + self.prefix
        if np.any(edges < 0) or np.any(rows >= self.q.shape[-2]) or np.any(columns >= self.k.shape[-2]):
            raise ValueError('Directed action patch IDs outside sequence')
        unique, inverse = np.unique(rows, return_inverse=True)
        # Gather each requested scalar from normalized rows, never patch-only softmax.
        probabilities = self.weights(unique, np.arange(self.k.shape[-2]))
        alpha = probabilities[:,inverse,columns]
        with torch.inference_mode():
            delta = self.v[0,:,columns] - self.head_output[0,:,rows]
            dh = delta.shape[-1]
            outputs = [(delta[j] @ self.projection_weight[:,j*dh:(j+1)*dh].T) * _tensor(torch,alpha[j])[:,None]
                       for j in range(delta.shape[0])]
            return _np(torch.stack(outputs))

    def edge_features(self, undirected_edges):
        """Three actual action features per layer: bidirectional lognorm/cos medians."""
        edges = np.asarray(undirected_edges,int).reshape(-1,2)
        forward = self.edge_actions(edges); backward = self.edge_actions(edges[:,::-1])
        fn = np.linalg.norm(forward,axis=-1); bn = np.linalg.norm(backward,axis=-1)
        cosine = np.divide(np.sum(forward*backward,axis=-1),fn*bn,
                           out=np.zeros_like(fn),where=fn*bn>1e-12)
        return readonly(np.c_[np.median(np.log(fn+1e-6),axis=0),
                               np.median(np.log(bn+1e-6),axis=0),np.median(cosine,axis=0)])


class CaptureSession:
    """Read-only native-hook session; usable independently for kernel fixtures."""
    def __init__(self, model, *, layers=(), attention_layers=(), before_layers=()):
        self.model = model
        self.layers = set(map(int,layers)); self.attention_layers = set(map(int,attention_layers))
        self.before_layers = set(map(int,before_layers))
        self.states = {}; self.before = {}; self.ropes = {}; self.attention = {}
        self.pre_final_ln = None; self._active = []; self._q_pre = {}; self._k_pre = {}

    @contextmanager
    def hooks(self):
        import torch
        import torch.nn.functional as functional
        handles = []; original_sdpa = functional.scaled_dot_product_attention
        def block_before(number):
            def save(module,args,kwargs):
                if number in self.before_layers:
                    self.before[number] = args[0].detach().clone()
                if number in self.before_layers or number in self.attention_layers:
                    rope = kwargs.get('rope', args[1] if len(args)>1 else None)
                    self.ropes[number] = None if rope is None else rope.detach().clone()
            return save
        def block_after(number):
            def save(module,args,output):
                if number in self.layers:
                    self.states[number] = output.detach().clone()
            return save
        def active_before(number):
            def save(module,args): self._active.append(number)
            return save
        def active_after(module,args,output):
            if self._active: self._active.pop()
        def qk_save(target,number):
            def save(module,args,output): target[number] = output.detach().clone()
            return save
        def norm_before(module,args): self.pre_final_ln = args[0].detach().clone()
        def observed_sdpa(q,k,v,*args,**kwargs):
            out = original_sdpa(q,k,v,*args,**kwargs)
            if self._active and self._active[-1] in self.attention_layers:
                number = self._active[-1]; attn = self.model.blocks[number-1].attn
                if args:
                    raise ArtifactUnavailable('Pinned native SDPA positional mask contract changed')
                if kwargs.get('dropout_p',0.) != 0. or kwargs.get('enable_gqa',False):
                    raise ArtifactUnavailable('Only frozen ungrouped-head SDPA observations are bound')
                if not isinstance(attn.norm,torch.nn.Identity) or attn.gate is not None:
                    raise ArtifactUnavailable('Head projection semantics require the pinned identity inner norm/no gate')
                self.attention[number] = AttentionObservation(q.detach().clone(),k.detach().clone(),v.detach().clone(),
                    out.detach().clone(),self._q_pre.get(number),self._k_pre.get(number),
                    attn.proj.weight.detach().clone(),kwargs.get('scale') or q.shape[-1]**-.5,
                    self.model.num_prefix_tokens,number,
                    None if kwargs.get('attn_mask') is None else kwargs['attn_mask'].detach().clone(),kwargs.get('is_causal',False))
            return out
        with _LOCK:
            try:
                for number,block in enumerate(self.model.blocks,1):
                    handles.append(block.register_forward_pre_hook(block_before(number),with_kwargs=True))
                    if number in self.layers: handles.append(block.register_forward_hook(block_after(number)))
                    if number in self.attention_layers:
                        handles.append(block.attn.register_forward_pre_hook(active_before(number)))
                        handles.append(block.attn.register_forward_hook(active_after,always_call=True))
                        handles.append(block.attn.q_norm.register_forward_hook(qk_save(self._q_pre,number)))
                        handles.append(block.attn.k_norm.register_forward_hook(qk_save(self._k_pre,number)))
                handles.append(self.model.norm.register_forward_pre_hook(norm_before))
                functional.scaled_dot_product_attention = observed_sdpa
                yield self
                if self.attention_layers != set(self.attention):
                    raise ArtifactUnavailable('Required actual native SDPA layer was not observed')
            finally:
                functional.scaled_dot_product_attention = original_sdpa
                for handle in handles: handle.remove()
                self._active.clear()


class InternalEncoder:
    """One real frozen worker model; per-episode raw parity gates all observations."""
    def __init__(self, encoder):
        if encoder.binding.get('execution_kind') != 'real_frozen_cpu_model' or not hasattr(encoder,'_real_forward'):
            raise ArtifactUnavailable('Internal resources require the actual frozen CPU factory model')
        self.encoder = encoder; self.forward = encoder._real_forward; self.model = self.forward._model
        self.torch = self.forward._torch
        from ics.cpu100.common import sha
        self.binding = dict(encoder.binding,internal_encoder_source_sha256=sha(__file__))
        if len(self.model.blocks)!=24 or self.model.num_prefix_tokens!=5:
            raise ArtifactUnavailable('Pinned native DINOv3-L24/five-prefix contract required')
        self.sessions = {}; self._episode_identity=None
        self.receipt = dict(actual_extra_forwards=0,actual_tail_blocks=0,
                                                forward_wall_seconds=0.,forward_cpu_seconds=0.,
                                                device='cpu',frozen=True,query_GT_read=False,parity={})

    def _input(self, rgb):
        torch = self.torch; value = np.asarray(rgb)
        if value.ndim!=3 or value.shape[-1]!=3 or value.shape[0]!=value.shape[1] or value.shape[0]%16:
            raise ValueError('Explicit square RGB canvas with side divisible by16 required')
        if not np.isfinite(value).all(): raise ValueError('Finite RGB required')
        if value.dtype==np.uint8:
            x = _tensor(torch,value).permute(2,0,1)/255
        else:
            if np.any(value<0) or np.any(value>1): raise ValueError('Floating RGB must be physical[0,1]')
            x = _tensor(torch,value).permute(2,0,1)
        return x

    def _normalized(self,x):
        mean=x.new_tensor((.485,.456,.406))[:,None,None]
        std=x.new_tensor((.229,.224,.225))[:,None,None]
        return ((x-mean)/std)[None]

    def encode_rgb(self, role, rgb, *, working_canvas=True, side=None, raw=False):
        if role not in ('r','q') or not working_canvas:
            raise ValueError('Encoder requires caller-recorded R/Q physical working canvas')
        if side is not None and tuple(np.asarray(rgb).shape[:2])!=(side,side):
            raise ValueError('Requested side differs from supplied physical RGB canvas')
        self.forward.check_frozen(); wall,cpu=time.monotonic(),time.process_time()
        with self.torch.inference_mode(),self.torch.autocast('cpu',enabled=False):
            output=self.model.forward_features(self._normalized(self._input(rgb)))
        self.receipt['actual_extra_forwards']+=1
        self.receipt['forward_wall_seconds']+=time.monotonic()-wall
        self.receipt['forward_cpu_seconds']+=time.process_time()-cpu
        self.forward.check_frozen()
        patches=_np(output[0,self.model.num_prefix_tokens:]); hw=np.asarray(rgb).shape[0]//16
        return readonly((patches if raw else unit(patches)).reshape(hw,hw,-1))

    def observe(self, ep, role, *, layers=(12,16,18,19,20,24), attention_layers=(12,16,20,23,24)):
        from ics.cpu100.common import rgb_view
        from .common import artifact
        if role not in ('r','q'): raise ValueError('R/Q role required')
        episode_identity=(ep.source_id,tuple(ep.producer.get('source_image_hashes',())))
        if episode_identity!=self._episode_identity:
            # Model reuse is worker-wide; multi-layer tensors are episode-only.
            self.sessions.clear();self._episode_identity=episode_identity
        image=rgb_view(ep,role); expected=np.asarray(artifact(ep,role+'_raw'))
        if expected.ndim==3: expected=expected.reshape(-1,expected.shape[-1])
        identity=(role,array_hash(image),tuple(layers),tuple(attention_layers))
        if identity in self.sessions: return self.sessions[identity]
        self.forward.check_frozen(); wall,cpu=time.monotonic(),time.process_time()
        session=CaptureSession(self.model,layers=layers,attention_layers=attention_layers,before_layers=(23,24))
        try:
            with self.torch.inference_mode(),self.torch.autocast('cpu',enabled=False),session.hooks():
                final=self.model.forward_features(self._normalized(self._input(image)))
        finally:
            self.receipt['actual_extra_forwards']+=1
            self.receipt['forward_wall_seconds']+=time.monotonic()-wall
            self.receipt['forward_cpu_seconds']+=time.process_time()-cpu
        raw=_np(final[0,5:]); maximum=float(np.max(np.abs(raw-expected))) if raw.shape==expected.shape else np.inf
        if raw.shape!=expected.shape or not np.allclose(raw,expected,atol=2e-5,rtol=2e-5):
            raise ArtifactUnavailable('Actual internal forward does not match sealed native raw final-LN: '+str(maximum))
        self.receipt['parity'][role]=dict(max_abs_error=maximum,atol=2e-5,rtol=2e-5,
                                          input_rgb_sha256=array_hash(image),native_raw_sha256=array_hash(expected))
        session.raw_final_ln=raw
        session.producer=dict(self.binding,source_image_hashes=ep.producer.get('source_image_hashes'),
                             observed_role=role,layer_index='one_based post-block before finalLN unless explicitly finalLN',
                             qk_position='native qk_norm before RoPE and actual SDPA after RoPE',prefix_tokens=5,
                             softmax_denominator='all native patch+prefix keys; no patch-submatrix renormalization')
        self.sessions[identity]=session
        self.forward.check_frozen();return session

    def input_gradient(self, role, rgb, scalar, *, working_canvas=True):
        """Genuine frozen input VJP; scalar(raw_finalLN,unit_finalLN)->Torch scalar."""
        if role not in ('r','q') or not working_canvas: raise ValueError('Explicit R/Q working canvas required')
        self.forward.check_frozen(); x=self._input(rgb).detach().requires_grad_(True)
        wall,cpu=time.monotonic(),time.process_time()
        with self.torch.enable_grad(),self.torch.autocast('cpu',enabled=False):
            raw=self.model.forward_features(self._normalized(x))[0,5:]
            norm=self.torch.nn.functional.normalize(raw,dim=-1)
            value=scalar(raw,norm)
            if value.ndim!=0: raise ValueError('Input-gradient objective must be a fixed scalar')
            gradient=self.torch.autograd.grad(value,x,only_inputs=True)[0]
        self.receipt['actual_extra_forwards']+=1
        self.receipt['forward_wall_seconds']+=time.monotonic()-wall
        self.receipt['forward_cpu_seconds']+=time.process_time()-cpu
        self.forward.check_frozen();return readonly(_np(gradient).transpose(1,2,0))

    def tail_replay(self, ep, role, *, patch_groups=None, private_special_tokens=True,
                    blocked_edges=None, head_gates=None, refresh_qkv=True, return_attention=True,
                    message_delta=None):
        """Continue actual last2 blocks+finalLN; each group has private prefix state."""
        import torch.nn.functional as functional
        session=self.observe(ep,role); original_sdpa=functional.scaled_dot_product_attention
        state=session.before[23]; n=state.shape[1]-5
        groups=None if patch_groups is None else np.asarray(patch_groups,int).ravel()
        if groups is not None and (groups.shape!=(n,) or np.any(groups<0) or not private_special_tokens):
            raise ValueError('Each patch needs a component ID and private special tokens')
        gates=np.ones((2,self.model.blocks[-1].attn.num_heads),np.float32) if head_gates is None else np.asarray(head_gates,float)
        if gates.shape!=(2,self.model.blocks[-1].attn.num_heads) or np.any((gates!=0)&(gates!=1)):
            raise ValueError('Tail head gates must be binary[2,H], zeroing the true projection component')
        regions=[np.arange(n)] if groups is None else [np.flatnonzero(groups==g) for g in sorted(set(groups))]
        final=np.empty((n,state.shape[-1]),np.float32); observations=[];wall,cpu=time.monotonic(),time.process_time()
        self.forward.check_frozen()
        with _LOCK,self.torch.inference_mode(),self.torch.autocast('cpu',enabled=False):
            for region in regions:
                ids=np.r_[np.arange(5),region+5]; selection=self.torch.as_tensor(ids,dtype=self.torch.long)
                current=state.index_select(1,selection).clone(); by_layer={}
                for offset,number in enumerate((23,24)):
                    block=self.model.blocks[number-1]; rope=session.ropes[number]
                    if groups is not None and rope is not None:
                        rope=rope.index_select(-2,self.torch.as_tensor(region,dtype=self.torch.long))
                    allowed=None
                    edge_input=blocked_edges.get(number,()) if isinstance(blocked_edges,dict) else blocked_edges
                    if edge_input is not None and len(edge_input):
                        allowed=self.torch.ones((len(ids),len(ids)),dtype=self.torch.bool)
                        positions={int(p):j+5 for j,p in enumerate(region)}
                        for a,b in np.asarray(edge_input,int).reshape(-1,2):
                            if a<0 or b<0 or a>=n or b>=n: raise ValueError('Blocked edges are directed native patch IDs')
                            if int(a) in positions and int(b) in positions: allowed[positions[int(a)],positions[int(b)]]=False
                    attn=block.attn
                    if not isinstance(attn.norm,self.torch.nn.Identity) or attn.gate is not None:
                        raise ArtifactUnavailable('True head gating is undefined for an unbound inner normalization/gate')
                    capture=CaptureSession(self.model,attention_layers=(number,))
                    gate_tensor=_tensor(self.torch,gates[offset])
                    def gate_projection(module,args):
                        value=args[0]; h=attn.num_heads
                        return (value.reshape(value.shape[0],value.shape[1],h,-1).mul(gate_tensor[None,None,:,None]).reshape_as(value),)
                    handle=attn.proj.register_forward_pre_hook(gate_projection)
                    delta_handle=None
                    if message_delta is not None:
                        def inject_message(module,args,output):
                            delta=np.asarray(message_delta(role,number,capture.attention[number],region),np.float32)
                            if delta.shape!=tuple(output.shape[1:]) or not np.isfinite(delta).all():
                                raise ValueError('Injected actual attention-output delta must include private prefix rows')
                            return output+_tensor(self.torch,delta)[None]
                        delta_handle=attn.proj.register_forward_hook(inject_message)
                    try:
                        with capture.hooks():
                            if not refresh_qkv:
                                captured_sdpa=functional.scaled_dot_product_attention
                                def fixed_route(q,k,v,*args,**kwargs):
                                    base=session.attention[number]
                                    q=base.q.index_select(-2,selection);k=base.k.index_select(-2,selection)
                                    capture._q_pre[number]=base.q_pre_rope.index_select(-2,selection)
                                    capture._k_pre[number]=base.k_pre_rope.index_select(-2,selection)
                                    return captured_sdpa(q,k,v,*args,**kwargs)
                                functional.scaled_dot_product_attention=fixed_route
                            current=block(current,rope=rope,attn_mask=allowed,is_causal=False)
                        if return_attention: by_layer[number]=capture.attention[number]
                    finally:
                        functional.scaled_dot_product_attention=original_sdpa;handle.remove()
                        if delta_handle is not None: delta_handle.remove()
                    self.receipt['actual_tail_blocks']+=1
                output=self.model.norm(current)[0,5:];final[region]=output.cpu().numpy()
                observations.append(dict(patch_ids=readonly(region),layers=by_layer))
        self.receipt['forward_wall_seconds']+=time.monotonic()-wall
        self.receipt['forward_cpu_seconds']+=time.process_time()-cpu
        self.forward.check_frozen()
        normal=groups is None and blocked_edges is None and np.all(gates==1) and refresh_qkv and message_delta is None
        if normal and not np.array_equal(final,session.raw_final_ln):
            error=float(np.max(np.abs(final-session.raw_final_ln)))
            raise ArtifactUnavailable('Normal native two-block replay is not bitwise final-LN parity: '+str(error))
        return dict(raw_final_ln=readonly(final),unit=readonly(unit(final)),observations=observations,
                    producer=session.producer,refresh_qkv=bool(refresh_qkv),private_special_tokens=bool(private_special_tokens),
                    normal_replay_bitwise_parity=bool(normal))

    def require(self, ep, name):
        """Owner-facing aliases; only requested full attention allocates N squared."""
        callbacks={'tail_replay':self.tail_replay,'frozen_encode_rgb':self.encode_rgb,
                   'encode_rgb':self.encode_rgb,'frozen_input_gradient':self.input_gradient}
        if name in callbacks: return callbacks[name]
        if name=='internal_encoder_binding': return self.binding
        if name in ('midlayers','attention_special_to_patch','head_messages_last','QKV','tail_attention'):
            sessions={role:self.observe(ep,role) for role in ('r','q')}
            if name=='midlayers':
                values={role:readonly(np.stack([_np(session.states[k][0,5:]) for k in (12,16,20,24)]))
                        for role,session in sessions.items()}
                return dict(values,blocks=[12,16,20,24],producers={role:s.producer for role,s in sessions.items()})
            if name=='attention_special_to_patch':
                values={role:readonly(np.stack([session.attention[k].special_to_patch() for k in (12,16,20,24)]))
                        for role,session in sessions.items()}
                return dict(values,blocks=[12,16,20,24],producers={role:s.producer for role,s in sessions.items()})
            if name=='head_messages_last':
                return {role:s.attention[24].head_messages() for role,s in sessions.items()}
            return {role:(s.attention[23],s.attention[24]) for role,s in sessions.items()}
        if name=='action_edge_features':
            def action_features(episode,role,edges):
                session=self.observe(episode,role)
                return readonly(np.c_[session.attention[23].edge_features(edges),session.attention[24].edge_features(edges)])
            return action_features
        if name.startswith(('q_','r_','tail_state_q','tail_state_r')):
            role=name[0] if name.startswith(('q_','r_')) else name[-1]
            session=self.observe(ep,role); feature=name[2:] if name.startswith(('q_','r_')) else 'tail_state'
            if feature in ('pre_final_ln','pre_finalLN'): return _np(session.pre_final_ln[0,5:])
            if feature=='block_minus6': return _np(session.states[19][0,5:])
            if feature=='tail_state': return _np(session.before[23][0])
            if feature=='attention_final': return session.attention[24].patch_attention()
            if feature=='attention_last2': return readonly(np.stack([session.attention[k].patch_attention() for k in (23,24)]))
            if feature=='head_output_final': return session.attention[24].patch_head_output()
            if feature=='layer_tokens_half_threequarter_final':
                return readonly(np.stack([unit(_np(session.states[k][0,5:])) for k in (12,18)]+[unit(session.raw_final_ln)]))
            if feature=='layer_tokens_mid_final':
                return readonly(np.stack([unit(_np(session.states[12][0,5:])),unit(session.raw_final_ln)]))
        raise ArtifactUnavailable('Exact internal resource not implemented: '+name)


def get_internal_encoder(model_dir, producer, *, threads=1):
    from ics.cpu100.encoder import get_cpu_encoder
    encoder=get_cpu_encoder(model_dir,producer,threads=threads,max_cached_views=16)
    if not hasattr(encoder,'_astra_internal'):
        encoder._astra_internal=InternalEncoder(encoder)
    return encoder._astra_internal
