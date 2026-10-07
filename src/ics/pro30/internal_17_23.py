"""Genuine frozen DINO state capture, terminal replay and suffix JVP.

No NumPy final-feature surrogate is used. Construct FrozenSuffixContext from
the existing actual Astra InternalEncoder and bind artifact
``pro30_internal_context``. The numerical-cost receipt separates encoding and
suffix work; raw native parity gates every episode.
"""
from __future__ import annotations
import time
import numpy as np
from .common import ArtifactUnavailable, require_artifact, array_hash, readonly
from ics.cpu100.common import rgb_view


class FrozenSuffixContext:
    def __init__(self,internal):
        if internal.binding.get('execution_kind')!='real_frozen_cpu_model':
            raise ArtifactUnavailable('Pro30 requires the actual pinned frozen CPU model')
        self.internal=internal;self.model=internal.model;self.torch=internal.torch
        if len(self.model.blocks)!=24 or self.model.num_prefix_tokens!=5:raise ArtifactUnavailable('24-block/5-prefix binding required')
        self.sessions={};self.identity=None
        self.receipt=dict(actual_original_forwards=0,actual_suffix_blocks=0,actual_JVPs=0,
            extra_encoding_wall_seconds=0.,suffix_wall_seconds=0.,jvp_wall_seconds=0.,query_GT_read=False,
            JVP_algorithm='torch.autograd.functional.jvp of real math-SDPA suffix',frozen_weights=True,parity={})

    def capture(self,ep,role):
        from ics.astra300.internal_encoder import CaptureSession, _np
        from ics.cpu100.encoder import assert_matches_producer
        assert_matches_producer(ep.producer,self.internal.binding)
        identity=(ep.source_id,id(ep.r),id(ep.q))
        if identity!=self.identity:self.sessions.clear();self.identity=identity
        if role in self.sessions:return self.sessions[role]
        image=rgb_view(ep,role);raw=np.asarray(require_artifact(ep,role+'_raw')).reshape(-1,ep.r.shape[1])
        self.internal.forward.check_frozen();start=time.perf_counter()
        session=CaptureSession(self.model,layers=(8,12,16,20,24),before_layers=(21,22,23,24))
        with self.torch.inference_mode(),self.torch.autocast('cpu',enabled=False),session.hooks():
            output=self.model.forward_features(self.internal._normalized(self.internal._input(image)))
        actual=_np(output[0,5:]);maximum=float(np.max(np.abs(actual-raw))) if actual.shape==raw.shape else np.inf
        if actual.shape!=raw.shape or not np.allclose(actual,raw,atol=2e-5,rtol=2e-5):
            raise ArtifactUnavailable('Pro30 true native forward does not match sealed finalLN:'+str(maximum))
        self.receipt['actual_original_forwards']+=1;self.receipt['extra_encoding_wall_seconds']+=time.perf_counter()-start
        self.receipt['parity'][role]=dict(max_abs_error=maximum,input_rgb_sha256=array_hash(image),raw_sha256=array_hash(raw))
        self.sessions[role]=session;self.internal.forward.check_frozen();return session

    def _tensor(self,array):
        # Create ordinary tensors, not requires_grad on an inference tensor.
        return self.torch.tensor(np.array(array,copy=True),dtype=self.torch.float32)

    def state(self,ep,role,layer,*,prefix=False):
        session=self.capture(ep,role);value=session.states[int(layer)][0].cpu().numpy()
        return readonly(value if prefix else value[5:])

    def normalize(self,raw,*,unit_output=True):
        import torch.nn.functional as F
        self.internal.forward.check_frozen()
        with self.torch.inference_mode():
            value=self.model.norm(self._tensor(raw))
            if unit_output:value=F.normalize(value,dim=-1)
        return readonly(value.cpu().numpy())

    def suffix(self,ep,role,state,*,start_layer=21,normalize=True):
        """One-based actual layers start_layer..24, native prefix and RoPE."""
        session=self.capture(ep,role);current=self._tensor(state)[None];start=time.perf_counter()
        with self.torch.inference_mode(),self.torch.autocast('cpu',enabled=False):
            for number in range(start_layer,25):
                current=self.model.blocks[number-1](current,rope=session.ropes[number],attn_mask=None,is_causal=False)
            if normalize:current=self.model.norm(current)
        self.receipt['actual_suffix_blocks']+=25-start_layer;self.receipt['suffix_wall_seconds']+=time.perf_counter()-start
        self.internal.forward.check_frozen();return readonly(current[0].cpu().numpy())

    def jvp(self,ep,role,state,direction,*,local_only=False):
        """True DG(H)[direction] for actual blocks21--24 + native finalLN.

        ``local_only`` is the explicit uncounted control: residual + actual
        per-token MLP updates only, followed by finalLN; attention omitted.
        """
        from torch.nn.attention import sdpa_kernel,SDPBackend
        session=self.capture(ep,role);H=self._tensor(state)[None].requires_grad_(True);V=self._tensor(direction)[None]
        ropes={number:(None if session.ropes[number] is None else self._tensor(session.ropes[number].cpu().numpy())) for number in range(21,25)}
        if H.shape!=V.shape:raise ValueError('Real suffix tangent must include prefix rows')
        def function(current):
            for number in range(21,25):
                block=self.model.blocks[number-1]
                if local_only:
                    # Execute the actual frozen block's residual/MLP/LayerScale
                    # branch. Only its attention output is replaced by zero;
                    # this does not guess Eva's gamma versus ls module layout.
                    original=block.attn.forward
                    def no_attention(value,*args,**kwargs):return self.torch.zeros_like(value)
                    block.attn.forward=no_attention
                    try:current=block(current,rope=ropes[number],attn_mask=None,is_causal=False)
                    finally:block.attn.forward=original
                else:current=block(current,rope=ropes[number],attn_mask=None,is_causal=False)
            return self.model.norm(current)
        start=time.perf_counter();self.internal.forward.check_frozen()
        # CPU flash attention double backward is not supported on all builds.
        # Math SDPA computes the same native softmax, with exact autograd JVP.
        with self.torch.enable_grad(),self.torch.autocast('cpu',enabled=False),sdpa_kernel(SDPBackend.MATH):
            original,tangent=self.torch.autograd.functional.jvp(function,H,V,create_graph=False,strict=True)
        self.receipt['actual_JVPs']+=1;self.receipt['actual_suffix_blocks']+=4
        self.receipt['jvp_wall_seconds']+=time.perf_counter()-start;self.internal.forward.check_frozen()
        if not self.torch.isfinite(tangent).all():raise FloatingPointError('Real suffix JVP nonfinite')
        return readonly(original[0].detach().cpu().numpy()),readonly(tangent[0].detach().cpu().numpy())

    def ordinary_perturbations(self,ep,role,state,directions,epsilon=1e-3):
        """Same six suffix calls for the registered perturbation-average control."""
        return [self.suffix(ep,role,np.asarray(state)+epsilon*np.asarray(d)) for d in directions]


def get_context(ep):
    try:context=require_artifact(ep,'pro30_internal_context')
    except ArtifactUnavailable:
        bound=require_artifact(ep,'pro30_encoder_context')
        internal=getattr(bound,'internal',bound)
        if not hasattr(internal,'_pro30_suffix_context'):internal._pro30_suffix_context=FrozenSuffixContext(internal)
        context=internal._pro30_suffix_context
    required=('state','normalize','suffix','jvp','capture','receipt')
    if not all(hasattr(context,name) for name in required):raise ArtifactUnavailable('Actual frozen Pro30 suffix context contract required')
    return context
