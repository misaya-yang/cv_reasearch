"""Ephemeral exact public-FoRIS context for ten independent user proposals.

No query annotations are accepted. Frozen encoder outputs are pair-local, never
written. All substitutions occur inside the actual public source prediction.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import inspect

import torch
import torch.nn.functional as F


@dataclass
class ViewBudget:
    max_extra_b2: int = 8
    extra_b2_calls: int = 0
    native_b2_calls: int = 0
    actual_b2_calls: int = 0
    cache: dict = field(default_factory=dict)


def _tensor_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_tensor_copy(v) for v in value)
    return value


@contextmanager
def _replace_instance(host, replacements):
    saved = {n: (n in host.__dict__, host.__dict__.get(n)) for n in replacements}
    try:
        for n, fn in replacements.items():
            setattr(host, n, fn)
        yield
    finally:
        for n, (owned, old) in saved.items():
            if owned:
                setattr(host, n, old)
            else:
                delattr(host, n)


class TenDirectionContext:
    """Public source state and legal inference data; replay does not encode.

    Overrides receive original_result, ctx, and original source arguments as
    named keywords. Part2 accepts either its entire five-tuple or a score map;
    grouping overrides return label IDs, not an independently predicted mask.
    """
    def __init__(self, host, encoder, support_pil, query_pil,
                 reference_mask_original, *, raw_native, layer_maps,
                 budget=None, new_view_callback=None, coordinate_metadata=None):
        self.host = host
        self.encoder = encoder
        self.support_rgb = support_pil.copy()
        self.query_rgb = query_pil.copy()
        self.reference_mask_original = reference_mask_original.detach().bool().clone()
        self.device = raw_native.device
        self.raw_native = raw_native.detach()
        self.work_hw = tuple(host._transform(query_pil).shape[-2:])
        self.grid_hw = tuple(raw_native.shape[-2:])
        self.reference_mask = F.interpolate(
            self.reference_mask_original[None, None].float().to(self.device),
            self.work_hw, mode='nearest')[0, 0].bool()
        self.reference_coverage = F.interpolate(
            self.reference_mask[None, None].float(), self.grid_hw,
            mode='area')[0, 0]
        self.refcov = self.reference_coverage
        self.support_work = host._transform(support_pil)[None].to(self.device)
        self.query_work = host._transform(query_pil).to(self.device)
        self.expected_raw_input = torch.cat(
            [self.support_work, self.query_work[None]], 0)[None]
        if tuple(raw_native.shape[:2]) != (1, 2):
            raise ValueError('Only actual public native B2 feature outputs are accepted')
        self.features = {k: F.normalize(v.detach(), p=2, dim=1)
                         for k, v in layer_maps.items()}
        self.features['final_raw'] = F.normalize(raw_native[0], p=2, dim=1)
        self.budget = budget if budget is not None else ViewBudget()
        self._new_view_callback = new_view_callback
        self.coordinate_metadata = coordinate_metadata or {
            'source': 'public square Resize, no padding',
            'query_original_hw': [query_pil.height, query_pil.width]}
        self._cluster_cache = []
        self.native = None
        self.replay_calls = 0
        self._active = False

    def initialize_native(self):
        """Decoder capture through exact B2 cache, then all identity hooks check."""
        native = self.replay()
        self.native = native
        self.features['final'] = native['part1'][0]
        self.ref_tokens = F.normalize(
            self.features['final'][0].flatten(1).T, dim=1)
        self.query_tokens = F.normalize(
            self.features['final'][1].flatten(1).T, dim=1)
        identity = self.replay(
            part2_override=lambda original_result, **kw: original_result,
            group_override=lambda labels, **kw: labels,
            seed_prior_override=lambda original_result, **kw: original_result,
            part4_override=lambda original_result, **kw: original_result)
        if not torch.equal(native['mask'], identity['mask']):
            raise RuntimeError('Public full-pipeline identity callbacks change native output')
        if not torch.equal(native['score'], identity['score']):
            raise RuntimeError('Identity callbacks change native continuous output')
        self.native['identity_exact'] = True
        return self.native

    def _public_set(self, ref_mask):
        h = self.host
        if h._ref_images is not None or h._ref_masks is not None or h._tgt_image is not None:
            raise RuntimeError('Host has foreign/unfinished public state')
        h.set_reference(self.support_rgb, ref_mask)
        h.set_target(self.query_rgb)
        if not torch.equal(h._ref_images, self.support_work):
            raise RuntimeError('Public support transform differs from frozen input')
        if not torch.equal(h._tgt_image, self.query_work):
            raise RuntimeError('Public query transform differs from frozen input')

    def replay(self, ref_mask=None, part2_override=None, group_override=None,
               seed_prior_override=None, part4_override=None):
        if self._active:
            raise RuntimeError('Nested source replay is forbidden; compute alternatives before override')
        mask = self.reference_mask_original if ref_mask is None else ref_mask
        if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError('Reference hypothesis must be a legal two-dimensional bool mask')
        if not mask.any():
            raise ValueError('Empty reference concept is an explicit abstention, not source failure')
        self._public_set(mask)
        h = self.host
        stages = {}
        part1 = None
        observed_score = None
        source_extractor = h._extract_features
        original = {n: getattr(h, n) for n in (
            '_part1_positional_debias', '_part2_background_suppression',
            '_locate_candidates', '_build_seed_cluster_prior', '_part3_clustering',
            '_semantic_disagreement_penalty', '_semantic_cluster_reweight_map',
            '_part4_semantic_consistency_correction', '_binarize_response', '_finalize_mask')}

        def cached_extract(imgs):
            if (imgs.shape != self.expected_raw_input.shape or
                imgs.dtype != self.expected_raw_input.dtype or
                imgs.stride() != self.expected_raw_input.stride() or
                not torch.equal(imgs, self.expected_raw_input)):
                raise RuntimeError('Encoder cache requested for a changed B2 image input')
            stages['raw_cache_hits'] = stages.get('raw_cache_hits', 0) + 1
            return self.raw_native

        def wrap(name, fn):
            signature = inspect.signature(fn)
            def call(*args, **kwargs):
                nonlocal part1, observed_score
                named = dict(signature.bind(*args, **kwargs).arguments)
                result = fn(*args, **kwargs)
                if name == '_part1_positional_debias':
                    part1 = result.detach().clone()
                    stages['part1_gate_recomputed'] = True
                    stages['part1_applied'] = not torch.equal(named['fmaps_norm'], result)
                if name == '_part2_background_suppression' and part2_override is not None:
                    replaced = part2_override(original_result=_tensor_copy(result), ctx=self, **named)
                    if isinstance(replaced, torch.Tensor):
                        replaced = (replaced, *result[1:])
                    if not isinstance(replaced, tuple) or len(replaced) != 5:
                        raise ValueError('Part2 substitution must preserve original five outputs')
                    result = replaced
                if name == '_build_seed_cluster_prior' and seed_prior_override is not None:
                    result = seed_prior_override(original_result=_tensor_copy(result), ctx=self, **named)
                if name == '_part4_semantic_consistency_correction' and part4_override is not None:
                    result = part4_override(original_result=_tensor_copy(result), ctx=self, **named)
                fields = {
                    '_part2_background_suppression': ('part2_score','part2_sf','part2_sbn','part2_mu_fg','part2_tgt_denoised'),
                    '_locate_candidates': ('candidate_hard','candidate_vote'),
                    '_build_seed_cluster_prior': ('seed_prior',),
                    '_part3_clustering': ('part3_score','part3_candidate_vote','part3_seed_prior'),
                    '_semantic_disagreement_penalty': ('part4_penalty',),
                    '_semantic_cluster_reweight_map': ('part4_cluster_delta',),
                    '_part4_semantic_consistency_correction': ('part4_score',),
                    '_binarize_response': ('pre_refinement_mask',),
                    '_finalize_mask': ('post_refinement_mask',)}.get(name, ())
                if fields:
                    if result is None:
                        raise RuntimeError('Public source returned no result for '+name)
                    values = result if isinstance(result, tuple) else (result,)
                    for field, value in zip(fields, values):
                        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                            raise ValueError('Non-finite/invalid source-stage output: '+field)
                        stages[field] = value.detach().clone()
                    if name == '_part2_background_suppression':
                        stages['sf'], stages['sbn'] = stages['part2_sf'], stages['part2_sbn']
                    if name == '_part4_semantic_consistency_correction':
                        observed_score = result.detach().clone()
                return result
            return call

        # Both clustering functions share the public source module's globals.
        scope = original['_build_seed_cluster_prior'].__func__.__globals__
        source_cluster = scope['agglomerative_clustering']
        active_stage = [None]
        def cluster(values, *args, **kwargs):
            labels = None
            for old_values, old_args, old_kwargs, old_labels in self._cluster_cache:
                if old_args == args and old_kwargs == kwargs and torch.equal(old_values, values):
                    labels = old_labels.clone(); break
            if labels is None:
                labels = source_cluster(values, *args, **kwargs)
                # At most two original group inputs; alternate hypotheses may differ.
                if len(self._cluster_cache) < 8:
                    self._cluster_cache.append((values.detach().clone(),args,kwargs,labels.detach().clone()))
            if group_override is not None and active_stage[0] in ('seed','semantic'):
                labels = group_override(values=values, labels=labels.clone(), stage=active_stage[0], ctx=self)
                if labels.shape != (values.shape[0],) or labels.dtype not in (torch.int32,torch.int64) or (labels < 0).any():
                    raise ValueError('Grouping substitution must be nonnegative row-aligned label IDs')
            stage_name = active_stage[0] or 'reference'
            stages[stage_name+'_cluster_labels'] = labels.detach().clone()
            return labels
        replacements = {'_extract_features': cached_extract}
        for name, fn in original.items():
            wrapped = wrap(name, fn)
            if name in ('_build_seed_cluster_prior','_semantic_cluster_reweight_map'):
                stage = 'seed' if name == '_build_seed_cluster_prior' else 'semantic'
                def grouped(*args, _wrapped=wrapped, _stage=stage, **kwargs):
                    old = active_stage[0]; active_stage[0] = _stage
                    try: return _wrapped(*args, **kwargs)
                    finally: active_stage[0] = old
                replacements[name] = grouped
            else:
                replacements[name] = wrapped
        self._active = True
        scope['agglomerative_clustering'] = cluster
        try:
            with _replace_instance(h, replacements):
                prediction = h.segment().reshape(self.work_hw).bool().clone()
            if stages.get('raw_cache_hits') != 1 or part1 is None or observed_score is None:
                raise RuntimeError('Incomplete public source prediction/cached encoder contract')
            self.replay_calls += 1
            return dict(mask=prediction,score=observed_score,stages=stages,part1=part1,
                        audit=dict(full_public_source=True,encoder_calls=0,
                                   reference_hypothesis_changed=ref_mask is not None,
                                   raw_cache_input_exact=True,part1_gate_recomputed=True))
        finally:
            scope['agglomerative_clustering'] = source_cluster
            self._active = False
            # Public segment resets these only on success. Failed own stage must not
            # contaminate independently proposed methods in the shared finite batch.
            h._ref_images = h._ref_masks = h._tgt_image = h._orig_tgt_size = None

    def finalize_work_mask(self, mask):
        if mask.shape != self.work_hw or mask.dtype != torch.bool:
            raise ValueError('Full working-grid bool mask required before common CRF')
        return self.host._finalize_mask(mask, self.query_work[None]).reshape(self.work_hw).bool()

    def new_view(self, support_rgb, reference_mask_original, query_rgb, view_key, metadata=None):
        if self._new_view_callback is None:
            raise RuntimeError('Prepared new-view callback required')
        identity = hashlib.sha256(
            support_rgb.tobytes()+str(support_rgb.size).encode()+
            query_rgb.tobytes()+str(query_rgb.size).encode()+
            reference_mask_original.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if view_key in self.budget.cache:
            old_identity, old_context = self.budget.cache[view_key]
            if old_identity != identity:
                raise RuntimeError('Same view key reused for different images/reference labels')
            return old_context
        if self.budget.extra_b2_calls >= self.budget.max_extra_b2:
            raise RuntimeError('Finite prepared extra-B2 view budget exhausted')
        self.budget.extra_b2_calls += 1
        child = self._new_view_callback(support_rgb,reference_mask_original,query_rgb,self.budget,metadata)
        self.budget.cache[view_key] = (identity,child)
        return child


@contextmanager
def capture_exact_native_and_middle(host, encoder):
    """One actual original n=1 B2 forward plus raw block6/12 hooks.

    The native final feature is returned verbatim by the original extractor;
    middle maps are block outputs normalized in their own channel coordinates.
    """
    model = getattr(encoder,'m',None)
    if model is None:
        model = getattr(encoder,'model',None)
    if model is None:
        model = getattr(encoder,'backbone',None)
    if model is None or not hasattr(model,'blocks'):
        raise RuntimeError('Actual timm DINO block container unavailable')
    observed = {}; handles = []
    for number in (6,12):
        def hook(module,args,output,_number=number):
            if _number in observed:
                raise RuntimeError('More than one B2 forward in native context capture')
            observed[_number] = output.detach().clone()
        handles.append(model.blocks[number-1].register_forward_hook(hook))
    original = host._extract_features
    def capture(imgs):
        if 'raw_native' in observed:
            raise RuntimeError('Native raw output captured more than once')
        observed['encoder_forward_calls'] = observed.get('encoder_forward_calls',0)+1
        result = original(imgs)
        observed['raw_native'] = result.detach().clone()
        observed['exact_input'] = imgs.detach().clone()
        return result
    try:
        with _replace_instance(host,{'_extract_features':capture}):
            yield observed
        h,w = observed['raw_native'].shape[-2:]
        n_prefix = int(getattr(model,'num_prefix_tokens',1))
        for number in (6,12):
            x = observed[number]
            if x.ndim != 3 or x.shape[0] != 2 or x.shape[1]-n_prefix != h*w:
                raise RuntimeError('Actual DINO middle block/token geometry mismatch')
            observed[number] = x[:,n_prefix:].transpose(1,2).reshape(2,-1,h,w)
    finally:
        for handle in handles:
            handle.remove()
