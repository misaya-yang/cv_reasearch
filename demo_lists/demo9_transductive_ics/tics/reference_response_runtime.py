"""One-block signed reference-value probes with explicit source contracts.

This integrates a proposed observation, not a trained segmentation method.
Requires an independently verified attention-source hash for production. No
downloads, feature files, compiler, GPU scheduling or fallback source rewrite.
"""
from contextlib import contextmanager
import hashlib
import inspect
import sys
from types import MethodType
import torch

from tics.reference_halfspace_probe import reference_axis, project_query_values
from scripts.native_attention_readout import eva_components, native_heads, project_heads


def attention_source_hash(attention):
    """Caller freezes this hash in the CPU preflight, not during GPU execution."""
    function = getattr(attention.forward, '__func__', attention.forward)
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def decoder_arguments(packet, *, static=False):
    """Explicit runtime-to-head bridge; never silently forwards label fields."""
    expected = {'support_final', 'support_middle', 'support_coverage',
                'query_final', 'query_middle', 'query_immediate_plus',
                'query_immediate_minus', 'query_response_plus', 'query_response_minus',
                'support_prefix', 'query_prefix', 'prefix_immediate_plus', 'prefix_immediate_minus'}
    features = packet['features']
    if set(features) != expected:
        raise ValueError('Runtime feature schema differs; no GT/class/area or missing fields allowed')
    if not isinstance(static, bool):
        raise ValueError('Declared decoder mode must be boolean')
    args = {name: features[name] for name in
            ('support_final', 'support_middle', 'support_coverage', 'query_final', 'query_middle')}
    args.update(immediate_plus=features['query_immediate_plus'],
                immediate_minus=features['query_immediate_minus'], static=static)
    args.update({name: features[name] for name in
                 ('support_prefix', 'query_prefix', 'prefix_immediate_plus', 'prefix_immediate_minus')})
    if not static:
        args.update(response_plus=features['query_response_plus'],
                    response_minus=features['query_response_minus'])
    return args


def finalize_decoder_logits(logits, query_rgb, source_finalize):
    """Zero threshold AFTER logit upsampling, then one original finalizer.

    source_finalize must be the original FoRIS _finalize_mask callback with
    its public query/original-size state already set. This function neither
    fabricates that state nor re-applies CRF to an already refined mask.
    Its caller checks the immutable source fingerprint and output size.
    """
    from tics.reference_response_decoder import logits_to_mask
    if (query_rgb.ndim != 4 or query_rgb.shape[:2] != (1, 3)
            or not query_rgb.is_floating_point() or not torch.isfinite(query_rgb).all()):
        raise ValueError('One finite public transformed query RGB required')
    if logits.ndim != 4 or logits.shape[:2] != (1, 1) or logits.device != query_rgb.device:
        raise ValueError('Single binary logit map on the same query device required')
    binary = logits_to_mask(logits, tuple(query_rgb.shape[-2:]))[0, 0]
    output = source_finalize(binary, query_rgb)
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise RuntimeError('Original source finalizer must return one binary mask')
    if output.dtype != torch.bool and not ((output == 0) | (output == 1)).all():
        raise RuntimeError('Finalizer returned continuous scores; no undeclared second threshold')
    return output.bool()


def _grid(sequence, prefix, grid):
    batch, tokens, channels = sequence.shape
    height, width = grid
    if tokens-prefix != height*width:
        raise ValueError('Actual patch tokens differ from declared image grid')
    return sequence[:, prefix:].transpose(1, 2).reshape(batch, channels, height, width)


def _head_grid(heads, prefix, grid):
    batch, count, tokens, dimension = heads.shape
    return _grid(heads.transpose(1, 2).reshape(batch, tokens, count*dimension), prefix, grid)


@contextmanager
def _selected_forward(backbone, selected, state, grid, coverage, rope_apply):
    """Return original objects for native/no-op; restore only our instance hook."""
    name, attention = selected
    modules = dict(backbone.named_modules())
    block_name = name.rsplit('.', 1)[0] if '.' in name else ''
    block = modules.get(block_name, backbone)
    original = attention.forward
    had_override = 'forward' in attention.__dict__
    old_override = attention.__dict__.get('forward')
    handle = None

    def capture_block(module, inputs, output):
        state['block_calls'] += 1
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise RuntimeError('Selected native block must return [B,tokens,C]')
        if state['phase'] == 'native':
            # Clone before applying even an in-place norm implementation.
            state['middle'] = backbone.norm(output.clone()).detach().clone()

    def probed(this, x, rope=None, attn_mask=None, is_causal=False):
        state['attention_calls'] += 1
        if state['attention_calls'] != 1:
            raise RuntimeError('Selected attention executed more than once in one encoder pass')
        if x.shape[0] != 2:
            raise RuntimeError('Probe requires the unchanged native [support,query] batch')
        actual = original(x, rope=rope, attn_mask=attn_mask, is_causal=is_causal)
        if not isinstance(actual, torch.Tensor) or actual.shape != x.shape:
            raise RuntimeError('Unsupported native projected-attention interface')
        q, k, v, qr, kr, gate = eva_components(this, x, rope, rope_apply)
        if not all(torch.isfinite(t).all() for t in (q, k, v, qr, kr, actual)):
            raise RuntimeError('Nonfinite native attention fields')
        prefix = this.num_prefix_tokens
        if x.shape[1]-prefix != grid[0]*grid[1]:
            raise RuntimeError('Support coverage cannot be aligned to native patch tokens')
        replica = project_heads(this, native_heads(this, qr, kr, v, attn_mask, is_causal), x, gate)
        if not torch.equal(actual, replica):
            error = float((actual-replica).abs().max())
            raise RuntimeError('Native attention replica differs; reject probe: '+str(error))
        if state['phase'] == 'native':
            state['x'] = x.detach().clone()
            state['attention_native'] = actual.detach().clone()
            state['qkv'] = tuple(t.detach().clone() for t in (q, k, v))
            state['prefix'] = prefix
            labels = coverage.flatten().to(device=v.device)
            state['axis'] = reference_axis(v[0, :, prefix:], labels >= .9, labels <= .1)
            # Read immediate PROJECTION/norm/gate differences too, not only A*dV.
            for branch in ('plus', 'minus'):
                changed = project_query_values(v, state['axis'], branch,
                                               query_index=1, prefix_tokens=prefix)
                projected = actual if changed is v else project_heads(
                    this, native_heads(this, qr, kr, changed, attn_mask, is_causal), x, gate)
                state['immediate_'+branch] = (actual-projected).detach().clone()
            state['native_exact'] = True
            return actual
        if state['phase'] not in ('plus', 'minus'):
            raise RuntimeError('Unknown probe phase')
        if not torch.equal(x, state['x']) or not torch.equal(actual, state['attention_native']):
            raise RuntimeError('Probe prefix trajectory differs before the selected intervention')
        for native, now in zip(state['qkv'], (q, k, v)):
            if not torch.equal(native, now):
                raise RuntimeError('Native Q/K/V drift before value intervention')
        changed = project_query_values(v, state['axis'], state['phase'],
                                       query_index=1, prefix_tokens=prefix)
        if not torch.equal(changed[0], v[0]) or not torch.equal(changed[:, :, :prefix], v[:, :, :prefix]):
            raise RuntimeError('Probe changed support or prefix VALUES')
        if changed is v:
            return actual
        projected = project_heads(this, native_heads(this, qr, kr, changed, attn_mask, is_causal), x, gate)
        delta = actual-projected
        if not torch.equal(delta, state['immediate_'+state['phase']]):
            raise RuntimeError('Replayed immediate response differs from native-side measurement')
        # Prefix QUERY outputs legitimately change; only prefix VALUES are fixed.
        # Preserve the complete original support attention output exactly.
        result = actual.clone()
        result[1] = projected[1]
        return result

    try:
        attention.forward = MethodType(probed, attention)
        handle = block.register_forward_hook(capture_block)
        yield
    finally:
        if handle is not None:
            handle.remove()
        if had_override:
            attention.forward = old_override
        elif 'forward' in attention.__dict__:
            del attention.forward
        state['hook_restored'] = (attention.__dict__.get('forward') is old_override
                                  if had_override else 'forward' not in attention.__dict__)


@torch.no_grad()
def collect_reference_responses(backbone, encode, images, support_coverage, *,
                                grid, layer=11, expected_attention_sha=None,
                                rope_apply=None, allow_cpu_fixture=False):
    """Collect same-shape native/plus/minus outputs without query annotation.

    encode(images) must return [2,C,h,w] from the SAME frozen original encoder.
    backbone is its actual timm object with named EvaAttention blocks and .norm.
    QKV remain transient tensors: this function NEVER writes features to disk.
    Caller owns the source/asset preflight, task split, CRF and decoder training.
    """
    if any(module.training for module in backbone.modules()):
        raise ValueError('Entire frozen encoder must be eval, including dropout/norms')
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[:2] != (2, 3):
        raise ValueError('Original [support,query] RGB batch [2,3,H,W] required')
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError('Finite transformed RGB required')
    if (len(grid) != 2 or any(not isinstance(v, int) or v <= 0 for v in grid)
            or support_coverage.shape != tuple(grid)):
        raise ValueError('Declared positive grid and aligned support coverage required')
    if (not support_coverage.is_floating_point() or not torch.isfinite(support_coverage).all()
            or ((support_coverage < 0) | (support_coverage > 1)).any()):
        raise ValueError('Support coverage must be finite in [0,1]')
    choices = [(name, module) for name, module in backbone.named_modules()
               if type(module).__name__ == 'EvaAttention']
    if not 0 <= layer < len(choices):
        raise ValueError('Selected original attention layer missing')
    selected = choices[layer]
    attention = selected[1]
    if 'forward' in attention.__dict__:
        raise RuntimeError('Fresh original attention required; do not overwrite another forward hook')
    fixture = type(attention).__module__ != 'timm.models.eva'
    if fixture and not (allow_cpu_fixture and images.device.type == 'cpu'):
        raise RuntimeError('Unsupported source; CPU fixture permission cannot enable GPU execution')
    source_sha = attention_source_hash(attention)
    if not fixture and (expected_attention_sha is None or source_sha != expected_attention_sha):
        raise RuntimeError('Original attention source differs from frozen installed-source preflight')
    if rope_apply is None:
        scope = sys.modules.get(type(attention).__module__)
        rope_apply = getattr(scope, 'apply_rot_embed_cat', None)
    state = dict(phase='native', attention_calls=0, block_calls=0, native_exact=False,
                 hook_restored=False)
    outputs = {}
    with _selected_forward(backbone, selected, state, grid, support_coverage, rope_apply):
        for phase in ('native', 'plus', 'minus'):
            state.update(phase=phase, attention_calls=0, block_calls=0)
            output = encode(images)
            if (not isinstance(output, torch.Tensor) or output.ndim != 4
                    or output.shape[0] != 2 or output.shape[-2:] != tuple(grid)
                    or not torch.isfinite(output).all()):
                raise RuntimeError('Original encoder must return finite [2,C,h,w]')
            if state['attention_calls'] != 1 or state['block_calls'] != 1:
                raise RuntimeError('Absent or repeated selected block; cached/unsupported encode callback')
            outputs[phase] = output.detach().clone()
            if phase != 'native' and not torch.equal(output[0], outputs['native'][0]):
                raise RuntimeError('Value probe changed the original reference encoder output')
    if not state['hook_restored']:
        raise RuntimeError('Original attention instance hook was not restored')
    prefix = state['prefix']
    middle = _grid(state['middle'], prefix, grid)
    qkv = [_head_grid(value, prefix, grid) for value in state['qkv']]
    static_middle = torch.cat([middle]+qkv, dim=1)
    prefix_qkv = [v.transpose(1, 2).reshape(v.shape[0], v.shape[2], -1)[:, :prefix]
                  for v in state['qkv']]
    static_prefix = torch.cat([state['middle'][:, :prefix]]+prefix_qkv, dim=-1)
    payload = dict(support_final=outputs['native'][0:1], query_final=outputs['native'][1:2],
                   support_middle=static_middle[0:1], query_middle=static_middle[1:2],
                   support_coverage=support_coverage[None, None].to(images.device),
                   query_immediate_plus=_grid(state['immediate_plus'], prefix, grid)[1:2],
                   query_immediate_minus=_grid(state['immediate_minus'], prefix, grid)[1:2],
                   query_response_plus=outputs['native'][1:2]-outputs['plus'][1:2],
                   query_response_minus=outputs['native'][1:2]-outputs['minus'][1:2],
                   support_prefix=static_prefix[0:1], query_prefix=static_prefix[1:2],
                   prefix_immediate_plus=state['immediate_plus'][1:2, :prefix],
                   prefix_immediate_minus=state['immediate_minus'][1:2, :prefix])
    # Keep the raw per-head prefix layout inspectable as transient metadata too.
    prefix_packet = dict(native_qkv=tuple(v[:, :, :prefix].detach() for v in state['qkv']),
                         immediate_plus=state['immediate_plus'][:, :prefix].detach(),
                         immediate_minus=state['immediate_minus'][:, :prefix].detach())
    audit = dict(state='COLLECTED_THREE_MATCHED_ENCODER_PASSES', layer_index=layer,
                 attention_source_sha256=source_sha, actual_installed_timm=not fixture,
                 CPU_fixture=fixture, batch_context='same [support,query] for all three passes',
                 pure_foreground_patches=state['axis'].foreground_count,
                 pure_background_patches=state['axis'].background_count,
                 enabled_heads=int(state['axis'].enabled.sum()),
                 contrast_norm=state['axis'].contrast_norm.detach().cpu().tolist(),
                 native_replica_exact=state['native_exact'], reference_output_exact=True,
                 original_instance_hook_restored=True, query_GT_received=False,
                 dtype=str(images.dtype), real_task_gain_measured=False,
                 prefix_features_supplied_to_decoder=True,
                 prefix_head_consumption_requires_composition_check=True)
    return dict(features=payload, prefix_information=prefix_packet, audit=audit)
