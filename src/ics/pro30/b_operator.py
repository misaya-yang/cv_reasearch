"""Source-bound current native FoRIS Part1--4 + fixed MEAN16 operator.

No CRF, old processed field, model construction, or new backbone encoding is
performed here. Changed representations use the original native APD decision
and the same bound U500; their encoder/suffix costs belong to their callers.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ics.astra300 import complete_baselines as released
from ics.astra300.common import ArtifactUnavailable, array_hash, as_episode, readonly
from ics.methods import mean_graph

CURRENT_CHECKPOINT_SHA256 = '45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941'
CURRENT_CONFIG_SHA256 = 'a71f705b0074e173540d0bdbd3aa940fa8d7d3c6c7f020a683004c46ca605b24'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_closure():
    """Every local Python dependency plus the unchanged embedded release."""
    root = Path(__file__).resolve().parents[3]
    local = ['src/ics/pro30/b_operator.py', 'src/ics/astra300/complete_baselines.py',
             'src/ics/astra300/common.py', 'src/ics/cpu100/common.py',
             'src/ics/methods/mean_graph.py', 'src/ics/methods/rcg.py',
             'src/ics/methods/direct_dino_features.py',
             'src/ics/pro30/operators_24_30.py', 'src/ics/pro30/common.py',
             'src/ics/cpu100/invariance_support.py', 'src/ics/__init__.py',
             'src/ics/astra300/__init__.py', 'src/ics/pro30/__init__.py',
             'src/ics/cpu100/__init__.py', 'src/ics/methods/__init__.py']
    released.source_closure()
    return dict(local_python_sha256={name: _sha(root / name) for name in local},
                released_archive_sha256=released.SOURCE_ARCHIVE_SHA256,
                released_FoRIS_member_sha256={name: value for name, value in released.SOURCE_HASHES.items()
                                             if name.startswith('foris/')})


def _basis_producer(ep, bundle):
    assets = released._assets(ep)
    spec = assets['positional_basis']
    path = spec.get('producer_path')
    expected = spec.get('producer_sha256')
    if not path or not expected or _sha(path) != expected:
        raise ArtifactUnavailable('B requires the existing SHA-bound native U500 producer JSON')
    producer = json.loads(Path(path).read_text())
    model = ep.producer.get('model_assets', ep.producer)
    if (producer.get('state') != 'NATIVE_BASIS_FROZEN'
            or producer.get('source_input') != 'normalized_black_image'
            or producer.get('real_encoder_execution') is not True
            or producer.get('synthetic_test_only') is not False
            or producer.get('query_GT_read') is not False
            or producer.get('checkpoint_sha256') != model.get('checkpoint_sha256')
            or producer.get('config_sha256') != model.get('config_sha256', model.get('model_config_sha256'))
            or producer.get('source_sha256') != released.SOURCE_HASHES['foris/models/foris.py']
            or producer.get('basis_file_sha256') != bundle['basis_binding']['sha256']
            or producer.get('basis_array_sha256') != bundle['basis_binding']['array_sha256']):
        raise ArtifactUnavailable('Native basis/cache/current-model/source producer identities differ')
    return dict(path=str(path), sha256=expected, producer=producer)


def _features(value, native, role):
    """Keep exact supplied final-LN values; original FoRIS normalizes them."""
    if value is None:
        return native[role], 'bound native FP32 final-LN before unit/Part1'
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu' or value.dtype not in (torch.float32, torch.float64):
            raise ArtifactUnavailable('B changed features must be actual CPU FP32/explicit FP64 arrays')
        value = value.detach().numpy()
    value = np.asarray(value)
    if (value.shape != (4096, 1024) or value.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(value).all()):
        raise ArtifactUnavailable('B requires finite same-coordinate final-LN [4096,1024] ' + role)
    return np.asarray(value, dtype=np.float32).copy(), 'caller-supplied current-model final-LN representation, source FP32 conversion'


def _native_gate(ep, host, encoder):
    """Run the unchanged source decision on native R/Q and original MR once."""
    host.set_reference(Image.fromarray(np.asarray(ep.r_rgb, np.uint8), 'RGB'),
                       torch.from_numpy(ep.reference_mask.copy()))
    host.set_target(Image.fromarray(np.asarray(ep.q_rgb, np.uint8), 'RGB'))
    encoder.verify_host_inputs(host)
    maps = torch.cat((encoder.maps['r'], encoder.maps['q']), dim=0)[None]
    with torch.inference_mode(), torch.autocast('cpu', enabled=False):
        normalized = F.normalize(maps, p=2, dim=2)
        gate = host._should_apply_positional_debias(normalized, host._ref_masks.unsqueeze(1), 1)
    host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None
    return bool(gate)


def _continuous_host(ep, host, encoder, native_gate, apply_apd):
    """Original predict executes all four stages; stop before its mask step.

The source body and Part1--4 functions are unchanged. Only the source decision
is locked and the two final mask callbacks are replaced by continuous capture.
Instance callbacks are restored even when a source stage raises an exception.
"""
    if apply_apd and not native_gate:
        raise ArtifactUnavailable('B cannot enable APD against a false locked native gate')
    names = ('_should_apply_positional_debias', '_part1_positional_debias',
             '_binarize_response', '_finalize_mask')
    old = {name: getattr(host, name) for name in names}
    captured = {}

    def gate(*_args, **_kwargs):
        captured['gate_calls'] = captured.get('gate_calls', 0) + 1
        host.should_debiass = apply_apd
        return apply_apd

    def part1(*args, **kwargs):
        value = old['_part1_positional_debias'](*args, **kwargs)
        captured['post_part1'] = value.detach().clone()
        return value

    def continuous(score_hw, *, target_hw):
        if tuple(target_hw) != (1024, 1024):
            raise ArtifactUnavailable('B source work geometry changed')
        captured['response'] = score_hw.detach().clone()
        return score_hw

    host._should_apply_positional_debias = gate
    host._part1_positional_debias = part1
    host._binarize_response = continuous
    host._finalize_mask = lambda value, _target: value
    try:
        with torch.inference_mode(), torch.autocast('cpu', enabled=False):
            output = released._run_host(ep, host, encoder)
        if (captured.get('gate_calls') != 1 or 'post_part1' not in captured
                or 'response' not in captured or output.dtype != torch.float32
                or tuple(output.shape) != (64, 64) or not torch.isfinite(output).all()):
            raise ArtifactUnavailable('Complete source Part1--4 continuous FP32 execution missing')
        if not torch.equal(output, captured['response']):
            raise ArtifactUnavailable('Source Part4 response changed before continuous capture')
        return captured['response'], captured['post_part1']
    finally:
        for name, function in old.items():
            setattr(host, name, function)
        host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None


class BOperator:
    """One legal native episode, its APD gate/U, and a small exact-result cache."""

    def __init__(self, native_episode):
        model = native_episode.producer.get('model_assets', native_episode.producer)
        config_hash = model.get('config_sha256', model.get('model_config_sha256'))
        if model.get('checkpoint_sha256') != CURRENT_CHECKPOINT_SHA256 or config_hash != CURRENT_CONFIG_SHA256:
            raise ArtifactUnavailable('Pro30 B is bound to the current verified checkpoint/config')
        setup_wall, setup_cpu = time.monotonic(), time.process_time()
        self.ep = native_episode
        self.bundle = released._episode_bundle(native_episode)
        self.basis_producer = _basis_producer(native_episode, self.bundle)
        self.closure = source_closure()
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._exact_area = None
        self.stats = dict(calls=0, cache_hits=0, complete_FoRIS_executions=0,
                          MEAN16_executions=0, actual_new_encoder_forwards=0,
                          native_gate_evaluations=0, algorithm_wall_seconds=0.,
                          algorithm_cpu_seconds=0.)
        host, encoder, _, _ = released._host(native_episode, 'foris', 'bilinear',
                                             self.bundle['raw'], self.bundle['basis'])
        self.native_gate = _native_gate(native_episode, host, encoder)
        self.stats.update(native_gate_evaluations=1, setup_wall_seconds=time.monotonic()-setup_wall,
                          setup_cpu_seconds=time.process_time()-setup_cpu)

    def _same_episode(self, ep):
        if (ep.source_id != self.ep.source_id or ep.producer != self.ep.producer
                or ep.query_geometry != self.ep.query_geometry
                or ep.reference_geometry != self.ep.reference_geometry
                or tuple(ep.original_shape) != tuple(self.ep.original_shape)
                or ep.q_hw != self.ep.q_hw or ep.r_hw != self.ep.r_hw):
            raise ArtifactUnavailable('A bound B operator cannot be reused on a different episode/geometry')
        for name in ('q', 'r', 'q_rgb', 'r_rgb', 'reference_mask', 'q_valid'):
            value, original = getattr(ep, name), getattr(self.ep, name)
            if value is not original and (value is None or not np.array_equal(value, original)):
                raise ArtifactUnavailable('B native source changed: ' + name)
        if not np.array_equal(ep.wvalid, self.ep.wvalid):
            raise ArtifactUnavailable('B native valid reference domain changed')
        if not np.array_equal(ep.wf, self.ep.wf):
            # M24--30 independently use exact original-pixel footprint areas for
            # A/C calibration. B retains its own original cached-MR coverage.
            if self._exact_area is None:
                from .operators_24_30 import exact_footprint
                self._exact_area = exact_footprint(self.ep.reference_mask, self.ep.r_hw,
                                                  self.ep.reference_geometry)
            area, valid = self._exact_area
            if not np.allclose(ep.wf, area, atol=1e-12, rtol=0) or not np.allclose(ep.wvalid, valid, atol=1e-12, rtol=0):
                raise ArtifactUnavailable('B reference labels differ from either its cache or the exact same original MR footprint')

    def __call__(self, ep, reference=None, query=None, *, apd='native', fixed_native_gate=True):
        if apd not in ('native', False) or fixed_native_gate is not True:
            raise ArtifactUnavailable('Only fixed original-native APD gate or explicit APD=False is defined')
        with self._lock, released._LOCK:
            self._same_episode(ep)
            r, r_kind = _features(reference, self.bundle['raw'], 'r')
            q, q_kind = _features(query, self.bundle['raw'], 'q')
            apply_apd = self.native_gate if apd == 'native' else False
            key = (array_hash(r), array_hash(q), apply_apd)
            self.stats['calls'] += 1
            if key in self._cache:
                self.stats['cache_hits'] += 1
                field, cached_info = self._cache[key]
                self._cache.move_to_end(key)
                info = deepcopy(cached_info)
                info.update(exact_B_cache_hit=True, this_call_new_encoder_forwards=0,
                            this_call_algorithm_wall_seconds=0., this_call_algorithm_cpu_seconds=0.,
                            requested_APD=apd, supplied_feature_stages=dict(r=r_kind, q=q_kind))
                info['producer']['supplied_feature_stages'] = dict(r=r_kind, q=q_kind)
                return field, info
            wall, cpu = time.monotonic(), time.process_time()
            host, encoder, _, config = released._host(self.ep, 'foris', 'bilinear',
                                                       dict(r=r, q=q), self.bundle['basis'])
            response, post_part1 = _continuous_host(self.ep, host, encoder, self.native_gate, apply_apd)
            if tuple(post_part1.shape) != (1, 2, 1024, 64, 64):
                raise ArtifactUnavailable('Original Part1 descriptor geometry changed')
            descriptors = {role: post_part1[0, index].permute(1, 2, 0).reshape(4096, 1024).numpy().copy()
                           for index, role in enumerate(('r', 'q'))}
            field, solve = mean_graph.predict(descriptors['q'], descriptors['r'],
                                              self.ep.wf.reshape(64, 64), response.numpy(), device='cpu')
            if field.dtype != np.float32 or field.shape != (64, 64) or not np.isfinite(field).all():
                raise ArtifactUnavailable('Actual complete MEAN16 continuous field missing')
            elapsed, elapsed_cpu = time.monotonic()-wall, time.process_time()-cpu
            self.stats['complete_FoRIS_executions'] += 1
            self.stats['MEAN16_executions'] += 1
            self.stats['algorithm_wall_seconds'] += elapsed
            self.stats['algorithm_cpu_seconds'] += elapsed_cpu
            native_inputs = np.array_equal(r, self.bundle['raw']['r']) and np.array_equal(q, self.bundle['raw']['q'])
            producer = dict(kind='current_native_complete_FoRIS_Part1_Part2_Part3_Part4_plus_MEAN_a025_l16_FP32',
                            checkpoint_sha256=CURRENT_CHECKPOINT_SHA256, config_sha256=CURRENT_CONFIG_SHA256,
                            source_image_hashes=self.bundle['raw_binding']['source_image_hashes'],
                            reference_mask_sha256=self.bundle['raw_binding']['reference_mask_sha256'],
                            native_feature_pack_binding=self.bundle['raw_binding'],
                            actual_input_representation_array_sha256=dict(r=key[0], q=key[1]),
                            input_is_bound_native_cache=bool(native_inputs),
                            supplied_feature_stages=dict(r=r_kind, q=q_kind),
                            native_APD_gate=self.native_gate, APD_applied=apply_apd,
                            gate_locked_to_original_native_R_Q_MR=True,
                            positional_basis=self.bundle['basis_binding'],
                            positional_basis_producer=self.basis_producer,
                            source_closure=self.closure,
                            FoRIS_continuous_Part4_array_sha256=array_hash(response.numpy()),
                            MEAN_graph_descriptor_stage='source channel-normalized Part1 output; no Part2 feature gate; FP32, no FP16 round-trip',
                            MEAN_graph_descriptor_array_sha256={role: array_hash(value) for role, value in descriptors.items()},
                            MEAN_reference_coverage='original native pack PIL-nearest1024 then exact16x16 patch average, verified against the same complete original MR; caller A/C footprints do not replace B labels',
                            MEAN_config=deepcopy(mean_graph.CONFIG), post_solve_minmax=False,
                            CRF=False, extra_encoder_forwards=0, query_GT_read=False)
            info = dict(complete_FoRIS_MEAN16=True, source_sha256=released.SOURCE_HASHES['foris/models/foris.py'],
                        checkpoint_sha256=CURRENT_CHECKPOINT_SHA256, config_sha256=CURRENT_CONFIG_SHA256,
                        native_gate_locked=True, native_APD_gate=self.native_gate, APD_applied=apply_apd,
                        requested_APD=apd, original_source_constants_and_quirks_retained=True,
                        host_config={name: value for name, value in config.items() if name != 'encoder'},
                        dependencies=dict(released._dependencies(), numpy=np.__version__, scipy=__import__('scipy').__version__),
                        validated_paired_feature_map_reads=encoder.calls,
                        cache_native_feature_reads=encoder.calls if native_inputs else 0,
                        supplied_representation_reads=0 if native_inputs else encoder.calls,
                        actual_new_encoder_forwards=0,
                        this_call_new_encoder_forwards=0, reference_and_query_RGB_source_verified=True,
                        supplied_feature_stages=dict(r=r_kind, q=q_kind),
                        additional_representation_cost='caller encoder/suffix/JVP cost is external and must be reported separately; it is not inferred from these arrays',
                        exact_B_cache_hit=False, this_call_algorithm_wall_seconds=elapsed,
                        this_call_algorithm_cpu_seconds=elapsed_cpu,
                        MEAN_solver=solve, continuous_field_array_sha256=array_hash(field),
                        field_space='native_physical64', renderer='caller source FP32 bilinear64-to1024 then strict threshold; fixed binary-to-original',
                        source_closure=self.closure, producer=producer, query_GT_read=False,
                        original_native_extraction_reused=True, historical_processed_MEAN_parity='not claimed',
                        quality='unknown; no score was read')
            field = readonly(field)
            self._cache[key] = (field, deepcopy(info))
            while len(self._cache) > 4:
                self._cache.popitem(last=False)
            return field, info


class _BProvider:
    """Lazy common aliases, retaining the original encoder/provider bookkeeping."""

    def __init__(self, operator, delegate):
        self.operator, self.delegate = operator, delegate
        self._cache = {} if delegate is None else delegate.cache
        self._receipt = {'query_GT_read': False} if delegate is None else delegate.receipt

    @property
    def cache(self):
        return self._cache

    @property
    def receipt(self):
        self._receipt['pro30_B_operator_stats'] = dict(self.operator.stats)
        return self._receipt

    def __getattr__(self, name):
        if self.delegate is None:
            raise AttributeError(name)
        return getattr(self.delegate, name)

    def require(self, ep, name):
        if name == 'pro30_B_operator':
            return self.operator
        if name in ('mean.continuous', 'pro30_B_continuous', 'pro30_B_producer'):
            field, info = self.operator(ep)
            self._receipt['pro30_B_native_producer'] = info['producer']
            return info['producer'] if name == 'pro30_B_producer' else field
        if self.delegate is None:
            raise ArtifactUnavailable('No original provider for B-external resource: ' + name)
        return self.delegate.require(ep, name)


def build_B_operator(native_episode):
    """Build using the existing raw1024 pack + current original U producer."""
    return BOperator(native_episode)


def bind_B(native_episode, *, assets=None):
    """Attach lazy B and mean.continuous; keep models and old host resources apart."""
    conflicting = {'mean.continuous', 'pro30_B_continuous', 'pro30_B_producer'} & set(native_episode.artifacts)
    if conflicting:
        raise ArtifactUnavailable('Prebound fields would shadow the actual lazy B operator: ' + repr(sorted(conflicting)))
    if assets is not None:
        native_episode = released.bind_assets(native_episode, assets)
    operator = build_B_operator(native_episode)
    return as_episode(native_episode,
                      artifacts=dict(native_episode.artifacts, pro30_B_operator=operator),
                      provider=_BProvider(operator, native_episode.provider))
