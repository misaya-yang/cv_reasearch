"""Inference-only Pro30 services over the existing native episode/provider.

The factory binds B before any card can replace reference footprint weights.
Internal observations remain lazy and execute the actual frozen CPU model;
neither legacy processed fields nor final tokens substitute for missing X.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib
import time

import numpy as np

from ics.astra300.common import ArtifactUnavailable, array_hash, readonly


B_METHODS = {1, 13, 16, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}
INTERNAL_METHODS = {8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}
B_RESOURCES = {'pro30_B_operator', 'mean.continuous', 'pro30_B_continuous',
               'pro30_B_producer', 'pro30_MEAN16_system'}
INTERNAL_ALIASES = {'internal_encoder', 'pro30_encoder_context',
                    'pro30_internal_context', 'FrozenSuffixContext'}
EXPLICIT_ONLY = {'pro30_trained_mask_token', 'pro30_mask_token_producer',
                 'pro30_independent_prefix_rope', 'pro30_independent_prefix_rope_producer',
                 'pro30_neutral_prefix_mass_targets', 'pro30_neutral_prefix_mass_producer',
                 'pro30_same_budget_controls'}


def _selected_numbers(modules, selected):
    known = set()
    for name in modules:
        package = 'ics.' + name if name.startswith('pro30.') else 'ics.astra300.' + name
        module = importlib.import_module(package)
        known.update(module.METHODS); known.update(module.CONTROLS)
    if not set(selected) <= known:
        raise ValueError('Runtime factory received an unregistered selected arm')
    numbers = set()
    for name in selected:
        head = name.split('__', 1)[0]
        if head.startswith('PRO30_M') and len(head) == 9 and head[-2:].isdigit():
            numbers.add(int(head[-2:]))
    return numbers


class _LazyArtifacts(dict):
    """Bridge M01's existing direct .get without eagerly preparing its system.

    The system is not enumerated in items(): unrelated cards copying artifacts
    must not trigger a full host calculation. Normal require_artifact accesses
    it through the provider. No original card's solver is modified.
    """
    def __init__(self, values, provider):
        super().__init__(values)
        self.provider = provider

    def get(self, name, default=None):
        if name == 'pro30_MEAN16_system' and name not in self:
            return self.provider.mean_system()
        return super().get(name, default)


class _SharedB:
    """Share original FoRIS/MEAN preparation with the unchanged M01 solver.

    A construction-only CG interception records the exact matrix/rhs/x0
    produced by mean_graph.py. Its dummy return is discarded, never stored as
    B. A requested B field always runs the original convergence solve. When B
    runs first, interception calls the actual solver and retains its system.
    """
    def __init__(self, operator):
        self.operator = operator
        self._system = None
        self._native_result = None
        self.receipt = dict(preparation_calls=0, native_solver_calls=0,
                            native_system_cache_hits=0, actual_new_encoder_forwards=0,
                            preparation_wall_seconds=0., preparation_cpu_seconds=0.,
                            solver_wall_seconds=0., solver_cpu_seconds=0.,
                            query_GT_read=False)

    def __getattr__(self, name):
        return getattr(self.operator, name)

    def _capture(self, function, *, solve):
        from ics.methods import mean_graph
        previous = mean_graph.cg
        captured = {}
        def record(matrix, rhs, *, x0, **options):
            if captured:
                raise ArtifactUnavailable('Expected one unchanged MEAN16 CG construction')
            captured.update(H=matrix.tocsr().copy(), rhs=readonly(rhs),
                            y=readonly(x0), solver_options=dict(options))
            if solve:
                wall, cpu = time.monotonic(), time.process_time()
                result = previous(matrix, rhs, x0=x0, **options)
                captured.update(full_solver_wall_seconds=time.monotonic()-wall,
                                full_solver_cpu_seconds=time.process_time()-cpu)
                return result
            # Only matrix preparation: the returned field is ignored below.
            return np.array(x0, copy=True), 0
        mean_graph.cg = record
        try:
            output = function()
        finally:
            mean_graph.cg = previous
        if not captured or captured['solver_options'] != dict(rtol=1e-7, atol=1e-9, maxiter=300):
            raise ArtifactUnavailable('Actual MEAN16 construction/solver constants differ')
        return captured, output

    def _finish_system(self, captured, producer):
        import torch
        import torch.nn.functional as F
        from . import b_operator as source
        from .certified_01 import _episode_identity, _system_integrity
        from ics.methods import mean_graph
        ep = self.operator.ep
        op = self.operator
        identity = _episode_identity(ep)
        raw_binding = op.bundle['raw_binding']
        source_binding = producer.get('native_feature_pack_binding')
        model = ep.producer.get('model_assets', ep.producer)
        expected_kinds = {'current_native_complete_FoRIS_Part1_Part2_Part3_Part4_MEAN16_system',
                          'current_native_complete_FoRIS_Part1_Part2_Part3_Part4_plus_MEAN_a025_l16_FP32'}
        source_verified = (
            producer.get('kind') in expected_kinds
            and all(mean_graph.CONFIG.get(k) == v for k, v in dict(alpha=.25,
                query_k=20, graph_lambda=16., fidelity_floor=.1, reference_purity=.9).items())
            and producer.get('checkpoint_sha256') == model.get('checkpoint_sha256')
            and producer.get('config_sha256') == model.get('config_sha256', model.get('model_config_sha256'))
            and raw_binding.get('source_image_hashes') == identity['source_image_hashes']
            and raw_binding.get('decoded_original_files_and_mask_weights_verified') is True
            and raw_binding.get('reference_mask_sha256') == ep.producer.get('reference_mask_sha256')
            and source_binding == raw_binding
            and op.closure.get('released_archive_sha256') == identity['locked_host_source_archive_sha256']
            and op.closure.get('released_FoRIS_member_sha256', {}).get('foris/models/foris.py') == source.released.SOURCE_HASHES['foris/models/foris.py']
            and producer.get('native_APD_gate') == op.native_gate
            and producer.get('APD_applied') == op.native_gate
            and (producer.get('native_gate_locked') is True
                 or producer.get('gate_locked_to_original_native_R_Q_MR') is True)
            and producer.get('positional_basis') == op.bundle['basis_binding']
            and producer.get('positional_basis_producer') == op.basis_producer
            and captured.get('solver_options') == dict(rtol=1e-7, atol=1e-9, maxiter=300))
        if not source_verified:
            raise ArtifactUnavailable('M01 system source must be the actual locked native B/verified RQ/MR/basis')
        def work_continuous(z):
            value = torch.from_numpy(np.asarray(z, np.float32).reshape(64, 64))[None, None]
            with torch.inference_mode():
                return F.interpolate(value, (1024, 1024), mode='bilinear',
                                     align_corners=False)[0, 0].numpy()
        def final_mask(work):
            # B has no CRF. This is the retained RCG/MEAN binary-work mapping.
            value = torch.from_numpy(np.asarray(work, np.float32))[None, None]
            with torch.inference_mode():
                return F.interpolate(value, tuple(ep.original_shape), mode='bilinear',
                                     align_corners=False)[0, 0].numpy() > .5
        self._system = dict(captured, native_hw=(64, 64),
                            a=readonly(np.asarray(captured['H'].sum(1)).ravel()),
                            work_continuous=work_continuous, final_mask=final_mask,
                            valid_work=np.ones((1024, 1024), bool),
                            producer=dict(producer, complete_locked_host=source_verified,
                                source_image_hashes=ep.producer['source_image_hashes'],
                                actual_finalizer='B no-CRF FP32continuous→1024>.5→binary-original>.5',
                                exact_system_source='unchanged ics.methods.mean_graph._run CG arguments',
                                system_binding=identity,
                                verified_source_binding=dict(native_pack_sha256=raw_binding['sha256'],
                                    source_image_hashes=raw_binding['source_image_hashes'],
                                    original_MR_byte_sha256=raw_binding['reference_mask_sha256'],
                                    original_MR_array_sha256=identity['original_MR_array_sha256'],
                                    released_source_archive_sha256=op.closure['released_archive_sha256'],
                                    native_gate_locked_to_verified_original_R_Q_MR=True,
                                    positional_basis_sha256=op.bundle['basis_binding']['sha256']),
                                preparation_receipt=self.receipt))
        self._system['producer']['system_payload_integrity'] = _system_integrity(self._system)

    def system(self):
        if self._system is not None:
            self.receipt['native_system_cache_hits'] += 1
            return self._system
        from . import b_operator as source
        from ics.methods import mean_graph
        op = self.operator
        with op._lock, source.released._LOCK:
            if np.any(op.ep.q_valid != 1):
                raise ArtifactUnavailable('M01 same B system requires the complete native physical query canvas')
            wall, cpu = time.monotonic(), time.process_time()
            host, encoder, _, config = source.released._host(
                op.ep, 'foris', 'bilinear', op.bundle['raw'], op.bundle['basis'])
            response, part1 = source._continuous_host(op.ep, host, encoder,
                                                      op.native_gate, op.native_gate)
            if tuple(part1.shape) != (1, 2, 1024, 64, 64):
                raise ArtifactUnavailable('M01 actual source Part1 descriptor geometry changed')
            descriptors = {role: part1[0, j].permute(1, 2, 0).reshape(4096, 1024).numpy().copy()
                           for j, role in enumerate(('r', 'q'))}
            captured, _discarded = self._capture(lambda: mean_graph.predict(
                descriptors['q'], descriptors['r'], op.ep.wf.reshape(64, 64),
                response.numpy(), device='cpu'), solve=False)
            elapsed, cpu_elapsed = time.monotonic()-wall, time.process_time()-cpu
            self.receipt['preparation_calls'] += 1
            self.receipt['preparation_wall_seconds'] += elapsed
            self.receipt['preparation_cpu_seconds'] += cpu_elapsed
            op.stats['complete_FoRIS_executions'] += 1
            op.stats['algorithm_wall_seconds'] += elapsed
            op.stats['algorithm_cpu_seconds'] += cpu_elapsed
            producer = dict(kind='current_native_complete_FoRIS_Part1_Part2_Part3_Part4_MEAN16_system',
                checkpoint_sha256=source.CURRENT_CHECKPOINT_SHA256,
                config_sha256=source.CURRENT_CONFIG_SHA256,
                source_sha256=source.released.SOURCE_HASHES['foris/models/foris.py'],
                source_closure=op.closure, native_feature_pack_binding=op.bundle['raw_binding'],
                positional_basis=op.bundle['basis_binding'], positional_basis_producer=op.basis_producer,
                native_APD_gate=op.native_gate, APD_applied=op.native_gate, native_gate_locked=True,
                FoRIS_continuous_Part4_array_sha256=array_hash(response.numpy()),
                MEAN_graph_descriptor_stage='actual FP32 source Part1; normalized by unchanged MEAN code',
                MEAN_graph_descriptor_array_sha256={k: array_hash(v) for k, v in descriptors.items()},
                host_config={k: v for k, v in config.items() if k != 'encoder'},
                validated_paired_feature_map_reads=encoder.calls, actual_new_encoder_forwards=0,
                query_GT_read=False, full_convergence_solver_prepaid=False,
                MEAN_config=dict(mean_graph.CONFIG), CRF=False, post_solve_minmax=False)
            self._finish_system(captured, producer)
        return self._system

    def __call__(self, ep, reference=None, query=None, *, apd='native', fixed_native_gate=True):
        from . import b_operator as source
        if apd not in ('native', False) or fixed_native_gate is not True:
            raise ArtifactUnavailable('Only the unchanged locked native APD contract is available')
        op = self.operator
        op._same_episode(ep)
        native = ((reference is None or np.array_equal(np.asarray(reference), op.bundle['raw']['r']))
                  and (query is None or np.array_equal(np.asarray(query), op.bundle['raw']['q'])))
        same_gate = apd == 'native' or op.native_gate is False
        if not native or not same_gate:
            return op(ep, reference=reference, query=query, apd=apd, fixed_native_gate=fixed_native_gate)
        with op._lock, source.released._LOCK:
            cache_hit = self._native_result is not None
            if self._native_result is None and self._system is None:
                captured, result = self._capture(lambda: op(ep, reference=reference, query=query,
                    apd=apd, fixed_native_gate=fixed_native_gate), solve=True)
                self._native_result = result
                self._finish_system(captured, result[1]['producer'])
                self.receipt['preparation_calls'] += 1
                self.receipt['native_solver_calls'] += 1
                self.receipt['solver_wall_seconds'] += captured['full_solver_wall_seconds']
                self.receipt['solver_cpu_seconds'] += captured['full_solver_cpu_seconds']
                self.receipt['preparation_wall_seconds'] += max(0., result[1]['this_call_algorithm_wall_seconds']-captured['full_solver_wall_seconds'])
                self.receipt['preparation_cpu_seconds'] += max(0., result[1]['this_call_algorithm_cpu_seconds']-captured['full_solver_cpu_seconds'])
            elif self._native_result is None:
                from ics.methods.mean_graph import cg
                system = self._system
                wall, cpu = time.monotonic(), time.process_time()
                z, status = cg(system['H'], system['rhs'], x0=system['y'], **system['solver_options'])
                if status:
                    raise RuntimeError('Original MEAN16 convergence solve failed: ' + str(status))
                field = readonly(z.reshape(64, 64).astype(np.float32))
                elapsed, elapsed_cpu = time.monotonic()-wall, time.process_time()-cpu
                self.receipt['native_solver_calls'] += 1
                self.receipt['solver_wall_seconds'] += elapsed
                self.receipt['solver_cpu_seconds'] += elapsed_cpu
                op.stats['calls'] += 1; op.stats['MEAN16_executions'] += 1
                op.stats['algorithm_wall_seconds'] += elapsed; op.stats['algorithm_cpu_seconds'] += elapsed_cpu
                producer = dict(system['producer'], kind='current_native_complete_FoRIS_Part1_Part2_Part3_Part4_plus_MEAN_a025_l16_FP32')
                info = dict(complete_FoRIS_MEAN16=True,
                    source_sha256=source.released.SOURCE_HASHES['foris/models/foris.py'],
                    checkpoint_sha256=source.CURRENT_CHECKPOINT_SHA256,
                    config_sha256=source.CURRENT_CONFIG_SHA256, native_gate_locked=True,
                    native_APD_gate=op.native_gate, APD_applied=op.native_gate,
                    requested_APD=apd, original_source_constants_and_quirks_retained=True,
                    this_call_algorithm_wall_seconds=elapsed, this_call_algorithm_cpu_seconds=elapsed_cpu,
                    shared_native_preparation=self.receipt, actual_new_encoder_forwards=0,
                    this_call_new_encoder_forwards=0, exact_B_cache_hit=False,
                    source_closure=op.closure, producer=producer, query_GT_read=False,
                    continuous_field_array_sha256=array_hash(field),
                    field_space='native_physical64', historical_processed_MEAN_parity='not claimed',
                    MEAN_solver=dict(relative_residual=float(np.linalg.norm(system['H']@z-system['rhs']) /
                        max(np.linalg.norm(system['rhs']), 1e-12)), solver_options=system['solver_options']),
                    quality='unknown; no score was read')
                self._native_result = field, info
            else:
                op.stats['calls'] += 1; op.stats['cache_hits'] += 1
            field, info = self._native_result
            returned = deepcopy(info)
            returned.update(requested_APD=apd, exact_B_cache_hit=cache_hit,
                            shared_native_preparation=dict(self.receipt))
            if cache_hit:
                returned.update(this_call_algorithm_wall_seconds=0., this_call_algorithm_cpu_seconds=0.)
            return field, returned


class RuntimeProvider:
    def __init__(self, ep, numbers):
        self.native_episode = ep
        self.delegate = ep.provider
        self.numbers = numbers
        self.operator = None
        self.b_error = None
        self._services = {}
        self._receipt = dict(factory='pro30.runtime_inputs:bind', selected_method_numbers=sorted(numbers),
            B_bound_before_A_C_footprint_changes=True, extra_base_feature_extractions=0,
            initial_extra_encoder_forwards=0, query_GT_read=False, unavailable_services={})
        if numbers & B_METHODS:
            try:
                from .b_operator import bind_B
                bound = bind_B(ep)
                self.operator = _SharedB(bound.artifacts['pro30_B_operator'])
            except Exception as error:
                self.b_error = type(error).__name__ + ': ' + str(error)
                self._receipt['unavailable_services']['complete_B'] = self.b_error

    @property
    def cache(self):
        return self.delegate.cache

    @property
    def receipt(self):
        original = self.delegate.receipt
        if self.operator is not None:
            self._receipt['shared_B'] = dict(self.operator.receipt)
            self._receipt['B_operator_stats'] = dict(self.operator.stats)
        suffix = self._services.get('pro30_internal_context')
        if suffix is not None:
            self._receipt['actual_suffix_context'] = dict(suffix.receipt)
        original['pro30_runtime_inputs'] = self._receipt
        return original

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def _B(self):
        if self.operator is None:
            reason = self.b_error or 'No selected card requested the complete B service'
            raise ArtifactUnavailable('Actual native with_current_basis B binding unavailable: ' + reason)
        return self.operator

    def mean_system(self):
        return self._B().system()

    def require(self, ep, name):
        if name in B_RESOURCES:
            operator = self._B()
            if name == 'pro30_B_operator':
                return operator
            if name == 'pro30_MEAN16_system':
                return operator.system()
            field, info = operator(ep)
            return info['producer'] if name == 'pro30_B_producer' else field
        if name in INTERNAL_ALIASES:
            if not self.numbers & INTERNAL_METHODS:
                raise ArtifactUnavailable('No selected Pro30 card requested actual internal observations')
            if 'internal_encoder' not in self._services:
                self._services['internal_encoder'] = self.delegate.require(self.native_episode, 'internal_encoder')
            internal = self._services['internal_encoder']
            if internal.binding.get('execution_kind') != 'real_frozen_cpu_model':
                raise ArtifactUnavailable('Actual pinned frozen CPU InternalEncoder required')
            if name in ('internal_encoder', 'pro30_encoder_context'):
                return internal
            if 'pro30_internal_context' not in self._services:
                from .internal_17_23 import FrozenSuffixContext
                self._services['pro30_internal_context'] = FrozenSuffixContext(internal)
            return self._services['pro30_internal_context']
        if name in ('pro30_native_query_graph', 'pro30_native_query_graph_producer'):
            if 'pro30_native_query_graph' not in self._services:
                from .runtime_query_graph import native_query_graph
                graph, producer = native_query_graph(ep)
                self._services.update(pro30_native_query_graph=graph,
                                      pro30_native_query_graph_producer=producer)
                self._receipt['native_query_graph'] = producer
            return self._services[name]
        if name in EXPLICIT_ONLY:
            raise ArtifactUnavailable('Exact Pro30 service must be separately source-bound; no substitute: ' + name)
        return self.delegate.require(ep, name)


def bind(ep, row, modules, selected, model_dir=None, threads=1):
    """Known runner factory: preserve native arrays, labels, RGB and provider.

    `row` is already inference-only and its artifacts/native provider are bound.
    The factory loads no new features, downloads nothing, and does no precheck.
    Missing B/X resources fail only the requesting arm, with their exact reason.
    """
    numbers = _selected_numbers(modules, selected)
    if getattr(ep.provider, 'row', None) != row:
        raise ValueError('Pro30 factory must follow binding of the same native input row')
    aliases = (B_RESOURCES | INTERNAL_ALIASES) & set(ep.artifacts)
    if aliases:
        raise ArtifactUnavailable('Runtime service aliases cannot be shadowed by external fields: ' + repr(sorted(aliases)))
    provider = RuntimeProvider(ep, numbers)
    provider._receipt.update(model_directory=model_dir, CPU_threads=int(threads),
        native_provider_reused=True, base_pack_sha256=row['sha256'])
    values = dict(ep.artifacts)
    if provider.operator is not None:
        values['pro30_B_operator'] = provider.operator
    return replace(ep, provider=provider, artifacts=_LazyArtifacts(values, provider))
