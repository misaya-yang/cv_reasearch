"""Shared, training-free source runtime from RGB, reference mask and DINO O24.

B is complete official FoRIS+CRF; M is original MEAN; U is original R1 VP_UN;
T is the original soft-coverage c readout. Optional I/J share one complete
official INSID3 core, with bilinear/CRF640 finalization. No previous score/mask
is an input. Boolean expressions consume the freshly computed source masks.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import builtins
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import types

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit

REPO = Path(os.environ.get('CV_RESEARCH_ROOT', Path.cwd())).resolve()
ASSETS = REPO.parent / 'cv_data'
BASE_OUT = ASSETS / 'a/x1012s2_runtime'
OUT = Path(os.environ.get('X1012S2_RUNTIME_OUT', BASE_OUT / 'mfu_boundary')).resolve()
RUNTIME = OUT / 'runtime'
R1 = ASSETS / 'a/x1012_R1'
R5 = ASSETS / 'a/x1012_R5_validation'
DOMAINS = ('coco', 'lvis', 'pascal_part', 'paco_part', 'suim', 'isaid', 'isic', 'lung', 'fundus', 'deepglobe_road')
CORE_SOURCES = ('B', 'M', 'U', 'T')
SOURCE_NAMES = CORE_SOURCES + ('I', 'J')
DEFAULT_RULES = ('(B | M | T) & U',
                 '(B & U) | (M & U) | (M & T) | (U & T)',
                 '(M & U) | (M & T) | (U & T) | (B & U & I) | (B & T & I) | (B & U & J) | (B & T & J)')
DEFAULT_LINEAR_RULES = ({'name': 'soft_M025_U050_T025', 'weights': {'M': .25, 'U': .5, 'T': .25}, 'threshold': .5},)
MFU_RULE = {'name': 'MFU', 'weights': {'M': .5, 'U': .5}, 'threshold': .5}
BOUNDARY_MODES = ('none', 'full_crf', 'add_only', 'delete_only', 'area16')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([list(value.shape), value.dtype.str]).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def case_key(binding):
    return str(binding['index']) if isinstance(binding['index'], str) else f"{binding['index']:06d}"


class BooleanRule:
    """Arbitrary Boolean expression structure, with no chain-length limit.

    Accepts names from the supplied source mapping, parentheses, &, |, ^, ~,
    set difference (-), and the equivalent and/or/not syntax. No Python eval.
    """
    def __init__(self, expression):
        self.expression = expression
        self.tree = ast.parse(expression, mode='eval').body
        self.symbols = set()
        self._validate(self.tree)

    def _validate(self, node):
        if isinstance(node, ast.Name):
            self.symbols.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (bool, int)) and node.value in (False, True, 0, 1):
            pass
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Invert, ast.Not)):
            self._validate(node.operand)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.Sub)):
            self._validate(node.left)
            self._validate(node.right)
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for value in node.values:
                self._validate(value)
        else:
            raise ValueError('Only Boolean names/constants/operators are allowed')

    def __call__(self, sources):
        missing = self.symbols - sources.keys()
        if missing:
            raise KeyError(f'Missing runtime sources: {sorted(missing)}')
        def apply(node):
            if isinstance(node, ast.Name):
                return sources[node.id]
            if isinstance(node, ast.Constant):
                return bool(node.value)
            if isinstance(node, ast.UnaryOp):
                return np.logical_not(apply(node.operand))
            if isinstance(node, ast.BoolOp):
                result = apply(node.values[0])
                operation = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
                for value in node.values[1:]:
                    result = operation(result, apply(value))
                return result
            left, right = apply(node.left), apply(node.right)
            if isinstance(node.op, ast.BitAnd):
                return np.logical_and(left, right)
            if isinstance(node.op, ast.BitOr):
                return np.logical_or(left, right)
            if isinstance(node.op, ast.BitXor):
                return np.logical_xor(left, right)
            return np.logical_and(left, np.logical_not(right))
        result = np.asarray(apply(self.tree), dtype=bool)
        shape = np.shape(next(iter(sources.values())))
        return np.broadcast_to(result, shape).copy()


class LinearRule:
    """Fixed FP64 weighted readouts; no normalization, clipping or fitting."""
    def __init__(self, specification):
        self.name = specification['name']
        self.weights = {name: float(weight) for name, weight in specification['weights'].items() if weight != 0}
        self.symbols = set(self.weights)
        self.threshold = float(specification.get('threshold', .5))
        assert self.weights and all(np.isfinite(x) for x in self.weights.values()) and np.isfinite(self.threshold)

    def __call__(self, frames):
        value = np.zeros_like(next(iter(frames.values())), dtype=np.float64)
        for name, weight in self.weights.items():
            value += weight * np.asarray(frames[name], np.float64)
        return value > self.threshold


def prepare(expressions, linear_specifications=DEFAULT_LINEAR_RULES):
    if (OUT / 'config.json').exists():
        config = check()
        assert config['expressions'] == list(expressions) and config['linear_rules'] == list(linear_specifications), 'Preserve frozen configuration; use another X1012S2_RUNTIME_OUT for other rules'
        return config
    for expression in expressions:
        rule = BooleanRule(expression)
        assert rule.symbols <= set(SOURCE_NAMES), f'Available runtime providers: {SOURCE_NAMES}'
    for specification in linear_specifications:
        assert LinearRule(specification).symbols <= set(SOURCE_NAMES)
    parent_rows = read(R1 / 'manifest.json')
    rows = [next(b for b in parent_rows if b['dataset'] == d) for d in DOMAINS]
    controls = {r['index']: r for r in read(ASSETS / 'a/x1012_B4/seed150/controls/sealed.json')['records']}
    r1 = {r['index']: r for r in read(R1 / 'sealed.json')['records']}
    insid3 = {r['index']: r for r in read(ASSETS / 'a/x1012_R6_insid3/development1360/sealed.json')['records']}
    soft_root = ASSETS / 'a/x1012s2_soft_composition'
    soft = {r['index']: r for r in read(soft_root / 'sealed.json')['records']}
    cases = []
    for binding in rows:
        name = case_key(binding)
        c, u = controls[name], r1[name]
        archives = {symbol: {'path': c['path'], 'sha256': c['sha256'], 'key': actual}
                    for symbol, actual in (('B', 'FoRIS'), ('M', 'MEAN'))}
        archives.update({symbol: {'path': u['prediction_path'], 'sha256': u['prediction_sha256'], 'key': actual}
                         for symbol, actual in (('U', 'VP_UN'), ('T', 'T'))})
        official = insid3[name]
        archives.update({symbol: {'path': official['prediction_path'], 'sha256': official['prediction_sha256'], 'key': actual}
                         for symbol, actual in (('I', 'cli/insid3.bilinear'), ('J', 'cli/insid3.crf640'))})
        mixture = soft[name]
        cases.append({'binding': binding, 'archives': archives,
                      'linear_archives': {'soft_M025_U050_T025': {'path': mixture['prediction_path'], 'sha256': mixture['prediction_sha256'], 'key': 'b0m1u2t1'}},
                      'continuous_sources': read(soft_root / 'source_records' / (name + '.json'))})
    sources = {}
    names = ('lab.py', 'plan_common.py', 'ics/__init__.py', 'ics/foris.py', 'ics/official_data.py',
             'ics/native_basis.py', 'ics/m4_crf_cached.py', 'ics/methods/__init__.py',
             'ics/methods/mean_control.py', 'ics/methods/rcg.py')
    for name in names:
        sources[name] = R5 / 'runtime' / name
    for folder in ('models', 'utils'):
        for path in sorted((ASSETS / 'third_party/foris_official' / folder).glob('*.py')):
            sources[str(Path('official') / folder / path.name)] = path
    snapshots = []
    for name, origin in sources.items():
        target = RUNTIME / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        snapshots.append({'source': str(origin), 'snapshot': str(target), 'sha256': sha(target)})
    for folder in ('models', 'utils'):
        path = RUNTIME / 'official' / folder / '__init__.py'
        if not path.exists():
            path.write_text('')
            snapshots.append({'source': 'namespace marker', 'snapshot': str(path), 'sha256': sha(path)})
    # Isolate official INSID3 imports, including dynamic refinement imports.
    for folder in ('models', 'utils'):
        for origin in sorted((ASSETS / 'third_party/INSID3' / folder).glob('*.py')):
            target = RUNTIME / 's2_insid3' / folder / origin.name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = origin.read_text().replace('from utils.', 'from s2_insid3.utils.').replace('from models.', 'from s2_insid3.models.')
            target.write_text(content)
            snapshots.append({'source': str(origin), 'source_sha256': sha(origin), 'snapshot': str(target), 'sha256': sha(target),
                              'change': 'imports only: utils/models -> s2_insid3.utils/models'})
    for relative in ('s2_insid3/__init__.py', 's2_insid3/utils/__init__.py'):
        path = RUNTIME / relative
        if not path.exists():
            path.write_text('')
            snapshots.append({'source': 'namespace marker', 'snapshot': str(path), 'sha256': sha(path)})
    # Extract only the four pure numerical functions. No image-descriptor or
    # fitted pixel-refinement source is copied, imported, or executed.
    un_source = ASSETS / 'a/x1012_R5_candidate/frozen/r5_candidate.py'
    code = un_source.read_text()
    functions = {node.name: ast.get_source_segment(code, node) for node in ast.parse(code).body
                 if isinstance(node, ast.FunctionDef)}
    selected = ('quality', 'modulate', 'graph', 'solve')
    header = ('import numpy as np\nimport torch\nimport torch.nn.functional as F\n'
              'from scipy import sparse\nfrom scipy.sparse.linalg import cg\nimport lab\nimport plan_common as pc\n'
              'STRENGTHS=tuple(float(x) for x in np.linspace(0,1,9))\n\n')
    target = RUNTIME / 'un_pure.py'
    target.write_text(header + '\n\n'.join(functions[name] for name in selected) + '\n')
    snapshots.append({'source': str(un_source), 'source_sha256': sha(un_source), 'selected_functions': selected,
                      'snapshot': str(target), 'sha256': sha(target)})
    boundary_source = REPO / 'scripts/x1012s2_boundary_composition.py'
    boundary_code = boundary_source.read_text()
    boundary_functions = {node.name: ast.get_source_segment(boundary_code, node) for node in ast.parse(boundary_code).body
                          if isinstance(node, ast.FunctionDef)}
    selected = ('blocks', 'unblocks', 'area_preserving_reorder')
    target = RUNTIME / 'boundary_pure.py'
    target.write_text('import numpy as np\n\n' + '\n\n'.join(boundary_functions[name] for name in selected) + '\n')
    snapshots.append({'source': str(boundary_source), 'source_sha256': sha(boundary_source), 'selected_functions': selected,
                      'change': 'unchanged AST source segments; imports limited to numpy',
                      'snapshot': str(target), 'sha256': sha(target)})
    (OUT / 'frozen').mkdir(exist_ok=True)
    shutil.copyfile(__file__, OUT / 'frozen/x1012s2_runtime.py')
    write(OUT / 'manifest.json', cases)
    write(OUT / 'runtime_sources.json', snapshots)
    config = {'state': 'SHARED_RUNTIME_FROZEN_BEFORE_PARITY_AND_COST', 'n': 10,
              'selection': 'first R1 manifest task in each of ten named domains; no GT selection',
              'sources': {'B': 'complete official FoRIS+CRF', 'M': 'original MEAN FP16 handoff and original FoRIS score',
                          'U': 'original R1 VP_UN strict midrange>.5 and rho>.9 rawT fallback',
                          'T': 'independent soft-reference-coverage c, bilinear1024,minmax,strict>.5',
                          'I': 'complete official INSID3 core with bilinear output',
                          'J': 'same INSID3 core and official CRF640 output'},
              'expressions': list(expressions), 'Boolean_rule_interface': 'arbitrary validated Boolean AST; no number-of-steps restriction',
              'linear_rules': list(linear_specifications),
              'candidate_formulas': {'MFU': 'FP64(.5*M + .5*U) > .5',
                                     'gated_MFU': 'reference rho > .9 ? T : MFU'},
              'candidate_dependency_optimization': 'Reference coverage/rho is computed before source dispatch. Gated true runs only c/T, with no FoRIS/M/VP/UN. Otherwise MFU uses FoRIS semantic core without CRF plus M/U.',
              'candidate_boundary_modes': {'none': 'MFU mask',
                                           'full_crf': 'one unchanged official FoRIS1024 CRF on the binary MFU mask',
                                           'add_only': 'MFU OR that CRF output',
                                           'delete_only': 'MFU AND that CRF output',
                                           'area16': 'existing exact area_preserving_reorder(MFU, CRF, FP64 continuous MFU p)'},
              'candidate_boundary_gate': 'gated_MFU with rho>.9 returns only T and skips boundary/CRF',
              'linear_readouts': 'B/I/J are binary0/1; M is original bilinear1024 with no minmax/clip; U/T are original FP32 bilinear1024 then minmax. FP64 weighted sum, strict threshold, no final normalization or CRF.',
              'dependency_optimization': 'M consumes FoRIS semantic score, not its CRF output. When B is unused, FoRIS runs its official bilinear finalizer and skips CRF. I/J share a core; I-only skips INSID3 CRF.',
              'sharing': ['one verified raw O24 load per distinct input', 'same decoded RGB/reference mask and transformed query',
                          'unit and unconditional positional projection shared by M/U/T', 'c shared by U and T',
                          'complete B score reused by M; no archive score read',
                          'I/J share one INSID3 core and the same raw O24 store; B normalized inputs reused through the official tensor-input API',
                          'M FP16 graph and U FP32 graph remain separate to preserve their formulas',
                          'B conditional-APD numerical path and U independent V/P are not falsely equated'],
              'no_fitting': True, 'no_pixel_refinement_code': True, 'new_encoder_calls': 0, 'query_GT_in_inference': False,
              'cpu_threads': 2, 'torch_interop_threads': 1, 'workers': 1, 'device': 'cpu',
              'timer': 'complete serial cached-input runtime including raw verification/I/O, RGB/reference-mask load, requested source providers and Boolean results; excluding one-time module/host/basis setup and artifact serialization',
              'benchmark': 'one untimed complete warmup; ten paired B-only/binary-four/binary-six/soft-MUT trials with rotating order; image-dependent CRF bilateral geometry cleared before each timed inference',
              'parity': 'compare all six natural masks after inference with sealed archives; report every XOR without replacing computed output',
              'source_sha256': sha(__file__), 'manifest_sha256': sha(OUT / 'manifest.json'),
              'runtime_sources_sha256': sha(OUT / 'runtime_sources.json'),
              'basis_sha256': sha(ASSETS / 'native_assets/positional_basis.pt'),
              'host': {'platform': platform.platform(), 'cpu_count': os.cpu_count()}}
    write(OUT / 'config.json', config)
    print({'state': config['state'], 'n': 10, 'expressions': expressions, 'config_sha256': sha(OUT / 'config.json')}, flush=True)
    return config


def check():
    config = read(OUT / 'config.json')
    assert config['source_sha256'] == sha(__file__)
    assert config['manifest_sha256'] == sha(OUT / 'manifest.json')
    assert config['runtime_sources_sha256'] == sha(OUT / 'runtime_sources.json')
    assert all(sha(row['snapshot']) == row['sha256'] for row in read(OUT / 'runtime_sources.json'))
    assert config['basis_sha256'] == sha(ASSETS / 'native_assets/positional_basis.pt')
    return config


@contextmanager
def inference_files(binding, archive_paths=()):
    """Deny query annotations and every non-input NPZ during timed inference."""
    row = binding['row']
    allowed_raw = {Path(binding['fit'][name]['payload_path']).resolve()
                   for name in ('reference_raw', 'query_raw_identity')}
    reference = Path(row['reference_mask_path']).resolve()
    denied = {Path(row[k]).resolve() for k in ('query_mask_path', 'query_ignore_mask_path') if row.get(k)} - {reference}
    denied.update(Path(path).resolve() for path in archive_paths)
    opened = []
    original = builtins.open, io.open, os.open
    def wrap(function):
        def guarded(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                resolved = Path(os.fsdecode(path)).resolve()
                if resolved in denied or (resolved.suffix == '.npz' and resolved not in allowed_raw):
                    raise RuntimeError('Only RGB/reference-mask/raw-O24 inputs are legal in timed inference: ' + str(resolved))
                opened.append(str(resolved))
            return function(path, *args, **kwargs)
        return guarded
    builtins.open, io.open, os.open = [wrap(function) for function in original]
    try:
        with torch.inference_mode():
            yield opened
    finally:
        builtins.open, io.open, os.open = original


class SharedReplay(torch.nn.Module):
    encoder = None
    encoder_calls = 0
    def __init__(self):
        super().__init__()
        self.clear()

    def clear(self):
        self.raw, self.maps, self.input_to_role = {}, {}, {}
        self.logical_requests, self.unique_payloads, self.logical_bytes = 0, set(), 0
        self.model_inputs = None
        self.provided_inputs = 0

    def bind(self, binding, provided=None):
        self.clear()
        loaded = {}
        for role, name in (('reference', 'reference_raw'), ('query', 'query_raw_identity')):
            metadata = binding['fit'][name]
            payload = Path(metadata['payload_path'])
            entry = Path(metadata.get('entry_path', payload.parent / 'entry.json'))
            info = read(entry)
            if metadata.get('entry_sha256'):
                assert sha(entry) == metadata['entry_sha256']
            assert payload.name == info['file']
            if provided is not None and role in provided:
                value = torch.as_tensor(provided[role]).detach().cpu()
                assert value.shape == (4096, 1024) and value.dtype == torch.float32 and torch.isfinite(value).all()
                self.raw[role] = value
                self.provided_inputs += 1
            elif str(payload) not in loaded:
                assert sha(payload) == info['file_sha256']
                if metadata.get('payload_sha256'):
                    assert metadata['payload_sha256'] == info['file_sha256']
                with np.load(payload, allow_pickle=False) as saved:
                    value = saved['O/24'].copy()
                assert value.shape == (4096, 1024) and value.dtype == np.float32 and np.isfinite(value).all()
                assert array_hash(value) == info['features']['O/24']['tensor_sha256']
                loaded[str(payload)] = torch.from_numpy(value)
                self.unique_payloads.add(str(payload))
                self.logical_bytes += payload.stat().st_size
                self.raw[role] = loaded[str(payload)]
            else:
                self.raw[role] = loaded[str(payload)]
            self.maps[role] = self.raw[role].reshape(64, 64, 1024).permute(2, 0, 1).contiguous()
            self.input_to_role[info['input_tensor_hash']] = role

    def get_intermediate_layers(self, inputs, n=1, reshape=True):
        assert n == 1 and reshape and inputs.dtype == torch.float32 and inputs.device.type == 'cpu'
        self.model_inputs = inputs.detach()
        result = []
        for value in inputs.numpy():
            digest = array_hash(value)
            assert digest in self.input_to_role, 'Actual transformed RGB has no matching cached O24'
            result.append(self.maps[self.input_to_role[digest]])
            self.logical_requests += 1
        return [torch.stack(result)]


class Runtime:
    def __init__(self, expressions, linear_specifications=DEFAULT_LINEAR_RULES):
        started = time.perf_counter()
        sys.path.insert(0, str(RUNTIME))
        self.lab = importlib.import_module('lab')
        self.lab.ROOT, self.lab.ASSETS = REPO, ASSETS
        self.lab.BASIS = ASSETS / 'native_assets/positional_basis.pt'
        self.lab.DEV = torch.device('cpu')
        self.pc = importlib.import_module('plan_common')
        sys.path[:0] = [str(RUNTIME), str(RUNTIME / 'official')]
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        native = self.pc.extension('ics_permutohedral_m4', ASSETS / 'runtime/macos/crf/ics_permutohedral_m4.so')
        prepared = self.pc.extension('ics_permutohedral_m4_prepared_v1', ASSETS / 'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so')
        class Layer(torch.nn.Module):
            def __init__(self, bilateral, theta_alpha, theta_beta, theta_gamma, nhwc=True):
                super().__init__()
                self.bilateral, self.parameters_ = bilateral, (theta_alpha, theta_beta, theta_gamma)
            def forward(self, value, image):
                value = value.detach().float().permute(0, 2, 3, 1).contiguous()
                image = image.detach().float().permute(0, 2, 3, 1).contiguous()
                return native.forward(value, image, self.bilateral, *self.parameters_).permute(0, 3, 1, 2).contiguous()
        module = types.ModuleType('PermutohedralFiltering')
        module.PermutohedralLayer = Layer
        sys.modules['PermutohedralFiltering'] = module
        sys.path.insert(0, str(ASSETS / 'third_party/crf_source/src'))
        from models.foris import FoRIS
        from ics.native_basis import reuse_native_basis
        from ics.m4_crf_cached import enable_on_crf
        from ics.foris import observe, run_foris
        from ics.official_data import load_inputs
        from ics.methods import mean_control
        from utils.refinement import crf_refine
        self.adapter = SharedReplay()
        with reuse_native_basis(FoRIS, self.lab.BASIS):
            self.host = FoRIS(encoder=self.adapter, image_size=1024, svd_components=500, tau=.6,
                              mask_refiner='crf', resize_to_orig_size=False, device='cpu').eval().requires_grad_(False)
        enable_on_crf(self.host._crf, prepared)
        assert self.host._crf_band_px == 10 and self.host._crf_p_core == .95
        self.crf_refine = crf_refine
        from s2_insid3.models.insid3 import INSID3
        from s2_insid3.utils.refinement import upsample_mask as insid3_upsample
        with reuse_native_basis(INSID3, self.lab.BASIS):
            self.insid3 = INSID3(encoder=self.adapter, image_size=1024, svd_components=500, tau=.6,
                                 merge_threshold=.2, mask_refiner='crf', crf_size=640,
                                 resize_to_orig_size=False, device='cpu').eval().requires_grad_(False)
        enable_on_crf(self.insid3._crf, prepared)
        original_finalize = self.insid3._finalize_mask
        self.insid3_capture = {}
        def capture_finalize(mask, target):
            self.insid3_capture['mask64'] = mask.detach().cpu().clone()
            self.insid3_capture['I'] = insid3_upsample(mask, 1024, 1024).detach().cpu()
            self.insid3_capture['core_seconds'] = time.perf_counter() - self.insid3_started
            tick = time.perf_counter()
            result = original_finalize(mask, target)
            self.insid3_capture['finalize_seconds'] = time.perf_counter() - tick
            return result
        self.insid3._finalize_mask = capture_finalize
        self.run_foris, self.load_inputs, self.mean = run_foris, load_inputs, mean_control
        self.observe_foris = observe
        self.un = importlib.import_module('un_pure')
        self.boundary = importlib.import_module('boundary_pure')
        self.rules = [BooleanRule(expression) for expression in expressions]
        self.linear_rules = [LinearRule(specification) for specification in linear_specifications]
        self.mfu_rule = LinearRule(MFU_RULE)
        self.basis = self.lab.basis()
        self.setup_seconds = time.perf_counter() - started

    def clear_image_crf(self):
        for host in (self.host, self.insid3):
            bilateral = host._crf.bilateral_filter
            bilateral._prepared_geometry = None
            bilateral._prepared_key = None
            bilateral._prepared_colours = None

    def run(self, binding, *, all_sources=True, providers=None, archive_paths=(), provided_raw=None, candidate=None, boundary='none'):
        """Fresh input execution. This API does not accept a previous score/mask."""
        assert candidate in (None, 'MFU', 'gated_MFU')
        assert boundary in BOUNDARY_MODES and (candidate is not None or boundary == 'none')
        if candidate is not None:
            assert providers is None, 'Candidate determines its source dependencies'
        active = ({'M', 'U'} if candidate is not None else
                  set(providers if providers is not None else (SOURCE_NAMES if all_sources else ('B',))))
        assert active <= set(SOURCE_NAMES)
        self.clear_image_crf()
        started, cost = time.perf_counter(), {}
        with inference_files(binding, archive_paths) as opened:
            tick = time.perf_counter()
            self.adapter.bind(binding, provided_raw)
            cost['shared_raw_O24_load_verify'] = time.perf_counter() - tick
            tick = time.perf_counter()
            reference, mask, query = self.load_inputs(binding['row'], ASSETS)
            cost['shared_RGB_reference_mask_load_verify'] = time.perf_counter() - tick
            tick = time.perf_counter()
            masks, fields, frames, detail = {}, {}, {}, {}
            coverage, rho, prepared = None, None, False
            if candidate is not None:
                self.host.set_reference(reference, mask)
                self.host.set_target(query)
                transformed, target = self.host._ref_masks[0].clone(), self.host._tgt_image.clone()
                coverage = self.lab.coverage64(transformed).flatten()
                rho = float(1 - (coverage * (coverage >= .9)).sum() / coverage.sum().clamp_min(1e-9))
                gate = candidate == 'gated_MFU' and rho > .9
                if gate:
                    active = {'T'}
                detail.update(candidate=candidate, candidate_boundary=boundary,
                              reference_gate_rho=rho, reference_gate_true=gate,
                              boundary_skipped_by_reference_gate=gate and boundary != 'none')
                prepared = True
                cost['shared_preprocess_and_reference_gate'] = time.perf_counter() - tick
                tick = time.perf_counter()
            if active & {'B', 'M'}:
                self.host.mask_refiner = 'crf' if 'B' in active else 'bilinear'
                if prepared:
                    try:
                        with self.observe_foris(self.host) as got:
                            final = self.host.segment().reshape(target.shape[-2:]).bool().clone()
                    finally:
                        self.host._ref_images = self.host._ref_masks = self.host._tgt_image = self.host._orig_tgt_size = None
                else:
                    final, got, transformed, target = self.run_foris(self.host, reference, mask, query)
                fields['B_score'] = got['score'].numpy().copy()
                if 'B' in active:
                    masks['B'] = final.numpy().copy()
                    frames['B'] = masks['B'].astype(np.float32)
                cost['B_complete_FoRIS_preprocess_and_CRF' if 'B' in active else 'FoRIS_semantic_core_without_CRF'] = time.perf_counter() - tick
            else:
                if not prepared:
                    self.host.set_reference(reference, mask)
                    self.host.set_target(query)
                    transformed, target = self.host._ref_masks[0].clone(), self.host._tgt_image.clone()
                self.adapter.model_inputs = torch.cat((self.host._ref_images, target[None]), dim=0)
                self.host._ref_images = self.host._ref_masks = self.host._tgt_image = self.host._orig_tgt_size = None
                cost['shared_preprocessing_without_FoRIS'] = time.perf_counter() - tick
            if active & set(CORE_SOURCES[1:]):
                tick = time.perf_counter()
                r_unit = self.lab.unit(self.adapter.raw['reference'])
                q_unit = self.lab.unit(self.adapter.raw['query'])
                r = self.lab.unit(r_unit - (r_unit @ self.basis) @ self.basis.T)
                q = self.lab.unit(q_unit - (q_unit @ self.basis) @ self.basis.T)
                if coverage is None:
                    coverage = self.lab.coverage64(transformed).flatten()
                    rho = float(1 - (coverage * (coverage >= .9)).sum() / coverage.sum().clamp_min(1e-9))
                mean_debias = None
                if 'M' in active:
                    native_fg = transformed[::16, ::16].flatten()
                    mean_debias = True
                    if native_fg.any():
                        target_mean = q_unit.mean(0)
                        mean_debias = bool(torch.dot(self.lab.unit(r_unit[native_fg].mean(0)), target_mean) / (target_mean.norm() + 1e-6) < .8)
                cost['shared_M_U_T_unit_projection_coverage'] = time.perf_counter() - tick
                if 'M' in active:
                    tick = time.perf_counter()
                    mean_q, mean_r = (q, r) if mean_debias else (q_unit, r_unit)
                    mean_field, mean_info = self.mean.predict(mean_q.half(), mean_r.half(), coverage.numpy().reshape(64, 64), fields['B_score'])
                    frames['M'] = self.lab.up(torch.from_numpy(np.ascontiguousarray(mean_field))).numpy()
                    masks['M'] = frames['M'] > .5
                    fields['M'] = mean_field
                    cost['M_original_FP16_mean_control_and_readout'] = time.perf_counter() - tick
                if active & {'U', 'T'}:
                    tick = time.perf_counter()
                    mu = .5 * (r.mean(0) + q.mean(0))
                    sim = self.lab.unit(q - mu) @ self.lab.unit(r - mu).T
                    maximum = sim.max()
                    kernel = torch.exp((sim - maximum) / .05)
                    df, db = kernel @ coverage, kernel @ (1 - coverage)
                    evidence = df + db
                    c = df / evidence.clamp_min(1e-30)
                    if candidate is None or 'T' in active or rho > .9:
                        frames['T'] = self.lab.minmax(self.lab.up(c.reshape(64, 64))).numpy()
                        masks['T'] = frames['T'] > .5
                        fields['T'] = c.numpy().reshape(64, 64)
                    cost['shared_T_U_kernel_c_and_T_readout'] = time.perf_counter() - tick
                    if 'U' not in active:
                        pass
                    elif rho > .9:
                        masks['U'], fields['U'] = masks['T'].copy(), fields['T'].copy()
                        frames['U'] = frames['T'].copy()
                        cost.update(U_modulation=0., U_independent_VP=0., U_graph_and_readout=0.)
                        detail['U_raw_T_fallback'] = True
                    else:
                        tick = time.perf_counter()
                        qunit = self.lab.unit(q)
                        centroids = self.lab.kmeans(qunit, 64, iters=6)
                        labels = (qunit @ centroids.T).argmax(1)
                        onehot = F.one_hot(labels, 64).float()
                        region = lambda value: onehot @ ((onehot.T @ value) / onehot.sum(0).clamp_min(1))
                        sim_max = sim.max(1).values
                        ind0, indn = region(c), region(c * self.lab.minmax(sim_max))
                        reverse = torch.softmax(sim / .05, dim=0)
                        back_f = reverse @ coverage / coverage.sum().clamp_min(1e-6)
                        c_mod, modulation = self.un.modulate(c, labels, ind0, indn,
                                                             sim_max.numpy().astype(np.float16), back_f.numpy().astype(np.float16))
                        cost['U_modulation'] = time.perf_counter() - tick
                        tick = time.perf_counter()
                        ref_map = r.T.reshape(1, 1, 1024, 64, 64).contiguous()
                        query_map = q.T.reshape(1, 1024, 64, 64).contiguous()
                        reference_mask = transformed[None, None].float()
                        prototype = self.lab.unit((r * coverage[:, None]).sum(0))
                        self.host._tgt_image = target
                        try:
                            candidates, vote = self.host._locate_candidates(ref_map, query_map, reference_mask, prototype, 1, 64, 64)
                            dense = torch.zeros((64, 64), dtype=torch.bool)
                            dense[::2, ::2] = True
                            prior = self.host._build_seed_cluster_prior(ref_map, query_map, reference_mask, candidates | dense, 1, 64, 64)
                        finally:
                            self.host._tgt_image = None
                        primary = c_mod.numpy().reshape(64, 64) + (.20 * (vote.numpy() - .5) + .25 * (prior.numpy() - .5))
                        cost['U_independent_VP'] = time.perf_counter() - tick
                        tick = time.perf_counter()
                        log_e = (evidence.clamp_min(1e-30).log() + maximum / .05).numpy().astype(np.float16).astype(np.float64)
                        explanation = expit(log_e - np.median(log_e)).astype(np.float32)
                        similarity = qunit @ qunit.T - torch.eye(4096) * 1e9
                        neighbors = similarity.topk(20, dim=1).indices.numpy().astype(np.uint16)
                        un_field, solver = self.un.solve(primary, explanation, self.un.graph(q, neighbors))
                        frames['U'] = self.lab.minmax(self.lab.up(torch.from_numpy(un_field))).numpy()
                        masks['U'], fields['U'] = frames['U'] > .5, un_field
                        cost['U_graph_and_readout'] = time.perf_counter() - tick
                        detail.update(U_raw_T_fallback=False, modulation=modulation, solver=solver)
                detail.update(reference_rho=rho, M_applied_debias=mean_debias)
                if 'M' in active:
                    detail['M_solver'] = mean_info
            if active & {'I', 'J'}:
                self.insid3_capture.clear()
                self.insid3.mask_refiner = 'crf' if 'J' in active else 'bilinear'
                shared_inputs = self.adapter.model_inputs
                assert shared_inputs.shape == (2, 3, 1024, 1024)
                self.insid3_started = time.perf_counter()
                self.insid3.set_reference(shared_inputs[0], transformed)
                self.insid3.set_target(shared_inputs[1])
                official = self.insid3.segment().cpu()
                masks['I'] = self.insid3_capture['I'].numpy().copy()
                frames['I'] = masks['I'].astype(np.float32)
                if 'J' in active:
                    masks['J'] = official.numpy().copy()
                    frames['J'] = masks['J'].astype(np.float32)
                fields['I_mask64'] = self.insid3_capture['mask64'].numpy()
                cost['I_J_shared_core_and_bilinear'] = self.insid3_capture['core_seconds']
                cost['J_official_CRF640_finalize' if 'J' in active else 'I_official_finalize'] = self.insid3_capture['finalize_seconds']
            tick = time.perf_counter()
            if candidate is None:
                combined = {f'rule{i:03d}': rule(masks) for i, rule in enumerate(self.rules) if rule.symbols <= masks.keys()}
                cost['Boolean_rules'] = time.perf_counter() - tick
                tick = time.perf_counter()
                for rule in self.linear_rules:
                    if rule.symbols <= frames.keys():
                        combined[rule.name] = rule(frames)
                cost['FP64_linear_rules'] = time.perf_counter() - tick
            else:
                combined = {candidate: masks['T'].copy() if detail['reference_gate_true'] else self.mfu_rule(frames)}
                cost['candidate_readout'] = time.perf_counter() - tick
                detail['candidate_crf_calls'] = 0
                if boundary != 'none' and not detail['reference_gate_true']:
                    initial = combined[candidate]
                    tick = time.perf_counter()
                    refined = self.crf_refine(self.host._crf, self.host._crf_band_px, self.host._crf_p_core,
                                              target[None], torch.from_numpy(initial)).numpy()
                    cost['candidate_single_official_CRF'] = time.perf_counter() - tick
                    detail['candidate_crf_calls'] = 1
                    tick = time.perf_counter()
                    if boundary == 'full_crf':
                        combined[candidate] = refined
                    elif boundary == 'add_only':
                        combined[candidate] = initial | refined
                    elif boundary == 'delete_only':
                        combined[candidate] = initial & refined
                    else:
                        probability = (frames['M'].astype(np.float64) + frames['U'].astype(np.float64)) / 2
                        assert np.array_equal(initial, probability > .5)
                        combined[candidate], info = self.boundary.area_preserving_reorder(initial, refined, probability)
                        detail['area16'] = info
                    cost['candidate_boundary_readout'] = time.perf_counter() - tick
        elapsed = time.perf_counter() - started
        detail.update(complete_cached_seconds=elapsed, stage_seconds=cost,
                      disjoint_stage_seconds_sum=sum(cost.values()), overhead_seconds=elapsed - sum(cost.values()),
                      raw_unique_payloads=len(self.adapter.unique_payloads), raw_payload_bytes=self.adapter.logical_bytes,
                      provided_in_memory_O24_inputs=self.adapter.provided_inputs,
                      raw_logical_model_requests=self.adapter.logical_requests, actual_input_files=sorted(set(opened)),
                      requested_sources=sorted(active), produced_sources=sorted(masks),
                      new_encoder_calls=0, query_GT_reads=0, fits=0, archived_score_mask_reads_in_inference=0)
        self.adapter.clear()
        return masks, combined, fields, detail

    def predict(self, binding, *, candidate='MFU', boundary='none', provided_raw=None):
        """Return the requested final 1024 mask and actual dependency/timing audit."""
        _, combined, _, detail = self.run(binding, candidate=candidate, boundary=boundary, provided_raw=provided_raw)
        return combined[candidate], detail


def archive_parity(runtime, case, masks, combined, fields=None):
    old = {}
    for symbol, spec in case['archives'].items():
        if symbol not in masks:
            continue
        assert sha(spec['path']) == spec['sha256']
        with np.load(spec['path'], allow_pickle=False) as saved:
            old[symbol] = runtime.pc.unpack(saved[spec['key']])
    xor = {symbol: int(np.count_nonzero(masks[symbol] != old[symbol])) for symbol in old}
    combo_xor = ({f'rule{i:03d}': int(np.count_nonzero(combined[f'rule{i:03d}'] != rule(old)))
                  for i, rule in enumerate(runtime.rules) if f'rule{i:03d}' in combined} if combined else {})
    linear_xor = {}
    for name, spec in case.get('linear_archives', {}).items():
        if name in combined:
            assert sha(spec['path']) == spec['sha256']
            with np.load(spec['path'], allow_pickle=False) as saved:
                expected = runtime.pc.unpack(saved[spec['key']])
            linear_xor[name] = int(np.count_nonzero(combined[name] != expected))
    differences = {}
    if fields is not None and case.get('continuous_sources'):
        source = case['continuous_sources']
        for symbol, path, field_key in (('M', source['M']['path'], source['M']['key']),
                                        ('U', source['R1_field_path'], 'VP_UN'),
                                        ('T', source['P0_field_path'], 'c32')):
            if symbol in fields:
                with np.load(path, allow_pickle=False) as saved:
                    expected = saved[field_key].astype(np.float32).reshape(64, 64)
                delta = np.abs(np.asarray(fields[symbol]).reshape(64, 64) - expected)
                differences[symbol] = {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean())}
    return {'source_mask_xor': xor, 'composed_archive_mask_xor': combo_xor,
            'linear_archive_mask_xor': linear_xor, 'continuous_field_difference': differences}


def parity():
    config = check()
    runtime = Runtime(config['expressions'], config['linear_rules'])
    root = OUT / 'parity'
    for name in ('predictions', 'fields', 'records'):
        (root / name).mkdir(parents=True, exist_ok=True)
    records = []
    for i, case in enumerate(read(OUT / 'manifest.json'), 1):
        binding, name = case['binding'], case_key(case['binding'])
        masks, combined, fields, detail = runtime.run(binding, archive_paths=[s['path'] for s in case['archives'].values()])
        comparison = archive_parity(runtime, case, masks, combined, fields)
        prediction, field = root / 'predictions' / (name + '.npz'), root / 'fields' / (name + '.npz')
        assert not prediction.exists() and not field.exists(), 'Preserve existing parity output'
        np.savez_compressed(prediction, **{k: runtime.pc.pack(v) for k, v in (masks | combined).items()})
        np.savez_compressed(field, **fields)
        record = {'index': name, 'dataset': binding['dataset'], 'prediction_path': str(prediction),
                  'prediction_sha256': sha(prediction), 'field_path': str(field), 'field_sha256': sha(field),
                  **comparison, **detail}
        write(root / 'records' / (name + '.json'), record)
        records.append(record)
        print({'phase': 'parity', 'completed': i, 'n': 10, 'dataset': binding['dataset'], **comparison}, flush=True)
    report = {'state': 'TEN_DOMAIN_SIX_SOURCE_PARITY_COMPLETE', 'n': 10, 'records': records,
              'config_sha256': sha(OUT / 'config.json'), 'source_xor_sum': {s: sum(r['source_mask_xor'][s] for r in records) for s in SOURCE_NAMES},
              'combination_xor_sum': {k: sum(r['composed_archive_mask_xor'][k] for r in records) for k in records[0]['composed_archive_mask_xor']},
              'linear_xor_sum': {k: sum(r['linear_archive_mask_xor'][k] for r in records) for k in records[0]['linear_archive_mask_xor']},
              'archive_replacement_used': False, 'new_encoder_calls': 0, 'query_GT_reads': 0, 'fits': 0}
    write(root / 'report.json', report)
    print({k: v for k, v in report.items() if k != 'records'}, flush=True)


def statistics(values):
    a = np.asarray(values, np.float64)
    return {'mean': float(a.mean()), 'median': float(np.median(a)), 'p95': float(np.quantile(a, .95)),
            'min': float(a.min()), 'max': float(a.max())}


def benchmark():
    config = check()
    parity_report = read(OUT / 'parity/report.json')
    assert parity_report['n'] == 10
    runtime = Runtime(config['expressions'], config['linear_rules'])
    cases = read(OUT / 'manifest.json')
    tick = time.perf_counter()
    runtime.run(cases[0]['binding'])
    warmup_seconds = time.perf_counter() - tick
    records = []
    modes = ('B_only', 'four_sources', 'six_sources', 'soft_MUT_core')
    providers = {'B_only': ('B',), 'four_sources': CORE_SOURCES, 'six_sources': SOURCE_NAMES, 'soft_MUT_core': ('M', 'U', 'T')}
    for i, case in enumerate(cases):
        shift = i % len(modes)
        order = modes[shift:] + modes[:shift]
        result = {'index': case_key(case['binding']), 'dataset': case['binding']['dataset'], 'order': order}
        for mode in order:
            masks, combined, fields, detail = runtime.run(case['binding'], providers=providers[mode],
                                                         archive_paths=[s['path'] for s in case['archives'].values()])
            result[mode] = detail | archive_parity(runtime, case, masks, combined, fields)
        result['total_cached_ratio'] = {mode: result[mode]['complete_cached_seconds'] / result['B_only']['complete_cached_seconds']
                                        for mode in modes[1:]}
        result['incremental_seconds'] = {mode: result[mode]['complete_cached_seconds'] - result['B_only']['complete_cached_seconds']
                                        for mode in modes[1:]}
        records.append(result)
        print({'phase': 'benchmark', 'completed': i + 1, 'n': 10, 'dataset': result['dataset'],
               'B_seconds': result['B_only']['complete_cached_seconds'],
               'four_source_seconds': result['four_sources']['complete_cached_seconds'],
               'six_source_seconds': result['six_sources']['complete_cached_seconds'],
               'soft_MUT_core_seconds': result['soft_MUT_core']['complete_cached_seconds'],
               'ratios': result['total_cached_ratio']}, flush=True)
    summary = {mode: statistics([r[mode]['complete_cached_seconds'] for r in records]) for mode in modes}
    paired = {mode: {'incremental_seconds': statistics([r['incremental_seconds'][mode] for r in records]),
                      'total_cached_ratio': statistics([r['total_cached_ratio'][mode] for r in records])} for mode in modes[1:]}
    stage_names = sorted({name for r in records for mode in modes for name in r[mode]['stage_seconds']})
    stage_summary = {mode: {name: statistics([r[mode]['stage_seconds'].get(name, 0.) for r in records]) for name in stage_names}
                     for mode in modes}
    report = {'state': 'SHARED_RUNTIME_PAIRED_COST_COMPLETE', 'n': 10, 'expressions': config['expressions'],
              'records': records, 'summary': summary, 'stage_summary': stage_summary, 'paired_cost': paired,
              'ratio_of_mean_cached_times': {mode: summary[mode]['mean'] / summary['B_only']['mean'] for mode in modes[1:]},
              'CPU_threads': 2, 'workers': 1, 'device': 'cpu', 'setup_seconds': runtime.setup_seconds,
              'warmup_seconds': warmup_seconds, 'new_encoder_calls': 0, 'query_GT_reads': 0, 'fits': 0,
              'archived_score_mask_reads_in_inference': 0, 'archive_checks_outside_timer': True,
              'config_sha256': sha(OUT / 'config.json'), 'parity_report_sha256': sha(OUT / 'parity/report.json'),
              'limits': 'Ten fixed single trials per pipeline are descriptive cached-O24 latency, not an RGB encoder timing or latency SLA.'}
    write(OUT / 'benchmark.json', report)
    lines = ['# Shared four/six-source runtime', '',
             'B=complete FoRIS+CRF, M=original MEAN, U=original R1 VP_UN(.5), T=soft-coverage c, I/J=one official INSID3 core with bilinear/CRF640 endpoints. No fitting or archived score/mask input during inference.',
             '', '|Pipeline|Mean s|Median s|P95 s|Max s|', '|---|---:|---:|---:|---:|']
    for mode in modes:
        s = summary[mode]
        lines.append(f"|{mode}|{s['mean']:.6f}|{s['median']:.6f}|{s['p95']:.6f}|{s['max']:.6f}|")
    lines += ['', 'Expressions: ' + ', '.join('`' + x + '`' for x in config['expressions']),
              'Ratios of mean complete cached times: ' + json.dumps(report['ratio_of_mean_cached_times']) + '.',
              'Input O24 verification/I/O and RGB/reference-mask loading are included. One-time model/basis setup and output serialization are excluded for both pipelines.',
              'Raw O24 is loaded once per input. M/U/T share normalization and projection; U/T share c; M uses the newly computed complete FoRIS score. M FP16 and U FP32 graphs remain distinct.',
              '', 'Parity XOR totals: ' + json.dumps(parity_report['source_xor_sum']) + '; composed: ' + json.dumps(parity_report['combination_xor_sum']),
              'Continuous-mixture XOR totals: ' + json.dumps(parity_report['linear_xor_sum']) + '. Soft-MUT runs no B or J CRF.',
              'Computed outputs were preserved; no archived outputs were substituted. Archives are opened only after the inference timer ends.',
              report['limits']]
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print({'state': report['state'], 'summary': summary, 'ratio_of_means': report['ratio_of_mean_cached_times']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'parity', 'benchmark', 'run', 'infer'))
    parser.add_argument('--rule', action='append', default=None, help='Repeat for any number of Boolean expressions')
    parser.add_argument('--linear-rule', action='append', default=None, help='JSON: name, weights mapping, threshold; repeat as needed')
    parser.add_argument('--candidate', choices=('MFU', 'gated_MFU'), default='MFU')
    parser.add_argument('--boundary', choices=BOUNDARY_MODES, default='none')
    parser.add_argument('--binding', help='JSON containing one existing RGB/reference-mask/raw-O24 binding')
    parser.add_argument('--output', help='Output NPZ path, outside this checkout')
    arguments = parser.parse_args()
    linear_specifications = [json.loads(value) for value in arguments.linear_rule] if arguments.linear_rule else DEFAULT_LINEAR_RULES
    if arguments.mode == 'prepare':
        prepare(arguments.rule or DEFAULT_RULES, linear_specifications)
    elif arguments.mode == 'parity':
        parity()
    elif arguments.mode == 'benchmark':
        benchmark()
    elif arguments.mode == 'infer':
        assert arguments.binding and arguments.output, '--binding and --output are required'
        config = check()
        output = Path(arguments.output).resolve()
        assert not output.is_relative_to(REPO) and not output.exists()
        binding = read(arguments.binding)
        binding = binding.get('binding', binding)
        runtime = Runtime(config['expressions'], config['linear_rules'])
        prediction, detail = runtime.predict(binding, candidate=arguments.candidate, boundary=arguments.boundary)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, prediction=runtime.pc.pack(prediction))
        write(output.with_suffix('.json'), detail | {'prediction_path': str(output), 'prediction_sha256': sha(output)})
        print({'candidate': arguments.candidate, 'boundary': arguments.boundary, 'output': str(output),
               'seconds': detail['complete_cached_seconds']}, flush=True)
    else:
        prepare(arguments.rule or DEFAULT_RULES, linear_specifications)
        frozen = OUT / 'frozen/x1012s2_runtime.py'
        for mode in ('parity', 'benchmark'):
            subprocess.run([sys.executable, '-B', str(frozen), mode], check=True)
