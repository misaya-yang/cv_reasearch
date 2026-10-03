#!/usr/bin/env python3
"""Standard-library-only audit of existing FROST source and native interfaces.

Reads source; never imports torch/timm/torchvision/einops, executes source, loads
weights/images, invokes CUDA/network, or changes official code. Output /tmp only.
Presence/AST acceptance is NOT installed runtime, numerical parity or task gain.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualified(node):
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute): return qualified(node.value) + '.' + node.attr
    return ast.unparse(node)


def functions(tree):
    rows = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            rows[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    rows[node.name + '.' + method.name] = method
    return rows


def signature(node):
    arguments = node.args
    return dict(parameters=[argument.arg for argument in arguments.posonlyargs + arguments.args],
        defaults=[ast.unparse(value) for value in arguments.defaults],
        keyword_only=[argument.arg for argument in arguments.kwonlyargs],
        decorators=[ast.unparse(value) for value in node.decorator_list], line=node.lineno)


def calls(node, name):
    return [call for call in ast.walk(node) if isinstance(call, ast.Call) and qualified(call.func) == name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', default='/tmp/demo9_frost_source')
    parser.add_argument('--native-foris', default='/tmp/demo9_native_foris.py')
    parser.add_argument('--native-wrapper', default=str(Path(__file__).resolve().parents[2] / 'demo4_incontext_seg/icx/common.py'))
    parser.add_argument('--out', default='/tmp/demo9_frost_native_contract_cpu.json')
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if not output.is_relative_to(Path('/tmp').resolve()):
        parser.error('--out must be under /tmp')
    root = Path(args.source_root)
    files = [root/'frost'/name for name in ('model.py', 'density.py', 'data.py', 'encoder.py')]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        output.write_text(json.dumps(dict(state='SOURCE_DEPENDENCY_MISSING', missing=missing), indent=2) + '\n')
        raise SystemExit(2)
    trees = {path.name: ast.parse(path.read_text(), filename=str(path)) for path in files}
    methods = {name: functions(tree) for name, tree in trees.items()}
    model, density, encoder = methods['model.py'], methods['density.py'], methods['encoder.py']
    constants = {}
    for node in trees['model.py'].body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name): constants[target.id] = node.value.value
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
        if not condition: raise AssertionError(name)
    check('constructor_explicit_encoder_and_raw_encoder',
          signature(model['FROST.__init__'])['parameters'] ==
          ['self', 'encoder', 'raw_encoder', 'image_size', 'device', 'resize_to_orig_size'])
    check('five_dimensional_callable_not_forward',
          'DINOv3FeatureExtractor.__call__' in encoder and 'DINOv3FeatureExtractor.forward' not in encoder)
    check('extractor_no_grad', 'torch.no_grad()' in signature(encoder['DINOv3FeatureExtractor.__call__'])['decorators'])
    check('feature_extractor_parameters_dtype_cast',
          any(ast.unparse(call) == 'x.to(enc_dtype)' for call in calls(encoder['DINOv3FeatureExtractor.__call__'], 'x.to')))
    check('last_layer_patch_maps',
          any({keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords} ==
              {'n': '1', 'reshape': 'True'} for call in calls(encoder['DINOv3FeatureExtractor.__call__'], 'self.encoder.get_intermediate_layers')))
    check('extractor_returns_float32', bool(calls(encoder['DINOv3FeatureExtractor.__call__'],
          "self.encoder.get_intermediate_layers(x, n=1, reshape=True)[0].float")))
    check('basis_raw_encoder_last_layer', bool(calls(model['FROST._build_positional_basis'],
          'enc.to(device).get_intermediate_layers')))
    check('basis_source_black_not_random', bool(calls(model['FROST._build_positional_basis'], 'torch.zeros')) and
          not calls(model['FROST._build_positional_basis'], 'torch.randn'))
    check('basis_rank250_shrink95', constants['_SVD_COMPONENTS'] == 250 and constants['_SHRINK_LAMBDA'] == .95)
    check('basis_not_compressed_coordinates', 'basis @ basis.T' in ast.unparse(model['FROST._debias_features']))
    predict = model['FROST.predict_mask']
    check('support_flip_before_triplet_concatenation',
          ast.unparse(predict).index('self._flip_augment') < ast.unparse(predict).index('torch.cat([ref_images, tgt_image]'))
    check('apd_always_called_without_native_choice', bool(calls(predict, 'self._debias_features')) and
          not calls(predict, 'self._should_apply_positional_debias'))
    check('complete_density_and_continuous_finalization', bool(calls(predict, 'density_ratio_predict')) and
          '_post_ell' in ast.unparse(predict) and bool(calls(predict, 'self._finalize_mask')))
    check('zero_posterior_boundary_and_ten_smoothing_steps', constants['_THRESHOLD'] == 0. and
          constants['_BILATERAL_N_ITERS'] == 10 and constants['_BILATERAL_ALPHA'] == .70)
    check('density_class_count_normalization',
          bool(calls(density['kde_log_posterior'], 'math.log')) and
          'log_p_fg - log_p_bg' in ast.unparse(density['kde_log_posterior']))
    check('no_autocast_checkpoint_or_automatic_dtype_conversion',
          not any(isinstance(node, ast.Call) and any(part in qualified(node.func)
              for part in ('autocast', 'checkpoint', '.half', '.bfloat16'))
              for tree in trees.values() for node in ast.walk(tree)))
    check('only_explicit_factory_reads_hub_weights', bool(calls(encoder['build_encoder'], 'torch.hub.load')) and
          not any(isinstance(node, ast.Call) and qualified(node.func) in
              ('torch.load', 'torch.hub.load', 'load_file', 'load_state_dict')
              for node in ast.walk(trees['model.py'])))
    relatives, externals = {}, set()
    for name, tree in trees.items():
        relative = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    path = root/'frost'/((node.module or '').replace('.', '/') + '.py')
                    relative.append(dict(module=node.module, path=str(path), present=path.is_file()))
                elif node.module and node.module != '__future__': externals.add(node.module.split('.')[0])
            elif isinstance(node, ast.Import):
                externals.update(alias.name.split('.')[0] for alias in node.names)
        relatives[name] = relative
    check('all_local_relative_source_dependencies_present', all(row['present'] for rows in relatives.values() for row in rows))
    packages = {}
    for module in sorted(externals | {'timm', 'safetensors'}):
        distribution = {'PIL': 'Pillow'}.get(module, module)
        try: version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError: version = None
        packages[module] = dict(spec_present=importlib.util.find_spec(module) is not None,
                                distribution_version=version, imported=module in sys.modules)
    native = Path(args.native_foris)
    wrapper = Path(args.native_wrapper)
    native_signatures = {}
    if native.is_file():
        nf = functions(ast.parse(native.read_text()))
        native_signatures = {name: signature(nf[name]) for name in (
            'FoRIS._extract_features', 'FoRIS._build_positional_basis',
            'FoRIS._part1_positional_debias', 'FoRIS._binarize_response')}
    check('no_third_party_runtime_imported', all(name not in sys.modules for name in ('torch', 'torchvision', 'timm', 'einops', 'PIL', 'numpy')))
    result = dict(state='STDLIB_FROST_SOURCE_CONTRACT_PASSED', checks=checks,
        checks_count=len(checks), source_root=str(root), source_sha256={str(path):digest(path) for path in files},
        signatures={name:{key:signature(value) for key,value in rows.items()} for name,rows in methods.items()},
        constants=constants, local_relative_dependencies=relatives, package_presence_only=packages,
        missing_package_init=not (root/'frost/__init__.py').is_file(),
        missing_factory='build_frost/__init__.py not present; direct namespace frost.model.FROST uses present local files',
        native_source_sha256={str(path):digest(path) for path in (native,wrapper) if path.is_file()},
        native_signatures=native_signatures,
        cache_contract=dict(full_FROST='one flattened ordered [S, flip(S), Q] batch3 for one-shot',
            native_FoRIS='ordered [S,Q] batch2; query-dependent optional rank500 APD',
            exact_reuse='Only identical full raw encoder input/batch/order/dtype/autocast/backend/weights and descriptor output context',
            not_exact='Cannot compose batch3 from batch2 plus B1 flip without whole-feature equality proof'),
        scope='Source/signature/AST/dependency-presence check only; no imports, CUDA, weights, data or numerical/runtime claim')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(state=result['state'],checks=result['checks_count'],out=str(output),
                         third_party_imported=False, missing_package_init=result['missing_package_init'])))


if __name__ == '__main__': main()
