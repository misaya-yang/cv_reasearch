#!/usr/bin/env python3
"""One finite RICE covariance experiment, full FoRIS plus matched controls.

Preparation and self checks are CPU-only. The GPU stage streams existing RGB,
never writes features, and freezes every prediction before opening query GT.
An inserted arm changes only Part2 score/sf/sbn; native geometry, clustering,
semantic consolidation and one native CRF call remain in the full pipeline.
"""
import argparse
import base64
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import MethodType
import zlib

HERE = Path(__file__).resolve().parent
ARMS = ('foris_crf', 'identity_density', 'plain_discriminant', 'fg_aug_fisher',
        'pooled_fisher', 'paired_only', 'paired_shuffle', 'foris_identity_density',
        'foris_pooled_fisher', 'foris_paired_only', 'foris_paired_shuffle')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False) + '\n'); temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


@contextmanager
def capture_native(host):
    """Observe original feature/Part2/pre-CRF output; restore owned methods."""
    stored = {}; owned = {}
    names = ('_extract_features', '_part1_positional_debias', '_part2_background_suppression', '_binarize_response')
    for name in names:
        original = getattr(host, name)
        owned[name] = (name in host.__dict__, host.__dict__.get(name))
        def wrapped(this, *args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            if _name in stored:
                raise RuntimeError('One exact native call expected: ' + _name)
            stored[_name] = result
            if _name == '_part1_positional_debias':
                inp = args[0] if args else kwargs['fmaps_norm']
                import torch
                stored['apd_applied'] = not torch.equal(inp, result)
            return result
        setattr(host, name, MethodType(wrapped, host))
    try:
        yield stored
    finally:
        for name, (was_owned, old) in owned.items():
            if was_owned: setattr(host, name, old)
            else: delattr(host, name)


@contextmanager
def frozen_native_replay(host, raw, part1, part2):
    """No encoder replay. Only continuous Part2 evidence is overridden."""
    replacements = {'_extract_features': lambda *a, **k: raw,
                    '_part1_positional_debias': lambda *a, **k: part1,
                    '_part2_background_suppression': lambda *a, **k: part2}
    saved = {name: (name in host.__dict__, host.__dict__.get(name)) for name in replacements}
    for name, callback in replacements.items():
        setattr(host, name, MethodType(lambda this, *a, _callback=callback, **k: _callback(*a, **k), host))
    try:
        yield
    finally:
        for name, (owned, old) in saved.items():
            if owned: setattr(host, name, old)
            else: delattr(host, name)


@contextmanager
def exact_clustering_cache(foris_module):
    """Reuse only bitwise-identical clustering inputs within one episode."""
    import torch
    original = foris_module.agglomerative_clustering
    cache, audit = [], dict(hits=0, misses=0, exact_only=True)
    def cached(x, *args, **kwargs):
        for old, aa, kk, result in cache:
            if args == aa and kwargs == kk and x.shape == old.shape and x.dtype == old.dtype and torch.equal(x, old):
                audit['hits'] += 1
                return result.clone()
        result = original(x, *args, **kwargs)
        cache.append((x.detach().clone(), args, kwargs.copy(), result.detach().clone()))
        audit['misses'] += 1
        return result
    foris_module.agglomerative_clustering = cached
    try: yield audit
    finally: foris_module.agglomerative_clustering = original


def bandwidth(projected_fg, projected_bg):
    """Reference-only median positive nearest same-label squared distance."""
    import torch
    nearest = []
    for x in (projected_fg, projected_bg):
        if len(x) < 2: continue
        distance = (x[:, None] - x[None]).square().sum(-1)
        distance.fill_diagonal_(float('inf'))
        minimum = distance.min(1).values
        nearest.append(minimum[torch.isfinite(minimum) & (minimum > 0)])
    valid = torch.cat(nearest) if nearest else projected_fg.new_empty(0)
    if not len(valid): return None
    answer = float(valid.median())
    return answer if answer > 0 else None


def kernel_logratio(query, foreground, background, directions):
    """Equal-prior Gaussian KDE, no post-projection normalization or host gain."""
    import torch
    q, fg, bg = query.double(), foreground.double(), background.double()
    if directions is not None:
        q, fg, bg = q @ directions, fg @ directions, bg @ directions
    tau = bandwidth(fg, bg)
    if tau is None: return None, dict(state='NO_POSITIVE_REFERENCE_BANDWIDTH')
    results = []
    for block in q.split(256):
        classes = []
        for a in (fg, bg):
            # Explicit norm formula preserves metric length, not cosine geometry.
            distance = (block.square().sum(1)[:, None] + a.square().sum(1)[None] - 2 * block @ a.T).clamp_min(0)
            classes.append((-distance / (2 * tau)).logsumexp(1) - __import__('math').log(len(a)))
        results.append(classes[0] - classes[1])
    score = torch.cat(results)
    if not torch.isfinite(score).all(): raise ArithmeticError('Nonfinite kernel evidence')
    return score, dict(state='KERNEL_EVIDENCE', bandwidth=tau, reference_only=True,
                        covariance_model='equal_prior_isotropic_Gaussian_in_declared_projected_coordinates',
                        posterior_calibration_claim=False, post_projection_normalization=False,
                        foreground_anchors=len(fg), background_anchors=len(bg))


def prepare(args):
    """Validate existing data and source, no weights/models or CUDA loaded."""
    import torch
    from unittest.mock import patch
    import numpy as np
    from PIL import Image
    if not args.prepared_manifest: raise ValueError('--prepared-manifest required')
    manifest = Path(args.prepared_manifest)
    if manifest.exists(): raise ValueError('Fresh manifest required')
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    with patch('torch.cuda.is_available', return_value=False):
        sys.path.insert(0, str(Path(args.prepared_root) / 'scripts')); import _paths
        sys.path.insert(0, _paths.DEMO4)
        from icx.common import coco_episodes
        sys.path.insert(0, str(HERE))
        from global_representation_probe import fixed_selection
        rows, assets, dimensions = [], set(), []
        for fold in range(4):
            episodes, _, base = coco_episodes(fold, 400, shot=1, seed=0)
            for e, (c, query, supports) in fixed_selection(episodes, 10):
                row = dict(e=e, c=c, fold=fold, support=supports[0], query=query)
                for role in ('support', 'query'):
                    image = Path(base) / row[role]
                    annotation = Path(_paths.COCO_ANN) / Path(row[role]).with_suffix('.png')
                    with Image.open(image) as value:
                        shape = value.size; value.verify()
                    with Image.open(annotation) as value:
                        if value.size != shape: raise ValueError('Original image/mask shape mismatch')
                        value.verify()  # Header/decode checks, no GT-quality selection.
                    assets.update((image.resolve(), annotation.resolve()))
                rows.append(row)
        modelroot = Path(args.shared_root) / 'models/dinov3-vitl16-timm'
        if not (modelroot / 'model.safetensors').exists(): raise ValueError('Existing offline DINO asset missing')
        assets.update((modelroot / 'model.safetensors', modelroot / 'config.json'))
        source = Path(args.foris_root) / 'models/foris.py'
        basis = Path(args.projection_basis)
        if not basis.is_file(): raise ValueError('Original frozen native basis required')
        assets.add(basis.resolve())
        code = [Path(__file__), HERE / 'analyze_rice_core.py', HERE / 'prepare_rice_core_queue.py',
                HERE / 'rice_statistics_cpu.py', HERE / 'global_representation_probe.py',
                HERE / 'experiment_resource_guard.py', HERE.parent / 'tics/rice_statistics.py',
                HERE.parent / 'tics/native_assets.py', HERE.parent / 'tics/rice_subspace.py',
                HERE.parent / 'tics/reference_views.py', HERE.parent / 'tics/__init__.py',
                HERE.parent / 'tics/imageset.py', HERE.parent / 'tics/propagate.py', source,
                Path(_paths.DEMO4) / 'icx/common.py', Path(args.prepared_root) / 'scripts/_paths.py']
        code += [p for p in (Path(args.foris_root) / 'utils').glob('*.py') if not p.name.startswith('.')]
        for path in code:
            compile(path.read_text(), str(path), 'exec')
        prepared = dict(schema='rice_core_assets_v1', state='PREPARED_ASSETS', fold_count=4, per_fold=10,
            seed=0, frozen_episodes=rows, arms=list(ARMS), data_root=str(base),
            annotation_root=_paths.COCO_ANN, prepared_root=args.prepared_root,
            foris_root=args.foris_root, source_hashes={str(p.resolve()): digest(p) for p in code},
            assets=[dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(assets)],
            projection_basis=dict(path=str(basis), sha256=digest(basis)),
            native_host_sha256=digest(source), live_data_CPU_checked=True, no_CUDA_or_model=True,
            selection='first protocol draw per class, smallest ten classes per fold; no GT-quality selection',
            fixed_config=dict(views=['original', 'global_RGB_mean_background', 'half_cycle_BG_pixel_permutation'],
                pure_FG=.9, pure_BG=.1, FG_cap=128, BG_cap=256, rank_cap=16,
                ridge='0.1*trace(pooled_augmented_within)/D shared across all geometry arms',
                bandwidth='median_positive_nearest_same_label_squared_distance',
                counterfactual_gamma=0, paired_shuffle_seed=2052,
                query_encoder_calls=1, encoder_precision='actual_unmodified_FoRIS_native',
                APD='original pair choice then same projection for all reference views',
                part2_insert='p-.5,p,1-p; original mu_fg and denoised target preserved',
                final_refiner='one original FoRIS CRF call for each arm; no double refinement'))
        write_json(manifest, prepared)
        return dict(state=prepared['state'], tasks=len(rows), files=len(assets), CUDA_initialized=torch.cuda.is_initialized())


def cpu_check(args):
    import torch
    from unittest.mock import patch
    torch.set_num_threads(1)
    with patch('torch.cuda.is_available', return_value=False):
        stats = module(HERE.parent / 'tics/rice_statistics.py', 'rice_core_cpu_statistics')
        rows = []
        for index in range(10):
            torch.manual_seed(2054 + index)
            x = torch.randn(3, 16, 6, dtype=torch.float64)
            x[1:] = x[0:1] + .1 * x[1:]
            cover = torch.tensor([1.] * 8 + [0.] * 8)
            s = stats.build_reference_statistics(x, cover, (4, 4))
            w, _ = stats.discriminant_projection(s['contrasts'], s['covariances']['paired_only'], s['ridge'])
            a, b = s['foreground_indices'], s['background_indices']
            score, audit = kernel_logratio(x[0], x[0, a], x[0, b], w)
            assert torch.isfinite(score).all() and audit['bandwidth'] > 0
            class Mock:
                def _extract_features(self, value): return value + 1
                def _part1_positional_debias(self, value, *a, **k): return value
                def _part2_background_suppression(self, *a, **k): return ('native',)
                def _binarize_response(self, value, **k): return value > 0
                def predict(self, value):
                    z = self._extract_features(value)
                    z = self._part1_positional_debias(z)
                    p = self._part2_background_suppression()
                    return self._binarize_response(z), p
            host = Mock(); source = host.predict(x)
            before = host.__dict__.copy()
            with capture_native(host) as packet: result = host.predict(x)
            assert torch.equal(result[0], source[0]) and host.__dict__ == before
            with frozen_native_replay(host, packet['_extract_features'], packet['_part1_positional_debias'], ('changed',)):
                replay = host.predict(x)
            assert torch.equal(replay[0], source[0]) and replay[1] == ('changed',) and host.__dict__ == before
            p = score.sigmoid(); assert torch.equal((p > .5), (score > 0))
            rows.append(dict(index=index, kernels_finite=True, native_noop_and_hook_restore=True, insertion_no_encoder=True))
    answer = dict(state='CPU_RICE_RUNTIME_PASSED', cases=rows, real_images=False,
                  pretrained_DINO=False, CUDA_initialized=torch.cuda.is_initialized())
    if args.out:
        path = Path(args.out)
        if path.exists(): raise ValueError('Preserve previous CPU checks')
        write_json(path, answer)
    return answer


def run(args):
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    if not args.allow_gpu or os.environ.get('DEMO9_CUDA_GUARD') != '1':
        raise RuntimeError('CPU-preflighted guarded GPU stage only')
    guard = json.loads(Path(args.resource_guard_state).read_text())
    if guard['state'] != 'GPU_RUNNING': raise RuntimeError('Actual running resource guard required')
    manifest = json.loads(Path(args.prepared_manifest).read_text())
    if manifest.get('state') != 'PREPARED_ASSETS' or manifest.get('arms') != list(ARMS):
        raise ValueError('Exactly frozen prepared RICE contract required')
    for asset in manifest['assets']:
        st = Path(asset['path']).stat()
        if (st.st_size, st.st_mtime_ns) != (asset['size'], asset['mtime_ns']): raise RuntimeError('Prepared data/weight drift')
    for source, expected in manifest['source_hashes'].items():
        if digest(source) != expected: raise RuntimeError('Frozen code drift')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()): raise ValueError('New output required')
    out.mkdir(parents=True, exist_ok=True)
    report = dict(state='RUNNING', records=[], arms=list(ARMS), manifest_sha256=digest(args.prepared_manifest),
                  scope='Four-fold first10/classes development mechanism subset; not independent test or final method score',
                  feature_cache_written=False, query_GT_used_for_prediction=False, config=manifest['fixed_config'])
    started = time.monotonic()
    def save():
        report['elapsed_s'] = time.monotonic() - started
        write_json(out / 'report.json', report)
    save()
    try:
        if not torch.cuda.is_available(): raise RuntimeError('No CUDA, no encoder loaded')
        torch.set_num_threads(4); torch.manual_seed(0)
        torch.cuda.set_per_process_memory_fraction(.3)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        sys.path.insert(0, str(Path(args.prepared_root) / 'scripts')); import _paths
        sys.path.insert(0, args.foris_root)
        import models.foris as foris_module
        from utils.data import build_transform
        sys.path.insert(0, _paths.DEMO4)
        from icx.common import TimmDINOv3
        sys.path.insert(0, str(HERE.parent))
        from tics.native_assets import reuse_native_basis
        from tics.reference_views import support_background_views
        statistics = module(HERE.parent / 'tics/rice_statistics.py', 'rice_live_statistics')
        with torch.inference_mode():
            encoder = TimmDINOv3().cuda().eval().requires_grad_(False)
            with reuse_native_basis(foris_module.FoRIS, args.projection_basis) as receipt:
                host = foris_module.FoRIS(encoder=encoder, image_size=1024, svd_components=500,
                    tau=.6, mask_refiner='crf', resize_to_orig_size=False, device='cuda').eval().requires_grad_(False)
            transform = build_transform(1024); report['basis_receipt'] = receipt
            for row in manifest['frozen_episodes']:
                begin = time.monotonic(); c = row['c']
                image_s = Image.open(Path(manifest['data_root']) / row['support']).convert('RGB')
                image_q = Image.open(Path(manifest['data_root']) / row['query']).convert('RGB')
                s, q = transform(image_s).cuda()[None], transform(image_q).cuda()[None]
                gold = torch.from_numpy((np.asarray(Image.open(Path(manifest['annotation_root']) / Path(row['support']).with_suffix('.png'))) == c + 1).copy()).cuda()
                mask = F.interpolate(gold[None, None].float(), (1024, 1024), mode='nearest')[0].bool()
                with exact_clustering_cache(foris_module) as cache_audit:
                    with capture_native(host) as packet:
                        native = host.predict(s, mask, q[0]).reshape(1024, 1024).bool().clone()
                    raw, part1 = packet['_extract_features'], packet['_part1_positional_debias']
                    with frozen_native_replay(host, raw, part1, packet['_part2_background_suppression']):
                        native_replay = host.predict(s, mask, q[0])
                    if not torch.equal(native, native_replay):
                        raise RuntimeError('Actual unchanged FoRIS replay is not exact')
                    base_support, query = part1[0, 0].flatten(1).T.float(), part1[0, 1].flatten(1).T.float()
                    grid = tuple(part1.shape[-2:])
                    if grid != (64, 64) or base_support.shape[1] != 1024: raise ValueError('Native feature coordinates changed')
                    def encode_view(rgb):
                        # The original view is reused exactly, no additional native S encoding.
                        if torch.equal(rgb, s): return part1[:, 0].clone()
                        value = host._extract_features(rgb[:, None])
                        value = F.normalize(value, p=2, dim=2)
                        if packet['apd_applied']: value = host._debias_features(value)
                        return value[:, 0]
                    bank = support_background_views(s, mask, encode_view)
                    views = torch.stack([a[0].flatten(1).T.float() for a in bank.maps])
                    coverage = F.interpolate(gold[None, None].float(), grid, mode='area')[0, 0].flatten()
                    stats = statistics.build_reference_statistics(views, coverage, grid)
                    predictions = {'foris_crf': native}; arm_info = {}; probabilities = {}
                    if stats['state'].startswith('DEGENERATE'):
                        for name in ARMS[1:]:
                            predictions[name] = native.clone(); arm_info[name] = dict(state=stats['state'], exact_native_fallback=True)
                    else:
                        fg, bg = stats['foreground_indices'], stats['background_indices']
                        for name, covariance_key in [('identity_density', None), ('plain_discriminant', 'identity'),
                            ('fg_aug_fisher', 'fg_aug'), ('pooled_fisher', 'pooled_fisher'),
                            ('paired_only', 'paired_only'), ('paired_shuffle', 'paired_shuffle')]:
                            directions, projection = (None, dict(state='FULL_IDENTITY_GEOMETRY')) if covariance_key is None else statistics.discriminant_projection(stats['contrasts'], stats['covariances'][covariance_key], stats['ridge'])
                            if directions is not None and directions.shape[1] == 0:
                                predictions[name] = native.clone(); arm_info[name] = dict(state='NO_DISCRIMINANT', projection=projection); continue
                            score, kde = kernel_logratio(query, base_support[fg], base_support[bg], directions)
                            if score is None:
                                predictions[name] = native.clone(); arm_info[name] = dict(**kde, projection=projection); continue
                            probability = score.sigmoid().reshape(grid); probabilities[name] = probability
                            hard = F.interpolate(score.reshape(1, 1, *grid), (1024, 1024), mode='bilinear', align_corners=False)[0, 0] > 0
                            predictions[name] = host._finalize_mask(hard, q).reshape(1024, 1024).bool().clone()
                            arm_info[name] = dict(state='COMPLETED', projection=projection, kernel=kde,
                                positive_kernel_tokens=int((score > 0).sum()), sigmoid_saturated_tokens=int(((probability == 0) | (probability == 1)).sum()))
                        old_tuple = packet['_part2_background_suppression']
                        for short in ('identity_density', 'pooled_fisher', 'paired_only', 'paired_shuffle'):
                            name = 'foris_' + short
                            if short not in probabilities:
                                predictions[name] = native.clone(); arm_info[name] = dict(state='CORRESPONDING_CORE_FALLBACK'); continue
                            p = probabilities[short].to(old_tuple[0].dtype)
                            new_part2 = (p - .5, p, 1 - p, old_tuple[3], old_tuple[4])
                            with frozen_native_replay(host, raw, part1, new_part2):
                                predictions[name] = host.predict(s, mask, q[0]).reshape(1024, 1024).bool().clone()
                            arm_info[name] = dict(state='COMPLETED', part2_output='p-.5,p,1-p',
                                rest_of_native_pipeline=True, encoder_replayed=False, native_CRF_calls=1)
                    if set(predictions) != set(ARMS): raise RuntimeError('Missing declared prediction')
                    torch.cuda.synchronize()
                    frozen = {name: pred.cpu().numpy().astype(bool) for name, pred in predictions.items()}
                    inference_s = time.monotonic() - begin
                    # ONLY NOW open query annotation; all candidate outputs are immutable.
                    truth_original = np.asarray(Image.open(Path(manifest['annotation_root']) / Path(row['query']).with_suffix('.png'))) == c + 1
                    truth_tensor = torch.from_numpy(truth_original.copy()).cuda()
                    truth_model = F.interpolate(truth_tensor[None, None].float(), (1024, 1024), mode='nearest')[0, 0].bool().cpu().numpy()
                    original_predictions = {name: (F.interpolate(pred[None, None].float(), truth_original.shape,
                        mode='bilinear', align_corners=False)[0, 0] > .5)
                        for name, pred in predictions.items()}
                    def iu(a, b): return [int((a & b).sum()), int((a | b).sum())]
                    def packed(a):
                        data = zlib.compress(np.packbits(a.flatten()).tobytes(), 6)
                        return dict(shape=list(a.shape), codec='zlib_np_packbits_big', data=base64.b64encode(data).decode())
                    baseline = frozen['foris_crf']
                    ledger = {name: dict(recovered_fn=int((~baseline & a & truth_model).sum()),
                        lost_tp=int((baseline & ~a & truth_model).sum()), added_fp=int((~baseline & a & ~truth_model).sum()),
                        removed_fp=int((baseline & ~a & ~truth_model).sum())) for name, a in frozen.items()}
                    item = dict(**row, iu={name: iu(a, truth_model) for name, a in frozen.items()},
                        original_iu={name: iu(a.cpu().numpy(), truth_original) for name, a in original_predictions.items()},
                        ledger=ledger, methods=arm_info, stats=stats['audit'], statistics_state=stats['state'],
                        native_APD=packet['apd_applied'], native_dtype=str(raw.dtype), native_capture_and_replay_exact=True,
                        prediction_bits={name: packed(a) for name, a in frozen.items()}, evaluator_bits=packed(truth_model),
                        query_GT_opened_after_all_predictions=True, feature_cache_written=False,
                        cost=dict(inference_s=inference_s, clustering_cache=cache_audit.copy(), support_encoder_calls=3, query_encoder_calls=1))
                    report['records'].append(item); save()
                    print(json.dumps(dict(completed=len(report['records']), fold=row['fold'], c=c,
                        inference_s=inference_s, native_IU=item['iu']['foris_crf'], paired_IU=item['iu']['foris_paired_only'])), flush=True)
                del raw, part1, packet, bank, views, stats, predictions, frozen, probabilities
        report['state'] = 'COMPLETED'; report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(); save()
    except BaseException as error:
        report.update(state='ERROR', error=repr(error)); save(); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-only', action='store_true'); p.add_argument('--self-check', action='store_true')
    p.add_argument('--prepared-root', default='/root/autodl-tmp/demo9')
    p.add_argument('--shared-root', default='/root/demo4_cache')
    p.add_argument('--foris-root', default='/root/autodl-tmp/demo8_local_verification/foris_source')
    p.add_argument('--prepared-manifest'); p.add_argument('--projection-basis')
    p.add_argument('--resource-guard-state'); p.add_argument('--allow-gpu', action='store_true'); p.add_argument('--out')
    a = p.parse_args()
    answer = cpu_check(a) if a.self_check else prepare(a) if a.prepare_only else run(a)
    if answer is not None: print(json.dumps(answer))


if __name__ == '__main__': main()
