#!/usr/bin/env python3
"""Ten CPU native-angular adapter fixtures; optional installed FoRIS source.

Default uses a clearly labelled synthetic mock when a complete existing
FoRIS source/dependency tree is unavailable. No pretrained encoder is loaded.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch
import torch
import torch.nn.functional as F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


class SyntheticHost:
    """Only a local API fixture; never claimed to be FoRIS scoring source."""
    def _extract_features(self, support, query):
        values = F.avg_pool2d(torch.cat((support, query.unsqueeze(0))), 16)
        values = torch.cat((values, values.square(), values[:, :2] - values[:, 1:3]), dim=1)
        return F.normalize(values.unsqueeze(0), dim=2)

    def _part1_positional_debias(self, maps, masks, n_refs):
        return maps

    def _part2_background_suppression(self, maps, ref_masks, n_refs, h, w):
        fg = F.interpolate(ref_masks.float().reshape(1, 1, *ref_masks.shape[-2:]), size=(h, w), mode='area')[0, 0] > .5
        tokens = maps[0, 0].movedim(0, -1)
        mu_fg = F.normalize(tokens[fg].mean(0), dim=0)
        mu_bg = F.normalize(tokens[~fg].mean(0), dim=0)
        mu_bg = F.normalize(mu_bg - (mu_fg * mu_bg).sum() * mu_fg, dim=0)
        target = maps[:, n_refs]
        sf_raw = torch.einsum('bchw,c->bhw', target, mu_fg)[0]
        sb_raw = torch.einsum('bchw,c->bhw', target, mu_bg)[0]
        sf = (sf_raw - sf_raw.min()) / (sf_raw.max() - sf_raw.min()).clamp_min(1e-6)
        sbn = (sb_raw - sb_raw.min()) / (sb_raw.max() - sb_raw.min()).clamp_min(1e-6)
        return sf_raw - .55 * sb_raw, sf, sbn, mu_fg, target * .8

    def _binarize_response(self, response):
        return response > 0

    def predict(self, support, mask, query):
        raw = self._extract_features(support, query)
        maps = self._part1_positional_debias(raw, mask.unsqueeze(1), 1)
        score, sf, sbn, mu, target = self._part2_background_suppression(maps, mask.unsqueeze(1), 1, 4, 4)
        # Uses BOTH retained downstream fields, not only the inserted score.
        downstream = torch.einsum('bchw,c->bhw', target, mu)[0]
        return F.interpolate(self._binarize_response(score + .1 * (sf - sbn) + .2 * downstream).float()[None, None],
                             size=(64, 64), mode='nearest')[0, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foris-root', type=Path, help='Existing complete source tree; never downloaded')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out and args.out.exists():
        raise ValueError('Preserve existing CPU receipts; choose a fresh output')
    torch.set_num_threads(1)
    base = Path(__file__).resolve().parents[1]
    with patch('torch.cuda.is_available', return_value=False):
        # Isolated package avoids historical tics.__init__ auto-discovery.
        package = types.ModuleType('_native_angular_cpu')
        package.__path__ = [str(base / 'tics')]
        sys.modules[package.__name__] = package
        math_kernel = module(base / 'tics/reference_nuisance_geometry.py', package.__name__ + '.reference_nuisance_geometry')
        adapter = module(base / 'tics/native_geometry_adapter.py', package.__name__ + '.native_geometry_adapter')
        replay = module(base / 'scripts/native_rice_core_experiment.py', '_native_angular_replay_cpu')
        actual_source = args.foris_root is not None
        source_hashes = {str(path): digest(path) for path in (
            Path(__file__), base / 'tics/native_geometry_adapter.py',
            base / 'tics/reference_nuisance_geometry.py', base / 'scripts/native_rice_core_experiment.py')}
        if actual_source:
            source_path = args.foris_root / 'models/foris.py'
            if not source_path.is_file() or not (args.foris_root / 'utils').is_dir():
                raise ValueError('--foris-root must contain existing models/foris.py and utils dependencies')
            sys.path.insert(0, str(args.foris_root))
            import models.foris as source
            class Encoder(torch.nn.Module):
                def get_intermediate_layers(self, rgb, n=1, reshape=True):
                    values = F.avg_pool2d(rgb, 16)
                    return [torch.cat((values, values.square(), values[:, :2] - values[:, 1:3]), dim=1)]
            with patch.object(source.FoRIS, '_build_positional_basis', lambda this, dev: torch.eye(8)[:, :2]):
                host = source.FoRIS(Encoder(), image_size=64, svd_components=2,
                                   mask_refiner='bilinear', resize_to_orig_size=False, device='cpu').eval()
            source_hashes[str(source_path)] = digest(source_path)
            for path in sorted((args.foris_root / 'utils').glob('*.py')):
                source_hashes[str(path)] = digest(path)
        else:
            source = None
            host = SyntheticHost()
        records = []
        generator = torch.Generator(device='cpu').manual_seed(2059)
        for index in range(10):
            s = torch.randn(1, 3, 64, 64, generator=generator)
            q = torch.randn(3, 64, 64, generator=generator)
            mask = torch.zeros(1, 64, 64, dtype=torch.bool)
            mask[:, 16:48, 16:48] = True
            before = {k: v for k, v in host.__dict__.items() if callable(v)}
            original_callback = host._part2_background_suppression
            cache = replay.exact_clustering_cache(source) if actual_source else nullcontext({})
            with torch.inference_mode(), cache as cache_audit:
                with replay.capture_native(host) as stored:
                    baseline = host.predict(s, mask, q).clone()
                raw, part1, part2 = (stored[k] for k in ('_extract_features', '_part1_positional_debias', '_part2_background_suppression'))
                assert part2 is not None
                dimension = part1.shape[2]
                identity, identity_audit = math_kernel.fit_angular_geometry(torch.zeros(dimension, dimension), .1)
                assert identity_audit['state'] == 'EXACT_IDENTITY'
                with replay.frozen_native_replay(host, raw, part1, part2):
                    unchanged = adapter.native_geometry_part2(host, part1, part2, identity, mask, (4, 4))
                    assert unchanged is part2
                    assert host._extract_features(s, q) is raw
                    assert host._part1_positional_debias(raw, mask, 1) is part1
                    assert host._part2_background_suppression(part1, mask, 1, 4, 4) is part2
                    identity_prediction = host.predict(s, mask, q)
                assert torch.equal(identity_prediction, baseline)
                covariance = torch.diag(torch.linspace(0., .6 + index / 10, dimension, dtype=torch.float64))
                geometry, geometry_audit = math_kernel.fit_angular_geometry(covariance, .1)
                assert torch.linalg.eigvalsh(geometry).min() > 0 and geometry_audit['full_rank']
                tokens = F.normalize(torch.randn(7, dimension, generator=generator, dtype=torch.float64), dim=1)
                transformed = math_kernel.transform_tokens(tokens, geometry)
                metric = geometry @ geometry.T
                numerator = tokens @ metric @ tokens.T
                norms = torch.sqrt(torch.diag(numerator))
                angular_error = float((transformed @ transformed.T - numerator / (norms[:, None] * norms[None])).abs().max())
                assert angular_error < 1e-12
                nullvector = torch.zeros(dimension, dtype=torch.float64); nullvector[0] = 1
                assert torch.allclose(geometry @ nullvector, nullvector, atol=1e-12, rtol=1e-12)
                native_transformed = math_kernel.transform_native_maps(part1, geometry)
                expected = original_callback(native_transformed, ref_masks=mask.unsqueeze(1), n_refs=1, h=4, w=4)
                original_snapshot = part1.clone()
                callback_calls = []
                def checked_callback(values, *, ref_masks, n_refs, h, w):
                    callback_calls.append(1)
                    assert n_refs == 1 and (h, w) == (4, 4)
                    assert torch.equal(ref_masks, mask.unsqueeze(1))
                    assert torch.equal(values, native_transformed)
                    return original_callback(values, ref_masks=ref_masks, n_refs=n_refs, h=h, w=w)
                explicit = adapter.native_geometry_part2(host, part1, part2, geometry, mask, (4, 4), native_callback=checked_callback)
                assert len(callback_calls) == 1
                assert torch.equal(part1, original_snapshot)
                with replay.frozen_native_replay(host, raw, part1, part2):
                    automatic = adapter.native_geometry_part2(host, part1, part2, geometry, mask, (4, 4))
                for result in (explicit, automatic):
                    assert result[3] is part2[3] and result[4] is part2[4]
                    assert all(torch.equal(result[k], expected[k]) for k in range(3))
                mixed = expected[:3] + (part2[3], part2[4])
                with replay.frozen_native_replay(host, raw, part1, explicit):
                    adapted_prediction = host.predict(s, mask, q)
                with replay.frozen_native_replay(host, raw, part1, mixed):
                    expected_prediction = host.predict(s, mask, q)
                assert torch.equal(adapted_prediction, expected_prediction)
                try:
                    with replay.frozen_native_replay(host, raw, part1, explicit):
                        def failing_callback(*args, **kwargs):
                            raise RuntimeError('intentional restoration fixture')
                        adapter.native_geometry_part2(host, part1, part2, geometry, mask, (4, 4), native_callback=failing_callback)
                except RuntimeError as error:
                    assert str(error) == 'intentional restoration fixture'
                assert {k: v for k, v in host.__dict__.items() if callable(v)} == before
                records.append(dict(index=index, identity_tuple_same_object=True, native_identity_prediction_exact=True,
                    full_rank=True, angular_metric_max_error=angular_error,
                    original_mu_fg_and_target_same_objects=True, native_scoring_exact=True,
                    modified_score_tail_exact=True, callbacks_restored_on_success_and_error=True,
                    changed_score_max=float((explicit[0] - part2[0]).abs().max()),
                    actual_source=actual_source, clustering_cache=dict(cache_audit)))
        assert not torch.cuda.is_initialized(), 'CPU checks must never initialize CUDA'
    receipt = dict(state='CPU_NATIVE_ANGULAR_ADAPTER_PASSED', cases=len(records), records=records,
                   source_hashes=source_hashes, actual_source=actual_source,
                   fixture='installed_FoRIS_with_mock_encoder' if actual_source else 'synthetic_API_mock_and_math',
                   realDINO=False, pretrained_DINO=False, CRF_executed=False,
                   CUDA_initialized=False, CUDA_discovery_suppressed=True, task_gain_measured=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
