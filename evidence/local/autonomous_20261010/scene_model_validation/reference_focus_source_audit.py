"""Independent legal-reference-only audit of all200 focus geometry/bindings."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN = REPO.parent/'cv_data/a/reference_focus200_20261010'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps([list(value.shape), value.dtype.str]).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def sums32(value):
    integral = np.pad(value, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return integral[32:, 32:]-integral[:-32, 32:]-integral[32:, :-32]+integral[:-32, :-32]


def main():
    begun = time.perf_counter()
    torch.set_num_threads(2)
    cfg, complete = read(RUN/'config.json'), read(RUN/'COMPLETE.json')
    assert sha(RUN/'config.json') == complete['config_sha256']
    assert sha(RUN/'input_bindings.json') == complete['input_bindings_sha256']
    assert sha(RUN/'selection_audit.json') == complete['selection_audit_sha256'] == 'd809c7c19b5dc09239ebfb152f79ecc798be0785642e1d7acd1931a9aabc47f1'
    assert sha(RUN/'source_manifest.json') == complete['source_manifest_sha256']
    assert sha(RUN/'tasks.json') == complete['tasks_sha256']
    manifest, bindings, audit = read(RUN/'source_manifest.json'), read(RUN/'input_bindings.json'), read(RUN/'selection_audit.json')
    reference_masks = {str(Path(r['reference_mask_path']).resolve()) for r in manifest}
    aliases = {str(Path(r['query_mask_path']).resolve()) for r in manifest} & reference_masks
    forbidden = {str(Path(r['query_mask_path']).resolve()) for r in manifest}-reference_masks
    forbidden |= {str((REPO.parent/'cv_data'/r['query_path']).resolve()) for r in manifest}
    attempts = []
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = str(Path(os.fsdecode(args[0])).resolve())
            if path in forbidden:
                attempts.append(path)
                raise RuntimeError('Forbidden query image or query annotation access')
    sys.addaudithook(guard)
    method_path = REPO/'src/ics/methods/reference_focus_head.py'
    assert sha(method_path) == '37c0f57254ff7d398b7cdfa5486d2bbd4c51a03958206e2d0fc6445329a48413'
    spec = importlib.util.spec_from_file_location('independent_focus_source_measure', method_path)
    head = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(head)
    records, changed = [], {'deepglobe_road': 0, 'paco_part': 0}
    for row, binding, selected in zip(manifest, bindings, audit['episodes']):
        assert row['episode_id'] == binding['episode_id'] == selected['episode_id']
        assert row['reference_rgb_hash'] == binding['reference_rgb_hash']
        assert row['reference_mask_hash'] == binding['reference_mask_hash'] == selected['reference_mask_hash']
        with Image.open(row['reference_mask_path']) as image:
            raw = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
        assert array_hash(raw) == row['reference_mask_hash']
        canvas = F.interpolate(torch.from_numpy(raw)[None, None].float(), (1024, 1024), mode='nearest')[0, 0]
        assert array_hash(canvas.numpy().astype(np.uint8)) == binding['canonical_mask_uint8_array_sha256'] == selected['canonical_mask_uint8_array_sha256']
        c16 = canvas.double().reshape(64, 16, 64, 16).mean((1, 3)).numpy()
        c8 = canvas.double().reshape(128, 8, 128, 8).mean((1, 3)).numpy()
        variance = c8.square() if hasattr(c8, 'square') else c8*c8
        variance = variance.reshape(64, 2, 64, 2).mean((1, 3))-c16*c16
        gains, foregrounds = sums32(variance), sums32(c16)
        best = max(((float(gains[y, x]), float(foregrounds[y, x]), -y, -x) for y in range(33) for x in range(33)))
        y0, x0 = -best[2]*16, -best[3]*16
        expected_box = [x0, y0, x0+512, y0+512]
        assert expected_box == binding['box_xyxy'] == selected['box_xyxy']
        assert best[0] == selected['refinement_variance_sum']
        focus_c = canvas[y0:y0+512, x0:x0+512].reshape(64, 8, 64, 8).mean((1, 3)).numpy()
        assert sha(binding['coverage64_path']) == binding['coverage64_file_sha256']
        with Path(binding['coverage64_path']).open('rb') as stream:
            saved_coverage = np.load(stream, allow_pickle=False)
        assert np.array_equal(focus_c, saved_coverage)
        assert array_hash(focus_c) == binding['coverage64_tensor_sha256'] == selected['focus_coverage_fp32_array_sha256']
        measure = head._reference_measure(canvas.double(), tuple(expected_box))
        sample = head._role_quadrature(measure)
        assert measure['diagnostics']['foreground_mass_absolute_error'] == measure['diagnostics']['background_mass_absolute_error'] == 0
        assert float(measure['physical_area'].sum()) == 1024**2
        assert sample['diagnostics']['foreground_occurrences'] == sample['diagnostics']['background_occurrences'] == 128
        assert abs(sample['diagnostics']['numerical_weighted_target_mean']) <= 1e-12
        assert sha(binding['entry_path']) == binding['entry_sha256']
        entry = read(binding['entry_path'])
        assert entry['input_tensor_hash'] == binding['input_tensor_hash']
        assert entry['file_sha256'] == binding['payload_sha256']
        assert entry['features']['O/24'] == binding['O24']
        changed[row['dataset']] += bool(selected['differs_from_initial_PIL_box'])
        records.append(dict(episode_id=row['episode_id'], box=expected_box, independent_box_exact=True,
                            independent_refinement_variance=best[0], focus_coverage_exact=True,
                            full_fg_mass=float(measure['foreground'].sum()), full_bg_mass=float(measure['background'].sum()),
                            role_occurrences=[sample['diagnostics']['foreground_occurrences'], sample['diagnostics']['background_occurrences']],
                            selected_source_tokens=sample['diagnostics']['sample_count'],
                            nF_nB_risk_weights=[sample['diagnostics']['foreground_risk_weight'], sample['diagnostics']['background_risk_weight']],
                            target_mean=sample['diagnostics']['numerical_weighted_target_mean'], actual_raw_binding_metadata_exact=True))
    assert len(records) == 200 and changed == {'deepglobe_road': 0, 'paco_part': 18}
    assert not attempts
    result = dict(status='PASSED_ALL200_REFERENCE_ONLY_INDEPENDENT_FOCUS_GEOMETRY_AND_MASS', n=200,
                  producer_COMPLETE_sha256=sha(RUN/'COMPLETE.json'), input_bindings_sha256=complete['input_bindings_sha256'],
                  corrected_audit_sha256=complete['selection_audit_sha256'], source_manifest_sha256=complete['source_manifest_sha256'],
                  validator_sha256=sha(__file__), head_sha256=sha(method_path), changed_boxes_vs_old_PIL=changed,
                  query_images_read=0, query_GT_reads=0, blocked_open_attempts=attempts,
                  legal_reference_mask_paths_also_query_mask_paths=len(aliases), real_feature_arrays_read=0,
                  all200_exact_max_variance_FG_mass_rowmajor_selection=True,
                  all200_exact_focus8x8_coverage_and_physical_mass=True, records=records,
                  actual_raw_payloads_rehashed=False,
                  note='Raw binding metadata checked; producer verified all payloads. Independent actual replay will read selected payloads after200 output seal.',
                  elapsed_seconds=time.perf_counter()-begun)
    (OUT/'reference_focus_source_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
