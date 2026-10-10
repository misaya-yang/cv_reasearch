"""Fundus released sampler/CLI geometry audit; no labels or prediction pixels."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import zipfile
sys.dont_write_bytecode = True
import numpy as np

REPO = Path(__file__).resolve().parents[4]
DATA = REPO.parent/'cv_data'
OUT = Path(__file__).resolve().parent
RUN = DATA/'a/fundus_foris_seed0_20261010'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def npy_header(archive, name):
    with archive.open(name) as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                                  else np.lib.format.read_array_header_2_0(stream))
    return dict(shape=list(shape), dtype=str(dtype), fortran=fortran)


def main():
    began = time.perf_counter()
    cfg, seal, prepared = read(RUN/'config.json'), read(RUN/'sealed.json'), read(RUN/'prepared.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED' and seal['n'] == cfg['n'] == 200
    assert sha(RUN/'config.json') == seal['config_sha256']
    assert sha(RUN/'manifest.json') == seal['manifest_sha256'] == cfg['manifest_sha256']
    assert sha(RUN/'inference.json') == seal['inference_sha256']
    assert sha(RUN/'prepared.json') == cfg['prepared_sha256']
    manifest, records = read(RUN/'manifest.json'), read(RUN/'inference.json')
    assert prepared['seed'] == 0 and prepared['shots'] == 1 and prepared['num_workers'] == 0 and prepared['shuffle'] is False
    assert prepared['folds']['fundus/-1']['official_length'] == prepared['folds']['fundus/-1']['selected_length'] == 200
    assert prepared['folds']['fundus/-1']['expected_class_ids'] == [0]
    forbidden = {str(Path(row['query_mask_path']).resolve()) for row in manifest}
    forbidden.update(str((DATA/row['query_annotation_path']).resolve()) for row in manifest)
    forbidden.update(str((DATA/row['reference_annotation_path']).resolve()) for row in manifest)
    forbidden.update(str(Path(row['reference_mask_path']).resolve()) for row in manifest)
    forbidden.update(str((RUN/name).resolve()) for name in ('report.json', 'episode_metrics.json', 'COMPLETE.json'))
    attempts, writes = [], []
    prefix = str(RUN.resolve())+os.sep
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).resolve())
        if path in forbidden:
            attempts.append(path)
            raise RuntimeError('Forbidden annotation or scoring-output access')
        mode, flags = args[1:3]
        write = ((isinstance(mode, str) and any(c in mode for c in 'wa+'))
                 or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))))
        if path.startswith(prefix) and write:
            writes.append(path)
            raise RuntimeError('Forbidden completed-run write')
    sys.addaudithook(guard)
    protected = [RUN/name for name in ('config.json', 'sealed.json', 'manifest.json', 'inference.json')]
    before = {str(path): sha(path) for path in protected}
    frozen = Path(cfg['frozen_assets'])
    for rel, digest in cfg['source_sha256'].items():
        assert sha(RUN/'frozen'/rel) == digest
    for rel, digest in cfg['external_source_sha256'].items():
        assert sha(frozen/rel) == digest
    author_root = frozen/'third_party/foris_official'
    for rel, digest in prepared['source']['files'].items():
        if rel != 'docs/data.md':
            assert sha(author_root/rel) == digest
    dataset_path = author_root/'datasets/fundus.py'
    assert sha(dataset_path) == 'f5c3b51a682f96f8631407d56db45427a1796e53b897576e70fffc99a211e919'
    spec = importlib.util.spec_from_file_location('independent_frozen_fundus_sampler', dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dataset = module.build(SimpleNamespace(data_root=str(DATA/'datasets/ics'), shots=1))
    assert dataset.split == 'test' and dataset.num == dataset.total_samples == len(dataset) == 200
    assert list(dataset.class_ids) == [0] and len(dataset.img_metadata_classwise['1']) == 200
    np.random.seed(0)
    paired = []
    for index, row in enumerate(manifest):
        query, reference, class_id = dataset.sample_episode(index)
        assert row['official_index'] == index and row['loader_class_id'] == class_id == 0
        assert str(Path(query).relative_to(DATA)) == row['query_path']
        assert [str(Path(path).relative_to(DATA)) for path in reference] == [row['reference_path']]
        assert query != reference[0] and row['reference_crop'] is None and row['query_crop'] is None
        assert row['shots'] == 1 and row['query_size_hw'] == row['query_mask_size_hw'] == [2048, 2048]
        paired.append([Path(query).name, Path(reference[0]).name])
    opts_spec = importlib.util.spec_from_file_location('independent_frozen_fundus_opts', author_root/'opts.py')
    opts = importlib.util.module_from_spec(opts_spec)
    opts_spec.loader.exec_module(opts)
    defaults = opts.get_args_parser().parse_args(['--dataset', 'fundus', '--crf-mask-refinement'])
    assert defaults.image_size == cfg['image_size'] == 1024 and defaults.shots == 1 and defaults.seed == 0
    assert defaults.fundus_split == 'test' and defaults.fundus_num_episodes is None and defaults.num_workers == 0
    assert defaults.svd_comps == cfg['svd_components'] == 500 and defaults.tau == cfg['tau'] == .6
    assert defaults.model_size == 'large' and defaults.crf_mask_refinement and cfg['resize_to_orig_size'] is False
    profile = read(RUN/'profile.json')
    assert profile == read(cfg['profile_path']) and sha(cfg['profile_path']) == cfg['profile_sha256']
    assert profile['model'] == 'DINOv3-L/16' and profile['encoder_dtype'] == profile['storage_dtype'] == 'float32'
    assert profile['producer_device'] == cfg['producer_device'] == 'mps'
    assert profile['preprocessing']['resize'] == [1024, 1024] and profile['features']['patch_grid'] == [64, 64]
    assert profile['weights_sha256'] == cfg['weights_sha256']
    assert profile['encoder_source_sha256'] == cfg['source_sha256']['src/ics/data.py']
    assert profile['observer_source_sha256'] == cfg['source_sha256']['src/ics/representations.py']
    assert profile['preprocessing']['source_sha256'] == cfg['external_source_sha256']['third_party/foris_official/utils/data.py']
    assert sha(frozen/'native_assets/positional_basis.pt') == cfg['basis_sha256']
    packed_shapes, entries = {}, set()
    for index, (row, rec) in enumerate(zip(manifest, records)):
        assert rec['episode_id'] == row['episode_id'] and rec['index'] == index
        assert rec['query_GT_reads'] == rec['query_label_attempts'] == rec['encoder_calls'] == 0
        assert [request['role'] for request in rec['raw']] == ['reference', 'query']
        for request in rec['raw']:
            entry = read(request['entry_path'])
            assert entry['profile_id'] == Path(cfg['profile_path']).parent.name
            assert entry['input_tensor_hash'] == request['input_tensor_hash'] and entry['file_sha256'] == request['payload_sha256']
            assert entry['features']['O/24']['tensor_sha256'] == request['tensor_sha256']
            assert entry['features']['O/24']['shape'] == [4096, 1024] and entry['features']['O/24']['dtype'] == 'float32'
            entries.add(request['key'])
        with zipfile.ZipFile(RUN/'predictions'/rec['filename']) as archive:
            cli = npy_header(archive, 'cli/foris.crf.npy')
            original = npy_header(archive, 'original/foris.crf.npy')
            assert cli == dict(shape=[1024*1024//8], dtype='uint8', fortran=False)
            assert original == dict(shape=[2048*2048//8], dtype='uint8', fortran=False)
            with archive.open('original_hw.npy') as stream:
                hw = np.load(stream, allow_pickle=False)
            assert hw.tolist() == [2048, 2048]
            packed_shapes[str(index)] = dict(cli=cli, original=original, original_hw=hw.tolist())
    assert len(entries) == cfg['unique_rgb_inputs'] == 173
    after = {str(path): sha(path) for path in protected}
    assert before == after and not attempts and not writes
    report = dict(status='PASSED_FUNDUS_RELEASED_DEFAULT200_AND_CLI_GEOMETRY_NO_GT', n=200,
                  public_source=prepared['source'], manifest_sha256=cfg['manifest_sha256'],
                  sampler=dict(default_split='test', default_num=None, real_pool=200, actual_default_draws=200,
                               shots=1, seed=0, dataloader_workers=0, all200_frozen_pairs_exact=True,
                               reference_never_query_same_file=True, retained_repeated_draws=True,
                               unique_query_photographs=len({row['query_photo_id'] for row in manifest})),
                  model_protocol=dict(backbone='DINOv3-L/16', weights_sha256=cfg['weights_sha256'],
                                      model_config_sha256=profile['model_config_sha256'], branch='O/24', channels=1024,
                                      patch_grid=[64, 64], input_canvas=[1024, 1024], producer='MPS FP32 B2',
                                      frozen_source_and_profile_identities_match=True, raw_unique_inputs=173,
                                      basis_sha256=cfg['basis_sha256'], svd_components=500, tau=.6,
                                      CRF='original solver/settings enabled as paper command flag'),
                  geometry=dict(primary_cli=[1024, 1024], additional_original=[2048, 2048],
                                factory_resize_to_orig_size=False, original_render='binary CLI result bilinear align_corners=False then >.5',
                                raw_GT_alignment='released inference.py direct torch nearest to actual prediction shape, >.5',
                                source_Fundus_mask_binarization='L uint8 values>=128; manifest stores this already-binary mask',
                                no_Fundus_ignore_map=True, all200_packed_array_headers_match=True),
                  metric='Fundus single class foreground pooled intersection / max(pooled union,1), consistent with released AverageMeter',
                  observations='Official set_reference/set_target/segment entry; source-stage taps return each original output unchanged.',
                  query_GT_reads=0, reference_mask_pixels_read=0, prediction_mask_pixels_read=0, real_feature_arrays_read=0,
                  segmentation_rerun=False, denied_input_open_attempts=attempts, completed_write_attempts=writes,
                  protected_hashes_unchanged=before == after, validator_sha256=sha(__file__),
                  source_locations={'episode_default':'datasets/fundus.py:42', 'sampling':'datasets/fundus.py:75',
                                    'CLI_native_geometry':'models/__init__.py:97', 'direct_GT_resize':'inference.py:83',
                                    'additional_original_render':'frozen/scripts/run_extended_foris.py:236'},
                  limitations=['Source-compatible CLI geometry and exact local seed0 draws do not assert CUDA numerical parity or reproduce the paper table value.',
                               'Bare CLI defaults do not enable CRF; this complete FoRIS baseline explicitly uses the published --crf-mask-refinement choice.',
                               'The paper source generic Fundus loader does not itself pin a FIVES archive hash; source-pool receipts separately identify FIVES v1.',
                               'Compute agent owns independent full prediction/field/payload/weights hashes and I/U. This complementary audit read headers/metadata only.'],
                  packed_headers=packed_shapes, elapsed_seconds=time.perf_counter()-began)
    (OUT/'fundus_protocol_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('packed_headers', 'public_source')}, indent=2))


if __name__ == '__main__':
    main()
