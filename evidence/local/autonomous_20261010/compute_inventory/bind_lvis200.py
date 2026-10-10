"""Read-only binding of the completed LVIS200 paired artifacts to raw O24.

No model construction, cache writes, query-mask reads, or segmentation occurs.
Only the original RGB transform is reapplied to identify existing raw entries.
"""
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
ASSETS = REPO.parent / 'cv_data'
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
from ics.official_data import load_inputs
from raw_feature_cache import canonical_hash, tensor_hash, file_hash


def index(path):
    return {v['episode_id']: v for v in (json.loads(line) for line in path.read_text().splitlines())}


def main():
    torch.set_num_threads(2)
    mean_run = ASSETS / 'a/lvis_mean200_20261008/run'
    foris_run = ASSETS / 'a/lvis_foris1400_score_reuse_20261009/run'
    profile_path = Path(json.loads((mean_run / 'raw_cache.json').read_text())['profile_path'])
    profile = json.loads(profile_path.read_text())
    profile_id = canonical_hash(profile)
    assert profile_path.parent.name == profile_id
    source = ASSETS / 'third_party/foris_official/utils/data.py'
    assert file_hash(source) == profile['preprocessing']['source_sha256']
    spec = importlib.util.spec_from_file_location('_inventory_official_transform', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transform = module.build_transform(1024)
    rows = json.loads((mean_run / 'manifest.json').read_text())
    mean_index, foris_index = index(mean_run / 'inference.jsonl'), index(foris_run / 'inference.jsonl')
    transformed = {}
    bindings = []
    for row in rows:
        reference, _, query = load_inputs(row, ASSETS)
        binding = dict(episode_id=row['episode_id'], row=row, raw={})
        for role, image in [('reference', reference), ('query', query)]:
            memo = row[role + '_rgb_hash']
            if memo not in transformed:
                x = transform(image).numpy()
                input_hash = tensor_hash(x)
                key = canonical_hash(dict(profile=profile_id, input_tensor_hash=input_hash))
                entry_path = profile_path.parent / key / 'entry.json'
                info = json.loads(entry_path.read_text())
                assert info['profile_id'] == profile_id and info['input_tensor_hash'] == input_hash
                assert 'O/24' in info['features']
                payload = entry_path.parent / info['file']
                assert payload.is_file() and info['file'] == info['file_sha256'] + '.npz'
                transformed[memo] = dict(key=key, entry_path=str(entry_path), payload_path=str(payload),
                    payload_sha256=info['file_sha256'], input_tensor_hash=input_hash,
                    O24=info['features']['O/24'])
            binding['raw'][role] = transformed[memo]
        for method, run, records in [('mean', mean_run, mean_index), ('foris.crf', foris_run, foris_index)]:
            record = records[row['episode_id']]
            p = run / 'predictions' / record['prediction_file']
            assert file_hash(p) == record['prediction_sha256']
            with np.load(p, allow_pickle=False) as values:
                keys = dict(original='original/' + method, cli='cli/' + method)
                assert all(k in values for k in keys.values())
            binding[method] = dict(path=str(p), sha256=record['prediction_sha256'], keys=keys)
        bindings.append(binding)
    result = dict(state='EXISTING_RAW_AND_PAIRED_PREDICTIONS_BOUND', n=len(bindings),
        manifest_path=str(mean_run / 'manifest.json'), manifest_sha256=file_hash(mean_run / 'manifest.json'),
        profile_path=str(profile_path), profile_sha256=file_hash(profile_path), profile_id=profile_id,
        unique_raw_inputs=len(transformed), encoder_calls=0, raw_cache_writes=0, query_GT_reads=0,
        verification='Exact original RGB preprocessing and input hash for every raw key; all 400 baseline prediction file hashes checked. Raw payload hashes and tensor hashes remain recorded metadata and are verified by the existing RawFeatureCache reader on use.',
        exposure='Completed, previously exposed development LVIS200; not independent benchmark confirmation.',
        rows=bindings)
    path = OUT / 'lvis200_source_binding.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['state', 'n', 'unique_raw_inputs', 'profile_path', 'encoder_calls', 'raw_cache_writes', 'query_GT_reads']}))
    print(path)


if __name__ == '__main__':
    main()
