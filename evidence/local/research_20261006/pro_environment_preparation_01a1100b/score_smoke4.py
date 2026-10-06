"""Score only after both M4 prediction parts are sealed;4 exposed smoke episodes."""
import hashlib
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional
torch.set_num_threads(1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def iu(mask, truth):
    return [int(np.count_nonzero(mask & truth)), int(np.count_nonzero(mask | truth))]


def edits(mask, native, truth):
    add, delete = mask & ~native, native & ~mask
    return dict(add_TP=int(np.count_nonzero(add & truth)), add_FP=int(np.count_nonzero(add & ~truth)),
                delete_TP=int(np.count_nonzero(delete & truth)), delete_FP=int(np.count_nonzero(delete & ~truth)))


def main():
    root = Path(sys.argv[1])
    metadata = json.loads((root/'evaluation_binding.json').read_text())
    expected = {row['id']: row for row in json.loads((root/'manifest4.json').read_text())}
    locations, seals, cost, first_config = {}, {}, [], None
    # Validate every prediction and producer before opening any query GT.
    partial = (root/'sealed_remaining/partial_seal.json').exists()
    parts = ('first_complete', 'remaining', 'case3') if partial else ('first_complete', 'remaining')
    for part in parts:
        run = root/('sealed_'+part)
        seal_path = run/('partial_seal.json' if partial and part == 'remaining' else 'sealed.json')
        seal = json.loads(seal_path.read_text())
        if seal['state'] not in ('ALL_PREDICTIONS_SEALED', 'SELECTED_COMPLETE_PREDICTIONS_SEALED'):
            raise ValueError('Complete predictions must be sealed before score')
        if sha(run/'config.json') != seal['config_sha256'] or sha(run/'inference_manifest.json') != seal['manifest_sha256']:
            raise ValueError('Configuration/manifest seal changed')
        config = json.loads((run/'config.json').read_text())
        if first_config is None:
            first_config = config
        elif any(config[field] != first_config[field] for field in
                 ('method', 'encoder_binding', 'native_binding', 'ref_canvas', 'source_sha256')):
            raise ValueError('Algorithm/encoder/control producer differs across split output parts')
        for path, digest in config['source_sha256'].items():
            if sha(path) != digest:
                raise ValueError('Inference code changed after sealing')
        for row in json.loads((run/'inference_manifest.json').read_text()):
            key = row['id']
            if key not in seal['prediction_sha256']:
                continue
            if row != expected[key] or key in locations:
                raise ValueError('Episode binding changed or duplicate occurrence')
            prediction, receipt = run/'predictions'/(key+'.npz'), run/'receipts'/(key+'.json')
            if sha(prediction) != seal['prediction_sha256'][key] or sha(receipt) != seal['receipt_sha256'][key]:
                raise ValueError('Prediction/receipt changed')
            if sha(row['native_npz']) != row['native_sha256']:
                raise ValueError('Bound native variant changed')
            record = json.loads(receipt.read_text())
            if (record['info']['query_gt_used'] or set(record['arms']) !=
                    {'paired', 'mean', 'class_lda', 'ce', 'ce_self', 'ref_canvas'}):
                raise ValueError('GT flag or six-arm contract mismatch')
            if any(sha(row[field]) != digest for field, digest in record['input_sha256'].items()):
                raise ValueError('RGB/reference mask changed')
            locations[key] = (prediction, record)
            cost.append(dict(id=key, method_wall_seconds=record['info']['wall_seconds'],
                             encoder_forwards=record['info']['encoder_forwards'],
                             encoder_seconds=sum(call['seconds'] for call in record['info'].get('encoder_calls', [])),
                             peak_process_rss_bytes=record['process_peak_rss_bytes'],
                             fallback={arm: value['fallback'] for arm, value in record['arms'].items()},
                             CE=record['info'].get('statistics', {}).get('ce', {}).get('solver')))
        seals[part] = sha(seal_path)
    if set(locations) != set(expected) or len(locations) != 4:
        raise ValueError('Require the entire prebound four-episode set')
    by_id = {item['id']: item for item in metadata}
    for key, (prediction, _) in locations.items():
        with np.load(prediction, allow_pickle=False) as archive:
            for arm in ('paired', 'mean', 'class_lda', 'ce', 'ce_self', 'ref_canvas'):
                for space, shape in (('work', (1024, 1024)),
                                     ('original', tuple(by_id[key]['original_query_hw']))):
                    mask = archive[arm+'_'+space]
                    if mask.shape != shape or not np.isin(mask, (0, 1)).all():
                        raise ValueError('Whole-mask binary shape/source query HW mismatch')
    source_rows = json.loads(Path('/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/bound600_v2/smoke4.json').read_text())
    comparison_controls = {}
    for source_row in source_rows:
        controls = {}
        for name in ('stored_mean.control', 'rcg.control', 'fine16.control', 'fine64.control'):
            descriptor = source_row['evaluation_controls'][name]
            controls[name] = dict(**descriptor, sha256=sha(descriptor['path']))
        comparison_controls[source_row['key']] = controls
    group_seal = dict(state='FOUR_EPISODES_ALL_PREDICTIONS_SEALED', component_seal_sha256=seals,
                      ids=list(expected), query_gt_read=False,
                      source_manifest_sha256=sha(root/'manifest4.json'), comparison_controls=comparison_controls)
    (root/'sealed4.json').write_text(json.dumps(group_seal, indent=2)+'\n')
    rows, accumulated, edit_sums = [], {}, {}
    for bound in metadata:
        key, class_id = bound['id'], bound['reference_class_id']
        with Image.open(bound['query_annotation']) as image:
            gt_original = np.asarray(image) == class_id
        if list(gt_original.shape) != bound['original_query_hw']:
            raise ValueError('Original query annotation/RGB HW mismatch')
        nearest_work = np.asarray(Image.fromarray(gt_original.astype(np.uint8)).resize(
            (1024, 1024), Image.Resampling.NEAREST)).astype(bool)
        if sha(bound['source_packet']) != bound['source_packet_sha256']:
            raise ValueError('Bound source packet changed before work GT scoring')
        with np.load(bound['source_packet'], allow_pickle=False) as packet:
            gt_work = np.unpackbits(packet['truth']).reshape(1024, 1024).astype(bool)
        truth = dict(work=gt_work, original=gt_original)
        prediction, record = locations[key]
        with np.load(expected[key]['native_npz'], allow_pickle=False) as native:
            native_masks = {space: native['mask_'+space].astype(bool) for space in ('work', 'original')}
        with np.load(prediction, allow_pickle=False) as archive:
            masks = {name: archive[name].astype(bool) for name in archive.files if not name.endswith('_coarse')}
        masks.update({'native_cache_variant_'+space: value for space, value in native_masks.items()})
        for name, descriptor in comparison_controls[key].items():
            if sha(descriptor['path']) != descriptor['sha256']:
                raise ValueError('Bound comparison control file changed')
            with np.load(descriptor['path'], allow_pickle=False) as archive:
                packed = archive[descriptor['key']].copy()
            if packed.shape != (131072,) or packed.dtype != np.uint8:
                raise ValueError('Expected sealed packed1024 comparison masks')
            work = np.unpackbits(packed).reshape(1024, 1024).astype(bool)
            original = functional.interpolate(torch.from_numpy(work.astype(np.float32))[None, None],
                         tuple(bound['original_query_hw']), mode='bilinear', align_corners=False)[0, 0].numpy() > .5
            safe_name = 'cached_'+name.replace('.', '_')+'_renderer_variant'
            masks[safe_name+'_work'], masks[safe_name+'_original'] = work, original
        for name, mask in masks.items():
            space = 'original' if name.endswith('_original') else 'work'
            if mask.shape != truth[space].shape:
                raise ValueError('Prediction/GT shape mismatch')
            counts = iu(mask, truth[space])
            change = edits(mask, native_masks[space], truth[space])
            rows.append(dict(id=key, class_id=class_id, arm=name, intersection=counts[0], union=counts[1],
                             IoU=counts[0]/max(counts[1], 1), **change))
            byclass = accumulated.setdefault(name, {})
            sums = byclass.setdefault(class_id, [0, 0])
            sums[0] += counts[0]
            sums[1] += counts[1]
            total_edits = edit_sums.setdefault(name, {label: 0 for label in change})
            for label, value in change.items():
                total_edits[label] += value
    scores = {name: 100*float(np.mean([value[0]/max(value[1], 1) for value in classes.values()]))
              for name, classes in accumulated.items()}
    report = dict(kind='four_exposed_RGB_episodes_full_M4_smoke_not_confirmation', episodes=4,
                  class_summed_mIoU=scores, four_edit_counts_vs_native_cache_variant=edit_sums,
                  costs=cost, component_prediction_seals=seals, protocol=dict(
                      primary='original-resolution class-summed I/U mean on actual original annotation',
                      secondary='same historical bound source packet.truth binary1024 after all predictions sealed; nearest-original resize is a separate diagnostic variant',
                      native='cached1024 full native plus Pro explicit binary-to-original bilinear reconstruction variant',
                      stored_controls='predeclared stored MEAN/RCG/fine16/fine64; original outputs reconstructed with same Pro binary renderer, variant explicitly named; historical producer not freshly reencoded',
                      current_model='actual local FP32 frozen timm DINOv3-L, no claimed official numerical equivalence',
                      exposure='first four reused public0 episodes; no independent confirmation',
                      confidence_intervals='not estimated from four smoke episodes', query_GT_only_after_all_four_sealed=True))
    for filename, value in (('report.json', report), ('episode_metrics.json', rows)):
        path = root/filename
        if path.exists() and not (root/('nearest_original_GT_variant_'+filename)).exists():
            path.rename(root/('nearest_original_GT_variant_'+filename))
        (root/filename).write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(dict(episodes=4, class_summed_mIoU=scores), indent=2))


if __name__ == '__main__':
    main()
