#!/usr/bin/env python3
"""Single authorized same-context object-removal response revision0; seal every mask before GT score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('infer', 'score'))
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '2' if args.stage == 'infer' else '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    from PIL import Image
    from ics.methods import object_removal_response as method
    torch.set_num_threads(2 if args.stage == 'infer' else 1)
    torch.set_num_interop_threads(1)
    binding = json.loads(args.binding.read_text())
    if args.stage == 'infer':
        if args.out.exists():
            raise FileExistsError('New single-case output namespace required')
        started = time.perf_counter()
        reference = np.asarray(Image.open(binding['reference_rgb']).convert('RGB')).copy()
        mask = np.asarray(Image.open(binding['reference_mask'])).copy()
        if mask.ndim != 2 or not np.isin(mask, (0, 1, 255)).all():
            raise ValueError('Already full binary original reference mask required')
        query = np.asarray(Image.open(binding['query_rgb']).convert('RGB')).copy()
        inputs = {key: sha(binding[key]) for key in ('reference_rgb', 'reference_mask', 'query_rgb', 'feature_export', 'packet_export')}
        features = torch.load(binding['feature_export'], map_location='cpu', weights_only=True, mmap=True)
        q, r = features['q'].numpy().copy(), features['r'].numpy().copy()
        with np.load(binding['packet_export'], allow_pickle=False) as packet:
            cov, score = packet['cov'].copy(), packet['score'].copy()  # No packet.truth/native/class.
        from ics.methods.huber_graph import make_mean_inputs
        from ics.methods.prepared_cpu_bundle import mean_base
        tick = time.perf_counter()
        graph_inputs, graph_origin = make_mean_inputs(q, r, cov, score)
        base, base_origin = mean_base(graph_inputs, graph_origin.get('graph_storage_dtype', 'float64'))
        base_seconds = time.perf_counter()-tick
        model_dir = Path(binding['model_dir'])
        encoder_binding = binding['encoder_binding']
        if sha(model_dir/'config.json') != encoder_binding['config_sha256'] or sha(model_dir/'model.safetensors') != encoder_binding['checkpoint_sha256']:
            raise ValueError('Frozen128 encoder assets changed')
        from ics.data import TimmDINOv3
        encoder = method.ActualRemoval128Encoder(TimmDINOv3(str(model_dir)))
        args.out.mkdir(parents=True)
        infer_binding = {key: binding[key] for key in ('id', 'reference_rgb', 'reference_mask', 'query_rgb', 'feature_export', 'packet_export', 'model_dir', 'encoder_binding')}
        dump(args.out/'input_binding.json', infer_binding)
        sources = [Path(__file__), Path(method.__file__), ROOT/'src/ics/methods/object_crop_cls.py', ROOT/'src/ics/methods/pro_message_extrapolation.py',
                   ROOT/'src/ics/methods/huber_graph.py', ROOT/'src/ics/methods/prepared_cpu_bundle.py', ROOT/'src/ics/data.py']
        config = dict(method=method.Config().__dict__, independent_new_method_count=0, revision=0,
                      descriptor='unit(unit(original)-unit(only-region-neutral-erased));actual FP32 Eva128 finalnorm CLS0 and same-crop patch response controls',
                      input_hashes=inputs, source_hashes={str(p): sha(p) for p in sources},
                      full_mask_known_BG_removal_reference_supervision=True, query_GT_in_inference=False,
                      native_cache_space='provided existing projected/quantized cache; new crop tokens raw actual128 encoder',
                      full_baseline='same exact recomputed MEAN', reference_mask_source='full original binary mask, not cov reconstruction')
        dump(args.out/'config.json', config)

        class LoggedEncoder:
            def __init__(self): self.calls = 0
            def __call__(self, image):
                tick = time.perf_counter()
                result = encoder(image)
                self.calls += 1
                print(json.dumps(dict(state='ACTUAL128_ENCODED', forward=self.calls, seconds=time.perf_counter()-tick)), flush=True)
                return result

        result = method.predict(reference, mask != 0, query, q, r, base, LoggedEncoder(),
                                 producer_binding=dict(producer=encoder_binding['producer'],
                                 frozen128=encoder_binding, native_inputs=inputs))
        for key, digest in inputs.items():
            if sha(binding[key]) != digest:
                raise ValueError('Input asset changed during single-case inference')
        predictions = {name+'_'+space: value.astype(np.uint8) for name, masks in result['masks'].items() for space, value in masks.items()}
        np.savez_compressed(args.out/'predictions.npz', **predictions)
        np.savez_compressed(args.out/'fields.npz', **result['fields'])
        if 'region_scores' in result:
            dump(args.out/'region_scores.json', result['region_scores'])
        result['info'].update(exact_shared_mean_seconds=base_seconds, shared_mean_origin=base_origin,
                              peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)),
                              complete_process_seconds=time.perf_counter()-started)
        dump(args.out/'receipt.json', result['info'])
        seal = dict(state='ALL_SINGLE_CASE_MASKS_SEALED', query_GT_opened=False,
                    hashes={name: sha(args.out/name) for name in ('input_binding.json', 'config.json', 'predictions.npz', 'fields.npz', 'receipt.json', 'region_scores.json')},
                    original_query_hw=list(query.shape[:2]))
        dump(args.out/'sealed.json', seal)
        print(json.dumps(dict(sealed=True, new_encoder_forwards=result['info']['new_encoder_forwards'],
                             process_seconds=result['info']['complete_process_seconds'])), flush=True)
    else:
        seal = json.loads((args.out/'sealed.json').read_text())
        if seal['state'] != 'ALL_SINGLE_CASE_MASKS_SEALED':
            raise ValueError('All full predictions must be sealed before query GT score')
        for name, digest in seal['hashes'].items():
            if sha(args.out/name) != digest:
                raise ValueError('Sealed prediction artifact changed')
        config = json.loads((args.out/'config.json').read_text())
        for path, digest in config['source_hashes'].items():
            if sha(path) != digest:
                raise ValueError('Immutable infer source changed')
        actual_annotation = np.asarray(Image.open(binding['query_annotation']))
        original_gt = actual_annotation == binding['reference_class_id']
        if list(original_gt.shape) != seal['original_query_hw']:
            raise ValueError('Real original query annotation/HW mismatch')
        with np.load(binding['packet_export'], allow_pickle=False) as packet:
            work_gt = np.unpackbits(packet['truth']).reshape(1024, 1024).astype(bool)
        metrics = {}
        with np.load(args.out/'predictions.npz', allow_pickle=False) as p:
            for name in p.files:
                space = 'original' if name.endswith('_original') else 'work'
                truth = original_gt if space == 'original' else work_gt
                mask, base = p[name].astype(bool), p['mean_'+space].astype(bool)
                add, delete = mask & ~base, base & ~mask
                intersection, union = int((mask & truth).sum()), int((mask | truth).sum())
                metrics[name] = dict(I=intersection, U=union, IoU=intersection/max(union, 1),
                    add_TP=int((add & truth).sum()), add_FP=int((add & ~truth).sum()),
                    delete_TP=int((delete & truth).sum()), delete_FP=int((delete & ~truth).sum()))
        with np.load(args.out/'fields.npz') as f:
            differences = {name: float(np.abs(f[name]-f['delta_cls']).max()) for name in f.files}
        diagnostics = {}
        if (args.out/'region_scores.json').exists():
            from scipy.stats import spearmanr
            rows = json.loads((args.out/'region_scores.json').read_text())
            token_gt = work_gt.reshape(64, 16, 64, 16).mean(axis=(1, 3)).ravel()
            purity = np.array([token_gt[np.asarray(row['region_ids'])].mean() for row in rows])
            for name in ('delta_cls', 'delta_patch_all', 'delta_patch_masked', 'zero_original_cls'):
                values = [row['scores'][name] for row in rows]
                active = np.array([value is not None for value in values])
                if active.sum() >= 2 and np.ptp(np.array(values, dtype=float)[active]) > 0:
                    statistic = spearmanr(np.array(values, dtype=float)[active], purity[active]).statistic
                else:statistic = np.nan
                diagnostics[name] = dict(legal_regions=len(rows), active_regions=int(active.sum()), GT_after_seal=True,
                    region_purity_spearman=float(statistic) if np.isfinite(statistic) else None,
                    no_independent_region_claim=True, constant_score=bool(active.any() and np.ptp(np.array(values,dtype=float)[active]) == 0))
        report = dict(kind='single_exposed_complete_contextual_RGB128_removal_revision0_feasibility', revision=0,
            independent_new_method_count=0, metrics=metrics, maximum_field_difference_from_deltaCLS=differences,
            region_diagnostic=diagnostics, query_GT_only_after_seal=True,
            quality_generalization='not established by one reused case', no_GT_threshold_or_parameter_selection=True)
        dump(args.out/'report.json', report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':main()
