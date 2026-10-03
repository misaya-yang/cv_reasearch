#!/usr/bin/env python3
"""Ten small orchestration cases, not pretrained DINO or segmentation gains."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from reference_response_experiment import HostPacketStore, original_binary, score_frozen, source_feature_replay
from tics.reference_response_decoder import ReferenceResponseDecoder
from tics.reference_response_training import MatchedResponseTrainer, TrainingPolicy, predict_logits
from tics.reference_response_runtime import finalize_decoder_logits
from tics.reference_response_protocol import validate_episode_manifest


def check():
    torch.set_num_threads(1)
    torch.manual_seed(31027)
    checks = []
    def record(name, **values):
        checks.append(dict(name=name, passed=True, **values))
    def name(split, uid):
        return f'{split}/COCO_{split}_{uid:012d}.jpg'
    def row(identity, category, split, a, b):
        return dict(id=identity, c=category, support=name(split, a), query=name(split, b))
    features = {key: torch.randn(1, channels, 4, 4) for key, channels in
        [('support_final', 8), ('query_final', 8), ('support_middle', 32), ('query_middle', 32),
         ('query_immediate_plus', 8), ('query_immediate_minus', 8),
         ('query_response_plus', 8), ('query_response_minus', 8)]}
    features['support_coverage'] = torch.zeros(1, 1, 4, 4)
    features['support_coverage'][:, :, :2] = 1
    for key, channels in [('support_prefix', 32), ('query_prefix', 32),
                          ('prefix_immediate_plus', 8), ('prefix_immediate_minus', 8)]:
        features[key] = torch.randn(1, 3, channels)
    packet = dict(features=features, audit=dict(CPU_fixture=True, query_GT_received=False))
    needed = sum(t.numel()*t.element_size() for t in features.values())
    store = HostPacketStore(needed)
    store.add('only', packet)
    copied = store.get('only', 'cpu')
    assert all(torch.equal(features[k], copied['features'][k]) for k in features)
    features['query_final'].add_(1)
    assert not torch.equal(features['query_final'], copied['features']['query_final'])
    try:
        store.add('overflow', packet)
    except MemoryError:
        pass
    else:
        raise AssertionError('RAM budget silently fell back to disk')
    record('lossless_FP32_host_packet_clone_budget_and_no_disk_fallback', packet_bytes=needed)
    class Source:
        def _extract_features(self, images):
            return 'original'
    host = Source()
    images = torch.randn(2, 3, 4, 4)
    with source_feature_replay(host, images, copied) as calls:
        result = host._extract_features(images[None])
        assert torch.equal(result[0, 0], copied['features']['support_final'][0])
        assert torch.equal(result[0, 1], copied['features']['query_final'][0])
        wrong = torch.empty(1, 2, 3, 8, 4)[:, :, :, ::2]
        wrong.copy_(images[None])
        try:
            host._extract_features(wrong)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Source cache accepted altered stride')
        assert len(calls) == 1
    assert host._extract_features(images) == 'original' and '_extract_features' not in host.__dict__
    record('public_source_same_input_values_dtype_shape_stride_and_hook_restoration')
    gold = np.zeros((7, 5), bool)
    gold[:4, :3] = True
    for i in range(10):
        metadata = dict(**row('test'+str(i), 0, 'val2014', 20+i*2, 21+i*2), fold=0)
        baseline = gold.copy()
        baseline[-1] = True
        response = gold.copy()
        static = np.zeros_like(gold)
        frozen = dict(foris_public=baseline, static=static, response=response)
        reads = []
        def truth(r):
            assert r == metadata and len(frozen) == 3
            reads.append(True)
            return gold
        scored = score_frozen(metadata, frozen, truth, packet['audit'])
        assert len(reads) == 1 and scored['original_iu']['response'] == [12, 12]
        assert scored['ledger']['response']['removed_fp'] == 5
        assert scored['original_oracle_iu']['pixel_false_positive_removed'] == [12, 12]
        assert scored['response_audit']['caller_masks_frozen_before_query_GT']
    record('ten_original_mask_rows_three_arms_freeze_before_query_truth_and_exact_pixel_ledger', cases=10)
    mask = torch.tensor([[True, False], [False, True]])
    original = original_binary(mask, (7, 5))
    assert np.array_equal(original, (F.interpolate(mask[None, None].float(), (7, 5),
        mode='bilinear', align_corners=False)[0, 0] > .5).numpy())
    record('source_causal_v3_original_binary_resize_exact')
    manifest = dict(schema='reference_response_episodes_v1', annotation_protocol='official_COCO20i',
        held_fold=0, coco_year=2014, test_scope='reused_development',
        train=[row('train', 1, 'train2014', 1, 2)], dev=[row('dev', 1, 'val2014', 3, 4)],
        test=[row('held', 0, 'val2014', 5, 6)])
    protocol = validate_episode_manifest(manifest, prior_development_photo_ids={5, 6})
    projection = {k: manifest[k] for k in ('schema', 'annotation_protocol', 'held_fold', 'coco_year', 'train', 'dev')}
    model = ReferenceResponseDecoder(final_channels=8, middle_channels=32, query_chunk_size=4)
    trainer = MatchedResponseTrainer(model, projection, protocol,
        policy=TrainingPolicy(max_epochs=2, min_epochs=1, patience=1, time_cap_seconds=60))
    target = torch.from_numpy(gold.copy())[None, None].float()
    updates, finalizer_calls = [], []
    def finalize(binary, rgb):
        assert rgb.shape == (1, 3, 7, 5)
        finalizer_calls.append(True)
        return binary
    def scorer(logits, r):
        assert r['id'] == 'dev'
        binary = finalize_decoder_logits(logits, torch.zeros(1, 3, 7, 5), finalize).numpy()
        return int((binary & gold).sum()), int((binary | gold).sum())
    with tempfile.TemporaryDirectory(prefix='response-entry-') as temp:
        output = Path(temp)/'train'
        receipt = trainer.run(output, feature_provider=lambda r: copied,
            training_label_provider=lambda r: target, dev_scorer=scorer, progress_callback=updates.append)
        assert receipt['state'] in ('PARTIAL_BUDGET', 'CONVERGED_PATIENCE') and receipt['completed_epochs'] > 0
        for arm, trained in trainer.models.items():
            checkpoint = torch.load(output/(arm+'_best.pt'), map_location='cpu', weights_only=False)
            assert checkpoint['contract'] == trainer.contract
            trained.load_state_dict(checkpoint['models'][arm])
            result = predict_logits(trained, copied, static=arm == 'static')
            assert torch.isfinite(result).all() and result.shape == (1, 1, 4, 4)
        assert len(finalizer_calls) == 2*receipt['completed_epochs']
        assert {'TRAIN_UPDATE', 'DEV_SCORE', 'EPOCH_COMMITTED'} <= {u['stage'] for u in updates}
    record('actual_matched_trainer_packet_bridge_progress_selected_best_and_single_finalizer',
           completed_epochs=receipt['completed_epochs'], source_refiner_is_fixture=True)
    return dict(state='CPU_RESPONSE_ENTRY_PASSED', checks=checks, cases=10, CUDA_initialized=torch.cuda.is_initialized(),
                pretrained_DINO_loaded=False, full_FoRIS_runtime_verified=False, real_task_gain_measured=False,
                source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__), BASE/'scripts/reference_response_experiment.py', BASE/'tics/reference_response_training.py')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve old receipt')
    result = check()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(state=result['state'], cases=result['cases'], checks=len(result['checks']),
                          real_task_gain_measured=False)))
