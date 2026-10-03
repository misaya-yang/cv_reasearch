#!/usr/bin/env python3
"""Finite synthetic CPU checks of actual decoder training; no DINO/task result.

All checkpoint fixtures live in TemporaryDirectory and are deleted. The only
persistent output is a new small JSON receipt; an existing receipt is preserved.
"""
import argparse
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tics.reference_response_decoder import ReferenceResponseDecoder, segmentation_loss
from tics.reference_response_protocol import validate_episode_manifest
from tics.reference_response_runtime import decoder_arguments
from tics.reference_response_training import (MatchedResponseTrainer, TrainingPolicy,
                                              class_miou, predict_logits,
                                              validated_training_manifest)


def image(split, index):
    return f'{split}/{index:012d}.jpg'


def fixture_manifest():
    def row(split, index, category):
        return dict(id=f'{split}{index}', c=category,
                    support=image('train2014' if split == 'train' else 'val2014', 2*index+1),
                    query=image('train2014' if split == 'train' else 'val2014', 2*index+2))
    full = dict(schema='reference_response_episodes_v1', annotation_protocol='official_COCO20i',
                held_fold=0, coco_year=2014, test_scope='frozen_confirmation',
                train=[row('train', i, 1) for i in range(3)],
                dev=[row('dev', 10+i, category) for i, category in enumerate((2, 2, 3))],
                test=[row('test', 30, 0)])
    receipt = validate_episode_manifest(full, prior_development_photo_ids=set())
    projection = {key: deepcopy(full[key]) for key in
                  ('schema', 'annotation_protocol', 'held_fold', 'coco_year', 'train', 'dev')}
    return projection, receipt


def packet_fixture(row, *, stochastic=False):
    # Native stream tensors are fixed per photo; optional noise exercises both
    # Python and torch RNG restoration without depending on an external cache.
    index = int(''.join(character for character in row['id'] if character.isdigit()))
    generator = torch.Generator().manual_seed(1200+index)
    def feature(channels):
        return torch.randn(1, channels, 2, 3, generator=generator)
    def prefix(channels):
        return torch.randn(1, 2, channels, generator=generator)
    fields = dict(support_final=feature(4), support_middle=feature(16),
                  support_coverage=torch.tensor([[[[0., .25, 1.], [.5, 1., 0.]]]]),
                  query_final=feature(4), query_middle=feature(16),
                  query_immediate_plus=feature(4), query_immediate_minus=feature(4),
                  query_response_plus=feature(4), query_response_minus=feature(4),
                  support_prefix=prefix(16), query_prefix=prefix(16),
                  prefix_immediate_plus=prefix(4), prefix_immediate_minus=prefix(4))
    if stochastic:
        fields['query_final'] = fields['query_final']+.01*(torch.randn_like(fields['query_final'])+random.random())
    return dict(features=fields)


class Providers:
    def __init__(self, trainer, *, fail_after_labels=None):
        self.trainer = trainer
        self.fail_after_labels = fail_after_labels
        self.labels_calls = 0
        self.score_calls = 0
        self.features_seen = []
        self.frozen_seen = []
        self.forward_events = []
        self.handles = [model.register_forward_hook(
            lambda model, inputs, output, arm=arm: self.forward_events.append((arm, model.training, torch.is_grad_enabled())))
            for arm, model in trainer.models.items()]

    def feature(self, row):
        assert row['id'].startswith(('train', 'dev'))
        assert set(row) == {'id', 'c', 'support', 'query'}
        self.features_seen.append(row['id'])
        return packet_fixture(row, stochastic=True)

    def labels(self, row):
        assert row['id'].startswith('train')
        if self.fail_after_labels is not None and self.labels_calls >= self.fail_after_labels:
            raise RuntimeError('Synthetic ERROR injection after completed epoch')
        self.labels_calls += 1
        labels = (torch.arange(20).reshape(1, 1, 4, 5) % 3 == 0).float()
        valid = torch.ones_like(labels, dtype=torch.bool)
        valid[:, :, 0, 0] = False
        return labels, valid

    def score(self, frozen_logits, row):
        assert row['id'].startswith('dev')
        assert frozen_logits.shape == (1, 1, 2, 3)
        assert not frozen_logits.requires_grad and frozen_logits.grad_fn is None
        assert torch.isfinite(frozen_logits).all()
        # Both maps, including response, already exist before any scoring label
        # for this row can be opened. No gradients/train mode remain in them.
        assert self.forward_events[-2:] == [('static', False, False), ('response', False, False)]
        self.frozen_seen.append(row['id'])
        epoch = self.score_calls//6+1
        self.score_calls += 1
        dev_index = int(row['id'][3:])-10
        # Deliberately reverses epoch ranking under per-episode average IoU.
        return ((9, 10), (0, 100), (1, 2))[dev_index] if epoch == 1 else ((1, 10), (50, 100), (1, 2))[dev_index]

    def close(self):
        for handle in self.handles:
            handle.remove()


def recursive_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            recursive_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            recursive_equal(a, b)
    else:
        assert left == right


def reject(function):
    try:
        function()
    except (ValueError, TypeError):
        return
    raise AssertionError('Invalid API input accepted')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=BASE/'results/native_membership_v1/response_training_cpu.json')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve existing CPU proof: '+str(args.out))
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(31027)
    checks = []
    def record(name, **evidence):
        checks.append(dict(name=name, passed=True, **evidence))
    projection, protocol = fixture_manifest()
    validated_training_manifest(projection, protocol)
    reject(lambda: validated_training_manifest(dict(projection, test=[]), protocol))
    bad = deepcopy(projection)
    bad['train'][0]['c'] = 0
    reject(lambda: validated_training_manifest(bad, protocol))
    bad = deepcopy(projection)
    bad['dev'][0]['query'] = image('val2014', 1)
    reject(lambda: validated_training_manifest(bad, protocol))
    record('external_protocol_then_train_dev_only_projection_rejects_extra_split_held_class_photo_leakage')

    model = ReferenceResponseDecoder(4, 16, query_chunk_size=3)
    policy = TrainingPolicy(max_epochs=3, min_epochs=2, patience=1, time_cap_seconds=120.)
    trainer = MatchedResponseTrainer(model, projection, protocol, policy=policy)
    for left, right in zip(trainer.models['static'].parameters(), trainer.models['response'].parameters()):
        assert torch.equal(left, right) and left.data_ptr() != right.data_ptr()
    assert trainer.optimizers['static'] is not trainer.optimizers['response']
    assert all(group['lr'] == .0003 and group['weight_decay'] == .01
               for optimizer in trainer.optimizers.values() for group in optimizer.param_groups)
    record('identical_initial_theta_independent_storage_and_fixed_AdamW',
           parameters_per_arm=sum(p.numel() for p in model.parameters()), lr=.0003, weight_decay=.01)

    packet = packet_fixture(projection['dev'][0])
    changed = deepcopy(packet)
    for key in ('query_response_plus', 'query_response_minus'):
        changed['features'][key].fill_(float('nan'))
    first = predict_logits(trainer.models['static'], packet, static=True)
    second = predict_logits(trainer.models['static'], changed, static=True)
    assert torch.equal(first, second)
    static_args = decoder_arguments(packet, static=True)
    assert 'response_plus' not in static_args and 'response_minus' not in static_args
    assert all(key in static_args for key in ('support_prefix', 'query_prefix', 'prefix_immediate_plus', 'prefix_immediate_minus'))
    dynamic_first = predict_logits(trainer.models['response'], packet, static=False)
    perturbed = deepcopy(packet)
    perturbed['features']['query_response_plus'].zero_()
    dynamic_second = predict_logits(trainer.models['response'], perturbed, static=False)
    assert not torch.equal(dynamic_first, dynamic_second)
    reject(lambda: predict_logits(model, packet, static=False, query_labels=torch.ones(1)))
    bad_packet = deepcopy(packet)
    bad_packet['features']['query_labels'] = torch.ones(1)
    reject(lambda: predict_logits(model, bad_packet, static=False))
    assert set(inspect.signature(predict_logits).parameters) == {'model', 'packet', 'static'}
    record('final_dynamic_streams_absent_from_static_immediate_and_prefix_present_prediction_rejects_labels',
           static_mask_shape=list(first.shape), dense_query_patches=first.numel())

    score1, totals1 = class_miou([(2, 9, 10), (2, 0, 100), (3, 1, 2)])
    score2, _ = class_miou([(2, 1, 10), (2, 50, 100), (3, 1, 2)])
    assert score2 > score1 and (.1+.5+.5)/3 < (.9+0+.5)/3
    assert totals1['2'] == dict(I=9, U=110)
    record('selection_uses_equal_class_sum_I_over_sum_U_not_episode_average',
           epoch1_class_miou=score1, epoch2_class_miou=score2,
           episode_mean_iou_ranking_is_opposite=True)

    with tempfile.TemporaryDirectory(prefix='response-training-cpu-') as temporary:
        root = Path(temporary)
        providers = Providers(trainer)
        whole = trainer.run(root/'whole', feature_provider=providers.feature,
                            training_label_provider=providers.labels, dev_scorer=providers.score)
        providers.close()
        assert whole['state'] == 'CONVERGED_PATIENCE' and whole['completed_epochs'] == 3
        assert providers.labels_calls == 9 and providers.score_calls == 18
        assert all(best['epoch'] == 2 for best in whole['best'].values())
        expected_shuffle = random.Random(policy.seed)
        for epoch in whole['history']:
            order = list(range(3))
            expected_shuffle.shuffle(order)
            assert epoch['episode_order'] == [projection['train'][i]['id'] for i in order]
        assert trainer.models['static'].output.weight.data_ptr() != trainer.models['response'].output.weight.data_ptr()
        assert not torch.equal(trainer.models['static'].output.weight, trainer.models['response'].output.weight)
        record('matched_episode_order_train_loss_only_labels_and_dev_after_both_predictions_frozen',
               epochs=3, train_episodes=3, dev_episodes=3, label_calls=9, scorer_calls=18,
               best_epoch_per_arm={arm: value['epoch'] for arm, value in whole['best'].items()})

        interrupted = MatchedResponseTrainer(model, projection, protocol, policy=policy)
        failure = Providers(interrupted, fail_after_labels=4)
        error = interrupted.run(root/'resumed', feature_provider=failure.feature,
                                training_label_provider=failure.labels, dev_scorer=failure.score)
        failure.close()
        assert error['state'] == 'ERROR' and error['completed_epochs'] == 1
        assert len(error['history']) == 1 and all(best['epoch'] == 1 for best in error['best'].values())
        resumed_trainer = MatchedResponseTrainer(model, projection, protocol, policy=policy)
        remaining = Providers(resumed_trainer)
        remaining.score_calls = 6
        resumed = resumed_trainer.run(root/'resumed', feature_provider=remaining.feature,
                                     training_label_provider=remaining.labels, dev_scorer=remaining.score,
                                     resume_error=True)
        remaining.close()
        assert resumed['state'] == whole['state'] and resumed['history'] == whole['history']
        for arm in trainer.models:
            recursive_equal(trainer.models[arm].state_dict(), resumed_trainer.models[arm].state_dict())
            recursive_equal(trainer.optimizers[arm].state_dict(), resumed_trainer.optimizers[arm].state_dict())
        whole_state = torch.load(root/'whole/last.pt', weights_only=False)
        resumed_state = torch.load(root/'resumed/last.pt', weights_only=False)
        recursive_equal(whole_state['rng'], resumed_state['rng'])
        recursive_equal(whole_state['shuffle_state'], resumed_state['shuffle_state'])
        assert set(whole_state) == {'contract', 'epoch', 'history', 'best', 'models', 'optimizers', 'rng', 'shuffle_state'}
        for arm in trainer.models:
            best_state = torch.load(root/'whole'/(arm+'_best.pt'), weights_only=False)
            assert best_state['epoch'] == 2 and best_state['optimizers'] and best_state['rng']
        record('ERROR_resume_matches_uninterrupted_weights_optimizers_RNG_shuffle_and_epoch_ledger_exact',
               failure_completed_epochs=1, resumed_completed_epochs=3, raw_feature_tensors_checkpointed=False)

        reject(lambda: resumed_trainer.run(root/'whole', feature_provider=remaining.feature,
                                           training_label_provider=remaining.labels, dev_scorer=remaining.score,
                                           resume_error=True))
        try:
            trainer.run(root/'whole', feature_provider=providers.feature,
                        training_label_provider=providers.labels, dev_scorer=providers.score)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing run overwritten')
        record('only_ERROR_resume_allowed_and_fresh_directory_is_exclusive')

        limited = MatchedResponseTrainer(model, projection, protocol,
            policy=TrainingPolicy(max_epochs=1, min_epochs=1, patience=1, time_cap_seconds=120.))
        limited_providers = Providers(limited)
        capped = limited.run(root/'epochs', feature_provider=limited_providers.feature,
                             training_label_provider=limited_providers.labels, dev_scorer=limited_providers.score)
        limited_providers.close()
        assert capped['state'] == 'PARTIAL_BUDGET' and capped['reason'] == 'max_epochs' and capped['completed_epochs'] == 1
        record('finite_epoch_budget_does_not_claim_convergence', completed_epochs=1, state=capped['state'])

        timed = MatchedResponseTrainer(model, projection, protocol,
            policy=TrainingPolicy(max_epochs=3, min_epochs=2, patience=1, time_cap_seconds=1e-9))
        def unopened(*args):
            raise AssertionError('Expired budget opened provider')
        expired = timed.run(root/'time', feature_provider=unopened, training_label_provider=unopened, dev_scorer=unopened)
        assert expired['state'] == 'PARTIAL_BUDGET' and expired['completed_epochs'] == 0
        assert expired['reason'] == 'time_cap_seconds'
        reject(lambda: timed.run(root/'time', feature_provider=unopened, training_label_provider=unopened,
                                 dev_scorer=unopened, resume_error=True))
        record('time_cap_is_explicit_partial_budget_no_provider_access_no_resume', completed_epochs=0)

        minimum = MatchedResponseTrainer(model, projection, protocol,
            policy=TrainingPolicy(max_epochs=4, min_epochs=3, patience=1, time_cap_seconds=120.))
        minimum_providers = Providers(minimum)
        def constant_score(logits, row):
            minimum_providers.score(logits, row)
            return (1, 2)
        converged = minimum.run(root/'minimum', feature_provider=minimum_providers.feature,
                                training_label_provider=minimum_providers.labels, dev_scorer=constant_score)
        minimum_providers.close()
        assert converged['state'] == 'CONVERGED_PATIENCE' and converged['completed_epochs'] == 3
        record('patience_requires_both_arms_and_declared_minimum_epochs', completed_epochs=3, min_epochs=3)
        fixture_path = root
    assert not fixture_path.exists()
    assert not torch.cuda.is_initialized()
    record('checkpoint_fixtures_autoclean_and_CUDA_uninitialized')

    report = dict(schema='reference_response_training_cpu_v1', state='CPU_TRAINING_PREPARATION_PASSED',
                  passed=len(checks), checks=checks, torch_version=torch.__version__, CPU_only=True,
                  CUDA_initialized=False, DINO_executed=False, real_task_gain_measured=False,
                  elapsed_seconds=time.monotonic()-started,
                  source_hashes={str(path.relative_to(BASE)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__), BASE/'tics/reference_response_training.py')},
                  limitations=['Synthetic tiny feature grids with the actual decoder only.',
                               'No image, pretrained encoder, original refiner, real-quality or GPU-readiness evidence.',
                               'Metadata receipt cannot verify image/mask files or completeness of exposure registry.',
                               'ERROR-only resume requires deterministic providers or saved Python/torch RNG; external state is caller-owned.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(state=report['state'], passed=report['passed'], receipt=str(args.out),
                          elapsed_seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
