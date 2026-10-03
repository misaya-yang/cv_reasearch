"""Finite matched-arm training preparation; no encoder, feature cache or scheduling.

The caller first validates the full episode protocol externally, then supplies ONLY
its TRAIN/DEV projection and validation receipt here. Feature providers yield the
runtime packet; training annotations enter only segmentation_loss. DEV annotations
are owned by a scorer called after each complete dense logit map is detached.
Checkpoints contain head/optimizer/RNG state, never packets or DINO observations.
Only an ERROR run may resume, replaying any incomplete epoch from its last complete
boundary. Providers must be deterministic per row (or use the saved Python/torch
RNG); external provider/cache state is not checkpointed.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import time

import torch

from tics.reference_response_decoder import ReferenceResponseDecoder, segmentation_loss
from tics.reference_response_protocol import coco_photo_id
from tics.reference_response_runtime import decoder_arguments


@dataclass(frozen=True)
class TrainingPolicy:
    max_epochs: int = 50
    min_epochs: int = 10
    patience: int = 8
    seed: int = 31027
    time_cap_seconds: float = 1200.

    def __post_init__(self):
        for name in ('max_epochs', 'min_epochs', 'patience', 'seed'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == 'seed' else 1):
                raise ValueError('Invalid finite policy: '+name)
        if self.min_epochs > self.max_epochs:
            raise ValueError('Minimum epochs exceeds maximum epochs')
        if isinstance(self.time_cap_seconds, bool) or not math.isfinite(self.time_cap_seconds) or self.time_cap_seconds <= 0:
            raise ValueError('A positive finite wall-time cap is mandatory')


def validated_training_manifest(train_dev, protocol_receipt):
    """Recheck a TRAIN/DEV-only projection against the external protocol receipt.

    Run reference_response_protocol.validate_episode_manifest BEFORE constructing
    this projection. No test rows, annotations or test provider enter this API.
    The receipt is metadata validation, not a guarantee of files/registry accuracy.
    """
    expected = {'schema', 'annotation_protocol', 'held_fold', 'coco_year', 'train', 'dev'}
    if set(train_dev) != expected:
        raise ValueError('Training projection must contain only declared TRAIN/DEV metadata')
    if (train_dev['schema'] != 'reference_response_episodes_v1'
            or train_dev['annotation_protocol'] != 'official_COCO20i'
            or protocol_receipt.get('state') != 'EPISODE_METADATA_CONTRACT_PASSED'
            or protocol_receipt.get('all_role_photo_splits_disjoint') is not True):
        raise ValueError('External episode protocol validation required')
    fold, year = train_dev['held_fold'], train_dev['coco_year']
    if type(fold) is not int or fold not in range(4) or year not in (2014, 2017) or protocol_receipt.get('held_fold') != fold:
        raise ValueError('Declared fold/year differs from validated protocol')
    photos, ids = {}, set()
    result = deepcopy(train_dev)
    for split in ('train', 'dev'):
        rows = result[split]
        if not isinstance(rows, list) or not rows:
            raise ValueError('Both TRAIN and DEV must be nonempty')
        photos[split] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {'id', 'c', 'support', 'query'}:
                raise ValueError('Only episode ID/class and support/query paths are permitted')
            if not isinstance(row['id'], str) or not row['id'] or row['id'] in ids:
                raise ValueError('Episode IDs must be globally unique')
            ids.add(row['id'])
            if type(row['c']) is not int or row['c'] not in range(80) or row['c'] % 4 == fold:
                raise ValueError('Held class or invalid class in TRAIN/DEV')
            pair = []
            for role in ('support', 'query'):
                photo = coco_photo_id(row[role])
                directory = ('train' if split == 'train' else 'val')+str(year)
                if PurePosixPath(row[role]).parts[0] != directory:
                    raise ValueError('Original split/year differs from protocol')
                pair.append(photo)
                photos[split].add(photo)
            if pair[0] == pair[1]:
                raise ValueError('Query must differ from support photo')
        counts = dict(Counter(row['c'] for row in rows))
        receipt_counts = {int(k): v for k, v in protocol_receipt['episodes_per_class'][split].items()}
        if (counts != receipt_counts or sorted(counts) != protocol_receipt[split+'_classes']
                or len(photos[split]) != protocol_receipt['unique_photos'][split]):
            raise ValueError('TRAIN/DEV projection differs from protocol receipt')
    if photos['train'] & photos['dev']:
        raise ValueError('TRAIN/DEV photographs overlap in any role')
    return result


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _arguments(packet, static):
    # Only exact runtime schema fields cross the decoder boundary. The frozen
    # encoder's autograd graph cannot become a trainable arm or loss input.
    return {key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in decoder_arguments(packet, static=static).items()}


def predict_logits(model, packet, *, static):
    """Complete dense prediction API: no row, labels, class or scorer argument."""
    model.eval()
    with torch.no_grad():
        logits = model(**_arguments(packet, static))
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite prediction')
    return logits.detach().clone()


def class_miou(count_rows):
    """Original-resolution class sum(I)/sum(U), then equal class mean (percent)."""
    totals = {}
    for category, intersection, union in count_rows:
        if any(type(v) is not int or v < 0 for v in (intersection, union)) or intersection > union:
            raise ValueError('DEV scorer must return nonnegative integer original-resolution I/U')
        pair = totals.setdefault(category, [0, 0])
        pair[0] += intersection
        pair[1] += union
    if not totals or any(union == 0 for _, union in totals.values()):
        raise ValueError('Each DEV class must have positive aggregate union')
    return 100*sum(i/u for i, u in totals.values())/len(totals), {
        str(c): {'I': i, 'U': u} for c, (i, u) in sorted(totals.items())}


def _atomic_save(path, state):
    temporary = path.with_suffix(path.suffix+'.tmp')
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_receipt(path, receipt):
    temporary = path.with_suffix('.json.tmp')
    try:
        with temporary.open('w') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _rng_state():
    state = dict(python=random.getstate(), torch=torch.get_rng_state())
    if torch.cuda.is_initialized():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


class MatchedResponseTrainer:
    """Two independent heads from identical theta, same finite episode order.

    AdamW is fixed at lr=3e-4, weight_decay=.01, one update per episode per arm.
    Both arms continue for the same epochs until BOTH satisfy the declared
    patience after min_epochs. Hitting max_epochs/time is PARTIAL_BUDGET, never
    a convergence claim. DEV loss and per-episode mean IoU cannot select heads.
    Caller owns actual DINO integration, original source refinement in scorer,
    and any separate held-out inference; this trainer has no such provider.
    """
    def __init__(self, initial_model, train_dev, protocol_receipt, *, policy=TrainingPolicy()):
        if not isinstance(initial_model, ReferenceResponseDecoder):
            raise TypeError('The actual ReferenceResponseDecoder is required')
        self.manifest = validated_training_manifest(train_dev, protocol_receipt)
        self.policy = policy
        self.models = {'static': deepcopy(initial_model), 'response': deepcopy(initial_model)}
        self.initial_hash = _state_hash(initial_model)
        if any(_state_hash(model) != self.initial_hash for model in self.models.values()):
            raise RuntimeError('Initial arm weights differ')
        for left, right in zip(self.models['static'].parameters(), self.models['response'].parameters()):
            if left.data_ptr() == right.data_ptr():
                raise RuntimeError('Arm parameters share storage')
        self.optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
                           for arm, model in self.models.items()}
        parameter = next(initial_model.parameters())
        self.contract = dict(manifest=self.manifest, policy=asdict(policy), initial_sha256=self.initial_hash,
                             final_channels=initial_model.final_channels,
                             middle_channels=initial_model.middle_channels,
                             query_chunk_size=initial_model.query_chunk_size, dtype=str(parameter.dtype),
                             device=str(parameter.device), torch_version=torch.__version__,
                             cuda_runtime=torch.version.cuda,
                             decoder_sha256=hashlib.sha256(Path(__file__).with_name('reference_response_decoder.py').read_bytes()).hexdigest(),
                             trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             runtime_sha256=hashlib.sha256(Path(__file__).with_name('reference_response_runtime.py').read_bytes()).hexdigest())

    def run(self, output_directory, *, feature_provider, training_label_provider, dev_scorer,
            resume_error=False, progress_callback=None):
        """Providers receive defensive metadata copies; never persisted features.

        training_label_provider(row) returns labels or (labels, boolean valid_mask).
        dev_scorer(frozen_logits, row) performs fixed full-mask original-resolution
        refinement/scoring and returns (I,U). No train loss is evaluated on DEV.
        A fresh directory is exclusive; only ERROR receipts authorize resumption.
        """
        directory = Path(output_directory)
        started = time.monotonic()
        elapsed_prior = 0.
        shuffle = random.Random(self.policy.seed)
        completed, history = 0, []
        best = {arm: dict(epoch=None, class_miou=None, stale_epochs=0) for arm in self.models}
        receipt_path = directory/'receipt.json'
        if resume_error:
            receipt = json.loads(receipt_path.read_text())
            if receipt['state'] != 'ERROR' or receipt['contract'] != self.contract:
                raise ValueError('Only an identical ERROR run may resume')
            # Only load trusted own checkpoints from this run directory.
            state = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
            if state['contract'] != self.contract:
                raise ValueError('Checkpoint contract differs')
            for arm in self.models:
                self.models[arm].load_state_dict(state['models'][arm])
                self.optimizers[arm].load_state_dict(state['optimizers'][arm])
            _restore_rng(state['rng'])
            shuffle.setstate(state['shuffle_state'])
            completed, history, best = state['epoch'], state['history'], state['best']
            elapsed_prior = receipt['elapsed_seconds']
        else:
            directory.mkdir(parents=True, exist_ok=False)
            random.seed(self.policy.seed)
            torch.manual_seed(self.policy.seed)

        def elapsed():
            return elapsed_prior+time.monotonic()-started

        def checkpoint():
            return dict(contract=self.contract, epoch=completed, history=history, best=best,
                        models={a: m.state_dict() for a, m in self.models.items()},
                        optimizers={a: o.state_dict() for a, o in self.optimizers.items()},
                        rng=_rng_state(), shuffle_state=shuffle.getstate())

        def budget_check():
            if elapsed() >= self.policy.time_cap_seconds:
                raise _BudgetExpired()

        status, reason = 'PARTIAL_BUDGET', 'max_epochs'
        try:
            if not resume_error:
                _atomic_save(directory/'last.pt', checkpoint())
            for epoch in range(completed+1, self.policy.max_epochs+1):
                budget_check()
                order = list(range(len(self.manifest['train'])))
                shuffle.shuffle(order)
                losses = {arm: [] for arm in self.models}
                for index in order:
                    budget_check()
                    row = self.manifest['train'][index]
                    packet = feature_provider(deepcopy(row))
                    targets = training_label_provider(deepcopy(row))
                    labels, valid = targets if isinstance(targets, tuple) else (targets, None)
                    for arm, model in self.models.items():
                        budget_check()
                        model.train()
                        optimizer = self.optimizers[arm]
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(**_arguments(packet, arm == 'static'))
                        loss = segmentation_loss(logits, labels, valid_mask=valid)['loss']
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite training loss')
                        loss.backward()
                        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                            raise FloatingPointError('Nonfinite training gradient')
                        optimizer.step()
                        if any(not torch.isfinite(p).all() for p in model.parameters()):
                            raise FloatingPointError('Nonfinite updated weights')
                        losses[arm].append(float(loss.detach()))
                        if progress_callback is not None:
                            progress_callback(dict(stage='TRAIN_UPDATE', epoch=epoch, arm=arm,
                                                   episode_id=row['id'], loss=losses[arm][-1]))
                counts = {arm: [] for arm in self.models}
                for row in self.manifest['dev']:
                    budget_check()
                    packet = feature_provider(deepcopy(row))
                    # Freeze BOTH arm predictions before opening this row's scorer.
                    frozen = {arm: predict_logits(model, packet, static=arm == 'static')
                              for arm, model in self.models.items()}
                    for arm in self.models:
                        budget_check()
                        i, u = dev_scorer(frozen[arm], deepcopy(row))
                        counts[arm].append((row['c'], i, u))
                        if progress_callback is not None:
                            progress_callback(dict(stage='DEV_SCORE', epoch=epoch, arm=arm,
                                                   episode_id=row['id'], I=i, U=u))
                budget_check()
                scores = {arm: class_miou(rows) for arm, rows in counts.items()}
                improved = []
                for arm, (score, _) in scores.items():
                    if best[arm]['class_miou'] is None or score > best[arm]['class_miou']:
                        best[arm] = dict(epoch=epoch, class_miou=score, stale_epochs=0)
                        improved.append(arm)
                    else:
                        best[arm]['stale_epochs'] += 1
                history.append(dict(epoch=epoch, episode_order=[self.manifest['train'][i]['id'] for i in order],
                                    train_loss={arm: sum(values)/len(values) for arm, values in losses.items()},
                                    dev={arm: dict(class_miou=score, class_counts=totals)
                                         for arm, (score, totals) in scores.items()}))
                completed = epoch
                state = checkpoint()
                _atomic_save(directory/'last.pt', state)
                for arm in improved:
                    _atomic_save(directory/(arm+'_best.pt'), state)
                if progress_callback is not None:
                    progress_callback(dict(stage='EPOCH_COMMITTED', epoch=epoch,
                                           train_loss=history[-1]['train_loss'],
                                           dev=history[-1]['dev']))
                if completed >= self.policy.min_epochs and all(b['stale_epochs'] >= self.policy.patience for b in best.values()):
                    status, reason = 'CONVERGED_PATIENCE', 'both_arms_patience'
                    break
        except _BudgetExpired:
            reason = 'time_cap_seconds'
        except Exception as exc:
            status, reason = 'ERROR', type(exc).__name__+': '+str(exc)
        # For incomplete epochs return ONLY the durable complete-epoch ledger.
        boundary = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
        # Incomplete-epoch work is not a selected model. Align the exposed
        # trainer state with its durable ledger as well as the receipt;
        # subsequent evaluation still loads each explicit best checkpoint.
        for arm in self.models:
            self.models[arm].load_state_dict(boundary['models'][arm])
            self.optimizers[arm].load_state_dict(boundary['optimizers'][arm])
        _restore_rng(boundary['rng'])
        receipt = dict(schema='reference_response_training_v1', state=status, reason=reason,
                       completed_epochs=boundary['epoch'], history=boundary['history'], best=boundary['best'],
                       elapsed_seconds=elapsed(), contract=self.contract,
                       optimizer=dict(name='AdamW', learning_rate=.0003, weight_decay=.01),
                       selection='original_resolution_class_sum_I_over_sum_U',
                       initial_arm_weights_identical=True, arm_parameters_independent=True,
                       feature_cache_written=False, real_task_gain_measured=False)
        _write_receipt(receipt_path, receipt)
        return receipt


class _BudgetExpired(Exception):
    pass
