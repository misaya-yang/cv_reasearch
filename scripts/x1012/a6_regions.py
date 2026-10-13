"""A6 region oracles and supervised diagnostics, with no new encoding.

CLI stages: produce -> upper -> supervised. Default cohort is the complete P0
explore manifest; seed150 is only an initial diagnostic/speed cohort. All natural
descriptors are sealed before query labels are opened. This script never opens
the confirm manifest and never turns supervised diagnostics into a C1 method.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

import numpy as np
import torch
import lab
import plan_common as pc
import plan_report as pr
import a6_helpers as ah
from p0_manifest import raw_metadata

BASE = lab.ASSETS / 'a/x1012_A6'
FIELDS = lab.ASSETS / 'a/x1012_P0/fields'
EXPLORE = lab.ASSETS / 'a/x1012_P0/explore.json'
SEED = 20261012


class BaselinePending(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def output(cohort):
    return BASE if cohort == 'explore' else BASE / 'seed150'


def rows(cohort):
    result = pc.read(EXPLORE) if cohort == 'explore' else pc.seed_rows(150)
    if any(b.get('split') == 'confirm' for b in result):
        raise RuntimeError('A6 accepts only explore rows, never confirm')
    keys = [pc.key(b) for b in result]
    if not result or len(keys) != len(set(keys)):
        raise ValueError('Empty cohort or duplicate episode keys')
    return result


def photo(binding, role='query'):
    return str((lab.ASSETS / binding['row'][role + '_path']).resolve())


@lru_cache(None)
def verified_file(path, expected=None):
    actual = pc.sha(path)
    if expected is not None and actual != expected:
        raise RuntimeError('File SHA256 changed: ' + str(path))
    return actual


@lru_cache(1)
def algorithm_signature():
    return digest({'a6': pc.sha(Path(__file__)), 'helpers': pc.sha(Path(ah.__file__)),
                   'lab': pc.sha(Path(lab.__file__)), 'basis': pc.sha(lab.BASIS),
                   'device': 'cpu', 'iters': 6, 'kmeans': ah.KMEANS, 'taus': ah.TAUS,
                   'spatial': 'four-neighbor average cosine; fixed K64'})


def raw_identity(payload):
    metadata = raw_metadata(str(payload))
    return digest([algorithm_signature(), metadata['input_sha256'], metadata['tensor_sha256']])


def binding_identity(binding):
    row = binding['row']
    return digest({'index': pc.key(binding), 'dataset': binding['dataset'], 'episode_id': row['episode_id'],
                   'fold': str(row['fold']), 'class': str(row['global_class_id']),
                   'raw': {role: raw_identity(binding['fit'][role]['payload_path'])
                           for role in ('reference_raw', 'query_raw_identity')},
                   'views': {key: row.get(key) for key in ('reference_path', 'reference_crop',
                              'reference_mask_path', 'reference_mask_hash', 'query_path', 'query_crop',
                              'query_mask_path', 'query_mask_hash', 'query_ignore_mask_path')}})


@lru_cache(8)
def debiased_raw(payload):
    metadata = raw_metadata(str(payload))
    verified_file(str(payload), metadata['payload_sha256'])
    with np.load(payload, allow_pickle=False) as archive:
        raw = archive['O/24'].copy()
    if raw.shape != (4096, 1024) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('Expected existing raw O/24 float32 tokens')
    with torch.inference_mode():
        return lab.debias(torch.from_numpy(raw)).cpu()


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


@lru_cache(16)
def load_regions(path):
    with np.load(path, allow_pickle=False) as archive:
        return {g: archive[g].astype(np.uint16) for g in ah.GEOMETRIES}


def query_regions(payload):
    identity = raw_identity(payload)
    path = BASE / 'query_regions' / (identity + '.npz')
    receipt = path.with_suffix('.json')
    if receipt.exists():
        metadata = pc.read(receipt)
        verified_file(str(path), metadata['sha256'])
        return path, metadata
    start = time.monotonic()
    q = debiased_raw(str(payload))
    values = {}
    reference_cached = BASE / 'reference_regions' / (identity + '.npz')
    for k in ah.KMEANS:
        if k == 64 and reference_cached.with_suffix('.json').exists():
            reference_metadata = pc.read(reference_cached.with_suffix('.json'))
            verified_file(str(reference_cached), reference_metadata['sha256'])
            with np.load(reference_cached, allow_pickle=False) as archive:
                values['kmeans64'] = archive['labels'].copy()
        else:
            centers = lab.kmeans(q, k, iters=6)
            values[f'kmeans{k}'] = ah.compact((q @ centers.T).argmax(1).numpy())
    tree, other = ah.average_tree_and_spatial(q)
    values.update(other)
    values['average_tree'] = tree
    save_npz(path, values)
    metadata = dict(path=str(path), sha256=pc.sha(path), raw=raw_metadata(str(payload)),
                    algorithm_signature=algorithm_signature(), seconds=time.monotonic() - start,
                    region_counts={g: int(values[g].max()) + 1 for g in ah.GEOMETRIES},
                    query_GT_reads=0, new_encoder_calls=0)
    pc.write(receipt, metadata)
    return path, metadata


def reference_regions(payload):
    identity = raw_identity(payload)
    query_path = BASE / 'query_regions' / (identity + '.npz')
    if query_path.with_suffix('.json').exists():
        metadata = pc.read(query_path.with_suffix('.json'))
        verified_file(str(query_path), metadata['sha256'])
        return load_regions(str(query_path))['kmeans64'], metadata
    path = BASE / 'reference_regions' / (identity + '.npz')
    receipt = path.with_suffix('.json')
    if not receipt.exists():
        r = debiased_raw(str(payload))
        centers = lab.kmeans(r, 64, iters=6)
        labels = ah.compact((r @ centers.T).argmax(1).numpy())
        save_npz(path, {'labels': labels})
        pc.write(receipt, dict(path=str(path), sha256=pc.sha(path), raw=raw_metadata(str(payload)),
                               algorithm_signature=algorithm_signature(), query_GT_reads=0, new_encoder_calls=0))
    metadata = pc.read(receipt)
    verified_file(str(path), metadata['sha256'])
    with np.load(path, allow_pickle=False) as archive:
        return archive['labels'].copy(), metadata


def standard_fields(binding, kernel, sim, shift, coverage, query_labels):
    receipt = FIELDS / 'records' / (pc.key(binding) + '.json')
    if receipt.exists():
        source = pc.read(receipt)
        verified_file(source['path'], source['sha256'])
        # Whitelist only fields generated by P0's GT-free inference.
        with np.load(source['path'], allow_pickle=False) as archive:
            c = archive['c32'].astype(np.float32)
            log_e = archive['log_e'].astype(np.float32)
            forward = archive['nn_query_reference'].astype(np.int64)
            backward = archive['nn_reference_query'].astype(np.int64)
            baseline = archive['mask/IND0'].copy()
        return c, log_e, forward, backward, baseline, dict(kind='P0_fields', receipt=str(receipt), **source)
    # Existing raw features only; no encoder fallback or query labels.
    df = kernel @ torch.as_tensor(coverage)
    explanation = kernel.sum(1)
    c = (df / explanation.clamp_min(1e-30)).numpy()
    log_e = (explanation.clamp_min(1e-30).log() + shift).numpy().astype(np.float16).astype(np.float32)
    forward, backward = sim.argmax(1).numpy(), sim.argmax(0).numpy()
    field = ah.region_means(c, query_labels)[query_labels]
    baseline = pc.pack(pc.readout(field))
    return c, log_e, forward, backward, baseline, dict(kind='existing_raw_CPU_fallback',
                log_e_storage='FP16 then FP32, matching P0 standard field storage', new_encoder_calls=0)


def infer(binding):
    name = pc.key(binding)
    receipt = BASE / 'inference_records' / (name + '.json')
    path = BASE / 'descriptors' / (name + '.npz')
    identity = binding_identity(binding)
    if receipt.exists():
        metadata = pc.read(receipt)
        if metadata['binding_identity'] != identity or metadata['algorithm_signature'] != algorithm_signature():
            raise RuntimeError('Preserve incompatible A6 descriptor cache: ' + name)
        verified_file(metadata['path'], metadata['sha256'])
        verified_file(metadata['query_regions']['path'], metadata['query_regions']['sha256'])
        return metadata
    start = time.monotonic()
    with pc.inference_only(binding):
        qpath, qreceipt = query_regions(binding['fit']['query_raw_identity']['payload_path'])
        labels = load_regions(str(qpath))
        reference_labels, rreceipt = reference_regions(binding['fit']['reference_raw']['payload_path'])
        q = debiased_raw(str(binding['fit']['query_raw_identity']['payload_path']))
        r = debiased_raw(str(binding['fit']['reference_raw']['payload_path']))
        coverage = lab.coverage64(lab.mask1024(binding['row']['reference_mask_path'])).flatten().numpy()
        mu = .5 * (r.mean(0) + q.mean(0))
        sim = lab.unit(q - mu) @ lab.unit(r - mu).T
        shift = float(sim.max()) / .05
        kernel = torch.exp((sim - sim.max()) / .05)
        c, log_e, forward, backward, baseline, source = standard_fields(
            binding, kernel, sim, shift, coverage, labels['kmeans64'])
        mutual = backward[forward] == np.arange(4096)
        fg_mutual = mutual * coverage[forward]
        token_f, token_b = ah.reference_kernels(kernel, coverage, reference_labels)
        arrays = {'mask/IND0': baseline}
        for geometry in ah.GEOMETRIES:
            arrays['features/' + geometry] = ah.describe(labels[geometry], c, log_e, mutual, fg_mutual,
                                                         token_f, token_b, shift)
        save_npz(path, arrays)
    metadata = dict(index=name, dataset=binding['dataset'], episode_id=binding['row']['episode_id'],
                    path=str(path), sha256=pc.sha(path), binding_identity=identity,
                    algorithm_signature=algorithm_signature(), query_regions=qreceipt,
                    reference_regions=rreceipt, field_source=source,
                    seconds=time.monotonic() - start, query_GT_reads=0, new_encoder_calls=0)
    pc.write(receipt, metadata)
    lab.free()
    return metadata


def freeze(cohort):
    out = output(cohort)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'config.json'
    if not path.exists():
        tasks = rows(cohort)
        sources = (Path(__file__), Path(ah.__file__), Path(pc.__file__), Path(pr.__file__), Path(lab.__file__))
        frozen = out / 'frozen'
        frozen.mkdir(exist_ok=True)
        for source in sources:
            shutil.copyfile(source, frozen / source.name)
        pc.write(out / 'manifest.json', tasks)
        pc.write(path, dict(state='FROZEN', cohort=cohort, n=len(tasks), complete_plan_cohort=cohort == 'explore',
                    domains=dict(Counter(b['dataset'] for b in tasks)), threads=4, device='CPU',
                    geometries=ah.GEOMETRIES, features=ah.FEATURES, temperature=.05,
                    region_readout='constant binary region labels, nearest to CLI1024; no bilinear/minmax',
                    cluster_inputs='unit positionally debiased O/24; independent of reference',
                    reference_regions='kmeans64 iters6; soft mask-weighted exact token-pair mean kernels',
                    split='SHA256(seed20261012, original query photograph path) parity; original crops grouped',
                    logistic='train-only standardization; soft BCE; valid region area / episode valid area; L2 1e-4',
                    source_sha256={str(p): pc.sha(p) for p in sources}, algorithm_signature=algorithm_signature(),
                    manifest_sha256=pc.sha(out / 'manifest.json'), query_GT_in_inference=False, new_encoder_calls=0))
    config = pc.read(path)
    if config['cohort'] != cohort or config['algorithm_signature'] != algorithm_signature():
        raise RuntimeError('A6 frozen configuration changed')
    for source, expected in config['source_sha256'].items():
        verified_file(source, expected)
    verified_file(str(out / 'manifest.json'), config['manifest_sha256'])
    return out, pc.read(out / 'manifest.json'), config


def produce(cohort):
    pc.setup('cpu')
    out, tasks, config = freeze(cohort)
    seal_path = out / 'descriptors_sealed.json'
    if seal_path.exists():
        print(dict(state='A6_DESCRIPTORS_ALREADY_SEALED', cohort=cohort, n=len(tasks)), flush=True)
        return
    # Same-query episodes adjacent: projection/clustering and raw tokens are reused.
    ordered = sorted(tasks, key=lambda b: (raw_identity(b['fit']['query_raw_identity']['payload_path']), pc.key(b)))
    start, last = time.monotonic(), time.monotonic()
    receipts = []
    for i, binding in enumerate(ordered, 1):
        receipts.append(infer(binding))
        if i <= 3 or i % 25 == 0 or time.monotonic() - last > 45:
            print(dict(stage='A6_produce', cohort=cohort, completed=i, n=len(tasks),
                       seconds=round(time.monotonic() - start, 1)), flush=True)
            last = time.monotonic()
    pc.write(seal_path, dict(state='ALL_GT_FREE_DESCRIPTORS_SEALED', cohort=cohort, n=len(tasks),
              seconds=time.monotonic() - start, records=receipts, manifest_sha256=config['manifest_sha256'],
              config_sha256=pc.sha(out / 'config.json'), unique_query_inputs=len({r['query_regions']['path'] for r in receipts}),
              query_GT_reads=0, new_encoder_calls=0, confirm_scored=False))
    print(dict(stage='A6_produce', state='SEALED', cohort=cohort, n=len(tasks)), flush=True)


def sealed(cohort):
    pc.setup('cpu')
    out, tasks, config = freeze(cohort)
    seal = pc.read(out / 'descriptors_sealed.json')
    if seal['state'] != 'ALL_GT_FREE_DESCRIPTORS_SEALED' or seal['n'] != len(tasks):
        raise RuntimeError('All descriptors must be sealed before any query labels')
    verified_file(str(out / 'config.json'), seal['config_sha256'])
    by_key = {r['index']: r for r in seal['records']}
    if set(by_key) != {pc.key(b) for b in tasks}:
        raise RuntimeError('Sealed descriptor cohort differs')
    for binding in tasks:
        receipt = by_key[pc.key(binding)]
        if receipt['binding_identity'] != binding_identity(binding):
            raise RuntimeError('Descriptor binding identity changed')
        verified_file(receipt['path'], receipt['sha256'])
        verified_file(receipt['query_regions']['path'], receipt['query_regions']['sha256'])
    return out, tasks, by_key


def baseline_bindings(tasks, out):
    # P0-2 may add B after descriptors were frozen. Read only the explore metadata.
    fresh = {pc.key(b): b for b in pc.read(EXPLORE)} if EXPLORE.exists() else {}
    result, missing = {}, []
    for original in tasks:
        key = pc.key(original)
        candidate = fresh.get(key, original)
        if binding_identity(candidate) != binding_identity(original):
            raise RuntimeError('P0 altered an episode identity rather than only its baseline')
        arm = candidate.get('old_arms', {}).get('B')
        if not arm:
            receipt = lab.ASSETS / 'a/x1012_P0/foris/new/records' / (key + '.json')
            if receipt.exists():
                record = pc.read(receipt)
                arm = dict(path=record['prediction_path'], sha256=record['prediction_sha256'], cli_key='cli/foris.crf')
        if not arm or not Path(arm['path']).exists():
            missing.append(key)
        else:
            verified_file(arm['path'], arm.get('sha256'))
            result[key] = arm
    if missing:
        state = dict(state='WAITING_P0_2_FORIS_BASELINES', missing_n=len(missing),
                     missing_keys=missing, descriptor_cohort_n=len(tasks), query_GT_reads=0, confirm_scored=False)
        pc.write(out / 'waiting_baselines.json', state)
        raise BaselinePending(f"A6 descriptors are sealed; {len(missing)} FoRIS baselines await P0-2")
    return result


def base_record(binding, metrics):
    return dict(index=pc.key(binding), episode_id=binding['row']['episode_id'], dataset=binding['dataset'],
                fold=binding['row']['fold'], class_id=binding['row']['global_class_id'],
                query_photo=photo(binding), metrics=metrics)


def targets(cohort, out, tasks, descriptors):
    seal_path = out / 'targets_sealed.json'
    if seal_path.exists():
        seal = pc.read(seal_path)
        for r in seal['records']:
            verified_file(r['path'], r['sha256'])
        return seal['records']
    baselines = baseline_bindings(tasks, out)  # No query GT until every B is available.
    start = time.monotonic()
    result = []
    for i, binding in enumerate(tasks, 1):
        key = pc.key(binding)
        receipt_path = out / 'target_records' / (key + '.json')
        if receipt_path.exists():
            receipt = pc.read(receipt_path)
            verified_file(receipt['path'], receipt['sha256'])
            result.append(receipt)
            continue
        gt, valid = pc.truth(binding)
        token_gt, token_valid = ah.token_pixel_counts(gt, valid)
        regions = load_regions(descriptors[key]['query_regions']['path'])
        values = {}
        for geometry in ah.GEOMETRIES:
            n, tp = ah.region_pixel_counts(regions[geometry], token_gt, token_valid)
            values['n/' + geometry], values['tp/' + geometry] = n, tp
        path = out / 'targets' / (key + '.npz')
        save_npz(path, values)
        arm = baselines[key]
        with np.load(arm['path'], allow_pickle=False) as archive:
            foris = pc.unpack(archive[arm['cli_key']])
        with np.load(descriptors[key]['path'], allow_pickle=False) as archive:
            baseline = pc.unpack(archive['mask/IND0'])
        metrics = {'FoRIS': pr.metric(foris, gt, valid), 'IND0': pr.metric(baseline, gt, valid)}
        receipt = dict(index=key, path=str(path), sha256=pc.sha(path),
                        descriptor_sha256=descriptors[key]['sha256'], GT=int(gt.sum()), valid=int(valid.sum()),
                        baseline_path=arm['path'], baseline_sha256=verified_file(arm['path'], arm.get('sha256')),
                        record=base_record(binding, metrics), query_GT_role='scoring/training diagnosis after descriptor seal')
        pc.write(receipt_path, receipt)
        result.append(receipt)
        if i <= 3 or i % 100 == 0:
            print(dict(stage='A6_targets', completed=i, n=len(tasks), seconds=round(time.monotonic() - start, 1)), flush=True)
    pc.write(seal_path, dict(state='DIAGNOSTIC_REGION_TARGETS_SEALED', cohort=cohort, n=len(tasks),
              descriptor_seal_sha256=pc.sha(out / 'descriptors_sealed.json'), records=result,
              new_encoder_calls=0, confirm_scored=False))
    return result


def report_name(cohort, suffix):
    if cohort == 'explore' and suffix == 'holdout':
        return 'A6'
    return ('A6-' if cohort == 'explore' else 'A6-seed150-') + suffix


def upper(cohort):
    out, tasks, descriptors = sealed(cohort)
    if (out / 'upper_gate.json').exists():
        print(dict(state='A6_UPPER_ALREADY_COMPLETE', cohort=cohort), flush=True)
        return
    target_records = targets(cohort, out, tasks, descriptors)
    records = [dict(r['record'], metrics=dict(r['record']['metrics'])) for r in target_records]
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[(r['dataset'], str(r['fold']), str(r['class_id']))].append(i)
    arms = ['FoRIS', 'IND0']
    start = time.monotonic()
    for geometry in ah.GEOMETRIES:
        majority, episode, pooled = ('GT_majority_' + geometry, 'oracle_episode_' + geometry, 'oracle_class_' + geometry)
        arms.extend((majority, episode, pooled))
        counts = []
        for i, receipt in enumerate(target_records):
            with np.load(receipt['path'], allow_pickle=False) as archive:
                n, tp = archive['n/' + geometry], archive['tp/' + geometry]
            if n.sum() != receipt['valid'] or tp.sum() != receipt['GT'] or (tp > n).any():
                raise RuntimeError('Regional geometry/count mismatch')
            counts.append((n, tp))
            gt_area, valid_area = receipt['GT'], receipt['valid']
            keep = tp / np.maximum(n, 1) > .5
            records[i]['metrics'][majority] = ah.metric_counts(keep, n, tp, gt_area, valid_area)
            keep = ah.selection(tp, n - tp, gt_area)
            records[i]['metrics'][episode] = ah.metric_counts(keep, n, tp, gt_area, valid_area)
        for indices in groups.values():
            tp = np.concatenate([counts[i][1] for i in indices])
            fp = np.concatenate([counts[i][0] - counts[i][1] for i in indices])
            keep = ah.selection(tp, fp, sum(target_records[i]['GT'] for i in indices))
            offset = 0
            for i in indices:
                n, inter = counts[i]
                selected = keep[offset:offset + len(n)]
                records[i]['metrics'][pooled] = ah.metric_counts(selected, n, inter,
                                       target_records[i]['GT'], target_records[i]['valid'])
                offset += len(n)
        print(dict(stage='A6_upper', geometry=geometry, seconds=round(time.monotonic() - start, 1)), flush=True)
    notes = ['所有区域/描述子先在GT-free inference_only内保存并全队列封存，之后才读取query GT。',
             'GT多数仅为像素错误最小化诊断，不称IoU上界。精确oracle按TP/(TP+FP)排序扫描前缀，分母保留全部GT。',
             'oracle_episode逐例最大化IoU；oracle_class联合同域(fold,global_class)所有区域最大化正式pooled-I/U。',
             '每个区域二值决策nearest到1024，训练读出与上界使用完全相同的区域几何。',
             'kmeans K16/32/64/128/256迭代6次；三个tau共享完整cosine/average树，无PCA；空间四邻接固定K64。',
             'oracle区间是固定GT选择掩码的描述性重采样，不代表无监督方法的泛化收益。',
             '完整explore为计划监督范围；seed150仅是首轮诊断，不能替代完整计划结论。']
    report = pr.report(report_name(cohort, 'upper'), out / 'upper_report', records, arms, notes,
                       extra={'cohort': cohort, 'complete_plan_cohort': cohort == 'explore'})
    small = {}
    for geometry in ah.GEOMETRIES:
        table = report['summary'].get('lvis', {}).get('oracle_episode_' + geometry)
        small[geometry] = table['size_bands']['<2%'] if table else {'n': 0, 'episode_mean_iou': None}
    available = [v['episode_mean_iou'] for v in small.values() if v['episode_mean_iou'] is not None]
    best = max(available) if available else None
    gate = dict(state='UPPER_COMPLETE', cohort=cohort, n=len(tasks), complete_plan_cohort=cohort == 'explore',
                lvis_small_target_episode_oracle=small, best_lvis_small_oracle=best,
                region_geometry_bottleneck=best < 70 if best is not None else None,
                supervised_diagnostics_enabled=best >= 70 if best is not None else True,
                gate='LVIS <2% target: exact per-episode union-oracle mean; GT-majority is not the gate',
                new_encoder_calls=0, confirm_scored=False)
    pc.write(out / 'upper_gate.json', gate)
    pc.write(out / 'region_gate.json', gate)
    print(gate, flush=True)


def split_manifest(tasks):
    train = np.array([ah.photo_train_half(photo(b), SEED) for b in tasks], dtype=bool)
    if train.all() or not train.any():
        raise ValueError('Photo hash split has an empty half')
    train_queries = {photo(b) for b, yes in zip(tasks, train) if yes}
    test_queries = {photo(b) for b, yes in zip(tasks, train) if not yes}
    train_refs = {photo(b, 'reference') for b, yes in zip(tasks, train) if yes}
    test_refs = {photo(b, 'reference') for b, yes in zip(tasks, train) if not yes}
    domains = tuple(d for d in lab.ALL_DOMAINS if any(b['dataset'] == d for b in tasks))
    detail = dict(seed=SEED, unit='original query photograph, all classes/crops/repeated episodes together',
                  train_indices=[pc.key(b) for b, yes in zip(tasks, train) if yes],
                  test_indices=[pc.key(b) for b, yes in zip(tasks, train) if not yes],
                  train_photo_n=len(train_queries), test_photo_n=len(test_queries), query_photo_overlap=0,
                  reference_photo_overlap=len(train_refs & test_refs),
                  cross_role_photo_overlap=len((train_refs | train_queries) & (test_refs | test_queries)),
                  independence='query-photo holdout; reference photos may be shared',
                  domains={d: {'train': int(sum(bool(yes) and b['dataset'] == d for b, yes in zip(tasks, train))),
                               'test': int(sum(not bool(yes) and b['dataset'] == d for b, yes in zip(tasks, train)))} for d in domains})
    if train_queries & test_queries:
        raise RuntimeError('A query photograph crossed the split')
    if any(not v['train'] or not v['test'] for v in detail['domains'].values()):
        raise ValueError('A domain has an empty photo-hash split; do not tune the hash using outcomes')
    return train, domains, detail


def geometry_training_data(geometry, tasks, descriptors, target_records, train):
    features, coverage, weights, lengths = [], [], [], []
    for binding, target, is_train in zip(tasks, target_records, train):
        with np.load(descriptors[pc.key(binding)]['path'], allow_pickle=False) as archive:
            x = archive['features/' + geometry]
        features.append(x)
        if is_train:
            with np.load(target['path'], allow_pickle=False) as archive:
                n, tp = archive['n/' + geometry], archive['tp/' + geometry]
            if len(x) != len(n):
                raise RuntimeError('Descriptor/target region mismatch')
            coverage.append((tp / np.maximum(n, 1)).astype(np.float32))
            weights.append((n / max(int(n.sum()), 1)).astype(np.float32))
        else:
            # Prediction needs only X and region lengths. Heldout label values
            # are not loaded here, even though the prior upper stage knows GT.
            coverage.append(np.zeros(len(x), np.float32))
            weights.append(np.zeros(len(x), np.float32))
        lengths.append(len(x))
    return (np.concatenate(features), np.concatenate(coverage), np.concatenate(weights),
            np.r_[0, np.cumsum(lengths)], np.asarray(lengths))


def supervised(cohort):
    out, tasks, descriptors = sealed(cohort)
    if (out / 'completion.json').exists():
        print(pc.read(out / 'completion.json'), flush=True)
        return
    gate = pc.read(out / 'upper_gate.json')
    if not gate['supervised_diagnostics_enabled']:
        done = dict(gate, state='COMPLETE_GEOMETRY_GATE_CLOSED',
                    reason='Even exact region IoU oracle below 70 on LVIS small targets; supervised head not run')
        pc.write(out / 'completion.json', done)
        report_path = lab.ROOT / 'docs/research/x1012_tasks' / (report_name(cohort, 'holdout') + '.md')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        upper_path = report_path.with_name(report_name(cohort, 'upper') + '.md')
        report_path.write_text(f'# A6 区域诊断\n\n队列：{cohort}，n={len(tasks)}。'
                  f'LVIS小目标最好的精确逐例区域IoU oracle为{gate["best_lvis_small_oracle"]:.3f}，低于70。'
                  '按计划先改区域几何，停止监督头拟合；此判断使用IoU oracle，不使用GT多数代替上界。\n\n'
                  f'完整按域、大小、错误率和配对bootstrap表：[区域上界报告]({upper_path})。'
                  '没有读取confirm或启动新编码。\n')
        return
    target_records = pc.read(out / 'targets_sealed.json')['records']
    if [r['index'] for r in target_records] != [pc.key(b) for b in tasks]:
        raise RuntimeError('Target ordering differs from manifest')
    for r in target_records:
        verified_file(r['path'], r['sha256'])
    train, domains, split = split_manifest(tasks)
    pc.write(out / 'photo_split.json', split)
    dataset = np.array([b['dataset'] for b in tasks])
    test_indices = np.flatnonzero(~train)
    prediction_seal = out / 'supervised_predictions_sealed.json'
    if not prediction_seal.exists():
        predictions = {int(i): {} for i in test_indices}
        model_receipts = []
        start = time.monotonic()
        for geometry in ah.GEOMETRIES:
            x, y, weight, offsets, lengths = geometry_training_data(geometry, tasks, descriptors, target_records, train)
            region_train = np.repeat(train, lengths)
            model_path = out / 'models' / (geometry + '.json')
            split_sha = pc.sha(out / 'photo_split.json')
            descriptor_sha = pc.sha(out / 'descriptors_sealed.json')
            target_sha = pc.sha(out / 'targets_sealed.json')
            if model_path.exists():
                saved = pc.read(model_path)
                if (saved['split_sha256'], saved['descriptor_seal_sha256'], saved['target_seal_sha256']) != (split_sha, descriptor_sha, target_sha):
                    raise RuntimeError('Preserve a model trained from different sealed inputs')
                models = saved['models']
            else:
                models = {'LODO': {}}

            def checkpoint_models():
                pc.write(model_path, dict(geometry=geometry, features=ah.FEATURES, models=models,
                          split_sha256=split_sha, descriptor_seal_sha256=descriptor_sha, target_seal_sha256=target_sha,
                          inference_inputs='sealed regional descriptors only',
                          query_GT_role='training half labels only; no heldout labels in fitting/standardization',
                          new_encoder_calls=0, deployment_method=False, confirm_scored=False))

            if 'holdout' not in models:
                models['holdout'] = ah.fit_logistic(x[region_train], y[region_train], weight[region_train])
                checkpoint_models()
            main = models['holdout']
            for i in test_indices:
                predictions[int(i)]['holdout/' + geometry] = ah.predict(main, x[offsets[i]:offsets[i + 1]])
            print(dict(stage='A6_fit', geometry=geometry, fit='A_to_B', seconds=round(time.monotonic() - start, 1)), flush=True)
            for domain in domains:
                selected = train & (dataset != domain)
                region_selected = np.repeat(selected, lengths)
                if domain not in models['LODO']:
                    model = ah.fit_logistic(x[region_selected], y[region_selected], weight[region_selected])
                    model['training_domains'] = [d for d in domains if d != domain]
                    model['heldout_domain'] = domain
                    models['LODO'][domain] = model
                    checkpoint_models()
                model = models['LODO'][domain]
                for i in np.flatnonzero(~train & (dataset == domain)):
                    predictions[int(i)]['LODO/' + geometry] = ah.predict(model, x[offsets[i]:offsets[i + 1]])
                print(dict(stage='A6_fit', geometry=geometry, fit='LODO', domain=domain,
                           seconds=round(time.monotonic() - start, 1)), flush=True)
            model_receipts.append(dict(path=str(model_path), sha256=pc.sha(model_path)))
            del x, y, weight, region_train, models
        receipts = []
        for i in test_indices:
            binding = tasks[i]
            with pc.inference_only(binding):
                path = out / 'supervised_predictions' / (pc.key(binding) + '.npz')
                save_npz(path, predictions[int(i)])
            receipts.append(dict(index=pc.key(binding), path=str(path), sha256=pc.sha(path), query_GT_in_inference=False))
        pc.write(prediction_seal, dict(state='HELDOUT_REGIONAL_PREDICTIONS_SEALED', n=len(receipts),
                  records=receipts, models=model_receipts, split_sha256=pc.sha(out / 'photo_split.json'),
                  descriptor_seal_sha256=pc.sha(out / 'descriptors_sealed.json'),
                  target_seal_sha256=pc.sha(out / 'targets_sealed.json'),
                  new_encoder_calls=0, confirm_scored=False, deployment_method=False))
    seal = pc.read(prediction_seal)
    verified_file(str(out / 'photo_split.json'), seal['split_sha256'])
    prediction_by_key = {r['index']: r for r in seal['records']}
    records = {'holdout': [], 'LODO': []}
    arms = ('FoRIS', 'IND0') + tuple('supervised_' + g for g in ah.GEOMETRIES)
    for i in test_indices:
        target = target_records[i]
        prediction = prediction_by_key[pc.key(tasks[i])]
        verified_file(prediction['path'], prediction['sha256'])
        with np.load(prediction['path'], allow_pickle=False) as p, np.load(target['path'], allow_pickle=False) as t:
            for mode in records:
                record = dict(target['record'], metrics=dict(target['record']['metrics']))
                for geometry in ah.GEOMETRIES:
                    record['metrics']['supervised_' + geometry] = ah.metric_counts(
                        p[mode + '/' + geometry], t['n/' + geometry], t['tp/' + geometry], target['GT'], target['valid'])
                records[mode].append(record)
    notes = ['监督区域头仅作信息诊断，使用训练query标签；不是无监督方法，不进入C1。',
             '完整explore按原query照片SHA256哈希切半，重复episode、类别、crop均跟随照片；参考照片可共享，重叠数在photo_split.json。',
             '标准化只拟合训练集；soft BCE按区域有效面积/该episode有效面积加权，固定L2=1e-4及sigmoid>0.5。',
             '主结果为所有域A半拟合→各域B半；LODO为其余域A半拟合→留出域同一B半，不用该域A半或B半拟合/选参数。',
             '每个固定区域几何独立报告，未按测试例或测试域GT选择K/tau；nearest硬区域读出与oracle一致。',
             '小样本seed150只作探针；只有完整explore报告覆盖本任务计划范围。']
    reports = {}
    for mode in ('holdout', 'LODO'):
        reports[mode] = pr.report(report_name(cohort, mode), out / (mode.lower() + '_report'),
                                  records[mode], arms, notes,
                                  extra=dict(cohort=cohort, complete_plan_cohort=cohort == 'explore', split=split))
    comparisons = {}
    for geometry in ah.GEOMETRIES:
        name = 'supervised_' + geometry
        gains = {mode: {d: reports[mode]['summary'][d][name]['delta_to_foris_pp'] for d in domains} for mode in reports}
        means = {mode: float(np.mean(list(value.values()))) for mode, value in gains.items()}
        comparisons[geometry] = dict(gain_pp=gains, ten_domain_mean_gain_pp=means,
                  mean_plus2_in_holdout_and_LODO=all(value >= 2 for value in means.values()),
                  every_domain_plus2_in_holdout_and_LODO=all(value >= 2 for mode in gains.values() for value in mode.values()),
                  all_domains_nondecreasing_in_holdout_and_LODO=all(value >= 0 for mode in gains.values() for value in mode.values()))
    final = dict(gate, state='COMPLETE_SUPERVISED_DIAGNOSTIC', supervised=comparisons,
                 mean_plus2_geometries=[g for g, v in comparisons.items() if v['mean_plus2_in_holdout_and_LODO']],
                 every_domain_plus2_geometries=[g for g, v in comparisons.items() if v['every_domain_plus2_in_holdout_and_LODO']],
                 all_domain_nondecreasing_geometries=[g for g, v in comparisons.items() if v['all_domains_nondecreasing_in_holdout_and_LODO']],
                 B4_role='regional supervised diagnostic only; no fitted head is a frozen C1 candidate',
                 test_episode_n=len(test_indices), train_episode_n=int(train.sum()),
                 confirm_scored=False, deployment_method=False, new_encoder_calls=0)
    pc.write(out / 'region_gate.json', final)
    pc.write(out / 'completion.json', final)
    print(dict(state=final['state'], cohort=cohort, n=len(tasks),
               mean_plus2_geometries=final['mean_plus2_geometries'], complete_plan_cohort=cohort == 'explore'), flush=True)


def self_check():
    """Small mathematical/geometry fixtures only; no project data or GT opened."""
    from itertools import product
    rng = np.random.default_rng(0)
    for unused in range(12):
        tp = rng.integers(0, 8, 6)
        fp = rng.integers(0, 12, 6)
        gt = int(tp.sum() + 3)
        keep = ah.selection(tp, fp, gt)
        found = tp[keep].sum() / (gt + fp[keep].sum())
        expected = max(sum(tp[i] * b[i] for i in range(6)) / (gt + sum(fp[i] * b[i] for i in range(6)))
                       for b in product((0, 1), repeat=6))
        assert abs(found - expected) < 1e-12
    assert ah.selection(np.array([60, 40]), np.array([0, 60]), 100).all()
    gt = rng.random((1024, 1024)) > .8
    valid = rng.random((1024, 1024)) > .1
    gt &= valid
    labels = rng.integers(0, 7, 4096).astype(np.uint16)
    token_gt, token_valid = ah.token_pixel_counts(gt, valid)
    n, tp = ah.region_pixel_counts(labels, token_gt, token_valid)
    keep = np.array([True, False, True, False, False, True, False])
    pixel = np.repeat(np.repeat(keep[labels].reshape(64, 64), 16, 0), 16, 1)
    assert ah.metric_counts(keep, n, tp, int(gt.sum()), int(valid.sum())) == pr.metric(pixel, gt, valid)
    kernel = torch.tensor([[1., 2., 3., 4.], [2., 4., 6., 8.]])
    foreground, background = ah.reference_kernels(kernel, np.array([1., .5, 0., 0.]), np.array([0, 0, 1, 1], np.uint16))
    assert np.allclose(foreground[:, 0], np.array([2. / 1.5, 4. / 1.5]))
    assert np.allclose(background[:, 1], np.array([3.5, 7.]))
    assert ah.photo_train_half('/photo/a') == ah.photo_train_half('/photo/a')
    x = np.array([[-2.], [-1.], [1.], [2.]], np.float32)
    model = ah.fit_logistic(x, np.array([0., .1, .9, 1.]), np.ones(4))
    assert np.array_equal(ah.predict(model, x), np.array([False, False, True, True]))
    print(dict(state='A6_SELF_CHECK_PASS', exact_oracle_fixtures=12, geometry_counts=True,
               reference_kernel=True, soft_logistic=True, project_data_read=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'upper', 'supervised', 'run', 'self-check'))
    parser.add_argument('--cohort', choices=('seed150', 'explore'), default='explore',
                        help='Default complete explore; seed150 is diagnostic only.')
    args = parser.parse_args()
    pc.setup('cpu')
    if args.mode == 'self-check':
        self_check()
        return
    try:
        if args.mode == 'run':
            produce(args.cohort)
            upper(args.cohort)
            supervised(args.cohort)
        else:
            globals()[args.mode](args.cohort)
    except BaselinePending as error:
        print(dict(state='WAITING_P0_2_FORIS_BASELINES', reason=str(error), confirm_scored=False), flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
