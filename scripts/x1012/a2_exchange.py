"""A2: exchange independent evidence and FoRIS decisions without query GT.

The small cohort is a first diagnostic; the default is the entire fixed explore
manifest. Prediction production and scoring are separate commands. This module
also supplies the same sealed-field workflow to the independently owned A7 file.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
import torch
import lab
import plan_common as pc

FIELDS = lab.ASSETS / 'a/x1012_P0/fields'
ARMS = ('FoRIS', 'FoRIS_pre', 'IND0', 'IND_to_FoRIS_pre',
        'IND_to_FoRIS', 'FoRIS_s2_to_IND0')
REQUIRED = ('c32', 'ind0_32', 'mask/IND0', 'regions64')


def cohort_rows(cohort):
    if cohort == 'seed150':
        return pc.seed_rows(150)
    if cohort == 'explore':
        return pc.read(lab.ASSETS / 'a/x1012_P0/explore.json')
    raise ValueError('Unknown cohort: ' + cohort)


def same_inputs(before, after):
    """P0 may bind baseline outputs later, but cannot change task inputs."""
    return (pc.key(before) == pc.key(after)
            and all(before[k] == after[k] for k in ('dataset', 'row', 'fit')))


def check_fields(rows, required):
    missing, sources = [], {}
    for b in rows:
        name = pc.key(b)
        path, record = FIELDS / 'data' / (name + '.npz'), FIELDS / 'records' / (name + '.json')
        if not path.is_file() or not record.is_file():
            missing.append(name)
            continue
        source = pc.read(record)
        assert source['index'] == name and source['episode_id'] == b['row']['episode_id']
        assert Path(source['path']).resolve() == path.resolve()
        assert source['new_encoder_calls'] == source['query_GT_reads'] == 0
        with np.load(path, allow_pickle=False) as z:
            assert set(required) <= set(z.files), 'Incomplete field schema: ' + name
        sources[name] = {'path': str(path), 'sha256': source['sha256'],
                         'record_path': str(record), 'record_sha256': pc.sha(record)}
    if missing:
        raise RuntimeError(f'P0 fields incomplete: {len(missing)}/{len(rows)}; first keys {missing[:10]}')
    return sources


def prepare_run(task, cohort, required, arms, method, sources):
    pc.setup('cpu')
    out = lab.ASSETS / ('a/x1012_' + task) / cohort
    current = cohort_rows(cohort)
    names = [pc.key(b) for b in current]
    assert len(names) == len(set(names)), 'Duplicate episode keys'
    inputs = check_fields(current, required)
    config_path = out / 'config.json'
    if not config_path.exists():
        for name in ('predictions', 'fields', 'records', 'frozen'):
            (out / name).mkdir(parents=True, exist_ok=True)
        for i, source in enumerate(sources):
            shutil.copyfile(source, out / 'frozen' / f'{i}_{source.name}')
        pc.write(out / 'manifest.json', current)
        pc.write(config_path, {'state': 'FROZEN', 'task': task, 'cohort': cohort,
                              'n': len(current), 'workers': 1, 'threads': 4,
                              'arms': list(arms), 'method': method, 'field_sources': inputs,
                              'source_sha256': {str(p): pc.sha(p) for p in sources},
                              'new_encoder_calls': 0, 'query_GT_in_inference': False})
    config = pc.read(config_path)
    assert config['task'] == task and config['cohort'] == cohort and config['arms'] == list(arms)
    assert config['field_sources'] == inputs, 'Frozen P0 fields changed'
    assert all(pc.sha(p) == h for p, h in config['source_sha256'].items()), 'Frozen execution source changed'
    frozen = pc.read(out / 'manifest.json')
    assert len(frozen) == len(current) and all(same_inputs(a, b) for a, b in zip(frozen, current))
    return out, frozen, config


def load_fields(b, config, required):
    source = config['field_sources'][pc.key(b)]
    assert pc.sha(source['record_path']) == source['record_sha256']
    assert pc.sha(source['path']) == source['sha256'], 'P0 field digest changed'
    with np.load(source['path'], allow_pickle=False) as z:
        return {k: z[k].copy() for k in required}


def checked_record(out, b, config):
    path = out / 'records' / (pc.key(b) + '.json')
    if not path.exists():
        return None
    record = pc.read(path)
    assert record['episode_id'] == b['row']['episode_id']
    assert record['input_field_sha256'] == config['field_sources'][pc.key(b)]['sha256']
    for prefix in ('prediction', 'field'):
        assert pc.sha(record[prefix + '_path']) == record[prefix + '_sha256']
    assert record['new_encoder_calls'] == record['query_GT_reads'] == 0
    return record


def save_record(out, b, config, predictions, fields, began, extra):
    name = pc.key(b)
    pred, field = out / 'predictions' / (name + '.npz'), out / 'fields' / (name + '.npz')
    assert not pred.exists() and not field.exists(), 'Preserve incomplete output: ' + name
    np.savez_compressed(pred, **predictions)
    np.savez_compressed(field, **fields)
    record = {'index': name, 'episode_id': b['row']['episode_id'], 'dataset': b['dataset'],
              'prediction_path': str(pred), 'prediction_sha256': pc.sha(pred),
              'field_path': str(field), 'field_sha256': pc.sha(field),
              'input_field_sha256': config['field_sources'][name]['sha256'],
              'seconds': time.monotonic() - began, 'new_encoder_calls': 0,
              'query_GT_reads': 0, **extra}
    pc.write(out / 'records' / (name + '.json'), record)
    return record


def seal_run(out, rows, records, began):
    assert [r['index'] for r in records] == [pc.key(b) for b in rows]
    seal = {'state': 'ALL_PREDICTIONS_SEALED', 'n': len(records), 'records': records,
            'manifest_sha256': pc.sha(out / 'manifest.json'),
            'config_sha256': pc.sha(out / 'config.json'),
            'seconds': time.monotonic() - began, 'new_encoder_calls': 0, 'query_GT_reads': 0}
    pc.write(out / 'sealed.json', seal)
    print({'state': seal['state'], 'n': seal['n'], 'out': str(out)}, flush=True)


def read_archive(b, required=True):
    entry = b.get('old_arms', {}).get('B')
    if entry is None:
        if required:
            raise RuntimeError('P0 FoRIS baseline not bound: ' + pc.key(b))
        return None, None
    digest = pc.sha(entry['path'])
    if 'sha256' in entry:
        assert digest == entry['sha256'], 'FoRIS archive digest changed'
    with np.load(entry['path'], allow_pickle=False) as z:
        bits = np.asarray(z[entry['cli_key']], dtype=np.uint8).copy()
    assert bits.size >= 1024 * 1024 // 8
    return pc.unpack(bits), {'index': pc.key(b), 'path': entry['path'],
                            'cli_key': entry['cli_key'], 'sha256': digest}


def score_task(task, cohort, stored_arms, notes, verify_replay=False):
    """Validate the complete seal and all baselines before the first GT read."""
    import plan_report as pr
    pc.setup('cpu')
    out = lab.ASSETS / ('a/x1012_' + task) / cohort
    seal = pc.read(out / 'sealed.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED'
    for name in ('manifest', 'config'):
        assert pc.sha(out / (name + '.json')) == seal[name + '_sha256']
    config, frozen = pc.read(out / 'config.json'), pc.read(out / 'manifest.json')
    current = cohort_rows(cohort)
    assert len(frozen) == len(current) == seal['n']
    assert all(same_inputs(a, b) for a, b in zip(frozen, current))
    assert all('B' in b.get('old_arms', {}) for b in current), 'Wait for all P0 baseline bindings before scoring'
    indexed = {r['index']: r for r in seal['records']}
    assert len(indexed) == len(current)
    baseline_receipts = []
    for b in current:
        record = checked_record(out, b, config)
        assert record == indexed[pc.key(b)]
        with pc.inference_only(b):
            archive, receipt = read_archive(b)
            with np.load(record['prediction_path'], allow_pickle=False) as z:
                assert set(z.files) == set(stored_arms)
                if verify_replay:
                    assert np.array_equal(pc.unpack(z['FoRIS']), archive), 'Source FoRIS archive mismatch: ' + pc.key(b)
        baseline_receipts.append(receipt)
    pc.write(out / 'baseline_verified.json', {'state': 'ALL_BASELINES_VERIFIED',
             'n': len(current), 'source_replay_bit_exact': verify_replay,
             'prediction_seal_sha256': pc.sha(out / 'sealed.json'), 'baselines': baseline_receipts})
    records = []
    arms = ('FoRIS',) + tuple(a for a in stored_arms if a != 'FoRIS')
    for b, receipt in zip(current, baseline_receipts):
        record = indexed[pc.key(b)]
        assert pc.sha(record['prediction_path']) == record['prediction_sha256']
        with np.load(record['prediction_path'], allow_pickle=False) as z:
            predictions = {a: pc.unpack(z[a]) for a in stored_arms}
        archive, latest = read_archive(b)
        assert latest == receipt, 'Baseline changed after validation'
        predictions['FoRIS'] = archive
        records.append(pr.score_record(b, predictions))
    label = task if cohort == 'explore' else task + '_seed150'
    result = pr.report(label, out, records, arms, notes,
                       extra={'cohort': cohort, 'fixed_manifest_sha256': seal['manifest_sha256'],
                              'baseline_verified_sha256': pc.sha(out / 'baseline_verified.json')})
    return out, result


def region_mean(field, labels):
    value = np.asarray(field, dtype=np.float32).reshape(-1)
    ids = np.asarray(labels, dtype=np.int64).reshape(-1)
    assert value.size == ids.size == 4096 and ids.min() >= 0 and ids.max() < 64
    counts = np.bincount(ids, minlength=64)
    sums = np.bincount(ids, weights=value, minlength=64)
    return (sums / np.maximum(counts, 1))[ids].astype(np.float32).reshape(64, 64)


@contextmanager
def independent_stage2(host, c, mu_fg):
    name = '_part2_stage2_contrastive_score'
    had, previous = name in host.__dict__, host.__dict__.get(name)
    calls = [0]
    def replacement(*args, **kwargs):
        calls[0] += 1
        return c, c, 1 - c, mu_fg
    setattr(host, name, replacement)
    try:
        yield
        assert calls[0] == 1, 'Stage-2 exchange was not applied exactly once'
    finally:
        if had:
            setattr(host, name, previous)
        else:
            delattr(host, name)


def produce(cohort):
    sources = (Path(__file__), Path(pc.__file__), Path(lab.__file__),
               lab.ROOT / 'src/ics/foris.py', lab.ROOT / 'scripts/cached_dino.py',
               lab.ASSETS / 'third_party/foris_official/models/foris.py',
               lab.ASSETS / 'third_party/foris_official/utils/refinement.py')
    method = {'IND_to_FoRIS': 'replace stage2 tuple by (c32,c32,1-c32,coverage-weighted IND reference prototype); original downstream incl CRF',
              'FoRIS_to_IND': 'actual s2 -> cached independent K64 region mean -> bilinear1024 -> minmax -> >.5',
              'archive_gate': 'source final mask bit-exact for bound archives; unbound new pairs verified before scoring'}
    out, rows, config = prepare_run('A2', cohort, REQUIRED, ARMS, method, sources)
    if (out / 'sealed.json').exists():
        print('A2 ' + cohort + ' already sealed', flush=True)
        return
    host, adapter = pc.host_cpu()
    from ics.foris import run_foris
    from ics.official_data import load_inputs
    began, records = time.monotonic(), []
    for i, b in enumerate(rows, 1):
        old = checked_record(out, b, config)
        if old is not None:
            records.append(old)
            continue
        started = time.monotonic()
        adapter.bind(b)
        try:
            with pc.inference_only(b):
                fields = load_fields(b, config, REQUIRED)
                c = torch.from_numpy(fields['c32']).reshape(64, 64)
                assert c.dtype == torch.float32 and torch.isfinite(c).all() and bool(((c >= 0) & (c <= 1)).all())
                reference, mask, query = load_inputs(b['row'], lab.ASSETS)
                final, got, transformed, _ = run_foris(host, reference, mask, query)
                archive, receipt = read_archive(b, required=False)
                if archive is not None:
                    assert np.array_equal(final.numpy(), archive), 'Source FoRIS archive mismatch: ' + pc.key(b)
                predictions = {'FoRIS': pc.pack(final.numpy()), 'FoRIS_pre': pc.pack(got['pre'].numpy()),
                               'IND0': fields['mask/IND0']}
                s2 = got['s2'].numpy().copy()
                reverse = region_mean(s2, fields['regions64'])
                predictions['FoRIS_s2_to_IND0'] = pc.pack(pc.readout(reverse))
                r = lab.debias(lab.feats(b, 'reference'))
                y = lab.coverage64(transformed).flatten()
                assert float(y.sum()) > 0
                mu_fg = lab.unit((r * y[:, None]).sum(0))
                with independent_stage2(host, c, mu_fg):
                    swapped, exchanged, _, _ = run_foris(host, reference, mask, query)
                assert torch.equal(exchanged['s2'], c)
                predictions['IND_to_FoRIS_pre'] = pc.pack(exchanged['pre'].numpy())
                predictions['IND_to_FoRIS'] = pc.pack(swapped.numpy())
                assert adapter.encoder is None and adapter.encoder_calls == 0
                saved_fields = {'ind_c': c.numpy(), 'foris_s2': s2, 'foris_s2_region': reverse,
                                'foris_pre': got['score'].numpy(),
                                'ind_to_foris_pre': exchanged['score'].numpy()}
            record = save_record(out, b, config, predictions, saved_fields, started,
                                 {'archive_bit_exact': True if receipt is not None else None,
                                  'archive_receipt': receipt, 'exchange_tuple': ['c', 'c', '1-c', 'IND_mu_fg']})
            records.append(record)
        finally:
            adapter.memo.clear()
        if i <= 3 or i % 50 == 0:
            print({'task': 'A2', 'cohort': cohort, 'completed': i, 'n': len(rows),
                   'seconds': round(time.monotonic() - began, 1)}, flush=True)
    seal_run(out, rows, records, began)


def score(cohort):
    out, result = score_task('A2', cohort, ARMS, [
        '两个交换方向使用固定完整队列，推理只读缓存O24/参考掩码/RGB；先封存全部预测，再核对所有FoRIS archive，最后读query GT。',
        '方向1完整替换(score,sf,sbn,mu_fg)为(c,c,1-c,独立参考原型)，原Part3/Part4及CRF不变；sf/sbn不随后续score更新。',
        '方向2取实际FoRIS s2，以独立K64区域均值、先上采样后minmax/>.5读出。另报交换前后pre，以区分CRF影响。',
        '方向1仍保留官方下游特征/先验/惩罚，是交换诊断，不是最多借两步的独立方法；c与互补背景也改变了原辅助证据的语义。',
    ], verify_replay=True)
    decision = {d: {'IND_evidence_equivalent_with_FoRIS_decision': abs(s['IND_to_FoRIS']['delta_to_foris_pp']) <= .5,
                    'IND_decision_equivalent_with_FoRIS_s2': abs(s['FoRIS_s2_to_IND0']['delta_to_foris_pp']) <= .5,
                    'IND_to_FoRIS_delta_pp': s['IND_to_FoRIS']['delta_to_foris_pp'],
                    'FoRIS_s2_to_IND0_delta_pp': s['FoRIS_s2_to_IND0']['delta_to_foris_pp']}
                for d, s in result['summary'].items()}
    pc.write(out / 'exchange_decisions.json', decision)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('produce', 'score'))
    parser.add_argument('--cohort', choices=('seed150', 'explore'), default='explore')
    args = parser.parse_args()
    produce(args.cohort) if args.mode == 'produce' else score(args.cohort)
