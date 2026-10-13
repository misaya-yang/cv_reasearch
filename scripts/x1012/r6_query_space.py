"""R6: fixed VP primary, 2x2 query graph space by unary reliability.

Raw O/24 is used only to organize the query graph. The cross-image evidence,
reference fallback, reciprocal20 graph formula, lambda16, solver and .5 readout
are unchanged. All 1360 natural outputs are sealed before the scorer opens GT.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.linalg import cg

REPO = Path(os.environ.get('CV_RESEARCH_ROOT', Path.cwd())).resolve()
ASSETS = REPO.parent / 'cv_data'
OUT = ASSETS / 'a/x1012_R6_query_space'
R1 = ASSETS / 'a/x1012_R1'
B4 = ASSETS / 'a/x1012_B4/seed150'
MOD = ASSETS / 'a/x1012_B4_modulated/region/seed150'
ARMS = ('debiased_G0', 'debiased_UN', 'raw_G0', 'raw_UN')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def initialize():
    global lab, pc, pr, graph, solve, CONFIG, INPUTS
    sys.path.insert(0, str(OUT / 'runtime'))
    lab = importlib.import_module('lab')
    lab.ROOT, lab.ASSETS, lab.DEV = REPO, ASSETS, torch.device('cpu')
    pc = importlib.import_module('plan_common')
    pr = importlib.import_module('plan_report')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    # Execute the unchanged two pure R1 functions without importing the old
    # task launcher, scorer, b1 runner or its mutable source dependencies.
    source = OUT / 'runtime/r1_graph_source.py'
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('graph', 'solve')]
    assert len(functions) == 2
    namespace = {'np': np, 'torch': torch, 'sparse': sparse, 'cg': cg, 'lab': lab}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    graph, solve = namespace['graph'], namespace['solve']
    CONFIG, INPUTS = read(OUT / 'config.json'), read(OUT / 'inputs.json')


def freeze(device):
    if device == 'mps':
        assert torch.backends.mps.is_available(), 'MPS unavailable; explicitly freeze the CPU fallback'
    runtime = OUT / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    mapping = {'lab.py': R1 / 'frozen/3_lab.py', 'plan_common.py': R1 / 'frozen/1_plan_common.py',
               'plan_report.py': R1 / 'frozen/2_plan_report.py', 'r1_graph_source.py': R1 / 'frozen/0_r1_reliability_graph.py'}
    for name, source in mapping.items():
        target = runtime / name
        if target.exists():
            assert sha(target) == sha(source), 'Preserve existing runtime'
        else:
            shutil.copyfile(source, target)
    manifests = read(R1 / 'manifest.json')
    assert len(manifests) == 1360
    receipts = {label: {r['index']: r for r in read(base / 'sealed.json')['records']}
                for label, base in (('R1', R1), ('B4', B4), ('MOD', MOD))}
    sources = {}
    for binding in manifests:
        name = str(binding['index']) if isinstance(binding['index'], str) else f"{binding['index']:06d}"
        sources[name] = {}
        for label in receipts:
            record = receipts[label][name]
            sources[name][label] = {k: record[k] for k in ('field_path', 'field_sha256')}
            if label == 'R1':
                sources[name][label].update({k: record[k] for k in ('prediction_path', 'prediction_sha256')})
    config = {'state': 'FROZEN_BEFORE_ALL_1360_NATURAL_OUTPUTS', 'n': 1360, 'arms': ARMS,
              'hypothesis': 'R1 VP_UN improves multiple domains but remains below strong controls on some domains; INSID3 separates debiased cross-image matching from raw-unit query organization. Test that space separation alone on the identical R1 primary.',
              'positive_decision': 'A paired raw-graph gain over both debiased_UN and raw_G0 supports the proposed space/reliability mechanism; full FoRIS and INSID3 comparisons and fresh-pair confirmation remain necessary.',
              'negative_decision': 'Close this exact raw-query reciprocal20 intervention; do not tune graph lambda, neighbors, threshold or combine other modules in this run.',
              'history_check': {'scope': 'Quick active-code/result-summary search, not an exhaustive originality search',
                                'known_prior': 'Old 60-pair raw-query graph used32NN, APD hard1NN unary and .5 absorption and did not beat strong baselines; different unary/graph/control design.',
                                'exact_2x2_found': False, 'originality_claim': False},
              'primary': 'unchanged R1 c_mod + (addition_V + addition_P), V=.20*(vote-.5), P=.25*(prior-.5)',
              'cross_image_features': 'unchanged debiased R/Q, single reference, tau_kernel .05',
              'raw_graph': 'unit(raw query O/24); exact cosine20NN excluding self; unchanged R1 reciprocal weights, mean-degree normalization and Laplacian',
              'reliability': 'reuse R1 explanation=sigmoid(P0_FP16_logE-median_logE); G0 a=a0, UN a=a0*explanation; normalize a by mean',
              'solver': 'unchanged R1 graph/solve source functions; diag(a)+16L, CG rtol1e-7, atol1e-9,maxiter300',
              'fallback': 'reference rho>.9: identical original T for all four arms',
              'readout': 'unchanged bilinear1024 align_corners=False, minmax, strict>.5',
              'debiased_controls': 'bitwise copies of R1 VP_G0/VP_UN; saved field readout must match each copied mask',
              'natural_order': 'all1360 four-arm predictions sealed before any new scoring invocation',
              'scoring': 'CLI1024 observed(fold,class) I/U pooling; paired query-photo1000 bootstrap; R1 controls reused only after seal',
              'resources': {'neighbor_device': device, 'MPS_processes': int(device == 'mps'),
                            'CPU_threads': 2, 'CPU_fallback_workers': 2, 'release_MPS_cache_each_case': True},
              'source_sha256': sha(__file__), 'runtime_sources': {str(runtime / k): sha(runtime / k) for k in mapping},
              'parent_seals': {str(base / 'sealed.json'): sha(base / 'sealed.json') for base in (R1, B4, MOD)},
              'manifest_sha256': sha(R1 / 'manifest.json'), 'reference_count': 1,
              'new_encoder_calls': 0, 'query_GT_in_inference': False, 'FoRIS_score_consumed': False,
              'domain_branch': False, 'training': False}
    if (OUT / 'config.json').exists():
        assert read(OUT / 'config.json') == json.loads(json.dumps(config)), 'Frozen R6 changed'
    else:
        (OUT / 'frozen').mkdir()
        shutil.copyfile(__file__, OUT / 'frozen/r6_query_space.py')
        shutil.copyfile(R1 / 'manifest.json', OUT / 'manifest.json')
        write(OUT / 'inputs.json', sources)
        config['inputs_sha256'] = sha(OUT / 'inputs.json')
        write(OUT / 'config.json', config)
    print({'phase': 'frozen', 'n': 1360, 'device': device, 'config_sha256': sha(OUT / 'config.json')}, flush=True)


def check_config():
    config = read(OUT / 'config.json')
    assert sha(__file__) == config['source_sha256']
    assert sha(OUT / 'manifest.json') == config['manifest_sha256']
    assert sha(OUT / 'inputs.json') == config['inputs_sha256']
    assert all(sha(path) == digest for path, digest in config['runtime_sources'].items())
    assert all(sha(path) == digest for path, digest in config['parent_seals'].items())
    return config


def raw_query(binding):
    metadata = binding['fit']['query_raw_identity']
    payload = Path(metadata['payload_path'])
    entry = Path(metadata.get('entry_path', payload.parent / 'entry.json'))
    info = read(entry)
    if metadata.get('entry_sha256'):
        assert sha(entry) == metadata['entry_sha256']
    assert sha(payload) == info['file_sha256'] and payload.name == info['file']
    if metadata.get('payload_sha256'):
        assert metadata['payload_sha256'] == info['file_sha256']
    with np.load(payload, allow_pickle=False) as source:
        value = source['O/24'].copy()
    assert value.shape == (4096, 1024) and value.dtype == np.float32 and np.isfinite(value).all()
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256(json.dumps([list(contiguous.shape), contiguous.dtype.str]).encode())
    digest.update(contiguous.tobytes())
    assert digest.hexdigest() == info['features']['O/24']['tensor_sha256']
    return torch.from_numpy(value), info['file_sha256']


def raw_neighbors(q):
    unit = F.normalize(q, dim=1)
    device = CONFIG['resources']['neighbor_device']
    if device == 'mps':
        gpu = unit.to('mps')
        similarity = gpu @ gpu.T - torch.eye(4096, device='mps') * 1e9
        neighbors = similarity.topk(20, dim=1).indices.cpu().numpy().astype(np.uint16)
        del gpu, similarity
        torch.mps.empty_cache()
    else:
        neighbors = np.empty((4096, 20), np.uint16)
        for start in range(0, 4096, 256):
            stop = min(start + 256, 4096)
            similarity = unit[start:stop] @ unit.T
            similarity[torch.arange(stop - start), torch.arange(start, stop)] -= 1e9
            neighbors[start:stop] = similarity.topk(20, dim=1).indices.numpy().astype(np.uint16)
    return neighbors


def produce_one(binding):
    name = pc.key(binding)
    record_path = OUT / 'records' / (name + '.json')
    if record_path.exists():
        record = read(record_path)
        for kind in ('prediction', 'field'):
            assert sha(record[kind + '_path']) == record[kind + '_sha256']
        return record
    started = time.monotonic()
    source = INPUTS[name]
    detail, fields, predictions = {}, {}, {}
    with pc.inference_only(binding):
        r1 = source['R1']
        assert sha(r1['prediction_path']) == r1['prediction_sha256']
        assert sha(r1['field_path']) == r1['field_sha256']
        with np.load(r1['prediction_path'], allow_pickle=False) as saved:
            for mode in ('G0', 'UN'):
                predictions['debiased_' + mode] = saved['VP_' + mode].copy()
        with np.load(r1['field_path'], allow_pickle=False) as saved:
            rho = float(saved['reference_rho'][0])
            explanation = saved['explanation'].copy()
            for mode in ('G0', 'UN'):
                fields['debiased_' + mode] = saved['VP_' + mode].copy()
                assert np.array_equal(pc.pack(pc.readout(fields['debiased_' + mode])), predictions['debiased_' + mode])
        if rho > .9:
            assert np.array_equal(predictions['debiased_G0'], predictions['debiased_UN'])
            for mode in ('G0', 'UN'):
                fields['raw_' + mode] = fields['debiased_' + mode].copy()
                predictions['raw_' + mode] = predictions['debiased_' + mode].copy()
            detail['raw_T_fallback'] = True
        else:
            for label in ('B4', 'MOD'):
                assert sha(source[label]['field_path']) == source[label]['field_sha256']
            with np.load(source['B4']['field_path'], allow_pickle=False) as saved:
                additions = saved['addition_V'] + saved['addition_P']
            with np.load(source['MOD']['field_path'], allow_pickle=False) as saved:
                primary = saved['c_mod'] + additions
            q, payload_sha = raw_query(binding)
            graph_start = time.monotonic()
            neighbors = raw_neighbors(q)
            w, laplacian = graph(q, neighbors)
            detail.update(raw_payload_sha256=payload_sha, graph_seconds=time.monotonic() - graph_start,
                          raw_T_fallback=False, raw_graph_edges=int(w.nnz))
            solver = {}
            for mode in ('G0', 'UN'):
                field, solver[mode] = solve(primary, explanation, laplacian, mode, None)
                fields['raw_' + mode] = field
                predictions['raw_' + mode] = pc.pack(pc.readout(field))
            fields.update(query_neighbors20=neighbors, graph_data=w.data, graph_indices=w.indices,
                          graph_indptr=w.indptr, primary=primary, explanation=explanation)
            detail['solver'] = solver
        fields['reference_rho'] = np.asarray([rho], np.float32)
        prediction_path, field_path = OUT / 'predictions' / (name + '.npz'), OUT / 'fields' / (name + '.npz')
        assert not prediction_path.exists() and not field_path.exists(), 'Preserve partial output'
        np.savez_compressed(prediction_path, **predictions)
        np.savez_compressed(field_path, **fields)
    record = {'index': name, 'dataset': binding['dataset'], 'episode_id': binding['row']['episode_id'],
              'prediction_path': str(prediction_path), 'prediction_sha256': sha(prediction_path),
              'field_path': str(field_path), 'field_sha256': sha(field_path), 'reference_rho': rho,
              'R1_prediction_sha256': r1['prediction_sha256'], 'debiased_control_xor': {'G0': 0, 'UN': 0},
              'seconds': time.monotonic() - started, 'reference_count': 1, 'new_encoder_calls': 0,
              'inference_query_GT_reads': 0, 'FoRIS_score_consumed': False, **detail}
    write(record_path, record)
    return record


def produce():
    config = check_config()
    initialize()
    for folder in ('records', 'predictions', 'fields'):
        (OUT / folder).mkdir(exist_ok=True)
    rows, records, started = read(OUT / 'manifest.json'), [], time.monotonic()
    if config['resources']['neighbor_device'] == 'mps':
        iterator = map(produce_one, rows)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'), initializer=initialize)
        iterator = pool.map(produce_one, rows, chunksize=1)
    try:
        for i, record in enumerate(iterator, 1):
            records.append(record)
            if i <= 3 or i % 50 == 0:
                print({'phase': 'R6-produce', 'completed': i, 'n': len(rows), 'seconds': round(time.monotonic() - started, 1)}, flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    seal = {'state': 'ALL_NATURAL_OUTPUTS_SEALED', 'n': len(records), 'records': records,
            'manifest_sha256': sha(OUT / 'manifest.json'), 'config_sha256': sha(OUT / 'config.json'),
            'wall_seconds': time.monotonic() - started, 'case_seconds_sum': sum(r['seconds'] for r in records),
            'new_encoder_calls': 0, 'inference_query_GT_reads': 0, 'debiased_control_xor': {'G0': 0, 'UN': 0},
            'fallback_n': sum(r['raw_T_fallback'] for r in records), 'resources': config['resources']}
    write(OUT / 'sealed.json', seal)
    print({'phase': 'R6-natural-sealed', 'n': len(records), 'seal_sha256': sha(OUT / 'sealed.json')}, flush=True)


def score_one(job):
    binding, record = job
    with np.load(record['prediction_path'], allow_pickle=False) as saved:
        predictions = {arm: pc.unpack(saved[arm]) for arm in ARMS}
    result = pr.score_record(binding, predictions)
    gt, valid = pc.truth(binding)
    baseline = predictions['debiased_UN']
    result['edits_vs_debiased_UN'] = {
        arm: {'add_TP': int((mask & ~baseline & gt & valid).sum()),
              'add_FP': int((mask & ~baseline & ~gt & valid).sum()),
              'delete_TP': int((~mask & baseline & gt & valid).sum()),
              'delete_FP': int((~mask & baseline & ~gt & valid).sum())}
        for arm, mask in predictions.items()}
    result['reference_rho'] = record['reference_rho']
    return result


def make_report(records, seal, seconds, insid3_path=None):
    baselines = ['FoRIS', 'MEAN', 'VP_previous', 'R1UN']
    extra = {'production': {k: v for k, v in seal.items() if k != 'records'}, 'score_seconds': seconds}
    if insid3_path is not None:
        independent = read(insid3_path)
        source = {r['index']: r for r in independent['records']}
        for record in records:
            matching = source[record['index']]
            assert matching['query_photo'] == record['query_photo']
            assert matching['metrics']['FoRIS'] == record['metrics']['FoRIS']
            record['metrics']['INSID3'] = matching['metrics']['INSID3']
        baselines.append('INSID3')
        extra['INSID3_report_sha256'] = sha(insid3_path)
        extra['INSID3_report_path'] = str(insid3_path)
    report = pr.report('R6_query_space', OUT, records, tuple(baselines) + ARMS, [
        '固定原R1的1360单参考开发任务及VP primary；只将query20NN与边权空间从debiased改为原始单位O24。',
        '2×2为图空间(debiased/raw)与可靠度(G0/UN)；kernel、primary、rho>.9原T、lambda16、strict>.5全部不变。',
        '两debiased臂逐位复用R1 VP_G0/VP_UN；全部1360自然四臂先封存后评分，无新增编码、无queryGT推理、无跨任务训练。',
        '旧60对已有不同raw-query图实验，本项只作精确机制对照，不声称原创；不按域选图/阈值。',
        'FoRIS、MEAN、旧VP和R1UN完整同批指标只在封存后的评分阶段复用；INSID3完成后再加入。'], extra=extra)
    contrasts = {}
    for treatment, control in (('raw_UN', 'debiased_UN'), ('raw_UN', 'raw_G0'),
                               ('raw_UN', 'debiased_G0'), ('raw_G0', 'debiased_G0')):
        paired = [dict(r, metrics={'FoRIS': r['metrics'][control], treatment: r['metrics'][treatment]}) for r in records]
        contrasts[treatment + '_minus_' + control] = pr.summary(paired, ('FoRIS', treatment))
    write(OUT / 'mechanism_contrasts.json', contrasts)
    completion = {'state': 'COMPLETE_WITH_INSID3' if insid3_path else 'COMPLETE_PENDING_INSID3_CONTROL',
                  'n': len(records), 'new_encoder_calls': 0, 'inference_query_GT_reads': 0,
                  'ten_domain_nonnegative_vs_FoRIS': [arm for arm in ARMS if all(s[arm]['delta_to_foris_pp'] >= 0 for s in report['summary'].values())],
                  'worst_domain_delta_to_FoRIS': {arm: min(s[arm]['delta_to_foris_pp'] for s in report['summary'].values()) for arm in ARMS},
                  'raw_UN_minus_debiased_UN': {domain: s['raw_UN']['delta_to_foris_pp'] for domain, s in contrasts['raw_UN_minus_debiased_UN'].items()},
                  'report_sha256': sha(OUT / 'report.json'), 'seal_sha256': sha(OUT / 'sealed.json'),
                  'config_sha256': sha(OUT / 'config.json'), 'score_seconds': seconds}
    write(OUT / 'completion.json', completion)
    print(completion, flush=True)


def score(insid3_path=None):
    check_config()
    initialize()
    seal, rows = read(OUT / 'sealed.json'), read(OUT / 'manifest.json')
    assert seal['state'] == 'ALL_NATURAL_OUTPUTS_SEALED' and seal['n'] == len(rows) == 1360
    assert seal['config_sha256'] == sha(OUT / 'config.json') and seal['manifest_sha256'] == sha(OUT / 'manifest.json')
    receipts = {r['index']: r for r in seal['records']}
    for record in receipts.values():
        for kind in ('prediction', 'field'):
            assert sha(record[kind + '_path']) == record[kind + '_sha256']
    started, records = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'), initializer=initialize) as pool:
        for i, record in enumerate(pool.map(score_one, ((b, receipts[pc.key(b)]) for b in rows), chunksize=1), 1):
            records.append(record)
            if i % 250 == 0:
                print({'phase': 'R6-score', 'completed': i, 'n': len(rows)}, flush=True)
    original = {r['index']: r for r in read(R1 / 'report.json')['records']}
    for record in records:
        old = original[record['index']]
        assert old['query_photo'] == record['query_photo']
        assert record['metrics']['debiased_G0'] == old['metrics']['VP_G0']
        assert record['metrics']['debiased_UN'] == old['metrics']['VP_UN']
        for arm in ('FoRIS', 'MEAN', 'VP_previous'):
            record['metrics'][arm] = old['metrics'][arm]
        record['metrics']['R1UN'] = old['metrics']['VP_UN']
    make_report(records, seal, time.monotonic() - started, insid3_path)


def add_insid3(path):
    assert path is not None
    check_config()
    initialize()
    report = read(OUT / 'report.json')
    make_report(report['records'], read(OUT / 'sealed.json'), report['extra']['score_seconds'], path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'produce', 'score', 'add-insid3'))
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    parser.add_argument('--insid3-report', type=Path)
    arguments = parser.parse_args()
    if arguments.mode == 'freeze':
        freeze(arguments.device)
    elif arguments.mode == 'produce':
        produce()
    elif arguments.mode == 'score':
        score(arguments.insid3_report)
    else:
        add_insid3(arguments.insid3_report)
