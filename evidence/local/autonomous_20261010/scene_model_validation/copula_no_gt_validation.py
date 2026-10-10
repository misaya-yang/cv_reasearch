"""Independent copula gauge/control checks and two frozen no-GT actual replays."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '2')
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
DATA = REPO.parent/'cv_data'
OUT = Path(__file__).resolve().parent
RUN = DATA/'a/reference_boundary_copula200_20261010'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    begun = time.perf_counter()
    cfg, seal = read(RUN/'config.json'), read(RUN/'sealed.json')
    assert seal['n'] == cfg['n'] == 200
    assert sha(RUN/'config.json') == seal['config_sha256']
    for path, key in (('manifest.json', 'manifest_sha256'), ('tasks.json', 'tasks_sha256')):
        assert sha(RUN/path) == cfg[key]
    for relative, digest in cfg['source_sha256'].items():
        assert sha(RUN/'frozen'/relative) == digest
    for key in ('profile', 'basis'):
        assert sha(cfg[key+'_path']) == cfg[key+'_sha256']
    manifests, tasks = read(RUN/'manifest.json'), read(RUN/'tasks.json')
    indices = (0, 100)
    forbidden = {str(Path(manifests[i]['query_mask_path']).resolve()) for i in indices}
    forbidden.update(str(Path(b['path']).resolve()) for i in indices for b in tasks[i]['baselines'].values())
    forbidden.update(str((RUN/name).resolve()) for name in ('edge_metrics.jsonl', 'report.json'))
    assert all(str(Path(manifests[i]['reference_mask_path']).resolve()) not in forbidden for i in indices)
    attempts, writes = [], []
    prefix = str(RUN.resolve())+os.sep
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).resolve())
        if path in forbidden:
            attempts.append(path)
            raise RuntimeError('Forbidden query GT, baseline mask or scoring output access')
        mode, flags = args[1:3]
        mutates = ((isinstance(mode, str) and any(c in mode for c in 'wa+'))
                   or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))))
        if path.startswith(prefix) and mutates:
            writes.append(path)
            raise RuntimeError('Forbidden sealed-probe write')
    sys.addaudithook(guard)
    protected = [RUN/name for name in ('config.json', 'sealed.json', 'manifest.json', 'tasks.json')]
    protected += [RUN/'fields'/seal['receipts'][i]['filename'] for i in indices]
    protected += [RUN/'records'/f'{i:06d}.json' for i in indices]
    before = {str(path): sha(path) for path in protected}
    sys.path[:0] = [str(RUN/'frozen/src'), str(RUN/'frozen/scripts'), str(DATA/'third_party/foris_official')]
    method_path = RUN/'frozen/src/ics/methods/reference_boundary_copula.py'
    spec = importlib.util.spec_from_file_location('independent_frozen_copula', method_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from raw_feature_cache import RawFeatureCache, tensor_hash
    from ics.official_data import load_inputs, array_hash
    from utils.data import build_transform
    torch.set_num_threads(cfg['threads'])
    torch.manual_seed(710)
    # Evaluate original joint/product densities with independent explicit products.
    r = F.normalize(torch.randn(20, 7), dim=1)
    q = F.normalize(torch.randn(12, 7), dim=1)
    c = (np.arange(20) % 9).astype(np.float32)/8
    banks = module._banks(c, (4, 5))
    edges = module._neighbors((3, 4))
    real, shuffled, _ = module._tables(q, r, edges, banks)
    rev_real, rev_shuffle, _ = module._tables(q, r, edges[:, ::-1].copy(), banks)
    active = np.unique(np.concatenate([b.reshape(-1) for b in banks['true'].values() if len(b)]))
    positions = np.full(len(r), -1, np.int64)
    positions[active] = np.arange(len(active))
    common_kernel = torch.exp((q @ r[torch.from_numpy(active)].T).double()/module.TEMPERATURE)
    independent_table = torch.zeros_like(real)
    density_errors = []
    for (y, z), bank in banks['true'].items():
        if not len(bank):
            continue
        ka = common_kernel[torch.from_numpy(edges[:, 0])][:, positions[bank[:, 0]]]
        kb = common_kernel[torch.from_numpy(edges[:, 1])][:, positions[bank[:, 1]]]
        joint = (ka*kb).mean(1)
        product = ka.mean(1)*kb.mean(1)
        density_errors.append(float((real[:, y, z]-(joint/product).log()).abs().max()))
        independent_joint = (ka[:, :, None]*kb[:, None, :]).mean((1, 2))
        independent_table[:, y, z] = (independent_joint/product).log()
    table = torch.randn(17, 2, 2, dtype=torch.float64)
    row_gauge = torch.randn(17, 2, 1, dtype=torch.float64)
    col_gauge = torch.randn(17, 1, 2, dtype=torch.float64)
    antisymmetric = torch.zeros_like(table)
    antisymmetric[:, 0, 1] = torch.randn(17, dtype=torch.float64)
    antisymmetric[:, 1, 0] = -antisymmetric[:, 0, 1]
    synthetic = dict(original_joint_product_kde_max_error=max(density_errors),
                     explicit_factorized_crossproduct_delta_error=float(module._delta(independent_table).abs().max()),
                     endpoint_reverse_true_delta_error=float((module._delta(real)-module._delta(rev_real)).abs().max()),
                     endpoint_reverse_shuffle_delta_error=float((module._delta(shuffled)-module._delta(rev_shuffle)).abs().max()),
                     rowcolumn_label_gauge_error=float((module._delta(table)-module._delta(table+row_gauge+col_gauge)).abs().max()),
                     antisymmetric_label_gauge_error=float((module._delta(table)-module._delta(table+antisymmetric)).abs().max()))
    assert max(synthetic.values()) < 1e-12, synthetic
    profile_path = Path(cfg['profile_path'])
    cache = RawFeatureCache(profile_path.parent.parent, read(profile_path))
    transform = build_transform(1024)
    basis = torch.load(cfg['basis_path'], map_location='cpu', weights_only=True)['basis'].float()
    projection = torch.eye(1024)-basis @ basis.T
    replays = []
    for index in indices:
        row, task, receipt = manifests[index], tasks[index], seal['receipts'][index]
        assert row['episode_id'] == task['episode_id'] == receipt['episode_id']
        fields_path = RUN/'fields'/receipt['filename']
        assert sha(fields_path) == receipt['fields_sha256']
        reference, mask, query = load_inputs(row, DATA)
        features, raw_checks = [], []
        for image, request in zip((reference, query), task['raw']):
            model_input = transform(image).numpy()
            assert cache.key(model_input) == request['key']
            assert tensor_hash(model_input) == request['input_sha256']
            entry_path = cache.folder/request['key']/'entry.json'
            assert sha(entry_path) == request['entry_sha256']
            assert read(entry_path)['file_sha256'] == request['payload_sha256']
            raw = cache.read(model_input, ('O/24',))['O/24']
            assert tensor_hash(raw) == request['tensor_sha256'] and raw.dtype == np.float32 and raw.shape == (4096, 1024)
            features.append(torch.from_numpy(raw))
            raw_checks.append(dict(role=request['role'], shape=list(raw.shape), input_sha256=request['input_sha256'],
                                   tensor_sha256=request['tensor_sha256'], payload_sha256=request['payload_sha256']))
        r, q = (F.normalize(x, dim=1) for x in features)
        binary = F.interpolate(mask.float()[None, None], (64, 64), mode='nearest')[0, 0].reshape(-1) > .5
        mu = q.mean(0)
        semantic = float(F.normalize(r[binary].mean(0), dim=0) @ mu/(mu.norm()+1e-6)) if binary.any() else None
        apd = semantic is None or semantic < .8
        if apd:
            r, q = (F.normalize(x @ projection.T, dim=1) for x in (r, q))
        coverage = F.interpolate(mask.float()[None, None], (1024, 1024), mode='nearest')[0, 0]
        coverage = coverage.reshape(64, 16, 64, 16).mean((1, 3)).reshape(-1).numpy()
        assert apd == receipt['apd_applied']
        result = module.probe(r, coverage, q, reference_grid_hw=(64, 64), query_grid_hw=(64, 64))
        repeat = module.probe(r, coverage, q, reference_grid_hw=(64, 64), query_grid_hw=(64, 64))
        edge_ids = result['edge_index']
        result['query_cosine'] = (q[edge_ids[:, 0]]*q[edge_ids[:, 1]]).sum(1).numpy()
        fields = {}
        with np.load(fields_path, allow_pickle=False) as saved:
            for name in saved.files:
                value = result[name]
                fields[name] = dict(exact_saved=np.array_equal(saved[name], value),
                                    maximum_absolute_error=float(np.max(np.abs(saved[name]-value))),
                                    shape=list(value.shape), dtype=str(value.dtype), tensor_hash=array_hash(value))
                if name != 'query_cosine':
                    fields[name]['exact_repeat'] = np.array_equal(value, repeat[name])
                assert fields[name]['exact_saved'] and fields[name].get('exact_repeat', True), (name, fields[name])
        states = {}
        for state, diagnostic in result['diagnostics']['construction']['states'].items():
            observed = np.asarray(diagnostic['true_pair_ids'], np.int64).reshape(-1, 2)
            null = np.asarray(diagnostic['fixed_shuffle_pair_ids'], np.int64).reshape(-1, 2)
            marginals = [np.array_equal(np.bincount(observed[:, j], minlength=4096),
                                       np.bincount(null[:, j], minlength=4096)) for j in (0, 1)]
            assert all(marginals) and len(observed) == len(null)
            assert len(observed) in (0, 64) and np.all(observed[:, 0] != observed[:, 1])
            states[state] = dict(occurrences=len(observed), both_endpoint_histograms_exact=all(marginals),
                                 unique_observed_pairs=diagnostic['unique_physical_true_pairs'],
                                 effective_observed_pairs=diagnostic['effective_physical_true_pairs'],
                                 null_self_pairs=diagnostic['shuffle_self_pair_count'],
                                 changed_pair_multiset_fraction=diagnostic['pair_multiset_changed_fraction'],
                                 fallback=diagnostic['missing_state_independence_fallback'])
        assert result['diagnostics']['construction'] == receipt['diagnostics']['construction']
        assert np.all(result['delta_factorized'] == 0)
        replays.append(dict(index=index, episode_id=row['episode_id'], dataset=row['dataset'],
                            APD=apd, semantic=semantic, raw=raw_checks, fields=fields, states=states,
                            full_state_mass_closure_error=result['diagnostics']['construction']['four_state_mass_closure_error'],
                            diagnostic_bank_ids_exact_saved=True, factorized_delta_bit_exact_zero=True))
    after = {str(path): sha(path) for path in protected}
    assert before == after and not attempts and not writes
    report = dict(status='PASSED_COPULA_SYNTHETIC_AND_TWO_ACTUAL_FROZEN_REPLAYS_NO_GT',
                  frozen_method_sha256=sha(method_path), frozen_runner_sha256=cfg['source_sha256']['scripts/probe_reference_boundary_copula.py'],
                  validator_sha256=sha(__file__), synthetic=synthetic, actual_replays=replays,
                  blocked_input_open_attempts=attempts, sealed_write_attempts=writes, protected_hashes_unchanged=before == after,
                  query_GT_reads=0, baseline_mask_reads=0, encoder_calls=0, new_masks=0, scoring_executed=False,
                  elapsed_seconds=time.perf_counter()-begun,
                  mathematical_result='delta removes antisymmetric L01-L10 and all label-row/column unaries; actual pairing remains through joint/product KDE.',
                  interpretation_limitations=['Ranking delta against a constant-zero product-null does not measure incremental relation evidence beyond a common lawful unary.',
                                              'True versus finite shuffle tests neighbor coupling including self-exclusion; null self-pairs are disclosed.',
                                              'Query-cosine control screens simple homogeneity but cannot rule out generic nonlinear smoothing.',
                                              'Independent patch-area same/cut is not physical pixel-boundary truth or segmentation mIoU.',
                                              'Before graph inference, compare conditional pair likelihood under identical lawful unary and assess incremental true versus shuffle/product control.'])
    (OUT/'copula_no_gt_validation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], synthetic=synthetic,
                         actual=[dict(index=r['index'], episode=r['episode_id'], exact_fields=len(r['fields']), states=r['states']) for r in replays],
                         query_GT_reads=0, sealed_unchanged=True, seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
