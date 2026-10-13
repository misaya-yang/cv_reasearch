"""Complete AND/OR-network search over six frozen DINO prediction pathways.

All monotone Boolean functions with f(0)=0 and f(1)=1 are executable finite
union/intersection networks. Generate the entire data-independent library first,
then rank programs on exposed development data. No model parameters are fitted.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import heapq
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
from pathlib import Path
import shutil
import time
import numpy as np
import x1012s2_composition_library as lib

OUT = lib.OUT.parent / 'boolean_network_v2'


def monotone_tables(n):
    tables = np.asarray([0, 1], np.uint64)
    sizes = [2]
    for variables in range(1, n + 1):
        shift = np.uint64(1 << (variables - 1))
        pieces = []
        complement = np.bitwise_not(tables)
        for lower in tables:
            upper = tables[(lower & complement) == 0]
            pieces.append(lower | (upper << shift))
        tables = np.concatenate(pieces)
        sizes.append(len(tables))
    return tables, sizes


def minimal_terms(table):
    result = []
    for pattern in range(64):
        if not ((table >> pattern) & 1):
            continue
        if all(not ((table >> (pattern ^ (1 << bit))) & 1)
               for bit in range(6) if pattern & (1 << bit)):
            result.append(pattern)
    return result


def expression(table):
    terms = minimal_terms(table)
    assert terms and all(terms)
    return '|'.join('(' + '&'.join(s for bit, s in enumerate(lib.SYMBOLS)
                                  if term & (1 << bit)) + ')' for term in terms)


def dependency_masks(tables):
    out = np.zeros(len(tables), np.uint8)
    for bit in range(6):
        low = np.uint64(sum(1 << p for p in range(64) if not (p & (1 << bit))))
        changes = ((tables ^ (tables >> np.uint64(1 << bit))) & low) != 0
        out |= changes.astype(np.uint8) << bit
    return out


def freeze():
    if (OUT / 'config.json').exists():
        result = lib.read(OUT / 'config.json')
        assert lib.sha(__file__) == result['source_sha256']
        print({'state': result['state'], 'n_rules': result['n_rules'], 'already_frozen': True})
        return
    first = lib.read(lib.OUT / 'config.json')
    assert lib.sha(lib.OUT / 'natural_bindings.json') == first['natural_bindings_sha256']
    tables, sizes = monotone_tables(6)
    tables = tables[(tables != 0) & (tables != np.iinfo(np.uint64).max)]
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / 'natural_truth_tables.npy', tables)
    shutil.copyfile(__file__, OUT / 'source.py')
    config = {'state': 'COMPLETE_NATURAL_AND_OR_NETWORK_LIBRARY_FROZEN',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_symbols_LSB_first': list(lib.SYMBOLS), 'n_rules': len(tables),
        'n_cases': first['n_cases'], 'monotone_function_counts_including_constants': sizes,
        'source_sha256': lib.sha(__file__),
        'truth_tables_sha256': lib.sha(OUT / 'natural_truth_tables.npy'),
        'natural_bindings_path': str(lib.OUT / 'natural_bindings.json'),
        'natural_bindings_sha256': first['natural_bindings_sha256'],
        'parent_library_config_sha256': lib.sha(lib.OUT / 'config.json'),
        'construction': 'For n variables enumerate every ordered pair of monotone(n-1) lower/upper slices satisfying lower<=upper pointwise. Remove only the constant0/constant1 functions, which cannot be formed from nonconstant source masks using AND/OR alone.',
        'operators': ['union', 'intersection'], 'branching': True, 'reuse': True,
        'operation_count_limit': None,
        'prediction': 'A selected table compiles to the union of its minimal positive conjunctions; at most20 conjunctions for six variables. This is an equivalent implementation, not an imposed search-depth limit.',
        'training': 'No feature/model parameter fitting. Exposed development GT selects one fixed operation network, exactly as it selects a method configuration; not a fresh validation claim.',
        'selection': 'Equal-weight ten-domain class mIoU, retain best per exact source dependency for actual latency evaluation.',
        'scope': 'All positive AND/OR compositions of these six existing binary pathways; does not cover continuous fields, extra observations, or other algorithms.',
        'latency': 'Measure complete required source computation and Boolean execution, with shared work counted once.'}
    lib.write(OUT / 'config.json', config)
    print({'state': config['state'], 'n_rules': len(tables), 'counts': sizes,
           'config_sha256': lib.sha(OUT / 'config.json')}, flush=True)


def summary_candidate(index, table, scores, domains, base_scores):
    sources = lib.essential_sources(table)
    return {'id': f'n{index:07d}', 'index': int(index), 'truth_table_hex': f'{table:016x}',
            'expression': expression(table), 'minimal_positive_terms': minimal_terms(table),
            'essential_sources': sources, 'macro10_class_miou': float(np.mean(scores)),
            'macro_delta_to_FoRIS_pp': float(np.mean(scores - base_scores)),
            'worst_delta_to_FoRIS_pp': float(np.min(scores - base_scores)),
            'domains': {d: {'class_miou': float(s), 'delta_to_FoRIS_pp': float(s - b)}
                        for d, s, b in zip(domains, scores, base_scores)}}


def score(histogram_path, batch_size):
    started = time.monotonic()
    config = lib.read(OUT / 'config.json')
    assert lib.sha(OUT / 'natural_truth_tables.npy') == config['truth_tables_sha256']
    assert lib.sha(lib.OUT / 'natural_bindings.json') == config['natural_bindings_sha256']
    tables = np.load(OUT / 'natural_truth_tables.npy', mmap_mode='r')
    hist = lib.read(histogram_path)
    assert hist['bit_order_LSB_first'] == list(lib.SYMBOLS)
    rows = hist['records']
    assert len(rows) == config['n_cases']
    frozen = {r['index']: r for r in lib.read(config['natural_bindings_path'])}
    for row in rows:
        b = frozen[row['index']]
        assert (row['dataset'], str(row['fold']), str(row['class_id'])) == (b['dataset'], str(b['fold']), str(b['class_id']))
        assert Path(row['query_photo']).resolve() == Path(b['query_photo']).resolve()
        row['query_photo'] = b['query_photo']
        for symbol in lib.SYMBOLS:
            a, c = row['prediction_sources'][symbol], b['sources'][symbol]
            assert (a['path'], a['mask_key'], a['file_sha256']) == (c['path'], c['key'], c['sha256'])
    domains = list(dict.fromkeys(r['dataset'] for r in rows))
    slots = sorted({(r['dataset'], str(r['fold']), str(r['class_id'])) for r in rows})
    si = {v: k for k, v in enumerate(slots)}
    fg, bg = np.zeros((len(slots), 64)), np.zeros((len(slots), 64))
    for r in rows:
        k = si[(r['dataset'], str(r['fold']), str(r['class_id']))]
        fg[k] += r['fg_count']
        bg[k] += r['bg_count']
    gt = fg.sum(1)
    slot_weights = np.zeros((len(slots), len(domains)))
    for col, d in enumerate(domains):
        selected = [k for k, slot in enumerate(slots) if slot[0] == d]
        slot_weights[selected, col] = 100 / len(selected)
    shifts = np.arange(64, dtype=np.uint64)
    source_tables = lib.source_truth_tables()
    source_lookup = np.array([[(table >> p) & 1 for p in range(64)]
                              for table in source_tables.values()], np.float64)
    source_scores = ((source_lookup @ fg.T) / np.maximum(gt[None, :] + source_lookup @ bg.T, 1)) @ slot_weights
    base_scores = source_scores[0]
    macros = np.lib.format.open_memmap(OUT / 'all_macro_scores.npy', mode='w+', dtype=np.float64, shape=(len(tables),))
    deps_file = np.lib.format.open_memmap(OUT / 'all_dependencies.npy', mode='w+', dtype=np.uint8, shape=(len(tables),))
    top = []
    best_dep = {}
    for start in range(0, len(tables), batch_size):
        stop = min(start + batch_size, len(tables))
        chunk = np.asarray(tables[start:stop])
        lookup = ((chunk[:, None] >> shifts[None, :]) & 1).astype(np.float64)
        intersection = lookup @ fg.T
        union = lookup @ bg.T
        union += gt[None, :]
        intersection /= np.maximum(union, 1)
        values = intersection @ slot_weights
        averages = values.mean(1)
        dependencies = dependency_masks(chunk)
        macros[start:stop] = averages
        deps_file[start:stop] = dependencies
        candidates = np.argpartition(averages, -min(64, len(averages)))[-64:]
        for local in candidates:
            record = (float(averages[local]), -int(dependencies[local]).bit_count(), -int(start + local), values[local].tolist())
            if len(top) < 64:
                heapq.heappush(top, record)
            elif record[:3] > top[0][:3]:
                heapq.heapreplace(top, record)
        for dep in np.unique(dependencies):
            selected = np.flatnonzero(dependencies == dep)
            local = selected[np.argmax(averages[selected])]
            record = (float(averages[local]), int(start + local), values[local].tolist())
            if int(dep) not in best_dep or record[0] > best_dep[int(dep)][0]:
                best_dep[int(dep)] = record
        if start == 0 or (start // batch_size) % 64 == 0 or stop == len(tables):
            progress = {'phase': 'complete_network_search', 'completed': stop, 'n': len(tables),
                        'seconds': round(time.monotonic() - started, 2), 'best_macro': max(r[0] for r in top)}
            lib.write(OUT / 'progress.json', progress)
            print(progress, flush=True)
    macros.flush(); deps_file.flush()
    top.sort(key=lambda r: r[:3], reverse=True)
    candidates = [summary_candidate(-r[2], int(tables[-r[2]]), np.asarray(r[3]), domains, base_scores) for r in top]
    dependency_candidates = {''.join(s for bit, s in enumerate(lib.SYMBOLS) if dep & (1 << bit)):
        summary_candidate(r[1], int(tables[r[1]]), np.asarray(r[2]), domains, base_scores)
        for dep, r in best_dep.items()}
    frontier = [v for v in dependency_candidates.values() if not any(
        set(other['essential_sources']) < set(v['essential_sources'])
        and other['macro10_class_miou'] >= v['macro10_class_miou'] - 1e-12
        for other in dependency_candidates.values())]
    frontier.sort(key=lambda r: -r['macro10_class_miou'])
    prior = lib.read(lib.OUT / 'report.json')
    assert candidates[0]['macro10_class_miou'] >= prior['best']['macro10_class_miou'] - 1e-9
    result = {'state': 'COMPLETE_EXPOSED_DEVELOPMENT_NETWORK_SEARCH',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'n_cases': len(rows),
        'n_rules_scored': len(tables), 'operation_count_limit': None,
        'config_sha256': lib.sha(OUT / 'config.json'),
        'histogram_path': str(histogram_path.resolve()), 'histogram_sha256': lib.sha(histogram_path),
        'source_sha256': lib.sha(__file__), 'score_seconds': time.monotonic() - started,
        'baselines': {s: {'macro10_class_miou': float(source_scores[k].mean()),
                            'domains': dict(zip(domains, source_scores[k].tolist()))}
                      for k, s in enumerate(lib.SYMBOLS)},
        'best': candidates[0], 'top64': candidates, 'best_per_exact_dependency': dependency_candidates,
        'score_dependency_frontier': frontier,
        'new_confirmation_done': False, 'selected_pipeline_latency_verified': False,
        'exposure': 'All1360 exposed development cases. This is discrete operation-program selection, not held-out evidence.',
        'scope': config['scope'], 'training': config['training']}
    lib.write(OUT / 'report.json', result)
    print({'state': result['state'], 'best': result['best'],
           'score_seconds': result['score_seconds']}, flush=True)


def apply_network(table, sources):
    result = np.zeros_like(sources['B'])
    for term in minimal_terms(table):
        conjunction = np.full_like(result, 255)
        for bit, symbol in enumerate(lib.SYMBOLS):
            if term & (1 << bit):
                conjunction &= sources[symbol]
        result |= conjunction
    return result


def materialize(hex_codes):
    config = lib.read(OUT / 'config.json')
    assert lib.sha(lib.OUT / 'natural_bindings.json') == config['natural_bindings_sha256']
    destination = OUT / 'materialized' / '_'.join(hex_codes)
    destination.mkdir(parents=True, exist_ok=True)
    tables = {code: int(code, 16) for code in hex_codes}
    records = []
    for b in lib.read(config['natural_bindings_path']):
        sources = lib.load_sources(b)
        masks = {code: apply_network(table, sources) for code, table in tables.items()}
        path = destination / (b['index'] + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                assert all(np.array_equal(old[k], v) for k, v in masks.items())
        else:
            np.savez_compressed(path, **masks)
        records.append({'index': b['index'], 'path': str(path), 'sha256': lib.sha(path)})
    lib.write(destination / 'sealed.json', {'state': 'SELECTED_NETWORK_MASKS_MATERIALIZED',
        'n': len(records), 'records': records, 'rules': {code: expression(table) for code, table in tables.items()},
        'config_sha256': lib.sha(OUT / 'config.json'), 'query_GT_in_inference': False})
    print({'state': 'MATERIALIZED', 'n': len(records), 'path': str(destination)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'score', 'materialize'))
    parser.add_argument('--histogram', type=Path, default=lib.ASSETS / 'a/x1012s2_error_overlap/joint6_1360.json')
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--hex-codes', nargs='+')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze()
    elif args.mode == 'score':
        score(args.histogram, args.batch_size)
    else:
        materialize(args.hex_codes)
