"""Training-free add/delete composition closure over six frozen DINO pathways.

This module constructs natural prediction rules without opening query labels.
There is no operation-count limit: enumeration stops only at algebraic closure.
Ranking on exposed development data lives in a separate scoring script.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO.parent / 'cv_data'
OUT = ASSETS / 'a/x1012s2_composition_search/boolean_v1'
SYMBOLS = ('B', 'M', 'U', 'T', 'I', 'J')
NAMES = {'B': 'complete_FoRIS_CRF', 'M': 'MEAN_a025_l16',
         'U': 'R1_VP_UN_tau05', 'T': 'independent_soft_coverage_T',
         'I': 'INSID3_bilinear', 'J': 'INSID3_CRF640'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def case_key(binding):
    i = binding['index']
    return i if isinstance(i, str) else f'{i:06d}'


def source_truth_tables():
    return {symbol: sum(1 << pattern for pattern in range(64) if (pattern >> bit) & 1)
            for bit, symbol in enumerate(SYMBOLS)}


def essential_sources(table):
    return [symbol for bit, symbol in enumerate(SYMBOLS)
            if any(((table >> p) & 1) != ((table >> (p ^ (1 << bit))) & 1)
                   for p in range(64))]


def enumerate_rules():
    columns = source_truth_tables()
    queue = deque([columns['B']])
    seen = {columns['B']: {'expression': 'B', 'operations': []}}
    ordered = []
    while queue:
        current = queue.popleft()
        path = seen[current]
        ordered.append({'id': f'r{len(ordered):05d}', 'truth_table_hex': f'{current:016x}',
                        'essential_sources': essential_sources(current), **path})
        for symbol in SYMBOLS:
            for op, result, token in (('add', current | columns[symbol], '|'),
                                      ('retain', current & columns[symbol], '&')):
                if result in seen:
                    continue
                seen[result] = {'expression': f'({path["expression"]}{token}{symbol})',
                                'operations': path['operations'] + [[op, symbol]]}
                queue.append(result)
    # Every successor is represented; closure, not a manually chosen depth.
    assert all((t | c) in seen and (t & c) in seen for t in seen for c in columns.values())
    source_ids = {symbol: next(r['id'] for r in ordered
                              if int(r['truth_table_hex'], 16) == table)
                  for symbol, table in columns.items()}
    return ordered, source_ids


def freeze():
    if (OUT / 'config.json').exists():
        config = read(OUT / 'config.json')
        assert config['library_source_sha256'] == sha(__file__)
        print({'state': config['state'], 'n_rules': config['n_rules'], 'already_frozen': True})
        return
    rules, source_ids = enumerate_rules()
    r1 = ASSETS / 'a/x1012_R1'
    control = ASSETS / 'a/x1012_B4/seed150/controls'
    insid3 = ASSETS / 'a/x1012_R6_insid3/development1360'
    manifest_path = r1 / 'manifest.json'
    rows = read(manifest_path)
    seals = {name: read(path / 'sealed.json')
             for name, path in (('R1', r1), ('controls', control), ('INSID3', insid3))}
    maps = {name: {r['index']: r for r in seal['records']} for name, seal in seals.items()}
    keys = {case_key(b) for b in rows}
    assert len(rows) == len(keys) == 1360
    assert all(set(records) == keys for records in maps.values())
    bindings = []
    for binding in rows:
        key = case_key(binding)
        r, c, i = (maps[name][key] for name in ('R1', 'controls', 'INSID3'))
        sources = {}
        for symbol, record, field, path_key, hash_key in (
            ('B', c, 'FoRIS', 'path', 'sha256'),
            ('M', c, 'MEAN', 'path', 'sha256'),
            ('U', r, 'VP_UN', 'prediction_path', 'prediction_sha256'),
            ('T', r, 'T', 'prediction_path', 'prediction_sha256'),
            ('I', i, 'cli/insid3.bilinear', 'prediction_path', 'prediction_sha256'),
            ('J', i, 'cli/insid3.crf640', 'prediction_path', 'prediction_sha256')):
            sources[symbol] = {'path': record[path_key], 'sha256': record[hash_key], 'key': field}
        bindings.append({'index': key, 'dataset': binding['dataset'],
                         'fold': binding['row']['fold'],
                         'class_id': binding['row']['global_class_id'],
                         'query_photo': str((ASSETS / binding['row']['query_path']).resolve()),
                         'reference_rho': r['reference_rho'], 'sources': sources})
    # Verify source-key spelling on actual natural assets before freezing.
    for symbol, spec in bindings[0]['sources'].items():
        assert sha(spec['path']) == spec['sha256']
        with np.load(spec['path'], allow_pickle=False) as z:
            assert z[spec['key']].size == 1024 * 1024 // 8, (symbol, z.files)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'rules.json', rules)
    write(OUT / 'natural_bindings.json', bindings)
    shutil.copyfile(__file__, OUT / 'library_source.py')
    config = {'state': 'ALL_NATURAL_BOOLEAN_RULES_FROZEN_BEFORE_COMBINATION_SCORING',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'training': False, 'new_encoder_calls_for_this_replay': 0,
              'source_symbols_in_low_to_high_bit_order': SYMBOLS, 'source_names': NAMES,
              'initial_state': 'B',
              'operators': {'add(X)': 'mask OR X: select missed pixels supported by X',
                            'retain(X)': 'mask AND X: remove pixels rejected by X'},
              'operation_count_limit': None, 'enumeration_stop': 'exact algebraic closure',
              'n_rules': len(rules), 'largest_shortest_expression_operations':
                  max(len(r['operations']) for r in rules),
              'source_rule_ids': source_ids, 'n_cases': len(rows),
              'development_exposure': 'All1360 already exposed; architecture selection only, never new validation.',
              'selection': 'Rank complete compositions by equal-weight ten-domain class mIoU; retain domain-level tradeoffs and dependency/latency Pareto comparisons. No individual-step gain requirement.',
              'latency': 'No depth constraint. Measure full selected pipeline with shared dependencies, relative to same-condition FoRIS; user example2s-to5s is a total-cost reference. Boolean mask lookup time alone is not pipeline latency.',
              'source_parent_seals': {str(path / 'sealed.json'): sha(path / 'sealed.json')
                  for path in (r1, control, insid3)},
              'manifest_path': str(manifest_path), 'manifest_sha256': sha(manifest_path),
              'library_source_sha256': sha(__file__), 'rules_sha256': sha(OUT / 'rules.json'),
              'natural_bindings_sha256': sha(OUT / 'natural_bindings.json')}
    write(OUT / 'config.json', config)
    print({'state': config['state'], 'n_rules': len(rules), 'n_cases': len(rows),
           'operation_count_limit': None, 'config_sha256': sha(OUT / 'config.json')}, flush=True)


def apply_rule(rule, sources):
    # Execute the original add/delete operations, not the scorer's truth table.
    result = sources['B'].copy()
    for operator, symbol in rule['operations']:
        if operator == 'add':
            result |= sources[symbol]
        elif operator == 'retain':
            result &= sources[symbol]
        else:
            raise ValueError(operator)
    return result


def load_sources(binding):
    by_path, sources = {}, {}
    for symbol, spec in binding['sources'].items():
        if spec['path'] not in by_path:
            assert sha(spec['path']) == spec['sha256']
            with np.load(spec['path'], allow_pickle=False) as z:
                by_path[spec['path']] = {k: z[k].copy() for k in z.files}
        sources[symbol] = by_path[spec['path']][spec['key']]
    return sources


def materialize(rule_ids):
    config = read(OUT / 'config.json')
    assert sha(OUT / 'rules.json') == config['rules_sha256']
    assert sha(OUT / 'natural_bindings.json') == config['natural_bindings_sha256']
    assert sha(__file__) == config['library_source_sha256']
    rules = {r['id']: r for r in read(OUT / 'rules.json')}
    selected = {k: rules[k] for k in rule_ids}
    destination = OUT / 'materialized' / ('_'.join(rule_ids))
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for b in read(OUT / 'natural_bindings.json'):
        sources = load_sources(b)
        masks = {key: apply_rule(rule, sources) for key, rule in selected.items()}
        path = destination / (b['index'] + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                assert all(np.array_equal(old[k], v) for k, v in masks.items())
        else:
            np.savez_compressed(path, **masks)
        records.append({'index': b['index'], 'prediction_path': str(path),
                        'prediction_sha256': sha(path)})
    write(destination / 'sealed.json', {'state': 'SELECTED_RULE_MASKS_MATERIALIZED',
         'n': len(records), 'rules': selected, 'records': records,
         'config_sha256': sha(OUT / 'config.json'), 'query_GT_in_inference': False,
         'training': False, 'new_encoder_calls': 0})
    print({'state': 'MATERIALIZED', 'path': str(destination), 'n': len(records), 'rules': rule_ids})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'materialize'))
    parser.add_argument('--rules', nargs='+')
    args = parser.parse_args()
    freeze() if args.mode == 'freeze' else materialize(args.rules)
