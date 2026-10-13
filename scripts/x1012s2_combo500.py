"""Two frozen full Boolean compositions on exposed R5, with natural T replay.

Freeze rules and source identities; produce and seal all 500 T and 1000
combination masks before a separate GT scoring call. No fitting or encoding.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent/'cv_data'
OUT = ASSETS/'a/x1012s2_combo500'
LIB = ASSETS/'a/x1012s2_composition_search/boolean_v1'
R1 = ASSETS/'a/x1012_R1'
R5C = ASSETS/'a/x1012_R5_candidate'
R5F = R5C/'final_tau0475'
R5B = ASSETS/'a/x1012_R5_validation'
R5S = ASSETS/'a/x1012_R5_results'
INS = ASSETS/'a/x1012_R6_insid3/r5_500'
RULE_IDS = ('r00097', 'r05681')
SYMBOLS = ('B', 'M', 'U', 'T', 'I', 'J')
ARM = {'B': 'FoRIS', 'M': 'MEAN', 'U': 'UN_midrange', 'T': 'T',
       'I': 'INS_bilinear', 'J': 'INS_crf640'}
BASELINES = ('FoRIS', 'MEAN', 'UN_midrange')
EDITS = ('recovered_TP', 'added_FP', 'deleted_TP', 'removed_FP')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def helpers():
    modules = {}
    for name, source in [('lab', R1/'frozen/3_lab.py'), ('plan_common', R1/'frozen/1_plan_common.py'),
                         ('plan_report', R1/'frozen/2_plan_report.py')]:
        target = OUT/'runtime'/(name+'.py')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        assert source.read_bytes() == target.read_bytes()
        spec = importlib.util.spec_from_file_location(name, target)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if name == 'lab':
            module.ROOT, module.ASSETS = ROOT, ASSETS
            module.DEV = torch.device('cpu')
        modules[name] = module
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return modules['lab'], modules['plan_common'], modules['plan_report']


lab, pc, pr = helpers()


def apply(rule, sources):
    result = sources['B'].copy()
    for op, symbol in rule['operations']:
        if op == 'add':
            result |= sources[symbol]
        elif op == 'retain':
            result &= sources[symbol]
        else:
            raise ValueError(op)
    return result


def prepare():
    for folder in ('predictions', 'records', 'frozen'):
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    path = OUT/'config.json'
    if not path.exists():
        library_config = read(LIB/'config.json')
        assert sha(LIB/'rules.json') == library_config['rules_sha256']
        selected = {r['id']: r for r in read(LIB/'rules.json') if r['id'] in RULE_IDS}
        assert tuple(selected) == RULE_IDS
        assert selected['r00097']['expression'] == '(((B|M)|T)&U)'
        truth = {s: ((np.arange(64) >> j) & 1).astype(np.uint8) for j, s in enumerate(SYMBOLS)}
        for rule in selected.values():
            actual = apply(rule, truth)
            expected = np.array([(int(rule['truth_table_hex'], 16) >> j) & 1 for j in range(64)], dtype=np.uint8)
            assert np.array_equal(actual, expected)
        rows = read(R5B/'manifest.baselines.json')
        old_u = {r['index']: r for r in read(R5F/'sealed.json')['records']}
        fields = {r['index']: r for r in read(R5C/'sealed.json')['records']}
        old_ins = {r['index']: r for r in read(INS/'sealed.json')['records']}
        assert len(rows) == len(old_u) == len(fields) == len(old_ins) == 500
        bindings = []
        for b in rows:
            key = pc.key(b)
            sources = {}
            for symbol, source_arm in (('B', 'B'), ('M', 'G')):
                s = b['old_arms'][source_arm]
                sources[symbol] = {'path': s['path'], 'key': s['cli_key'], 'sha256': s['sha256']}
            u, field, ins = old_u[key], fields[key], old_ins[key]
            sources['U'] = {'path': u['prediction_path'], 'key': 'UN_midrange', 'sha256': u['prediction_sha256']}
            for symbol, actual in (('I', 'insid3.bilinear'), ('J', 'insid3.crf640')):
                sources[symbol] = {'path': ins['prediction_path'], 'key': 'cli/'+actual, 'sha256': ins['prediction_sha256']}
            bindings.append({'index': key, 'dataset': b['dataset'], 'sources': sources,
                             'T_field_path': field['field_path'], 'T_field_sha256': field['field_sha256'],
                             'T_field_key': 'c32', 'T_output_key': 'T', 'binding': b})
        write(OUT/'bindings.json', bindings)
        write(OUT/'rules.json', selected)
        dependencies = [LIB/'config.json', LIB/'rules.json', R5B/'manifest.baselines.json',
                        R5B/'baseline/sealed.json', R5F/'sealed.json', R5C/'sealed.json', INS/'sealed.json']
        config = {
            'state': 'TWO_RULES_AND_T_REPLAY_FROZEN_BEFORE_500_NATURAL_OUTPUTS',
            'n': 500, 'primary_rule': 'r00097', 'fixed_six_source_control': 'r05681', 'rules': selected,
            'selection': 'Both exact rule IDs requested after 1360 exposed-development closure search; no selection or modification using this R5 replay.',
            'exposure': 'R5 500 already exposed earlier; this is development reproduction, not a new blind test.',
            'T': 'saved R5 c32 float32 -> reshape64x64 -> unchanged pc.readout: bilinear1024 align_corners=False -> minmax -> strict>.5; all500, no additional encoder or parameter fitting',
            'U': 'existing frozen R5 UN_midrange with original .5, not final .475',
            'prediction_format': 'np.packbits 1024x1024; existing B/M/U/I/J used byte-identically; T and both exact operation chains generated before query GT scoring',
            'all500_sealed_before_score': True, 'reference_count': 1, 'new_encoder_calls': 0,
            'training': False, 'parameter_search': False, 'domain_branch': False,
            'metric': 'observed (fold,class) pooled I/U then equal slots; ten-domain unweighted macro',
            'paired_CI': '1000 query-photo cluster resamples independently within each domain, observed slots in each draw; seed20261013; same draws for all arms; macro averages ten sampled domain scores',
            'CI_scope': 'Descriptive fixed-rule paired intervals on exposed data; no claim of blind confirmation or correction for earlier development selection.',
            'resources': {'device': 'cpu', 'workers': 1, 'threads': 1},
            'source_sha256': sha(__file__), 'source_parent_sha256': {str(p): sha(p) for p in dependencies},
            'helper_sha256': {str(OUT/'runtime'/(n+'.py')): sha(OUT/'runtime'/(n+'.py')) for n in ('lab', 'plan_common', 'plan_report')},
            'bindings_sha256': sha(OUT/'bindings.json'), 'rules_sha256': sha(OUT/'rules.json'),
        }
        shutil.copyfile(__file__, OUT/'frozen/x1012s2_combo500.py')
        write(path, config)
    config = read(path)
    assert sha(__file__) == config['source_sha256'], 'Frozen source changed'
    assert all(sha(p) == h for p, h in config['source_parent_sha256'].items()), 'Parent input changed'
    assert all(sha(p) == h for p, h in config['helper_sha256'].items()), 'Helper drift'
    assert sha(OUT/'bindings.json') == config['bindings_sha256']
    assert sha(OUT/'rules.json') == config['rules_sha256']
    return read(OUT/'bindings.json'), config


def load_existing(row):
    by_path, sources = defaultdict(list), {}
    for symbol, entry in row['sources'].items():
        by_path[entry['path']].append((symbol, entry))
    for path, entries in by_path.items():
        assert sha(path) == entries[0][1]['sha256']
        with np.load(path, allow_pickle=False) as archive:
            for symbol, entry in entries:
                sources[symbol] = archive[entry['key']].copy()
    assert all(v.dtype == np.uint8 and v.size == 1024*1024//8 for v in sources.values())
    return sources


def produce():
    rows, config = prepare()
    assert not (OUT/'sealed.json').exists(), 'Natural outputs already sealed; do not overwrite'
    began, records = time.monotonic(), []
    for i, row in enumerate(rows, 1):
        key = row['index']
        target = OUT/'predictions'/(key+'.npz')
        assert not target.exists(), 'Preserve partial natural output: '+str(target)
        start = time.monotonic()
        with pc.inference_only(row['binding']):
            sources = load_existing(row)
            assert sha(row['T_field_path']) == row['T_field_sha256']
            with np.load(row['T_field_path'], allow_pickle=False) as z:
                c = z['c32'].copy()
                rho = float(z['reference_rho'][0])
            t0 = time.monotonic()
            sources['T'] = pc.pack(pc.readout(c))
            t_seconds = time.monotonic()-t0
            if rho > .9:
                assert np.array_equal(sources['T'], sources['U']), 'Existing raw-T fallback differs'
            t0 = time.monotonic()
            predictions = {'T': sources['T'], **{name: apply(rule, sources) for name, rule in config['rules'].items()}}
            rule_seconds = time.monotonic()-t0
            # The two frozen library rules differ only on state5: B=U=1,
            # M=T=I=J=0. This checks actual packed operation execution.
            expected = sources['B'] & sources['U'] & ~sources['M'] & ~sources['T'] & ~sources['I'] & ~sources['J']
            assert np.array_equal(predictions['r00097'] ^ predictions['r05681'], expected)
            assert not np.any(predictions['r05681'] & ~predictions['r00097'])
            np.savez_compressed(target, **predictions)
        receipt = {'index': key, 'dataset': row['dataset'], 'prediction_path': str(target),
                   'prediction_sha256': sha(target), 'T_field_path': row['T_field_path'],
                   'T_field_sha256': row['T_field_sha256'], 'reference_rho': rho,
                   'T_equals_U_fallback': rho > .9, 'new_encoder_calls': 0, 'query_GT_reads': 0,
                   'T_readout_seconds': t_seconds, 'packed_rule_seconds': rule_seconds,
                   'seconds': time.monotonic()-start}
        records.append(receipt)
        write(OUT/'records'/(key+'.json'), receipt)
        if i <= 2 or i % 100 == 0:
            print({'phase': 'natural-produce', 'completed': i, 'n': 500, 'seconds': round(time.monotonic()-began, 1)}, flush=True)
    seal = {'state': 'ALL500_T_AND_TWO_FIXED_COMPOSITIONS_SEALED', 'n': len(rows),
            'T_masks': 500, 'composition_masks': 1000, 'records': records,
            'config_sha256': sha(OUT/'config.json'), 'bindings_sha256': sha(OUT/'bindings.json'),
            'rules_sha256': sha(OUT/'rules.json'), 'new_encoder_calls': 0, 'query_GT_reads': 0,
            'wall_seconds': time.monotonic()-began, 'T_readout_seconds_sum': sum(r['T_readout_seconds'] for r in records),
            'packed_rule_seconds_sum': sum(r['packed_rule_seconds'] for r in records),
            'latency_scope': 'cached field/mask replay only, not complete pipeline latency'}
    write(OUT/'sealed.json', seal)
    print({'phase': 'natural-produce', 'state': seal['state'], 'n': 500, 'seal_sha256': sha(OUT/'sealed.json')}, flush=True)


def edit_counts(candidate, baseline, gt, valid):
    add = candidate & ~baseline & valid
    remove = baseline & ~candidate & valid
    at, dt = int((add & gt).sum()), int((remove & gt).sum())
    return [at, int(add.sum())-at, dt, int(remove.sum())-dt]


def bootstrap(records, arms, summary):
    rng = np.random.default_rng(20261013)
    output, samples = {}, {}
    for d in lab.ALL_DOMAINS:
        rows = [r for r in records if r['dataset'] == d]
        photos = sorted({r['query_photo'] for r in rows})
        slots = sorted({(str(r['fold']), str(r['class_id'])) for r in rows})
        pm, sm = {p: i for i, p in enumerate(photos)}, {s: i for i, s in enumerate(slots)}
        count = np.zeros((len(photos), len(slots)))
        iu = np.zeros((len(arms), 2, len(photos), len(slots)))
        for r in rows:
            p, s = pm[r['query_photo']], sm[(str(r['fold']), str(r['class_id']))]
            count[p, s] += 1
            for j, a in enumerate(arms):
                iu[j, :, p, s] += (r['metrics'][a]['I'], r['metrics'][a]['U'])
        weights = rng.multinomial(len(photos), np.full(len(photos), 1/len(photos)), size=1000)
        present = (weights @ count) > 0
        draws = {}
        for j, a in enumerate(arms):
            inter, union = weights @ iu[j, 0], weights @ iu[j, 1]
            draws[a] = ((100*inter/np.maximum(union, 1))*present).sum(1)/present.sum(1)
        samples[d] = draws
        output[d] = {}
        for rule in RULE_IDS:
            output[d][rule] = {}
            for base in BASELINES+('r00097',):
                delta = draws[rule]-draws[base]
                output[d][rule][base] = {'delta_pp': summary[d][rule]['class_miou']-summary[d][base]['class_miou'],
                    'paired_query_photo_bootstrap95_pp': np.quantile(delta, [.025, .975]).tolist()}
    macro = {}
    for rule in RULE_IDS:
        macro[rule] = {}
        for base in BASELINES+('r00097',):
            delta = np.mean([samples[d][rule]-samples[d][base] for d in lab.ALL_DOMAINS], axis=0)
            macro[rule][base] = {'delta_pp': float(np.mean([summary[d][rule]['class_miou']-summary[d][base]['class_miou'] for d in lab.ALL_DOMAINS])),
                'paired_query_photo_bootstrap95_pp': np.quantile(delta, [.025, .975]).tolist()}
    return {'domains': output, 'macro10': macro, 'seed': 20261013, 'repeats': 1000,
            'macro_definition': 'equal ten-domain mean; independent within-domain photo-cluster resampling'}


def score():
    rows, config = prepare()
    seal = read(OUT/'sealed.json')
    assert seal['n'] == 500 and seal['state'] == 'ALL500_T_AND_TWO_FIXED_COMPOSITIONS_SEALED'
    assert seal['config_sha256'] == sha(OUT/'config.json')
    assert seal['bindings_sha256'] == sha(OUT/'bindings.json')
    receipts = {r['index']: r for r in seal['records']}
    assert all(sha(r['prediction_path']) == r['prediction_sha256'] for r in receipts.values())
    old = {r['index']: r for r in read(R5S/'report.json')['records']}
    old_ins = {r['index']: r for r in read(INS/'episodes.json')}
    arms = tuple(ARM.values())+RULE_IDS
    began, records = time.monotonic(), []
    for i, row in enumerate(rows, 1):
        key = row['index']
        source = load_existing(row)
        predictions = {ARM[s]: pc.unpack(p) for s, p in source.items()}
        with np.load(receipts[key]['prediction_path'], allow_pickle=False) as z:
            predictions.update({a: pc.unpack(z[a]) for a in ('T',)+RULE_IDS})
        gt, valid = pc.truth(row['binding'])
        b = row['binding']
        record = {'index': key, 'dataset': b['dataset'], 'fold': b['row']['fold'],
                  'class_id': b['row']['global_class_id'],
                  'query_photo': str((ASSETS/b['row']['query_path']).resolve()),
                  'metrics': {a: pr.metric(predictions[a], gt, valid) for a in arms},
                  'edits': {a: {base: edit_counts(predictions[a], predictions[base], gt, valid)
                                for base in BASELINES} for a in RULE_IDS}}
        expected = {a: old[key]['metrics'][a] for a in BASELINES}
        expected.update(INS_bilinear=old_ins[key]['metrics']['insid3.bilinear'],
                        INS_crf640=old_ins[key]['metrics']['insid3.crf640'])
        for a, metric in expected.items():
            assert all(record['metrics'][a][k] == metric[k] for k in ('I', 'U', 'TP', 'FP', 'FN', 'GT', 'predicted')), (key, a)
        for a in RULE_IDS:
            m = record['metrics'][a]
            for base in BASELINES:
                bm = record['metrics'][base]
                at, af, dt, df = record['edits'][a][base]
                assert m['TP'] == bm['TP']+at-dt and m['FP'] == bm['FP']+af-df
        records.append(record)
        if i <= 2 or i % 100 == 0:
            print({'phase': 'exposed-score', 'completed': i, 'n': 500, 'seconds': round(time.monotonic()-began, 1)}, flush=True)
    summary = pr.summary(records, arms)
    paired = bootstrap(records, arms, summary)
    edits = {d: {a: {base: np.sum([r['edits'][a][base] for r in records if r['dataset'] == d], axis=0).tolist()
                       for base in BASELINES} for a in RULE_IDS} for d in lab.ALL_DOMAINS}
    result = {'state': 'COMPLETE_FIXED_RULE_EXPOSED500_REPRODUCTION', 'n': 500,
              'primary_rule': 'r00097', 'fixed_control_rule': 'r05681', 'rules': config['rules'],
              'summary': summary, 'macro10_class_miou': {a: float(np.mean([summary[d][a]['class_miou'] for d in lab.ALL_DOMAINS])) for a in arms},
              'paired_comparisons': paired, 'edit_columns': EDITS,
              'edit_columns_zh': ['补回TP', '加入FP', '删除TP', '移除FP'], 'edits': edits, 'records': records,
              'config_sha256': sha(OUT/'config.json'), 'seal_sha256': sha(OUT/'sealed.json'),
              'bindings_sha256': sha(OUT/'bindings.json'), 'score_seconds': time.monotonic()-began,
              'existing_controls_integer_identity': True, 'new_encoder_calls': 0, 'training': False,
              'new_blind_confirmation': False, 'post_score_rule_changes': False,
              'full_pipeline_latency_measured_here': False,
              'exposure': config['exposure']}
    write(OUT/'report.json', result)
    compact = {k: v for k, v in result.items() if k != 'records'}
    compact['report_path'], compact['report_sha256'] = str(OUT/'report.json'), sha(OUT/'report.json')
    write(OUT/'completion.json', compact)
    write(ROOT/'evidence/research/x1012s2_combo500.json', compact)
    lines = ['# 冻结组合在已曝光500配对上的完整复现', '',
             '**这是已曝光R5开发复现，不是新的盲测。两个规则在读本次GT前冻结；全部500原T与1000组合自然mask封存后才评分；评分后不改规则。**', '',
             '主规则 `r00097 = ((B|M|T)&U)`；固定六源对照 `r05681 = ((B&(I|J))|M|T)&U`。U是原UN(.5)，T只重放既有c32原读出。新增编码0、拟合0。', '',
             '|域|FoRIS|MEAN|UN(.5)|r00097|r05681|r00097−FoRIS|r00097−MEAN|r00097−UN|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d, table in summary.items():
        values = [table[a]['class_miou'] for a in BASELINES+RULE_IDS]
        values += [paired['domains'][d]['r00097'][a]['delta_pp'] for a in BASELINES]
        lines.append('|'+d+'|'+'|'.join(f'{x:.3f}' for x in values)+'|')
    values = [result['macro10_class_miou'][a] for a in BASELINES+RULE_IDS]
    values += [paired['macro10']['r00097'][a]['delta_pp'] for a in BASELINES]
    lines += ['|十域均值|'+'|'.join(f'{x:.3f}' for x in values)+'|', '',
              '|范围|规则|对照|差值pp|照片配对95%区间|', '|---|---|---|---:|---|']
    for scope, table in [*paired['domains'].items(), ('macro10', paired['macro10'])]:
        for a in RULE_IDS:
            for base in BASELINES+('r00097',):
                if a == base:
                    continue
                v = table[a][base]; lo, hi = v['paired_query_photo_bootstrap95_pp']
                lines.append(f"|{scope}|{a}|{base}|{v['delta_pp']:+.3f}|[{lo:+.3f},{hi:+.3f}]|")
    lines += ['', '四项增删顺序为补回TP、加入FP、删除TP、移除FP。逐域对三个基线均给完整整数：', '',
              '|域|规则|对照|补回TP|加入FP|删除TP|移除FP|', '|---|---|---|---:|---:|---:|---:|']
    for d, table in edits.items():
        for a in RULE_IDS:
            for base in BASELINES:
                lines.append('|'+d+'|'+a+'|'+base+'|'+'|'.join(str(x) for x in table[a][base])+'|')
    lines += ['', 'CI为固定规则在曝光数据上的描述性照片聚类bootstrap；没有校正此前1360选择。十域均值区间对各域分别重采照片后等权平均。',
              f'自然mask：`{OUT/"predictions"}`；封存SHA `{result["seal_sha256"]}`。',
              f'冻结配置SHA `{result["config_sha256"]}`；完整逐例评分 `{OUT/"report.json"}`。',
              '这里只测缓存读出/布尔操作耗时，不把它当完整方法延迟。',
              '[紧凑可复算结果](../../evidence/research/x1012s2_combo500.json)。']
    (ROOT/'docs/research/COMPOSITION_500_REPLAY_20261012.md').write_text('\n'.join(lines)+'\n')
    print({'phase': 'complete', 'n': 500, 'macro10_class_miou': result['macro10_class_miou'],
           'macro_paired': paired['macro10'], 'seconds': round(time.monotonic()-began, 1)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'produce', 'score'))
    args = parser.parse_args()
    if args.mode == 'freeze':
        rows, config = prepare()
        print({'state': config['state'], 'n': len(rows)}, flush=True)
    elif args.mode == 'produce':
        produce()
    else:
        score()
