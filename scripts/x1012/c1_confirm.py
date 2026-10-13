"""C1: freeze one scored B4 arm, produce all cached confirm masks, then score.

Selection reads explore evidence only. Confirm GT is opened exclusively by the
score command, after natural predictions and bound FoRIS baselines are sealed.
There is one immutable candidate per C1 directory; scoring never changes it.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import numpy as np
import lab
import plan_common as pc
import plan_report as pr
import b4_minimal as b4
import a2_exchange as workflow
import c1_inference as ci

OUT = lab.ASSETS / 'a/x1012_C1'
P0 = lab.ASSETS / 'a/x1012_P0'
CONFIRM = P0 / 'confirm.json'
ARMS = ('FoRIS', 'candidate')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def input_identity(binding):
    return {k: binding[k] for k in ('index', 'dataset', 'row', 'fit', 'split', 'pair_signature') if k in binding}


def choose_candidate(report, arm=None):
    assert report['state'] == 'COMPLETE' and report['task'] in (
        'B4', 'B4_seed150', 'B4_graph', 'B4_graph_seed150',
        'B4_modulated_region', 'B4_modulated_region_seed150',
        'B4_modulated_graph', 'B4_modulated_graph_seed150', 'B4_assembly', 'B4_assembly_seed150')
    assert set(report['summary']) == set(lab.ALL_DOMAINS), 'B4 selection needs its ten-domain table'
    method = report.get('extra', {}).get('method', {})
    basis = method.get('decision_basis', 'region_mean')
    modulation = method.get('evidence_modulation')
    assert basis in ('region_mean', 'K64_region_mean', 'reciprocal_graph')
    assert modulation in (None, 'A4_max_J_continuous')
    if report['task'].startswith(('B4_graph', 'B4_modulated_graph')):
        assert basis == 'reciprocal_graph', 'Graph report must declare its graph basis'
    if report['task'].startswith('B4_modulated'):
        assert modulation == 'A4_max_J_continuous', 'Modulated report must declare its evidence'
    definitions = method['candidates']
    names = [a for a in report['arms'] if a in definitions and not definitions[a].get('common_crf', a.endswith('_crf'))]
    assert names
    stats = {}
    for name in names:
        delta = {d: report['summary'][d][name]['delta_to_foris_pp'] for d in lab.ALL_DOMAINS}
        stats[name] = {'worst_domain_delta_pp': min(delta.values()),
                       'equal_domain_mean_delta_pp': float(np.mean(list(delta.values()))),
                       'FoRIS_stage_count': definitions[name]['FoRIS_stage_count'],
                       'per_domain_delta_pp': delta, 'ten_domains_ge_minus_05': all(v >= -.5 for v in delta.values())}
    if arm is None:
        arm = max(names, key=lambda a: (stats[a]['worst_domain_delta_pp'],
                                        stats[a]['equal_domain_mean_delta_pp'], -stats[a]['FoRIS_stage_count']))
        selection = 'max worst-domain delta; tie max ten-domain equal mean; tie fewer stages; no automatic CRF'
    else:
        assert arm in report['arms'] and (arm in definitions or arm.removesuffix('_crf') in definitions), 'Arm must have been scored in B4'
        selection = 'root-specified fixed scored arm'
    definition = definitions.get(arm, definitions.get(arm.removesuffix('_crf')))
    base = definition.get('base_arm', arm.removesuffix('_crf'))
    delta = {d: report['summary'][d][arm]['delta_to_foris_pp'] for d in lab.ALL_DOMAINS}
    candidate = {'kind': 'b4', 'source_arm': arm, 'base_arm': base, 'decision_basis': basis,
                 'evidence_modulation': modulation,
                 'lineage_source_arm': definition.get('source_arm', base),
                 'underlying_source_arm': definition.get('underlying_source_arm', definition.get('source_arm', base)),
                 'active_score_terms': list(definition['active_score_terms']),
                 'dependencies': list(definition['computed_dependencies']),
                 'FoRIS_stage_count': definition['FoRIS_stage_count'],
                 'common_crf': definition.get('common_crf', arm.endswith('_crf')), 'dataset_branch': False,
                 'ten_domains_ge_minus_05_in_explore': all(v >= -.5 for v in delta.values()),
                 'explore_delta_pp': delta, 'selection_rule': selection}
    assert candidate['FoRIS_stage_count'] <= 2
    assert candidate['FoRIS_stage_count'] == len(candidate['dependencies'])
    if 'reference_rho_gate' in definition:
        assert definition['reference_rho_gate'] == .9 and definition['reference_rho_fallback'] == 'raw_T'
        candidate.update(reference_rho_gate=.9, reference_rho_fallback='raw_T',
                         triggered_FoRIS_stage_count=0, triggered_common_crf=False)
    return candidate, stats


def evidence_paths():
    choices = {
        'A1': [lab.ASSETS / 'a/x1012_A1/report.json'],
        'A2': [lab.ASSETS / 'a/x1012_A2/explore/report.json', lab.ASSETS / 'a/x1012_A2/seed150/report.json'],
        'A6': [lab.ASSETS / 'a/x1012_A6/holdout_report/report.json', lab.ASSETS / 'a/x1012_A6/upper_report/report.json'],
        'B2': [lab.ASSETS / 'a/x1012_B2/report.json'], 'B3': [lab.ASSETS / 'a/x1012_B3/report.json'],
        'A7': [lab.ASSETS / 'a/x1012_A7/explore/report.json', lab.ASSETS / 'a/x1012_A7/seed150/report.json'],
    }
    return {name: next((p for p in paths if p.is_file()), None) for name, paths in choices.items()}


def freeze(args):
    pc.setup('cpu')
    if (OUT / 'config.json').exists():
        config = pc.read(OUT / 'config.json')
        if args.arm is not None:
            assert args.arm == config['candidate']['source_arm'], 'C1 candidate is immutable; no refit/reselection'
        check_freeze(config)
        print({'task': 'C1', 'state': 'ALREADY_FROZEN', 'candidate': config['candidate']['source_arm']}, flush=True)
        return
    report_path = args.source_report.resolve()
    report = pc.read(report_path)
    candidate, comparisons = choose_candidate(report, args.arm)
    source_method = report['extra']['method']
    assert pc.sha(lab.BASIS) == source_method['basis_sha256'], 'Scored B4 positional basis changed'
    for path, sha in source_method['source_sha256'].items():
        assert pc.sha(path) == sha, 'Scored B4 source changed: ' + path
    rows = pc.read(CONFIRM)
    assert rows and all(b.get('split') == 'confirm' for b in rows)
    assert len({pc.key(b) for b in rows}) == len(rows) and set(b['dataset'] for b in rows) == set(lab.ALL_DOMAINS)
    for name in ('frozen', 'predictions', 'fields', 'records'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    source_files = (Path(__file__), Path(ci.__file__), Path(ci.a7.__file__), Path(ci.a4m.__file__), Path(b4.__file__), Path(workflow.__file__),
                    Path(pc.__file__), Path(pr.__file__), Path(lab.__file__), Path(ci.raw_core.__file__),
                    lab.ROOT / 'scripts/raw_feature_cache.py', lab.BASIS,
                    lab.ASSETS / 'third_party/foris_official/models/foris.py',
                    lab.ASSETS / 'third_party/foris_official/utils/data.py',
                    lab.ASSETS / 'third_party/foris_official/utils/clustering.py',
                    lab.ASSETS / 'third_party/foris_official/utils/refinement.py')
    for i, path in enumerate(source_files):
        shutil.copyfile(path, OUT / 'frozen' / f'{i}_{path.name}')
    shutil.copyfile(report_path, OUT / 'frozen/B4_explore_report.json')
    evidence = {'B4': {'path': str(report_path), 'sha256': pc.sha(report_path),
                       'n': report['n'], 'task': report['task'], 'cohort': report['extra'].get('cohort')}}
    for name, path in evidence_paths().items():
        if path is not None:
            shutil.copyfile(path, OUT / 'frozen' / (name + '_evidence.json'))
            evidence[name] = {'path': str(path), 'sha256': pc.sha(path), 'available_at_freeze': True}
        else:
            evidence[name] = {'available_at_freeze': False}
    pc.write(OUT / 'manifest.json', rows)
    config = {'state': 'FROZEN_BEFORE_CONFIRM_GT', 'task': 'C1', 'candidate': candidate,
              'n': len(rows), 'per_domain_n': dict(Counter(b['dataset'] for b in rows)),
              'confirm_manifest_at_freeze_path': str(CONFIRM), 'confirm_manifest_at_freeze_sha256': pc.sha(CONFIRM),
              'confirm_input_digest': digest([input_identity(b) for b in rows]),
              'manifest_sha256': pc.sha(OUT / 'manifest.json'), 'source_B4_method': source_method,
              'explore_evidence': evidence, 'explore_comparisons': comparisons,
              'constants': {'kernel_temperature': .05, 'shared_center': '.5*(mean(R)+mean(Q))',
                            'regions': 'P0 K64, 6 iterations, same deterministic initialization',
                            'decision_basis': candidate['decision_basis'],
                            'evidence_modulation': candidate['evidence_modulation'],
                            'modulation_grid': list(ci.a4m.STRENGTHS),
                            'modulation_rule': 'original A4 maxJ; FP16 sim_max/back_f; FP32 ind0/raw-novelty indn endpoint masks; lower-s ties; B/V/P/K still use original c',
                            'reference_rho_gate': candidate.get('reference_rho_gate'),
                            'reference_rho_rule': 'rho=1-sum(y*(y>=.9))/sum(y); strict>.9 -> raw T readout(original c), skip maxJ/graph/FoRIS/CRF',
                            'readout': 'selected region/graph field -> bilinear1024 -> minmax -> >.5',
                            'graph': {'input': 'frozen primary c32 or c_mod32 + sum(active additions)', 'lambda': 16,
                                      'neighbors': 20, 'neighbor_method': 'CPU exact unit(q) cosine top20 excluding self, block256',
                                      'confidence': 'A7 s=minmax(input); A=(.1+abs(2*s-1))/mean(.1+abs(2*s-1))',
                                      'rank_correction': False},
                            'background': .55, 'vote': .20, 'prior': .25, 'tau': .6},
              'recomputed_confirm_fields': 'FP32 P0 c/IND0/K64 formula from existing O24 on CPU; no P0 confirm field cache required',
              'CPU_MPS_scope': 'same fixed formula; no cross-device pixel-equality gate',
              'baseline_policy': 'complete P0 FoRIS sealed and bound before score; never consumed by method',
              'workers': 1, 'threads': 4, 'source_sha256': {str(p): pc.sha(p) for p in source_files},
              'new_encoder_calls': 0, 'confirm_GT_read_at_freeze': False, 'post_score_reselection_allowed': False}
    pc.write(OUT / 'config.json', config)
    path = lab.ROOT / 'docs/research/x1012_tasks/C1-freeze.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# C1 候选冻结', '', f"固定候选：{candidate['source_arm']}；决策基座：{candidate['decision_basis']}；证据调制：{candidate['evidence_modulation']}；FoRIS依赖{candidate['FoRIS_stage_count']}步；公共CRF：{candidate['common_crf']}。",
             f"选择规则：{candidate['selection_rule']}。开发集十域均≥−0.5：{candidate['ten_domains_ge_minus_05_in_explore']}。",
             f"固定参考rho回退：{candidate.get('reference_rho_gate')}；仅严格超过阈值时直接原T，跳过所有修正/maxJ/graph/CRF。",
             '未达开发门槛时也只确认一次最接近的固定组合，不冒称已达到独立基座目标。确认集不参与选择。',
             f"确认清单：{len(rows)}例；各域例数：{config['per_domain_n']}。",
             f"confirm原文件SHA256：{config['confirm_manifest_at_freeze_sha256']}；自然输入digest：{config['confirm_input_digest']}。",
             '允许P0随后只绑定派生FoRIS基线；R/Q/掩码路径与已有O24身份、完整确认队列保持冻结。',
             '推理CPU单进程4线程，只重算同一独立证据与所选0/1/2步。没有DINO模型、编码、窗口或域条件选择。全部自然掩码封存后才评分。',
             '确认评分后禁止换候选、改常数、重新拟合；如需研究新方法，返回explore并另划未暴露确认集。',
             '', '## 方法与出处', '', json.dumps(candidate, ensure_ascii=False, indent=2), '',
             '完整常数、依赖、源码SHA和已评分开发报告出处：' + str(OUT / 'config.json') + '。',
             '证据快照：' + str(OUT / 'frozen') + '。']
    path.write_text('\n'.join(lines) + '\n')
    print({'task': 'C1', 'state': config['state'], 'candidate': candidate['source_arm'],
           'development_gate_passed': candidate['ten_domains_ge_minus_05_in_explore'], 'n': len(rows)}, flush=True)


def check_freeze(config):
    assert pc.sha(OUT / 'manifest.json') == config['manifest_sha256']
    assert all(pc.sha(path) == sha for path, sha in config['source_sha256'].items()), 'Frozen C1 source/asset changed'
    rows = pc.read(OUT / 'manifest.json')
    assert digest([input_identity(b) for b in rows]) == config['confirm_input_digest']
    return rows


def save_npz(path, values):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def produce(max_new=None):
    pc.setup('cpu')
    config = pc.read(OUT / 'config.json')
    rows = check_freeze(config)
    if (OUT / 'sealed.json').exists():
        verify_predictions(config)
        print({'task': 'C1', 'state': 'ALREADY_SEALED', 'n': len(rows)}, flush=True)
        return
    candidate = config['candidate']
    host = ci.build_host(candidate)
    records, new = [], 0
    start, cpu = time.monotonic(), time.process_time()
    for i, binding in enumerate(rows, 1):
        name = pc.key(binding)
        receipt_path = OUT / 'records' / (name + '.json')
        if receipt_path.exists():
            record = pc.read(receipt_path)
            assert record['binding_input_digest'] == digest(input_identity(binding))
            for kind in ('prediction', 'field'):
                assert pc.sha(record[kind + '_path']) == record[kind + '_sha256']
            records.append(record)
            continue
        if max_new is not None and new >= max_new:
            print({'task': 'C1', 'state': 'PARTIAL_RESUMABLE', 'new': new, 'complete': len(records), 'n': len(rows)}, flush=True)
            return
        tick = time.monotonic()
        predictions, fields, detail = ci.infer(binding, candidate, host)
        pred, field = OUT / 'predictions' / (name + '.npz'), OUT / 'fields' / (name + '.npz')
        # Atomic writes permit recovery if interrupted before the record commit.
        save_npz(pred, predictions); save_npz(field, fields)
        record = {'index': name, 'episode_id': binding['row']['episode_id'], 'dataset': binding['dataset'],
                  'binding_input_digest': digest(input_identity(binding)), 'candidate': candidate['source_arm'],
                  'prediction_path': str(pred), 'prediction_sha256': pc.sha(pred),
                  'field_path': str(field), 'field_sha256': pc.sha(field),
                  'seconds': time.monotonic() - tick, **detail}
        pc.write(receipt_path, record); records.append(record); new += 1
        if new <= 3 or i % 50 == 0:
            typical = float(np.median([r['seconds'] for r in records]))
            print({'task': 'C1', 'completed': len(records), 'n': len(rows),
                   'seconds': round(time.monotonic() - start, 1),
                   'estimated_remaining_seconds': round((len(rows) - len(records)) * typical, 1)}, flush=True)
    pc.write(OUT / 'sealed.json', {'state': 'ALL_CONFIRM_NATURAL_PREDICTIONS_SEALED', 'n': len(rows),
             'records': records, 'manifest_sha256': pc.sha(OUT / 'manifest.json'),
             'config_sha256': pc.sha(OUT / 'config.json'), 'candidate': candidate['source_arm'],
             'invocation_wall_seconds': time.monotonic() - start, 'invocation_cpu_seconds': time.process_time() - cpu,
             'episode_seconds_sum': sum(r['seconds'] for r in records), 'new_encoder_calls': 0, 'query_GT_reads': 0})
    print({'task': 'C1', 'state': 'ALL_CONFIRM_NATURAL_PREDICTIONS_SEALED', 'n': len(rows)}, flush=True)


def verify_predictions(config):
    rows = check_freeze(config)
    seal = pc.read(OUT / 'sealed.json')
    assert seal['state'] == 'ALL_CONFIRM_NATURAL_PREDICTIONS_SEALED'
    assert seal['manifest_sha256'] == config['manifest_sha256'] and pc.sha(OUT / 'config.json') == seal['config_sha256']
    indexed = {r['index']: r for r in seal['records']}
    assert len(rows) == len(indexed) == seal['n']
    for b in rows:
        record = indexed[pc.key(b)]
        assert record['binding_input_digest'] == digest(input_identity(b))
        assert record['new_encoder_calls'] == record['query_GT_reads'] == record['complete_FoRIS_calls'] == 0
        expected = 0 if record.get('reference_rho_gate_triggered', False) else config['candidate']['FoRIS_stage_count']
        assert record['FoRIS_stage_count'] == expected <= 2
        if record.get('reference_rho_gate_triggered', False):
            assert record['actual_stage_calls'] == [] and record['decision_basis'] == 'raw_T'
        for kind in ('prediction', 'field'):
            assert pc.sha(record[kind + '_path']) == record[kind + '_sha256']
    return rows, indexed, seal


def score():
    pc.setup('cpu')
    config = pc.read(OUT / 'config.json')
    if (OUT / 'report.json').exists():
        report = pc.read(OUT / 'report.json')
        assert report['state'] == 'COMPLETE' and report['task'] == 'C1'
        check_freeze(config)
        print({'task': 'C1', 'state': 'ALREADY_COMPLETE', 'candidate': config['candidate']['source_arm']}, flush=True)
        return report
    rows, indexed, seal = verify_predictions(config)
    current = pc.read(CONFIRM)
    assert digest([input_identity(b) for b in current]) == config['confirm_input_digest'], 'Confirm natural input cohort changed'
    baseline_path = P0 / 'foris/sealed.json'
    baseline_seal = pc.read(baseline_path)
    assert baseline_seal['state'] == 'NEW_PAIR_PREDICTIONS_SEALED'
    baseline_records = {r['index']: r for r in baseline_seal['records']}
    assert all('B' in b.get('old_arms', {}) for b in current), 'FoRIS must be sealed/bound before C1 scoring'
    receipts = []
    for binding in current:
        name = pc.key(binding)
        source = baseline_records[name]
        entry = binding['old_arms']['B']
        assert entry['path'] == source['prediction_path'] and entry['sha256'] == source['prediction_sha256']
        with ci.raw_core.legal_inference(binding):
            _, receipt = workflow.read_archive(binding)
        receipts.append(receipt)
    # GT may be read after this point; this marker never changes candidate/config.
    pc.write(OUT / 'score_started.json', {'state': 'CONFIRM_SCORING_STARTED_FIXED_CANDIDATE',
             'config_sha256': pc.sha(OUT / 'config.json'), 'natural_seal_sha256': pc.sha(OUT / 'sealed.json'),
             'FoRIS_seal_sha256': pc.sha(baseline_path), 'baseline_receipts': receipts})
    records = []
    for binding, receipt in zip(current, receipts):
        record = indexed[pc.key(binding)]
        with np.load(record['prediction_path'], allow_pickle=False) as z:
            prediction = pc.unpack(z['candidate'])
        original, latest = workflow.read_archive(binding)
        assert latest == receipt
        records.append(pr.score_record(binding, {'FoRIS': original, 'candidate': prediction}))
    result = pr.report('C1', OUT, records, ARMS, [
        '唯一候选在确认GT暴露前从已评分B4开发表冻结；最差域差值、十域均值、少步骤为自动排序，或root显式指定已测arm。确认集不参与选择。',
        '2136个固定confirm任务全部同规则；CPU重算原P0独立字段，仅从已有O24/参考掩码获取证据，0新增编码。决策基座与A4 maxJ证据开关随单份已评分报告冻结；调制九点不读GT，B/V/P/K始终由original c生成，再对固定主证据加修正作region或graph读出。不作域/例组件选择。',
        '所选0/1/2步依赖逐例如实记录；若源候选有固定rho>.9门控，触发时直接原T且实际0步，不做maxJ/graph/CRF/FoRIS修正。完整自然预测先seal，再验证P0完整FoRIS基线封存与绑定，最后评分。',
        '照片隔离及旧照片新配对的边界沿用P0划分，不额外宣称全部为全新图片。原开发门槛未通过也只确认一次最接近组合。',
        '任何负域如实报告；确认后禁止换候选、改常数或重新拟合。没有A6监督head或B2窗口；唯一可用回退为源开发表已测的固定rho>.9原T规则。',
    ], extra={'candidate_freeze': config, 'confirm_scored': True,
              'natural_seal_sha256': pc.sha(OUT / 'sealed.json'), 'FoRIS_seal_sha256': pc.sha(baseline_path),
              'production_cost': {k: v for k, v in seal.items() if k != 'records'}})
    deltas = {d: s['candidate']['delta_to_foris_pp'] for d, s in result['summary'].items()}
    ci_positive = [d for d, s in result['summary'].items() if s['candidate']['paired_query_photo_bootstrap95_pp'][0] > 0]
    verdict = {'state': 'COMPLETE', 'task': 'C1', 'candidate': config['candidate']['source_arm'],
               'decision_basis': config['candidate']['decision_basis'],
               'evidence_modulation': config['candidate']['evidence_modulation'],
               'n': len(rows), 'negative_domains': [d for d, x in deltas.items() if x < 0],
               'positive_point_domain_n': sum(x > 0 for x in deltas.values()),
               'positive_95CI_domains': ci_positive, 'positive_95CI_domain_n': len(ci_positive),
               'all_ten_domain_deltas_positive': len(deltas) == 10 and all(x > 0 for x in deltas.values()),
               'delta_to_foris_pp': deltas, 'confirm_scored': True, 'post_score_parameter_changes': False,
               'development_goal_passed': config['candidate']['ten_domains_ge_minus_05_in_explore'],
               'new_encoder_calls': 0}
    result['confirm_scored'] = True; result['confirmation_verdict'] = verdict
    pc.write(OUT / 'report.json', result)
    pc.write(OUT / 'completion.json', verdict)
    path = lab.ROOT / 'docs/research/x1012_tasks/C1.md'
    # pr's common diagnostic footer is incorrect for this specific confirm run.
    text = path.read_text().replace('全部为开发/诊断结果，未评分confirm。', '本表为预先冻结候选的固定confirm一次评分。')
    path.write_text(text)
    with path.open('a') as f:
        f.write('\n## 确认判据\n\n' + json.dumps(verdict, ensure_ascii=False, indent=2) + '\n')
    print(verdict, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'produce', 'score'))
    parser.add_argument('--source-report', type=Path, default=lab.ASSETS / 'a/x1012_B4/explore/report.json')
    parser.add_argument('--arm', help='root-specified scored B4 arm; otherwise fixed max-min/mean/fewer-step rule')
    parser.add_argument('--max-new', type=int, help='produce this many new episodes, then resume with the same frozen candidate')
    args = parser.parse_args()
    if args.max_new is not None and args.max_new < 1: parser.error('--max-new must be positive')
    if args.mode == 'freeze': freeze(args)
    elif args.mode == 'produce': produce(args.max_new)
    else: score()
