"""The requested short report, inspectable tables, and T8 PNG contact sheets."""
from __future__ import annotations

import csv
import json
import numpy as np
import common as c
import score

TABLE_ARMS = {
    'T1': ('foris_noCRF', 'mean', 'T', 'candidate', 'oracle_candidate', 'oracle_foris'),
    **{task: arms for task, arms in score.ARMS.items() if task not in ('T0', 'T1', 'appendix')},
}


def fmt(value, signed=True):
    if value is None:
        return '—'
    return f'{value:+.2f}' if signed else f'{value:.2f}'


def cell(value, ci=None):
    return fmt(value) if ci is None else f'{fmt(value)} [{fmt(ci[0])},{fmt(ci[1])}]'


def relative_table(records, arms, reference='foris', intervals=None, label='变体'):
    lines = ['|' + label + '|' + '|'.join(c.DOMAIN_NAMES) + '|等权均值|≥' + reference + '域数|',
             '|---|' + '|'.join(['---:'] * 12) + '|']
    for arm in arms:
        deltas, cells = [], []
        for domain in c.DOMAINS:
            subset = [r for r in records if r['dataset'] == domain]
            delta = score.miou(subset, arm) - score.miou(subset, reference)
            deltas.append(delta)
            ci = None if intervals is None else intervals.get((domain, arm, reference))
            cells.append(cell(delta, ci))
        lines.append('|' + arm + '|' + '|'.join(cells) + '|' + fmt(float(np.mean(deltas))) +
                     '|' + str(sum(delta >= 0 for delta in deltas)) + '|')
    return lines


def area_table(records, arms, reference='foris'):
    lines = ['|变体（小／中／大）|' + '|'.join(c.DOMAIN_NAMES) + '|',
             '|---|' + '|'.join(['---:'] * 10) + '|']
    for arm in arms:
        cells = []
        for domain in c.DOMAINS:
            values = []
            for band in ('<2%', '2-20%', '>20%'):
                subset = [r for r in records if r['dataset'] == domain and score.area_band(r) == band]
                values.append(float(np.mean([score.episode_iou(r, arm) - score.episode_iou(r, reference)
                                             for r in subset])) if subset else None)
            cells.append('/'.join(fmt(value) for value in values))
        lines.append('|' + arm + '|' + '|'.join(cells) + '|')
    return lines


def save_tsv(path, header, rows):
    with c.Path(path).open('w') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(header)
        writer.writerows(rows)


def failure_report(task, error):
    c.REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# NEXT_EXPERIMENT_PLAN_20261013 执行报告', '',
             f'执行在 {task} 停止：`{type(error).__name__}: {error}`。没有更改科学定义来通过身份闸门。', '',
             '|任务|是否完成|身份闸门|', '|---|---|---|']
    for name in (f'T{i}' for i in range(10)):
        path = c.task_path(name) / 'scored.json'
        status = c.read(path) if path.exists() else {}
        lines.append(f"|{name}|{'是' if status else '否'}|{status.get('identity_gate', '未通过' if name == task else '未评')}|")
    gate = c.task_path(task) / 'gate_failure.json'
    if gate.exists():
        lines += ['', f'原始闸门差异：`{gate}`。']
    c.REPORT.write_text('\n'.join(lines) + '\n')

def auxiliary_tables(records):
    destination = c.REPORT.parent
    t3 = []
    for domain in c.DOMAINS:
        eligible = [r for r in records if r['dataset'] == domain and r['rho'] > .9]
        for label, low, high in (('all', 0, 4096), ('<=16', 0, 16), ('17-64', 17, 64), ('65-256', 65, 256), ('>256', 257, 4096)):
            subset = [r for r in eligible if low <= r['touched'] <= high]
            means = [float(np.mean([score.episode_iou(r, arm) for r in subset])) if subset else None
                     for arm in ('foris', 'T', 'VP_no_fallback')]
            t3.append([domain, label, len(subset), *means])
    save_tsv(destination / 'T3_touched.tsv', ['dataset', 'touched', 'n', 'FoRIS', 'T', 'VP_UN'], t3)
    t6 = []
    for domain in c.DOMAINS:
        all_rows = [r for r in records if r['dataset'] == domain]
        subset = [r for r in all_rows if r.get('rewritten', False)]
        t6.append([domain, len(all_rows), len(subset), len(subset) / len(all_rows),
                   *[float(np.mean([score.episode_iou(r, arm) for r in subset])) if subset else None
                     for arm in ('candidate', 'S-self')]])
    save_tsv(destination / 'T6_rewrites.tsv', ['dataset', 'n', 'rewrite_n', 'rewrite_fraction', 'before', 'after'], t6)
    cluster_counts = []
    for binding in c.rows():
        with np.load(c.task_path('T5') / 'clusters' / (c.key(binding) + '.npz'), allow_pickle=False) as archive:
            cluster_counts.append((binding['dataset'], int(archive['n'])))
    clusters = {domain: np.percentile([n for d, n in cluster_counts if d == domain],
                                     [0, 10, 25, 50, 75, 90, 100]).tolist() for domain in c.DOMAINS}
    c.write(destination / 'T5_clusters.json', clusters)


def tint(image, mask, color):
    result = image.copy()
    result[mask] = (.45 * result[mask] + .55 * np.asarray(color)).astype(np.uint8)
    return result


def case_sheets(selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from ics.official_data import decoded_rgb
    paths = []
    destination = c.task_path('T8') / 'cases'
    destination.mkdir(parents=True, exist_ok=True)
    for domain in ('coco', 'isic', 'suim'):
        tail = [(b, record) for b, group, record in selected if b['dataset'] == domain and group == 'tail'][:24]
        for page in range(0, len(tail), 6):
            figure, axes = plt.subplots(6, 7, figsize=(21, 18))
            for ax in axes.ravel():
                ax.axis('off')
            for row, (binding, base) in enumerate(tail[page:page + 6]):
                meta = binding['row']
                reference = np.asarray(decoded_rgb(c.ASSETS / meta['reference_path'], meta.get('reference_crop')).resize((512, 512)))
                query = np.asarray(decoded_rgb(c.ASSETS / meta['query_path'], meta.get('query_crop')).resize((512, 512)))
                refmask = c.rc.lab.mask1024(meta['reference_mask_path']).numpy()[::2, ::2]
                gt, _ = c.truth(binding)
                foris = c.rc.lab.arm_cli(binding, 'B').numpy()[::2, ::2]
                with np.load(c.prediction_path('T0', binding), allow_pickle=False) as archive:
                    candidate = c.unpack(archive['candidate'])[::2, ::2]
                fields = c.load_fields(binding)
                images = [(tint(reference, refmask, (0, 255, 0)), f"R+mask {binding['index']} rho={base['rho']:.2f}"),
                          (tint(query, gt[::2, ::2], (255, 0, 0)), 'Q+GT'),
                          (tint(query, foris, (0, 100, 255)), f"FoRIS {score.episode_iou(base, 'foris'):.1f}"),
                          (tint(query, candidate, (255, 0, 255)), f"U {score.episode_iou(base, 'candidate'):.1f}"),
                          (c.up(c.foreground_field(fields))[::2, ::2], 'candidate field'),
                          (c.up(fields['explanation'])[::2, ::2], 'e'),
                          (c.up(fields['addition_P'])[::2, ::2], 'P')]
                for column, (image, title) in enumerate(images):
                    axes[row, column].imshow(image, cmap='viridis')
                    axes[row, column].set_title(title, fontsize=9)
            figure.tight_layout()
            path = destination / f'{domain}_{page // 6 + 1:02d}.png'
            figure.savefig(path, dpi=60)
            plt.close(figure)
            paths.append(str(path))
    return paths


def render():
    records = score.merged_records()
    if not records or any(not (c.task_path(f'T{i}') / 'scored.json').exists() for i in range(10)):
        raise RuntimeError('The final report requires completed T0--T9')
    c.initialize()
    destination = c.REPORT.parent
    destination.mkdir(parents=True, exist_ok=True)
    scopes = {'全池': records, 'S': [r for r in records if r['split'] == 'S'],
              'T': [r for r in records if r['split'] == 'T']}
    rng = np.random.default_rng(c.SEED)
    interval_cache = {}
    t1_rows = []
    for scope, subset in scopes.items():
        for domain in c.DOMAINS:
            domain_rows = [r for r in subset if r['dataset'] == domain]
            values = [score.miou(domain_rows, arm) for arm in ('foris', 'foris_noCRF', 'mean', 'T', 'candidate',
                                                             'oracle_candidate', 'oracle_foris')]
            for reference in ('foris', 'mean'):
                interval_cache[(scope, domain, 'candidate', reference)] = score.paired_ci(domain_rows, 'candidate', reference, rng)
            t1_rows.append([scope, domain, len(domain_rows), *values,
                            values[4] - values[0], *interval_cache[(scope, domain, 'candidate', 'foris')],
                            values[4] - values[2], *interval_cache[(scope, domain, 'candidate', 'mean')]])
    save_tsv(destination / 'T1_baselines.tsv',
             ['scope', 'dataset', 'n', 'FoRIS', 'FoRIS_noCRF', 'MEAN', 'T', 'candidate',
              'oracle_candidate', 'oracle_foris', 'U-F', 'U-F_lo', 'U-F_hi', 'U-M', 'U-M_lo', 'U-M_hi'], t1_rows)
    lines = ['# NEXT_EXPERIMENT_PLAN_20261013 执行报告', '',
             '13,613 个已有缓存配对；CLI-1024；(fold,class) 池化 I/U 后槽等权；iSAID 扣 ignore。全部是曝光开发数据。',
             '差值单位 pp。区间为 query 照片簇配对 bootstrap 500 次，seed=20261013。自然预测全部落盘后才评分。',
             'S/T 由 query 照片 id 的 SHA256 前8位奇偶决定；所有保留决策只看 S，最终 T 评分一次。', '',
             '## 1 完成状态', '', '|任务|完成|身份闸门|耗时（秒）|', '|---|---|---|---:|']
    for task in (f'T{i}' for i in range(10)):
        status = c.read(c.task_path(task) / 'scored.json')
        produced = c.read(c.task_path(task) / 'complete.json') if (c.task_path(task) / 'complete.json').exists() else {}
        elapsed = status['seconds'] + produced.get('seconds', 0)
        lines.append(f"|{task}|是|{status['identity_gate']}|{elapsed:.1f}|")
    lines += ['', '## 2 T1–T7', '',
              '下表均相对 FoRIS。面积表每格为 <2% / 2–20% / >20% 的逐例平均 IoU 差；空档记 —。',
              '[T1 完整绝对分数、例数、两组区间及 T 半表](T1_baselines.tsv)。',
              '[T3 touched 分箱](T3_touched.tsv)；[T5 聚类数分位](T5_clusters.json)；[T6 改写比例与前后分数](T6_rewrites.tsv)。', '']
    for task, arms in TABLE_ARMS.items():
        lines += [f'### {task}：全池', '']
        intervals = {(domain, arm, reference): value for (scope, domain, arm, reference), value in interval_cache.items()
                     if scope == '全池'} if task == 'T1' else None
        lines += relative_table(records, arms, intervals=intervals)
        lines += ['']
        intervals = {(domain, arm, reference): value for (scope, domain, arm, reference), value in interval_cache.items()
                     if scope == 'S'} if task == 'T1' else None
        lines += relative_table(scopes['S'], arms, intervals=intervals, label='变体（S 半）')
        lines += [''] + area_table(records, arms) + ['']
    lines += ['## 3 T8 前缀归因', '',
              'LVIS 仅 GT>20%；其它三域用全域。尾部取额外错误像素最高10%，其余固定种子等量抽样。值为该错误区域像素加权前景比例。', '']
    tables = c.read(c.task_path('T8') / 'prefix_tables.json')
    columns = ['foris/' + name for name in ('fg', 'bg', 'vote', 'prior', 'penalty', 'pre', 'FoRIS')]
    columns += ['own/' + name for name in ('c', 'c_mod', 'c_mod+V', 'c_mod+V+P', 'no_e', 'candidate')]
    for domain, cells in tables.items():
        lines += [f'**{domain}**', '', '|组/区域|n|像素|' + '|'.join(columns) + '|',
                  '|---|---:|---:|' + '|'.join(['---:'] * len(columns)) + '|']
        lines += ['|' + f"{row['group']}/{row['region']}|{row['n']}|{row['pixels']}|" +
                  '|'.join(f'{row[col]:.3f}' for col in columns) + '|' for row in cells]
        lines += ['']
    t8 = c.read(c.task_path('T8') / 'scored.json')
    lines += [f"逐例描述量：`{c.task_path('T8') / 'descriptions.jsonl'}`。案例图：`{c.task_path('T8') / 'cases'}`（COCO/ISIC/SUIM 各4页，各24例，每页6行，dpi60）。", '',
              '## 4 T9 固定顺序组装', '', '|尝试加入|S 等权均值变化|S 最差域变化|保留|', '|---|---:|---:|---|']
    state = c.read(c.task_path('T9') / 'selection.json')
    for step in state['steps']:
        lines.append(f"|{step['name']}|{step['equal_domain_delta']:+.4f}|{step['worst_domain_delta']:+.4f}|{'是' if step['keep'] else '否'}|")
    lines += ['', '判据：等权变化 ≥0 且最差域变化 ≥−0.3。保留：' + ('、'.join(state['retained']) or '无') + '。', '']
    t9_rows = []
    for scope in ('T', '全池'):
        subset = scopes[scope]
        lines += [f'**最终 {scope}**', '']
        for reference in ('foris', 'mean'):
            intervals = {(domain, 'final', reference): score.paired_ci(
                [r for r in subset if r['dataset'] == domain], 'final', reference, rng) for domain in c.DOMAINS}
            lines += [f'相对 {reference}：'] + relative_table(subset, ('final',), reference, intervals) + ['']
            for domain in c.DOMAINS:
                domain_rows = [r for r in subset if r['dataset'] == domain]
                t9_rows.append([scope, domain, reference, len(domain_rows), score.miou(domain_rows, 'final'),
                                score.miou(domain_rows, 'final') - score.miou(domain_rows, reference),
                                *intervals[(domain, 'final', reference)]])
        lines += area_table(subset, ('final',)) + ['']
    save_tsv(destination / 'T9_final.tsv', ['scope', 'dataset', 'reference', 'n', 'final', 'delta', 'lo', 'hi'], t9_rows)
    lines += ['## 5 执行偏差', '',
              '- T1 的“五条线”定义表只列四条自然线；第五条报告原 T（供 T3 回退对照），没有增加候选变体。',
              '- T1 要求 T 半表与公共约定的 T9 一次评分有文字冲突；T1 的 T 半汇总仅在 T9 选择冻结后生成，没有用于任何保留判断。',
              '- T8 中“LVIS（仅 GT>20%）”按仅限定 LVIS 执行；COCO/ISIC/SUIM 保留原域面积范围。',
              '- T9 C1 沿用明确范围 ρ≤.9；G1 取消回退的原 ρ>.9 任务仍不加 CRF。', '',
              f'逐例整数：`{c.EVIDENCE}`；代码：`scripts/x1013/run.py all --workers 6`。']
    if records and 'MU_control' in records[0]:
        lines += ['', '## 附录：固定 M/U 对照', '',
                  '来自用户在共享聊天中的追加要求；原计划完成后执行，不参与 T9。公式：ρ>.9 用原 T，否则 .5M+.5U>.5，FP64，无 CRF。',
                  '下表相对原 U，M 是 T1 原 MEAN 连续场的绝对读出，U 是冻结候选的 min-max 读出。', '']
        appendix_intervals = {(domain, 'MU_control', 'candidate'): score.paired_ci(
            [r for r in records if r['dataset'] == domain], 'MU_control', 'candidate', rng) for domain in c.DOMAINS}
        lines += relative_table(records, ('MU_control',), 'candidate', appendix_intervals)
        c.write(destination / 'MU_control.json', {domain: {
            'n': len([r for r in records if r['dataset'] == domain]),
            'delta_to_U': score.miou([r for r in records if r['dataset'] == domain], 'MU_control') -
                          score.miou([r for r in records if r['dataset'] == domain], 'candidate'),
            'CI95': appendix_intervals[(domain, 'MU_control', 'candidate')]} for domain in c.DOMAINS})
    auxiliary_tables(records)
    score.save_records(c.EVIDENCE, records)
    if len(lines) > 300:
        raise RuntimeError(f'Report too long: {len(lines)} lines')
    c.REPORT.write_text('\n'.join(lines) + '\n')
    c.write(destination / 'summary.json', {'n': len(records), 'S': len(scopes['S']), 'T': len(scopes['T']),
            'retained': state['retained'], 'baseline_table': t1_rows, 'final_table': t9_rows,
            'T8_case_paths': t8['case_paths'], 'report_lines': len(lines)})
    print(f'Report: {c.REPORT}, {len(lines)} lines', flush=True)


if __name__ == '__main__':
    render()
