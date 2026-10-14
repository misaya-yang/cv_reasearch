"""Render the revised T1'/T6/E1 report without opening the held-out T half."""
from __future__ import annotations

import numpy as np
import common as c
import score


def pct(value):
    return '—' if value is None else f'{value:+.2f}'


def merge(base, additions):
    result = {row['index']: dict(row) for row in base}
    for row in additions:
        result[row['index']].update(row)
    return list(result.values())


def cell_delta(records, arm, reference, rng):
    delta = score.miou(records, arm) - score.miou(records, reference)
    low, high = score.paired_ci(records, arm, reference, rng)
    star = '*' if low > 0 or high < 0 else ''
    return f'{delta:+.2f}{star} [{low:+.2f},{high:+.2f}]'


def relative_table(records, arms, reference='foris', intervals=True):
    rng = np.random.default_rng(c.SEED)
    header = ['变体'] + [d for d in c.DOMAINS] + ['等权均值', '≥FoRIS 域数']
    rows = ['|' + '|'.join(header) + '|', '|' + '|'.join(['---'] * len(header)) + '|']
    for arm in arms:
        deltas, cells = [], []
        for domain in c.DOMAINS:
            subset = [r for r in records if r['dataset'] == domain]
            delta = score.miou(subset, arm) - score.miou(subset, reference)
            deltas.append(delta)
            cells.append(cell_delta(subset, arm, reference, rng) if intervals else pct(delta))
        rows.append('|' + '|'.join([arm, *cells, pct(float(np.mean(deltas))),
                                    str(sum(delta >= 0 for delta in deltas)) + '/10']) + '|')
    return rows


def relative_to_mean_table(records, arms):
    domains = ('coco', 'isic', 'suim')
    header = ['变体'] + list(domains)
    rows = ['|' + '|'.join(header) + '|', '|' + '|'.join(['---'] * len(header)) + '|']
    rng = np.random.default_rng(c.SEED)
    for arm in arms:
        cells = []
        for domain in domains:
            subset = [r for r in records if r['dataset'] == domain]
            cells.append(cell_delta(subset, arm, 'M', rng))
        rows.append('|' + '|'.join([arm, *cells]) + '|')
    return rows


def e2_selection(records):
    candidates = ('C1', 'UFraw+', 'Ugm+', 'UF+', 'M+', 'MFU')
    reference = score.domain_scores(records, 'foris')
    summaries = []
    for arm in candidates:
        values = {domain: score.miou([r for r in records if r['dataset'] == domain], arm) -
                  reference[domain] for domain in c.DOMAINS}
        summaries.append((min(values.values()), float(np.mean(list(values.values()))), arm))
    return max(summaries, key=lambda item: (item[0], item[1]))


def render_t8_report():
    task = c.task_path('T8')
    status_path = task / 'scored.json'
    if not status_path.exists():
        return None
    status = c.read(status_path)
    tables = c.read(task / 'prefix_tables.json')
    selection_hash = __import__('hashlib').sha256((task / 'selection.json').read_bytes()).hexdigest()
    lines = [
        '# T8 prefix diagnosis (S half only)', '',
        f"The frozen tail/control sample contains {status['n']} S-half cases; T labels were not read and no encoder calls were made.",
        'Rates are fractions of selected error-region pixels classified foreground at each cached output prefix. They are a targeted diagnostic, not dataset-wide accuracy.', '',
        'In the tail group, FoRIS `pre` foreground rates are 0.046–0.070 in candidate-only false-positive pixels and 0.912–0.964 in candidate-missed true-positive pixels; the final FoRIS mask is 0 and 1, respectively. The candidate `c_mod+V+P` prefix marks 0.077–0.381 of the latter pixels foreground, while the final candidate mask marks none. This shows disagreement already present in the frozen prefix masks for these selected cases; it does not establish a general correction rule.', '',
        '|Domain|Group|Region|Images|Pixels|FoRIS pre|FoRIS final|Own c_mod+V+P|Own no-e|Candidate final|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|'
    ]
    for domain in ('coco', 'isic', 'suim', 'lvis'):
        for row in tables[domain]:
            lines.append(f"|{domain}|{row['group']}|{row['region']}|{row['n']}|{row['pixels']}|"
                         f"{row['foris/pre']:.3f}|{row['foris/FoRIS']:.3f}|"
                         f"{row['own/c_mod+V+P']:.3f}|{row['own/no_e']:.3f}|{row['own/candidate']:.3f}|")
    lines += ['', f"Selection SHA256: `{selection_hash}`.",
              f"Per-case records: `{task / 'descriptions.jsonl'}`; case sheets: `{task / 'cases'}`.", '']
    path = c.ROOT / 'docs/research/x1013_tasks/T8_S_REPORT.md'
    path.write_text('\n'.join(lines))
    return path


def area_table(records, arms):
    bands = ('<2%', '2-20%', '>20%')
    header = ['变体'] + list(bands)
    rows = ['|' + '|'.join(header) + '|', '|' + '|'.join(['---'] * len(header)) + '|']
    for arm in arms:
        cells = []
        for band in bands:
            subset = [r for r in records if score.area_band(r) == band]
            if not subset:
                cells.append('—')
            else:
                delta = np.mean([score.episode_iou(r, arm) - score.episode_iou(r, 'foris')
                                 for r in subset])
                cells.append(pct(float(delta)))
        rows.append('|' + '|'.join([arm, *cells]) + '|')
    return rows


def absolute_table(records, arms):
    header = ['指标'] + list(c.DOMAINS) + ['等权均值']
    rows = ['|' + '|'.join(header) + '|', '|' + '|'.join(['---'] * len(header)) + '|']
    for arm in arms:
        values = [score.miou([r for r in records if r['dataset'] == d], arm) for d in c.DOMAINS]
        rows.append('|' + '|'.join([arm, *(f'{v:.2f}' for v in values), f'{np.mean(values):.2f}']) + '|')
    return rows


def effect_table(records, pairs):
    header = ['作用'] + list(c.DOMAINS) + ['等权均值']
    rows = ['|' + '|'.join(header) + '|', '|' + '|'.join(['---'] * len(header)) + '|']
    for label, arm, reference in pairs:
        values = [score.miou([r for r in records if r['dataset'] == d], arm) -
                  score.miou([r for r in records if r['dataset'] == d], reference)
                  for d in c.DOMAINS]
        rows.append('|' + '|'.join([label, *(pct(v) for v in values), pct(float(np.mean(values)))]) + '|')
    return rows


def render():
    t0 = score.read_records('T0')
    t1 = score.read_records('T1')
    t6 = score.read_records('T6')
    e1 = score.read_records('E1')
    if not all((t0, t1, t6, e1)):
        raise RuntimeError('Revised report requires T0, T1 S, T6 S and E1 S records.')
    s0 = [r for r in t0 if r['split'] == 'S']
    s1 = merge(s0, t1)
    s6 = merge(s0, t6)
    identity = c.read(c.task_path('T1') / 'identity_summary.json')
    e1_identity = c.read(c.task_path('E1') / 'ufraw_identity.json')
    lines = [
        '# 修订计划执行报告：T1′、T6 与 E1', '',
        '执行依据：[NEXT_EXPERIMENT_PLAN_20261013B.md](../NEXT_EXPERIMENT_PLAN_20261013B.md)。',
        '数据为已曝光开发缓存；所有预测覆盖 13,613 例，T1′/T6/E1 的指标仅用 S 半（6,752 例）。T 半尚未评分。',
        '指标为 CLI-1024，(fold,class) 池化 I/U 后槽等权；区间为按 query 照片成组的 500 次配对 bootstrap。* 表示 95% 区间不含 0。', '',
        '## T1′ MEAN 重放与身份差异', '',
        f"MEAN 全池重算；缓存覆盖 {identity['n_cached']}/13,613。身份差异仅比较预测掩码，不读取标签：XOR p50={identity['xor_pixels'].get('p50')}、p90={identity['xor_pixels'].get('p90')}、p99={identity['xor_pixels'].get('p99')}、max={identity['xor_pixels'].get('max')}；超过4像素 {identity['n_over_4_pixels']} 例。原先触发停止的单例保留在 `../cv_data/a/x1013_T1/gate_failure.json`。", '',
        '重算与缓存 MEAN 的 class mIoU 差仅在 S 半缓存子集上计算，以保留 T 半；单位 pp。', '',
        '|域|缓存 S 例数|重算 MEAN − 缓存 MEAN|', '|---|---:|---:|'
    ]
    deltas = identity['miou_delta_recomputed_minus_cached_S']
    for domain in c.DOMAINS:
        cached_n = sum(1 for r in t1 if r['dataset'] == domain and 'mean_cached' in r)
        if domain in deltas:
            lines.append(f'|{domain}|{cached_n}|{deltas[domain]:+.3f}|')
        else:
            lines.append(f'|{domain}|{cached_n}|—|')
    lines += ['', 'MEAN、FoRIS、FoRIS-noCRF、T、候选与两个严格同分组 oracle 的 S 半绝对表：', '']
    lines += absolute_table(s1, ('foris', 'foris_noCRF', 'mean', 'T', 'candidate',
                                  'oracle_candidate', 'oracle_foris')) + ['',
        '## T6 角色对称（S 半）', '',
        '相对完整 FoRIS 的 class mIoU 差，单位 pp；面积表为逐例 IoU 差。', '']
    lines += relative_table(s6, ('S-self', 'S-avg', 'S-ref')) + ['', *area_table(s6, ('S-self', 'S-avg', 'S-ref'))]
    candidate_scores = score.domain_scores(s0, 'candidate')
    self_scores = score.domain_scores(s6, 'S-self')
    d_self = {d: self_scores[d] - candidate_scores[d] for d in c.DOMAINS}
    self_keep = float(np.mean(list(d_self.values()))) >= 0 and min(d_self.values()) >= -.3
    lines += ['', f"T6 的 S-self 对照 E2 入选条件：等权均值变化 {np.mean(list(d_self.values())):+.3f} pp；最差域变化 {min(d_self.values()):+.3f} pp；{'满足' if self_keep else '不满足'}。", '',
        '## E1 CRF 对等比较（S 半）', '',
        '相对完整 FoRIS 的 class mIoU 差，单位 pp；括号为95%区间。', '']
    e1_arms = ('C1', 'C2', 'UFraw', 'UFraw+', 'Ugm+', 'UF+', 'M', 'M+', 'MFU')
    lines += relative_table(e1, e1_arms) + ['',
        '面积档逐例 IoU 差：', '']
    lines += area_table(e1, e1_arms) + ['',
        'COCO、ISIC、SUIM 相对 MEAN（M）的 class mIoU 差，单位 pp；括号为95%区间：', '']
    lines += relative_to_mean_table(e1, e1_arms) + ['',
        'UFraw 对 FoRIS-noCRF（双方都不做 CRF）的配对差：', '']
    lines += relative_table(e1, ('UFraw',), reference='foris_noCRF') + ['',
        'CRF 自身的作用（class mIoU 差，单位 pp）：', '']
    lines += effect_table(e1, (('C1 − 候选', 'C1', 'candidate'),
                               ('UFraw+ − UFraw', 'UFraw+', 'UFraw'),
                               ('M+ − M', 'M+', 'M'))) + ['',
        '对应面积档逐例 IoU 差：', '']
    crf_area = ['|作用|<2%|2-20%|>20%|', '|---|---:|---:|---:|']
    for label, arm, reference in (('C1 − 候选', 'C1', 'candidate'),
                                  ('UFraw+ − UFraw', 'UFraw+', 'UFraw'),
                                  ('M+ − M', 'M+', 'M')):
        cells = []
        for band in ('<2%', '2-20%', '>20%'):
            subset = [r for r in e1 if score.area_band(r) == band]
            delta = np.mean([score.episode_iou(r, arm) - score.episode_iou(r, reference)
                             for r in subset]) if subset else None
            cells.append(pct(float(delta)) if delta is not None else '—')
        crf_area.append('|' + '|'.join([label, *cells]) + '|')
    lines += crf_area + ['',
        f"UFraw 身份核对（S 半、排除 iSAID，与旧 W10 U@Fraw 对照）：{e1_identity}", '',
        f"E2 的 S 半预选（固定集合，不含未通过条件的 S-self 替换臂）：{e2_selection(e1)}；按预定规则选臂，T 半尚未用于选择。", '',
        'E1 逐例结果：[evidence/research/x1013_E1_iu.jsonl.gz](../../../evidence/research/x1013_E1_iu.jsonl.gz)。']
    e2_report = c.ROOT / 'docs/research/x1013_tasks/E2_T_REPORT.md'
    if e2_report.exists():
        content = e2_report.read_text().splitlines()
        if content and content[0].startswith('# '):
            content = content[1:]
        lines += ['', '## E2 报告半结果', '', *content]
    else:
        lines += ['', 'E2 的报告半评分尚未开始。']
    t8_report = render_t8_report()
    if t8_report:
        lines += ['', '## T8 S 半前缀诊断', '',
                  'T8 使用 S 半 322 例（161 尾部、161 对照），未读取 T 半标签；完整逐域前缀表和案例路径见 [T8 S 半报告](T8_S_REPORT.md)。']
    else:
        lines += ['', 'T8 是可选诊断，本轮未运行。']
    c.REPORT.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {c.REPORT} ({len(lines)} lines)')


if __name__ == '__main__':
    render()
