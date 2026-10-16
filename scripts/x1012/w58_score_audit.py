"""Compare repaired W45/W49 scores with preserved ledgers; no model execution.

Run after both exact produce/report commands. Writes W58 summary and per-case evidence.
"""
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from diagnostic_metrics import classify_diagnostic, legacy_classification, DIAGNOSTIC_CLASSES, SCORING_VERSION

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT.parent / 'cv_data/a/x1013_W58_score_repair'
EV = ROOT / 'evidence/research'


def read(name):
    with gzip.open(EV / name, 'rt') as f:
        rows = list(map(json.loads, f))
    assert len({r['index'] for r in rows}) == len(rows)
    return {r['index']: r for r in rows}


def iou(v):
    return v[0] / max(v[1], 1)


def probe_support(cases):
    """Inspect existing S query labels only; no fitting or feature/model execution."""
    sys.path[:0] = [str(ROOT / 'scripts/x1012'), str(ROOT / 'scripts/x1013')]
    import w31_demonstration as w31
    from w45_solvability import FOLD
    import plan_common as pc
    by = {r['index']: r for r in cases}
    rows = []
    for b in w31.rows():
        r = by[str(b['index'])]
        if not r['catastrophic']:
            continue
        gt, _ = pc.truth(b)
        y = gt.reshape(64, 16, 64, 16).mean((1, 3)).ravel()
        rows.append(dict(index=r['index'], dataset=r['dataset'], old=r['old_class'], new=r['diagnostic'],
                         folds=[dict(nonzero=int((y[FOLD == v] > 0).sum()),
                                     majority=int((y[FOLD == v] >= .5).sum()), mass=float(y[FOLD == v].sum()))
                                for v in (False, True)]))
    summary = {}
    for name, sel in (('old_representation', [r for r in rows if r['old'] == 'representation']),
                      ('new_unresolved', [r for r in rows if r['new'] == 'unresolved_by_probes']),
                      ('all_catastrophic', rows)):
        summary[name] = dict(n=len(sel), **{
            'any_fold_' + label: sum(min(f[field] for f in r['folds']) < cutoff for r in sel)
            for label, field, cutoff in (('no_soft_positive', 'nonzero', 1), ('no_majority_positive', 'majority', 1),
                                         ('fewer_than3_soft_positive', 'nonzero', 3), ('fewer_than3_majority_positive', 'majority', 3))})
    (EV / 'w58_probe_support.json').write_text(json.dumps(dict(
        definition='Original W45 soft coverage labels; fixed training FOLD; original S only',
        summary=summary, cases=rows), indent=1) + '\n')
    return summary


def run():
    old45, new45 = read('w45_solvability_S.jsonl.gz'), read('w45_solvability_S_exact.jsonl.gz')
    old49, new49 = read('w49_token_resolution_S.jsonl.gz'), read('w49_token_resolution_S_exact.jsonl.gz')
    sh, w48, w51 = read('w23_simple_pipeline_S.jsonl.gz'), read('w48_reference_specificity_S.jsonl.gz'), read('w51_competition_S.jsonl.gz')
    assert len(new45) == 1285
    assert list(old45) == list(new45) == list(old49) == list(new49)
    manifest = json.loads((OUT / 'inputs.json').read_text())
    for path, sha in manifest['files'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha, path
    rows = []
    for k, r in new45.items():
        a, b, v = old45[k], old49[k], new49[k]
        assert r['scoring'] == v['scoring'] == SCORING_VERSION
        for name in ('dataset', 'fold', 'cls', 'photo', 'gt', 'fallback', 'qfrac', 'rho', 'SH', 'PQ', 'PR'):
            assert r[name] == a[name], (k, name, r[name], a[name])
        assert r['SH'] == sh[k]['SH'] and v['N64'] == b['N64']
        for name, natural in (('F', 'SH'), ('PQ', 'PQ'), ('PR', 'PR')):
            assert r['cut_' + name] + 1e-12 >= iou(r[natural])
            assert r['cut_' + name + '_all'] + 1e-12 >= a['cut_' + name]
        for g in (64, 128, 256):
            assert v[f'C{g}_all'] + 1e-12 >= b[f'C{g}']
        assert v['C64'] + 1e-12 >= iou(v['N64'])
        rows.append(dict(index=k, dataset=r['dataset'], photo=r['photo'], gt=r['gt'],
                         catastrophic=iou(sh[k]['SH+']) < .3, old_class=legacy_classification(a),
                         exact_legacy_class=legacy_classification(r), diagnostic=classify_diagnostic(r),
                         old_cut={s: a['cut_' + s] for s in ('F', 'PQ', 'PR')},
                         exact_cut={s: r['cut_' + s] for s in ('F', 'PQ', 'PR')},
                         old_C={g: b[f'C{g}'] for g in (64, 128, 256)},
                         exact_C={g: v[f'C{g}'] for g in (64, 128, 256)},
                         sh=iou(r['SH']), shplus=iou(sh[k]['SH+']), pq=iou(r['PQ']), pr=iou(r['PR']),
                         coverage_natural=iou(v['N64'])))
    cat = [r for r in rows if r['catastrophic']]
    non = [r for r in rows if not r['catastrophic']]
    def means(group):
        return dict(n=len(group), old_cut={s: float(np.mean([r['old_cut'][s] for r in group])) for s in ('F', 'PQ', 'PR')},
                    exact_cut={s: float(np.mean([r['exact_cut'][s] for r in group])) for s in ('F', 'PQ', 'PR')},
                    old_C={g: float(np.mean([r['old_C'][g] for r in group])) for g in (64, 128, 256)},
                    exact_C={g: float(np.mean([r['exact_C'][g] for r in group])) for g in (64, 128, 256)},
                    old_class=dict(Counter(r['old_class'] for r in group)),
                    exact_legacy_class=dict(Counter(r['exact_legacy_class'] for r in group)),
                    diagnostic=dict(Counter(r['diagnostic'] for r in group)))
    original60 = [r for r in cat if r['old_class'] == 'representation' and r['old_C'][64] < .5]
    low = [r for r in cat if r['exact_C'][64] < .5]
    unresolved = [r for r in cat if r['diagnostic'] == 'unresolved_by_probes']
    summary = dict(scoring=SCORING_VERSION, n=len(rows), natural_predictions_exactly_reproduced=len(rows),
                   preserved_source_hashes=len(manifest['files']), all=means(rows), catastrophic=means(cat), noncatastrophic=means(non),
                   original60_n=len(original60), original60_now_C64_ge_half=sum(r['exact_C'][64] >= .5 for r in original60),
                   original60_now_PQ_ge_half=sum(r['exact_cut']['PQ'] >= .5 for r in original60),
                   all_low_C64=sum(r['exact_C'][64] < .5 for r in rows), catastrophic_low_C64=len(low),
                   unresolved_low_C64=sum(r['exact_C'][64] < .5 for r in unresolved),
                   catastrophic_precrf_already_ge_half=sum(r['sh'] >= .5 for r in cat),
                   low_C64_reconstructed_at128=sum(r['exact_C'][128] >= .5 for r in low),
                   low_C64_reconstructed_at256=sum(r['exact_C'][256] >= .5 for r in low),
                   original_to_exact_legacy=dict(Counter(r['old_class'] + ' -> ' + r['exact_legacy_class'] for r in cat)),
                   old_to_evidence_class=dict(Counter(r['old_class'] + ' -> ' + r['diagnostic'] for r in cat)),
                   sh_field_beats_exact_coverage=[r['index'] for r in rows if r['exact_cut']['F'] > r['exact_C'][64] + 1e-9],
                   natural_beats_old_oracle={s: sum(r[n] > r['old_cut'][s] + 1e-9 for r in rows) for s, n in (('F', 'sh'), ('PQ', 'pq'), ('PR', 'pr'))},
                   natural_coverage_beats_old_C64=sum(r['coverage_natural'] > r['old_C'][64] + 1e-9 for r in rows),
                   valid_effects={s: dict(n=sum(new45[r['index']]['cut_' + s] != new45[r['index']]['cut_' + s + '_all'] for r in rows),
                                         max_delta=max(new45[r['index']]['cut_' + s] - new45[r['index']]['cut_' + s + '_all'] for r in rows)) for s in ('F', 'PQ', 'PR')},
                   coverage_valid_effects={str(g): sum(new49[r['index']][f'C{g}'] != new49[r['index']][f'C{g}_all'] for r in rows) for g in (64, 128, 256)},
                   by_domain={d: means([r for r in cat if r['dataset'] == d]) for d in sorted({r['dataset'] for r in cat})})
    # Only re-group sealed natural outputs; historical W48 oracle cuts are not re-scored here.
    summary['regrouped_w48_w51'] = {}
    for group in DIAGNOSTIC_CLASSES:
        sel = [r for r in cat if r['diagnostic'] == group]
        if not sel:
            continue
        keys = [r['index'] for r in sel]
        errs = [w51[k]['err_true'] for k in keys]
        summary['regrouped_w48_w51'][group] = dict(n=len(sel),
            self0_iou=float(np.mean([iou(w48[k]['self']) for k in keys])),
            true_iou=float(np.mean([iou(w48[k]['true']) for k in keys])),
            error_share={s: float(np.mean([r[s] / max(sum(r.values()), 1) for r in errs])) for s in errs[0]},
            mismatched_n=sum('mis' in w48[k] for k in keys))
    OUT.mkdir(exist_ok=True, parents=True)
    summary['probe_support'] = probe_support(rows)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (EV / 'w58_score_repair_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with gzip.open(EV / 'w58_score_repair_cases.jsonl.gz', 'wt') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
