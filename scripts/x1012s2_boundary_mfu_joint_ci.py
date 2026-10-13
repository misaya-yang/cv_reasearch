"""Joint photo-cluster CI for the already fixed MFU boundary family.

Freeze this scoring-only rule before new scores: shared query photographs get
the same bootstrap weight in both exposed development cohorts. No GT is read.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import x1012s2_boundary_mfu as run
import numpy as np

OUT, pc = run.OUT, run.pc


def freeze():
    path = OUT/'joint_ci_config.json'
    manifest = pc.read(OUT/'manifest.json')
    overlap = {}
    for d in run.lab.ALL_DOMAINS:
        photos = [{str((run.ASSETS/r['binding']['row']['query_path']).resolve()) for r in manifest
                   if r['cohort'] == c and r['dataset'] == d} for c in run.COHORTS]
        overlap[d] = {'development1360_photos': len(photos[0]), 'development500_photos': len(photos[1]),
                      'shared_photos': len(photos[0] & photos[1]), 'union_photos': len(photos[0] | photos[1])}
    config = {'state': 'JOINT_PHOTO_CLUSTER_SCORE_RULE_FROZEN_BEFORE_NEW_METRICS',
              'method': 'For each domain draw N query-photo clusters with replacement from the union of both cohorts; the same photo multiplicity applies to all of its observations in both cohorts. Recompute observed fold/class pooled I/U, then equal-domain and equal-cohort means.',
              'motivation': 'Cohorts share query photographs; independent cohort bootstrap is retained only as a labeled diagnostic.',
              'repeats': 1000, 'seed': 20261013, 'photo_overlap': overlap,
              'query_GT_reads': 0, 'new_natural_predictions': False, 'fixed_inference_unchanged': True,
              'source_sha256': pc.sha(__file__), 'manifest_sha256': pc.sha(OUT/'manifest.json'),
              'inference_config_sha256': pc.sha(OUT/'config.json')}
    if path.exists():
        assert pc.read(path) == config
    else:
        assert not (OUT/'report.json').exists(), 'Freeze joint scoring before new results'
        pc.write(path, config)
        shutil.copyfile(__file__, OUT/'frozen/joint_ci_x1012s2_boundary_mfu.py')
    return config


def score():
    from scipy.sparse import csr_matrix
    config = freeze()
    report = pc.read(OUT/'report.json')
    assert report['n'] == 1860 and report['query_GT_only_after_all18600_masks_sealed']
    rows, names = report['records'], report['arms']
    rng = np.random.default_rng(config['seed'])
    draws = {a: [] for a in names}
    for d in run.lab.ALL_DOMAINS:
        rs = [r for r in rows if r['dataset'] == d]
        photos = sorted({r['query_photo'] for r in rs})
        pm = {p: i for i, p in enumerate(photos)}
        assert len(photos) == config['photo_overlap'][d]['union_photos']
        weights = rng.multinomial(len(photos), np.full(len(photos), 1/len(photos)), size=config['repeats'])
        cohort_draws = {}
        for cohort in run.COHORTS:
            cr = [r for r in rs if r['cohort'] == cohort]
            slots = sorted({(str(r['fold']), str(r['class_id'])) for r in cr})
            sm = {s: i for i, s in enumerate(slots)}
            pi = [pm[r['query_photo']] for r in cr]
            si = [sm[(str(r['fold']), str(r['class_id']))] for r in cr]
            shape = (len(photos), len(slots))
            presence = csr_matrix((np.ones(len(cr)), (pi, si)), shape=shape)
            present = (weights @ presence) > 0
            assert np.all(present.sum(1) > 0)
            cohort_draws[cohort] = {}
            for a in names:
                ii = csr_matrix(([r['metrics'][a]['I'] for r in cr], (pi, si)), shape=shape)
                uu = csr_matrix(([r['metrics'][a]['U'] for r in cr], (pi, si)), shape=shape)
                values = ((100*(weights @ ii)/np.maximum(weights @ uu, 1))*present).sum(1)/present.sum(1)
                cohort_draws[cohort][a] = values
        for a in names:
            draws[a].append(np.mean([cohort_draws[c][a] for c in run.COHORTS], axis=0))
    macro_draws = {a: np.mean(v, axis=0) for a, v in draws.items()}
    joint = run.make_contrasts(macro_draws, report['equal_cohort_macro10'])
    result = {'state': 'COMPLETE_JOINT_SHARED_PHOTO_CLUSTER_INTERVALS',
              'equal_cohort_macro10': joint, 'config': config,
              'report_sha256': pc.sha(OUT/'report.json'), 'seal_sha256': pc.sha(OUT/'sealed.json'),
              'source_sha256': pc.sha(__file__), 'query_GT_reads': 0,
              'point_estimates_unchanged': True, 'natural_predictions_unchanged': True}
    pc.write(OUT/'joint_photo_cluster_ci.json', result)
    original = OUT/'paired_contrasts_independent_cohort.json'
    if not original.exists():
        shutil.copyfile(OUT/'paired_contrasts.json', original)
    paired = pc.read(original)
    paired['equal_cohort_macro10_independent_cohort_diagnostic'] = paired['equal_cohort_macro10']
    paired['equal_cohort_macro10'] = joint
    paired['equal_cohort_macro10_method'] = config['method']
    paired['joint_photo_cluster_ci_path'] = str(OUT/'joint_photo_cluster_ci.json')
    pc.write(OUT/'paired_contrasts.json', paired)
    completion = pc.read(OUT/'completion.json')
    completion['joint_CI'] = {'path': str(OUT/'joint_photo_cluster_ci.json'),
                               'sha256': pc.sha(OUT/'joint_photo_cluster_ci.json'),
                               'method': config['method'], 'shared_query_photos': sum(v['shared_photos'] for v in config['photo_overlap'].values())}
    pc.write(OUT/'completion.json', completion)
    md = OUT/'report.md'
    marker = '两队列总效应的区间使用跨队列共同query照片簇'
    if marker not in md.read_text():
        with md.open('a') as handle:
            handle.write('\n'+marker+'重采样；同照片在两队列取得相同权重。此评分规则在新分数产生前冻结，点估计和自然mask未改。paired_contrasts.json中的equal_cohort_macro10已使用该区间；独立队列区间仅保留为明确标记的诊断。\n')
    print({'state': result['state'], 'shared_query_photos': completion['joint_CI']['shared_query_photos'],
           'winner': report['winner'], 'winner_vs_G_mfu': joint[report['winner']+'__minus__G_mfu']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'score'))
    args = parser.parse_args()
    run.initialize_cpu()
    if args.mode == 'freeze':
        config = freeze()
        print({'state': config['state'], 'photo_overlap': config['photo_overlap']}, flush=True)
    else:
        score()
