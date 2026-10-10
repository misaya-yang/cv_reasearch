"""Measure genuine reference-neighbor dependence, before choosing a graph model.

This is an information experiment on fixed cached Deep100/PACO100, not a new
segmentation result. Four role-conditioned copulas strip all additive endpoint
effects. A fixed endpoint shuffle preserves sampled marginal feature clouds.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name, '2')
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent / 'cv_data'
SOURCE = DATA / 'a/autonomous_scene_reconstruction600_20261010'
OUT = DATA / 'a/reference_boundary_copula200_20261010'
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
from autonomous_scene_reconstruction import read, sha, write


def prepare():
    assert not (OUT / 'config.json').exists(), 'Do not replace an experiment'
    config = read(SOURCE / 'config.json')
    seal = read(SOURCE / 'sealed.json')
    assert seal['n'] == 600 and seal['config_sha256'] == sha(SOURCE / 'config.json')
    manifest, tasks = read(SOURCE / 'manifest.json')[:200], read(SOURCE / 'tasks.json')[:200]
    assert len(manifest) == 200 and {r['dataset'] for r in manifest} == {'deepglobe_road', 'paco_part'}
    OUT.mkdir(parents=True, exist_ok=True)
    files = ['scripts/probe_reference_boundary_copula.py', 'scripts/autonomous_scene_reconstruction.py',
             'scripts/raw_feature_cache.py', 'src/ics/__init__.py', 'src/ics/official_data.py',
             'src/ics/methods/__init__.py', 'src/ics/methods/reference_boundary_copula.py']
    frozen = {}
    for relative in files:
        dest = OUT / 'frozen' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, dest)
        frozen[relative] = sha(dest)
    shutil.copyfile(REPO / 'evidence/local/difficult_region_signal_20261010/study.py', OUT / 'frozen/metric_source.py')
    write(OUT / 'manifest.json', manifest)
    write(OUT / 'tasks.json', tasks)
    write(OUT / 'config.json', dict(n=200, source_sha256=frozen,
        profile_path=config['profile_path'], profile_sha256=config['profile_sha256'],
        basis_path=config['basis_path'], basis_sha256=config['basis_sha256'],
        manifest_sha256=sha(OUT / 'manifest.json'), tasks_sha256=sha(OUT / 'tasks.json'),
        source_seal_sha256=sha(SOURCE / 'sealed.json'), threads=2,
        metric='soft independent patch-area role product for same/cut edge; not a pixel-boundary or segmentation mIoU',
        regions=['all', 'FoRIS_foreground_incident', 'FoRIS_border'],
        signals=['true copula delta', 'fixed endpoint shuffle delta', 'factorized zero', 'query endpoint cosine'],
        inference_reads='exact whole R/Q O24 and legal reference mask only',
        query_GT_in_inference=False, encoder_calls=0,
        decision='information test only; no graph optimization or new masks',
        exposure='Same exposed Deep custom100/PACO100, no untouched confirmation',
        metric_source_sha256=sha(OUT / 'frozen/metric_source.py')))
    for folder in ('fields', 'records'):
        (OUT / folder).mkdir(exist_ok=True)


def setup():
    cfg = read(OUT / 'config.json')
    for name, key in (('manifest.json', 'manifest_sha256'), ('tasks.json', 'tasks_sha256')):
        assert sha(OUT / name) == cfg[key]
    for name, digest in cfg['source_sha256'].items():
        assert sha(OUT / 'frozen' / name) == digest
    assert sha(OUT / 'frozen/metric_source.py') == cfg['metric_source_sha256']
    assert sha(__file__) == cfg['source_sha256']['scripts/probe_reference_boundary_copula.py']
    assert sha(cfg['profile_path']) == cfg['profile_sha256']
    assert sha(cfg['basis_path']) == cfg['basis_sha256']
    sys.path[:0] = [str(OUT / 'frozen/src'), str(OUT / 'frozen/scripts'), str(DATA / 'third_party/foris_official')]
    import numpy as np
    import torch
    import torch.nn.functional as F
    from raw_feature_cache import RawFeatureCache
    from ics.official_data import load_inputs, array_hash
    from utils.data import build_transform
    path = OUT / 'frozen/src/ics/methods/reference_boundary_copula.py'
    spec = importlib.util.spec_from_file_location('frozen_reference_copula', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(cfg['threads'])
    profile = Path(cfg['profile_path'])
    cache = RawFeatureCache(profile.parent.parent, read(profile))
    basis = torch.load(cfg['basis_path'], map_location='cpu', weights_only=True)['basis'].float()
    return cfg, np, torch, F, cache, load_inputs, array_hash, build_transform(1024), torch.eye(1024)-basis@basis.T, module


def generate():
    assert not (OUT / 'sealed.json').exists(), 'Do not replace sealed fields'
    cfg, np, torch, F, cache, load_inputs, _, transform, projection, module = setup()
    begun, receipts = time.monotonic(), []
    for i, (row, task) in enumerate(zip(read(OUT / 'manifest.json'), read(OUT / 'tasks.json'))):
        filename = f'{i:06d}.npz'
        record_path = OUT / 'records' / f'{i:06d}.json'
        if record_path.exists():
            rec = read(record_path)
            assert rec['episode_id'] == row['episode_id'] and sha(OUT / 'fields' / filename) == rec['fields_sha256']
            receipts.append(rec)
            continue
        tick = time.monotonic()
        reference, mask, query = load_inputs(row, DATA)
        features = []
        for image, request in zip((reference, query), task['raw']):
            x = transform(image).numpy()
            assert cache.key(x) == request['key']
            assert sha(cache.folder / request['key'] / 'entry.json') == request['entry_sha256']
            features.append(torch.from_numpy(cache.read(x, ('O/24',))['O/24']))
        loading_seconds = time.monotonic()-tick
        tick = time.monotonic()
        r, q = (F.normalize(x, dim=1) for x in features)
        binary = F.interpolate(mask.float()[None, None], (64, 64), mode='nearest')[0, 0].reshape(-1) > .5
        mu = q.mean(0)
        semantic = float(F.normalize(r[binary].mean(0), dim=0) @ mu / (mu.norm()+1e-6)) if binary.any() else None
        apd = semantic is None or semantic < .8
        if apd:
            r, q = (F.normalize(x @ projection.T, dim=1) for x in (r, q))
        mask1024 = F.interpolate(mask.float()[None, None], (1024, 1024), mode='nearest')[0, 0]
        coverage = mask1024.reshape(64, 16, 64, 16).mean((1, 3)).reshape(-1).numpy()
        result = module.probe(r, coverage, q, reference_grid_hw=(64, 64), query_grid_hw=(64, 64))
        edges = np.asarray(result['edge_index'])
        cosine = (q[edges[:, 0]]*q[edges[:, 1]]).sum(1).numpy()
        fields = {name: np.asarray(result[name]) for name in ('edge_index', 'delta_true', 'delta_fixed_shuffle',
                                                           'delta_factorized', 'log_copula_true', 'log_copula_fixed_shuffle')}
        fields['query_cosine'] = cosine
        assert np.all(fields['delta_factorized'] == 0) and len(edges) == 8064
        assert all(np.isfinite(x).all() for x in fields.values())
        np.savez_compressed(OUT / 'fields' / filename, **fields)
        rec = dict(episode_id=row['episode_id'], filename=filename, fields_sha256=sha(OUT / 'fields' / filename),
                   loading_seconds=loading_seconds, head_seconds=time.monotonic()-tick,
                   apd_applied=apd, diagnostics=result['diagnostics'], query_GT_reads=0,
                   baseline_mask_reads=0, encoder_calls=0, raw_writes=0)
        write(record_path, rec)
        receipts.append(rec)
        if (i+1) % 20 == 0:
            state = dict(state='INFERENCE', n=i+1, total=200, seconds=time.monotonic()-begun)
            write(OUT / 'activity.json', state)
            print(json.dumps(state), flush=True)
    write(OUT / 'sealed.json', dict(n=200, receipts=receipts, config_sha256=sha(OUT / 'config.json'),
                                   inference_seconds=time.monotonic()-begun, query_GT_read=False))


def score():
    cfg, np, torch, F, _, _, array_hash, _, _, _ = setup()
    from PIL import Image
    # Reuse the already independently checked exact weighted rank statistic.
    script = OUT / 'frozen/metric_source.py'
    spec = importlib.util.spec_from_file_location('fixed_rank_statistic', script)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    seal = read(OUT / 'sealed.json')
    assert seal['n'] == 200 and seal['config_sha256'] == sha(OUT / 'config.json')
    records = []
    for row, task, rec in zip(read(OUT / 'manifest.json'), read(OUT / 'tasks.json'), seal['receipts']):
        assert row['episode_id'] == rec['episode_id']
        assert sha(OUT / 'fields' / rec['filename']) == rec['fields_sha256']
        with Image.open(row['query_mask_path']) as im:
            gt = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
        assert array_hash(gt) == row['query_mask_hash']
        truth = F.interpolate(torch.from_numpy(gt).float()[None, None], (1024, 1024), mode='nearest')[0, 0].numpy()
        c = truth.reshape(64, 16, 64, 16).mean((1, 3)).reshape(-1)
        baseline = task['baselines']['foris.crf']
        assert sha(baseline['path']) == baseline['sha256']
        with np.load(baseline['path'], allow_pickle=False) as z:
            b = np.unpackbits(z[baseline['keys']['cli1024']], count=1024*1024).reshape(1024, 1024)
        b = b.reshape(64, 16, 64, 16).mean((1, 3)).reshape(-1) > .5
        with np.load(OUT / 'fields' / rec['filename'], allow_pickle=False) as z:
            i, j = z['edge_index'].T
            cut = c[i]*(1-c[j])+(1-c[i])*c[j]
            same = 1-cut
            rois = dict(all=np.ones(len(i), bool), FoRIS_foreground_incident=b[i] | b[j], FoRIS_border=b[i] != b[j])
            values = {}
            for roi, keep in rois.items():
                values[roi] = {name: metric.weighted_rank(z[name][keep], same[keep], cut[keep])
                               for name in ('delta_true', 'delta_fixed_shuffle', 'delta_factorized', 'query_cosine')}
            entry = dict(episode_id=row['episode_id'], dataset=row['dataset'], regions=values,
                         source_pair_diagnostics=rec['diagnostics'])
            records.append(entry)
    (OUT / 'edge_metrics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
    report = {}
    for dataset in ('deepglobe_road', 'paco_part'):
        group = [r for r in records if r['dataset'] == dataset]
        report[dataset] = {}
        for roi in cfg['regions']:
            report[dataset][roi] = {}
            for name in ('delta_true', 'delta_fixed_shuffle', 'delta_factorized', 'query_cosine'):
                valid = [r['regions'][roi][name] for r in group if r['regions'][roi][name]['auc'] is not None]
                report[dataset][roi][name] = dict(n=len(valid), mean_auc=float(np.mean([r['auc'] for r in valid])) if valid else None,
                                                mean_ap=float(np.mean([r['ap'] for r in valid])) if valid else None)
    write(OUT / 'report.json', dict(results=report, metric=cfg['metric'], exposure=cfg['exposure'],
                                   seal_sha256=sha(OUT / 'sealed.json'), metrics_sha256=sha(OUT / 'edge_metrics.jsonl'),
                                   no_segmentation_mIoU_claim=True))
    write(OUT / 'activity.json', dict(state='COMPLETE', n=200))
    print(json.dumps(dict(state='COMPLETE', root=str(OUT))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'generate', 'score'))
    args = parser.parse_args()
    {'prepare': prepare, 'generate': generate, 'score': score}[args.command]()
