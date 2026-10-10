"""Test reference/query class-conditional transfer, without a new segmenter.

The fixed600 panel has already been exposed and sealed by its parent experiment.
generate nevertheless denies all query-label/baseline files and seals its fields
before score opens query labels. This is a diagnostic of a necessary first-moment
condition for reference-based label-shift calibration, not a prior estimator with
guarantees or an original method.
"""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '2')
import argparse
import builtins
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parent/'cv_data'
PARENT = DATA/'a/autonomous_scene_reconstruction600_20261010'
DEFAULT = DATA/'a/reference_query_mixture600_20261010'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


@contextmanager
def deny(paths):
    forbidden = {str(Path(p).resolve()) for p in paths}
    state = {'attempts': 0}
    prior = (builtins.open, io.open)
    def guard(original):
        def opened(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and str(Path(os.fsdecode(file)).resolve()) in forbidden:
                state['attempts'] += 1
                raise PermissionError('Diagnostic generation attempted query-label/baseline access')
            return original(file, *args, **kwargs)
        return opened
    builtins.open, io.open = map(guard, prior)
    try:
        yield state
    finally:
        builtins.open, io.open = prior


def check(out):
    global REPO, DATA, PARENT
    cfg = read(out/'config.json')
    REPO, DATA, PARENT = (Path(cfg[key]) for key in ('repo_root', 'data_root', 'parent_root'))
    assert sha(__file__) == cfg['source_sha256']
    for rel, digest in cfg['parent_hashes'].items():
        assert sha(PARENT/rel) == digest, ('Parent drift', rel)
    assert sha(cfg['profile_path']) == cfg['profile_sha256']
    assert sha(cfg['basis_path']) == cfg['basis_sha256']
    return cfg


def imports(out):
    cfg = check(out)
    # All dependency imports come from the immutable completed parent snapshot.
    sys.path[:0] = [str(PARENT/'frozen/src'), str(PARENT/'frozen/scripts'),
                   str(DATA/'third_party/foris_official')]
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ics.official_data import load_inputs, array_hash
    from raw_feature_cache import RawFeatureCache, tensor_hash
    from utils.data import build_transform
    torch.set_num_threads(2)
    return cfg, np, torch, F, load_inputs, array_hash, RawFeatureCache, tensor_hash, build_transform


def moments(r, q, cov):
    """Coordinate along the reference conditional-mean segment, FP64."""
    import torch
    r, q, c = r.double(), q.double(), cov.double()
    assert bool((c >= 0).all() and (c <= 1).all())
    if not (float(c.sum()) > 0 and float((1-c).sum()) > 0):
        raise ValueError('Diagnostic requires two lawful positive-area reference roles')
    fg = (c[:, None]*r).sum(0)/c.sum()
    bg = ((1-c)[:, None]*r).sum(0)/(1-c).sum()
    delta = fg-bg
    scale = float(delta.square().sum())
    if scale <= 1e-20:
        return None
    mu = q.mean(0)
    score = ((q-bg) @ delta)/scale
    source_score = ((r-bg) @ delta)/scale
    estimate = float(score.mean())
    closed = float((mu-bg) @ delta/scale)
    assert abs(estimate-closed) < 1e-12
    clipped = max(0., min(1., estimate))
    residual = mu-bg-clipped*delta
    source_fg = float((c*source_score).sum()/c.sum())
    source_bg = float(((1-c)*source_score).sum()/(1-c).sum())
    assert abs(source_fg-1) < 1e-10 and abs(source_bg) < 1e-10
    return dict(query_score=score.numpy(), centroids=torch.stack((fg, bg, mu)).numpy(),
        diagnostics=dict(reference_prior=float(c.mean()), unconstrained_query_prior_coordinate=estimate,
            constrained_query_prior_coordinate=clipped,
            coordinate_outside_unit_interval=not 0 <= estimate <= 1,
            conditional_mean_separation=scale**.5,
            convex_segment_residual_over_source_separation=float(residual.norm()/delta.norm()),
            source_conditional_fg_coordinate=source_fg, source_conditional_bg_coordinate=source_bg))


def synthetic_checks():
    import torch
    r = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    c = torch.tensor([1., 1., 0., 0.])
    q = torch.cat((r[:1].repeat(3, 1), r[2:3].repeat(7, 1)))
    value = moments(r, q, c)
    assert abs(value['diagnostics']['unconstrained_query_prior_coordinate']-.3) < 1e-14
    assert value['diagnostics']['convex_segment_residual_over_source_separation'] < 1e-14
    assert moments(torch.ones_like(r), q, c) is None
    mixed = torch.tensor([.8, .2, .1, .3])
    same = moments(r, r, mixed)
    assert abs(same['diagnostics']['unconstrained_query_prior_coordinate']-float(mixed.double().mean())) < 1e-12
    return dict(exact_label_shift_prior=True, zero_source_separation_is_explicit_null=True,
                fractional_area_self_reconstruction=True)


def prepare(out):
    assert not (out/'config.json').exists(), 'Refuse to replace a prepared diagnostic'
    parent = read(PARENT/'config.json')
    seal = read(PARENT/'sealed.json')
    assert seal['n'] == len(seal['receipts']) == 600
    assert seal['config_sha256'] == sha(PARENT/'config.json') and seal['query_GT_read'] is False
    assert sha(PARENT/'manifest.json') == parent['manifest_sha256']
    assert sha(PARENT/'tasks.json') == parent['tasks_sha256']
    manifest, tasks = read(PARENT/'manifest.json'), read(PARENT/'tasks.json')
    assert len(manifest) == len(tasks) == 600
    assert [r['dataset'] for r in manifest] == ['deepglobe_road']*100+['paco_part']*100+['coco']*200+['lvis']*200
    hashes = {rel: sha(PARENT/rel) for rel in ('config.json', 'manifest.json', 'tasks.json', 'sealed.json')}
    for rel, digest in parent['source_sha256'].items():
        assert sha(PARENT/'frozen'/rel) == digest
        hashes['frozen/'+rel] = digest
    external = parent['external_source_sha256']
    for path, digest in external.items():
        assert sha(path) == digest
    out.mkdir(parents=True, exist_ok=True)
    (out/'fields').mkdir(exist_ok=True)
    (out/'records').mkdir(exist_ok=True)
    (out/'frozen').mkdir(exist_ok=True)
    shutil.copyfile(__file__, out/'frozen/diagnose.py')
    forbidden = [r['query_mask_path'] for r in manifest]
    forbidden += [b['path'] for t in tasks for b in t['baselines'].values()]
    write(out/'config.json', dict(n=600, created_utc=datetime.now(timezone.utc).isoformat(),
        repo_root=str(REPO), data_root=str(DATA), parent_root=str(PARENT),
        source_sha256=sha(__file__), parent_hashes=hashes, external_source_sha256=external,
        profile_path=parent['profile_path'], profile_sha256=parent['profile_sha256'],
        basis_path=parent['basis_path'], basis_sha256=parent['basis_sha256'], denied_paths=forbidden,
        representations=['rawunit', 'source_apd'], primary='source_apd',
        question='Do reference and query class-conditional first moments transfer sufficiently for a source-based label-shift prior?',
        coordinate='s(q)=(q-mu_RB).(mu_RF-mu_RB)/||mu_RF-mu_RB||^2; pi_hat=mean_Q s',
        calibration_assumption='mu_QF=mu_RF and mu_QB=mu_RB; necessary moment condition only',
        score='exact pixel-area conditional means, pi error, convex-line residual, and source/prototype ranking',
        geometry='lawful R mask Torch nearest1024 then16pixel area; QGT same geometry after diagnostic seal',
        not_a_segmenter=True, no_threshold_search=True, no_class_prior_selected_by_query_GT=True,
        encoder_constructions=0, cache_writes=0, query_GT_in_generate=False,
        exposure=parent['exposure'], threads=2))
    print(json.dumps(dict(state='PREPARED', n=600, root=str(out))), flush=True)


def generate(out):
    cfg, np, torch, F, load_inputs, array_hash, RawFeatureCache, tensor_hash, build_transform = imports(out)
    assert not (out/'sealed.json').exists(), 'Do not replace a sealed diagnostic'
    profile = Path(cfg['profile_path'])
    cache, transform = RawFeatureCache(profile.parent.parent, read(profile)), build_transform(1024)
    basis = torch.load(cfg['basis_path'], map_location='cpu', weights_only=True)['basis'].float()
    projection = torch.eye(1024)-basis @ basis.T
    manifest, tasks = read(PARENT/'manifest.json'), read(PARENT/'tasks.json')
    parent_receipts = read(PARENT/'sealed.json')['receipts']
    started, records = time.monotonic(), []
    with deny(cfg['denied_paths']) as guard, torch.inference_mode():
        for index, (row, task) in enumerate(zip(manifest, tasks)):
            recpath = out/'records'/f'{index:06d}.json'
            if recpath.exists():
                rec = read(recpath)
                assert rec['episode_id'] == row['episode_id']
                assert sha(out/'fields'/rec['filename']) == rec['fields_sha256']
                records.append(rec)
                continue
            reference, mask, query = load_inputs(row, DATA)
            arrays = []
            for image, request in zip((reference, query), task['raw']):
                x = transform(image).numpy()
                assert cache.key(x) == request['key']
                assert sha(cache.folder/request['key']/'entry.json') == request['entry_sha256']
                array = cache.read(x, ('O/24',))['O/24']
                assert tensor_hash(array) == request['tensor_sha256']
                arrays.append(torch.from_numpy(array))
            r, q = [F.normalize(x, dim=1) for x in arrays]
            hard = F.interpolate(mask.float()[None, None], (64, 64), mode='nearest')[0, 0].flatten() > .5
            mu = q.mean(0)
            semantic = float(F.normalize(r[hard].mean(0), dim=0) @ mu/(mu.norm()+1e-6)) if hard.any() else None
            apd = semantic is None or semantic < .8
            parent_rec = read(PARENT/'records'/f'{index:06d}.json')
            assert parent_rec == parent_receipts[index]
            assert apd == parent_rec['apd_applied']
            mask1024 = F.interpolate(mask.float()[None, None], (1024, 1024), mode='nearest')[0, 0]
            c = mask1024.reshape(64, 16, 64, 16).double().mean((1, 3)).flatten()
            result, values = {}, {}
            for rep in cfg['representations']:
                rr, qq = (r, q) if rep == 'rawunit' or not apd else [F.normalize(x @ projection.T, dim=1) for x in (r, q)]
                value = moments(rr, qq, c)
                if value is None:
                    result[rep] = dict(degenerate=True)
                else:
                    result[rep] = dict(degenerate=False, **value['diagnostics'])
                    values[rep+'/query_coordinate'] = value['query_score']
                    values[rep+'/centroids'] = value['centroids']
            filename = f'{index:06d}.npz'
            np.savez_compressed(out/'fields'/filename, **values)
            rec = dict(index=index, episode_id=row['episode_id'], dataset=row['dataset'], apd=apd,
                filename=filename, fields_sha256=sha(out/'fields'/filename), representations=result,
                raw_requests=task['raw'], reference_mask_array_sha256=array_hash(mask.numpy().astype(np.uint8)),
                query_GT_reads=0, baseline_reads=0, encoder_forward=0, raw_cache_writes=0)
            write(recpath, rec)
            records.append(rec)
            if (index+1) % 50 == 0:
                print(json.dumps(dict(state='GENERATING', completed=index+1, n=600, seconds=time.monotonic()-started)), flush=True)
        assert guard['attempts'] == 0
    assert len(records) == 600
    (out/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records))
    check(out)
    write(out/'sealed.json', dict(state='ALL_DIAGNOSTIC_FIELDS_SEALED', n=600,
        config_sha256=sha(out/'config.json'), records_sha256=sha(out/'records.jsonl'),
        seconds=time.monotonic()-started, query_GT_reads=0, baseline_reads=0,
        synthetic_checks=synthetic_checks()))
    print(json.dumps(dict(state='SEALED', n=600)), flush=True)


def auc(score, c):
    import numpy as np
    order = np.argsort(score, kind='stable')
    s, pos, neg = score[order], c[order], 1-c[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1])+1]
    p, n = np.add.reduceat(pos, starts), np.add.reduceat(neg, starts)
    return float((p*(np.cumsum(n)-.5*n)).sum()/(p.sum()*n.sum()))


def score(out):
    cfg, np, torch, F, _, array_hash, *_ = imports(out)
    seal = read(out/'sealed.json')
    assert seal['state'] == 'ALL_DIAGNOSTIC_FIELDS_SEALED' and seal['n'] == 600
    assert sha(out/'config.json') == seal['config_sha256']
    assert sha(out/'records.jsonl') == seal['records_sha256']
    assert not (out/'results.json').exists(), 'Do not replace an analysis result'
    from PIL import Image
    manifest, rows = read(PARENT/'manifest.json'), []
    records = [json.loads(x) for x in (out/'records.jsonl').read_text().splitlines()]
    for row, rec in zip(manifest, records):
        assert row['episode_id'] == rec['episode_id']
        assert sha(out/'fields'/rec['filename']) == rec['fields_sha256']
        with Image.open(row['query_mask_path']) as image:
            mask = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
        assert array_hash(mask) == row['query_mask_hash']
        full = F.interpolate(torch.from_numpy(mask).double()[None, None], (1024, 1024), mode='nearest')[0, 0]
        c = full.reshape(64, 16, 64, 16).mean((1, 3)).numpy().flatten()
        prior = float(c.mean())
        assert 0 < prior < 1
        result = {}
        with np.load(out/'fields'/rec['filename'], allow_pickle=False) as z:
            for rep, info in rec['representations'].items():
                if info['degenerate']:
                    result[rep] = dict(degenerate=True)
                    continue
                s = z[rep+'/query_coordinate']
                fg, bg = float((c*s).sum()/c.sum()), float(((1-c)*s).sum()/(1-c).sum())
                raw_error = info['unconstrained_query_prior_coordinate']-prior
                decomposition = prior*(fg-1)+(1-prior)*bg
                assert abs(raw_error-decomposition) < 1e-12
                result[rep] = dict(**info, query_prior_GT=prior,
                    signed_prior_error=info['constrained_query_prior_coordinate']-prior,
                    absolute_prior_error=abs(info['constrained_query_prior_coordinate']-prior),
                    query_conditional_fg_coordinate=fg, query_conditional_bg_coordinate=bg,
                    query_conditional_gap=fg-bg, foreground_shift_bias=prior*(fg-1),
                    background_shift_bias=(1-prior)*bg, raw_prior_error=raw_error,
                    bias_decomposition_error=abs(raw_error-decomposition),
                    prototype_direction_area_auc=auc(s, c))
        rows.append(dict(episode_id=row['episode_id'], dataset=row['dataset'], representations=result))
    groups = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        group = [r for r in rows if r['dataset'] == dataset]
        groups[dataset] = {}
        for rep in cfg['representations']:
            valid = [r['representations'][rep] for r in group if not r['representations'][rep]['degenerate']]
            values = {}
            for key in valid[0] if valid else ():
                if isinstance(valid[0][key], (int, float)) and not isinstance(valid[0][key], bool):
                    v = np.asarray([r[key] for r in valid], dtype=float)
                    values[key] = dict(mean=float(v.mean()), median=float(np.median(v)),
                        p10=float(np.quantile(v, .1)), p90=float(np.quantile(v, .9)))
            groups[dataset][rep] = dict(n=len(group), valid=len(valid), moments=values,
                negative_or_zero_query_role_gap=sum(r['query_conditional_gap'] <= 0 for r in valid),
                coordinate_outside_unit_interval=sum(r['coordinate_outside_unit_interval'] for r in valid))
    (out/'scored_episodes.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    write(out/'results.json', dict(n=600, groups=groups, diagnostic_only=True,
        source_sha256=cfg['source_sha256'], seal_sha256=sha(out/'sealed.json'),
        scored_sha256=sha(out/'scored_episodes.jsonl'),
        scope='Necessary first-moment label-shift check; no new segmentation or threshold selection',
        exposure=cfg['exposure'], statistical_claim='Descriptive exposed-panel moments; no significance or untouched-generalization claim'))
    print(json.dumps(dict(state='COMPLETE', n=600)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'generate', 'score', 'synthetic'))
    parser.add_argument('--out', type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == 'synthetic':
        print(json.dumps(synthetic_checks()))
    elif args.command == 'prepare':
        prepare(out)
    elif args.command == 'generate':
        generate(out)
    else:
        score(out)


if __name__ == '__main__':
    main()
