#!/usr/bin/env python3
"""One frozen encoder producer and bounded CPU replay of the supplied Astra method.

Inference never opens query annotations. It stores the same FP16 APD cache and
FP32 packet inputs as the earlier evidence cache, plus full FoRIS and raw-origin
masks. Every complete candidate is sealed before the separate score command.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_npz(path, value):
    import numpy as np
    tmp = Path(str(path) + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **value)
    tmp.replace(path)


def load_astra(directory):
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('frozen_astra_run', Path(directory) / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(job):
    """Exact supplied predict_one, unchanged; query labels are not in this cache."""
    import numpy as np
    import torch
    from ics.experiment import sha
    key, feature, packet, astra, destination = job
    torch.set_num_threads(1)
    method = load_astra(astra)
    started = time.monotonic()
    feat = torch.load(feature, weights_only=True, map_location='cpu')
    q, r = feat['q'].float().numpy(), feat['r'].float().numpy()
    with np.load(packet, allow_pickle=False) as p:
        masks, audit = method.predict_one(q, r, p['cov'], p['score'], p['fg_max'], p['bg_max'])
        masks.update({k: p[k].copy() for k in ('native', 'model.raw_nn', 'model.raw_mean')})
    if any(m.dtype != np.uint8 or m.shape != (131072,) for m in masks.values()):
        raise ValueError('All complete masks must use the packed 1024 grid')
    write_npz(Path(destination), masks)
    return dict(key=key, arms=sorted(masks), prediction_sha256=sha(destination),
                feature_sha256=sha(feature), packet_sha256=sha(packet), audit=audit,
                replay_seconds=time.monotonic() - started)


def source_hashes(astra):
    from ics.experiment import sha
    paths = [Path(__file__), REPO / 'src/ics/foris.py', REPO / 'src/ics/data.py',
             REPO / 'src/ics/native_basis.py', REPO / 'src/ics/methods/stage_bank.py']
    paths += sorted(Path(astra).glob('*.py'))
    return {str(p.resolve()): sha(p) for p in paths}


def infer(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from ics.experiment import load_rows, sha
    from ics.data import CACHE
    from ics.foris import build_host, run_foris
    from ics.methods.stage_bank import down_bilinear, model_rows, render_mask

    rows = load_rows(a.manifest)
    man = json.loads(a.manifest.read_text())
    if len(rows) != a.expected or (a.expected < 20 and not a.smoke):
        raise ValueError('Exact cohort size required; small execution checks require --smoke')
    if not torch.cuda.is_available():
        raise RuntimeError('A real GPU is required')
    if a.out.exists():
        raise FileExistsError('Use a fresh run directory; never overwrite a partial run')
    if shutil.disk_usage(a.out.parent).free < len(rows) * 2 * 4096 * 1024 * 2 + 2 * 1024**3:
        raise RuntimeError('Insufficient disk for the FP16 pair cache and output reserve')
    missing = []
    for row in rows:
        # Query annotation files are not accessed during inference or its preflight.
        paths = [Path(man['data_root']) / row[k] for k in ('support', 'query')]
        paths += [Path(man['annotation_root']) / Path(row['support']).with_suffix('.png')]
        missing += [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(str(missing))
    for folder in ('predictions', 'cache/evidence_v1/feat', 'results/extent_v1/run/packets'):
        (a.out / folder).mkdir(parents=True, exist_ok=False)
    write_json(a.out / 'manifest.json', rows)
    sources = source_hashes(a.astra)
    assets = [Path(man['projection_basis']), Path(man['foris_root'])/'models/foris.py',
              Path(CACHE)/'models/dinov3-vitl16-timm/model.safetensors',
              Path(CACHE)/'models/dinov3-vitl16-timm/config.json']
    protocol = dict(source_sha256=sources, source_manifest_sha256=sha(a.manifest),
                    source_manifest=man, astra=str(a.astra.resolve()), expected=a.expected,
                    resolution=1024, encoder_instances=1, worker_count=a.workers,
                    feature_definition='FP16 of observed FoRIS Part1/APD output, not raw DINO',
                    packet_definition='FP32 unquantized APD maxima; area reference coverage; original response',
                    origin_definition='raw normalized l24 nearest reference token, canonical bilinear/nearest/centre labels',
                    candidate='unchanged supplied run.predict_one; fixed external_mean__delete',
                    asset_sha256={str(p):sha(p) for p in assets},
                    parameter_selection='none on this cohort', query_gt_usage='score command after all predictions sealed',
                    native='complete public FoRIS with identical 1024 CRF',
                    insid3='supplied insid3_default cached complete rule; not a bitwise claim about a separate BF16 entrypoint',
                    smoke=a.smoke)
    write_json(a.out / 'protocol.json', protocol)
    # Model initialization is in the parent only. Spawned workers only import the cache algorithms.
    host = build_host(man, 'cuda')
    completed, pending, extraction = {}, set(), {}
    begin = time.monotonic()

    def collect(block):
        nonlocal pending
        ready, pending = wait(pending, timeout=None if block else 0, return_when=FIRST_COMPLETED)
        for future in ready:
            result = future.result()
            completed[result['key']] = result
            write_json(a.out / 'progress.json', dict(extracted=len(extraction), predicted=len(completed),
                       total=len(rows), seconds=time.monotonic()-begin, query_gt_opened=False))
            print(json.dumps(dict(event='candidate_complete', key=result['key'], n=len(completed),
                                  seconds=round(result['replay_seconds'], 2))), flush=True)

    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        with torch.inference_mode():
            for row in rows:
                while len(pending) >= 2 * a.workers:
                    collect(True)
                key, started = row['key'], time.monotonic()
                with Image.open(Path(man['data_root']) / row['support']) as im:
                    support = im.convert('RGB')
                with Image.open(Path(man['data_root']) / row['query']) as im:
                    query = im.convert('RGB')
                with Image.open(Path(man['annotation_root']) / Path(row['support']).with_suffix('.png')) as im:
                    mask = torch.from_numpy((np.asarray(im) == row['c'] + 1).copy())
                native, got, ref_mask, _ = run_foris(host, support, mask, query)
                raw = F.normalize(got['raw'][0], dim=1)
                deb = got['deb'][0]
                q, r = deb[-1].flatten(1).T, deb[0].flatten(1).T
                if q.shape != (4096, 1024) or r.shape != q.shape:
                    raise ValueError('Unexpected FoRIS feature shape')
                cov = F.interpolate(ref_mask[None, None].float(), (64, 64), mode='area')[0, 0]
                sim, fg = q @ r.T, cov.reshape(-1) >= .5
                zeros = torch.zeros(4096, device=q.device)
                _, origins = model_rows(raw[-1].flatten(1).T, raw[0].flatten(1).T, down_bilinear(ref_mask))
                packet = dict(cov=cov, score=got['score'].float(),
                              fg_max=sim[:, fg].amax(1) if bool(fg.any()) else zeros,
                              bg_max=sim[:, ~fg].amax(1) if bool((~fg).any()) else zeros)
                packet = {k: v.cpu().numpy() for k, v in packet.items()}
                packet.update(native=np.packbits(native.cpu().numpy()), pre=np.packbits(got['pre'].cpu().numpy()))
                packet.update({k: np.packbits(render_mask(v).cpu().numpy()) for k, v in origins.items()})
                feature = a.out / 'cache/evidence_v1/feat' / (key + '.pt')
                pp = a.out / 'results/extent_v1/run/packets' / (key + '.npz')
                torch.save(dict(q=q.half().cpu(), r=r.half().cpu(),
                                debiased=bool((raw-deb).abs().max() > 1e-4)), feature)
                write_npz(pp, packet)
                torch.cuda.synchronize()
                extraction[key] = dict(seconds=time.monotonic()-started,
                    rgb_sha256={role: sha(Path(man['data_root'])/row[role]) for role in ('support','query')},
                    reference_mask_sha256=sha(Path(man['annotation_root'])/Path(row['support']).with_suffix('.png')))
                pending.add(pool.submit(replay, (key, str(feature), str(pp), str(a.astra.resolve()),
                                               str(a.out/'predictions'/(key+'.npz')))))
                print(json.dumps(dict(event='encoder_complete', key=key, n=len(extraction),
                                      seconds=round(extraction[key]['seconds'],2))), flush=True)
                del native, got, raw, deb, q, r, cov, sim, packet, origins
                collect(False)
            while pending:
                collect(True)
    if set(completed) != {r['key'] for r in rows} or sources != source_hashes(a.astra):
        raise RuntimeError('Incomplete predictions or source changed')
    arms = next(iter(completed.values()))['arms']
    if any(v['arms'] != arms for v in completed.values()):
        raise RuntimeError('Different arms across episodes')
    write_json(a.out/'audit.json', dict(extraction=extraction, candidates=completed, seconds=time.monotonic()-begin))
    write_json(a.out/'sealed.json', dict(state='ALL_PREDICTIONS_SEALED', n=len(rows), arms=arms,
               manifest_sha256=sha(a.out/'manifest.json'), protocol_sha256=sha(a.out/'protocol.json'),
               predictions={k:v['prediction_sha256'] for k,v in completed.items()},
               inputs={k:{x:v[x] for x in ('feature_sha256','packet_sha256')} for k,v in completed.items()},
               source_sha256=sources, query_gt_opened=False))
    print('ALL_PREDICTIONS_SEALED', len(rows), flush=True)


def score(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from ics.experiment import sha, unpack
    seal = json.loads((a.out/'sealed.json').read_text())
    protocol = json.loads((a.out/'protocol.json').read_text())
    rows = json.loads((a.out/'manifest.json').read_text())
    if seal['state'] != 'ALL_PREDICTIONS_SEALED' or seal['n'] != len(rows):
        raise ValueError('Incomplete sealed cohort')
    for key, file in [('manifest_sha256','manifest.json'),('protocol_sha256','protocol.json')]:
        if sha(a.out/file) != seal[key]:
            raise ValueError('Sealed metadata changed')
    if seal['source_sha256'] != source_hashes(Path(protocol['astra'])):
        raise ValueError('Source changed after inference')
    if (a.out/'report.json').exists():
        raise FileExistsError('Score output already exists')
    sys.path.insert(0, protocol['astra'])
    from statistics_randomstate import analyze
    scored = []
    torch.set_num_threads(1)
    for row in rows:
        key = row['key']; path = a.out/'predictions'/(key+'.npz')
        if sha(path) != seal['predictions'][key]:
            raise ValueError('Prediction changed: '+key)
        annotation = Path(protocol['source_manifest']['annotation_root']) / Path(row['query']).with_suffix('.png')
        with Image.open(annotation) as im:
            truth = torch.from_numpy((np.asarray(im) == row['c']+1).copy())
        truth = F.interpolate(truth[None,None].float(), (1024,1024), mode='nearest')[0,0].bool().numpy()
        with np.load(path, allow_pickle=False) as p:
            masks = {k:unpack(p[k]) for k in seal['arms']}
        iu, edits = {}, {}
        for name, mask in masks.items():
            iu[name] = [int((mask & truth).sum()), int((mask | truth).sum())]
            edits[name] = {}
            for origin in ('model.raw_nn','native','RCG','RCG_count_matched_delete'):
                add, remove = mask & ~masks[origin], masks[origin] & ~mask
                edits[name][origin] = dict(add_TP=int((add&truth).sum()), add_FP=int((add&~truth).sum()),
                                          delete_TP=int((remove&truth).sum()), delete_FP=int((remove&~truth).sum()))
        scored.append(dict(row, iu=iu, edits=edits, query_annotation_sha256=sha(annotation)))
    primary = 'external_mean__delete'
    stats = analyze(scored, baseline='native', contrasts=[(primary,k) for k in
        ('RCG','MEAN_CONTROL','insid3_default','RCG_count_matched_delete','MEAN_count_matched_delete','external_mean_delete_sameK_RCG')])
    (a.out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    write_json(a.out/'report.json', dict(n=len(rows), primary=primary, statistics=stats,
               cohort_exposure=protocol['source_manifest'].get('exposure','unspecified'),
               protocol='1024 class-summed I/U; paired 2000 RandomState(0) photo-connected bootstrap',
               prediction_seal_sha256=sha(a.out/'sealed.json'), query_gt_used_for_selection=False,
               edit_totals={name:{origin:{k:sum(r['edits'][name][origin][k] for r in scored)
                            for k in ('add_TP','add_FP','delete_TP','delete_FP')}
                            for origin in ('model.raw_nn','native','RCG','RCG_count_matched_delete')}
                            for name in seal['arms']}))
    print(json.dumps(stats['point_estimates_pp'], indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('infer','score'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--astra', type=Path)
    p.add_argument('--expected', type=int, default=600)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if not 1 <= a.workers <= 6:
        p.error('workers must be between 1 and 6')
    if a.stage == 'infer' and (a.manifest is None or a.astra is None):
        p.error('infer requires manifest and astra')
    (infer if a.stage == 'infer' else score)(a)


if __name__ == '__main__':
    main()
