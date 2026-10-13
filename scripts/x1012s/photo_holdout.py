"""Thin new-photo evaluation wrapper for the already frozen M/U combination.

Reuses the exact runtime and statistical functions from confirm1000. No method
selection or fitted parameters. Predictions and scoring are separate commands.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '2'
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / 'cv_data'
OUT = ASSETS / 'a/x1013_photo_holdout'
RUNTIME = ASSETS / 'a/x1012s2_runtime/mfu_boundary'
WEIGHTS = ASSETS / 'demo4_cache/models/dinov3-vitl16-timm'
PROGRAM = ROOT / 'evidence/research/composition_final_program_20261013.json'
PROGRAM_SHA = '342b57abf45355fb0b65df86dbbc1cd16eaef730e9a4cf6b93eb7baf4007cc00'
ARMS = ('main', 'U', 'FoRIS', 'MEAN', 'G_mfu')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def runtime():
    source = RUNTIME / 'frozen/x1012s2_runtime.py'
    task = module('photo_holdout_runtime', source)
    config = task.check()
    return task, task.Runtime(config['expressions'], [task.MFU_RULE])


def freeze(out, manifest):
    assert sha(PROGRAM) == PROGRAM_SHA, 'The accepted primary formula changed'
    rows = read(manifest)
    assert rows and len({str(b['index']) for b in rows}) == len(rows)
    for b in rows:
        assert b['canonical_reference_photo'] != b['canonical_query_photo']
    record = dict(manifest_sha256=sha(manifest), program_sha256=PROGRAM_SHA,
                  arms=ARMS, primary='main-U', seed=2026101303, bootstrap_repeats=2000,
                  n=len(rows), domains=dict(Counter(b['dataset'] for b in rows)),
                  inference_source_sha256=sha(RUNTIME / 'frozen/x1012s2_runtime.py'),
                  scorer_source_sha256=sha(ROOT / 'scripts/x1012s2_confirm_score.py'),
                  wrapper_source_sha256=sha(__file__),
                  backbone_frozen=True, fitted_models=0,
                  weights_sha256=sha(WEIGHTS / 'model.safetensors'))
    path = out / 'evaluation.json'
    if path.exists():
        old = read(path)
        for k, v in record.items():
            assert old[k] == (list(v) if isinstance(v, tuple) else v), k
    else:
        record['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
        write(path, record)
    return rows


def five_masks(task, rt, binding, features):
    import numpy as np
    import torch
    masks, mixed, _, detail = rt.run(binding, providers=('B', 'M', 'U', 'T'), provided_raw=features)
    gate = detail['reference_rho'] > .9
    parent = masks['T'] if gate else mixed['MFU']
    if gate:
        main = parent.copy()
    else:
        with task.inference_files(binding):
            _, _, query = rt.load_inputs(binding['row'], ASSETS)
            rgb = rt.host._transform(query)[None]
            main = rt.crf_refine(rt.host._crf, rt.host._crf_band_px, rt.host._crf_p_core,
                                 rgb, torch.from_numpy(parent)).numpy()
    result = dict(main=main, U=masks['U'], FoRIS=masks['B'], MEAN=masks['M'], G_mfu=parent)
    assert all(x.shape == (1024, 1024) and x.dtype == np.bool_ for x in result.values())
    return result, detail


def worker_init():
    global WORKER_TASK, WORKER_RUNTIME
    WORKER_TASK, WORKER_RUNTIME = runtime()


def save_prediction(out, binding, features, metadata):
    import numpy as np
    task, rt = WORKER_TASK, WORKER_RUNTIME
    began = time.monotonic()
    masks, detail = five_masks(task, rt, binding, features)
    key = str(binding['index'])
    path = out / 'predictions' / (key + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **{a: rt.pc.pack(masks[a]) for a in ARMS})
    temp.replace(path)
    record = dict(index=key, dataset=binding['dataset'], prediction_path=str(path),
                  prediction_sha256=sha(path), reference_rho=detail['reference_rho'],
                  query_GT_reads=0, fits=0, head_and_save_seconds=time.monotonic()-began, **metadata)
    write(out / 'records' / (key + '.json'), record)
    return key


def predict(out, manifest):
    import numpy as np
    import torch
    rows = freeze(out, manifest)
    task, rt = runtime()
    encoder_source = (ROOT / 'src/ics/data.py').read_text()
    node = next(n for n in ast.parse(encoder_source).body if isinstance(n, ast.ClassDef) and n.name == 'TimmDINOv3')
    path = out / 'frozen/encoder.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    text = 'import json\nimport torch\n' + ast.get_source_segment(encoder_source, node) + '\n'
    if path.exists():
        assert path.read_text() == text
    else:
        path.write_text(text)
    encoder = module('photo_holdout_encoder', path).TimmDINOv3(str(WEIGHTS)).to('mps').eval().requires_grad_(False)
    assert torch.backends.mps.is_available() and not any(p.requires_grad for p in encoder.parameters())
    # Group repeated reference inputs to avoid re-encoding them. No scores enter ordering.
    ordered = sorted(rows, key=lambda b: (b['row']['reference_rgb_hash'], str(b['index'])))
    cache = OrderedDict()
    started = time.monotonic()
    pending, completed = set(), 0
    # One MPS encoder, two CPU head workers. Bound live tensors to four episodes.
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init) as pool, torch.inference_mode():
        for position, original in enumerate(ordered):
            key = str(original['index'])
            record_path = out / 'records' / (key + '.json')
            if record_path.exists():
                old = read(record_path)
                assert old['manifest_sha256'] == sha(manifest)
                assert sha(old['prediction_path']) == old['prediction_sha256']
                completed += 1
                continue
            began = time.monotonic()
            b = json.loads(json.dumps(original))
            reference, _, query = rt.load_inputs(b['row'], ASSETS)
            inputs = {'reference': rt.host._transform(reference), 'query': rt.host._transform(query)}
            features, b['fit'], encoded = {}, {}, 0
            input_hashes = {role: task.array_hash(x.numpy()) for role, x in inputs.items()}
            missing = [role for role in inputs if input_hashes[role] not in cache]
            fresh = {}
            if missing:
                batch = torch.stack([inputs[role] for role in missing]).to('mps')
                maps = encoder.get_intermediate_layers(batch, n=1, reshape=True)[0]
                torch.mps.synchronize()
                raw_batch = maps.cpu().permute(0, 2, 3, 1).reshape(len(missing), 4096, 1024).contiguous()
                fresh = dict(zip(missing, raw_batch.unbind(0)))
                encoded = len(missing)
            for role, x in inputs.items():
                h = input_hashes[role]
                if h in cache:
                    raw = cache.pop(h)
                    cache[h] = raw
                else:
                    raw = fresh[role]
                    assert raw.dtype == torch.float32 and torch.isfinite(raw).all()
                    if role == 'reference':
                        cache[h] = raw
                        while len(cache) > 4:
                            cache.popitem(last=False)
                features[role] = raw
                entry = out / 'inputs' / (h + '.json')
                meta = dict(file='in_memory.npz', input_tensor_hash=h,
                            storage='FP32 tensor supplied directly; no feature payload written',
                            raw_tensor_sha256=task.array_hash(raw.numpy()))
                if entry.exists():
                    assert read(entry)['input_tensor_hash'] == h
                else:
                    write(entry, meta)
                b['fit']['reference_raw' if role == 'reference' else 'query_raw_identity'] = {
                    'payload_path': str(entry.parent / 'in_memory.npz'), 'entry_path': str(entry)}
            pending.add(pool.submit(save_prediction, out, b, features,
                                    dict(manifest_sha256=sha(manifest), new_encoded_images=encoded,
                                         input_and_encoder_seconds=time.monotonic()-began)))
            if len(pending) >= 4:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    completed += 1
                if completed % 10 < len(done):
                    print(json.dumps(dict(done=completed, total=len(rows), seconds=round(time.monotonic()-started, 1))), flush=True)
        for future in pending:
            future.result()
            completed += 1
    print(json.dumps(dict(done=completed, total=len(rows), seconds=round(time.monotonic()-started, 1))), flush=True)
    records = [read(out / 'records' / (str(b['index']) + '.json')) for b in rows]
    assert all(sha(r['prediction_path']) == r['prediction_sha256'] for r in records)
    write(out / 'sealed.json', dict(state='ALL_FIXED_NATURAL_OUTPUTS_COMPLETE', n=len(rows),
          manifest_sha256=sha(manifest), evaluation_sha256=sha(out/'evaluation.json'),
          arms=ARMS, records=records, completed_at_utc=datetime.now(timezone.utc).isoformat()))


def score(out, manifest):
    import numpy as np
    import torch
    from PIL import Image
    rows = freeze(out, manifest)
    seal = read(out / 'sealed.json')
    assert seal['n'] == len(rows) and seal['manifest_sha256'] == sha(manifest)
    assert seal['evaluation_sha256'] == sha(out / 'evaluation.json')
    by_key = {r['index']: r for r in seal['records']}
    assert set(by_key) == {str(b['index']) for b in rows}
    helper = module('holdout_score_helpers', ROOT / 'scripts/x1012s2_confirm_score.py')
    config = read(out / 'evaluation.json')
    assert sha(ROOT / 'scripts/x1012s2_confirm_score.py') == config['scorer_source_sha256']
    torch.set_num_threads(2)
    records = []
    def resize_mask(path):
        with Image.open(path) as image:
            x = torch.from_numpy((np.asarray(image.convert('L')) > 0).astype(np.float32))
        return torch.nn.functional.interpolate(x[None, None], (1024, 1024), mode='nearest')[0, 0].numpy() > .5
    write(out / 'scoring_started.json', dict(time=datetime.now(timezone.utc).isoformat(), seal_sha256=sha(out/'sealed.json')))
    for b in rows:
        saved, row = by_key[str(b['index'])], b['row']
        assert sha(saved['prediction_path']) == saved['prediction_sha256']
        gt = resize_mask(row['query_mask_path'])
        valid = ~resize_mask(row['query_ignore_mask_path']) if row.get('query_ignore_mask_path') else np.ones_like(gt)
        g, v = np.packbits(gt & valid), np.packbits(valid)
        ng, nv = int((gt & valid).sum()), int(valid.sum())
        with np.load(saved['prediction_path'], allow_pickle=False) as payload:
            assert set(payload.files) == set(ARMS)
            masks = {a: payload[a].copy() for a in ARMS}
        records.append(dict(index=str(b['index']), dataset=b['dataset'], fold=row['fold'],
              class_id=row.get('global_class_id', row['loader_class_id']),
              query_photo=b['canonical_query_photo'], reference_photo=b['canonical_reference_photo'],
              reference_rho=saved['reference_rho'],
              metrics={a: helper.packed_metric(masks[a], g, v, ng, nv) for a in ARMS},
              edits={a: helper.edit_vector(masks['main'], masks[a], g, v) for a in ARMS if a != 'main'}))
    summary = helper.point_summary(records, ARMS)
    paired = helper.global_photo_bootstrap(records, ARMS, summary, repeats=2000, seed=2026101303)
    paired['covered_domains_macro'] = paired.pop('macro10')
    report = dict(state='FIXED_NEW_PHOTO_HOLDOUT_SCORED', n=len(rows), primary='main-U',
                  domains=list(summary), summary=summary, paired_comparisons=paired,
                  macro={a: float(np.mean([d[a]['class_miou'] for d in summary.values()])) for a in ARMS},
                  reference_reuse=Counter(r['reference_photo'] for r in records),
                  interval_scope='Conditional on fixed support assignment; global query-photo bootstrap, reaggregate observed slots each draw.',
                  evaluation_sha256=sha(out/'evaluation.json'), seal_sha256=sha(out/'sealed.json'))
    write(out / 'scores/records.json', records)
    write(out / 'scores/report.json', report)
    print(json.dumps(dict(n=len(rows), macro=report['macro'], primary=paired['covered_domains_macro']['U']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'predict', 'score'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    manifest = args.manifest or args.out / 'manifest.json'
    {'freeze': freeze, 'predict': predict, 'score': score}[args.mode](args.out, manifest)
