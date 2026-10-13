"""E1: compare fixed extent rules after the same CRF, choosing only on S."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
import time
import numpy as np
import torch
import common as c
import produce
import score

ARMS = ('C1', 'C2', 'UFraw', 'UFraw+', 'Ugm+', 'UF+')


def top_mask(u, k):
    """Use W10's tie-inclusive top-k rule and clamp its requested extent."""
    k = int(min(max(k, 1), u.size - 1))
    threshold = np.partition(u.ravel(), u.size - k)[u.size - k]
    return u >= threshold


def predict(binding):
    fields = c.load_fields(binding)
    candidate = produce.candidate(fields)
    fallback = produce.fallback(fields)
    import repo_methods as repo

    if fallback:
        # Explicit C2 control follows the table in plan B; other arms preserve fallback.
        return {'C1': c.pack(candidate),
                'C2': c.pack(produce.crf(binding, candidate)),
                'UFraw': c.pack(candidate), 'UFraw+': c.pack(candidate),
                'Ugm+': c.pack(candidate), 'UF+': c.pack(candidate)}

    u = c.rc.lab.minmax(torch.from_numpy(c.up(fields['VP_UN']))).numpy()
    k_u = int(candidate.sum())
    foris_raw = c.readout(repo.foris_score(binding))
    foris_full = c.rc.lab.arm_cli(binding, 'B').numpy()
    k_fraw, k_f = int(foris_raw.sum()), int(foris_full.sum())
    ufraw = top_mask(u, k_fraw)
    ugm = top_mask(u, np.sqrt(k_fraw * k_u))
    uf = top_mask(u, k_f)

    # C1 and C2 are identical on non-fallback tasks, so compute that CRF once.
    refined_candidate = c.pack(produce.crf(binding, candidate))
    return {'C1': refined_candidate, 'C2': refined_candidate,
            'UFraw': c.pack(ufraw),
            'UFraw+': c.pack(produce.crf(binding, ufraw)),
            'Ugm+': c.pack(produce.crf(binding, ugm)),
            'UF+': c.pack(produce.crf(binding, uf))}


def worker(binding):
    path = c.prediction_path('E1', binding)
    if not path.exists():
        with c.rc.pc.inference_only(binding):
            c.save_npz(path, **predict(binding))
    return str(binding['index'])


def score_one(binding):
    fields = c.load_fields(binding)
    gt, valid = c.truth(binding)
    record = c.metadata(binding, fields)
    record.update(gt=int(gt.sum()), split='S', rho=float(fields['reference_rho'][0]))
    with np.load(c.prediction_path('E1', binding), allow_pickle=False) as archive:
        for name in ARMS:
            record[name] = c.iu(c.unpack(archive[name]), gt, valid)
    with np.load(c.prediction_path('T0', binding), allow_pickle=False) as archive:
        record['candidate'] = c.iu(c.unpack(archive['candidate']), gt, valid)
    record['foris'] = c.iu(c.rc.lab.arm_cli(binding, 'B').numpy(), gt, valid)
    import repo_methods as repo
    record['foris_noCRF'] = c.iu(c.readout(repo.foris_score(binding)), gt, valid)
    return record


def compare_ufraw(records):
    source = c.ROOT / 'evidence/research/w10_area_transfer_S.jsonl.gz'
    if not source.exists():
        return {'state': 'missing_w10_reference', 'path': str(source)}
    with gzip.open(source, 'rt') as stream:
        old = {str(row['index']): row for row in map(json.loads, stream)}
    diffs_i, diffs_u = [], []
    compared = 0
    for row in records:
        if row['dataset'] == 'isaid' or row['index'] not in old:
            continue
        prior = old[row['index']].get('U@Fraw')
        if prior is None:
            continue
        diffs_i.append(abs(int(row['UFraw'][0]) - int(prior[0])))
        diffs_u.append(abs(int(row['UFraw'][1]) - int(prior[1])))
        compared += 1
    summary = {'state': 'REPORT_ONLY', 'n': compared}
    for label, values in (('abs_I', diffs_i), ('abs_U', diffs_u)):
        a = np.asarray(values, dtype=np.int64)
        summary[label] = ({f'p{q}': float(np.percentile(a, q)) for q in (50, 90, 99)}
                          | {'max': int(a.max())}) if a.size else {}
    return summary


def run(workers):
    out = c.task_path('E1')
    out.mkdir(parents=True, exist_ok=True)
    bindings = c.rows()
    todo = [b for b in bindings if not c.prediction_path('E1', b).exists()]
    started = time.monotonic()
    print(f'E1 predict: {len(todo)} remaining / {len(bindings)}', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool:
        for number, _ in enumerate(pool.map(worker, todo, chunksize=2), 1):
            if number % 200 == 0 or number == len(todo):
                print(f'E1: {number}/{len(todo)}, {time.monotonic() - started:.1f}s', flush=True)
    c.write(out / 'complete.json', {'state': 'PREDICTIONS_COMPLETE', 'n': len(bindings),
                                    'seconds': time.monotonic() - started,
                                    'new_encoder_calls': 0})
    if (out / 'scored.json').exists():
        return
    selected = [b for b in bindings if c.half(b) == 'S']
    print(f'E1 score S: {len(selected)}', flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=c.initialize) as pool:
        records = list(pool.map(score_one, selected, chunksize=8))
    score.save_records(out / 'iu.jsonl.gz', records)
    score.save_records(c.ROOT / 'evidence/research/x1013_E1_iu.jsonl.gz', records)
    identity = compare_ufraw(records)
    c.write(out / 'ufraw_identity.json', identity)
    c.write(out / 'scored.json', {'n': len(records), 'pool_n': len(bindings),
                                  'scope': 'S only', 'T_scored': False,
                                  'identity_gate': 'REPORT_ONLY',
                                  'seconds': time.monotonic() - started,
                                  'UFraw_identity': identity})
    print(f'E1 score complete; UFraw identity: {identity}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error('The plan allows at most six CPU processes.')
    run(args.workers)


if __name__ == '__main__':
    main()
