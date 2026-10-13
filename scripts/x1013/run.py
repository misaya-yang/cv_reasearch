"""Execute the fixed T0--T9 plan, with restartable task boundaries.

  python -B scripts/x1013/run.py all --workers 6
  python -B scripts/x1013/run.py T0 --workers 6

Only a fully produced task is scored. S is used for every retention decision;
final T metrics are first opened after the T9 selection is complete.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import common as c
from produce import worker

MOTIVATION = {
    'T0': 'W02 gives the exact 13613-pair candidate; equality permits replay, failure stops dependent tasks.',
    'T1': 'W02 lacks matched MEAN/noCRF: establish the strong control; a larger MEAN deficit raises the priority of class guidance.',
    'T2': 'B4 used 1360 heavy-tailed cases: identify useful components by domain/area; dispensable components become removal proposals, without changing this fixed chain.',
    'T3': 'rho>.9 mixes thin structures with tiny blobs: test only G1; a gain supports its trigger, a loss keeps the original fallback.',
    'T4': 'ISIC and large targets have fragmented evidence, while MEAN has gains: own-feature guidance can confirm class evidence; failure rejects these fixed variants.',
    'T5': 'SUIM/ISIC homogeneous targets fragment: fixed tau=.6 integration can support cluster coherence; failure is retained unchanged.',
    'T6': 'The candidate wins small targets and loses large ones: fixed role symmetry tests figure/ground asymmetry; failure closes these three rules.',
    'T7': 'FoRIS includes CRF while the candidate does not: matched C1/C2 isolate its contribution; retain only under the fixed S criteria.',
    'T8': 'A few tasks dominate extra errors: replay original prefixes to locate E_FP/E_FN; their attribution determines the next targeted mechanism.',
    'T9': 'Assemble only G1/F6a/F1/S-self/C1 in fixed order; S selects, T reports once; ten nonnegative T domains would motivate later unseen-data confirmation.',
    'appendix': 'User-requested fixed M/U control after the plan: paired intervals diagnose complementarity; it cannot change T9.',
}


def produce(task, workers, bindings=None):
    bindings = c.rows() if bindings is None else bindings
    destination = c.task_path(task)
    destination.mkdir(parents=True, exist_ok=True)
    todo = [b for b in bindings if not c.prediction_path(task, b).exists()
            or (task == 'T0' and not c.fields_path(b).exists())]
    started = time.monotonic()
    print(f'{task} produce: {len(todo)} remaining / {len(bindings)}', flush=True)
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=c.initialize) as pool:
        for number, _ in enumerate(pool.map(worker, ((task, b) for b in todo), chunksize=4), 1):
            if number % 200 == 0 or number == len(todo):
                print(f'{task}: {number}/{len(todo)}, {time.monotonic() - started:.1f}s', flush=True)
    elapsed = time.monotonic() - started
    old = c.read(destination / 'complete.json') if (destination / 'complete.json').exists() else {}
    c.write(destination / 'complete.json', {'state': 'PREDICTIONS_COMPLETE', 'n': len(bindings),
            'seconds': old.get('seconds', 0) + elapsed, 'new_encoder_calls': 0})


def execute(task, args):
    print(f'{task} rationale: {MOTIVATION[task]}', flush=True)
    if task in ('T8', 'T9'):
        from diagnostics import run_t8, run_t9
        (run_t8 if task == 'T8' else run_t9)(args.workers)
    else:
        produce(task, args.workers)
        if not args.produce_only:
            from score import score_task
            score_task(task, args.workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['all', 'appendix', *(f'T{i}' for i in range(10))])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--produce-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error('The plan allows at most six CPU processes.')
    tasks = [f'T{i}' for i in range(10)] if args.task == 'all' else [args.task]
    for task in tasks:
        try:
            execute(task, args)
        except Exception as error:
            from report import failure_report
            failure_report(task, error)
            raise
    if args.task == 'all' and not args.produce_only:
        execute('appendix', args)
    if not args.produce_only and args.task == 'all':
        from report import render
        render()


if __name__ == '__main__':
    main()
