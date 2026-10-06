"""Owned process-group RSS guard for explicitly authorized Pro M3 cache runs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

BASE = Path('/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b')
SNAPSHOT = BASE / 'code_318a5564bafd'
PYTHON = '/root/miniconda3/bin/python'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def rss_group(pgid):
    total = 0
    pids = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            pid = int(directory.name)
            if os.getpgid(pid) != pgid:
                continue
            lines = (directory / 'status').read_text().splitlines()
            rss = next((int(line.split()[1]) * 1024 for line in lines if line.startswith('VmRSS:')), 0)
            total += rss
            pids.append(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total, pids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('infer', 'score'), required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    label = args.out.name + '.' + args.action
    log = args.out.parent / (label + '.log')
    report_path = args.out.parent / (label + '.guard.json')
    if args.workers < 1 or args.threads < 1 or args.workers * args.threads > 6:
        raise ValueError('This owned run is limited to6 numerical CPU threads')
    thread_count = str(args.threads)
    env = dict(os.environ, PYTHONPATH='/root/demo4_cache/env:' + str(SNAPSHOT / 'src'),
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=thread_count, OPENBLAS_NUM_THREADS=thread_count,
               MKL_NUM_THREADS=thread_count, NUMEXPR_NUM_THREADS=thread_count, VECLIB_MAXIMUM_THREADS=thread_count)
    command = [PYTHON, str(SNAPSHOT / 'scripts/run_pro_role_prediction.py'), args.action, '--out', str(args.out)]
    if args.action == 'infer':
        command += ['--manifest', str(args.manifest), '--root', '/', '--workers', str(args.workers), '--threads', str(args.threads)]
        if args.limit:
            command += ['--limit', str(args.limit)]
    available = sorted(os.sched_getaffinity(0))
    budget = args.workers * args.threads
    # Use a dedicated high-index slice instead of the low-index profile cores.
    allowed = available[-budget:]
    command = ['taskset', '-c', ','.join(map(str, allowed)), *command]
    started = time.perf_counter()
    peak = 0
    stopped = None
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=SNAPSHOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            rss, pids = rss_group(process.pid)
            peak = max(peak, rss)
            if rss > 8 * 1024 ** 3:
                stopped = 'owned_process_group_RSS_exceeded8GiB'
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(.25)
        return_code = process.wait()
    report = dict(command=command, return_code=return_code, stopped=stopped,
                  elapsed_seconds=time.perf_counter() - started,
                  measured_peak_owned_process_group_RSS_bytes=peak,
                  RSS_limit_bytes=8 * 1024 ** 3, CPU_affinity=allowed,
                  workers=args.workers, threads=args.threads, encoder_forwards=0,
                  log_path=str(log), out_path=str(args.out))
    write(report_path, report)
    print(json.dumps(report), flush=True)
    if return_code or stopped:
        raise SystemExit(return_code or 1)


if __name__ == '__main__':
    main()
