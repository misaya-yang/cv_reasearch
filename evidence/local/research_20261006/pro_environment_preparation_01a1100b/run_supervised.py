"""Owned M4 CLI subprocess: four threads,8GiB RSS and first-episode180s ceiling."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    namespace, part = Path(sys.argv[1]), sys.argv[2]
    if part not in ('first', 'first_complete', 'remaining', 'case3'):
        raise ValueError('Explicit first/remaining part required')
    code = namespace/('code_ce' if part != 'first' and (namespace/'code_ce').exists() else 'code')
    is_first = part.startswith('first')
    manifest_part = 'first' if is_first else part
    command = ['/root/miniconda3/bin/python', str(code/'scripts/run_pro_paired_environment.py'),
               '--manifest', str(namespace/('manifest_'+manifest_part+'.json')),
               '--expected', '3' if part == 'remaining' else '1', '--out', str(namespace/('sealed_'+part)),
               '--model-dir', '/root/demo4_cache/models/dinov3-vitl16-timm',
               '--encoder-binding', str(namespace/'encoder_binding.json'),
               '--native-binding', str(namespace/'native_binding.json'), '--device', 'cpu', '--threads', '4']
    environment = dict(os.environ, PYTHONPATH='/root/demo4_cache/env:'+str(code/'src'),
                       PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4')
    log_path = namespace/(part+'.log')
    started, peak, state = time.perf_counter(), 0, 'STARTED'
    with log_path.open('w') as log:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        record = dict(pid=child.pid, command=command, owned_new_session=True, part=part,
                      maximum_rss_bytes=(4 if part == 'case3' else 8)*1024**3,
                      first_deadline_seconds=(180 if part == 'first' else 720) if is_first else None,
                      clarified_first_budget='primary clarified180s is no-progress screen; progressing full first allowed720s')
        (namespace/(part+'_ownership.json')).write_text(json.dumps(record, indent=2)+'\n')
        while child.poll() is None:
            try:
                status = Path(f'/proc/{child.pid}/status').read_text().splitlines()
                rss = max(int(line.split()[1])*1024 for line in status if line.startswith(('VmRSS:', 'VmHWM:')))
                peak = max(peak, rss)
            except (FileNotFoundError, ValueError):
                pass
            if peak > record['maximum_rss_bytes']:
                state = 'OWNED_PROCESS_RSS_LIMIT'
            elif is_first and time.perf_counter()-started > (180 if part == 'first' else 720):
                state = 'FIRST_EPISODE_180S_TIMEOUT' if part == 'first' else 'FIRST_EPISODE_720S_TIMEOUT'
            elif part != 'first' and time.time()-log_path.stat().st_mtime > 180:
                state = 'NO_FORWARD_PROGRESS_180S_TIMEOUT'
            if state != 'STARTED':
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(.5)
        if state == 'STARTED':
            state = 'COMPLETED' if child.returncode == 0 else 'FAILED'
    record.update(state=state, exit_code=child.returncode, wall_seconds=time.perf_counter()-started,
                  measured_peak_rss_bytes=peak, log=str(log_path))
    (namespace/(part+'_supervisor.json')).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    sys.exit(0 if state == 'COMPLETED' else 1)


if __name__ == '__main__':
    main()
