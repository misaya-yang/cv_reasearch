"""One owned actual128-view CLS feasibility process,2CPU/8GiB,no-progress guard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    root = Path(sys.argv[1])
    code = root/'code'
    command = ['/root/miniconda3/bin/python', str(code/'scripts/run_object_crop_cls.py'), 'infer',
               '--binding', str(root/'binding.json'), '--out', str(root/'single')]
    environment = dict(os.environ, PYTHONPATH='/root/demo4_cache/env:'+str(code/'src'),
        PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    started, peak = time.perf_counter(), 0
    log_path = root/'infer.log'
    with log_path.open('w') as log:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state = 'RUNNING'
        ownership = dict(pid=child.pid, command=command, owned_new_session=True, CPU_threads=2,
                         RSS_limit_bytes=8*1024**3, one_case_only=True)
        (root/'ownership.json').write_text(json.dumps(ownership, indent=2)+'\n')
        while child.poll() is None:
            try:
                lines = Path(f'/proc/{child.pid}/status').read_text().splitlines()
                peak = max(peak, max(int(line.split()[1])*1024 for line in lines if line.startswith(('VmRSS:', 'VmHWM:'))))
            except (FileNotFoundError, ValueError):pass
            if peak > 8*1024**3:state = 'OWNED_RSS_LIMIT'
            elif time.time()-log_path.stat().st_mtime > 180:state = 'NO_PROGRESS180_SECONDS'
            elif time.perf_counter()-started > 600:state = 'SINGLE_FEASIBILITY600_SECOND_LIMIT'
            if state != 'RUNNING':
                os.killpg(child.pid, signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL);child.wait()
                break
            time.sleep(.5)
        if state == 'RUNNING':state = 'COMPLETED' if child.returncode==0 else 'FAILED'
    receipt = dict(ownership, state=state, exit_code=child.returncode,
                   wall_seconds=time.perf_counter()-started, measured_peak_rss_bytes=peak)
    (root/'supervisor.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    sys.exit(0 if state=='COMPLETED' else 1)


if __name__=='__main__':main()
