"""Add the previously fastest implicit-factor baseline after the in-flight trial."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/takeover_20261001_v1/local_student_head_v1'


def main():
    while True:
        status=json.loads((BASE/'perf_p128/report.json').read_text())['status']
        if status in ('COMPLETED_EXCLUSIVE_COMPILED_DECODER_TIMING','ERROR'):break
        time.sleep(10)
    # The first supervisor has finished its measurements. Release of its CUDA
    # context is handled by the next runner's normal exclusivity check.
    target=BASE/'perf_p128_fourarm'
    if target.exists():raise FileExistsError(target)
    with (BASE/'perf_fourarm.log').open('x') as log:
        code=subprocess.call([sys.executable,'tools/takeover_compact_perf.py','--output-dir',str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if code:raise RuntimeError('Four-arm timing failed; inspect preserved report/traceback')


if __name__=='__main__':main()
