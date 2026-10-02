#!/usr/bin/env python3
"""Run only the frozen native table rows using the prepared evaluator and an actual CUDA memory cap."""
import argparse, os, runpy, sys
from pathlib import Path
import torch
ap=argparse.ArgumentParser();ap.add_argument('--file',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--prepared-root',default='/root/autodl-tmp/demo9');a=ap.parse_args()
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC','.3')))
torch.backends.cuda.matmul.allow_tf32=False
script=Path(a.prepared_root)/'scripts/run_episodes.py'
sys.argv=[str(script),'--file',a.file,'--M','15','--rounds','4','--k','5',
          '--variants','agree:tophalf,none:all','--every','25','--out',a.out]
runpy.run_path(str(script),run_name='__main__')
