#!/usr/bin/env python3
"""Delete only this finished queue's temporary metric tensors, retain small evidence."""
import argparse
import json
from pathlib import Path
import os


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metric-plan',required=True,type=Path)
    p.add_argument('--evaluation',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path)
    a=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    plan=json.loads(a.metric_plan.read_text())
    if json.loads(a.evaluation.read_text()).get('state')!='COMPLETED':raise ValueError('Do not remove active or failed-stage inputs')
    tensors=Path(plan['tensor_root']).resolve()
    tensors.relative_to(root)
    if tensors==root or tensors.is_symlink() or not tensors.is_dir():raise ValueError('Explicit owned tensor directory required')
    paths=sorted(tensors.rglob('*.pt')); resolved={str(p.resolve()) for p in paths}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            for fd in (proc/'fd').iterdir():
                try:target=os.readlink(fd)
                except OSError:continue
                if target in resolved:raise RuntimeError('Tensor still open by PID '+proc.name)
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
    receipt=[]
    for path in paths:
        if path.is_symlink():raise ValueError('Do not follow a tensor symlink')
        path.resolve().relative_to(tensors)
        receipt.append(dict(path=str(path),bytes=path.stat().st_size))
    for item in receipt:Path(item['path']).unlink()
    value=dict(state='COMPLETED',reason='All frozen predictions scored; temporary full-coordinate inputs no longer in use',
        files=receipt,released_bytes=sum(v['bytes'] for v in receipt),checkpoints_and_metrics_preserved=True)
    a.receipt.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(state='COMPLETED',files=len(receipt),released_bytes=value['released_bytes'])))


if __name__=='__main__':main()
