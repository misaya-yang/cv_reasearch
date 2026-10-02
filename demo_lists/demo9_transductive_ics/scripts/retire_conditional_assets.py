"""Remove only retired own conditional-study tensors after small evidence backup.

Dry-run by default. Standard library only, usable in no-GPU mode. Never touch
shared weights/data, symlinks, or files referenced by live process FDs/maps.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

NAMES=('conditional_ranker_pilot_v1','conditional_ranker_train_v1',
 'conditional_ranker_development_v2','conditional_ranker_development_v2_controls',
 'conditional_ranker_development_v2_controls_pilot10','conditional_ranker_development_v2_train',
 'conditional_ranker_development_v3_cf_train','conditional_counterfactual_v1',
 'conditional_counterfactual_availability_fixed10')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--apply',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve();results=root/'results'
    if root.name!='demo9_transductive_ics':raise ValueError('Own direction root required')
    directories=[results/name for name in NAMES if (results/name).is_dir()]
    tensors=[f for d in directories for f in d.rglob('*.pt') if not f.is_symlink()]
    targets={str(f.resolve()) for f in tensors}
    if any(not f.resolve().is_relative_to(root) for f in tensors):raise ValueError('Tensor outside own root')
    referenced={}
    proc=Path('/proc')
    if not proc.is_dir() and a.apply:raise RuntimeError('Linux live-reference audit unavailable; dry-run only')
    if proc.is_dir():
        for process in proc.iterdir():
            if not process.name.isdigit() or int(process.name)==os.getpid():continue
            try:
                for fd in (process/'fd').iterdir():
                    try:
                        value=str(fd.resolve())
                        if value in targets:referenced.setdefault(value,set()).add(process.name)
                    except OSError:pass
                for line in (process/'maps').read_text().splitlines():
                    fields=line.split(None,5)
                    if len(fields)==6 and fields[-1] in targets:referenced.setdefault(fields[-1],set()).add(process.name)
            except (OSError,PermissionError):continue
    if referenced:raise RuntimeError('In-use assets protected: '+json.dumps({k:sorted(v) for k,v in referenced.items()}))
    records=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size) for f in tensors]
    receipt=dict(state='DRY_RUN',files=len(records),bytes=sum(r['bytes'] for r in records),records=records,
                 reason='Conditional ranking and CF training retired by latest user direction; no raw cache for new encoder method',time=time.time())
    if a.apply:
        archive=results/'conditional_studies_retired_evidence.tgz'
        with tarfile.open(archive,'w:gz') as t:
            for d in directories:
                for f in d.rglob('*'):
                    if f.is_file() and not f.is_symlink() and f.suffix in ('.json','.jsonl','.log','.txt'):
                        t.add(f,arcname=str(f.relative_to(root)))
            for f in (root/'scripts').glob('*.py'):
                if f.name.startswith(('conditional_','offline_','global_','experiment_resource_guard')):
                    t.add(f,arcname=str(f.relative_to(root)))
        receipt.update(archive=str(archive),archive_bytes=archive.stat().st_size,
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        for f in tensors:f.unlink()
        receipt['state']='REMOVED_RETIRED_TENSORS'
        (results/'conditional_retirement_receipt.json').write_text(json.dumps(receipt))
    print(json.dumps({k:v for k,v in receipt.items() if k!='records'}),flush=True)

if __name__=='__main__':main()
