#!/usr/bin/env python3
"""Bounded CPU handoff: verify frozen source, then attach generated native basis."""
import argparse
import hashlib
import json
from pathlib import Path


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',required=True,type=Path)
    p.add_argument('--basis',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    value=json.loads(a.template.read_text())
    if value.get('state')!='PREPARED_SOURCE':raise ValueError('Explicit frozen source template required')
    for path,sha in value['files'].items():
        if checksum(path)!=sha:raise RuntimeError('Prepared code changed: '+path)
    if not a.basis.is_file() or a.basis.stat().st_size<1:raise RuntimeError('Native basis producer missing')
    value['files'][str(a.basis.resolve())]=checksum(a.basis)
    value['basis_contract']='Original INSID3 normalized BLACK-image FP32 U500; produced by native baseline'
    if a.out.exists():raise RuntimeError('Preserve existing receipt; no overwrite')
    a.out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(state='PREPARED_SOURCE',files=len(value['files']))))


if __name__=='__main__':main()
