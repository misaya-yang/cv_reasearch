#!/usr/bin/env python3
"""Inspect/validate ten INDEPENDENT cards; CPU-only, no default GPU queue.

No --run switch exists. This prepares concrete algorithm/test artifacts and
records CPU checks, not pretrained-runtime readiness or method superiority.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BOOK=ROOT/'results/prepared_independent_experiments.json'
CHECKS=[
 ('global_content_attention.py',['--self-check']),
 ('native_attention_readout_cpu.py',[]),
 ('native_readout_probe.py',['--self-check']),
 ('reference_metric_cpu.py',['--out',str(ROOT/'results/reference_metric_cpu.json')]),
 ('train_reference_metric.py',['--synthetic-smoke','--out',str(ROOT/'results/reference_metric_trainer_cpu.json')]),
 ('reference_views_cpu.py',['--out',str(ROOT/'results/reference_views_cpu.json')]),
 ('reference_prompt_cpu.py',['--out',str(ROOT/'results/reference_prompt_cpu.json')]),
 ('unbalanced_transport_cpu.py',[]),
 ('reference_diffusion_cpu.py',[]),
 ('host_signed_field_cpu.py',['--out',str(ROOT/'results/host_signed_field_cpu.json')]),
 ('reference_feature_probe.py',['--self-check']),
 ('finite_bank_oracle.py',['--self-check']),
 ('experiment_resource_guard.py',['--self-check']),
]


def validate(book):
    cards=book['cards']
    if len(cards)!=10 or {c['id'] for c in cards}!={f'E{i}' for i in range(1,11)}:
        raise ValueError('Exactly ten distinct independent experiment cards required')
    if sum(c['kind']=='method' for c in cards)!=7:
        raise ValueError('Seven algorithms; diagnostics cannot count as inventions')
    sources={}
    for c in cards:
        if c.get('depends_on_cards') or c.get('requires_other_cards'):
            raise ValueError('No successive pass/fail prerequisites')
        if set(c['prediction_card'])!={'assumption','prediction','pass_decision','fail_decision'}:
            raise ValueError('Each experiment requires all four decision lines')
        if not c['controls'] or not c['cost_cap_stop']:
            raise ValueError('Strong controls and finite cost/failure policies required')
        for name in c['implementation_files']:
            p=(ROOT/name).resolve()
            if ROOT not in p.parents or not p.is_file():raise ValueError('Missing/non-owned code artifact: '+name)
            code=p.read_text();compile(code,str(p),'exec')
            sources[name]=hashlib.sha256(code.encode()).hexdigest()
    return dict(state='PREPARED_CARDS_VALIDATED',cards=10,algorithms=7,diagnostics=3,
                serial_dependencies=0,source_sha256=sources,
                full_pretrained_runtime_verified=False,segmentation_gain_verified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validate',action='store_true');p.add_argument('--list',action='store_true')
    p.add_argument('--card',choices=[f'E{i}' for i in range(1,11)])
    p.add_argument('--show',action='store_true');p.add_argument('--cpu-checks',action='store_true')
    a=p.parse_args();book=json.loads(BOOK.read_text());receipt=validate(book)
    if a.list:
        for c in book['cards']:print(c['id'],c['kind'],c['title'])
    if a.show:
        if not a.card:p.error('--show requires --card')
        print(json.dumps(next(c for c in book['cards'] if c['id']==a.card),ensure_ascii=False,indent=2))
    if a.cpu_checks:
        records=[]
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        for name,options in CHECKS:
            r=subprocess.run([sys.executable,str(ROOT/'scripts'/name),*options],env=environment,
                             capture_output=True,text=True,timeout=90)
            record=dict(script=name,passed=r.returncode==0,returncode=r.returncode,
                        output=r.stdout[-20000:],error=r.stderr[-4000:])
            records.append(record);print(json.dumps(dict(script=name,passed=record['passed'])),flush=True)
            if r.returncode:break
        receipt.update(cpu_checks=records,cpu_checks_passed=len(records)==len(CHECKS) and all(r['passed'] for r in records),
                       scope='Synthetic/source-operation CPU acceptance; no images/pretrained weights/GPU experiment',
                       automatic_GPU_queue=False)
    if a.validate or a.cpu_checks:
        target=ROOT/'results/prepared_methods_cpu_receipt.json'
        target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('source_sha256','cpu_checks')},ensure_ascii=False))
        if a.cpu_checks and not receipt['cpu_checks_passed']:raise SystemExit(1)
    if not (a.validate or a.list or a.show or a.cpu_checks):p.print_help()


if __name__=='__main__':main()
