#!/usr/bin/env python3
"""Write a finite GPU batch plan from an existing manifest; never start it."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--cohort',choices=['smoke4','first20','exposed220','dev241'],default='smoke4')
    p.add_argument('--run-name',required=True)
    p.add_argument('--python',default='/root/miniconda3/bin/python')
    a=p.parse_args();root=a.workspace.resolve()
    if not a.run_name or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in a.run_name):
        p.error('run-name must contain only letters, digits, underscore or hyphen')
    config=root/'evidence/local/research_20261005'
    manifest=config/(a.cohort+'.json')
    out=root/'outputs'/a.run_name
    if out.exists():raise FileExistsError('Use a new run name')
    plan_dir=root/'launch'/a.run_name
    plan_dir.mkdir(parents=True,exist_ok=False)
    code=[str(p.relative_to(root)) for p in sorted((root/'src/ics').rglob('*.py'))]
    code += ['scripts/run_mechanisms.py','scripts/run_intervention.py','scripts/score_forward_run.py','scripts/run_gpu_batch.py']
    plan={'platform':'autodl','cuda_python':a.python,'stages':[{
        'name':'complete_mechanism_batch','kind':'gpu','cwd':str(root),
        'argv':[a.python,'scripts/run_gpu_batch.py','--manifest',str(manifest),'--out',str(out)],
        'timeout_seconds':1800 if a.cohort=='smoke4' else 3600 if a.cohort=='first20' else 14400,
        'code_files':code,
        'requires':[{'path':str(manifest)},{'path':'evidence/local/research_20261005/server_inventory.json'}],
        'produces':[{'path':str(out/'completion.json'),'json_equals':{'state':'COMPLETED'}}],
        'success_checks':[{'path':str(out/'completion.json'),'json_equals':{'state':'COMPLETED'}}]}],
        'authorization':'Prepared only. Run after the user switches on an existing GPU. No rental or shutdown authorization.',
        'cost':'Unknown price and actual GPU runtime. Timeout is a hard ceiling, not a runtime estimate.',
        'cohort':a.cohort}
    (plan_dir/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(plan_dir/'plan.json')

if __name__=='__main__':main()
