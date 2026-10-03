"""Prepare forty-design -> conditional200-development finite queue, no CUDA."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

BASE = Path(__file__).resolve().parents[1]
from experiment_resource_guard import ResourceGuard


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.relative_to(BASE)
    if directory.exists(): raise ValueError('Fresh finite queue only')
    parent_path = BASE/'results/native_membership_v1/causal_v3/manifest40.json'
    parent = json.loads(parent_path.read_text())
    code = [BASE/'scripts'/n for n in ['query_boundary_cut.py','query_boundary_experiment.py',
        'analyze_query_boundary.py','prepare_query_boundary.py','native_rice_core_experiment.py','experiment_resource_guard.py']]
    code += [BASE/'tics'/n for n in ['native_assets.py','native_decision_trace.py','reference_witness_audit.py',
                                   '__init__.py','imageset.py','propagate.py']]
    inherited = {p:h for p,h in parent['source_hashes'].items() if '/foris_source/' in p or p.endswith('/icx/common.py')}
    for p,h in inherited.items():
        if sha(p) != h: raise ValueError('Pinned strong source drift')
        code.append(Path(p))
    for p in code: compile(p.read_text(),str(p),'exec')
    sources = {str(p):sha(p) for p in code}
    # The existing sampler returns metadata ONLY; no image/model instantiated.
    sys.path.insert(0,'/root/autodl-tmp/demo4')
    from icx.common import coco_episodes
    sampled,_,_ = coco_episodes(0,n=200,shot=1,seed=0)
    from PIL import Image
    rows = [dict(e=e,c=int(row['c']),fold=0,support=row['support'],query=row['query']) for e,row in enumerate(sampled)]
    assets = {a['path']:a for a in parent['assets'] if a['path'].endswith(('model.safetensors','config.json','positional_basis.pt'))}
    for row in rows:
        for role in ('support','query'):
            image = Path(parent['data_root'])/row[role]
            mask = Path(parent['annotation_root'])/Path(row[role]).with_suffix('.png')
            with Image.open(image) as picture: size=picture.size
            with mask.open('rb') as stream:
                header=stream.read(24)
            import struct
            if header[:8]!=b'\x89PNG\r\n\x1a\n' or struct.unpack('>II',header[16:24])!=size:
                raise ValueError('Original JPEG/official PNG geometry mismatch')
            if role=='support':
                with Image.open(mask) as labels:
                    if not labels.histogram()[row['c']+1]:raise ValueError('Empty official reference')
            for p in (image,mask):
                st=p.stat();assets[str(p)]=dict(path=str(p),size=st.st_size,mtime_ns=st.st_mtime_ns)
    previous=json.loads((parent_path.parent/'experiment/report.json').read_text())
    masks={f"{r['fold']}:{r['e']}":hashlib.sha256(zlib.decompress(base64.b64decode(
        r['prediction_bits']['foris_stateful']['data']))).hexdigest() for r in previous['records']}
    directory.mkdir()
    next_parent=directory/'manifest200.json'
    next_parent.write_text(json.dumps({**parent,'frozen_episodes':rows,'fold_count':1,'per_fold':200,
                                      'assets':list(assets.values()),'scope':'fold0 first200 seed0 reused development'}))
    plans=[]
    for count,path,required in [(40,parent_path,True),(200,next_parent,False)]:
        p=directory/f'execution{count}.json'
        p.write_text(json.dumps(dict(stage_name=f'query_boundary{count}',parent_manifest=str(path),
            parent_manifest_sha256=sha(path),source_hashes=sources,native_mask_sha256=masks,
            require_all_native_identity=required,scope=f'official COCO20i seed0 reusedDEV{count}; no method score')))
        plans.append(p)
    guard=directory/'guard.json'
    stages=[]
    for count,p,limit in [(40,plans[0],400),(200,plans[1],1400)]:
        out=directory/f'experiment{count}'
        report=out/'report.json';analysis=out/'analysis.json'
        requires=[dict(path=str(p))]
        if count==200:requires.append(dict(path=str(directory/'experiment40/analysis.json'),
                                          json_equals=dict(further_expansion_signal=True)))
        stages.append(dict(name=f'query_boundary{count}',kind='gpu',cwd=str(BASE),timeout_seconds=limit,
            argv=['/root/miniconda3/bin/python',str(BASE/'scripts/query_boundary_experiment.py'),
                  '--plan',str(p),'--out',str(out),'--guard-state',str(guard),'--allow-gpu'],
            code_files=list(sources),requires=requires,
            produces=[dict(path=str(report),json_equals=dict(state='COMPLETED'))],
            success_checks=[dict(path=str(report),json_equals=dict(state='COMPLETED'))],
            env=dict(DEMO9_CUDA_GUARD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                PYTHONPATH='/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions')))
        stages.append(dict(name=f'boundary_analysis{count}',kind='cpu',role='handoff',cwd=str(BASE),timeout_seconds=30,
            argv=['/root/miniconda3/bin/python',str(BASE/'scripts/analyze_query_boundary.py'),'--report',str(report),'--out',str(analysis)],
            requires=[dict(path=str(report),json_equals=dict(state='COMPLETED'))],
            produces=[dict(path=str(analysis),json_equals=dict(state='CPU_FROZEN_OUTPUT_ANALYSIS'))],
            success_checks=[dict(path=str(analysis),json_equals=dict(state='CPU_FROZEN_OUTPUT_ANALYSIS'))],env=dict(CUDA_VISIBLE_DEVICES='')))
    queue=dict(platform='autodl',cuda_python='/root/miniconda3/bin/python',stages=stages,
               CPU_preparation_complete=True,no_training=True,no_downloads=True)
    (directory/'queue.json').write_text(json.dumps(queue))
    if not ResourceGuard(queue,guard).build_preflight():raise ValueError('Finite preflight failed; no GPU')
    print(json.dumps(dict(state='QUERY_BOUNDARY_FINITE_CPU_PREPARED',GPU_started=False,
                         stage_count=4,first200_reference_geometry_checked=True)))


if __name__=='__main__':main()
