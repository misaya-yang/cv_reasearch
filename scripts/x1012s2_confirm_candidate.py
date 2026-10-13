"""Freeze and produce the predetermined ten MFU boundary arms on confirm1000.

Uses sealed original M64 and sealed U/T64. No fitting, method selection,
query-ground-truth access or scoring entry point. Each source readout must
reproduce its natural mask before the one official FoRIS1024 CRF call.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import importlib
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable] = '2'

import hashlib
import json
import numpy as np
import torch

REPO = Path('/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch')
ASSETS = REPO.parent / 'cv_data'
CONFIRM = ASSETS / 'a/x1012s2_confirm1000'
SOURCES = ASSETS / 'a/x1012s2_runtime/confirm1000_sources'
OUT = CONFIRM / 'candidate_sources'
RUNTIME = OUT / 'runtime'
BASE_ARMS = ('mfu','mfu_crf','mfu_or_crf','mfu_and_crf','mfu_area16')
ARMS = BASE_ARMS + tuple('G_'+name for name in BASE_ARMS)
TIMES = ('source_seconds','query_preprocess_seconds','official_crf_seconds','area_reorder_seconds',
         'setop_seconds','gate_seconds','save_seconds','case_seconds')


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare():
    if (OUT/'config.json').exists():
        return check()
    baseline=read(CONFIRM/'baseline/sealed.json')
    sources=read(SOURCES/'sealed.json')
    assert baseline['n']==sources['n']==1000
    assert baseline['foris_complete']==baseline['mean_complete']==1000
    assert baseline['manifest_sha256']==sources['manifest_sha256']==sha(CONFIRM/'manifest.json')
    assert baseline['new_encoder_calls']==baseline['query_GT_reads']==0
    rows=read(CONFIRM/'manifest.baselines.json')
    source_records={r['index']:r for r in sources['records']}
    assert len(rows)==len(source_records)==1000 and {b['index'] for b in rows}==set(source_records)
    for folder in ('runtime','frozen','records','predictions'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    dependencies=[]
    for path in sorted((CONFIRM/'runtime').rglob('*.py')):
        target=RUNTIME/path.relative_to(CONFIRM/'runtime')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        dependencies.append(dict(source=str(path),snapshot=str(target),sha256=sha(target)))
    # Copy the exact existing area algorithm, not an approximation or new rule.
    boundary_source=REPO/'scripts/x1012s2_boundary_composition.py'
    text=boundary_source.read_text();tree=ast.parse(text)
    names=('blocks','unblocks','area_preserving_reorder')
    functions=[next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name) for name in names]
    segments=[ast.get_source_segment(text,node) for node in functions]
    helper=RUNTIME/'area_exact.py'
    helper.write_text('"""Verbatim area functions from frozen development boundary source."""\nimport numpy as np\n\n'+'\n\n'.join(segments)+'\n')
    dependencies.append(dict(source=str(boundary_source),source_sha256=sha(boundary_source),
                             extracted_functions=names,snapshot=str(helper),sha256=sha(helper)))
    frozen=OUT/'frozen/x1012s2_confirm_candidate.py';shutil.copyfile(Path(__file__),frozen)
    dependencies.append(dict(source=str(Path(__file__).resolve()),snapshot=str(frozen),sha256=sha(frozen)))
    for path in (boundary_source,REPO/'scripts/x1012s2_boundary_mfu.py'):
        target=OUT/'frozen'/path.name;shutil.copyfile(path,target)
        dependencies.append(dict(source=str(path),snapshot=str(target),sha256=sha(target)))
    write(OUT/'runtime_sources.json',dependencies)
    shutil.copyfile(CONFIRM/'manifest.json',OUT/'manifest.json')
    bindings=[]
    for b in rows:
        source=source_records[b['index']]
        assert source['dataset']==b['dataset'] and source['query_GT_reads']==source['new_encoder_calls']==0
        assert bool(source['U_raw_T_fallback'])==(float(source['reference_rho'])>.9)
        bindings.append(dict(index=b['index'],dataset=b['dataset'],binding=b,
                             M_field=b['mean64'],M_mask=b['old_arms']['G'],
                             UT_field_path=source['field_path'],UT_field_sha256=source['field_sha256'],
                             UT_prediction_path=source['prediction_path'],UT_prediction_sha256=source['prediction_sha256'],
                             reference_rho=float(source['reference_rho']),gate_true=float(source['reference_rho'])>.9))
    write(OUT/'bindings.json',bindings)
    development=read(ASSETS/'a/x1012s2_boundary_mfu/config.json')
    assert tuple(development['arms'])==ARMS
    config=dict(state='TEN_FIXED_CONFIRM_MFU_BOUNDARY_ARMS_FROZEN_BEFORE_NATURAL_OUTPUTS',n=1000,arms=ARMS,
                p='Original float32 M/U frames cast to FP64, (M+U)/2, strict>.5; no additional normalization or clipping',
                M='Stored M64 field -> original CPU FP32 bilinear1024, no minmax; strict>.5 must reproduce sealed MEAN',
                U='Stored U64 field -> original CPU FP32 bilinear1024 and minmax; strict>.5 must reproduce sealed U',
                T='Stored T64 field -> original CPU FP32 bilinear1024 and minmax; strict>.5 must reproduce sealed T',
                source_identity='Before each CRF call, M/U/T natural masks must be bit-exact. No source repairs, fallback readouts or input mutation.',
                mfu='p>.5',mfu_crf='One unchanged official FoRIS1024 CRF(mfu, original query RGB preprocessing)',
                mfu_or_crf='mfu OR same CRF output',mfu_and_crf='mfu AND same CRF output',
                mfu_area16='Exact development area_preserving_reorder: descending CRF binary, descending continuous p, ascending row-major; preserve each16x16 original MFU count',
                gate='G_arm = original sealed T bits if reference_rho>.9 else arm, for all five base arms',
                crf_parameters=development['crf_parameters'],workers=2,threads_per_worker=2,device='cpu',
                reference_count=1,new_encoder_calls=0,training=False,parameter_fitting=False,
                query_GT_reads=0,query_GT_scored=False,selection='No selection here. Primary formula is chosen only using the two development cohorts.',
                all10000_natural_masks_before_any_confirm_scoring=True,
                manifest_sha256=sha(OUT/'manifest.json'),bindings_sha256=sha(OUT/'bindings.json'),
                baseline_seal_path=str(CONFIRM/'baseline/sealed.json'),baseline_seal_sha256=sha(CONFIRM/'baseline/sealed.json'),
                UT_seal_path=str(SOURCES/'sealed.json'),UT_seal_sha256=sha(SOURCES/'sealed.json'),
                development_config_path=str(ASSETS/'a/x1012s2_boundary_mfu/config.json'),
                development_config_sha256=sha(ASSETS/'a/x1012s2_boundary_mfu/config.json'),
                runtime_sources_sha256=sha(OUT/'runtime_sources.json'),
                official_refinement_sha256=sha(ASSETS/'third_party/foris_official/utils/refinement.py'),
                basis_sha256=sha(ASSETS/'native_assets/positional_basis.pt'),
                latency_scope='Boundary library generation only; existing source/encoder production excluded. All five ungated arms are retained, so CRF is executed even when gated arms copy T.')
    write(OUT/'config.json',config)
    print(dict(state=config['state'],n=1000,arms=ARMS,config_sha256=sha(OUT/'config.json')),flush=True)
    return config


def check():
    config=read(OUT/'config.json')
    assert sha(OUT/'manifest.json')==config['manifest_sha256']==sha(CONFIRM/'manifest.json')
    assert sha(OUT/'bindings.json')==config['bindings_sha256']
    assert sha(OUT/'runtime_sources.json')==config['runtime_sources_sha256']
    for r in read(OUT/'runtime_sources.json'):assert sha(r['snapshot'])==r['sha256']
    for key in ('baseline','UT','development_config'):
        path_key=key+'_seal_path' if key in ('baseline','UT') else key+'_path'
        hash_key=key+'_seal_sha256' if key in ('baseline','UT') else key+'_sha256'
        assert sha(config[path_key])==config[hash_key]
    assert sha(ASSETS/'third_party/foris_official/utils/refinement.py')==config['official_refinement_sha256']
    return config


def initialize():
    global PC,LAB,HOST,ADAPTER,LOAD,CRF_REFINE,AREA,CONFIG_SHA
    sys.path.insert(0,str(RUNTIME))
    import lab
    lab.ROOT,lab.ASSETS=REPO,ASSETS;lab.BASIS=ASSETS/'native_assets/positional_basis.pt'
    import plan_common as pc
    sys.path.insert(0,str(RUNTIME))
    pc.setup('cpu');torch.set_num_threads(2);torch.set_num_interop_threads(1)
    HOST,ADAPTER=pc.host_cpu()
    from ics.official_data import load_inputs
    from utils.refinement import crf_refine
    from area_exact import area_preserving_reorder
    PC,LAB,LOAD,CRF_REFINE,AREA=pc,lab,load_inputs,crf_refine,area_preserving_reorder
    assert HOST.image_size==1024 and HOST._crf_band_px==10 and HOST._crf_p_core==.95
    assert ADAPTER.encoder is None and ADAPTER.encoder_calls==0
    CONFIG_SHA=sha(OUT/'config.json')


def load_sources(row):
    field=row['M_field'];natural=row['M_mask']
    for path,h in ((field['path'],field['sha256']),(natural['path'],natural['sha256']),
                   (row['UT_field_path'],row['UT_field_sha256']),
                   (row['UT_prediction_path'],row['UT_prediction_sha256'])):assert sha(path)==h
    with np.load(field['path'],allow_pickle=False) as z:
        m=LAB.up(torch.as_tensor(z[field['key']],dtype=torch.float32).reshape(64,64)).numpy()
    with np.load(natural['path'],allow_pickle=False) as z:m_bits=z[natural['cli_key']].copy()
    with np.load(row['UT_field_path'],allow_pickle=False) as z:
        u=LAB.minmax(LAB.up(torch.as_tensor(z['U'],dtype=torch.float32).reshape(64,64))).numpy()
        t=LAB.minmax(LAB.up(torch.as_tensor(z['T'],dtype=torch.float32).reshape(64,64))).numpy()
    with np.load(row['UT_prediction_path'],allow_pickle=False) as z:
        u_bits,t_bits=z['U'].copy(),z['T'].copy()
    xors={name:int(np.unpackbits(PC.pack(frame>.5)^bits).sum())
          for name,frame,bits in (('M',m,m_bits),('U',u,u_bits),('T',t,t_bits))}
    assert not any(xors.values()), (row['index'],'Original source natural-mask mismatch',xors)
    assert all(x.dtype==np.float32 and x.shape==(1024,1024) and np.isfinite(x).all() for x in (m,u,t))
    p=(m.astype(np.float64)+u.astype(np.float64))/2
    a=p>.5
    assert np.array_equal(a,(.5*m.astype(np.float64)+.5*u.astype(np.float64))>.5)
    return a,p,t_bits,xors


def produce_one(row):
    key=row['index'];record_path=OUT/'records'/(key+'.json')
    if record_path.exists():
        old=read(record_path);assert old['config_sha256']==CONFIG_SHA
        assert sha(old['prediction_path'])==old['prediction_sha256'];return old
    began=time.monotonic()
    try:
        with PC.inference_only(row['binding']):
            tick=time.monotonic();a,p,t_bits,xors=load_sources(row);source_seconds=time.monotonic()-tick
            tick=time.monotonic();_,_,query=LOAD(row['binding']['row'],ASSETS)
            HOST.set_target(query);image=HOST._tgt_image.clone().unsqueeze(0)
            assert image.dtype==torch.float32 and tuple(image.shape)==(1,3,1024,1024)
            preprocess=time.monotonic()-tick
            tick=time.monotonic();crf=CRF_REFINE(HOST._crf,HOST._crf_band_px,HOST._crf_p_core,image,torch.from_numpy(a)).numpy()
            crf_seconds=time.monotonic()-tick
            tick=time.monotonic();area,area_info=AREA(a,crf,p);area_seconds=time.monotonic()-tick
            tick=time.monotonic()
            masks={'mfu':PC.pack(a),'mfu_crf':PC.pack(crf),'mfu_or_crf':PC.pack(a|crf),
                   'mfu_and_crf':PC.pack(a&crf),'mfu_area16':PC.pack(area)}
            setop=time.monotonic()-tick
            tick=time.monotonic()
            assert row['gate_true']==(row['reference_rho']>.9)
            masks.update({'G_'+name:t_bits if row['gate_true'] else masks[name] for name in BASE_ARMS})
            gate=time.monotonic()-tick
            assert ADAPTER.encoder is None and ADAPTER.encoder_calls==0
            path=OUT/'predictions'/(key+'.npz');assert not path.exists(),'Preserve incomplete natural output'
            tick=time.monotonic();tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**masks);tmp.replace(path)
            save=time.monotonic()-tick
        record=dict(index=key,dataset=row['dataset'],cohort='confirm1000',
                    episode_id=row['binding']['row']['episode_id'],prediction_path=str(path),prediction_sha256=sha(path),
                    config_sha256=CONFIG_SHA,reference_rho=row['reference_rho'],gate_true=row['gate_true'],
                    source_identity_xor=xors,mfu_arithmetic_identity_xor=0,area_reorder=area_info,
                    M_field_sha256=row['M_field']['sha256'],UT_field_sha256=row['UT_field_sha256'],
                    UT_prediction_sha256=row['UT_prediction_sha256'],
                    timing=dict(source_seconds=source_seconds,query_preprocess_seconds=preprocess,
                                official_crf_seconds=crf_seconds,area_reorder_seconds=area_seconds,
                                setop_seconds=setop,gate_seconds=gate,save_seconds=save,case_seconds=time.monotonic()-began),
                    seconds=time.monotonic()-began,official_crf_calls=1,threads=torch.get_num_threads(),
                    new_encoder_calls=0,query_GT_reads=0,query_GT_scored=False,training=False,parameter_fitting=False)
        write(record_path,record);return record
    finally:
        HOST._ref_images=HOST._ref_masks=HOST._tgt_image=HOST._orig_tgt_size=None


def produce():
    config=check()
    if (OUT/'sealed.json').exists():
        print(dict(state=read(OUT/'sealed.json')['state'],already_sealed=True),flush=True);return
    rows=read(OUT/'bindings.json');began=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for i,record in enumerate(pool.map(produce_one,rows,chunksize=1),1):
            records.append(record)
            if i<=3 or i%50==0:
                progress=dict(completed=i,n=len(rows),dataset=record['dataset'],seconds=round(time.monotonic()-began,3),
                              source_identity_xor=record['source_identity_xor'],query_GT_reads=0)
                write(OUT/'progress.json',progress);print(progress,flush=True)
    timing={k:dict(mean=float(np.mean([r['timing'][k] for r in records])),
                   sum=float(sum(r['timing'][k] for r in records))) for k in TIMES}
    seal=dict(state='ALL1000_TEN_FIXED_MFU_BOUNDARY_ARMS_SEALED_UNSCORED',n=1000,mask_n=10000,arms=ARMS,
              records=records,config_sha256=sha(OUT/'config.json'),manifest_sha256=config['manifest_sha256'],
              bindings_sha256=config['bindings_sha256'],wall_seconds=time.monotonic()-began,timing=timing,
              source_identity_xor_sum={s:sum(r['source_identity_xor'][s] for r in records) for s in ('M','U','T')},
              gate_true_n=sum(r['gate_true'] for r in records),official_crf_calls=sum(r['official_crf_calls'] for r in records),
              logical_selected_gated_crf_seconds=sum(r['timing']['official_crf_seconds'] for r in records if not r['gate_true']),
              workers=2,threads_per_worker=2,new_encoder_calls=0,query_GT_reads=0,query_GT_scored=False,
              training=False,parameter_fitting=False,primary_formula_selected=False)
    assert len(records)==1000 and not any(seal['source_identity_xor_sum'].values())
    write(OUT/'sealed.json',seal)
    print({k:v for k,v in seal.items() if k not in ('records','timing')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','produce','run'))
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='produce':produce()
    else:prepare();produce()
