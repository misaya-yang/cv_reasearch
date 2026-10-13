"""Complete released INSID3 on cached O24: bilinear and official CRF640.

No fitting, FoRIS score input, feature encoding, or query-label inference.
Official candidate localization, clustering, seed selection, aggregation, and
finalization execute unmodified. R5's 500 pairs are already exposed diagnostics.
The 1360 development tasks are also exposed. Predictions are sealed before any
query masks are read for scoring.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sys
import time
import types

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
sys.dont_write_bytecode = True

REPO = Path('/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch')
ASSETS = REPO.parent / 'cv_data'
OUT = ASSETS / 'a/x1012_R6_insid3'
RUNTIME = OUT / 'runtime'
OFFICIAL = RUNTIME / 'official'
OLD = ASSETS / 'a/insid3_task_structure60_20261010'
TRACE = ASSETS / 'a/reference_mask_task_trace_20261010'
R5 = ASSETS / 'a/x1012_R5_validation'
CANDIDATE = ASSETS / 'a/x1012_R5_candidate/final_tau0475'
ARMS = ('insid3.bilinear', 'insid3.crf640')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze_runtime():
    if (OUT / 'runtime_sources.json').exists():
        for row in read(OUT / 'runtime_sources.json'):
            assert sha(row['snapshot']) == row['sha256']
        return
    sources = {}
    for name in ('lab.py', 'plan_common.py', 'p0_manifest.py', 'raw_feature_cache.py',
                 'ics/__init__.py', 'ics/official_data.py', 'ics/native_basis.py', 'ics/m4_crf_cached.py'):
        sources[name] = R5 / 'runtime' / name
    origin = ASSETS / 'third_party/INSID3'
    for folder in ('models', 'utils'):
        for path in sorted((origin / folder).glob('*.py')):
            sources[str(Path('official') / folder / path.name)] = path
    records = []
    for relative, source in sources.items():
        target = RUNTIME / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        assert source.is_file(), 'Missing INSID3 runtime source ' + str(source)
        if target.exists():
            assert sha(target) == sha(source)
        else:
            shutil.copyfile(source, target)
        records.append(dict(source=str(source), snapshot=str(target), sha256=sha(target)))
    for folder in ('models', 'utils'):
        marker = OFFICIAL / folder / '__init__.py'
        if not marker.exists():
            marker.write_text('"""Isolated unmodified INSID3 modules."""\n')
            records.append(dict(source='isolated namespace marker', snapshot=str(marker), sha256=sha(marker)))
    frozen = OUT / 'frozen/r6_insid3.py'
    frozen.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), frozen)
    records.append(dict(source=str(Path(__file__).resolve()), snapshot=str(frozen), sha256=sha(frozen)))
    write(OUT / 'runtime_sources.json', records)


def modules():
    sys.path.insert(0, str(RUNTIME))
    lab = importlib.import_module('lab')
    lab.ROOT, lab.ASSETS = REPO, ASSETS
    lab.BINDINGS = ASSETS / 'a/consensus_reference_calibration6380_v1_20261011/source_bindings.json'
    lab.BASIS = ASSETS / 'native_assets/positional_basis.pt'
    pc = importlib.import_module('plan_common')
    sys.path.insert(0, str(RUNTIME))
    return lab, pc


def prepare():
    freeze_runtime()
    if (OUT / 'config.json').exists():
        check()
        print({'state': 'R6_INSID3_ALREADY_FROZEN'}, flush=True)
        return
    lab, pc = modules()
    cases500 = read(R5 / 'manifest.json')
    development = read(ASSETS / 'a/x1012_R4_global/manifest.json')
    assert len(cases500) == 500 and len(development) == 1360
    assert [pc.key(b) for b in development] == [pc.key(b) for b in pc.seed_rows(150)]
    # Ten old natural tasks cover all three archived domains; no query labels.
    pairs = read(TRACE / 'pairs.json')
    chosen = ([p for p in pairs if p['dataset'] == 'coco'][:4] +
              [p for p in pairs if p['dataset'] == 'paco_part'][:3] +
              [p for p in pairs if p['dataset'] == 'suim'][:3])
    probes = []
    for pair in chosen:
        row = copy.deepcopy(pair['row'])
        row.update(reference_mask_path=pair['variants']['A']['path'],
                   reference_mask_hash=pair['variants']['A']['array_hash'])
        raw = {}
        for old_role, name in (('reference', 'reference_raw'), ('query', 'query_raw_identity')):
            entry_path = Path(pair['raw_binding'][old_role]['entry'])
            assert sha(entry_path) == pair['raw_binding'][old_role]['entry_sha256']
            entry = read(entry_path)
            raw[name] = dict(payload_path=str(entry_path.parent / entry['file']))
        probes.append(dict(index=pair['id'], dataset=pair['dataset'], row=row, fit=raw,
                           old_insid3=dict(prediction_path=str(OLD / 'predictions' / (pair['id'] + '.npz')),
                                          state_path=str(OLD / 'states' / (pair['id'] + '.npz')))))
    for cohort, entries in [('probe10', probes), ('r5_500', cases500), ('development1360', development)]:
        write(OUT / cohort / 'manifest.json', entries)
        for folder in ('predictions', 'records'):
            (OUT / cohort / folder).mkdir(parents=True, exist_ok=True)
    old_config = read(OLD / 'config.json')
    for relative in ('models/insid3.py', 'utils/clustering.py', 'utils/data.py', 'utils/refinement.py'):
        assert sha(OFFICIAL / relative) == old_config['source_sha256'][str(ASSETS / 'third_party/INSID3' / relative)]
    controls = [R5 / 'manifest.baselines.json', R5 / 'baseline/sealed.json',
                CANDIDATE / 'config.json', CANDIDATE / 'sealed.json',
                TRACE / 'pairs.json', OLD / 'config.json', OLD / 'sealed.json']
    controls += [ASSETS / 'native_assets/positional_basis.pt',
                 ASSETS / 'runtime/macos/crf/ics_permutohedral_m4.so',
                 ASSETS / 'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so']
    config = dict(state='R6_INSID3_FROZEN', arms=list(ARMS), reference_count=1,
                  official=dict(image_size=1024, svd_components=500, tau=.6, merge_threshold=.2,
                                mask_refiner='crf', crf_size=640, resize_to_orig_size=False,
                                device='cpu'),
                  bilinear='Read-only tap before original _finalize_mask; original upsample_mask(mask64,1024,1024)',
                  crf='Original INSID3 _finalize_mask at its default CRF640, followed by original binary bilinear to CLI1024',
                  parameter_fitting=False, cross_task_training=False, query_GT_in_inference=False,
                  new_encoder_calls=0, FoRIS_score_consumed=False,
                  positional_basis='Reuse frozen native 1024x500 basis; no black-image encoding/SVD',
                  workers=2, threads_per_worker=4, torch_interop_threads=1, seed=0,
                  runtime_sources_sha256=sha(OUT / 'runtime_sources.json'),
                  manifest_sha256={c: sha(OUT / c / 'manifest.json') for c in ('probe10', 'r5_500', 'development1360')},
                  controls_sha256={str(p): sha(p) for p in controls},
                  cohort_scope={'r5_500': 'Already exposed 500 old-photo directed pairs; diagnostic only',
                                'development1360': 'Fixed exposed development tasks, same pc.seed_rows(150)'},
                  metric='CLI1024 observed (fold,class) pooled intersection/union, then equal-slot mean; photo-cluster bootstrap',
                  runtime_parity='CPU cache FP32, same cached DINO inputs as FoRIS; published hub/CUDA numerical identity unclaimed')
    write(OUT / 'config.json', config)
    print(dict(state=config['state'], cohorts={c:len(read(OUT / c / 'manifest.json')) for c in config['manifest_sha256']},
               official=config['official']), flush=True)


def check():
    config = read(OUT / 'config.json')
    assert sha(OUT / 'runtime_sources.json') == config['runtime_sources_sha256']
    for row in read(OUT / 'runtime_sources.json'):
        assert sha(row['snapshot']) == row['sha256'], 'Frozen source changed'
    for cohort, expected in config['manifest_sha256'].items():
        assert sha(OUT / cohort / 'manifest.json') == expected
    for path, expected in config['controls_sha256'].items():
        assert sha(path) == expected, 'Frozen input changed: ' + path
    return config


def init_worker():
    global LAB, PC, HOST, ADAPTER, LOAD, UPSAMPLE, CAPTURE
    LAB, PC = modules()
    PC.setup('cpu')
    import numpy as np
    import torch
    torch.set_num_interop_threads(1)
    np.random.seed(0); random.seed(0)
    native = PC.extension('ics_permutohedral_m4', ASSETS / 'runtime/macos/crf/ics_permutohedral_m4.so')
    prepared = PC.extension('ics_permutohedral_m4_prepared_v1', ASSETS / 'runtime/macos/crf/prepared_v1/ics_permutohedral_m4_prepared_v1.so')
    class Layer(torch.nn.Module):
        def __init__(self, bilateral, theta_alpha, theta_beta, theta_gamma, nhwc=True):
            super().__init__(); self.bilateral=bilateral; self.parameters_=(theta_alpha, theta_beta, theta_gamma)
        def forward(self, values, image):
            v=values.detach().float().permute(0,2,3,1).contiguous()
            c=image.detach().float().permute(0,2,3,1).contiguous()
            return native.forward(v,c,self.bilateral,*self.parameters_).permute(0,3,1,2).contiguous()
    replacement=types.ModuleType('PermutohedralFiltering'); replacement.PermutohedralLayer=Layer
    sys.modules['PermutohedralFiltering']=replacement
    sys.path.insert(0, str(ASSETS / 'third_party/crf_source/src'))
    sys.path.insert(0, str(OFFICIAL))
    from models.insid3 import INSID3
    from utils.refinement import upsample_mask
    from ics.native_basis import reuse_native_basis
    from ics.m4_crf_cached import enable_on_crf
    from ics.official_data import load_inputs
    ADAPTER = PC.Replay()
    with reuse_native_basis(INSID3, ASSETS / 'native_assets/positional_basis.pt'):
        HOST = INSID3(encoder=ADAPTER, image_size=1024, svd_components=500, tau=.6,
                      merge_threshold=.2, mask_refiner='crf', crf_size=640,
                      resize_to_orig_size=False, device='cpu').eval().requires_grad_(False)
    enable_on_crf(HOST._crf, prepared)
    LOAD, UPSAMPLE = load_inputs, upsample_mask
    CAPTURE = {}
    original = HOST._finalize_mask
    def finalize(mask, target):
        CAPTURE['mask64'] = mask.detach().cpu().clone()
        CAPTURE['bilinear'] = upsample_mask(mask, 1024, 1024).detach().cpu().clone()
        CAPTURE['precrf_elapsed'] = time.monotonic() - CAPTURE['started']
        began = time.monotonic()
        result = original(mask, target)
        CAPTURE['crf_seconds'] = time.monotonic() - began
        return result
    HOST._finalize_mask = finalize


def infer(cohort, binding):
    import numpy as np
    import torch
    root=OUT/cohort; name=PC.key(binding); record_path=root/'records'/(name+'.json')
    if record_path.exists():
        old=read(record_path)
        assert sha(old['prediction_path']) == old['prediction_sha256']
        return old
    prediction=root/'predictions'/(name+'.npz')
    assert not prediction.exists(), 'Preserve incomplete prediction ' + name
    began=time.monotonic(); CAPTURE.clear(); ADAPTER.bind(binding)
    try:
        with PC.inference_only(binding):
            reference, mask, query=LOAD(binding['row'],ASSETS)
            HOST.set_reference(reference,mask); HOST.set_target(query)
            CAPTURE['started']=time.monotonic()
            crf=HOST.segment().cpu().numpy()
            bilinear=CAPTURE['bilinear'].numpy()
            coarse=CAPTURE['mask64'].numpy()
            assert crf.shape==bilinear.shape==(1024,1024) and coarse.shape==(64,64)
            assert ADAPTER.encoder is None and ADAPTER.encoder_calls==0
            values={'cli/insid3.bilinear':PC.pack(bilinear), 'cli/insid3.crf640':PC.pack(crf),
                    'mask64/insid3':PC.pack(coarse)}
            probe={}
            if cohort=='probe10':
                source=binding['old_insid3']
                with np.load(source['prediction_path'],allow_pickle=False) as z:
                    prior=PC.unpack(z['A/official_bilinear'])
                with np.load(source['state_path'],allow_pickle=False) as z:
                    prior64=z['A/official64']
                probe=dict(archive_bilinear_xor=int(np.count_nonzero(bilinear!=prior)),
                           archive_mask64_xor=int(np.count_nonzero(coarse!=prior64)),
                           archive_prediction_sha256=sha(source['prediction_path']),
                           archive_state_sha256=sha(source['state_path']))
        temporary=prediction.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,**values); temporary.replace(prediction)
        record=dict(index=name,dataset=binding['dataset'],episode_id=binding['row']['episode_id'],
                    prediction_path=str(prediction),prediction_sha256=sha(prediction),
                    seconds=time.monotonic()-began,core_plus_bilinear_seconds=CAPTURE['precrf_elapsed'],
                    crf640_seconds=CAPTURE['crf_seconds'],worker_pid=os.getpid(),
                    reference_count=1,parameter_fitting=False,new_encoder_calls=0,query_GT_reads=0,
                    FoRIS_score_consumed=False,**probe)
        write(record_path,record)
        return record
    finally:
        HOST.reset_state(); ADAPTER.memo.clear(); CAPTURE.clear()


def produce(cohort):
    config=check(); root=OUT/cohort
    lab,pc=modules()
    if (root/'sealed.json').exists():
        return read(root/'sealed.json')
    if cohort!='probe10':
        proof=read(OUT/'probe10/sealed.json')
        assert proof['n']==10 and proof['archive_bilinear_xor']==proof['archive_mask64_xor']==0
    rows=read(root/'manifest.json'); began=time.monotonic(); records=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=init_worker) as pool:
        futures={pool.submit(infer,cohort,b):b['index'] for b in rows}
        for i,future in enumerate(as_completed(futures),1):
            result=future.result(); records.append(result)
            if cohort=='probe10' and (result['archive_bilinear_xor'] or result['archive_mask64_xor']):
                raise RuntimeError('Official cached INSID3 parity mismatch: '+result['index'])
            if i<=10 or i%50==0:
                progress=dict(cohort=cohort,completed=i,total=len(rows),seconds=round(time.monotonic()-began,3),
                              dataset=result['dataset'],new_encoder_calls=0,query_GT_reads=0)
                write(root/'progress.json',progress); print(progress,flush=True)
    by_key={r['index']:r for r in records}
    seal=dict(state='ALL_INSID3_NATURAL_OUTPUTS_SEALED',cohort=cohort,n=len(rows),arms=list(ARMS),
              seconds=time.monotonic()-began,case_seconds_sum=sum(r['seconds'] for r in records),
              core_plus_bilinear_seconds_sum=sum(r['core_plus_bilinear_seconds'] for r in records),
              crf640_seconds_sum=sum(r['crf640_seconds'] for r in records),
              config_sha256=sha(OUT/'config.json'),manifest_sha256=config['manifest_sha256'][cohort],
              records=[by_key[pc.key(b)] for b in rows],reference_count=1,parameter_fitting=False,
              new_encoder_calls=0,query_GT_reads=0,FoRIS_score_consumed=False,
              archive_bilinear_xor=sum(r.get('archive_bilinear_xor',0) for r in records),
              archive_mask64_xor=sum(r.get('archive_mask64_xor',0) for r in records))
    write(root/'sealed.json',seal)
    print({k:v for k,v in seal.items() if k!='records'},flush=True)
    return seal


def score(cohort):
    assert cohort in ('r5_500','development1360')
    config=check(); lab,pc=modules(); pc.setup('cpu')
    import numpy as np
    root=OUT/cohort; seal=read(root/'sealed.json'); rows=read(root/'manifest.json')
    assert seal['state']=='ALL_INSID3_NATURAL_OUTPUTS_SEALED' and len(rows)==seal['n']
    if (root/'report.json').exists():
        return read(root/'report.json')
    source_rows=read(R5/'manifest.baselines.json') if cohort=='r5_500' else rows
    controls={pc.key(b):b for b in source_rows}
    candidates={r['index']:r for r in read(CANDIDATE/'sealed.json')['records']} if cohort=='r5_500' else {}
    arms=['FoRIS',*ARMS]+(['R5_candidate','R5_UN_midrange','R5_VP_previous'] if candidates else [])
    # Verify every natural prediction before the first query annotation is read.
    for b,record in zip(rows,seal['records']):
        assert record['index']==pc.key(b) and sha(record['prediction_path'])==record['prediction_sha256']
        original=controls[pc.key(b)]['old_arms']['B']
        if original.get('sha256'): assert sha(original['path'])==original['sha256']
        if candidates:
            r=candidates[pc.key(b)]; assert sha(r['prediction_path'])==r['prediction_sha256']
    began=time.monotonic(); records=[]
    for b,result in zip(rows,seal['records']):
        name=pc.key(b); baseline=controls[name]['old_arms']['B']
        with np.load(result['prediction_path'],allow_pickle=False) as z:
            predictions={a:pc.unpack(z['cli/'+a]) for a in ARMS}
        with np.load(baseline['path'],allow_pickle=False) as z:
            predictions['FoRIS']=pc.unpack(z[baseline['cli_key']])
        if candidates:
            with np.load(candidates[name]['prediction_path'],allow_pickle=False) as z:
                predictions.update(R5_candidate=pc.unpack(z['candidate']),R5_UN_midrange=pc.unpack(z['UN_midrange']),
                                   R5_VP_previous=pc.unpack(z['VP_previous']))
        gt,valid=pc.truth(b)
        metrics={a:pc.pixel_metrics(predictions[a],gt,valid) for a in arms}
        records.append(dict(index=name,dataset=b['dataset'],fold=b['row']['fold'],class_id=b['row']['global_class_id'],
                            query_photo=str((ASSETS/b['row']['query_path']).resolve()),metrics=metrics))
    summary=pc.summarize(records,arms,bootstrap=1000,seed=2026101206)
    deltas={d:{a:values[a]['class_miou']-values['FoRIS']['class_miou'] for a in arms if a!='FoRIS'}
            for d,values in summary.items()}
    report=dict(state='R6_INSID3_SCORED_AFTER_SEAL',cohort=cohort,n=len(rows),arms=arms,summary=summary,
                delta_to_foris_pp=deltas,metric=config['metric'],scope=config['cohort_scope'][cohort],
                prediction_seal_sha256=sha(root/'sealed.json'),config_sha256=sha(OUT/'config.json'),
                score_seconds=time.monotonic()-began,new_encoder_calls=0,parameter_fitting=False,
                query_GT_in_inference=False,query_GT_only_after_seal=True)
    write(root/'episodes.json',records); write(root/'report.json',report)
    lines=['# Complete official INSID3, '+cohort,'',config['cohort_scope'][cohort]+'.',
           '','CLI1024; official bilinear and original CRF640 are reported separately. Single reference, no fitting or encoder calls.',
           '','|Domain|n|'+'|'.join(arms)+'|','|---|---:|'+'|'.join(['---:']*len(arms))+'|']
    counts=Counter(b['dataset'] for b in rows)
    for d,values in summary.items():
        lines.append('|'+d+'|'+str(counts[d])+'|'+'|'.join(f"{values[a]['class_miou']:.3f}" for a in arms)+'|')
    lines += ['',f"Natural production wall: {seal['seconds']:.3f}s; scoring: {report['score_seconds']:.3f}s.",
              'The reported masks and scores come from the official full candidate/cluster/seed/merge chain; no stage-bank approximation or FoRIS score input.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(dict(state=report['state'],cohort=cohort,n=len(rows),delta_to_foris_pp=deltas,
               score_seconds=report['score_seconds']),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','probe','produce','score','run'))
    parser.add_argument('--cohort',choices=('r5_500','development1360'),default='r5_500')
    args=parser.parse_args()
    if args.mode=='prepare': prepare()
    elif args.mode=='probe': produce('probe10')
    elif args.mode=='produce': produce(args.cohort)
    elif args.mode=='score': score(args.cohort)
    else:
        prepare(); produce('probe10'); produce('r5_500'); score('r5_500')
        produce('development1360'); score('development1360')
