#!/usr/bin/env python3
"""PLAN B: exact manifest complement, accepted annotation truth, reusable tokens.

CPU-only server entry; this script does not encode images or download assets.
Run --phase truth first when inspecting annotation provenance, then --phase
compute with the same output and A's report. --phase all executes both in that
order. A failed fresh600 truth gate prevents every holdout inference call.

  python scripts/prepare_holdout_tokens.py --out OUTPUT --phase truth \
    --annotation-source-metadata .../claude_official/batch0.json \
    --annotation-source-metadata .../claude_official/batch1.json
  python scripts/prepare_holdout_tokens.py --out OUTPUT --phase compute \
    --a-report ACCEPTED_A/report.json [same source arguments]

Defaults deliberately use the original ICS annotation tree, not demo4_cache's
different PNGs. Class c maps to annotation value c+1, as run_order_tokens did.
Expected601 is historical text, not a sample-count gate: the observed1200 minus
fresh600 manifest may be600. No row or feature is manufactured to fill a gap.
Outputs retain FP32 episode fields and FP16[N,4096] tokens.npz compatible with
the historical record. Here rcg explicitly means canonical RCG-on-native-s2,
not the historical fresh600 RCG-on-final-score. GT is stored/scored only.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
SERVER = Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs')
IDENTITY = ('support', 'query', 'c', 'fold', 'key')
ANNOTATIONS = Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')
_BENCH = None
_CONFIG = None
_PREFIX = None


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def array_sha(value):
    import numpy as np
    array=np.ascontiguousarray(value);value=hashlib.sha256()
    value.update(array.dtype.str.encode());value.update(str(array.shape).encode());value.update(array.tobytes())
    return value.hexdigest()


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_rows(path):
    data = json.loads(Path(path).read_text())
    data = data.get('rows', data.get('episodes')) if isinstance(data, dict) else data
    if not isinstance(data, list) or not data:
        raise ValueError('Nonempty actual rows/episodes manifest required: '+str(path))
    result = []
    for raw in data:
        row = dict(raw)
        if not all(name in row for name in IDENTITY):
            raise ValueError('Exact support/query/class/fold/key identity absent')
        row['c'], row['fold'] = int(row['c']), int(row['fold'])
        for name in ('key', 'support', 'query'):
            row[name] = str(row[name])
        if (not row['key'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in row['key'])
                or row['support'] == row['query']):
            raise ValueError('Unsafe key or identical reference/query: '+row['key'])
        for name in ('support', 'query'):
            if Path(row[name]).is_absolute() or '..' in Path(row[name]).parts:
                raise ValueError('Annotation identity must be a safe relative photo path')
        result.append(row)
    return result


def identity(row):
    return tuple(row[name] for name in IDENTITY)


def exact_complement(source, fresh):
    """No key-only matching, class coercion, or forced601 count."""
    fresh_ids = {identity(row) for row in fresh}
    by_identity = {}; duplicates = []; key_owner = {}
    for index, row in enumerate(source):
        ident = identity(row)
        if row['key'] in key_owner and key_owner[row['key']] != ident:
            raise ValueError('One source key denotes different exact episodes: '+row['key'])
        key_owner[row['key']] = ident
        if ident in by_identity:
            previous = by_identity[ident]
            if any(row.get(k) != previous.get(k) for k in ('feature_export', 'packet_export')):
                raise ValueError('Duplicate exact identity has contradictory cache bindings: '+row['key'])
            duplicates.append(dict(source_row_index=index,identity=list(ident)))
        else:
            by_identity[ident] = row
    selected = [r for ident, r in by_identity.items() if ident not in fresh_ids]
    semantic_groups = {}
    for row in source:
        # Same R/Q/class across aliases/folds is exposure, never independence.
        semantic_groups.setdefault((row['support'],row['query'],row['c']), []).append(list(identity(row)))
    fresh_semantic = {(r['support'],r['query'],r['c']) for r in fresh}
    aliases = [dict(identity=list(identity(r)),reason='same support/query/class as fresh600 under another key/fold')
               for r in selected if (r['support'],r['query'],r['c']) in fresh_semantic]
    if aliases:
        # These remain part of the literal exact complement, but are flagged as
        # exposed aliases rather than being silently counted as independent.
        alias_ids = {tuple(x['identity']) for x in aliases}
        selected = [dict(r, fresh_semantic_alias=identity(r) in alias_ids) for r in selected]
    report = dict(expected_manifest_rows=1200,actual_manifest_rows=len(source),
                  unique_source_exact_identities=len(by_identity),fresh_rows=len(fresh),
                  unique_fresh_exact_identities=len(fresh_ids),
                  exact_shared_identities=len(set(by_identity)&fresh_ids),
                  fresh_identities_missing_from_source=[list(i) for i in sorted(fresh_ids-set(by_identity))],
                  expected_holdout_rows_in_PLAN=601,actual_holdout_rows=len(selected),
                  count_discrepancy_vs_PLAN=len(selected)-601,
                  exact_identity_fields=list(IDENTITY),duplicate_source_rows=duplicates,
                  semantic_duplicate_groups=[v for v in semantic_groups.values() if len(v)>1],
                  fresh_semantic_aliases_in_exact_complement=aliases,
                  unique_holdout_support_query_class=len({(r['support'],r['query'],r['c']) for r in selected}),
                  no_rows_or_features_manufactured=True,
                  exposure='previously reused development data; never independent confirmation')
    return selected, report


def source_binding(args):
    files = dict(input_manifest=args.input_manifest,fresh_rows=args.fresh/'rows.json',
                 fresh_tokens=args.fresh/'tokens.npz')
    bound = {name:dict(path=str(path),sha256=sha(path)) for name,path in files.items()}
    metadata = []
    for path in args.annotation_source_metadata:
        document = json.loads(path.read_text())
        if str(Path(document.get('annotation_root',''))) != str(args.annotation_root):
            raise ValueError('Original producer metadata annotation_root differs from selected source: '+str(path))
        metadata.append(dict(path=str(path),sha256=sha(path),annotation_root=document['annotation_root']))
    if not metadata:
        raise ValueError('Bind actual claude_official/batch0/1 metadata with --annotation-source-metadata')
    # Truth acceptance can run while A's inference functions are being finished.
    # Its binding includes only executable truth code, not unrelated A edits.
    names = ('scripts/prepare_holdout_tokens.py','scripts/run_order_tokens.py')
    return dict(files=bound,annotation_root=str(args.annotation_root),annotation_source_metadata=metadata,
                source_hashes={name:sha(ROOT/name) for name in names},
                identity_rule=list(IDENTITY),truth_rule='original PNG ==c+1 -> Torch nearest1024 -> avg_pool2d16 -> FP32 coverage64',
                truth_acceptance_rule='fresh600 generated and stored truth strict>.5 aggregate disagreement<=.001',
                reference_rule='the same annotation transform must reproduce stored reference coverage exactly',
                default_annotation_source='original ICS root recovered from claude_official batch0/1, not demo4_cache')


def verify_binding(binding):
    for item in list(binding['files'].values())+binding['annotation_source_metadata']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Input/annotation producer source changed: '+item['path'])
    for name, expected in binding['source_hashes'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Executable source changed during this run: '+name)


def snapshot_sources(out, hashes):
    for name,expected in hashes.items():
        content=(ROOT/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=expected:
            raise ValueError('Executable source changed before snapshot: '+name)
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=expected:
            raise ValueError('Refuse to overwrite a different frozen source: '+str(target))
        if not target.exists():target.write_bytes(content)


def initialize_worker(config):
    global _BENCH, _CONFIG, _PREFIX
    _CONFIG=config
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]=str(config['threads'])
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(config['threads']);torch.set_num_interop_threads(1)
    if config.get('stage')=='compute':
        import bench_evidence
        _BENCH=bench_evidence
    _PREFIX=None


def annotation_coverage(row, role):
    """This function is shared by fresh acceptance and holdout truth generation."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    path=Path(_CONFIG['annotation_root'])/Path(row[role]).with_suffix('.png')
    content=path.read_bytes()
    with Image.open(io.BytesIO(content)) as image:
        labels=np.asarray(image)
        if labels.ndim != 2 or labels.dtype.kind not in 'uib':
            raise ValueError('Actual integer label PNG required: '+str(path))
        target=(labels==row['c']+1).copy()
    original=torch.from_numpy(target)
    with torch.inference_mode():
        work=F.interpolate(original[None,None].float(),(1024,1024),mode='nearest')
        coverage=F.avg_pool2d(work,16)[0,0].numpy().copy()
    return coverage,dict(path=str(path),sha256=hashlib.sha256(content).hexdigest(),
                         role=role,class_id=row['c'],annotation_value=row['c']+1,
                         original_hw=list(target.shape),original_positive_pixels=int(target.sum()),
                         work_positive_pixels=int(work.sum()),coverage_array_sha256=array_sha(coverage))


def truth_task(row):
    import numpy as np
    try:
        truth,q_receipt=annotation_coverage(row,'query')
        reference,r_receipt=annotation_coverage(row,'support')
        if r_receipt['original_positive_pixels']==0:
            raise ValueError('Original reference prompt is empty: '+row['key'])
        return dict(key=row['key'],truth=truth,ref=reference,
                    query_annotation=q_receipt,reference_annotation=r_receipt)
    except Exception as error:
        return dict(key=row['key'],error=repr(error),traceback=traceback.format_exc())


def pool_results(rows, config, worker):
    context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=config['workers'],mp_context=context,
            initializer=initialize_worker,initargs=(config,)) as pool:
        # At most two tasks/worker in flight: predictable memory and prompt stop.
        iterator=iter(enumerate(rows));pending={}
        for _ in range(config['workers']*2):
            item=next(iterator,None)
            if item is None:break
            index,row=item;pending[pool.submit(worker,row)]=index
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                index=pending.pop(future)
                yield index,future.result()
                item=next(iterator,None)
                if item is not None:
                    new_index,row=item;pending[pool.submit(worker,row)]=new_index


def truth_phase(args, fresh, selected, binding, selection, config):
    import numpy as np
    started=time.monotonic();receipts=[None]*len(fresh)
    with np.load(args.fresh/'tokens.npz',allow_pickle=False) as pack:
        truth=pack['truth'].copy().reshape(len(fresh),64,64)
        reference=pack['ref'].copy().reshape(len(fresh),64,64)
    if truth.dtype!=np.float16 or reference.dtype!=np.float16:
        raise ValueError('Original fresh600 FP16 truth/ref storage required')
    for index,result in pool_results(fresh,dict(config,stage='truth'),truth_task):
        if 'error' not in result:
            generated=result.pop('truth');generated_ref=result.pop('ref')
            result.update(truth_mask_different_tokens=int(np.count_nonzero((generated>.5)!=(truth[index].astype(np.float32)>.5))),
                          truth_max_abs_difference=float(np.max(np.abs(generated-truth[index].astype(np.float32)))),
                          reference_coverage_exact=bool(np.array_equal(generated_ref,reference[index].astype(np.float32))),
                          reference_max_abs_difference=float(np.max(np.abs(generated_ref-reference[index].astype(np.float32)))))
        receipts[index]=result
        write(args.out/'fresh_truth_receipts.json',receipts)
        write(args.out/'progress.json',dict(state='FRESH600_TRUTH_ACCEPTANCE',completed=sum(x is not None for x in receipts),total=len(fresh)))
    errors=[r for r in receipts if 'error' in r]
    different=sum(r.get('truth_mask_different_tokens',0) for r in receipts)
    reference_failures=[r['key'] for r in receipts if not r.get('reference_coverage_exact',False)]
    fraction=different/(len(fresh)*4096)
    gate=dict(evaluated=len(fresh),errors=errors,mask_different_tokens=different,
              mask_disagreement_fraction=fraction,fixed_tolerance_fraction=.001,
              truth_passed=not errors and fraction<=.001,
              reference_coverage_exact=not reference_failures,reference_mismatch_keys=reference_failures,
              passed=not errors and fraction<=.001 and not reference_failures,
              query_truth_used_for_evidence=False)
    write(args.out/'truth_acceptance.json',gate)
    if not gate['passed']:
        write(args.out/'report.json',dict(state='STOPPED_TRUTH_SOURCE_OR_ACCEPTANCE',truth_acceptance=gate,
              selection=selection,source_binding=binding,holdout_inference_executed=False))
        return False
    directory=args.out/'truth_fields';directory.mkdir()
    receipts=[None]*len(selected)
    for index,result in pool_results(selected,dict(config,stage='truth'),truth_task):
        if 'error' in result:
            receipts[index]=result;write(args.out/'holdout_truth_receipts.json',receipts)
            raise ValueError('Holdout truth generation failed: '+repr(result))
        truth64=result.pop('truth');ref64=result.pop('ref')
        path=directory/(result['key']+'.npz')
        np.savez_compressed(path,truth=truth64,ref=ref64)
        result.update(truth_field_path=str(path),truth_field_sha256=sha(path),
                      positive_majority_tokens=int(np.count_nonzero(truth64>.5)),
                      reference_majority_tokens=int(np.count_nonzero(ref64>.5)))
        receipts[index]=result
        write(args.out/'holdout_truth_receipts.json',receipts)
        write(args.out/'progress.json',dict(state='HOLDOUT_TRUTH_GENERATION',completed=sum(x is not None for x in receipts),total=len(selected)))
    # Hash consistency detects changed PNG contents across repeated image use.
    annotation_hashes={}
    for receipt in json.loads((args.out/'fresh_truth_receipts.json').read_text())+receipts:
        for name in ('query_annotation','reference_annotation'):
            item=receipt[name];previous=annotation_hashes.setdefault(item['path'],item['sha256'])
            if previous!=item['sha256']:raise ValueError('Annotation changed between episodes: '+item['path'])
    verify_binding(binding)
    write(args.out/'truth_ready.json',dict(state='TRUTH_READY',n=len(selected),source_binding=binding,
          selection=selection,truth_acceptance=gate,rows_sha256=sha(args.out/'rows.json'),
          holdout_truth_receipts_sha256=sha(args.out/'holdout_truth_receipts.json'),
          annotation_hashes=annotation_hashes,elapsed_seconds=time.monotonic()-started,
          holdout_inference_executed=False,query_truth_used_for_evidence=False))
    return True


def accepted_a_report(path,binding):
    import bench_evidence as bench
    data=json.loads(path.read_text());acceptance=data.get('acceptance',{})
    one=data.get('acceptance1',acceptance.get('acceptance1',{}))
    second=data.get('acceptance2',acceptance.get('acceptance2',{}))
    if not isinstance(second,dict):raise ValueError('A native-s2 acceptance is incomplete')
    two=second.get('summary',second)
    if not one.get('passed') or two.get('evaluated')!=600:
        raise ValueError('A must supply completed600 acceptance1/2; no inferential fallback')
    difference=abs(float(two['actual_class_mIoU'])-float(two['saved_class_mIoU']))
    grade=bench.acceptance2_grade(two)
    if not grade['continue_ladder']:
        raise ValueError('A native-s2 acceptance has not satisfied updated PLAN')
    for name in ('scripts/bench_evidence.py','src/ics/methods/stage_bank.py','src/ics/methods/rcg.py'):
        if data.get('source_hashes',{}).get(name)!=binding['source_hashes'][name]:
            raise ValueError('A report and B executable source differ: '+name)
    expected_inputs=dict(rows_sha256=binding['files']['fresh_rows']['sha256'],
                         tokens_sha256=binding['files']['fresh_tokens']['sha256'],
                         input_manifest_sha256=binding['files']['input_manifest']['sha256'])
    if data.get('input_hashes')!=expected_inputs:
        raise ValueError('A and B must use the same actual fresh600/1200 source cohort')
    if data.get('query_GT_used_for_evidence') is not False:
        raise ValueError('A must explicitly exclude queryGT from evidence')
    return dict(path=str(path),sha256=sha(path),native_s2_difference_pp=difference,grade=grade)


def inference_task(row):
    global _PREFIX
    import numpy as np
    from ics.methods import rcg
    start=time.monotonic()
    try:
        if _PREFIX is None:_PREFIX=_BENCH.part2_prefix()
        prefix,prefix_sha=_PREFIX
        feature_path=_BENCH.resolve(row['feature_export'],_CONFIG['input_base'])
        packet_path=_BENCH.resolve(row['packet_export'],_CONFIG['input_base'])
        q,r,features=_BENCH.features(feature_path)
        packet,packet_sha=_BENCH.packet(packet_path)  # never reads its queryGT member
        fg,reference_source=_BENCH.native_foreground(row,_CONFIG['annotation_root'],packet['cov'])
        evidence=_BENCH.native_evidence(fg,prefix)
        # Only cached features and known reference labels reach either head.
        native_s2=evidence(q,r,packet['cov'])
        s2=_BENCH.ladder_evidence(6,{'s2':native_s2})(q,r,packet['cov'])
        field,solver=rcg.predict(q,r,packet['cov'],s2,device='cpu')
        if s2.shape!=(64,64) or field.shape!=(64,64) or not (np.isfinite(s2).all() and np.isfinite(field).all()):
            raise ValueError('Complete finite native64 fields required')
        # QueryGT is first loaded into this worker AFTER both inference calls.
        truth_path=Path(_CONFIG['out'])/'truth_fields'/(row['key']+'.npz')
        if sha(truth_path)!=_CONFIG['truth_receipts'][row['key']]['truth_field_sha256']:
            raise ValueError('Prevalidated truth field changed: '+row['key'])
        with np.load(truth_path,allow_pickle=False) as saved:
            truth,reference=saved['truth'].copy(),saved['ref'].copy()
        if not np.array_equal(reference,packet['cov']):
            raise ValueError('Bound native reference cov differs from generated original label: '+row['key'])
        receipt=_CONFIG['truth_receipts'][row['key']]
        if reference_source['reference_annotation_sha256']!=receipt['reference_annotation']['sha256']:
            raise ValueError('Reference PNG changed after truth acceptance')
        s2=s2.astype(np.float32,copy=False);field=field.astype(np.float32,copy=False)
        output=Path(_CONFIG['out'])/'fields'/(row['key']+'.npz')
        np.savez_compressed(output,truth=truth,ref=reference,s2=s2,rcg=field,score=packet['score'])
        s2_mask=_BENCH.mask(s2,'raw');rcg_mask=_BENCH.mask(field,'rcg')
        return dict(key=row['key'],field_path=str(output),field_sha256=sha(output),
                    s2_IU=_BENCH.counts(s2_mask,truth).tolist(),rcg_IU=_BENCH.counts(rcg_mask,truth).tolist(),
                    truth_majority_tokens=int(np.count_nonzero(truth>.5)),
                    s2_positive_tokens=int(s2_mask.sum()),rcg_positive_tokens=int(rcg_mask.sum()),
                    s2_FP16_storage_mask_changes=int(np.count_nonzero(s2_mask!=_BENCH.mask(s2.astype(np.float16),'raw'))),
                    rcg_FP16_storage_mask_changes=int(np.count_nonzero(rcg_mask!=(field.astype(np.float16)>.5))),
                    features=features,packet_sha256=packet_sha,reference_source=reference_source,
                    native_s2_prefix_source_sha256=prefix_sha,native_s2_info=evidence.last_info,
                    solver=solver,seconds=time.monotonic()-start,actual_new_encoder_forwards=0,
                    query_GT_used_for_evidence=False,query_GT_role='stored and scored only after inference')
    except Exception as error:
        return dict(key=row['key'],error=repr(error),traceback=traceback.format_exc(),fallback='none')


def compute_phase(args,selected,binding,selection,config):
    import numpy as np
    import bench_evidence as bench
    from ics.methods import rcg,stage_bank
    ready=json.loads((args.out/'truth_ready.json').read_text())
    if ready['source_binding']!=binding or not ready['truth_acceptance']['passed']:
        raise ValueError('Fresh600 truth must pass with exactly the current source binding before holdout')
    if sha(args.out/'rows.json')!=ready['rows_sha256'] or sha(args.out/'holdout_truth_receipts.json')!=ready['holdout_truth_receipts_sha256']:
        raise ValueError('Truth-ready row/receipt source changed')
    inference_names=('scripts/bench_evidence.py','src/ics/methods/stage_bank.py','src/ics/methods/rcg.py')
    inference_sources={name:sha(ROOT/name) for name in inference_names}
    a=accepted_a_report(args.a_report,dict(binding,source_hashes=dict(binding['source_hashes'],**inference_sources)))
    bench.assert_recipe()
    snapshot_sources(args.out,inference_sources)
    truth_receipts=json.loads((args.out/'holdout_truth_receipts.json').read_text())
    config=dict(config,stage='compute',truth_receipts={r['key']:r for r in truth_receipts})
    (args.out/'fields').mkdir()
    receipts=[None]*len(selected);started=time.monotonic()
    for index,result in pool_results(selected,config,inference_task):
        receipts[index]=result;write(args.out/'receipts.json',receipts)
        write(args.out/'progress.json',dict(state='HOLDOUT_NATIVE_S2_AND_RCG',completed=sum(r is not None for r in receipts),total=len(selected)))
        if 'error' in result:
            raise ValueError('Native holdout inference failed without fallback: '+repr(result))
    arrays={name:np.empty((len(selected),4096),np.float16) for name in ('truth','ref','s2','rcg','score')}
    for index,result in enumerate(receipts):
        if sha(result['field_path'])!=result['field_sha256']:raise ValueError('Complete episode output changed')
        with np.load(result['field_path'],allow_pickle=False) as pack:
            for name in arrays:arrays[name][index]=pack[name].reshape(-1).astype(np.float16)
    arrays['iu_s2']=np.array([r['s2_IU'] for r in receipts],np.int64)
    arrays['iu_rcg']=np.array([r['rcg_IU'] for r in receipts],np.int64)
    arrays['rcg_s2']=arrays['rcg']  # explicit alias of the same actual field
    np.savez_compressed(args.out/'tokens.npz',**arrays)
    verify_binding(binding)
    for name,expected in inference_sources.items():
        if sha(ROOT/name)!=expected:raise ValueError('Inference executable source changed: '+name)
    classes=np.array([r['c'] for r in selected])
    report=dict(state='HOLDOUT_TOKENS_COMPLETE',n=len(selected),classes=len(np.unique(classes)),
                selection=selection,source_binding=binding,inference_source_hashes=inference_sources,
                accepted_A=a,truth_acceptance=ready['truth_acceptance'],
                original_cache_stage='already FoRIS Part1 processed FP16; no raw/no-APD claim',
                inference_API='bench.native_foreground -> bench.native_evidence(bench.part2_prefix) -> bench.ladder_evidence(6), canonical rcg.predict CPU',
                native_s2_CONFIG=stage_bank.CONFIG,RCG_CONFIG=rcg.CONFIG,
                rcg_field_origin='recomputed native_s2, not original packet final score',
                rcg_input='s2',fresh_original_rcg_input='FoRIS final score',
                field_producers=dict(truth='bound original query label nearest1024/avgpool16',
                    ref='bound original reference label nearest1024/avgpool16, exact packet cov',
                    s2='A native Part2 prefix, reference native bilinear FG, processed FP16 features',
                    rcg='canonical fixed rcg.predict(q,r,ref,recomputed_s2)',
                    rcg_s2='same actual array as rcg; explicit input-stage alias',
                    score='actual packet FoRIS final score, saved without recalculation'),
                absent_fields_not_fabricated=['nbr','back_fg','back_bg','native'],
                fields_storage='FP32 episode files; FP16 flat tokens.npz for historical compatibility',
                FP32_field_class_mIoU=dict(s2=bench.miou(arrays['iu_s2'],classes),rcg_s2=bench.miou(arrays['iu_rcg'],classes)),
                FP16_storage_mask_changes=dict(s2=sum(r['s2_FP16_storage_mask_changes'] for r in receipts),
                                               rcg=sum(r['rcg_FP16_storage_mask_changes'] for r in receipts)),
                query_GT_used_for_evidence=False,query_GT_role='truth generation, storage and post-inference scoring only',
                actual_new_encoder_forwards=0,workers=args.workers,threads_per_worker=args.threads,
                elapsed_compute_seconds=time.monotonic()-started,
                exposure='reused development complement; not independent confirmation',
                output_files={n:dict(path=str(args.out/n),sha256=sha(args.out/n)) for n in ('tokens.npz','rows.json','receipts.json')})
    write(args.out/'report.json',report)
    write(args.out/'manifest.json',dict(report,rows=selected))
    write(args.out/'progress.json',dict(state=report['state'],completed=len(selected)))


def run(args):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]=str(args.threads)
    binding=source_binding(args)
    fresh=read_rows(args.fresh/'rows.json');source=read_rows(args.input_manifest)
    if len(fresh)!=600 or len({identity(r) for r in fresh})!=600:
        raise ValueError('Original fresh600 must contain600 unique exact episode identities')
    selected,selection=exact_complement(source,fresh)
    if not selected:raise ValueError('Actual manifest has no complement; no fabricated holdout')
    if any(not all(k in r for k in ('feature_export','packet_export')) for r in selected):
        raise ValueError('Actual manifest must bind every selected feature/packet')
    config=dict(workers=args.workers,threads=args.threads,annotation_root=str(args.annotation_root),
                input_base=str(args.input_base or args.input_manifest.parent),out=str(args.out))
    if args.phase!='compute':
        args.out.mkdir(parents=True,exist_ok=False)
        write(args.out/'rows.json',selected);write(args.out/'selection.json',selection)
        write(args.out/'source_binding.json',binding)
        snapshot_sources(args.out,binding['source_hashes'])
    else:
        if json.loads((args.out/'rows.json').read_text())!=selected:
            raise ValueError('Resumed compute exact complement differs from accepted truth rows')
    try:
        if args.phase in ('truth','all') and not truth_phase(args,fresh,selected,binding,selection,config):
            return 2
        if args.phase in ('compute','all'):
            if args.a_report is None:raise ValueError('--a-report is required before native holdout inference')
            compute_phase(args,selected,binding,selection,config)
        return 0
    except Exception as error:
        write(args.out/'report.json',dict(state='FAILED_STOP',error=repr(error),traceback=traceback.format_exc(),
              selection=selection,source_binding=binding,query_GT_used_for_evidence=False,fallback='none'))
        write(args.out/'progress.json',dict(state='FAILED_STOP',error=repr(error)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fresh',type=Path,default=SERVER/'claude_order_fresh600')
    parser.add_argument('--input-manifest',type=Path,default=SERVER/'confirm1200_conditional_v1/inputs/manifest.json')
    parser.add_argument('--input-base',type=Path)
    parser.add_argument('--annotation-root',type=Path,default=ANNOTATIONS)
    parser.add_argument('--annotation-source-metadata',type=Path,action='append',default=[])
    parser.add_argument('--a-report',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase',choices=('truth','compute','all'),default='all')
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.workers<=16 or not 1<=args.threads<=2 or args.workers*args.threads>32:
        parser.error('At most16 CPU workers ×2 threads,32 total')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
