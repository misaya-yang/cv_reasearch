#!/usr/bin/env python3
"""CPU fresh600 evidence acceptance and position-level comparison harness.

Reuse the completed CPU acceptance1 without rerunning its600 RCG calls:
  python scripts/bench_evidence.py --acceptance1-provider PROVIDER_DIRECTORY \
    --provider-runner ORIGINAL_ACCEPTANCE1_CPU_SCRIPT --out NEW_DIRECTORY

An evidence callable takes (q, r, cov) and returns a finite64x64 field.
The main metric is the precise protocol behind PLAN's64.52 anchor:
saved FP16 area truth -> FP32 -> strict truth>.5, class-summed I/U.
Raw evidence is minmax>.5; canonical RCG output is directly>.5.

Acceptance uses the user-updated fixed gates: absolute class mIoU difference
<=.3pp and aggregate mask disagreement<=1%. These tolerances were chosen after
acceptance1 was observed, and that post-hoc decision is recorded explicitly.
Field drift and FP16 quantisation discrepancies remain diagnostics.
No constants are searched and neither rcg.py nor stage_bank.py is modified.

The six user-fixed levels end at genuine native s2: FG mean; subtract.55 BG
mean; hardest20% BG tokens; orthogonalize that BG mean; FG prototype maximum;
FG prototype LSE T.07. Acceptance2 must pass before the ladder runs. Separate
accept2/ladder phases preserve fields and avoid repeating accepted work.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import inspect
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SERVER = Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9/outputs')
DEFAULT_FRESH = SERVER / 'claude_order_fresh600'
DEFAULT_INPUTS = SERVER / 'confirm1200_conditional_v1/inputs/manifest.json'
N = 600
IDENTITY = ('key', 'support', 'query', 'c', 'fold')
GATE = dict(absolute_class_mIoU_difference_pp=.3, aggregate_mask_disagreement_fraction=.01,
            chosen_post_hoc_after_acceptance1=True, fixed_before_acceptance2=True,
            source='User-updated PLAN step3 2026-10-07; not a pre-registered acceptance1 tolerance')
LEVELS = (
    ('L1','FG mean cosine'), ('L2','minus.55 all-BG mean'),
    ('L3','hardest20% BG token mean'), ('L4','orthogonalized hard-BG mean'),
    ('L5','FG prototype maximum'), ('L6','FG prototype LSE T.07: native s2'))
NO_APD = 'Not measurable from the already Part1-processed FP16 cache; no inverse or fresh extraction.'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def array_sha(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    value = hashlib.sha256()
    value.update(a.dtype.str.encode()); value.update(str(a.shape).encode()); value.update(a.tobytes())
    return value.hexdigest()


def write(path, value):
    target = Path(path); temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(target)


def rows(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get('rows', data.get('episodes'))
    if not isinstance(data, list):
        raise ValueError('A row list or explicit rows/episodes object is required: '+str(path))
    result = []
    for value in data:
        row = dict(value)
        if not all(k in row for k in IDENTITY):
            raise ValueError('Exact key/support/query/c/fold identity missing in '+str(path))
        row['c'], row['fold'] = int(row['c']), int(row['fold'])
        row['key'], row['support'], row['query'] = map(str, (row['key'], row['support'], row['query']))
        if not row['key'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in row['key']):
            raise ValueError('Unsafe episode key: '+row['key'])
        if row['support'] == row['query']:
            raise ValueError('The reference and query are the same photo: '+row['key'])
        result.append(row)
    if len({r['key'] for r in result}) != len(result):
        raise ValueError('Duplicate keys: '+str(path))
    return result


def exact_join(fresh, source):
    if len(fresh) != N or len(source) != 1200:
        raise ValueError('Require ordered fresh600 rows and the full1200 input manifest')
    indexed = {r['key']: r for r in source}; joined = []
    for row in fresh:
        if row['key'] not in indexed:
            raise ValueError('Fresh key absent from1200 input manifest: '+row['key'])
        other = indexed[row['key']]
        mismatch = {k: [row[k], other[k]] for k in IDENTITY if row[k] != other[k]}
        if mismatch:
            raise ValueError('Exact episode identity mismatch '+row['key']+': '+repr(mismatch))
        if not all(k in other for k in ('feature_export', 'packet_export')):
            raise ValueError('Explicit feature_export/packet_export missing: '+row['key'])
        joined.append(dict(row, feature_export=other['feature_export'], packet_export=other['packet_export']))
    return joined


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else Path(base)/path


def token_fields(path):
    import numpy as np
    fields = {}
    with np.load(path, allow_pickle=False) as saved:
        for name in ('truth', 'ref', 'score', 'rcg', 's2'):
            field = saved[name].copy()
            if field.shape not in ((N, 4096), (N, 64, 64)) or not np.isfinite(field).all():
                raise ValueError('Invalid fresh600 '+name+' shape/values')
            if field.dtype != np.float16:
                raise ValueError('Expected original FP16 token field '+name+'; do not silently change storage provenance')
            fields[name] = field.reshape(N, 64, 64)
    for name in ('truth', 'ref'):
        if fields[name].min() < 0 or fields[name].max() > 1:
            raise ValueError('Invalid saved area coverage: '+name)
    return fields


def features(path):
    import numpy as np
    import torch
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as saved:
            values = {k: saved[k].copy() for k in ('q', 'r')}
            marker = bool(saved['debiased'].item()) if 'debiased' in saved else None
    else:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        values = {k: saved[k].detach().cpu().numpy() for k in ('q', 'r')}
        marker = bool(saved['debiased']) if 'debiased' in saved else None
    for role, value in values.items():
        if value.shape != (4096, 1024) or value.dtype != np.float16 or not np.isfinite(value).all():
            raise ValueError('Exact retained post-Part1 FP16[4096,1024] required: '+role)
        if np.any(np.linalg.norm(value.astype(np.float32), axis=1) == 0):
            raise ValueError('Zero norm cached token')
    # Preserve the FP16->FP32 contents. Canonical rcg.predict performs its own
    # normalization; historical run_rcg2.solve did not. Do not hide that drift.
    return values['q'].astype(np.float32), values['r'].astype(np.float32), dict(
        feature_sha256=sha(path), cache_q_array_sha256=array_sha(values['q']),
        cache_r_array_sha256=array_sha(values['r']), q_r_dtype='float16',
        debiased=marker, stage='retained FoRIS Part1 processed cache, not raw DINO',
        conversion='FP16->FP32 only; no normalization in the loader')


def packet(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        # Only current-reference coverage and source fields; packet query truth
        # is never indexed. Truth for scoring comes from fresh tokens.npz.
        values = {k: saved[k].copy() for k in ('cov', 'score', 's2')}
    for name, value in values.items():
        if value.shape != (64, 64) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Source packet must retain actual FP32 '+name)
    return values, sha(path)


def native_foreground(row, annotation_root, cov):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from ics.methods import stage_bank as bank
    path = Path(annotation_root)/Path(row['support']).with_suffix('.png')
    with Image.open(path) as image:
        original = torch.from_numpy((np.asarray(image) == row['c']+1).copy())
    if not bool(original.any()):
        raise ValueError('Empty original reference mask: '+row['key'])
    mask = F.interpolate(original[None, None].float(), (1024, 1024), mode='nearest')[0, 0] > .5
    area = F.interpolate(mask[None, None].float(), (64, 64), mode='area')[0, 0].numpy()
    difference = np.abs(area-cov)
    if not np.array_equal(area, cov):
        raise ValueError('Original reference nearest1024/area64 does not reproduce source cov '+row['key']+
                         ': maxdiff='+str(float(difference.max())))
    fg = bank.down_bilinear(mask)
    if not bool(fg.any()):
        raise ValueError('Native bilinear reference downsample lacks foreground')
    return fg, dict(reference_annotation_sha256=sha(path),
        reference_work_mask_array_sha256=array_sha(mask.numpy()), foreground_array_sha256=array_sha(fg.numpy()),
        native_foreground_rule='stage_bank.down_bilinear: bilinear>.5, then nearest, then centroid fallback',
        foreground_tokens=int(fg.sum()), area_majority_foreground_disagreement=int((fg.numpy() != (cov.ravel()>.5)).sum()),
        original_reference_area_cov_exact=True)


def part2_prefix():
    """Execute the unchanged current foris prefix through native Part2 only.

    Cutting before the named Part3 `rh` assignment avoids unused query
    clustering. All native Part2 statements and existing helper functions are
    retained. This is source extraction, not a replacement score recipe.
    """
    from ics.methods import stage_bank as bank
    source = inspect.getsource(bank.foris)
    tree = ast.parse(source); function = tree.body[0]; body = []
    for statement in function.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'rh' for t in statement.targets):
            break
        body.append(statement)
    else:
        raise ValueError('Current stage_bank.foris Part3 boundary not found; refusing an assumed prefix')
    body += ast.parse("return {'fg':sim_fg,'bg':sim_bg}, {'FG_prototypes':int(protos.shape[0]),'FG_tokens':int(f.shape[0]),'BG_tokens':int(b.shape[0]),'hard_BG_tokens':max(1,int(c['hard_bg']*b.shape[0])) if b.shape[0] else 0,'native_BG_mean_orthogonal_to_FG_mean':True,'BG_coefficient':c['bg_weight'],'temperature':t}, {'mu_fg':mu_fg,'mu_bg_orth':mu_bg,'mu_bg_hard':F.normalize(hard.mean(0),dim=0) if b.shape[0] else torch.zeros_like(mu_fg),'mu_bg_all':F.normalize(b.mean(0),dim=0) if b.shape[0] else torch.zeros_like(mu_fg),'fg_prototypes':protos}").body
    function.name = '_actual_current_part2'; function.body = body
    namespace = dict(bank.__dict__)
    exec(compile(ast.fix_missing_locations(tree), str(Path(bank.__file__))+':Part2_prefix', 'exec'), namespace)
    return namespace['_actual_current_part2'], hashlib.sha256(source.encode()).hexdigest()


def native_evidence(fg, prefix):
    """Bind only the known reference mask; query labels are never captured."""
    def evidence(q, r, cov):
        import torch
        import torch.nn.functional as F
        from ics.methods import stage_bank as bank
        with torch.inference_mode():
            q = F.normalize(torch.from_numpy(q), dim=1); r = F.normalize(torch.from_numpy(r), dim=1)
            terms, info, auxiliary = prefix(q, r, fg, None, None, None, bank.CONFIG['tau'], mode='never')
            zero = torch.zeros_like(terms['fg'])
            terms.update(vote=zero, prior=zero, penalty=zero, delta=zero)
            field = bank.response(terms, upto='bg').reshape(64, 64).numpy()
        evidence.last_info = info
        evidence.last_aux = {name:value.numpy().copy() for name,value in auxiliary.items()}
        return field
    evidence.last_info = {}
    evidence.last_aux = {}
    return evidence



def mask(field, backend):
    import numpy as np
    import torch
    from ics.methods import stage_bank as bank
    field = np.asarray(field, np.float32).reshape(64, 64)
    if not np.isfinite(field).all():
        raise ValueError('Nonfinite64x64 evidence field')
    return (bank.unit(torch.from_numpy(field)).numpy() if backend == 'raw' else field) > .5


def counts(prediction, truth, *, soft=False):
    import numpy as np
    if soft:
        t = np.asarray(truth, np.float32)
        return np.array([(prediction*t).sum(dtype=np.float64),
                         t.sum(dtype=np.float64)+(prediction*(1-t)).sum(dtype=np.float64)])
    target = np.asarray(truth, np.float32) > .5
    return np.array([(prediction & target).sum(), (prediction | target).sum()], dtype=np.int64)


def miou(values, classes, weights=None):
    from ics.experiment import metric
    return metric(values,classes,weights)


def compare(actual, saved, backend):
    import numpy as np
    a, b = np.asarray(actual, np.float32), np.asarray(saved, np.float32)
    difference = np.abs(a-b)
    result = dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
        actual_array_sha256=array_sha(a), saved_array_sha256=array_sha(saved),
        mask_different_tokens=int(np.count_nonzero(mask(a, backend) != mask(b, backend))),
        field_differences_are_diagnostics_not_an_extra_pass_threshold=True)
    if np.asarray(saved).dtype == np.float16:
        s = np.asarray(saved)
        previous = np.nextafter(s, np.full_like(s, -np.inf)).astype(np.float32)
        following = np.nextafter(s, np.full_like(s, np.inf)).astype(np.float32)
        lower, upper = (previous+b)/2, (following+b)/2
        result.update(saved_dtype='float16', FP16_roundtrip_different_tokens=int(np.count_nonzero(a.astype(np.float16) != s)),
            outside_pure_FP16_storage_rounding_cell=int(np.count_nonzero((a<lower)|(a>upper))),
            max_FP16_storage_halfULP=float(np.maximum(b-lower, upper-b).max()),
            quantization_policy='Exact adjacent-representable midpoint cells, no tuned atol/rtol; input quantization, re-normalization and execution drift stay separate from storage quantization.')
    return result


def summaries(episodes, classes):
    import numpy as np
    result = {}
    for name in ('acceptance1', 'acceptance2'):
        successful = [e for e in episodes if name in e and 'error' not in e[name]]
        if len(successful) != N:
            result[name] = dict(passed=False, evaluated=len(successful), expected=N,
                                errors=[dict(key=e['key'], error=e[name]['error']) for e in episodes if 'error' in e.get(name, {})])
            continue
        got = np.array([e[name]['actual_IU'] for e in episodes]); reference = np.array([e[name]['saved_IU'] for e in episodes])
        mismatches = sum(e[name]['mask_different_tokens'] for e in episodes)
        actual_miou, saved_miou = miou(got, classes), miou(reference, classes)
        fraction = mismatches/(N*4096)
        result[name] = dict(passed=abs(actual_miou-saved_miou)<=GATE['absolute_class_mIoU_difference_pp'] and fraction<=GATE['aggregate_mask_disagreement_fraction'],
            evaluated=N, mask_different_tokens=int(mismatches), actual_class_mIoU=miou(got, classes),
            saved_class_mIoU=miou(reference, classes), per_episode_IU_equal=bool(np.array_equal(got, reference)),
            class_mIoU_difference_pp=actual_miou-saved_miou,mask_disagreement_fraction=fraction,
            maximum_field_error=max(e[name]['max_abs'] for e in episodes),
            mean_field_error=float(np.mean([e[name]['mean_abs'] for e in episodes])),
            FP16_roundtrip_different_tokens=sum(e[name]['FP16_roundtrip_different_tokens'] for e in episodes),
            outside_pure_FP16_storage_rounding_cell=sum(e[name]['outside_pure_FP16_storage_rounding_cell'] for e in episodes),
            gate=dict(GATE))
    return result



def snapshot(out):
    names = ['scripts/bench_evidence.py', 'scripts/run_order_tokens.py', 'scripts/run_rcg2.py',
             'scripts/score_order_tokens.py', 'scripts/score_size_threshold.py',
             'src/ics/methods/rcg.py', 'src/ics/methods/stage_bank.py', 'src/ics/experiment.py']
    names += ['src/ics/__init__.py','src/ics/methods/__init__.py']
    names += ['scripts/score_reverse_landing.py']
    hashes = {}
    for name in names:
        data = (ROOT/name).read_bytes(); target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def acceptance1_provider(args, fresh, joined, saved, classes, source_hashes):
    """Verify actual input/source bindings, then inspect already computed fields.

    The original producer did not hash/seal its field and receipt outputs. Their
    current consumed hashes are recorded explicitly; no historical output seal
    is invented. Every original feature hash and the ordered replay diagnostics
    are checked before accepting its report as a usable provider.
    """
    import numpy as np
    from ics.methods import rcg
    provider = args.acceptance1_provider
    paths = {name:provider/name for name in ('report.json','receipts.json','replayed_fields.npz')}
    report = json.loads(paths['report.json'].read_text())
    receipts = json.loads(paths['receipts.json'].read_text())
    expected = dict(n=N, CPU_only=True, decomposition_started=False,
        token_source_sha256=sha(args.fresh/'tokens.npz'), rows_sha256=sha(args.fresh/'rows.json'),
        feature_manifest_sha256=sha(args.input_manifest), rcg_source_sha256=source_hashes['src/ics/methods/rcg.py'],
        runner_sha256=sha(args.provider_runner))
    for name,value in expected.items():
        if report.get(name)!=value:
            raise ValueError('Acceptance1 provider source/contract mismatch '+name)
    if report.get('rcg_config')!=rcg.CONFIG or report.get('classes')!=len(np.unique(classes)):
        raise ValueError('Acceptance1 provider canonical RCG/class configuration mismatch')
    if len(receipts)!=N:
        raise ValueError('Acceptance1 provider lacks600 ordered receipts')
    with np.load(paths['replayed_fields.npz'],allow_pickle=False) as z:
        if set(z.files)!={'rcg'}:raise ValueError('Unexpected replay field members')
        fields=z['rcg'].copy()
    if fields.shape!=(N,64,64) or fields.dtype!=np.float32 or not np.isfinite(fields).all():
        raise ValueError('Invalid original replay fields')
    episodes=[]; actual=[]; reference=[]; quantized=[]
    for index,(row,receipt) in enumerate(zip(joined,receipts)):
        write(args.out/'progress.json',dict(state='verifying_existing_acceptance1',index=index,total=N,current_key=row['key']))
        path=resolve(row['feature_export'],args.input_base or args.input_manifest.parent)
        if (receipt.get('key')!=row['key'] or Path(receipt.get('feature_path',''))!=path
            or receipt.get('feature_sha256')!=sha(path) or receipt.get('source_cache_precision')!='torch.float16'
            or not isinstance(receipt.get('debiased'),bool)):
            raise ValueError('Acceptance1 original feature/order/processed-stage mismatch '+row['key'])
        got=compare(fields[index],saved['rcg'][index],'rcg')
        if (receipt['raw_cut_mask_different_tokens']!=got['mask_different_tokens']
            or receipt['FP16_cut_mask_different_tokens']!=int(np.count_nonzero((fields[index].astype(np.float16)>.5)!=(saved['rcg'][index]>.5)))
            or receipt['max_abs_error']!=got['max_abs'] or receipt['mean_abs_error']!=got['mean_abs']):
            raise ValueError('Existing replay field/receipt diagnostics mismatch '+row['key'])
        got.update(actual_IU=counts(mask(fields[index],'rcg'),saved['truth'][index]).tolist(),
            saved_IU=counts(mask(saved['rcg'][index],'rcg'),saved['truth'][index]).tolist(),
            actual_soft_IU=counts(mask(fields[index],'rcg'),saved['truth'][index],soft=True).tolist(),
            saved_soft_IU=counts(mask(saved['rcg'][index],'rcg'),saved['truth'][index],soft=True).tolist(),
            reused_original_receipt=receipt,actual_new_RCG_calls=0)
        actual.append(got['actual_IU']);reference.append(got['saved_IU'])
        quantized.append(counts(fields[index].astype(np.float16)>.5,saved['truth'][index]).tolist())
        episodes.append(dict(key=row['key'],support=row['support'],query=row['query'],c=row['c'],fold=row['fold'],acceptance1=got))
    actual=np.array(actual);reference=np.array(reference);quantized=np.array(quantized)
    expected_scores=dict(stored=miou(reference,classes),replay=miou(actual,classes),replay_FP16_compatibility=miou(quantized,classes))
    # This is report arithmetic consistency, not an acceptance tolerance. The
    # output gate below still requires exact masks and per-episode I/U.
    for name,value in expected_scores.items():
        if abs(report['score'][name]-value)>1e-12:
            raise ValueError('Original provider class-IoU report mismatch '+name)
    if (report['raw_mask_different_tokens']!=sum(e['acceptance1']['mask_different_tokens'] for e in episodes)
        or report['episodes_raw_mask_different']!=sum(e['acceptance1']['mask_different_tokens']>0 for e in episodes)
        or report['maximum_field_error']!=max(e['acceptance1']['max_abs'] for e in episodes)):
        raise ValueError('Original provider overall difference report mismatch')
    bindings={name:dict(path=str(path),sha256=sha(path)) for name,path in paths.items()}
    bindings['runner']=dict(path=str(args.provider_runner),sha256=sha(args.provider_runner))
    return episodes,dict(files=bindings,producer_report=report,actual_new_RCG_calls=0,
        original_output_seal='Producer reported input/source hashes but no field/receipt output seal; current consumed output hashes recorded, not retrospectively invented.')



def write_npz(path, values):
    import numpy as np
    target=Path(path);temporary=target.with_suffix(target.suffix+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**values)
    temporary.replace(target)


def input_hashes(args):
    return dict(rows_sha256=sha(args.fresh/'rows.json'),tokens_sha256=sha(args.fresh/'tokens.npz'),
                input_manifest_sha256=sha(args.input_manifest))


def versions():
    result={}
    for name in ('numpy','torch','scipy','scikit-learn','Pillow'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]='distribution metadata unavailable'
    return result


def assert_recipe():
    from ics.methods import stage_bank,rcg
    for key,value in dict(tau=.6,lse_temperature=.07,bg_weight=.55,hard_bg=.2).items():
        if stage_bank.CONFIG[key]!=value:raise ValueError('Native Part2 constant changed: '+key)
    for key,value in dict(alpha=.5,query_k=20,cross_image_k=10,source_purity=.9,confidence_floor=.1).items():
        if rcg.CONFIG[key]!=value:raise ValueError('Fixed RCG constant changed: '+key)
    if rcg.CONFIG['lambda']!=16 or rcg.CONFIG['cg_maxiter']!=300:
        raise ValueError('Fixed RCG solve changed')


def acceptance2_grade(summary):
    if summary.get('evaluated')!=N:
        return dict(continue_ladder=False,band='incomplete',baseline=None,mask_one_percent_rule='not evaluated')
    gap=abs(summary['class_mIoU_difference_pp']);mask_ok=summary['mask_disagreement_fraction']<=.01
    if gap<=.3 and mask_ok:
        return dict(continue_ladder=True,band='within.3pp_and1percent',baseline='stored_s2',mask_one_percent_rule='passed')
    if .3<gap<=2.:
        return dict(continue_ladder=True,band='.3_to2pp_use_recomputed_endpoint',baseline='recomputed_s2',
            mask_one_percent_rule='passed' if mask_ok else 'exceeded; continuation follows the explicit graded PLAN exception, not source equivalence',
            source_equivalence_claim=False)
    return dict(continue_ladder=False,band='above2pp' if gap>2 else 'within.3pp_but_mask_rule_failed',
                baseline=None,mask_one_percent_rule='passed' if mask_ok else 'failed')


def worker_init(threads):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]=str(threads)
    import torch
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)


_FROZEN_WORKER=None
def frozen_dispatch(frozen,operation,index,row,cfg):
    """Execute the saved script/dependency closure, never a changing live body."""
    global _FROZEN_WORKER
    if _FROZEN_WORKER is None:
        import importlib.util
        for name in list(sys.modules):
            if name=='ics' or name.startswith('ics.'):del sys.modules[name]
        sys.path.insert(0,str(Path(frozen)/'src'))
        spec=importlib.util.spec_from_file_location('_frozen_evidence_bench',Path(frozen)/'scripts/bench_evidence.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        _FROZEN_WORKER=module
    return _FROZEN_WORKER.compute_worker(operation,index,row,cfg)


def ladder_evidence(level, auxiliary):
    """Known-reference-only evidence(q,r,cov)->64x64; no query labels."""
    def evidence(q,r,cov):
        import torch
        import torch.nn.functional as F
        with torch.inference_mode():
            x=F.normalize(torch.from_numpy(q),dim=1)
            if level==6:return auxiliary['s2'].copy()
            fg=x@torch.from_numpy(auxiliary['mu_fg'])
            if level==1:value=fg
            elif level==2:value=fg-.55*(x@torch.from_numpy(auxiliary['mu_bg_all']))
            elif level==3:value=fg-.55*(x@torch.from_numpy(auxiliary['mu_bg_hard']))
            elif level==4:value=fg-.55*(x@torch.from_numpy(auxiliary['mu_bg_orth']))
            elif level==5:value=(x@torch.from_numpy(auxiliary['fg_prototypes']).T).max(1).values-.55*(x@torch.from_numpy(auxiliary['mu_bg_orth']))
            else:raise ValueError('Exactly six user-fixed levels')
        return value.reshape(64,64).numpy()
    return evidence


_SAVED_S2=None
_SAVED_REF=None
def compute_worker(operation,index,row,cfg):
    import numpy as np
    import torch
    from ics.methods import rcg,stage_bank
    global _SAVED_S2,_SAVED_REF
    started=time.monotonic()
    record=dict(index=index,key=row['key'],query_GT_indexed_in_worker=False)
    try:
        assert_recipe()
        feature_path=Path(row['feature_export']);packet_path=Path(row['packet_export'])
        q,r,provenance=features(feature_path);pack,packet_hash=packet(packet_path)
        if _SAVED_REF is None:
            with np.load(cfg['tokens_path'],allow_pickle=False) as z:_SAVED_REF=z['ref'].astype(np.float32).reshape(N,64,64)
        if not np.array_equal(pack['cov'],_SAVED_REF[index]):
            raise ValueError('Packet coverage differs from the exact saved fresh600 reference coverage '+row['key'])
        record.update(features=provenance,packet_sha256=packet_hash)
        if operation=='accept2':
            fg,prompt=native_foreground(row,cfg['annotation_root'],pack['cov'])
            prefix,prefix_sha=part2_prefix();head=native_evidence(fg,prefix)
            begin=time.monotonic();s2=head(q,r,pack['cov']);part2_seconds=time.monotonic()-begin
            values=dict(s2=s2,**head.last_aux)
            if any(value.dtype!=np.float32 or not np.isfinite(value).all() for value in values.values()):
                raise ValueError('Nonfinite/wrong precision native Part2 or auxiliary field')
            path=Path(cfg['out'])/'acceptance2_fields'/(row['key']+'.npz')
            begin=time.monotonic();write_npz(path,values);save_seconds=time.monotonic()-begin
            record.update(state='complete',field_path=str(path),field_sha256=sha(path),
                field_members=list(values),field_array_hashes={k:array_sha(v) for k,v in values.items()},
                reference_mask=prompt,part2=head.last_info,prefix_source_sha256=prefix_sha,
                native_part2_seconds=part2_seconds,save_seconds=save_seconds,
                auxiliary_scope='Only reference means/prototypes already in native Part2 plus full-BG reference mean; no other-level query predictions before acceptance passes')
        else:
            if _SAVED_S2 is None:
                with np.load(cfg['tokens_path'],allow_pickle=False) as z:_SAVED_S2=z['s2'].astype(np.float32).reshape(N,64,64)
            entry=cfg['acceptance2_receipts'][row['key']]
            if entry['features']['feature_sha256']!=provenance['feature_sha256'] or entry['packet_sha256']!=packet_hash:
                raise ValueError('Ladder input differs from accepted native Part2')
            source=Path(entry['field_path'])
            if sha(source)!=entry['field_sha256']:raise ValueError('Accepted Part2/auxiliary field changed')
            with np.load(source,allow_pickle=False) as z:auxiliary={name:z[name].copy() for name in z.files}
            values={};timings={};solvers={}
            for number,(name,title) in enumerate(LEVELS,1):
                begin=time.monotonic();field=ladder_evidence(number,auxiliary)(q,r,pack['cov'])
                with torch.inference_mode():unit=stage_bank.unit(torch.from_numpy(field)).numpy()
                evidence_seconds=time.monotonic()-begin
                begin=time.monotonic();graph,info=rcg.predict(q,r,pack['cov'],field,device='cpu')
                values[name+'.evidence']=field;values[name+'.unit']=unit;values[name+'.rcg']=graph
                timings[name]=dict(evidence_and_unit_seconds=evidence_seconds,RCG_seconds=time.monotonic()-begin,
                    reused_native_part2_preparation=number==6)
                solvers[name]=info
            baseline=auxiliary['s2'] if cfg['baseline']=='recomputed_s2' else _SAVED_S2[index]
            with torch.inference_mode():unit=stage_bank.unit(torch.from_numpy(baseline)).numpy()
            # Under the graded middle band, L6 is the exact same evidence and
            # complete RCG call as the baseline. Reuse that completed field.
            if cfg['baseline']=='recomputed_s2':
                graph=values['L6.rcg'];baseline_seconds=0.;baseline_solver=solvers['L6']
            else:
                begin=time.monotonic();graph,baseline_solver=rcg.predict(q,r,pack['cov'],baseline,device='cpu')
                baseline_seconds=time.monotonic()-begin
            values.update(baseline_evidence=baseline,baseline_unit=unit,baseline_rcg=graph)
            if any(value.shape!=(64,64) or value.dtype!=np.float32 or not np.isfinite(value).all() for value in values.values()):
                raise ValueError('Every saved ladder field must be finite FP32 physical64x64')
            path=Path(cfg['out'])/'ladder_fields'/(row['key']+'.npz')
            begin=time.monotonic();write_npz(path,values)
            record.update(state='complete',field_path=str(path),field_sha256=sha(path),
                field_members=list(values),field_array_hashes={k:array_sha(v) for k,v in values.items()},
                timings=timings,solvers=solvers,baseline=cfg['baseline'],baseline_RCG_seconds=baseline_seconds,
                baseline_solver=baseline_solver,save_seconds=time.monotonic()-begin,
                reused_acceptance2_field_sha256=entry['field_sha256'],
                native_part2_preparation_seconds_from_acceptance2=entry['native_part2_seconds'])
        record['worker_wall_seconds']=time.monotonic()-started
    except Exception as error:
        record.update(state='failed',error=repr(error),traceback=traceback.format_exc(),worker_wall_seconds=time.monotonic()-started)
    return record


def parallel_phase(args,operation,joined,cfg):
    results=[None]*N
    (args.out/('acceptance2_fields' if operation=='accept2' else 'ladder_fields')).mkdir()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(args.threads,)) as pool:
        pending={pool.submit(frozen_dispatch,str(args.out/'source'),operation,i,row,cfg):i for i,row in enumerate(joined)}
        for completed,future in enumerate(concurrent.futures.as_completed(pending),1):
            index=pending[future]
            try:record=future.result()
            except Exception as error:record=dict(index=index,key=joined[index]['key'],state='failed',error=repr(error),traceback=traceback.format_exc())
            results[index]=record
            write(args.out/(operation+'_receipts.json'),results)
            write(args.out/'progress.json',dict(state=operation,completed=completed,total=N,last_key=record['key']))
    return results


def acceptance2_result(args,receipts,joined,saved,classes,base):
    import numpy as np
    episodes=[];complete=[]
    for index,(row,record) in enumerate(zip(joined,receipts)):
        if record['state']!='complete':
            episodes.append(dict(key=row['key'],acceptance2=dict(error=record['error'])));continue
        source=Path(record['field_path'])
        if sha(source)!=record['field_sha256']:raise ValueError('Part2 field changed before scoring')
        with np.load(source,allow_pickle=False) as z:field=z['s2'].copy()
        details=compare(field,saved['s2'][index],'raw')
        details.update(actual_IU=counts(mask(field,'raw'),saved['truth'][index]).tolist(),
            saved_IU=counts(mask(saved['s2'][index],'raw'),saved['truth'][index]).tolist())
        episodes.append(dict(key=row['key'],acceptance2=details));complete.append(record)
    summary=summaries(episodes,classes)['acceptance2'];grade=acceptance2_grade(summary)
    result=dict(state='ACCEPTANCE2_CONTINUE' if grade['continue_ladder'] else 'STOPPED_ACCEPTANCE2',n=N,
        ordered600_keys=base['ordered600_keys'],input_hashes=base['input_hashes'],source_hashes=base['source_hashes'],
        summary=summary,grade=grade,gate=dict(GATE),reference_area_binding='exact; never weakened by graded readout tolerances',
        receipts={record['key']:record for record in complete},decomposition_executed=False)
    write(args.out/'acceptance2_episodes.json',episodes);write(args.out/'acceptance2_report.json',result)
    return result


def reuse_acceptance2(args,base,joined,saved,classes):
    provider=args.acceptance2_provider;path=provider/'acceptance2_report.json'
    result=json.loads(path.read_text())
    if result.get('ordered600_keys')!=base['ordered600_keys'] or result.get('input_hashes')!=base['input_hashes']:
        raise ValueError('Acceptance2 provider is a different ordered cohort/source')
    if result.get('source_hashes')!=base['source_hashes'] or result.get('gate')!=GATE:
        raise ValueError('Acceptance2 source closure or fixed gate differs; do not silently reuse another recipe')
    if not result.get('grade',{}).get('continue_ladder') or len(result.get('receipts',{}))!=N:
        raise ValueError('Acceptance2 provider did not pass the current graded continuation rule')
    receipts=[result['receipts'][row['key']] for row in joined]
    verified=acceptance2_result(args,receipts,joined,saved,classes,base)
    if verified['summary']!=result['summary'] or verified['grade']!=result['grade']:
        raise ValueError('Existing native Part2 fields no longer reproduce their provider summary')
    verified['reused_provider']=dict(path=str(path),sha256=sha(path),new_Part2_predictions=0)
    return verified


def failure_sets(saved):
    """Fixed stored-RCG diagnostic selection from score_reverse_landing.py."""
    import numpy as np
    truth=saved['truth'].astype(np.float32).reshape(N,4096)>.5
    reference=saved['rcg'].astype(np.float32).reshape(N,4096)
    best=np.zeros(N)
    for level in np.linspace(.1,.9,33):
        prediction=reference>level
        i=(prediction&truth).sum(1);u=(prediction|truth).sum(1)
        best=np.maximum(best,i/np.maximum(u,1))
    wrong=(reference>.5)&~truth
    comparable=(best<.5)&(truth.sum(1)>0)&(wrong.sum(1)>0)
    return truth,wrong,comparable,dict(source='scripts/score_reverse_landing.py',
        selection='Original stored RCG best binary-token IoU<.5 on fixed33 levels.1..9; true and wrong-kept regions nonempty',
        failure_count=int((best<.5).sum()),comparable_count=int(comparable.sum()),
        definition='target: truth>.5; wrong: original storedRCG>.5 and not target; never reselect with candidate scores')


def bootstrap_samples(values,classes,weights):
    import numpy as np
    result=np.zeros(len(weights));present_count=np.zeros(len(weights),np.int64)
    for cls in np.unique(classes):
        keep=classes==cls;counts=weights[:,keep]@np.asarray(values[keep],np.float64)
        present=weights[:,keep].sum(1)>0
        result+=(counts[:,0]/np.maximum(counts[:,1],1))*present
        present_count+=present
    return result/np.maximum(present_count,1)*100


def ladder_result(args,receipts,fresh,saved,classes,accept2):
    import numpy as np
    from ics.experiment import photo_groups
    if any(record['state']!='complete' for record in receipts):
        return dict(state='LADDER_FAILED_STOP',complete=sum(record['state']=='complete' for record in receipts),
                    errors=[r for r in receipts if r['state']!='complete'],decomposition_executed=True)
    truth,wrong,comparable,failure_info=failure_sets(saved)
    arrays={(name,backend):np.zeros((N,2),np.int64) for name in [level[0] for level in LEVELS]+['baseline'] for backend in ('raw','rcg')}
    winners={key:0 for key in arrays};means={key:[] for key in arrays}
    for index,record in enumerate(receipts):
        path=Path(record['field_path'])
        if sha(path)!=record['field_sha256']:raise ValueError('Ladder field changed before scoring')
        with np.load(path,allow_pickle=False) as z:
            for name in [level[0] for level in LEVELS]+['baseline']:
                for backend in ('raw','rcg'):
                    field=z[(name+'.unit' if backend=='raw' else name+'.rcg') if name!='baseline' else 'baseline_'+('unit' if backend=='raw' else 'rcg')].copy()
                    arrays[name,backend][index]=counts(field>.5,saved['truth'][index])
                    if comparable[index]:
                        flat=field.ravel();target_mean=float(flat[truth[index]].mean(dtype=np.float64));wrong_mean=float(flat[wrong[index]].mean(dtype=np.float64))
                        winners[name,backend]+=target_mean>wrong_mean
                        means[name,backend].append(dict(key=fresh[index]['key'],target_mean=target_mean,wrong_region_mean=wrong_mean))
    groups=photo_groups(fresh);number=int(groups.max())+1
    draws=np.random.RandomState(0).randint(number,size=(2000,number))
    weights=np.stack([np.bincount(draw,minlength=number) for draw in draws])[:,groups]
    np.save(args.out/'bootstrap_photo_draws.npy',draws)
    baseline_samples={backend:bootstrap_samples(arrays['baseline',backend],classes,weights) for backend in ('raw','rcg')}
    table=[]
    for name,title in LEVELS:
        row=dict(level=name,title=title)
        for backend in ('raw','rcg'):
            values=arrays[name,backend];score=miou(values,classes);reference=miou(arrays['baseline',backend],classes)
            delta=bootstrap_samples(values,classes,weights)-baseline_samples[backend]
            row[backend]=dict(class_mIoU=score,baseline_class_mIoU=reference,gain_vs_matched_s2=score-reference,
                paired_ci95=np.percentile(delta,[2.5,97.5]).tolist(),
                failing_target_mean_above_wrong_region=int(winners[name,backend]),comparable_failure_count=failure_info['comparable_count'])
        row['timings']={key:dict(mean_seconds=float(np.mean([r['timings'][name][key] for r in receipts])),
            median_seconds=float(np.median([r['timings'][name][key] for r in receipts])),
            p95_seconds=float(np.percentile([r['timings'][name][key] for r in receipts],95)))
            for key in ('evidence_and_unit_seconds','RCG_seconds')}
        table.append(row)
    write_npz(args.out/'counts.npz',{name+'.'+backend:value for (name,backend),value in arrays.items()})
    write(args.out/'failure_region_means.json',{name+'.'+backend:value for (name,backend),value in means.items()})
    lines=['| level | direct positions mIoU; Δ vs matched s2 [95% CI] | RCG positions mIoU; Δ vs matched s2 [95% CI] | target>wrong, direct/RCG |',
           '|---|---|---|---|']
    for row in table:
        def cell(backend):
            value=row[backend];lo,hi=value['paired_ci95']
            return f"{value['class_mIoU']:.4f}; {value['gain_vs_matched_s2']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        lines.append(f"| {row['level']} {row['title']} | {cell('raw')} | {cell('rcg')} | {row['raw']['failing_target_mean_above_wrong_region']}/{failure_info['comparable_count']} ; {row['rcg']['failing_target_mean_above_wrong_region']}/{failure_info['comparable_count']} |")
    lines.append('| no APD | unavailable in this processed cache | unavailable | — |')
    text='\n'.join(lines)+'\n\nPosition-level binary-token class IoU; reused development fresh600. Baseline: '+accept2['grade']['baseline']+'. Native s2 replay drift: '+str(accept2['summary']['class_mIoU_difference_pp'])+' pp. Gates were chosen after acceptance1.\n'
    (args.out/'table.md').write_text(text)
    return dict(state='LADDER_COMPLETE',n=N,table=table,no_APD=NO_APD,baseline=accept2['grade']['baseline'],
        acceptance2=accept2['summary'],acceptance2_grade=accept2['grade'],gate=dict(GATE),failure_region_protocol=failure_info,
        baseline_failing_target_mean_above_wrong_region={backend:int(winners['baseline',backend]) for backend in ('raw','rcg')},
        bootstrap=dict(draws=2000,seed=0,rng='RandomState(0)',unit='connected support/query photographs',groups=number,
            same_draws_for_all_candidates_and_both_matched_baselines=True),
        bootstrap_class_denominator='Only classes present in each resampled draw, exactly as ics.experiment.metric; class-presence uses sampled row mass, not prediction union.',
        table_path=str(args.out/'table.md'),decomposition_executed=True,
        scientific_scope='Position-level component comparisons; the14pp historical complete-mask gap is not algebraically decomposed by adding these gains.',
        endpoint_level6_reuses_actual_accepted_native_part2=True)


def run(args):
    worker_init(args.threads)
    import numpy as np
    from ics.methods import stage_bank,rcg
    args.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    write(args.out/'progress.json',dict(state='validating_inputs'))
    try:
        assert_recipe();fresh=rows(args.fresh/'rows.json');joined=exact_join(fresh,rows(args.input_manifest));saved=token_fields(args.fresh/'tokens.npz')
        for row in joined:
            for name in ('feature_export','packet_export'):row[name]=str(resolve(row[name],args.input_base or args.input_manifest.parent))
        classes=np.array([row['c'] for row in fresh]);source_hashes=snapshot(args.out)
        base=dict(input_hashes=input_hashes(args),ordered600_keys=[row['key'] for row in fresh],source_hashes=source_hashes,
            exact_join_fields=list(IDENTITY),truth_protocol='saved FP16 area coverage→FP32→strict>.5; binary64x64 token I/U, class sums then macro mean',
            raw_backend='stage_bank.unit then strict>.5',RCG_backend='original rcg.predict CPU output direct>.5, no new minmax',
            RCG_is_not_pure_graph='unchanged reference CSLS density/rank guide alpha.5, score-dependent fidelity, reciprocal20NN and lambda16 solve',
            source_risks=['Historical score was FP32 before field storage; current tokens retain FP16',
                'Original run_rcg2.solve omitted re-normalization and allowed600CG steps; canonical RCG normalizes and allows300',
                'Already Part1-processed FP16 q/r cannot recover raw/no-APD states'],
            reference_annotation_root=str(args.annotation_root),gate=dict(GATE),levels=list(LEVELS),
            acceptance2_graded_rule=dict(within_03='Original stored-s2 baseline only if aggregate mask disagreement<=1%',
                between_03_and2='Continue with recomputed source-s2 endpoint and matched baselines; explicitly report token1% status',
                above2='Stop; no parameter adjustment'),
            native_part2_constants=dict(tau=.6,FG_LSE_temperature=.07,BG_weight=.55,hard_BG_token_fraction=.2,orthogonal_norm_floor=1e-8),
            workers=args.workers,threads=args.threads,phase=args.phase,query_GT_used_for_evidence=False,new_encoder_forwards=0,
            dependency_versions=versions(),
            native_part2=stage_bank.CONFIG,RCG_config=rcg.CONFIG,no_APD=NO_APD)
        write(args.out/'config.json',base);write(args.out/'ordered_rows.json',fresh)
        reports,provider=acceptance1_provider(args,fresh,joined,saved,classes,source_hashes)
        first=summaries(reports,classes)['acceptance1'];write(args.out/'acceptance1_episodes.json',reports)
        result=dict(state='STOPPED_ACCEPTANCE1' if not first['passed'] else 'ACCEPTANCE1_PASSED',acceptance1=first,
                    acceptance1_provider=provider,acceptance2='NOT_RUN',decomposition_executed=False)
        if first['passed'] and args.phase!='accept1':
            cfg=dict(out=str(args.out),annotation_root=str(args.annotation_root),tokens_path=str(args.fresh/'tokens.npz'))
            if args.acceptance2_provider is not None:
                second=reuse_acceptance2(args,base,joined,saved,classes)
            else:
                # Bind the first reference's actual source before launching the
                # parallel corpus. This is an input identity check, not a cost
                # precheck, and it does not substitute area-majority foreground.
                first_packet,_=packet(Path(joined[0]['packet_export']))
                if not np.array_equal(first_packet['cov'],saved['ref'][0].astype(np.float32)):
                    raise ValueError('First packet coverage differs from saved fresh ref')
                native_foreground(joined[0],args.annotation_root,first_packet['cov'])
                receipt=parallel_phase(args,'accept2',joined,cfg)
                second=acceptance2_result(args,receipt,joined,saved,classes,base)
            result.update(state=second['state'],acceptance2=second)
            if second['grade']['continue_ladder'] and args.phase in ('ladder','all'):
                cfg.update(acceptance2_receipts=second['receipts'],baseline=second['grade']['baseline'])
                receipt=parallel_phase(args,'ladder',joined,cfg)
                result.update(ladder_result(args,receipt,fresh,saved,classes,second))
        for relative,digest in source_hashes.items():
            if sha(ROOT/relative)!=digest:raise ValueError('Source changed during finite benchmark: '+relative)
        if input_hashes(args)!=base['input_hashes']:raise ValueError('Ordered cohort/source changed')
        for binding in provider['files'].values():
            if sha(binding['path'])!=binding['sha256']:raise ValueError('Consumed acceptance1 provider changed')
        result.update(n=N,input_hashes=base['input_hashes'],source_hashes=source_hashes,ordered600_keys=base['ordered600_keys'],
            gate=dict(GATE),workers=args.workers,threads=args.threads,elapsed_seconds=time.monotonic()-started,
            actual_new_acceptance1_RCG_calls=0,query_GT_used_for_evidence=False,exposure='reused development; not independent confirmation')
        write(args.out/'report.json',result);write(args.out/'progress.json',dict(state=result['state'],elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(state=result['state'],acceptance1=first),ensure_ascii=False),flush=True)
        return 2 if result['state'].startswith(('STOPPED','FAILED','LADDER_FAILED')) else 0
    except Exception as error:
        write(args.out/'report.json',dict(state='FAILED_STOP',error=repr(error),traceback=traceback.format_exc(),
            decomposition_executed=False,actual_new_acceptance1_RCG_calls=0,elapsed_seconds=time.monotonic()-started))
        write(args.out/'progress.json',dict(state='FAILED_STOP',error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fresh', type=Path, default=DEFAULT_FRESH)
    parser.add_argument('--input-manifest', type=Path, default=DEFAULT_INPUTS)
    parser.add_argument('--input-base', type=Path)
    parser.add_argument('--annotation-root', type=Path, default=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--acceptance1-provider', type=Path, required=True, help='Completed original acceptance1 report/receipts/replayed_fields directory; never rerun600')
    parser.add_argument('--provider-runner', type=Path, required=True, help='Actual acceptance1_cpu.py bytes matching provider runner_sha256')
    parser.add_argument('--phase', choices=('accept1','accept2','ladder','all'), default='accept2')
    parser.add_argument('--acceptance2-provider',type=Path,help='Passed/graded600 acceptance2 directory, reused by ladder')
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if min(args.workers,args.threads)<1 or args.workers*args.threads>32:
        parser.error('Positive CPU workers*threads<=32 required')
    if args.phase=='ladder' and args.acceptance2_provider is None:
        parser.error('ladder requires an existing --acceptance2-provider; no acceptance2 rerun')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
