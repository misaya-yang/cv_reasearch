"""Fixed reference-only paired focus diagnostics; no query raw/labels/fields."""
import builtins
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
RUN = REPO.parent/'cv_data/a/reference_focus_head200_20261010'
HEAD_SHA = '37c0f57254ff7d398b7cdfa5486d2bbd4c51a03958206e2d0fc6445329a48413'


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(json.dumps([list(a.shape), a.dtype.str]).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def rank(score, fg, bg):
    f, b, s = np.asarray(fg, np.float64), np.asarray(bg, np.float64), np.asarray(score, np.float64)
    keep = f+b > 0
    f, b, s = f[keep], b[keep], s[keep]
    if not f.sum() or not b.sum(): return None
    order = np.argsort(s, kind='stable')
    s, f, b = s[order], f[order], b[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1])+1]
    f, b = np.add.reduceat(f, starts), np.add.reduceat(b, starts)
    return float(np.sum(f*(np.cumsum(b)-.5*b))/(f.sum()*b.sum()))


def role_stats(features, fg, bg):
    x = features.double()
    af, ab = fg/fg.sum(), bg/bg.sum()
    mf, mb = af @ x, ab @ x
    vf, vb = (af*((x-mf).square().sum(1))).sum(), (ab*((x-mb).square().sum(1))).sum()
    return dict(contrast_norm=float((mf-mb).norm()),
        mean_cosine=float(mf @ mb/(mf.norm()*mb.norm()).clamp_min(1e-15)),
        fg_variance=float(vf), bg_variance=float(vb),
        contrast_to_within_variance=float((mf-mb).square().sum()/(vf+vb).clamp_min(1e-15)))


def score_stats(score, fg, bg):
    s = score.detach().numpy()
    f, b = fg.numpy(), bg.numpy()
    return dict(auc=rank(s, f, b), fg_mean=float(np.dot(s, f)/f.sum()),
        bg_mean=float(np.dot(s, b)/b.sum()),
        fg_negative_mass_fraction=float(f[s <= 0].sum()/f.sum()),
        bg_positive_mass_fraction=float(b[s > 0].sum()/b.sum()),
        binary_role_loss=float(.5*np.dot(f, (s-1)**2)/f.sum()+.5*np.dot(b, (s+1)**2)/b.sum()))


def drift_stats(actual, derived, coverage):
    x, z = actual.double(), derived.double()
    delta = x-z
    f, b = coverage/coverage.sum(), (1-coverage)/(1-coverage).sum()
    mf, mb = f @ delta, b @ delta
    result = dict(common_mean_drift_norm=float((.5*(mf+mb)).norm()),
        role_contrast_drift_norm=float((mf-mb).norm()),
        fg_mean_drift_norm=float(mf.norm()), bg_mean_drift_norm=float(mb.norm()),
        aligned_cosine_mean=float((x*z).sum(1).mean()),
        aligned_cosine_fg_mean=float(f @ (x*z).sum(1)), aligned_cosine_bg_mean=float(b @ (x*z).sum(1)))
    for name, a in (('foreground', f), ('background', b)):
        mx, mz = a @ x, a @ z
        vx = (a*((x-mx).square().sum(1))).sum()
        vz = (a*((z-mz).square().sum(1))).sum()
        cross = (a*((x-mx)*(z-mz)).sum(1)).sum()
        result[name+'_centered_trace_correlation'] = float(cross/(vx*vz).sqrt().clamp_min(1e-15))
        md = a @ delta
        result[name+'_within_drift_variance'] = float((a*((delta-md).square().sum(1))).sum())
    return result


def raw(request, profile_id):
    info = read(request['entry_path'])
    assert sha(request['entry_path']) == request['entry_sha256']
    assert info['profile_id'] == profile_id and info['input_tensor_hash'] == request['input_tensor_hash']
    assert info['file_sha256'] == request['payload_sha256'] and sha(request['payload_path']) == request['payload_sha256']
    with np.load(request['payload_path'], allow_pickle=False) as z: a = z['O/24'].copy()
    assert a.shape == (4096,1024) and a.dtype == np.float32 and array_sha(a) == request['tensor_sha256']
    return torch.from_numpy(a)


@torch.inference_mode()
def main():
    assert not (ROOT/'reference.json').exists(), 'Never replace completed reference analysis'
    cfg, seal = read(RUN/'config.json'), read(RUN/'sealed.json')
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED' and seal['n'] == 200
    assert sha(RUN/'config.json') == seal['config_sha256']
    tasks, manifest, flags = read(RUN/'tasks.json'), read(RUN/'manifest.json'), read(RUN/'source_apd_flags.json')
    assert sha(RUN/'tasks.json') == cfg['tasks_sha256'] and sha(RUN/'manifest.json') == cfg['manifest_sha256']
    flag_rows = flags if isinstance(flags, list) else flags['rows']
    flag_map = {r['episode_id']: r['apply_apd'] for r in flag_rows}
    module_path = RUN/'frozen/src/ics/methods/reference_focus_head.py'
    assert sha(module_path) == HEAD_SHA
    spec = importlib.util.spec_from_file_location('focus_mechanism_frozen_head', module_path)
    head = importlib.util.module_from_spec(spec); spec.loader.exec_module(head)
    basis = torch.load(RUN/'positional_basis.pt', map_location='cpu', weights_only=True)['basis'].float()
    projection = torch.eye(1024)-basis@basis.T
    profile_id = Path(cfg['profile_path']).parent.name
    recipe = dict(head_sha256=HEAD_SHA, source_seal_sha256=sha(RUN/'sealed.json'),
        tasks_sha256=cfg['tasks_sha256'], source_apd_flags_sha256=sha(RUN/'source_apd_flags.json'),
        metrics=['aligned real-derived cosine','full physical role source fit/score',
            'paired within-role trace correlation/drift variance',
            'selected role kernel overlap/effective patterns',
            '2fold physical64px checkerboard source-heldout cross-representation fit'],
        cv='Inside focus, both whole/fine observations of a physical64x64 block are held out together; outside whole remains legal training support. Two checkerboard folds; correlated same-image diagnostic only.',
        query_GT_read=False, query_features_read=False, new_encoder_calls=0,
        queries='No query raw, labels, predictions or fields in this reference phase',
        no_parameter_search=True, threads=1, fixed_source_budget=128, ridge=.01,
        postseal_query_plan='Fine128 field ordering/zero-sign versus same canonical pixel masses; full grid/FoRIS foreground/frozen actual-derived disagreements; fixed6 fields; no masks generated or thresholds chosen')
    (ROOT/'recipe.json').write_text(json.dumps(recipe, indent=2)+'\n')
    denied = {str(Path(r['query_mask_path']).resolve()) for r in manifest}
    allowed_reference = {str(Path(r['reference_mask_path']).resolve()) for r in manifest}
    denied -= allowed_reference
    original_builtin, original_io = builtins.open, io.open
    def guard(original):
        def opener(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                p = Path(path).resolve()
                if str(p) in denied or p.parent in {RUN/'fields', RUN/'predictions'} or p.name in {'report.json','scored_episodes.jsonl'}:
                    raise PermissionError('Reference phase attempted query/scoring artifact')
            return original(path, *args, **kwargs)
        return opener
    builtins.open, io.open = guard(original_builtin), guard(original_io)
    started, results = time.perf_counter(), []
    try:
        for i, (row, task) in enumerate(zip(manifest, tasks)):
            tick = time.perf_counter()
            assert row['episode_id'] == task['episode_id']
            whole = raw(task['raw'][0], profile_id)
            focus_binding = task['focus']
            request = dict(focus_binding, tensor_sha256=focus_binding['O24']['tensor_sha256'])
            focus = raw(request, profile_id)
            with Image.open(row['reference_mask_path']) as image:
                m = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
            assert array_sha(m) == row['reference_mask_hash']
            mask = F.interpolate(torch.from_numpy(m).float()[None,None], (1024,1024),mode='nearest')[0,0]
            assert array_sha(mask.numpy().astype(np.uint8)) == focus_binding['canonical_mask_uint8_array_sha256']
            box = tuple(focus_binding['box_xyxy']); measure = head._reference_measure(mask.double(), box)
            sample = head._role_quadrature(measure); apd = bool(flag_map[row['episode_id']])
            w = head._representation(whole, apd, projection)
            f = head._representation(focus, apd, projection)
            d = head._representation(head._derived_focus(whole, box), apd, projection)
            features = {'actual':torch.cat((w,f)), 'derived':torch.cat((w,d))}
            fits = {name:head._fit(x,measure,sample) for name,x in features.items()}
            full = {}; scores = {}
            for name, x in features.items():
                fits[name]['diagnostics']['role_features'] = role_stats(x, measure['foreground']/measure['foreground'].sum(), measure['background']/measure['background'].sum())
                for target, xx in features.items():
                    score = xx.double()@fits[name]['coefficient']+fits[name]['bias']; scores[name,target] = score
                    full[name+'->'+target] = score_stats(score,measure['foreground'],measure['background'])
                    for view, sl in (('whole',slice(0,4096)),('focus',slice(4096,None))):
                        full[name+'->'+target+'.'+view] = score_stats(score[sl],measure['foreground'][sl],measure['background'][sl])
            # Fixed compact semantic kernel diagnostics on sampled source atoms.
            kernel = {}
            pf,pb = sample['foreground_risk']*2,sample['background_risk']*2
            for name,x in features.items():
                selected=x[sample['ids']]; K=torch.exp(((selected@selected.T).double()-1)/.07)
                ff,bb,fb=pf@K@pf,pb@K@pb,pf@K@pb
                kernel[name]=dict(fg_effective_patterns=float(1/ff),bg_effective_patterns=float(1/bb),
                    cross_role_normalized_overlap=float(fb/(ff*bb).sqrt()))
            # No target labels or mode/temperature adjustments are selected from CV.
            xx,yy=box[0]//16,box[1]//16
            coarse_fold=(torch.arange(32)[:,None]//4+torch.arange(32)[None,:]//4)%2
            cv=[]
            for fold in (0,1):
                held_whole=torch.zeros((64,64),dtype=torch.bool);held_whole[yy:yy+32,xx:xx+32]=coarse_fold==fold
                held_focus=(coarse_fold==fold).repeat_interleave(2,0).repeat_interleave(2,1)
                held=torch.cat((held_whole.reshape(-1),held_focus.reshape(-1)))
                train=dict(measure)
                for key in ('foreground','background'): train[key]=measure[key].clone();train[key][held]=0
                if not float(train['foreground'].sum())>0 or not float(train['background'].sum())>0:
                    cv.append(dict(fold=fold,missing_training_role=True));continue
                ss=head._role_quadrature(train)
                for name,x in features.items():
                    fitted=head._fit(x,train,ss)
                    for target,xx_features in features.items():
                        score=xx_features.double()@fitted['coefficient']+fitted['bias']
                        cv.append(dict(fold=fold,train=name,test=target,
                            heldout=score_stats(score[held],measure['foreground'][held],measure['background'][held])))
            cosine=float(fits['actual']['coefficient']@fits['derived']['coefficient']/(fits['actual']['coefficient'].norm()*fits['derived']['coefficient'].norm()).clamp_min(1e-15))
            results.append(dict(index=i,episode_id=row['episode_id'],dataset=row['dataset'],apply_apd=apd,
                source_raw_keys=[task['raw'][0]['key'],focus_binding['key']],
                full_source_scores=full, fits={n:v['diagnostics'] for n,v in fits.items()},
                paired_focus_drift=drift_stats(f,d,measure['coverage'][4096:]), kernel=kernel,
                coefficient_cosine=cosine,bias_actual_minus_derived=float(fits['actual']['bias']-fits['derived']['bias']),
                source_holdout_cv=cv,seconds=time.perf_counter()-tick))
            if (i+1)%25==0: print(json.dumps({'reference_completed':i+1,'seconds':time.perf_counter()-started}),flush=True)
    finally:
        builtins.open, io.open = original_builtin, original_io
    payload=dict(state='REFERENCE_ONLY_SEALED',n=len(results),recipe_sha256=sha(ROOT/'recipe.json'),
        script_sha256=sha(__file__),query_GT_reads=0,query_raw_reads=0,raw_writes=0,encoder_calls=0,
        reference_raw_requests=2*len(results),results=results,seconds=time.perf_counter()-started)
    (ROOT/'reference.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':payload['state'],'n':len(results),'seconds':payload['seconds']}),flush=True)


if __name__=='__main__': main()
