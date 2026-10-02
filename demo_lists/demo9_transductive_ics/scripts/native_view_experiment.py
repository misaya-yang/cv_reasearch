#!/usr/bin/env python3
"""E4 support-background OR E7 RGB-lattice views, existing frozen DINO assets.

No downloads/training/pool/cache. Actual INSID3 anchor plus matched final-layer
no-APD FROST-component density anchors, NOT full FoRIS/FROST reproduction.
Every prediction is frozen before the evaluator opens query annotations.
GPU execution requires --allow-gpu AND explicit owner evidence; root owns queue.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tics.reference_views import (support_background_views, patch_lattice_views,
    background_holdout_factory, fit_support_view_weights, combine_view_outputs)
from tics.readout_baselines import (frost_whitening, frost_bandwidth, frost_kde_ratio,
    frost_candidate_gate, frost_bilateral, frost_finalize, frost_style_dense_readout)
from tics.native_assets import reuse_native_basis
from global_representation_probe import fixed_selection, pixel_ledger, summary


class DensityDegenerate(ValueError):
    """Known insufficient labeled anchors ONLY, never a numeric/shape fault."""
    pass


def density_evidence(support, fg, valid, query):
    """Same normalized final-descriptor whitening/LOO/KDE/source gate per view.

    Invalid support patches are neither positive nor negative anchors. Query
    validity is handled by caller's geometric contribution mask after density.
    """
    reference = F.normalize(support.float().flatten(1).T, dim=1)
    valid = valid.bool().flatten()
    label = fg.bool().flatten()[valid]
    reference = reference[valid]
    if min(int(label.sum()), int((~label).sum())) < 2:
        raise DensityDegenerate('Need >=2 valid support anchors of each label')
    transform = frost_whitening(reference[label], reference[~label])
    white_ref = reference @ transform
    white_query = F.normalize(query.float().flatten(1).T, dim=1) @ transform
    white_fg, white_bg = F.normalize(white_ref[label], dim=1), F.normalize(white_ref[~label], dim=1)
    sigma, loo = frost_bandwidth(white_fg, white_bg)
    margin = frost_kde_ratio(F.normalize(white_query, dim=1), white_fg, white_bg, sigma)
    gate = frost_candidate_gate(white_query, [white_ref], [label])
    if not torch.isfinite(margin).all():
        raise RuntimeError('Nonfinite density margin; numerical fault must not become baseline fallback')
    shape = query.shape[-2:]
    return margin.reshape(shape), gate.reshape(shape), dict(sigma=sigma, loo_margin=loo,
        fg_anchors=int(label.sum()), bg_anchors=int((~label).sum()))


def density_holdout(feature, train_fg, train_bg, held_valid):
    """No held labels arrive; raw logits -> sigmoid probability for Brier fit."""
    probabilities = []
    for b in range(feature.shape[0]):
        valid = train_fg[b, 0] | train_bg[b, 0]
        if min(int(train_fg[b].sum()), int(train_bg[b].sum())) < 2:
            # Ineligible sample is never scored by the weight fitter.
            probabilities.append(torch.full_like(feature[b, 0], .5))
            continue
        margin, _, _ = density_evidence(feature[b], train_fg[b, 0], valid, feature[b])
        probabilities.append(margin.sigmoid())
    return torch.stack(probabilities)[:, None]


def insufficient_anchor_fallback(predictions, host, failure):
    """Prespecified fallback is scoped to insufficient-FG/BG exceptions."""
    if not isinstance(failure,DensityDegenerate):
        raise RuntimeError('Unrecognized error must propagate, not become baseline fallback') from failure
    result=dict(predictions)
    result.setdefault('native_dense',host)
    result['uniform_views']=result['native_dense']
    result['fitted_views']=result['native_dense']
    return result


def decode_margin(margin, gate, native_query, rgb_grid, fraction, model_shape):
    if not torch.isfinite(margin).all():raise RuntimeError('Nonfinite ensemble margin')
    geometry = F.normalize(native_query.float().flatten(1).T, dim=1)
    smooth = frost_bilateral(margin.flatten(), geometry, rgb_grid.reshape(-1, 3), margin.shape)
    if not torch.isfinite(smooth).all():raise RuntimeError('Nonfinite bilateral density; cannot declare a prediction')
    return frost_finalize(smooth.reshape(margin.shape), gate, fraction, model_shape)


def predict_bank(sbank, qbank, gold_grid, native_query, rgb_grid, fraction, weights):
    """Fixed density-PROBABILITY averaging + weighted gate majority.

    The singleton matched anchor equals original no-APD component composition.
    Bilateral/finalization uses ORIGINAL native query geometry for every arm.
    """
    if len(qbank.maps) not in (1, len(sbank.maps)):
        raise ValueError('Background query has one view; lattice query shares four origins')
    probabilities, gates, infos = [], [], []
    query_valid = []
    for j, (support, valid) in enumerate(zip(sbank.maps, sbank.valid)):
        qi = 0 if len(qbank.maps) == 1 else j
        margin, gate, info = density_evidence(support[0], gold_grid, valid[0, 0], qbank.maps[qi][0])
        # FP64 posterior conversion delays saturation; explicit finite clipping
        # of ensemble logit is recorded, never called source-identical math.
        probability=margin.double().sigmoid()
        info['sigmoid_saturated_tokens']=int(((probability==0)|(probability==1)).sum())
        probabilities.append(probability[None, None])
        gates.append(gate.float()[None, None])
        query_valid.append(qbank.valid[qi])
        infos.append(info)
    from dataclasses import replace
    # Combination validity is QUERY geometric coverage; support invalidity was
    # already removed from both anchor classes in density_evidence.
    combine_bank = replace(sbank, valid=tuple(query_valid))
    probability, evidence = combine_view_outputs(tuple(probabilities), combine_bank, weights)
    epsilon=torch.finfo(probability.dtype).eps
    clipped=(probability<epsilon)|(probability>1-epsilon)
    margin=torch.logit(probability.clamp(epsilon,1-epsilon)).float()
    gate, _ = combine_view_outputs(tuple(gates), combine_bank, weights)
    prediction = decode_margin(margin[0, 0], (gate[0, 0] >= .5) & evidence[0, 0],
                               native_query, rgb_grid, fraction, (1024, 1024))
    return prediction, dict(views=infos,query_valid_fractions=[float(v.float().mean()) for v in query_valid],
                           no_evidence_pixels=int((~evidence).sum()),gate_patches=int(((gate>=.5)&evidence).sum()),
                           ensemble_logit_clipped_tokens=int((clipped&evidence).sum()),
                           combination='sigmoid KDE margin probability mean -> finite ensemble logit; weighted candidate majority',
                           geometry='original native final descriptor, same bilateral/finalize')


def selfcheck():
    """Real method functions, synthetic density maps; no GPU/DINO/data access."""
    from reference_views_cpu import run
    view_report = run()
    torch.manual_seed(997)
    from tics.reference_views import ViewBank
    fg = torch.zeros(4, 4, dtype=torch.bool);fg[:2] = True
    s = torch.randn(1, 8, 4, 4);q = torch.randn(1, 8, 4, 4)
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    sbank = ViewBank(('original',), (torch.zeros(1,3,64,64),), (s,), (valid,), ((0,0),), (64,64), {})
    qbank = ViewBank(('original',), sbank.rgb, (q,), (valid,), ((0,0),), (64,64), {})
    rgb = torch.rand(4, 4, 3)
    actual, info = predict_bank(sbank, qbank, fg, q[0], rgb, .5, torch.ones(1,1))
    expected, _ = frost_style_dense_readout([s[0]], [fg], q[0], q[0], rgb, .5, (1024,1024))
    assert torch.equal(actual, expected), 'Singleton must equal unchanged component anchor'
    hold = torch.zeros(1,1,4,4,dtype=torch.bool);hold[...,0,:] = True
    tf = fg[None,None] & ~hold;tb = ~fg[None,None] & ~hold
    probability = density_holdout(s, tf, tb, hold)
    assert probability.shape == tf.shape and torch.isfinite(probability).all()
    try:density_evidence(s[0], fg, torch.zeros_like(fg), q[0]);raise AssertionError('Degenerate support accepted')
    except DensityDegenerate as exc:
        fallback=insufficient_anchor_fallback({},expected,exc)
        assert all(torch.equal(value,expected) for value in fallback.values())
    from unittest.mock import patch
    with patch(__name__+'.frost_kde_ratio',return_value=torch.full((16,),float('nan'))):
        try:density_evidence(s[0],fg,valid[0,0],q[0]);raise AssertionError('Nonfinite margin accepted')
        except DensityDegenerate:raise AssertionError('Numerical fault mislabeled as insufficient anchors')
        except RuntimeError as exc:
            assert 'Nonfinite density margin' in str(exc)
    with patch(__name__+'.frost_bilateral',return_value=torch.full((16,),float('nan'))):
        try:decode_margin(torch.zeros(4,4),fg,q[0],rgb,.5,(1024,1024));raise AssertionError('Nonfinite smoother accepted')
        except RuntimeError as exc:assert 'Nonfinite bilateral density' in str(exc)
    for fault in (ValueError('wrong coordinate/shape'),RuntimeError('nonfinite')):
        try:insufficient_anchor_fallback({},expected,fault);raise AssertionError('Unexpected fault became fallback')
        except RuntimeError:pass
    print(json.dumps(dict(state='CPU_SELFCHECK_PASSED',view_cases=view_report['cases_count'],
        singleton_native_density_prediction_exact=True,real_recipe_holdout_probability=True,
        insufficient_anchor_baseline_fallback=True,nonfinite_density_error_propagated=True,
        nonfinite_bilateral_error_propagated=True,
        shape_coordinate_errors_not_fallback=True,
        cuda_initialized=torch.cuda.is_initialized(),scope='fake encoder/interface + actual dense functions; no DINO gain')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('background','lattice'), default='background')
    p.add_argument('--prepared-root', default='/root/autodl-tmp/demo9')
    p.add_argument('--prepared-manifest')
    p.add_argument('--projection-basis', help='Optional NATIVE_BASIS_FROZEN normalized-black-image receipt/tensor')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--out')
    p.add_argument('--allow-gpu', action='store_true')
    p.add_argument('--gpu-owner', help='Parent queue/job ownership evidence, recorded verbatim')
    p.add_argument('--selfcheck', '--self-check', action='store_true')
    a = p.parse_args()
    if a.selfcheck:selfcheck();return
    if not a.allow_gpu or not a.gpu_owner:
        p.error('GPU mode requires --allow-gpu and --gpu-owner; root must launch prepared queue')
    if not a.out or not a.prepared_manifest or not (1 <= a.limit <= 10) or not (0 <= a.fold < 4):
        p.error('Fresh --out and CPU-prepared --prepared-manifest, limit1..10, fold0..3 required')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Fresh output required; no overwrite or resume')
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(a.prepared_manifest).read_text())
    report = dict(state='PREPARING',args=vars(a),records=[],seed=0,
        method=a.method,scope='Bounded development experiment; no full FoRIS/FROST or all-fold gain claim',
        recipe='Final native L24/C1024 descriptors; no-APD FROST components per view; same original native geometry',
        combination='sigmoid KDE margin probability mean -> finite ensemble logit; weighted candidate majority; uniform same encoded views/calibration budget',
        precision='Original INSID3 _extract_features native BF16 source; full maps converted to FP32; no serialized feature replay',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (Path(__file__),HERE.parent/'tics/reference_views.py',HERE.parent/'tics/readout_baselines.py',
             HERE.parent/'tics/native_assets.py')},
        manifest_sha256=hashlib.sha256(Path(a.prepared_manifest).read_bytes()).hexdigest())
    def save():
        report['class_miou']=summary(report['records'])
        report['fallback_counts']={name:sum(row['armstate'].get(name)!='PREDICTED' for row in report['records'])
                                   for name in ('insid3_native','native_dense','uniform_views','fitted_views')}
        tmp=out/'report.tmp';tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(out/'report.json')
    save()
    if not torch.cuda.is_available():
        report.update(state='NO_GPU',reason='No CUDA; no CPU production fallback/model load');save();return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get('DEMO4_GPU_FRAC','.3')))
    sys.path.insert(0,str(Path(a.prepared_root)/'scripts'));import _paths
    sys.path.insert(0,_paths.DEMO4)
    from icx.common import build_model,coco_episodes,DEV
    from utils.data import load_image,load_mask,downsample_mask
    from PIL import Image
    started=time.monotonic()
    try:
        episodes,_,base=coco_episodes(a.fold,400,shot=1,seed=0)
        selected=fixed_selection(episodes,a.limit)
        frozen=[dict(e=e,c=c,support=r[0],query=q) for e,(c,q,r) in selected]
        if frozen != manifest['frozen_episodes'][:a.limit] or manifest['fold'] != a.fold:
            raise RuntimeError('Frozen standard episodes differ from prepared manifest; no model constructed')
        for asset in manifest.get('assets',[]):
            path=Path(asset['path'])
            if path.stat().st_size != asset['size']:raise RuntimeError('Prepared asset size changed')
        report['frozen_episodes']=frozen;save()
        from models.insid3 import INSID3
        # Source uses a NORMALIZED BLACK IMAGE for this native SP basis.
        # Constructor tensors must be ordinary no-grad tensors, not inference
        # tensors; reuse is process-scoped and restores the original method.
        with torch.no_grad(),reuse_native_basis(INSID3,a.projection_basis) as basis_receipt:
            model=build_model().eval()
        report['projection_basis']=basis_receipt or dict(source_input='normalized_black_image',reused=False)
        for parameter in model.parameters():parameter.requires_grad_(False)
        if len(model.encoder.m.blocks)!=24:raise RuntimeError('Requires existing DINOv3 ViT-L/24')
        runtime_sources={str(Path(inspect.getfile(obj))):Path(inspect.getfile(obj)) for obj in
            (build_model,load_image,load_mask,downsample_mask,model._extract_features.__func__,type(model.encoder))}
        report['runtime_source_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest()
                                          for name,path in runtime_sources.items()}
        norms=[t for t in model._transform.transforms if hasattr(t,'mean') and hasattr(t,'std')]
        if len(norms)!=1:raise RuntimeError('Actual preprocessing must expose exactly one mean/std Normalize')
        mean=torch.tensor(norms[0].mean,device=DEV).reshape(1,3,1,1)
        std=torch.tensor(norms[0].std,device=DEV).reshape(1,3,1,1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():
            raise RuntimeError('Source normalization has invalid mean/std')
        report['normalization']=dict(mean=norms[0].mean,std=norms[0].std,shape=[1024,1024])
        report['native_geometry']='Source original query final L24/C1024/64x64, source pair-BF16, used unchanged for all bilateral arms'
        report['state']='RUNNING';save()
        with torch.inference_mode():
            for e,(c,q,refs) in selected:
                unit_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
                si=Image.open(Path(base)/refs[0]).convert('RGB');qi=Image.open(Path(base)/q).convert('RGB')
                s=load_image(si,model._transform,DEV)[0];t=load_image(qi,model._transform,DEV)[0]
                if s.shape!=(1,3,1024,1024) or t.shape!=s.shape:raise RuntimeError('Native source preprocessing shape changed')
                sg=torch.from_numpy((np.array(Image.open(Path(_paths.COCO_ANN)/str(Path(refs[0]).with_suffix('.png'))))==c+1).copy())
                sm=load_mask(sg,1024,DEV)
                if sm.shape!=(1,1024,1024) or not ((sm==0)|(sm==1)).all():
                    raise RuntimeError('Source support mask must be binary [1,1024,1024]; do not coerce soft mask by bool')
                # Query annotation has NOT been opened. RGB inversion is for
                # deterministic interventions only; unchanged values below are
                # restored from exact original normalized source pixels.
                sr=s*std+mean;tr=t*std+mean
                counter=dict(source_pair_forwards=0,adapter_requests=0,original_map_reuses=0)
                def pair_encode(sp,tp):
                    counter['source_pair_forwards']+=1
                    raw=model._extract_features(torch.cat([sp,tp]).unsqueeze(0)).float()[0]
                    if raw.shape!=(2,1024,64,64):raise RuntimeError('Unexpected source final native descriptor shape')
                    if not torch.isfinite(raw).all():raise RuntimeError('Source encoder returned nonfinite native descriptor')
                    return raw
                native=pair_encode(s,t)
                def support_encoder(rgb):
                    counter['adapter_requests']+=1
                    if torch.equal(rgb,sr):counter['original_map_reuses']+=1;return native[:1]
                    normalized=torch.where(rgb==sr,s,(rgb-mean)/std)
                    return pair_encode(normalized,t)[:1]
                def query_encoder(rgb):
                    counter['adapter_requests']+=1
                    if torch.equal(rgb,tr):counter['original_map_reuses']+=1;return native[1:]
                    normalized=torch.where(rgb==tr,t,(rgb-mean)/std)
                    return pair_encode(s,normalized)[1:]
                from tics.reference_views import ViewBank
                allvalid=torch.ones(1,1,64,64,dtype=torch.bool,device=DEV)
                single_s=ViewBank(('original',),(sr,),(native[:1],),(allvalid,),((0,0),),(1024,1024),{})
                single_q=ViewBank(('original',),(tr,),(native[1:],),(allvalid,),((0,0),),(1024,1024),{})
                gold=downsample_mask(sm.unsqueeze(1),64,64).reshape(64,64).bool() if sm.any() else torch.zeros(64,64,dtype=torch.bool,device=DEV)
                rgb_grid=F.interpolate(tr,size=(64,64),mode='bilinear',align_corners=False)[0].permute(1,2,0)
                fraction=float(sg.float().mean());predictions={};infos={};fallback=None;view_weights=None;fallback_arms=set()
                host=model.predict_mask(s,sm,t).reshape(1024,1024).bool() if sm.any() else torch.zeros(1024,1024,dtype=torch.bool,device=DEV)
                predictions['insid3_native']=host
                try:
                    if min(int(gold.sum()),int((~gold).sum()))<2:
                        raise DensityDegenerate('Native support requires >=2 valid anchors of each label')
                    baseline,baseinfo=frost_style_dense_readout([native[0]],[gold],native[1],native[1],
                        rgb_grid,fraction,(1024,1024))
                    predictions['native_dense']=baseline
                    if baseinfo['ell_smooth'] is not None and not torch.isfinite(baseinfo['ell_smooth']).all():
                        raise RuntimeError('Unchanged native dense anchor has nonfinite smoothed density')
                    infos['native_dense']=dict(sigma=baseinfo['sigma'],loo_margin=baseinfo['loo_margin'],
                        candidate_patches=int(baseinfo['candidate_grid'].sum()),scope='unchanged no-APD component anchor')
                    if a.method=='background':
                        sbank=support_background_views(sr,sm[:,None].bool(),support_encoder)
                        qbank=single_q
                        factory=background_holdout_factory(sr,sm[:,None].bool(),support_encoder)
                    else:
                        sbank=patch_lattice_views(sr,support_encoder);qbank=patch_lattice_views(tr,query_encoder);factory=None
                    view_weights=fit_support_view_weights(sbank,gold[None,None],density_holdout,
                        region_view_factory=factory,min_train_anchors=2)
                    for name,weight in [('uniform_views',view_weights.uniform_weights),('fitted_views',view_weights.weights)]:
                        prediction,info=predict_bank(sbank,qbank,gold,native[1],rgb_grid,fraction,weight)
                        predictions[name]=prediction;infos[name]=info
                except DensityDegenerate as exc:
                    # Prespecified degeneracy fallback only; do not mask OOM,
                    # source feature mismatch or an unexpected programming error.
                    fallback=repr(exc)
                    fallback_arms={'uniform_views','fitted_views'}
                    if 'native_dense' not in predictions:fallback_arms.add('native_dense')
                    predictions=insufficient_anchor_fallback(predictions,host,exc)
                # Freeze ALL method outputs before evaluator accesses query GT.
                predictions={name:value.detach().clone() for name,value in predictions.items()}
                qgt=torch.from_numpy((np.array(Image.open(Path(_paths.COCO_ANN)/str(Path(q).with_suffix('.png'))))==c+1).copy())
                truth=load_mask(qgt,1024,DEV)[0].bool()
                row=dict(e=e,c=c,support=refs[0],query=q,allrole_photo_ids=[refs[0],q],
                    iu={},original_iu={},pixels={},reader_info=infos,
                    armstate={name:('BASELINE_FALLBACK' if name in fallback_arms else 'PREDICTED') for name in predictions},
                    fallback=fallback,source_context='Each altered RGB map replaces its role in native ordered B2(S,Q)',
                    cost=dict(counter,host_prediction_additional_pair_forwards=1 if sm.any() else 0,
                              elapsed_s=time.monotonic()-unit_start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved()),
                    weights=None if view_weights is None else dict(weights=view_weights.weights.cpu().tolist(),
                        uniform=view_weights.uniform_weights.cpu().tolist(),valid=view_weights.valid.cpu().tolist(),
                        eligible=view_weights.eligible_regions.cpu().tolist(),reasons=view_weights.reasons,
                        metadata=view_weights.metadata),query_gt_opened_after_prediction=True)
                if not fallback and view_weights is not None and not bool(view_weights.valid.all()):
                    row['armstate']['fitted_views']='UNIFORM_WEIGHT_FALLBACK'
                for name,prediction in predictions.items():
                    row['iu'][name]=[int((prediction&truth).sum()),int((prediction|truth).sum())]
                    resized=F.interpolate(prediction[None,None].float(),size=tuple(qgt.shape),mode='bilinear',align_corners=False)[0,0]>.5
                    g=qgt.to(DEV);row['original_iu'][name]=[int((resized&g).sum()),int((resized|g).sum())]
                    row['pixels'][name]=pixel_ledger(predictions['native_dense'],prediction,truth)
                report['records'].append(row);report['elapsed_s']=time.monotonic()-started;save()
                print(json.dumps(dict(count=len(report['records']),e=e,c=c,scores=report['class_miou'],fallback=fallback,cost=row['cost'])),flush=True)
                del native,predictions,single_s,single_q,view_weights,s,t,sr,tr,host,sm,gold
                if 'sbank' in locals():del sbank
                if 'qbank' in locals():del qbank
        report['state']='COMPLETED';save()
    except BaseException as exc:
        report.update(state='ERROR',error=repr(exc),elapsed_s=time.monotonic()-started);save();raise


if __name__=='__main__':main()
