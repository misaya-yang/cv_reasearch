"""One frozen pixel-information probe; no segmentation candidate or encoder.

Freeze -> infer all400 with GT-file guards -> seal -> score. The 12-channel
descriptor and 13-parameter reference logistic readout are fixed, with a
same-capacity16-pixel-block-constant descriptor control. No query label is
available to fitting, field inference, model selection or preprocessing.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
import builtins
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.special import expit
from scipy.stats import rankdata
import torch
import torch.nn.functional as F
torch.set_num_threads(2)

REPO = next(p for p in Path(__file__).resolve().parents if (p/'src/ics').is_dir())
DATA = REPO.parent/'cv_data'
HERE = REPO/'evidence/local/autonomous_20261010/physical_pixel_evidence'
ARMS = ('pixel', 'patch_constant')
DESCRIPTOR_NAMES = [f'{kind}_{channel}' for kind in ('rgb', 'contrast', 'gradient', 'texture') for channel in ('r','g','b')]
SIDE, PATCH, TILE, GUARD, ROLE_SAMPLES, LAMBDA = 1024, 16, 64, 8, 4096, .01


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(json.dumps([list(x.shape), x.dtype.str]).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


@contextmanager
def deny_files(paths):
    denied = {str(Path(p).resolve()) for p in paths}
    state = {'attempts': 0}
    originals = (builtins.open, io.open)
    def wrapper(original):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and str(Path(os.fsdecode(file)).resolve()) in denied:
                state['attempts'] += 1
                raise PermissionError('Query-label or baseline access during field inference')
            return original(file, *args, **kwargs)
        return guarded
    builtins.open, io.open = [wrapper(v) for v in originals]
    try:
        yield state
    finally:
        builtins.open, io.open = originals


def rgb(row, role):
    with Image.open(DATA/row[role+'_path']) as image:
        value = np.asarray(image.convert('RGB')).copy()
    crop = row.get(role+'_crop')
    if crop is not None:
        x0, y0, x1, y1 = crop
        value = value[y0:y1, x0:x1]
    assert array_hash(value) == row[role+'_rgb_hash']
    assert list(value.shape[:2]) == row[role+'_size_hw']
    # Exactly torchvision's released PIL Resize((1024,1024)), bilinear.
    return np.asarray(Image.fromarray(value).resize((SIDE, SIDE), Image.Resampling.BILINEAR)).copy()


def mask(row, role, hw=(SIDE,SIDE)):
    with Image.open(row[role+'_mask_path']) as image:
        value = (np.asarray(image.convert('L')) > 0).astype(np.uint8)
    assert array_hash(value) == row[role+'_mask_hash']
    assert list(value.shape) == row[role+'_mask_size_hw']
    return F.interpolate(torch.from_numpy(value)[None,None].float(), size=hw,
                         mode='nearest')[0,0].numpy() > .5


def descriptor(image):
    """12 fixed channels. No learned/upstream feature network or image labels."""
    value = image.astype(np.float32)/255
    mean = value.mean((0,1), dtype=np.float64).astype(np.float32)
    std = np.maximum(value.std((0,1), dtype=np.float64).astype(np.float32), 1/255)
    z = (value-mean)/std
    smooth = gaussian_filter(z, (2,2,0), mode='reflect', truncate=3.)
    contrast = z-smooth
    dx = gaussian_filter(z, (1,1,0), order=(0,1,0), mode='reflect', truncate=3.)
    dy = gaussian_filter(z, (1,1,0), order=(1,0,0), mode='reflect', truncate=3.)
    gradient = np.sqrt(dx*dx+dy*dy)
    texture = np.sqrt(np.maximum(gaussian_filter(contrast*contrast, (2,2,0),
                                               mode='reflect', truncate=3.), 0))
    out = np.concatenate((z, contrast, gradient, texture), axis=2).astype(np.float32)
    assert out.shape == (SIDE,SIDE,12) and np.isfinite(out).all()
    return out


def pool(value):
    h,w = value.shape[:2]
    assert h % PATCH == w % PATCH == 0
    return value.reshape(h//PATCH,PATCH,w//PATCH,PATCH,*value.shape[2:]).mean((1,3), dtype=np.float64).astype(np.float32)


def expand(value):
    return value.repeat(PATCH, axis=0).repeat(PATCH, axis=1)


def source_split():
    y,x = np.indices((SIDE,SIDE))
    interior = ((x%TILE >= GUARD)&(x%TILE < TILE-GUARD)&
                (y%TILE >= GUARD)&(y%TILE < TILE-GUARD))
    train = interior&(((x//TILE+y//TILE)%2)==0)
    holdout = interior&(((x//TILE+y//TILE)%2)==1)
    assert not (train&holdout).any()
    return train,holdout


def source_occurrences(label, train):
    ids = []
    counts = []
    for role in (True,False):
        eligible = np.flatnonzero((label == role)&train)
        counts.append(len(eligible))
        if not len(eligible):
            return None,counts
        k = np.floor((np.arange(ROLE_SAMPLES,dtype=np.float64)+.5)*len(eligible)/ROLE_SAMPLES).astype(int)
        ids.append(eligible[k])
    return np.concatenate(ids),counts


def fit(features, label):
    """Convex balanced logistic loss;13 params, common .01 slope ridge."""
    x = np.column_stack((features.astype(np.float64),np.ones(len(features))))
    y = label.astype(np.float64)
    assert len(y) == 2*ROLE_SAMPLES and y.sum() == ROLE_SAMPLES
    beta = np.zeros(x.shape[1],np.float64)
    ridge = np.full(x.shape[1],LAMBDA); ridge[-1] = 0
    def objective(v):
        score = x@v
        return float(np.mean(np.logaddexp(0,score)-y*score)+.5*np.sum(ridge*v*v))
    for step in range(40):
        score = x@beta
        p = expit(score)
        gradient = x.T@(p-y)/len(y)+ridge*beta
        if np.max(np.abs(gradient)) < 1e-8:
            break
        hessian = (x.T*(p*(1-p)))@x/len(y)+np.diag(ridge)
        direction = np.linalg.solve(hessian,gradient)
        before = objective(beta)
        scale = 1.
        for _ in range(30):
            proposal = beta-scale*direction
            if objective(proposal) <= before:
                beta = proposal
                break
            scale *= .5
        else:
            raise RuntimeError('Convex readout line search failed')
    score = x@beta
    gradient = x.T@(expit(score)-y)/len(y)+ridge*beta
    assert np.max(np.abs(gradient)) < 1e-7, 'Readout did not converge'
    return beta,dict(iterations=step+1,objective=objective(beta),gradient_max_abs=float(np.max(np.abs(gradient))),
                     unique_source_occurrences=int(len(np.unique(features,axis=0))),parameters=13)


def rank_metrics(score, truth, roi=None, score_is_logit=True):
    keep = np.ones(truth.shape,bool) if roi is None else roi
    y = truth[keep]
    s = score[keep].astype(np.float64)
    n1,n0 = int(y.sum()),int((~y).sum())
    if not n1 or not n0:
        return dict(valid=False,n_fg=n1,n_bg=n0)
    rank = rankdata(s,method='average')
    auc = float((rank[y].sum()-n1*(n1+1)/2)/(n1*n0))
    out = dict(valid=True,n_fg=n1,n_bg=n0,auc=auc)
    if score_is_logit:
        losses = np.logaddexp(0,s)-y*s
        out['balanced_nll'] = float(.5*(losses[y].mean()+losses[~y].mean()))
    return out


def patch_rows(value):
    h,w=value.shape
    return value.reshape(h//PATCH,PATCH,w//PATCH,PATCH).transpose(0,2,1,3).reshape(-1,PATCH*PATCH)


def within_patch(score, truth, roi=None):
    valid = np.ones(truth.shape,bool) if roi is None else roi
    y = patch_rows(truth&valid)
    v = patch_rows(valid)
    s = np.where(v,patch_rows(score),np.inf)
    f = y.sum(1).astype(np.int64)
    b = v.sum(1)-f
    pairs = f*b
    if not pairs.sum():
        return dict(valid=False,mixed_patches=0,pairs=0)
    ranks = rankdata(s,method='average',axis=1)
    concordance = (ranks*y).sum(1)-f*(f+1)/2
    concordance = np.maximum(concordance,0)
    auc = float(concordance.sum()/pairs.sum())
    assert -.000000001 <= auc <= 1.000000001
    return dict(valid=True,mixed_patches=int((pairs>0).sum()),pairs=int(pairs.sum()),
                concordance=float(concordance.sum()),auc=auc)


def constant_decoder_ceiling(truth, grid):
    """Exact restricted block-constant IoU oracle; never an inference rule."""
    h,w = truth.shape
    yy = np.floor(np.arange(h)*grid/h).astype(np.int64)
    xx = np.floor(np.arange(w)*grid/w).astype(np.int64)
    ids = (yy[:,None]*grid+xx[None,:]).reshape(-1)
    area = np.bincount(ids,minlength=grid*grid).astype(np.int64)
    fg = np.bincount(ids,weights=truth.reshape(-1),minlength=grid*grid).astype(np.int64)
    assert fg.sum() == truth.sum() and area.sum() == truth.size
    valid = area > 0
    coverage = fg[valid]/area[valid]
    ordered = np.argsort(-coverage,kind='stable')
    f = fg[valid][ordered]
    false_positive = (area[valid]-fg[valid])[ordered]
    intersections = np.cumsum(f)
    unions = int(truth.sum())+np.cumsum(false_positive)
    iou = intersections/np.maximum(unions,1)
    k = int(np.argmax(iou)) if len(iou) else 0
    mixed = (fg>0)&(fg<area)
    return dict(grid_hw=[grid,grid],frame_hw=[h,w],block_constant_gt_oracle_iou=float(iou[k]) if len(iou) else 0.,
                intersection=int(intersections[k]) if len(iou) else 0,union=int(unions[k]) if len(iou) else int(truth.sum()),
                selected_nonempty_blocks=k+1 if len(iou) else 0,foreground_pixels=int(truth.sum()),
                mixed_blocks=int(mixed.sum()),fg_mass_in_mixed_blocks=int(fg[mixed].sum()),
                fg_mass_in_mixed_fraction=float(fg[mixed].sum()/max(truth.sum(),1)))


def tests():
    # Exhaustive all2x2 masks and16 block subsets certify the oracle formula,
    # with nonuniform block areas as well as the tie-heavy binary cases.
    checked = 0
    for bits in itertools.product((False,True),repeat=4):
        t=np.array(bits,bool).reshape(2,2)
        got=constant_decoder_ceiling(t,2)['block_constant_gt_oracle_iou']
        expected=max((np.logical_and(t,np.array(p,bool).reshape(2,2)).sum()/max(np.logical_or(t,np.array(p,bool).reshape(2,2)).sum(),1)
                      for p in itertools.product((False,True),repeat=4)))
        assert got == expected
        checked+=1
    t=np.array([[1,0,0],[0,1,1],[0,0,1]],bool)
    ids=(np.floor(np.arange(3)*2/3).astype(int)[:,None]*2+np.floor(np.arange(3)*2/3).astype(int)[None,:])
    expected=max((np.logical_and(t,np.array(p,bool)[ids]).sum()/max(np.logical_or(t,np.array(p,bool)[ids]).sum(),1)
                  for p in itertools.product((False,True),repeat=4)))
    assert constant_decoder_ceiling(t,2)['block_constant_gt_oracle_iou'] == expected
    rng=np.random.default_rng(7)
    truth=rng.random((16,16)) < .4
    value=rng.integers(0,8,size=(16,16)).astype(np.float32)
    roi=rng.random((16,16)) < .7
    for keep in (np.ones((16,16),bool),roi):
        positive=value[truth&keep];negative=value[(~truth)&keep]
        brute=((positive[:,None]>negative[None,:])+.5*(positive[:,None]==negative[None,:])).mean()
        assert abs(within_patch(value,truth,keep)['auc']-brute)<1e-14
        assert abs(rank_metrics(value,truth,keep)['auc']-brute)<1e-14
        assert within_patch(np.ones_like(value),truth,keep)['auc'] == .5
    # Meaningful matched capacity identity: block-constant source and query
    # descriptors make both fits/outputs byte identical under same samples.
    synthetic=rng.normal(size=(64,64,12)).astype(np.float32)
    synthetic=expand(synthetic)
    label=rng.random((SIDE,SIDE)) < .4
    train,_=source_split()
    occurrence,_=source_occurrences(label,train)
    a,_=fit(synthetic.reshape(-1,12)[occurrence],label.reshape(-1)[occurrence])
    b,_=fit(expand(pool(synthetic)).reshape(-1,12)[occurrence],label.reshape(-1)[occurrence])
    assert np.array_equal(a,b)
    from torchvision.transforms import Resize
    for hw in ((93,157),(1024,1024),(2048,2048)):
        fixture=Image.fromarray(rng.integers(0,256,size=(*hw,3),dtype=np.uint8))
        assert np.array_equal(np.asarray(Resize((SIDE,SIDE))(fixture)),
                              np.asarray(fixture.resize((SIDE,SIDE),Image.Resampling.BILINEAR)))
    return dict(exact_oracle_binary_fixtures=checked,nonuniform_block_fixture=True,
                exact_roc_and_patch_tie_fixtures=True,same_capacity_identity=True,
                official_PIL_resize_parity=True,query_GT_reads=0)


def freeze():
    assert not (HERE/'FROZEN.json').exists(), 'Never replace frozen protocol'
    pilot=DATA/'a/joint_role_pilot200_20261010'
    fundus=DATA/'a/fundus_foris_seed0_20261010'
    rows=[];parents={};sources=read(pilot/'source_index.json')
    for root in (pilot,fundus):
        manifest=read(root/'manifest.json')
        inference=lines(root/'inference.jsonl') if (root/'inference.jsonl').exists() else read(root/'inference.json')
        seal=read(root/'sealed.json')
        assert seal['state']=='ALL_PREDICTIONS_SEALED' and seal['n']==len(manifest)==200
        assert sha(root/'manifest.json') == seal['manifest_sha256']
        fname='inference.jsonl' if (root/'inference.jsonl').exists() else 'inference.json'
        assert sha(root/fname)==seal['inference_sha256']
        parents[str(root)]={name:sha(root/name) for name in ('manifest.json',fname,'sealed.json','config.json')}
        for i,(row,rec) in enumerate(zip(manifest,inference)):
            assert row['episode_id']==rec['episode_id']
            baseline=root/'predictions'/rec['filename']
            assert sha(baseline)==rec['prediction_sha256']
            item=dict(index=len(rows),episode=row,baseline_path=str(baseline),baseline_sha256=sha(baseline),
                      baseline_cli_key=('cli1024/foris.crf' if root==pilot else 'cli/foris.crf'))
            if root==fundus:
                field=root/'fields'/rec['filename']
                item.update(foris_field_path=str(field),foris_field_sha256=sha(field),foris_field_key='score')
            else:
                old=sources[row['episode_id']]
                if 'source_field_path' in old:
                    field=Path(old['source_field_path'])
                    assert sha(field)==old['source_field_sha256']
                else:
                    field=Path(old['mean_path']).parent.parent/'fields'/Path(old['mean_path']).name
                assert field.exists()
                item.update(foris_field_path=str(field),foris_field_sha256=sha(field),foris_field_key='score')
            rows.append(item)
    assert len(rows)==400
    protocol=dict(state='ANALYSIS_DESIGN_FROZEN_BEFORE_QUERY_GT',created_utc=datetime.now(timezone.utc).isoformat(),n=400,
        parents=parents,order='sealed Deep100,PACO100,then released Fundusseed0all200; retain repeated photos and episodes',
        canonical_rgb='official PIL bilinear1024-square, exact lawful parent crop; verified preprocessor parity',
        mask='torch nearest raw legal R mask directly to1024; query mask denied until all400 scores sealed',
        descriptor=dict(channels=DESCRIPTOR_NAMES,rgb='per-image channel mean/std, floor1/255; unsupervised R/Q separately',
            contrast='z-Gaussian(z,sigma2,reflect,truncate3)',gradient='magnitude Gaussian first derivative sigma1',
            texture='sqrt Gaussian(contrast squared,sigma2)',scales_fixed_not_swept=True),
        source_split=dict(tile=TILE,guard=GUARD,train='even64-square checkerboard tile interiors',holdout='odd tile interiors',
            disclosure='same-image spatial holdout correlated; max local descriptor support12pixels, train/holdout gap17pixels; global unsupervised normalization shared'),
        fitting=dict(capacity=13,role_samples_each=ROLE_SAMPLES,occurrences='fixed row-major physical-area midpoint quantiles',
            loss='equalFG/BG logistic; intercept free, slope L2lambda.01',
            normalization='same reference training-region unsupervised real12-channel mean/std floor.01 for both arms',
            optimizer='FP64 convex Newton tol1e-8, cap40, exact Hessian with backtracking',
            no_source_role='skip both appearance arms if either role absent in training tiles, still compute geometricceilings'),
        matched_control='average same12 actual descriptors inside aligned16canonicalpixelblocks, same sample IDs/labels/loss/lambda/scales/parameter count; apply same coarse transform to query',
        score=dict(primary='within16patchFG/BG pair concordance on query; fixedconstantcontrolallties.5',
            transfer_control='same withinpatch and fullimage AUC/NLL on spatial reference holdout',
            rois=['whole','saved complete FoRISFG','saved complete FoRISBG'],
            foris_auc='bilinear64scoreto1024 from sealed exact whole-source field; diagnostic only, common physicalROI',
            geometric_oracle='exact maxIoU over64x64 and128x128 block-constant binary decoders, CLI1024 and originalframes; GTdiagnostic NEVERachievable construction norO24informationbound',
            no_parameter_selection=True,no_threshold_selection=True,no_segmentation_candidate=True),
        limits=['no new DINO input/encoder/cache writes','appearance readout is classical; not an original contribution',
            'withinpatch signal can be boundary detail only; fullquery and FoRISFG AUC assess taskidentity transfer',
            'nearestblock decoder ceiling excludes bilinear or learned intra-block patterns and says nothing about O24 global receptivefield',
            'all Deep/PACO already exposed; Fundus newly audited but same released draw/pool repeated'],
        CPU_threads=2,workers=1,packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','Pillow')})
    write(HERE/'protocol.json',protocol);write(HERE/'tasks.json',rows)
    write(HERE/'synthetic_validation.json',tests())
    (HERE/'frozen').mkdir(exist_ok=True)
    shutil.copy2(__file__,HERE/'frozen/diagnose.py')
    for name in ('protocol.json','tasks.json'):
        shutil.copy2(HERE/name,HERE/'frozen'/name)
    receipt=dict(state='FROZEN_BEFORE_ANY_QUERY_GT_ANALYSIS',source_sha256=sha(__file__),
                 protocol_sha256=sha(HERE/'protocol.json'),tasks_sha256=sha(HERE/'tasks.json'),
                 synthetic_validation_sha256=sha(HERE/'synthetic_validation.json'),query_GT_reads=0)
    write(HERE/'FROZEN.json',receipt)
    print(json.dumps(receipt),flush=True)


def verify_frozen():
    frozen=read(HERE/'FROZEN.json')
    assert sha(__file__)==frozen['source_sha256']
    assert sha(HERE/'protocol.json')==frozen['protocol_sha256']
    assert sha(HERE/'tasks.json')==frozen['tasks_sha256']
    return frozen,read(HERE/'tasks.json')


def infer():
    frozen,tasks=verify_frozen()
    assert not (HERE/'sealed.json').exists()
    (HERE/'fields').mkdir(exist_ok=True);(HERE/'records').mkdir(exist_ok=True)
    train,holdout=source_split();started=time.monotonic();records=[]
    # Guard every prepared query annotation and every baseline, not just own row.
    denied=[]
    for task in tasks:
        row=task['episode']
        denied.extend([row['query_mask_path'],task['baseline_path'],task['foris_field_path']])
        for rel in row.get('source_files',{}):
            if ('Ground truth' in rel and rel!=row.get('reference_annotation_path')) or rel==row.get('query_annotation_path'):
                denied.append(DATA/rel)
    with deny_files(denied) as guard:
        # Literal query annotations may also occur as another episode's legal
        # reference annotation. We open only frozen reference_mask_path, which
        # is in the distinct reference subtree and is an authorized source.
        for task in tasks:
            i,row=task['index'],task['episode'];recpath=HERE/'records'/f'{i:06d}.json'
            if recpath.exists():
                rec=read(recpath);assert sha(HERE/'fields'/rec['filename'])==rec['field_sha256']
                records.append(rec);continue
            tick=time.monotonic()
            rimage,qimage=rgb(row,'reference'),rgb(row,'query')
            label=mask(row,'reference');occurrence,counts=source_occurrences(label,train)
            rec=dict(index=i,episode_id=row['episode_id'],dataset=row['dataset'],source_role_pixels=counts,
                     query_GT_reads=0,encoder_calls=0,raw_cache_writes=0,canonical_R_rgb_hash=array_hash(rimage),canonical_Q_rgb_hash=array_hash(qimage))
            filename=f'{i:06d}.npz'
            if occurrence is None:
                np.savez_compressed(HERE/'fields'/filename,valid=np.array(False))
                rec.update(valid=False,reason='reference training region absent role')
            else:
                rf,qf=descriptor(rimage),descriptor(qimage)
                # One common label-free source normalization for matched fits.
                mean=rf[train].mean(0,dtype=np.float64)
                std=np.maximum(rf[train].std(0,dtype=np.float64),.01)
                rc,qc=pool(rf),pool(qf)
                coords=np.unravel_index(occurrence,label.shape)
                actual_samples=rf[coords].astype(np.float64)
                control_samples=rc[coords[0]//PATCH,coords[1]//PATCH].astype(np.float64)
                y=label.reshape(-1)[occurrence]
                betas={};diagnostics={};arrays={'valid':np.array(True)}
                for arm,samples in zip(ARMS,(actual_samples,control_samples)):
                    beta,diagnostics[arm]=fit((samples-mean)/std,y)
                    betas[arm]=beta.tolist()
                    slope=beta[:-1]/std;intercept=beta[-1]-mean@slope
                    if arm=='pixel':
                        arrays['reference_pixel']=(rf@slope+intercept).astype(np.float32)
                        arrays['query_pixel']=(qf@slope+intercept).astype(np.float32)
                    else:
                        arrays['reference_patch_constant64']=(rc@slope+intercept).astype(np.float32)
                        arrays['query_patch_constant64']=(qc@slope+intercept).astype(np.float32)
                np.savez_compressed(HERE/'fields'/filename,**arrays)
                rec.update(valid=True,common_sample_ids_sha256=array_hash(occurrence),common_labels_sha256=array_hash(y),
                           source_normalization_mean=mean.tolist(),source_normalization_std=std.tolist(),
                           betas=betas,fit=diagnostics,
                           descriptor_tensors_not_saved=True,stored_real_pixel_scores_sha256={k:array_hash(v) for k,v in arrays.items()})
            rec.update(filename=filename,field_sha256=sha(HERE/'fields'/filename),seconds=time.monotonic()-tick)
            write(recpath,rec);records.append(rec)
            if (i+1)%25==0:
                print(json.dumps(dict(stage='infer',completed=i+1,n=400,seconds=time.monotonic()-started,query_GT_reads=0)),flush=True)
        assert guard['attempts']==0
    write(HERE/'inference.json',records)
    seal=dict(state='ALL400_APPEARANCE_FIELDS_SEALED_BEFORE_QUERY_GT',n=len(records),
              inference_sha256=sha(HERE/'inference.json'),FROZEN_sha256=sha(HERE/'FROZEN.json'),
              field_sha256s=[r['field_sha256'] for r in records],query_GT_reads=0,query_label_attempts=0,
              encoder_calls=0,raw_cache_writes=0,wall_seconds=time.monotonic()-started,CPU_threads=2,workers=1)
    write(HERE/'sealed.json',seal);print(json.dumps({k:v for k,v in seal.items() if k!='field_sha256s'}),flush=True)


def score():
    _,tasks=verify_frozen();seal=read(HERE/'sealed.json');records=read(HERE/'inference.json')
    assert seal['state']=='ALL400_APPEARANCE_FIELDS_SEALED_BEFORE_QUERY_GT' and seal['n']==len(records)==400
    assert sha(HERE/'inference.json')==seal['inference_sha256']
    train,holdout=source_split();output=[];started=time.monotonic()
    for task,rec in zip(tasks,records):
        row=task['episode'];i=task['index']
        assert row['episode_id']==rec['episode_id']
        path=HERE/'fields'/rec['filename'];assert sha(path)==rec['field_sha256']==seal['field_sha256s'][i]
        truth=mask(row,'query');rtruth=mask(row,'reference')
        item=dict(index=i,episode_id=row['episode_id'],dataset=row['dataset'],valid=rec['valid'],
                  reference_photo_id=row['reference_photo_id'],query_photo_id=row['query_photo_id'],
                  geometry=dict(cli={str(grid):constant_decoder_ceiling(truth,grid) for grid in (64,128)}))
        original=mask(row,'query',tuple(row['query_size_hw']))
        item['geometry']['original']={str(grid):constant_decoder_ceiling(original,grid) for grid in (64,128)}
        assert sha(task['baseline_path'])==task['baseline_sha256']
        with np.load(task['baseline_path'],allow_pickle=False) as old:
            base=np.unpackbits(old[task['baseline_cli_key']],count=SIDE*SIDE).reshape(SIDE,SIDE).astype(bool)
        assert sha(task['foris_field_path'])==task['foris_field_sha256']
        with np.load(task['foris_field_path'],allow_pickle=False) as old:
            foris=old[task['foris_field_key']]
            foris=F.interpolate(torch.from_numpy(foris.astype(np.float64)).reshape(1,1,64,64),
                                (SIDE,SIDE),mode='bilinear',align_corners=False)[0,0].numpy()
        rois=dict(whole=np.ones(truth.shape,bool),foris_fg=base,foris_bg=~base)
        item['foris']={name:rank_metrics(foris,truth,roi,score_is_logit=False) for name,roi in rois.items()}
        if rec['valid']:
            with np.load(path,allow_pickle=False) as fields:
                item['source']={};item['query']={}
                for arm in ARMS:
                    if arm=='pixel':
                        rscore=fields['reference_pixel'];qscore=fields['query_pixel']
                    else:
                        rscore=expand(fields['reference_patch_constant64']);qscore=expand(fields['query_patch_constant64'])
                    item['source'][arm]=dict(global_rank=rank_metrics(rscore,rtruth,holdout),
                                             within_patch=within_patch(rscore,rtruth,holdout))
                    item['query'][arm]={name:dict(global_rank=rank_metrics(qscore,truth,roi),
                                                 within_patch=within_patch(qscore,truth,roi)) for name,roi in rois.items()}
                    if arm=='patch_constant':
                        for metrics in [item['source'][arm]]+list(item['query'][arm].values()):
                            m=metrics['within_patch']
                            if m['valid']:assert m['auc']==.5
        output.append(item)
        if (i+1)%25==0:
            print(json.dumps(dict(stage='score',completed=i+1,n=400,seconds=time.monotonic()-started)),flush=True)
    write(HERE/'episodes.json',output)
    datasets={}
    def summary(values):
        valid=[v for v in values if v['valid']]
        if not valid:return dict(n=0)
        out=dict(n=len(valid),macro_auc=float(np.mean([v['auc'] for v in valid])))
        if 'pairs' in valid[0]:
            pairs=sum(v['pairs'] for v in valid);c=sum(v['concordance'] for v in valid)
            out.update(pooled_within_patch_auc=c/pairs,pairs=pairs)
        elif 'balanced_nll' in valid[0]:
            out['macro_balanced_nll']=float(np.mean([v['balanced_nll'] for v in valid]))
        return out
    for name in sorted({r['dataset'] for r in output}):
        rows=[r for r in output if r['dataset']==name];valid=[r for r in rows if r['valid']]
        ds=dict(n=len(rows),appearance_valid=len(valid),query_unique_photos=len({r['query_photo_id'] for r in rows}),
                reference_unique_photos=len({r['reference_photo_id'] for r in rows}),source={},query={},foris={},geometry={})
        for arm in ARMS:
            ds['source'][arm]={m:summary([r['source'][arm][m] for r in valid]) for m in ('global_rank','within_patch')}
            ds['query'][arm]={roi:{m:summary([r['query'][arm][roi][m] for r in valid]) for m in ('global_rank','within_patch')}
                              for roi in ('whole','foris_fg','foris_bg')}
        for roi in ('whole','foris_fg','foris_bg'):
            # FoRIS ranking paired to cases on which source probe had bothroles.
            ds['foris'][roi]=summary([r['foris'][roi] for r in valid])
        for frame in ('cli','original'):
            ds['geometry'][frame]={}
            for grid in ('64','128'):
                g=[r['geometry'][frame][grid] for r in rows]
                ceilings=[r['block_constant_gt_oracle_iou'] for r in g]
                ds['geometry'][frame][grid]=dict(n=len(g),mean_per_episode_ceiling=float(np.mean(ceilings)),
                    median_per_episode_ceiling=float(np.median(ceilings)),min_per_episode_ceiling=float(np.min(ceilings)),
                    pooled_union_weighted_chosen_episode_oracle=sum(r['intersection'] for r in g)/max(sum(r['union'] for r in g),1),
                    fg_mass_in_mixed_fraction=sum(r['fg_mass_in_mixed_blocks'] for r in g)/max(sum(r['foreground_pixels'] for r in g),1))
        ds['paired_query_within_patch_pixel_minus_control']={roi:float(np.mean([r['query']['pixel'][roi]['within_patch']['auc']-.5
                                                    for r in valid if r['query']['pixel'][roi]['within_patch']['valid']]))
                                                           for roi in ('whole','foris_fg','foris_bg')}
        ds['paired_query_global_pixel_minus_control']={roi:float(np.mean([r['query']['pixel'][roi]['global_rank']['auc']-
                                                      r['query']['patch_constant'][roi]['global_rank']['auc']
                                                    for r in valid if r['query']['pixel'][roi]['global_rank']['valid']]))
                                                           for roi in ('whole','foris_fg','foris_bg')}
        datasets[name]=ds
    result=dict(state='SINGLE_FROZEN_INFORMATION_DIAGNOSIS_COMPLETE',n=400,datasets=datasets,
                sealed_sha256=sha(HERE/'sealed.json'),protocol_sha256=sha(HERE/'protocol.json'),
                episodes_sha256=sha(HERE/'episodes.json'),score_wall_seconds=time.monotonic()-started,
                no_segmentation_predictions_or_mIoU_head=True,no_parameters_changed_after_freeze=True,
                encoder_calls=0,raw_cache_writes=0,query_GT_in_field_inference=0,
                limits=read(HERE/'protocol.json')['limits'])
    write(HERE/'results.json',result)
    print(json.dumps(dict(state=result['state'],n=400,result=str(HERE/'results.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','infer','score','tests'])
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='infer':infer()
    elif args.action=='score':score()
    else:print(json.dumps(tests()),flush=True)
