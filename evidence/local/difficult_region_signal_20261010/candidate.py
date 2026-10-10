"""Reference-only calibration of the frozen ridge signal, then complete masks."""
from collections import defaultdict
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DATA = REPO.parent/'cv_data'
PILOT = DATA/'a/joint_role_pilot200_20261010'
RAW = ROOT/'raw'
OUT = ROOT/'candidate_reference_v1'
SUPPORT = RAW/'frozen/study.py'
spec = importlib.util.spec_from_file_location('frozen_signal_math', SUPPORT)
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)
np, torch, F = math.np, math.torch, math.F
read, lines, sha, write, resize = math.read, math.lines, math.sha, math.write, math.resize


def reference_threshold(score, coverage, fit_ids):
    """Choose a strict threshold using only held-out legal reference patches."""
    keep = np.ones(len(score), dtype=bool)
    keep[fit_ids] = False
    fallback = None
    if coverage[keep].sum() == 0 or (1-coverage[keep]).sum() == 0:
        keep[:] = True
        fallback = 'heldout_reference_role_empty_use_full_reference'
    values, fg = score[keep], coverage[keep]
    order = np.argsort(values, kind='stable')
    values, fg = values[order], fg[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1]
    sizes = np.diff(np.r_[starts, len(values)])
    pos = np.add.reduceat(fg, starts)
    tp, fp = np.cumsum(pos[::-1]), np.cumsum((sizes-pos)[::-1])
    iou = tp/(fg.sum()+fp)
    best = int(np.argmax(iou))
    k = len(starts)-1-best
    threshold = float(values[starts[k-1]]) if k else float(np.nextafter(values[0], -np.inf))
    selected = score[keep] > threshold
    exact_iou = float(coverage[keep][selected].sum()/(coverage[keep].sum()+(1-coverage[keep][selected]).sum()))
    assert abs(exact_iou-float(iou[best])) < 1e-12
    mfg = float(np.sum(score[keep]*coverage[keep])/coverage[keep].sum())
    mbg = float(np.sum(score[keep]*(1-coverage[keep]))/(1-coverage[keep]).sum())
    return dict(threshold=threshold, separation=mfg-mbg, foreground_mean=mfg, background_mean=mbg,
        reference_iou=exact_iou, calibration_patches=int(keep.sum()), fallback=fallback,
        scope='source-reference supervision, same-image neighboring patches; not independent query validation')


def quantile_ids(c):
    ids = []
    for w in (c, 1-c):
        cumulative = np.cumsum(w)
        ids.extend(np.searchsorted(cumulative, (np.arange(128)+.5)*cumulative[-1]/128, side='left'))
    return np.unique(ids)


def source_choice(hg, hl, baseline1024, source64):
    """One least-squares mixture of two views, fit to existing query context."""
    span = float(source64.max()-source64.min())
    norm = (source64-source64.min())/max(span, 1e-6)
    confidence = np.abs(2*resize(norm, (1024,1024))-1) if span else np.zeros((1024,1024))
    foreground = confidence*baseline1024
    background = confidence*~baseline1024
    roles = int(foreground.sum()>0)+int(background.sum()>0)
    if not roles:
        return .5, dict(reason='zero_context_confidence', denominator=0)
    if foreground.sum(): foreground /= roles*foreground.sum()
    if background.sum(): background /= roles*background.sum()
    wf, wb = math.mass128(foreground), math.mass128(background)
    g, delta = hg.ravel(), (hl-hg).ravel()
    w = wf+wb
    denominator = float(np.sum(w*delta**2))
    numerator = float(np.sum(delta*((wf-wb)-w*g)))
    alpha = float(np.clip(numerator/denominator, 0, 1)) if denominator else .5
    return alpha, dict(reason='balanced_existing_query_context_least_squares', denominator=denominator,
                       numerator=numerator, query_GT_read=False)


def generate():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'sealed.json').exists(), 'Never overwrite completed candidate'
    (OUT/'predictions').mkdir(exist_ok=True); (OUT/'records').mkdir(exist_ok=True)
    raw_seal = read(RAW/'sealed.json')
    assert sha(SUPPORT) == raw_seal['script_sha256']
    source = {r['episode_id']:r for r in lines(PILOT/'inference.jsonl')}
    diagnostic_path = DATA/'a/cache_mechanism_study_20261010/frontend_fields_episodes.jsonl'
    diagnostic = {r['episode_id']:r for r in lines(diagnostic_path)}
    cfg = read(PILOT/'config.json'); cache_folder = Path(cfg['raw_profile']).parent
    basis_path = DATA/'native_assets/positional_basis.pt'
    assert sha(basis_path) == read(RAW/'config.json')['basis_sha256']
    basis = torch.load(basis_path, map_location='cpu', weights_only=True)['basis'].float()
    projection = torch.eye(1024)-basis@basis.T
    manifest = read(PILOT/'manifest.json')
    config = dict(manifest_sha256=sha(PILOT/'manifest.json'), raw_seal_sha256=sha(RAW/'sealed.json'),
        signal_math_sha256=sha(SUPPORT), generator_sha256=sha(__file__), diagnostic_sha256=sha(diagnostic_path),
        signal='source_apd.ridge from frozen raw study; same source APD branch for all views',
        calibration='strict-threshold source IoU on reference patches outside fit subset; full-reference fallback only if a role is empty',
        bounded_evidence='clip((ridge_score-reference_threshold)/reference_class_mean_separation, -1,1); zero if nonpositive separation',
        scene_mixture='closed-form alpha in [0,1] minimizing balanced confidence-weighted squared error to existing query FoRIS labels; no query GT',
        anchor='g0 + .5*h, original full FoRIS sign and minmax source response magnitude; exact tie inherits baseline',
        controls=['unshifted zero threshold global/local', 'reference-calibrated standalone global/local',
                  'anchor global/local/fixed_mean/scene_selected'], query_GT_in_prediction=False,
        note='Exploratory development; ROI evidence motivated selection before this complete-output evaluation.')
    if (OUT/'config.json').exists(): assert read(OUT/'config.json') == config
    else:
        write(OUT/'config.json',config)
        frozen=OUT/'frozen'; frozen.mkdir(exist_ok=True)
        (frozen/'candidate.py').write_bytes(Path(__file__).read_bytes())
        (frozen/'signal_math.py').write_bytes(SUPPORT.read_bytes())
    receipts=[]
    for i, (row, raw_record) in enumerate(zip(manifest,raw_seal['receipts'])):
        assert row['episode_id']==raw_record['episode_id']
        begin=time.monotonic()
        dest=OUT/'predictions'/raw_record['filename']; record_path=OUT/'records'/f'{i:06d}.json'
        if record_path.exists():
            receipt=read(record_path); assert sha(dest)==receipt['prediction_sha256']; receipts.append(receipt); continue
        request=raw_record['raw_receipts'][0]; assert request['role']=='reference_full'
        folder=cache_folder/request['key']
        with (folder/'entry.lock').open('rb') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH)
            info=read(folder/'entry.json'); assert info['file_sha256']==request['payload_sha256']
            assert sha(folder/info['file'])==request['payload_sha256']
            with np.load(folder/info['file'],allow_pickle=False) as z: reference=z['O/24'].copy()
            assert math.array_sha(reference)==request['tensor_sha256']
            fcntl.flock(lock,fcntl.LOCK_UN)
        r=F.normalize(torch.from_numpy(reference),dim=1)
        if raw_record['apd_applied']: r=F.normalize(r@projection.T,dim=1)
        c=math.mask(row,'reference').reshape(64,16,64,16).mean(axis=(1,3)).ravel()
        fits,fit_info=math.reference_fit(r,c)
        coefficient,bias=fits['ridge']
        score_r=(r.double()@coefficient+bias).numpy()
        ids=quantile_ids(c); assert len(ids)==fit_info['fit_tokens']
        calibration=reference_threshold(score_r,c,ids)
        field_path=RAW/'fields'/raw_record['filename']; assert sha(field_path)==raw_record['sha256']
        with np.load(field_path,allow_pickle=False) as z:
            scores={v:z['source_apd.ridge.'+v].copy() for v in ('global','local4')}
        threshold,scale=calibration['threshold'],calibration['separation']
        evidence={v:np.clip((s-threshold)/scale,-1,1) if scale>np.finfo(float).eps else np.zeros_like(s) for v,s in scores.items()}
        src=source[row['episode_id']]; baseline_path=PILOT/'predictions'/src['filename']
        assert sha(baseline_path)==src['prediction_sha256']
        shape=tuple(row['query_size_hw'])
        with np.load(baseline_path,allow_pickle=False) as z:
            baseline=np.unpackbits(z['original/foris.crf'],count=np.prod(shape)).reshape(shape).astype(bool)
            baseline1024=np.unpackbits(z['cli1024/foris.crf'],count=1024**2).reshape(1024,1024).astype(bool)
        diag=diagnostic[row['episode_id']]
        assert sha(diag['source_score_path'])==diag['source_score_sha256']
        with np.load(diag['source_score_path'],allow_pickle=False) as z: source64=z['score'].reshape(64,64).astype(float)
        alpha,selection=source_choice(evidence['global'],evidence['local4'],baseline1024,source64)
        mixed={'global':evidence['global'],'local4':evidence['local4'],
               'fixed_mean':.5*(evidence['global']+evidence['local4']),
               'scene_selected':(1-alpha)*evidence['global']+alpha*evidence['local4']}
        span=float(source64.max()-source64.min())
        norm=(source64-source64.min())/max(span,1e-6)
        confidence=np.abs(2*resize(norm,shape)-1) if span else np.zeros(shape)
        g0=(2*baseline.astype(float)-1)*confidence
        zero=(g0>0)|((g0==0)&baseline); assert np.array_equal(zero,baseline)
        predictions={}
        for view,score in scores.items():
            pixels=resize(score,shape)
            predictions['ridge.zero.'+view]=np.packbits(pixels>0)
            predictions['reference_iou.'+view]=np.packbits(pixels>threshold)
        for view,h in mixed.items():
            decision=g0+.5*resize(h,shape)
            prediction=(decision>0)|((decision==0)&baseline)
            predictions['anchor.'+view]=np.packbits(prediction)
        np.savez_compressed(dest, original_hw=np.array(shape), **predictions)
        receipt=dict(episode_id=row['episode_id'],filename=raw_record['filename'],prediction_sha256=sha(dest),
            calibration=calibration,local_weight=alpha,selection=selection,raw_reference=request,
            seconds=time.monotonic()-begin,zero_anchor_matches=True)
        write(record_path,receipt);receipts.append(receipt)
        if (i+1)%40==0:print(json.dumps(dict(completed=i+1,total=200)),flush=True)
    write(OUT/'sealed.json',dict(n=len(receipts),receipts=receipts,config_sha256=sha(OUT/'config.json'),
        generator_sha256=sha(__file__),query_GT_read=False,encoder_calls=0,new_raw_writes=0))
    print(json.dumps(dict(stage='generated',n=len(receipts))),flush=True)


def aggregate(records):
    groups=defaultdict(lambda:np.zeros(2,dtype=np.int64))
    for r in records:
        for method,iu in r['iu'].items(): groups[r['dataset'],method,r['fold'],r['class_id']]+=iu
    folds=defaultdict(list)
    deep=defaultdict(lambda:np.zeros(2,dtype=np.int64))
    for (dataset,method,fold,cls),(i,u) in groups.items():
        if dataset=='deepglobe_road':deep[method]+=[i,u]
        else:folds[method,fold].append(100*i/u if u else 0)
    answer={'deepglobe_road':{method:100*i/u if u else 0 for method,(i,u) in deep.items()},'paco_part':{}}
    for method in sorted({m for m,f in folds}):answer['paco_part'][method]=float(np.mean([np.mean(v) for (m,f),v in folds.items() if m==method]))
    return answer


def score():
    seal=read(OUT/'sealed.json');assert sha(OUT/'config.json')==seal['config_sha256']
    assert sha(OUT/'frozen/candidate.py')==seal['generator_sha256']
    manifest=read(PILOT/'manifest.json')
    old={r['episode_id']:r for r in lines(PILOT/'score/scored_episodes.jsonl')}
    source={r['episode_id']:r for r in lines(PILOT/'inference.jsonl')}
    records=[]
    for row,rec in zip(manifest,seal['receipts']):
        assert row['episode_id']==rec['episode_id']
        with math.Image.open(row['query_mask_path']) as im:truth=(np.asarray(im.convert('L'))>0).astype(np.uint8)
        assert math.array_sha(truth)==row['query_mask_hash']
        shape=tuple(row['query_size_hw']);truth=resize(truth,shape,'nearest')>.5
        path=OUT/'predictions'/rec['filename'];assert sha(path)==rec['prediction_sha256']
        src=source[row['episode_id']]
        with np.load(PILOT/'predictions'/src['filename'],allow_pickle=False) as z:
            baseline=np.unpackbits(z['original/foris.crf'],count=truth.size).reshape(shape).astype(bool)
        b_iu=[int((baseline&truth).sum()),int((baseline|truth).sum())]
        assert b_iu==old[row['episode_id']]['iu']['original']['foris.crf']
        value=dict(episode_id=row['episode_id'],dataset=row['dataset'],fold=row['fold'],class_id=old[row['episode_id']]['class_id'],
                   query_photo_id=old[row['episode_id']]['query_photo_id'],iu=dict(old[row['episode_id']]['iu']['original']),edits={})
        with np.load(path,allow_pickle=False) as z:
            for method in z.files:
                if method=='original_hw':continue
                prediction=np.unpackbits(z[method],count=truth.size).reshape(shape).astype(bool)
                value['iu'][method]=[int((prediction&truth).sum()),int((prediction|truth).sum())]
                value['edits'][method]=[int((prediction&~baseline&truth).sum()),int((prediction&~baseline&~truth).sum()),
                    int((~prediction&baseline&truth).sum()),int((~prediction&baseline&~truth).sum())]
        records.append(value)
    (OUT/'scored_episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    answer=aggregate(records)
    write(OUT/'report.json',dict(n=len(records),frame='original',miou=answer,
        metric='Deep totalI/U; PACO observed-class pooledI/U then equal class/fold mean, 87 strata',
        new_encoder_calls=0,source_seal_sha256=sha(OUT/'sealed.json'),scored_episodes_sha256=sha(OUT/'scored_episodes.jsonl')))
    print(json.dumps(answer,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['generate','score']);args=parser.parse_args()
    begin=time.monotonic()
    generate() if args.stage=='generate' else score()
    print(json.dumps(dict(seconds=time.monotonic()-begin)),flush=True)
