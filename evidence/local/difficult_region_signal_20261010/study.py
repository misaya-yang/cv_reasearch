"""Frozen-field and raw-cache study of target/distractor signal in fixed ROIs."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import fcntl
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DATA = REPO.parent / 'cv_data'
PILOT = DATA / 'a/joint_role_pilot200_20261010'
MECHANISM = DATA / 'a/cache_mechanism_study_20261010'


def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def array_sha(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(json.dumps([list(x.shape), x.dtype.str]).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def resize(x, shape, mode='bilinear'):
    value = torch.as_tensor(np.asarray(x).copy(), dtype=torch.float64)[None, None]
    opts = {} if mode == 'nearest' else {'align_corners': False}
    return F.interpolate(value, shape, mode=mode, **opts)[0, 0].numpy()


def mask(row, role):
    with Image.open(row[role + '_mask_path']) as im:
        value = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
    assert array_sha(value) == row[role + '_mask_hash']
    return resize(value, (1024, 1024), 'nearest') > .5


def mass128(binary):
    return binary.reshape(128, 8, 128, 8).sum(axis=(1, 3)).reshape(-1).astype(np.float64)


def weighted_rank(score, fg, bg):
    valid = np.isfinite(score)
    total_f, total_b = float(fg.sum()), float(bg.sum())
    fg, bg, score = fg[valid], bg[valid], score[valid]
    keep = fg + bg > 0
    fg, bg, score = fg[keep], bg[keep], score[keep]
    nfg, nbg = float(fg.sum()), float(bg.sum())
    result = dict(fg=nfg, bg=nbg, excluded_fg=total_f-nfg, excluded_bg=total_b-nbg)
    if not nfg or not nbg:
        return dict(result, auc=None, ap=None, tpr_at_fpr05=None)
    order = np.argsort(score, kind='stable')
    s, f, b = score[order], fg[order], bg[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1])+1]
    f, b = np.add.reduceat(f, starts), np.add.reduceat(b, starts)
    auc = float(np.sum(f * (np.cumsum(b)-.5*b))/(nfg*nbg))
    tp, fp = np.cumsum(f[::-1]), np.cumsum(b[::-1])
    ap = float(np.sum(f[::-1] * tp/(tp+fp))/nfg)
    tpr = float(np.max(np.r_[0., tp[fp/nbg <= .05]/nfg]))
    return dict(result, auc=auc, ap=ap, tpr_at_fpr05=tpr, prevalence=nfg/(nfg+nbg))


def fixtures():
    assert weighted_rank(np.array([0., 1.]), np.array([0., 1.]), np.array([1., 0.]))['auc'] == 1
    tied = weighted_rank(np.array([.5]), np.array([3.]), np.array([7.]))
    assert tied['auc'] == .5 and abs(tied['ap']-.3) < 1e-12
    assert weighted_rank(np.array([.5]), np.array([0.]), np.array([7.]))['auc'] is None


def existing_fields():
    out = ROOT / 'existing'
    out.mkdir(exist_ok=True)
    assert not (out/'sealed.json').exists(), 'Existing complete fields must not be overwritten'
    (out/'fields').mkdir(exist_ok=True)
    manifest = read(PILOT/'manifest.json')
    source = {r['episode_id']: r for r in lines(PILOT/'inference.jsonl')}
    diagnostic = {r['episode_id']: r for r in lines(MECHANISM/'frontend_fields_episodes.jsonl')}
    write(out/'config.json', dict(manifest_sha256=sha(PILOT/'manifest.json'), script_sha256=sha(__file__),
        signals=['P0', 'retained_peak', 'nearest_coverage', 'whole_competition', 'foris_source'],
        query_GT_in_field_generation=False, existing_source_sha256=sha(MECHANISM/'frontend_fields_episodes.jsonl')))
    receipts = []
    for i, row in enumerate(manifest):
        rec = source[row['episode_id']]
        path = PILOT/'fields'/rec['filename']
        assert sha(path) == rec['fields_sha256']
        with np.load(path, allow_pickle=False) as z:
            b0, alpha, P0 = (z[k].astype(np.float64) for k in ('b0', 'alpha', 'P0'))
            ids, valid = z['refpatch'], z['valid']
        c = mask(row, 'reference').reshape(64, 16, 64, 16).mean(axis=(1, 3)).ravel()
        coverage = np.stack((1-c, c), axis=1)
        weights = coverage / coverage.sum(0)
        logw = np.full((4096, 2), -np.inf)
        good = coverage > 0
        logw[good] = np.log(weights[good]).astype(np.float32).astype(np.float64)
        roles = np.r_[np.zeros(8, dtype=int), np.ones(8, dtype=int)]
        statew = logw[np.maximum(ids, 0), roles]
        recovered = np.full(b0.shape, -np.inf)
        recovered[valid] = .07*(np.log(b0[valid])+np.log(alpha[:, roles])[valid]-statew[valid])
        gap = recovered[:, 8:].max(1)-recovered[:, :8].max(1)
        nearest = ids[np.arange(len(ids)), recovered.argmax(1)]
        diag = diagnostic[row['episode_id']]
        assert sha(diag['old_competition_field']) == diag['old_competition_field_sha256']
        with np.load(diag['old_competition_field'], allow_pickle=False) as z: h = z['h'].copy()
        assert sha(diag['source_score_path']) == diag['source_score_sha256']
        with np.load(diag['source_score_path'], allow_pickle=False) as z:
            score = z['score'] if 'score' in z else z[z.files[0]]
            score = np.asarray(score).reshape(64, 64).copy()
        fields = dict(P0=P0[:, 1].reshape(128, 128), retained_peak=gap.reshape(128, 128),
            nearest_coverage=c[nearest].reshape(128, 128), whole_competition=resize(h.reshape(64, 64), (128, 128)),
            foris_source=resize(score, (128, 128)))
        dest = out/'fields'/rec['filename']
        np.savez_compressed(dest, **fields)
        receipts.append(dict(episode_id=row['episode_id'], filename=rec['filename'], sha256=sha(dest),
                             pilot_fields_sha256=rec['fields_sha256']))
    write(out/'sealed.json', dict(n=len(receipts), receipts=receipts, script_sha256=sha(__file__),
                                 query_GT_read=False, encoder_calls=0))
    print(json.dumps(dict(stage='existing_fields', n=len(receipts))), flush=True)


def reference_fit(reference, coverage):
    """Old native200 role-balanced ridge and Huber, same subset/lambda/loss."""
    ids = []
    for role in (coverage, 1-coverage):
        cumulative = np.cumsum(role)
        ids.extend(np.searchsorted(cumulative, (np.arange(128)+.5)*cumulative[-1]/128, side='left'))
    ids = np.unique(ids)
    x = reference[ids].double()
    c = torch.from_numpy(coverage[ids]).double()
    y = 2*c-1
    weights = .5*c/c.sum() + .5*(1-c)/(1-c).sum()

    def fit(w):
        mx = (w[:, None]*x).sum(0)/w.sum()
        my = (w*y).sum()/w.sum()
        xc, yc = x-mx, y-my
        matrix = xc@xc.T + torch.diag(.01/w.clamp_min(1e-15))
        coefficient = xc.T@torch.linalg.solve(matrix, yc)
        return coefficient, my-mx@coefficient

    coefficient, bias = fit(weights)
    fits = {'ridge': (coefficient.clone(), bias.clone())}
    for _ in range(12):
        residual = x@coefficient+bias-y
        robust = weights*torch.minimum(torch.ones_like(residual), .5/residual.abs().clamp_min(1e-12))
        coefficient, bias = fit(robust)
    fits['huber'] = coefficient, bias
    full_c = torch.from_numpy(coverage).to(reference)
    fg = F.normalize((reference*full_c[:, None]).sum(0), dim=0)
    bg = F.normalize((reference*(1-full_c[:, None])).sum(0), dim=0)
    fits['prototype'] = (fg-bg).double(), torch.zeros((), dtype=torch.float64)
    return fits, dict(fit_tokens=len(ids), coverage_mean=float(coverage.mean()),
                     selected_target_max=float(y.max()), selected_target_min=float(y.min()))


def raw_fields():
    out = ROOT/'raw'
    out.mkdir(exist_ok=True); (out/'fields').mkdir(exist_ok=True); (out/'records').mkdir(exist_ok=True)
    assert not (out/'sealed.json').exists(), 'Do not overwrite complete fields'
    cfg = read(PILOT/'config.json')
    manifest = read(PILOT/'manifest.json')
    source = {r['episode_id']:r for r in lines(PILOT/'inference.jsonl')}
    profile_path = Path(cfg['raw_profile'])
    assert sha(profile_path) == cfg['raw_profile_sha256']
    cache_folder = profile_path.parent
    sys.path[:0] = [str(PILOT/'frozen/src'), str(PILOT/'frozen/scripts'), str(DATA/'third_party/foris_official')]
    from ics.official_data import load_inputs
    from raw_feature_cache import tensor_hash, canonical_hash
    from utils.data import build_transform
    transform = build_transform(1024)
    basis_path = DATA/'native_assets/positional_basis.pt'
    basis = torch.load(basis_path, map_location='cpu', weights_only=True)['basis'].float()
    assert tuple(basis.shape) == (1024, 500)
    projection = torch.eye(1024)-basis@basis.T
    frozen = out/'frozen'; frozen.mkdir(exist_ok=True)
    config = dict(manifest_sha256=sha(PILOT/'manifest.json'), script_sha256=sha(__file__),
        basis_sha256=sha(basis_path), profile_sha256=sha(profile_path),
        methods=['prototype', 'ridge', 'huber'], representations=['raw', 'source_apd'], views=['global', 'local4'],
        labels='exact area coverage from legal reference mask; no query labels',
        fit='old native200 quantile subset128 per role; balanced weights, y=2c-1; lambda=.01; Huber delta=.5,12 IRLS',
        apd='original whole-query semantic branch (<.8); same native frozen projection for whole/reference/local',
        query_GT_read=False, new_encoder_calls=0)
    if (out/'config.json').exists(): assert read(out/'config.json') == config, 'Resume source changed'
    else:
        write(out/'config.json', config)
        (frozen/'study.py').write_bytes(Path(__file__).read_bytes())
        historical = subprocess.check_output(['git', 'show', '6e986b9^:src/ics/cpu100/invariance_support.py'], cwd=REPO)
        (frozen/'historical_invariance_support.py').write_bytes(historical)
    receipts = []
    for i, row in enumerate(manifest):
        rec = source[row['episode_id']]; output = out/'fields'/rec['filename']; record_file = out/'records'/f'{i:06d}.json'
        if record_file.exists():
            receipt = read(record_file); assert sha(output) == receipt['sha256']; receipts.append(receipt); continue
        start = time.monotonic()
        reference, reference_mask, query = load_inputs(row, DATA)
        canonical_query = transform.transforms[0](query)
        images = [reference, query] + [canonical_query.crop(box) for box in ((0,0,512,512),(512,0,1024,512),(0,512,512,1024),(512,512,1024,1024))]
        arrays, raw_receipts = [], []
        for image, request in zip(images, rec['raw_inputs']):
            model_input = transform(image).numpy()
            key = canonical_hash(dict(profile=cache_folder.name, input_tensor_hash=tensor_hash(model_input)))
            assert key == request['key'], (row['episode_id'], request['role'])
            folder = cache_folder/key
            # A shared read lock preserves raw-cache locking without opening assets for writing.
            with (folder/'entry.lock').open('rb') as lock:
                fcntl.flock(lock, fcntl.LOCK_SH)
                info = read(folder/'entry.json')
                assert info['profile_id'] == cache_folder.name
                assert info['input_tensor_hash'] == tensor_hash(model_input)
                assert info['file_sha256'] == request['payload_sha256']
                assert Path(info['file']).name == info['file'] == info['file_sha256']+'.npz'
                assert sha(folder/info['file']) == request['payload_sha256']
                with np.load(folder/info['file'], allow_pickle=False) as z: array = z['O/24'].copy()
                assert array.shape == (4096,1024) and array.dtype == np.float32
                assert tensor_hash(array) == request['tensor_sha256'] == info['features']['O/24']['tensor_sha256']
                fcntl.flock(lock, fcntl.LOCK_UN)
            arrays.append(torch.from_numpy(array)); raw_receipts.append(request)
        io_seconds = time.monotonic()-start
        c = mask(row, 'reference').reshape(64,16,64,16).mean(axis=(1,3)).ravel()
        whole_ref, whole_query = (F.normalize(v, dim=1) for v in arrays[:2])
        binary = resize(reference_mask.numpy(), (64,64), 'nearest').ravel() > .5
        mu_q = whole_query.mean(0)
        semantic = float(F.normalize(whole_ref[binary].mean(0), dim=0)@mu_q/(mu_q.norm()+1e-6)) if binary.any() else None
        apply_apd = semantic is None or semantic < .8
        global_raw = F.interpolate(arrays[1].reshape(64,64,1024).permute(2,0,1)[None], (128,128), mode='bilinear', align_corners=False)[0].permute(1,2,0).reshape(-1,1024)
        global_unit = F.normalize(global_raw, dim=1)
        local_unit = torch.cat((torch.cat((arrays[2].reshape(64,64,1024), arrays[3].reshape(64,64,1024)), dim=1),
                                torch.cat((arrays[4].reshape(64,64,1024), arrays[5].reshape(64,64,1024)), dim=1)), dim=0).reshape(-1,1024)
        local_unit = F.normalize(local_unit, dim=1)
        del arrays, global_raw
        fields, fitted = {}, {}
        for representation in ('raw', 'source_apd'):
            if representation == 'source_apd' and apply_apd:
                r = F.normalize(whole_ref@projection.T, dim=1)
                qg = F.normalize(global_unit@projection.T, dim=1)
                ql = F.normalize(local_unit@projection.T, dim=1)
            else: r, qg, ql = whole_ref, global_unit, local_unit
            fits, fit_info = reference_fit(r, c)
            fitted[representation] = fit_info
            coefficients = torch.stack([fits[m][0] for m in fits], dim=1)
            biases = torch.stack([fits[m][1] for m in fits])
            for view, features in [('global',qg), ('local4',ql)]:
                values = features.double()@coefficients+biases
                for j, method in enumerate(fits): fields[f'{representation}.{method}.{view}'] = values[:,j].numpy().reshape(128,128)
        np.savez_compressed(output, **fields)
        receipt = dict(episode_id=row['episode_id'], filename=rec['filename'], sha256=sha(output),
            raw_receipts=raw_receipts, input_and_raw_seconds=io_seconds, total_seconds=time.monotonic()-start,
            apd_applied=apply_apd, source_semantic_score=semantic, fit=fitted)
        write(record_file, receipt); receipts.append(receipt)
        write(out/'activity.json', dict(completed=len(receipts), total=len(manifest), last=row['episode_id']))
        if (i+1)%20 == 0: print(json.dumps(dict(completed=i+1, total=200, seconds_last=receipt['total_seconds'])), flush=True)
    write(out/'sealed.json', dict(n=len(receipts), receipts=receipts, script_sha256=sha(__file__),
                                 query_GT_read=False, encoder_calls=0, cache_writes=0))
    print(json.dumps(dict(stage='raw_fields', n=len(receipts))), flush=True)


def evaluate(which):
    fixtures()
    out = ROOT / which
    seal = read(out/'sealed.json')
    manifest = read(PILOT/'manifest.json')
    infer = {r['episode_id']: r for r in lines(PILOT/'inference.jsonl')}
    output, grouped = [], defaultdict(list)
    for row, rec in zip(manifest, seal['receipts']):
        assert row['episode_id'] == rec['episode_id']
        path = out/'fields'/rec['filename']; assert sha(path) == rec['sha256']
        with np.load(path, allow_pickle=False) as z: fields = {k: z[k].reshape(-1).astype(float) for k in z.files}
        old = infer[row['episode_id']]
        pred = PILOT/'predictions'/old['filename']; assert sha(pred) == old['prediction_sha256']
        with np.load(pred, allow_pickle=False) as z:
            base = np.unpackbits(z['cli1024/foris.crf'], count=1024**2).reshape(1024, 1024).astype(bool)
            unary = np.unpackbits(z['cli1024/role.unary'], count=1024**2).reshape(1024, 1024).astype(bool)
        truth = mask(row, 'query')
        rois = {'global': np.ones(base.shape, dtype=bool), 'foris_foreground': base, 'new_unary_foreground': unary & ~base}
        r = dict(episode_id=row['episode_id'], dataset=row['dataset'], fold=row['fold'], class_id=row['loader_class_id'], rois={})
        for roi, region in rois.items():
            fg, bg = mass128(region & truth), mass128(region & ~truth)
            values = {key: weighted_rank(score, fg, bg) for key, score in fields.items()}
            r['rois'][roi] = values
            for key, value in values.items(): grouped[row['dataset'], roi, key].append(value)
            if 'retained_peak' in fields:
                reject = fields['retained_peak'] < -1e-5
                r['rois'][roi]['peak_BG_counts'] = dict(target=float(fg.sum()), distractor=float(bg.sum()),
                    target_rejected=float(fg[reject].sum()), distractor_rejected=float(bg[reject].sum()))
        output.append(r)
    (out/'episodes.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in output))
    summary = {}
    for (dataset, roi, signal), values in grouped.items():
        key = '/'.join((dataset, roi, signal))
        valid = [v for v in values if v['auc'] is not None]
        summary[key] = dict(n_valid=len(valid), macro={m:float(np.mean([v[m] for v in valid])) if valid else None
            for m in ('auc', 'ap', 'tpr_at_fpr05', 'prevalence')},
            fg=sum(v['fg'] for v in values), bg=sum(v['bg'] for v in values))
    peak = {}
    if which == 'existing':
        for dataset in ('deepglobe_road', 'paco_part'):
            rows = [r['rois']['new_unary_foreground']['peak_BG_counts'] for r in output if r['dataset'] == dataset]
            peak[dataset] = {k:sum(v[k] for v in rows) for k in rows[0]}
    write(out/'report.json', dict(n=len(output), roi_frame='128 grid with canonical1024 pixel masses; not original-resolution mIoU',
        summary=summary, new_foreground_peak_BG=peak, source_seal_sha256=sha(out/'sealed.json'),
        episodes_sha256=sha(out/'episodes.jsonl'), evaluator_sha256=sha(__file__)))
    print(json.dumps(dict(stage='evaluate', source=which, n=len(output), summary=summary, peak_BG=peak)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['existing', 'raw', 'evaluate'])
    parser.add_argument('--source', default='existing')
    args = parser.parse_args()
    begin = time.monotonic()
    if args.stage == 'existing': existing_fields()
    elif args.stage == 'raw': raw_fields()
    else: evaluate(args.source)
    print(json.dumps(dict(seconds=time.monotonic()-begin)), flush=True)
