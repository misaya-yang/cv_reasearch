#!/usr/bin/env python3
"""Finite existing-cache replay. No encoder constructor/forward, training or downloads.
Original public score binarizer + original image transform + original CRF finalizer.
GT is opened only after the five frozen supplied arms and matched naive prediction
are saved with a SHA. Old241 are design DEV, never independent confirmation.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

CARD = [
    'Assumption: equal-bank shared correspondence introduces useful FG-to-BG evidence beyond independent matching.',
    'Prediction: joint_flip original-grid delta > 0pp and > score_same_count delta; nonzero actions; exact baseline 10/10.',
    'Match: retain only as 10-design-DEV task signal; quantify deleted FP and lost TP, not a paper result.',
    'Mismatch: exact replay failure stops before GT; empty/worse-than-naive or TP-dominated flips refute this fixed sampled-star candidate; no rescue/sweep.'
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pick10(man):
    groups = [[r for r in man['episodes'] if r['fold'] == f] for f in range(4)]
    rows = []
    for j in range(3):
        for f in range(4):
            if len(rows) == 10:
                return rows
            r = groups[f][j]
            rows.append({k:r[k] for k in ('fold','e','c','support','query')})
    return rows


def metadata_only(a):
    from PIL import Image
    import numpy as np
    import torch
    man = json.loads(Path(a.manifest).read_text())
    source = Path(man['foris_root'])
    text = (source/'utils/data.py').read_text()
    if 'transforms.Resize(size=(image_size, image_size))' not in text:
        raise RuntimeError('unverified public transform: do not invent full-frame metadata')
    rows = pick10(man)
    for r in rows:
        with Image.open(Path(man['data_root'])/r['query']) as im:
            r['query_original_hw'] = [im.height, im.width]
        with Image.open(Path(man['data_root'])/r['support']) as im:
            r['source_original_hw'] = [im.height, im.width]
        name = '%d_%d_%d' % (r['fold'],r['e'],r['c'])
        for p in (Path(a.features)/(name+'.pt'),Path(a.packets)/(name+'.npz')):
            if not p.is_file():
                raise RuntimeError('missing existing cache; never encode: '+str(p))
        feat=torch.load(Path(a.features)/(name+'.pt'),map_location='cpu',weights_only=True)
        if set(feat)!={'q','r','debiased'} or any(feat[k].dtype!=torch.float16 or tuple(feat[k].shape)!=(4096,1024) for k in ('q','r')):
            raise RuntimeError('actual q/r complete grid format not verified')
        with np.load(Path(a.packets)/(name+'.npz'),allow_pickle=False) as z:
            if any(z[k].shape!=(64,64) or z[k].dtype!=np.float32 for k in ('score','cov')):
                raise RuntimeError('actual64x64 score/coverage grid is not verified')
        del feat
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt = dict(state='METADATA_PREPARED_FULL_PATH_NOT_YET_EXECUTED', episodes=rows,
                   manifest=str(a.manifest), manifest_sha256=sha(a.manifest), design_DEV=True,
                   source_full_frame_mapping_reviewed=True, feature_grid_hw=[64,64],
                   source_work_hw=[1024,1024],query_work_hw=[1024,1024],
                   transform_definition='source build_transform(1024): PIL square Resize, ToTensor, ImageNet Normalize; no padding/crop',
                   source_sha256={n:sha(source/n) for n in ('models/foris.py','utils/data.py','utils/refinement.py')},
                   encoder_calls=0,query_GT_read=False,card=CARD)
    (out/'metadata_preflight.json').write_text(json.dumps(receipt,indent=1))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('episodes','card')}),flush=True)


def source_finalizer(man, device):
    import torch
    sys.path.insert(0,man['foris_root'])
    from models.foris import FoRIS
    from utils.data import build_transform
    from utils.refinement import init_crf
    # Deliberately avoid __init__: that would load an encoder and build an SVD basis.
    host = FoRIS.__new__(FoRIS)
    torch.nn.Module.__init__(host)
    host.device = torch.device(device)
    host.image_size = 1024
    host.resize_to_orig_size = False
    host.mask_refiner = 'crf'
    host._transform = build_transform(1024)
    host._crf, host._crf_band_px, host._crf_p_core = init_crf(1024,device)
    def forbidden(*args,**kwargs):
        raise RuntimeError('encoder/whole forward forbidden in existing-cache replay')
    host._extract_features = forbidden
    host.predict = forbidden
    return host.eval().requires_grad_(False)


def unpack_mask(bits, shape):
    import numpy as np
    bits = np.asarray(bits)
    if bits.dtype != np.uint8 or bits.ndim != 1 or len(bits) != (shape[0]*shape[1]+7)//8:
        raise RuntimeError('native packed mask does not match verified source working dimensions')
    return np.unpackbits(bits)[:shape[0]*shape[1]].reshape(shape).astype(bool)


def run(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from export_cached_relational_packet import export_packet
    from relational_exclusion_cpu import relational_exclusion, finalize_deletion, MODES
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    previous=None
    if (out/'report.json').exists():
        if not a.resume:
            raise RuntimeError('fresh scientific output required, or explicit --resume')
        previous=json.loads((out/'report.json').read_text())
        if previous['state'] not in ('COMPLETED_FINITE_BUDGET','ERROR'):
            raise RuntimeError('resume requires a finished limited/error run, never a healthy active run')
    elif a.resume:
        raise RuntimeError('no frozen result exists to resume')
    if not (out/'metadata_preflight.json').exists():
        raise RuntimeError('metadata-only preparation must precede run')
    meta = json.loads((out/'metadata_preflight.json').read_text())
    if meta['manifest_sha256'] != sha(a.manifest):
        raise RuntimeError('frozen metadata manifest drift')
    for name,value in meta['source_sha256'].items():
        if sha(Path(man['foris_root'])/name) != value:
            raise RuntimeError('public source drift')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if a.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.4)
    rows=meta['episodes'];records=[];prior_elapsed=0.0;resume_snapshot=None
    if previous is not None:
        records=[json.loads(line) for line in (out/'episodes.jsonl').read_text().splitlines() if line.strip()]
        if len(records)!=previous['episodes'] or previous.get('native_exact')!=len(records) or len(records)>len(rows):
            raise RuntimeError('stream/report count differs; no automatic guessing on resume')
        for old,wanted in zip(records,rows):
            if any(old['row'][k]!=wanted[k] for k in ('fold','e','c','support','query')):
                raise RuntimeError('completed prefix does not match frozen ten-pair order')
            name='%d_%d_%d'%(wanted['fold'],wanted['e'],wanted['c'])
            if not old['predictions_frozen_before_query_GT'] or not old['native_pre_and_final_exact']:
                raise RuntimeError('prefix lacks frozen/exact provenance')
            if sha(out/'predictions'/(name+'.npz'))!=old['predictions_sha256']:
                raise RuntimeError('frozen first-pair prediction bytes changed; do not regenerate')
        prior_elapsed=float(previous['elapsed_s'])
        resume_snapshot=out/('report.before_resume_%03d_%d.json'%(len(records),time.time_ns()))
        resume_snapshot.write_bytes((out/'report.json').read_bytes())
    host=source_finalizer(man,a.device)
    begin=time.monotonic()
    (out/'predictions').mkdir(exist_ok=True)
    (out/'witnesses').mkdir(exist_ok=True)
    report = dict(state='RUNNING', episodes=len(records),native_exact=len(records),encoder_calls=0,training=False,
                  resumed_frozen_episodes=len(records),resume_report_snapshot=str(resume_snapshot) if resume_snapshot else None,
                  prior_elapsed_s=prior_elapsed,total_budget_seconds=a.budget_seconds,
                  gpu_center_requested=a.gpu_center,gpu_center_scope='remaining pairs only; original first prediction reused unchanged',
                  runner_sha256=sha(Path(__file__)),
                  design_DEV=True,feature_precision='cached native Part1 q/r float16; core normalizes in float64',
                  scope='full public finalizer replay on 10 frozen old DEV; not independent method score',card=CARD)
    def save():
        report['episodes'] = len(records)
        report['elapsed_s'] = prior_elapsed+time.monotonic()-begin
        if records:
            names = list(records[0]['original_iu'])
            report['episode_mean_original_IoU_pp'] = {n:100*float(np.mean([r['original_iu'][n][0]/max(r['original_iu'][n][1],1) for r in records])) for n in names}
            values={}
            for n in names:
                by={}
                for r in records:
                    c=r['row']['c'];by.setdefault(c,[0,0])
                    by[c]=[x+y for x,y in zip(by[c],r['original_iu'][n])]
                values[n]=100*float(np.mean([i/max(u,1) for i,u in by.values()]))
            report['class_sum_original_IoU_pp']=values
        tmp=out/'report.json.tmp';tmp.write_text(json.dumps(report,indent=1));tmp.replace(out/'report.json')
    save()
    try:
        with torch.inference_mode(), open(out/'episodes.jsonl','a') as stream:
            for row in rows[len(records):]:
                start=time.monotonic();name='%d_%d_%d'%(row['fold'],row['e'],row['c'])
                qp=Image.open(Path(man['data_root'])/row['query']).convert('RGB')
                if [qp.height,qp.width]!=row['query_original_hw']:
                    raise RuntimeError('original RGB shape drift')
                tgt=host._transform(qp)[None].to(a.device)
                if tuple(tgt.shape)!= (1,3,1024,1024):
                    raise RuntimeError('full square working transform was not reproduced')
                with np.load(Path(a.packets)/(name+'.npz'),allow_pickle=False) as z:
                    score=z['score'].copy();saved=unpack_mask(z['native'],(1024,1024));stored_pre=unpack_mask(z['pre'],(1024,1024))
                if score.shape!=(64,64) or score.dtype!=np.float32:
                    raise RuntimeError('only pinned full 64x64 float32 score is supported')
                pre=host._binarize_response(torch.from_numpy(score).to(a.device),target_hw=(1024,1024))
                if not np.array_equal(pre.cpu().numpy(),stored_pre):
                    raise RuntimeError('public pre-finalizer mask is not bit exact; stop before GT')
                reproduced=host._finalize_mask(pre,tgt).cpu().numpy().astype(bool)
                if not np.array_equal(reproduced,saved):
                    raise RuntimeError('public finalizer native mask differs: %d pixels; stop before GT'%int((reproduced!=saved).sum()))
                report['native_exact']+=1;save()
                clean_path=out/(name+'_clean.npz')
                metadata=dict(pair_id=name,representation_id='public FoRIS actual Part1 output, saved row-major fp16',
                              feature_provenance='pinned evidence_cache.py q/r, no new encoding',
                              source_coverage_provenance='source reference mask area coverage in old extent packet cov',
                              cache_alignment_provenance='same original241 manifest/cache; score pre and complete native CRF rechecked bit exact before GT',
                              feature_grid_hw=[64,64],source_work_hw=[1024,1024],query_work_hw=[1024,1024],
                              query_original_hw=row['query_original_hw'],mapping_contract='full_grid_regular_no_padding')
                # Temporary export is the supplied exact adapter. Its NPZ must share the source pair stem.
                clean_dir=out/'temporary_inputs';clean_dir.mkdir(exist_ok=True);clean_path=clean_dir/(name+'.npz')
                export_packet(features_path=Path(a.features)/(name+'.pt'),packet_path=Path(a.packets)/(name+'.npz'),
                              metadata=metadata,native_patch_mode='score_native',out_path=clean_path)
                with np.load(clean_path,allow_pickle=False) as z:
                    arrays={k:z[k].copy() for k in z.files}
                gpu_audit=dict(enabled=False,mode='original CPU selector')
                if a.gpu_center:
                    import relational_exclusion_cpu as core_module
                    from gpu_center_selector import gpu_center_selection
                    with gpu_center_selection(enabled=True,verify=True) as gpu_audit:
                        selector=core_module.select_role_banks
                        def checked_selector(*args,**kwargs):
                            answer=selector(*args,**kwargs)
                            if gpu_audit['parity_mismatches']:
                                raise RuntimeError('GPU-center role-bank parity mismatch; stop before any prediction/GT')
                            return answer
                        core_module.select_role_banks=checked_selector
                        try:
                            result=relational_exclusion(**arrays)
                        finally:
                            core_module.select_role_banks=selector
                else:
                    result=relational_exclusion(**arrays)
                fields=result['deletions']
                k=int(fields['joint_flip'].sum());naive=np.zeros((64,64),bool)
                eligible=np.flatnonzero(arrays['native_patch_fg'])
                order=eligible[np.lexsort((eligible,score.reshape(-1)[eligible]))]
                naive.reshape(-1)[order[:k]]=True
                fields['score_same_count']=naive
                final=dict(native=saved.copy());details={}
                def callback(mask):
                    return host._finalize_mask(torch.from_numpy(mask).to(a.device),tgt).cpu().numpy().astype(bool)
                for arm in list(MODES)+['score_same_count']:
                    x=finalize_deletion(fields[arm],native_pre_final=stored_pre,native_final=saved,
                                        finalizer=callback,mapping_contract='full_grid_nearest')
                    final[arm]=x['final_mask'];details[arm]=x['metadata']
                # Every label-free mask, including control, is frozen before any query label access.
                path=out/'predictions'/(name+'.npz')
                original={n:(F.interpolate(torch.from_numpy(v)[None,None].float(),tuple(row['query_original_hw']),mode='bilinear',align_corners=False)[0,0].numpy()>.5) for n,v in final.items()}
                frozen={n:np.packbits(v) for n,v in final.items()}
                frozen.update({'original__'+n:np.packbits(v) for n,v in original.items()})
                np.savez_compressed(path,**frozen)
                prediction_sha=sha(path)
                (out/'witnesses'/(name+'.json')).write_text(json.dumps(dict(records=result['records'],metadata=result['metadata'],finalizer=details,gpu_center_audit=gpu_audit)))
                clean_path.unlink()  # never retain another full-feature cache
                clean_path.with_suffix('.json').rename(out/'witnesses'/(name+'_input.json'))
                del arrays
                # GT is evaluation only; no masks or decisions can change below this point.
                truth_o=np.asarray(Image.open(Path(man['annotation_root'])/Path(row['query']).with_suffix('.png')))==row['c']+1
                truth_m=F.interpolate(torch.from_numpy(truth_o.copy())[None,None].float(),(1024,1024),mode='nearest')[0,0].numpy().astype(bool)
                def iu(p,t):return [int((p&t).sum()),int((p|t).sum())]
                if list(truth_o.shape)!=row['query_original_hw']:
                    raise RuntimeError('query annotation/RGB dimensions differ; frozen predictions cannot change')
                counts={n:dict(deleted_FP=int((saved&~p&~truth_m).sum()),lost_TP=int((saved&~p&truth_m).sum()),
                               changed_patch_count=int(fields[n].sum())) for n,p in final.items() if n!='native'}
                native_iu=iu(saved,truth_m);D=int((saved&~final['joint_flip']).sum());FP=int((saved&~truth_m).sum())
                oracle_model_iu=[native_iu[0],native_iu[1]-min(D,FP)]
                native_original_iu=iu(original['native'],truth_o)
                Do=int((original['native']&~original['joint_flip']).sum());FPo=int((original['native']&~truth_o).sum())
                oracle_original_iu=[native_original_iu[0],native_original_iu[1]-min(Do,FPo)]
                rec=dict(row=row,model_iu={n:iu(p,truth_m) for n,p in final.items()},
                         original_iu={n:iu(p,truth_o) for n,p in original.items()},deletion_counts=counts,
                         oracle_same_final_pixel_budget_model_iu=oracle_model_iu,
                         oracle_same_final_pixel_budget_original_iu=oracle_original_iu,
                         oracle_scope='post-freeze deletion-only arithmetic diagnostic; not star-family or CRF-method score',
                         predicted_patch_budget=k,model_final_deleted_pixels=D,predictions_sha256=prediction_sha,
                         predictions_frozen_before_query_GT=True,native_pre_and_final_exact=True,gpu_center_audit=gpu_audit,
                         input_coordinates='64 full regular cells over source/query square1024 Resize; no padding/crop',seconds=time.monotonic()-start)
                records.append(rec);stream.write(json.dumps(rec)+'\n');stream.flush();save()
                print(json.dumps(dict(pair=name,episodes=len(records),joint_flip_IU=rec['original_iu']['joint_flip'],native_IU=rec['original_iu']['native'],same_count_IU=rec['original_iu']['score_same_count'],seconds=rec['seconds'])),flush=True)
                if prior_elapsed+time.monotonic()-begin>=a.budget_seconds:
                    report['state']='COMPLETED_FINITE_BUDGET';save();return
        report['state']='COMPLETED';save()
    except BaseException as e:
        report.update(state='ERROR',error=repr(e));save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--features',required=True);p.add_argument('--packets',required=True)
    p.add_argument('--out',required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--device',default='cuda');p.add_argument('--budget-seconds',type=float,default=600)
    p.add_argument('--resume',action='store_true');p.add_argument('--gpu-center',action='store_true')
    a=p.parse_args();metadata_only(a) if a.prepare else run(a)

if __name__=='__main__':main()
