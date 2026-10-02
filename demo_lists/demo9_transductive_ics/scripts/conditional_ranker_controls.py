#!/usr/bin/env python3
"""NEXT-card controls on a completed full-token pilot; active source files untouched.

Small overlays keep original K4 masks/counts as an immutable prefix, add native
multi-reference naive, cached naive, and cached AG4 current candidates, and attach
probe_select's four primitive scores plus four products. No feature-cache copy.
Native candidate counts use official model-size masks; original-size counts are
separate. Cached readouts reuse ORIGINAL FP16 pair-conditioned tokens and rebuild
debiased features/clusters; this is not the old M15/400-episode protocol.

Output report.records[].path points to the original episode; overlay_path has
replacement candidate/scalar fields to merge after loading that episode. This
script does not alter training or data files used by an active run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import shutil

import numpy as np
import torch
import torch.nn.functional as F

SCORES=('rtg','cyc','link','fms','rtg*cyc','rtg*link','cyc*link','rtg*cyc*link')


def safe_downsample(mask,h,w,official):
    """Official utility crashes on an empty mask; emptiness is not absence evidence."""
    if not bool(mask.any()):return torch.zeros(h,w,dtype=torch.bool,device=mask.device)
    return official(mask.reshape(1,1,*mask.shape[-2:]),h,w).reshape(h,w).bool()


def unpack_native(payload,device):
    size=int(payload['candidate_native_mask_size']);packed=payload['candidate_native_mask_bits']
    values=np.unpackbits(np.asarray(packed),axis=1,count=size*size)
    return torch.from_numpy(values).reshape(len(values),size,size).to(device).bool()


def score_candidates(s,masks,sources):
    """Exact old single-source formula; explicitly averaged extension for multi-source.

    masks are query PATCH masks. sources[k] is [(image, source_patch_mask),...].
    The sole source of direct is annotated image 0: cyc=rtg, link=1, exactly old
    probe_select. Nonempty per-donor sources use original formula. Multi-source
    candidates average cyc/link/fms equally across actual nonempty references;
    that extension was absent in probe_select and is marked in provenance.
    """
    q=1;gold=s.gt64[0];link_cache={};fms_cache={};scores=[]
    def link(j,source):
        if j==0:return 1.
        key=(j,source.detach().cpu().numpy().tobytes())
        if key not in link_cache:link_cache[key]=s.iou(s.predict(0,[j],[source]),gold) if source.any() else 0.
        return link_cache[key]
    def fms(j,source):
        key=(j,source.detach().cpu().numpy().tobytes())
        if key not in fms_cache:fms_cache[key]=float(s.nnv(j,q)[0][source].mean()) if source.any() else 0.
        return fms_cache[key]
    for mask,refs in zip(masks,sources):
        rtg=s.iou(s.predict(0,[q],[mask]),gold) if mask.any() else 0.
        cycles=[];links=[];strengths=[]
        for j,source in refs:
            cycles.append(rtg if j==0 else (s.iou(s.predict(j,[q],[mask]),source) if mask.any() and source.any() else 0.))
            links.append(link(j,source));strengths.append(fms(j,source))
        cyc=float(np.mean(cycles)) if cycles else 0.;lnk=float(np.mean(links)) if links else 0.
        strength=float(np.mean(strengths)) if strengths else 0.
        scores.append([rtg,cyc,lnk,strength,rtg*cyc,rtg*lnk,cyc*lnk,rtg*cyc*lnk])
    return torch.tensor(scores,dtype=torch.float32)


def replay_set(payload,model,cluster,ImageSet,cluster_protos,device):
    """Label-independent clusters; inference object has only gold support truth."""
    original=torch.cat([payload['reference_tokens'][None],payload['query_tokens'][None],payload['donor_tokens']]).to(device).float()
    original=F.normalize(original,dim=-1);n,p,c=original.shape;h=int(p**.5)
    if h*h!=p:raise ValueError('Token grid must be square')
    labels=[];prototypes=[]
    for image in original:
        lab=torch.as_tensor(cluster(image,model.tau),device=device).long().reshape(-1)
        labels.append(lab);prototypes.append(cluster_protos(image,lab,int(lab.max())+1)[0])
    basis=model.positional_basis.float().to(device)
    debiased=F.normalize(original-(original@basis)@basis.T,dim=-1)
    legal=torch.zeros(n,h,h,dtype=torch.bool,device=device);legal[0]=payload['reference_mask'].to(device).bool()
    packed=np.zeros((n,model.image_size*model.image_size//8),dtype=np.uint8)
    s=ImageSet(dict(c=int(payload['c']),names=payload['photo_ids'],fq=debiased,
                   lab=torch.stack(labels),Po=prototypes,gt64=legal,gt_bits=packed,S=model.image_size))
    if bool(s.gt64[1:].any()) or bool(s.gt.any()):raise AssertionError('Evaluator truth entered replay object')
    return s


def candidate_counts(predictions,host_truth,original_truth):
    native=[];original=[]
    for prediction in predictions:
        native.append([int((prediction&host_truth).sum()),int((prediction|host_truth).sum())])
        resized=F.interpolate(prediction.reshape(1,1,*prediction.shape).float(),size=original_truth.shape,mode='bilinear',align_corners=False)[0,0]>.5
        original.append([int((resized&original_truth).sum()),int((resized|original_truth).sum())])
    return torch.tensor(native,dtype=torch.long),torch.tensor(original,dtype=torch.long)


def build_overlay(payload,model,s,images,support_host,host_truth,original_truth,official_downsample,propagate):
    native=unpack_native(payload,s.fd.device);core=len(native);donors=list(range(2,s.n));q=1
    if core!=1+len(donors):raise ValueError('Expected fixed core direct plus one candidate per donor')
    patches=[safe_downsample(m,s.h,s.h,official_downsample).reshape(-1) for m in native]
    p1={0:s.gt64[0],1:patches[0]};p1.update({j:payload['donor_masks'][j-2].to(s.fd.device).reshape(-1).bool() for j in donors})
    sources=[[(0,p1[0])]]+[[(j,p1[j])] for j in donors]
    provenance=list(payload['candidate_provenance'])
    # Genuine native naive: official joint encoding of all nonempty references.
    donor_native=[];native_mask_checks=[]
    for j in donors:
        prediction=model.predict_mask(images[0],support_host,images[j]).reshape(model.image_size,model.image_size).bool()
        donor_native.append(prediction)
        rebuilt=safe_downsample(prediction,s.h,s.h,official_downsample).reshape(-1)
        native_mask_checks.append(dict(donor_slot=j-2,exact_patch_mask=bool(torch.equal(rebuilt,p1[j])),
                                      changed_patches=int((rebuilt!=p1[j]).sum()),empty_native=not bool(prediction.any())))
    live=[k for k,p in enumerate(donor_native) if bool(p.any())]
    if support_host.any():
        ref_images=torch.cat([images[0]]+[images[donors[k]] for k in live])
        ref_masks=torch.cat([support_host]+[donor_native[k][None] for k in live])
        naive_native=model.predict_mask(ref_images,ref_masks,images[q]).reshape(model.image_size,model.image_size).bool()
    else:naive_native=torch.zeros_like(native[0])
    naive_cached=s.up(s.predict(q,[0]+donors,[p1[0]]+[p1[j] for j in donors],backward='majority'))
    trace={};rounds=propagate(s,list(range(1,s.n)),rounds=4,k=5,trust='agree',rule='tophalf',P1=p1,track=trace)
    current=s.up(rounds[-1][q]);before=rounds[-2]
    selected=[j for j in donors if trace['ok'][-1][j] and bool(before[j].any())]
    extras=[naive_native,naive_cached,current];native=torch.cat([native,torch.stack(extras)])
    patches += [safe_downsample(m,s.h,s.h,official_downsample).reshape(-1) for m in extras]
    sources += [[(0,p1[0])]+[(donors[k],p1[donors[k]]) for k in live],
                [(0,p1[0])]+[(j,p1[j]) for j in donors if bool(p1[j].any())],
                [(0,p1[0])]+[(j,before[j]) for j in selected]]
    provenance += [dict(kind='naive_native_multi',encoding_context=[payload['photo_ids'][0]]+[payload['photo_ids'][donors[k]] for k in live]+[payload['photo_ids'][q]]),
                   dict(kind='naive_cached_native_p1',backward='majority'),
                   dict(kind='current_ag4_cached_native_p1',rounds=4,trust='agree',rule='tophalf',k=5,selected_donor_slots=[j-2 for j in selected])]
    for p in provenance[core:]:p['scalar_definition']='Equal-mean extension over nonempty actual reference sources; original probe_select defined single-source candidates only'
    scalar=score_candidates(s,patches,sources)
    # All inference/nominee statistics above were frozen before evaluator labels.
    iu,oiu=candidate_counts(native,host_truth,original_truth)
    expected=payload['candidate_iu'].cpu().long()
    if not torch.equal(iu[:core],expected):raise AssertionError('Saved core native I/U changed under official mask evaluator')
    if 'candidate_original_iu' in payload and not torch.equal(oiu[:core],payload['candidate_original_iu'].cpu().long()):
        raise AssertionError('Saved core original-resolution I/U changed')
    cached_direct=s.up(s.predict(1,[0],[s.gt64[0]]))
    audit=dict(core_native_iu_exact=True,core_original_iu_exact=True,
               cached_direct_vs_native_exact=bool(torch.equal(cached_direct,native[0])),
               cached_direct_vs_native_changed_pixels=int((cached_direct!=native[0]).sum()),
               cached_direct_vs_native_iou=s.iou(cached_direct,native[0]),donor_native_replay=native_mask_checks,
               empty_core_candidates=[i for i,m in enumerate(native[:core]) if not bool(m.any())])
    return dict(e=int(payload['e']),c=int(payload['c']),split=payload['split'],photo_ids=payload['photo_ids'],
        source_candidate_count=core,candidate_state='COMPLETE',candidate_masks=torch.stack(patches).reshape(len(patches),s.h,s.h).cpu(),
        candidate_iu=iu,candidate_model_iu=iu,candidate_iou=(iu[:,0].double()/iu[:,1].clamp_min(1)).float(),
        candidate_original_iu=oiu,candidate_original_iou=(oiu[:,0].double()/oiu[:,1].clamp_min(1)).float(),
        candidate_native_mask_bits=np.packbits(native.cpu().numpy().reshape(len(native),-1),axis=1),
        candidate_native_mask_size=model.image_size,candidate_provenance=provenance,
        scalar_features=scalar,scalar_feature_names=list(SCORES),audit=audit,
        candidate_label_resolution=payload['candidate_label_resolution'],
        controls_contract='Original core prefix unchanged. Full old 8-score formulas on current FP16 pair tokens reconstructed in native positional complement; native multi-shot naive joint encoding, cached AG4 adapter with native P1. Not M15 or historical 59.1 protocol.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report');ap.add_argument('--out');ap.add_argument('--limit',type=int,default=10)
    ap.add_argument('--prepared-root',default='/root/autodl-tmp/demo9');ap.add_argument('--max-bytes',type=int,default=200_000_000)
    ap.add_argument('--self-check',action='store_true');a=ap.parse_args()
    if a.self_check:self_check();return
    if not a.report or not a.out:ap.error('--report and --out are required')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    sys.path.insert(0,str(Path(a.prepared_root)/'scripts'));import _paths
    sys.path.insert(0,_paths.DEMO4)
    from icx.common import build_model,DEV
    from utils.data import load_image,load_mask,downsample_mask
    from utils.clustering import agglomerative_clustering
    from tics.imageset import ImageSet,cluster_protos
    from tics import propagate
    from PIL import Image
    source=json.loads(Path(a.report).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if Path(a.report).resolve().parent==out.resolve():raise ValueError('Use a distinct controls output directory')
    # A bounded parallel first10 control probe is real work, not duplicate work.
    # Before the serial full stage, wait without CUDA allocation and reuse only
    # overlays verified against these exact immutable source episode files.
    pilot=out.with_name(out.name+'_pilot10')
    if not out.name.endswith('_pilot10') and pilot.exists():
        pidfile=pilot/'controls.pid';deadline=time.time()+900
        while pidfile.exists():
            try:pid=int(pidfile.read_text().strip());os.kill(pid,0)
            except (ProcessLookupError,ValueError):break
            if time.time()>deadline:raise RuntimeError('Parallel control probe still active after15min; inspect, do not duplicate')
            time.sleep(2)
        known={int(e['e']):Path(e['path']) for e in source['records']}
        for old in pilot.glob('controls_*.pt'):
            overlay=torch.load(old,weights_only=False,map_location='cpu')
            e=int(old.stem.split('_')[-1]);original=known.get(e)
            if original is None:continue
            if overlay.get('source_episode_sha256')!=hashlib.sha256(original.read_bytes()).hexdigest():
                raise RuntimeError('Parallel overlay input hash differs; do not reuse')
            target=out/old.name
            if not target.exists():shutil.copy2(old,target)
    script_paths=[Path(__file__),Path(a.prepared_root)/'scripts/probe_select.py',Path(a.prepared_root)/'tics/imageset.py',Path(a.prepared_root)/'tics/propagate.py',Path(_paths.DEMO4)/'INSID3/models/insid3.py']
    result=dict(state='RUNNING',args=vars(a),source_report=a.report,records=[],
                overlay_schema='Load records.path original full tokens, then update from records.overlay_path; never drop original full tokens',
                scalar_formulas='Exact probe_select rtg/cyc/link/fms plus 4 products for the original direct/donor candidates; explicit equal-source averaging for multi-reference additions',
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in script_paths})
    start=time.time();model=None
    def save():
        result.update(elapsed_s=time.time()-start,bytes=sum(p.stat().st_size for p in out.glob('controls_*.pt')))
        tmp=out/'report.tmp';tmp.write_text(json.dumps(result));tmp.replace(out/'report.json')
    save()
    try:
        for entry in source['records'][:a.limit]:
            path=Path(entry['path']);overlay_path=out/f'controls_{int(entry["e"]):04d}.pt'
            if overlay_path.exists():
                result['records'].append(dict(entry,overlay_path=str(overlay_path),resumed=True));save();continue
            payload=torch.load(path,weights_only=False,map_location='cpu')
            if payload.get('candidate_state')!='COMPLETE' or payload.get('donor_token_state')!='COMPLETE':
                raise RuntimeError(f'Input candidate/donor stage incomplete: {path}')
            if model is None:model=build_model()
            s=replay_set(payload,model,agglomerative_clustering,ImageSet,cluster_protos,DEV)
            base=Path(os.environ.get('DEMO4_CACHE','/root/demo4_cache'))/'data/COCO2014'
            images=[]
            for name in payload['photo_ids']:images.append(load_image(Image.open(base/name).convert('RGB'),model._transform,DEV)[0])
            # Only annotated support truth is supplied to predictors. Query labels stay in evaluator variables.
            annotation=Path(_paths.COCO_ANN)
            support=torch.from_numpy(np.array(Image.open(annotation/str(Path(payload['photo_ids'][0]).with_suffix('.png')))))==payload['c']+1
            support_host=load_mask(support,model.image_size,DEV)
            query_truth=torch.from_numpy(np.array(Image.open(annotation/str(Path(payload['photo_ids'][1]).with_suffix('.png')))))==payload['c']+1
            host_truth=load_mask(query_truth,model.image_size,DEV)[0]
            with torch.inference_mode():overlay=build_overlay(payload,model,s,images,support_host,host_truth,query_truth.to(DEV),downsample_mask,propagate)
            overlay['source_episode_path']=str(path);overlay['source_episode_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            tmp=overlay_path.with_suffix('.tmp');torch.save(overlay,tmp)
            if result['bytes']+tmp.stat().st_size>a.max_bytes:tmp.unlink();result['state']='BUDGET_STOP';break
            tmp.replace(overlay_path)
            result['records'].append(dict(entry,overlay_path=str(overlay_path),candidate_count=len(overlay['candidate_iu']),audit=overlay['audit']))
            save();print(len(result['records']),payload['e'],round(result['elapsed_s'],1),overlay['audit'],flush=True)
            del payload,s,images,overlay
        if result['state']=='RUNNING':result['state']='COMPLETED'
        save()
    except BaseException as exc:result.update(state='ERROR',error=repr(exc));save();raise


def self_check():
    from types import SimpleNamespace
    count=0
    def official(mask,h,w):
        nonlocal count;count+=1
        if not mask.any():raise AssertionError('Empty mask reached buggy official fallback')
        return F.interpolate(mask.float(),size=(h,w),mode='nearest')[0,0]>.5
    assert not safe_downsample(torch.zeros(8,8,dtype=torch.bool),4,4,official).any() and count==0
    torch.set_num_threads(1)
    for seed in range(10):
        g=torch.Generator().manual_seed(seed);gold=torch.rand(16,generator=g)>.5;donor=torch.rand(16,generator=g)>.5
        masks=[torch.rand(16,generator=g)>.5 for _ in range(2)]
        def predict(target,refs,ms):return torch.roll(ms[0],target-refs[0])
        def iou(a,b):return float((a&b).sum())/max(float((a|b).sum()),1)
        nn=lambda t,j:(torch.linspace(.1,.9,16),torch.arange(16))
        fake=SimpleNamespace(gt64=[gold],predict=predict,iou=iou,nnv=nn)
        got=score_candidates(fake,masks,[[(0,gold)],[(2,donor)]]).numpy()
        rtg=np.array([iou(predict(0,[1],[mask]),gold) if mask.any() else 0 for mask in masks])
        cyc=np.array([rtg[0],iou(predict(2,[1],[masks[1]]),donor) if masks[1].any() else 0])
        link=np.array([1,iou(predict(0,[2],[donor]),gold)])
        fms=np.array([float(nn(0,1)[0][gold].mean()),float(nn(2,1)[0][donor].mean())])
        expected=np.stack([rtg,cyc,link,fms,rtg*cyc,rtg*link,cyc*link,rtg*cyc*link],1)
        assert np.allclose(got,expected),seed
        packed=np.packbits(torch.stack(masks).numpy(),axis=1)
        assert torch.equal(unpack_native(dict(candidate_native_mask_size=4,candidate_native_mask_bits=packed),'cpu').reshape(2,-1),torch.stack(masks))
    print('CPU 10-episode synthetic smoke passed: exact original 8-score formulas, direct/source conventions, packed native masks, empty-mask guard.')


if __name__=='__main__':main()
