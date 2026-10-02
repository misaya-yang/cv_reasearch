#!/usr/bin/env python3
"""ONE preregistered train-only same-RGB counterfactual data-availability check.

Fixed FIRST minimal_recombined_examples of the supplied feasibility document;
never select pairs or replacements by query prediction/IoU. Two legal support
annotations generate an A4+B4 native hypothesis union. Freeze that union, the
physical donor pool, P1 union, provenance and direct_index=0 for BOTH tasks.
Only gold reference mask, derived scores and evaluator labels change. Full
pair-context token tensors are stored once, with two small task overlays.

Not a method-gain test or a family gate. A failure says this fixed pair cannot
teach the prespecified flip, not that other eligible pairs lack information.
No automatic expansion. Existing frozen DINO/official data only; no downloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def photo_id(name):
    import re
    stem=Path(name).stem;match=re.fullmatch(r'COCO_(?:train|val)2014_(\d+)',stem)
    return 'COCO:'+str(int(match.group(1))) if match else stem


def validate_pair(feasibility,manifest):
    # Metadata filtering was frozen by the earlier feasibility study. No oracle
    # or performance label is read here, and no fallback pair is searched.
    pair=feasibility['minimal_recombined_examples'][0]
    names=[pair['support'],pair['query']]+pair['donors']
    if len(pair['donors'])!=3 or len(set(map(photo_id,names)))!=5:
        raise ValueError('Need fixed distinct support/query/three donors')
    train=set(photo_id(p) for r in manifest['records'] if r['split']=='train' for p in r['image_ids'])
    held=set(photo_id(p) for r in manifest['records'] if r['split']!='train' for p in r['image_ids'])
    if set(map(photo_id,names))&held:raise ValueError('Counterfactual role photo overlaps dev/test')
    if not set(map(photo_id,names))<=train:raise ValueError('Counterfactual role photo outside NEW training cohort')
    classes=set(int(r['c']) for r in manifest['records'] if r['split']=='train')
    if len(pair['classes'])!=2 or not set(pair['classes'])<=classes:
        raise ValueError('Counterfactual task classes must both belong to NEW train')
    return pair,names


class PairMemo:
    """Memoize actual native ordered RGB contexts, never features by photo alone."""
    def __init__(self,model,images):
        self.model=model;self.images=images;self.original=model._extract_features
        self.cache={};self.key=None;model._extract_features=self.extract
    def extract(self,imgs):
        if self.key is None:raise RuntimeError('No native RGB context assigned')
        if self.key not in self.cache:self.cache[self.key]=self.original(imgs)
        return self.cache[self.key]
    def predict(self,target,refs,masks):
        keep=[k for k,m in enumerate(masks) if bool(m.any())]
        if not keep:return torch.zeros(self.model.image_size,self.model.image_size,dtype=torch.bool,device=self.images[0].device)
        refs=[refs[k] for k in keep];masks=[masks[k] for k in keep];self.key=tuple(refs+[target])
        return self.model.predict_mask(torch.cat([self.images[j] for j in refs]),
            torch.stack([m.reshape(self.model.image_size,self.model.image_size) for m in masks]),
            self.images[target]).reshape(self.model.image_size,self.model.image_size).bool()
    def tokens(self,key,slot):
        pair=F.normalize(self.cache[tuple(key)].float(),p=2,dim=2)[0]
        return pair[slot].flatten(1).T.half().cpu()
    def close(self):self.model._extract_features=self.original


def oracle_diagnostic(iou_a,iou_b):
    a=np.asarray(iou_a,float);b=np.asarray(iou_b,float)
    ba=np.flatnonzero(np.isclose(a,a.max(),atol=1e-12,rtol=0)).tolist()
    bb=np.flatnonzero(np.isclose(b,b.max(),atol=1e-12,rtol=0)).tolist()
    disjoint=not bool(set(ba)&set(bb))
    # Stronger tie handling: every OTHER-task best candidate must lose >=.1.
    regret_a=float(a.max()-max(a[bb]));regret_b=float(b.max()-max(b[ba]))
    checks=dict(both_oracles_ge_50=bool(a.max()>=.5 and b.max()>=.5),
                optimal_sets_disjoint=disjoint,both_cross_regrets_ge_10=bool(regret_a>=.1 and regret_b>=.1))
    return dict(oracle_iou=[float(a.max()),float(b.max())],best_indices=[ba,bb],
                cross_regret=[regret_a,regret_b],checks=checks,pair_teaches_prespecified_flip=all(checks.values()),
                scope='Fixed-pair training-data/interface diagnostic only. Failure does not refute other eligible pairs.')


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--feasibility',default='/tmp/demo9_counterfactual_feasibility120_v2.json')
    ap.add_argument('--manifest');ap.add_argument('--prepared-root',default='/root/autodl-tmp/demo9')
    ap.add_argument('--out');ap.add_argument('--limit',type=int,default=1);ap.add_argument('--max-bytes',type=int,default=100_000_000)
    ap.add_argument('--self-check',action='store_true');a=ap.parse_args()
    if a.self_check:self_check();return
    if a.limit!=1:ap.error('This frozen premise card permits --limit 1 only; no automatic sweep')
    if not a.out:ap.error('--out is required')
    feasibility=json.loads(Path(a.feasibility).read_text());manifest_path=Path(a.manifest or feasibility['source_manifest'])
    manifest=json.loads(manifest_path.read_text());manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if feasibility.get('source_manifest_sha256')!=manifest_sha:raise ValueError('Feasibility/new-cohort manifest SHA mismatch')
    pair,names=validate_pair(feasibility,manifest)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);report_path=out/'report.json'
    if report_path.exists():raise RuntimeError('Use a fresh own counterfactual output directory; never overwrite a prior premise check')
    start=time.time();report=dict(state='RUNNING',args=vars(a),dataset='COCO-20i',fold=0,split='new_train_only',pair=pair,
        source_manifest_sha256=manifest_sha,photo_purge=True,records=[],
        four_lines=dict(assumption='Same RGB/candidates can express two legitimate targets solely specified by the support mask.',
            prediction='Full RGB feature patches identical across gold tasks; shared K8/P1/provenance/direct0 exactly fixed. Both oracle>=.5, best sets disjoint, cross regrets>=.1.',
            match='This fixed pair may teach reference conditioning; stop after one and ask primary for the next data decision, no method claim.',
            mismatch='This pair cannot teach the prespecified flip or interface is invalid. Preserve diagnostic, stop GPU; do not cherry-pick replacements or reject all eligible pairs.'),
        inference_allowlist=['reference_tokens','query_tokens','donor_tokens','candidate_masks','donor_masks','reference_mask','scalar_features','direct_index'],
        forbidden_model_metadata=['e','c','classes','pair_id','task_tag','photo_ids','generation_classes','source_train_episode','candidate_iou','candidate_iu','candidate_provenance.hypothesis_id'],
        contract='Same shared path for both tasks; two legal semantic generation passes, fixed union before query evaluator; source concept IDs and provenance hypothesis names are audit-only.')
    def save():
        report['elapsed_s']=time.time()-start
        tmp=report_path.with_suffix('.tmp');tmp.write_text(json.dumps(report,allow_nan=False));tmp.replace(report_path)
    save();memo=None
    try:
        os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
        sys.path.insert(0,str(Path(a.prepared_root)/'scripts'));import _paths
        sys.path.insert(0,_paths.DEMO4)
        from icx.common import build_model,DEV
        from utils.data import load_image,load_mask,downsample_mask
        from utils.clustering import agglomerative_clustering
        from tics.imageset import ImageSet,cluster_protos
        from conditional_ranker_controls import safe_downsample,score_candidates,replay_set,candidate_counts
        from PIL import Image
        model=build_model();base=Path(os.environ.get('DEMO4_CACHE','/root/demo4_cache'))/'data/COCO2014';annroot=Path(_paths.COCO_ANN)
        images=[load_image(Image.open(base/name).convert('RGB'),model._transform,DEV)[0] for name in names]
        support_ann=torch.from_numpy(np.array(Image.open(annroot/str(Path(names[0]).with_suffix('.png'))))).to(DEV)
        golds=[load_mask(support_ann==c+1,model.image_size,DEV)[0] for c in pair['classes']]
        if not all(bool(g.any()) for g in golds):raise ValueError('One legal support annotation is empty')
        memo=PairMemo(model,images);native=[];provenance=[];p1_groups=[]
        with torch.inference_mode():
            # Actual encoding invariant audit: independent native extraction of the
            # same ordered RGB pair; masks never enter this operation.
            rgb_pair=torch.cat(images[:2])[None]
            raw_a=memo.original(rgb_pair);raw_b=memo.original(rgb_pair)
            feature_exact=bool(torch.equal(raw_a,raw_b));changed=int((raw_a!=raw_b).sum())
            if not feature_exact:raise AssertionError('Identical native RGB context produced different features')
            memo.cache[(0,1)]=raw_a;del raw_b
            for group,gold in enumerate(golds):
                p1=[memo.predict(j,[0],[gold]) for j in range(1,len(names))];p1_groups.append(p1)
                native.append(p1[0]);provenance.append(dict(kind='direct_1shot' if group==0 else 'shared_direct_alternative',hypothesis_id=f'h{len(native)-1:02d}'))
                for j in range(2,len(names)):
                    native.append(memo.predict(1,[j],[p1[j-1]]))
                    provenance.append(dict(kind='donor_two_hop',donor_slot=j-2,hypothesis_id=f'h{len(native)-1:02d}',
                        empty_donor_p1=not bool(p1[j-1].any()),caveat='Empty P1 is not semantic absence'))
            candidates=torch.stack(native);patches=[safe_downsample(m,64,64,downsample_mask).reshape(-1) for m in candidates]
            donor_components=torch.stack([torch.stack([safe_downsample(p1_groups[g][j-1],64,64,downsample_mask).cpu() for g in range(2)]) for j in range(2,len(names))])
            donor_union=donor_components.any(1)
            shared=dict(pair_id='counterfactual_fixed_first_pair',split='train',photo_ids=names,direct_index=0,
                reference_tokens=memo.tokens((0,1),0),query_tokens=memo.tokens((0,1),1),
                donor_tokens=torch.stack([memo.tokens((0,j),1) for j in range(2,len(names))]),
                donor_masks=donor_union,donor_masks_components=donor_components,donor_token_state='COMPLETE',
                candidate_masks=torch.stack(patches).reshape(len(patches),64,64).cpu(),candidate_state='COMPLETE',
                candidate_native_mask_bits=np.packbits(candidates.cpu().numpy().reshape(len(candidates),-1),axis=1),
                candidate_native_mask_size=model.image_size,candidate_provenance=provenance,
                token_context=dict(ordered_photos=names[:2],pair_specific=True),
                donor_token_context=[dict(ordered_photos=[names[0],name],pair_specific=True) for name in names[2:]],
                shared_feature_hashes={key:tensor_sha(value) for key,value in [('reference_tokens',memo.tokens((0,1),0)),('query_tokens',memo.tokens((0,1),1))]},
                disabled_metadata='No class/concept/source identities as model input; pair/provenance names are audit only.')
            # Shared donor physical tensors and union evidence are fixed. Scoring
            # uses each candidate's frozen source P1 component, not target labels.
            score_rows=[];reference_masks=[]
            replay_payload=dict(shared,e=0,c=pair['classes'][0],reference_mask=safe_downsample(golds[0],64,64,downsample_mask).cpu())
            s=replay_set(replay_payload,model,agglomerative_clustering,ImageSet,cluster_protos,DEV)
            for condition,gold in enumerate(golds):
                reference=safe_downsample(gold,64,64,downsample_mask).reshape(-1)
                s.gt64[0]=reference;reference_masks.append(reference.reshape(64,64).cpu())
                sources=[]
                for group in range(2):
                    sources.append([(0,reference)])
                    for j in range(2,len(names)):
                        sources.append([(j,donor_components[j-2,group].to(DEV).reshape(-1))])
                score_rows.append(score_candidates(s,patches,sources))
            # Bank and all GT-free score rows are frozen before reading QUERY labels.
            query_ann=torch.from_numpy(np.array(Image.open(annroot/str(Path(names[1]).with_suffix('.png'))))).to(DEV)
            labels=[]
            for c in pair['classes']:
                original=query_ann==c+1;host=load_mask(original,model.image_size,DEV)[0]
                iu,oiu=candidate_counts(candidates,host,original);labels.append((iu,oiu))
        memo.close();memo=None
        shared_path=out/'shared_pair.pt';tmp=shared_path.with_suffix('.tmp');torch.save(shared,tmp)
        if tmp.stat().st_size>a.max_bytes:tmp.unlink();raise RuntimeError('Shared pilot exceeds frozen byte cap')
        tmp.replace(shared_path);shared_sha=hashlib.sha256(shared_path.read_bytes()).hexdigest()
        native_ious=[]
        for condition,((iu,oiu),mask,scores) in enumerate(zip(labels,reference_masks,score_rows)):
            label=(iu[:,0].double()/iu[:,1].clamp_min(1)).float();native_ious.append(label.tolist())
            task=dict(e=condition,c=pair['classes'][condition],split='train',task_tag=f'condition{condition}',
                pair_id=shared['pair_id'],source_shared_path=str(shared_path),source_shared_sha256=shared_sha,
                reference_mask=mask,scalar_features=scores,
                scalar_feature_names=['rtg','cyc','link','fms','rtg*cyc','rtg*link','cyc*link','rtg*cyc*link'],
                candidate_iu=iu,candidate_model_iu=iu,candidate_iou=label,candidate_original_iu=oiu,
                candidate_original_iou=(oiu[:,0].double()/oiu[:,1].clamp_min(1)).float(),
                candidate_label_resolution='Native official1024-square IU; original annotation HxW IU separate; never 64IoU labels',
                disabled_metadata=report['forbidden_model_metadata'])
            path=out/f'task_{condition}.pt';torch.save(task,path)
            report['records'].append(dict(e=condition,c=pair['classes'][condition],split='train',path=str(shared_path),overlay_path=str(path),counterfactual=True,
                                         candidate_count=len(candidates),source_shared_sha256=shared_sha))
        report.update(state='COMPLETED',diagnostic=oracle_diagnostic(*native_ious),
            audit=dict(same_rgb_patch_features_exact=feature_exact,changed_feature_values=changed,
                candidate_union_shared=True,p1_union_shared=True,provenance_shared=True,donors_shared=True,direct_index_both=0,
                shared_tensor_file_count=1,finite_scalar_arrays=[bool(torch.isfinite(x).all()) for x in score_rows],
                support_masks_different=not bool(torch.equal(reference_masks[0],reference_masks[1])),
                total_bytes=sum(p.stat().st_size for p in out.glob('*.pt'))),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        save();print(json.dumps(dict(state=report['state'],elapsed_s=report['elapsed_s'],diagnostic=report['diagnostic'],audit=report['audit'])),flush=True)
    except BaseException as exc:
        report.update(state='ERROR',error=repr(exc));save();raise
    finally:
        if memo is not None:memo.close()


def self_check():
    torch.set_num_threads(1)
    for i in range(10):
        train=[dict(c=0,split='train',image_ids=['s','q','d1','d2','d3']),dict(c=4,split='train',image_ids=['s4','q4','a','b','c'])]
        held=[dict(c=48,split='dev',image_ids=['held_s','held_q','held_d'])]
        pair=dict(support='s',query='q',donors=['d1','d2','d3'],classes=[0,4])
        got,names=validate_pair(dict(minimal_recombined_examples=[pair]),dict(records=train+held));assert got==pair
        a=np.array([.8,.1,.7,.05]);b=np.array([.1,.9,.05,.7])
        assert oracle_diagnostic(a,b)['pair_teaches_prespecified_flip']
        assert not oracle_diagnostic(a,a)['pair_teaches_prespecified_flip']
        assert not oracle_diagnostic(a,np.zeros(4))['pair_teaches_prespecified_flip']
        swapped=dict(pair,donors=['d1','d2','held_s'])
        try:validate_pair(dict(minimal_recombined_examples=[swapped]),dict(records=train+held));raise AssertionError('Leak passed')
        except ValueError:pass
        rgb=torch.randn(2,16,8,generator=torch.Generator().manual_seed(i)).half()
        assert tensor_sha(rgb)==tensor_sha(rgb.clone())
    print('CPU10 synthetic checks passed: fixed first pair/new-train photo purge, same-RGB tensor hashes, valid/non-switching/unsupported oracle diagnostics; no pair performance selection.')


if __name__=='__main__':main()
