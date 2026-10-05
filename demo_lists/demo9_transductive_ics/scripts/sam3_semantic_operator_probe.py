#!/usr/bin/env python3
"""DEV measurement of source semantic-operator alignment; no predicate selector.

Capture source exact-100 without proposal pruning and query outputs for at most
six selected words. The existing two-route host
is read from its frozen cache. --score is a separate CPU operation. No query
annotation is opened during capture. This is not a tested method.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score', action='store_true')
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--stitch', required=True, type=Path)
    p.add_argument('--sam3', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--vocabulary', type=Path)
    p.add_argument('--limit', type=int, default=241)
    p.add_argument('--memory-fraction', type=float, default=.4)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('operator_probe_stitch', a.stitch)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    import numpy as np
    from PIL import Image

    def unpack(path, shape):
        n, h, w = shape
        with np.load(path, allow_pickle=False) as z:
            return np.unpackbits(z['proposal_query'], axis=1)[:, :h*w].reshape(n, h, w).astype(bool)

    man = json.loads(a.manifest.read_text())
    if a.score:
        recs = [json.loads(s) for s in (a.out/'observations.jsonl').read_text().splitlines()]
        arms = ('H', 'S', 'A', 'best_H_episode', 'best_A_episode')
        for r in recs:
            # Only this CPU stage opens a query annotation, after capture ends.
            truth = np.asarray(Image.open(Path(man['annotation_root']) / Path(r['query']).with_suffix('.png'))) == r['c']+1
            with np.load(a.out/r['mask_file'], allow_pickle=False) as z:
                h, w = r['canvas_shape']
                maps = np.unpackbits(z['query'], axis=1)[:, :h*w].reshape(-1, h, w).astype(bool)
                visual = np.unpackbits(z['visual'])[:h*w].reshape(h, w).astype(bool)
                named = np.unpackbits(z['named'])[:h*w].reshape(h, w).astype(bool)
            by_id = dict(zip(r['query_mask_word_ids'], maps))
            sem = by_id[r['t_sem']]
            masks = dict(H=named if r['cached_named_conf'] > r['visual_conf'] else visual,
                         S=sem, A=sem if r['query_conf'][str(r['t_sem'])] > r['visual_conf'] else visual)
            def full_iu(mask):
                full = np.asarray(Image.fromarray(mask.astype('uint8')).resize(truth.shape[::-1], Image.Resampling.NEAREST)) > 0
                return [int((full & truth).sum()), int((full | truth).sum())]
            visual_iu, named_iu, sem_iu = full_iu(visual), full_iu(named), full_iu(sem)
            ratio = lambda v:v[0]/max(v[1],1)
            masks['best_H_episode'] = named if ratio(named_iu)>ratio(visual_iu) else visual
            masks['best_A_episode'] = sem if ratio(sem_iu)>ratio(visual_iu) else visual
            r['iu'] = {}
            for arm, mask in masks.items():
                r['iu'][arm] = full_iu(mask)
        report = dict(dataset='COCO-20i', split='exposed DEV', episodes=len(recs), seed=0,
                      confidence_weight='legacy best-j confidence, not max proposal confidence',
                      method_status='Operator measurement only; no witness/program arm implemented',
                      diagnostic_scope='best_*_episode uses labels per episode; it is not a pooled class-mIoU upper bound', arms={})
        for arm in arms:
            row = S.paired(recs, lambda r: r['iu'][arm], lambda r: r['iu'][arm], draws=1)
            report['arms'][arm] = {'miou': row['miou']}
        for arm, base in [('A','H'), ('S','H')]:
            d = S.paired(recs, lambda r: r['iu'][arm], lambda r: r['iu'][base])
            d['fold_gain'] = [S.paired([r for r in recs if r['fold']==f], lambda r:r['iu'][arm], lambda r:r['iu'][base], draws=1)['gain'] for f in sorted({r['fold'] for r in recs})]
            delta = [r['iu'][arm][0]/max(r['iu'][arm][1],1)-r['iu'][base][0]/max(r['iu'][base][1],1) for r in recs]
            d.update(up=sum(x>0 for x in delta), down=sum(x<0 for x in delta), drops_over_10=sum(x<-.1 for x in delta))
            report['arms'][arm]['over_'+base] = d
        (a.out/'readout.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    if os.environ.get('DEMO9_CUDA_GUARD') != '1':
        raise SystemExit('Capture must run through experiment_resource_guard.py.')
    if not all((a.sam3, a.checkpoint, a.vocabulary)):
        raise SystemExit('Capture requires --sam3, --checkpoint, --vocabulary.')
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import v2
    sys.path.insert(0, str(a.sam3))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.data_misc import FindStage
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    model = build_sam3_image_model(bpe_path=str(a.sam3/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'), device='cuda',
                                  checkpoint_path=str(a.checkpoint), load_from_HF=False,
                                  eval_mode=True, enable_inst_interactivity=False, compile=False).float().eval()
    S.configure_fp32_mlp_backend(model)
    proc = Sam3Processor(model, device='cuda', confidence_threshold=.5)
    vocab = json.loads(a.vocabulary.read_text())
    named_root = a.run/'reference_use/open_fast'
    named_rows = {tuple(r['key']):r for r in map(json.loads, (named_root/'variant.jsonl').read_text().splitlines())}
    recs = [r for f in sorted(a.run.glob('predictions_shard*.jsonl')) for r in map(json.loads,f.read_text().splitlines()) if r.get('candidate_file')][:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'masks').mkdir(exist_ok=True)
    journal = a.out/'observations.jsonl'
    done = {tuple(r['key']) for r in map(json.loads,journal.read_text().splitlines())} if journal.exists() else set()
    start = time.monotonic()
    with torch.inference_mode(), journal.open('a') as stream:
        parts = [model.backbone.forward_text(vocab[i:i+64],device='cuda') for i in range(0,len(vocab),64)]
        keys = [k for k,v in parts[0].items() if torch.is_tensor(v)]
        bdim = {k:(1 if parts[0][k].shape[0]!=len(vocab[:64]) else 0) for k in keys}
        text = {k:torch.cat([p[k] for p in parts],dim=bdim[k]) for k in keys}

        def features(image):
            return model.backbone.forward_image(proc.transform(v2.functional.to_image(image).to('cuda')).unsqueeze(0))

        def ground(b, ids, masks):
            sel = torch.tensor(ids,device='cuda')
            bo = {**b, **{k:text[k].index_select(bdim[k],sel) for k in keys}}
            stage = FindStage(img_ids=torch.zeros(len(ids),device='cuda',dtype=torch.long),
                              text_ids=torch.arange(len(ids),device='cuda'),input_boxes=None,
                              input_boxes_mask=None,input_boxes_label=None,input_points=None,input_points_mask=None)
            prompt, pm, bo = model._encode_prompt(bo,stage,model._get_dummy_prompt(len(ids)))
            bo, enc, _ = model._run_encoder(bo,stage,prompt,pm)
            o = {'encoder_hidden_states':enc['encoder_hidden_states'],'prev_encoder_out':{'encoder_out':enc,'backbone_out':bo}}
            o, hs = model._run_decoder(memory=o['encoder_hidden_states'],pos_embed=enc['pos_embed'],src_mask=enc['padding_mask'],out=o,
                                      prompt=prompt,prompt_mask=pm,encoder_out=enc)
            prob = (o['pred_logits'].sigmoid()*o['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
            if not masks: return prob, None, None
            image_ids = stage.img_ids if bo.get('id_mapping') is None else bo['id_mapping'][stage.img_ids]
            model._run_segmentation_heads(out=o,backbone_out=bo,img_ids=image_ids,vis_feat_sizes=enc['vis_feat_sizes'],
                                         encoder_hidden_states=o['encoder_hidden_states'],prompt=prompt,prompt_mask=pm,hs=hs)
            sem = o['semantic_seg'].reshape(len(ids),*o['semantic_seg'].shape[-2:])
            return prob, o['pred_masks'], sem

        for r in recs:
            key = tuple(r[k] for k in ('fold','e','c'))
            if key in done: continue
            begin = time.monotonic(); old = named_rows[key]
            ref, query = (Image.open(Path(man['data_root'])/r[k]).convert('RGB') for k in ('support','query'))
            mask = np.asarray(Image.open(Path(man['annotation_root'])/Path(r['support']).with_suffix('.png'))) == r['c']+1
            with torch.autocast('cuda',dtype=torch.bfloat16): rb = features(ref)
            # Same ROI and pooled coarse grid as the existing open_fast source.
            L = rb['backbone_fpn'][-1].shape[-1]; H,W = mask.shape
            ys,xs = np.nonzero(mask)
            cx0,cx1,cy0,cy1 = xs.min()/W*L,(xs.max()+1)/W*L,ys.min()/H*L,(ys.max()+1)/H*L
            mx,my = max(2.,.2*(cx1-cx0)),max(2.,.2*(cy1-cy0))
            x0,x1,y0,y1 = max(0,int(cx0-mx)),min(L,int(np.ceil(cx1+mx))),max(0,int(cy0-my)),min(L,int(np.ceil(cy1+my)))
            cut = lambda f:f[...,y0*(f.shape[-2]//L):y1*(f.shape[-2]//L),x0*(f.shape[-1]//L):x1*(f.shape[-1]//L)]
            rb = dict(rb); rb['backbone_fpn']=[cut(f) for f in rb['backbone_fpn']];rb['vision_pos_enc']=[cut(f) for f in rb['vision_pos_enc']]
            if torch.is_tensor(rb.get('vision_features')):rb['vision_features']=rb['backbone_fpn'][-1]
            mask=mask[int(y0/L*H):int(np.ceil(y1/L*H)),int(x0/L*W):int(np.ceil(x1/L*W))]
            cb=dict(rb);cb['backbone_fpn']=[F.avg_pool2d(f,3) for f in rb['backbone_fpn']]
            cb['vision_pos_enc']=[q[...,1::3,1::3][...,:f.shape[-2],:f.shape[-1]] for q,f in zip(rb['vision_pos_enc'],cb['backbone_fpn'])]
            if torch.is_tensor(cb.get('vision_features')):cb['vision_features']=cb['backbone_fpn'][-1]
            coarse=torch.zeros(len(vocab),device='cuda')
            for i in range(0,len(vocab),128):
                ids=list(range(i,min(i+128,len(vocab))))
                with torch.autocast('cuda',dtype=torch.bfloat16):coarse[ids]=ground(cb,ids,False)[0].float().max(1).values
            shortlist=coarse.topk(100).indices.tolist(); stats={}; sm={}
            for i in range(0,100,32):
                ids=shortlist[i:i+32]
                with torch.autocast('cuda',dtype=torch.bfloat16):prob,prop,sem=ground(rb,ids,True)
                prob=prob.float(); prop=prop>0; sem=sem>0
                target=F.interpolate(torch.from_numpy(mask).float().cuda()[None,None],prop.shape[-2:],mode='area')[0,0]>.5
                inter=(prop&target).flatten(2).sum(2).float();iou=inter/(prop.flatten(2).sum(2)+target.sum()-inter).clamp(min=1)
                best=(prob*iou).argmax(1)
                st=F.interpolate(torch.from_numpy(mask).float().cuda()[None,None],sem.shape[-2:],mode='area')[0,0]>.5
                for k,t in enumerate(ids):
                    area=int(sem[k].sum());it=int((sem[k]&st).sum());j=int(best[k]); cf=float(prob[k,j]);pi=float(iou[k,j])
                    si=it/max(area+int(st.sum())-it,1)
                    stats[t]=dict(word=vocab[t],top_conf=float(prob[k].max()),best_j_conf=cf,best_proposal_iou=pi,
                                  semantic_iou=si,semantic_area=area,semantic_intersection=it,semantic_false_positive=area-it)
                    sm[t]=sem[k].cpu().numpy()
            t_sem=min(shortlist,key=lambda t:(-stats[t]['best_j_conf']*stats[t]['semantic_iou'],t))
            positive=sorted(shortlist,key=lambda t:(-stats[t]['best_j_conf']*stats[t]['semantic_iou'],t))[:3]
            envelope=np.logical_or.reduce([sm[t] for t in positive]); st_cpu=st.cpu().numpy()
            excess=envelope&~st_cpu
            neg_score={t:stats[t]['best_j_conf']*int((sm[t]&excess).sum())/max(stats[t]['semantic_area'],1) for t in shortlist if t not in positive}
            negative=sorted([t for t in neg_score if neg_score[t]>0],key=lambda t:(-neg_score[t],t))[:3]
            selected=sorted(set(positive+negative));qm=[];qc={};qb=None
            _,_,width,height=S.rectangles()[1]
            for t in selected:
                if vocab[t]==old['name']:
                    qm.append(unpack(named_root/old['candidate_file'],old['proposal_shape'])[old['semantic_row']])
                    qc[t]=old['query_top_score']
                else:
                    if qb is None:qb=features(query)
                    prob,_,sem=ground(qb,[t],True);qc[t]=float(prob.max())
                    qm.append((F.interpolate(sem[:,None].float(),(height,width),mode='bilinear',align_corners=False)[0,0]>0).cpu().numpy())
            raw=unpack(a.run/r['candidate_file'],r['proposal_shape']);md=r['proposal_metadata'];valid=[j for j in range(len(md)) if md[j][1]>0]
            visual_conf=max((md[j][0] for j in valid),default=0.)
            visual=raw[[j for j in valid if md[j][0]>=.7*visual_conf]].any(0) if valid else np.zeros((height,width),bool)
            native_named=unpack(named_root/old['candidate_file'],old['proposal_shape'])[old['semantic_row']]
            file='masks/%d_%d_%d.npz'%key
            np.savez_compressed(a.out/file,query=np.packbits(np.asarray(qm).reshape(len(selected),-1),axis=1),
                                source=np.packbits(np.asarray([sm[t] for t in sorted(set(positive+negative))]).reshape(len(set(positive+negative)),-1),axis=1),
                                visual=np.packbits(visual),named=np.packbits(native_named))
            row=dict(key=list(key),fold=key[0],e=key[1],c=key[2],support=r['support'],query=r['query'],source=stats,
                     shortlist=shortlist,coarse_conf={t:float(coarse[t]) for t in shortlist},query_conf=qc,t_sem=t_sem,
                     positive=positive,negative=negative,negative_scores=neg_score,source_shape=list(st_cpu.shape),
                     positive_envelope_area=int(envelope.sum()),positive_excess_area=int(excess.sum()),source_mask_word_ids=sorted(set(positive+negative)),
                     query_mask_word_ids=selected,canvas_shape=[height,width],mask_file=file,visual_conf=visual_conf,
                     cached_named_conf=old['query_top_score'],cached_name=old['name'],seconds=time.monotonic()-begin,
                     query_annotation_opened=False,witness_selector_implemented=False)
            stream.write(json.dumps(row)+'\n');stream.flush();done.add(key)
            print(json.dumps(dict(episodes=len(done),key=list(key),seconds=row['seconds'],query_masks=len(selected),t_sem=vocab[t_sem],query_annotation_opened=False)),flush=True)
    print(json.dumps(dict(state='CAPTURE_COMPLETED',episodes=len(done),elapsed_s=time.monotonic()-start,query_annotation_opened=False)))


if __name__ == '__main__':
    main()
