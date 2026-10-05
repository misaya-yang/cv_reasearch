#!/usr/bin/env python3
"""Capture a fixed top-eight SAM3 entity bank for a first identity-signal test.

This only collects observations. It does not implement RLGT pixel editing.
Reference labels are allowed; query annotations are never opened by capture.
The real model path, memory and throughput have not been tested.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def read_signal(a):
    """One frozen signal read; single-proposal counterfactuals are not masks."""
    import numpy as np
    from PIL import Image
    man = json.loads(a.manifest.read_text())
    episodes = [json.loads(s) for s in (a.out / 'observations.jsonl').read_text().splitlines()]
    if len(episodes) != 241:
        raise SystemExit('Task signal read requires all DEV241 episodes; smoke is capture only.')
    with np.load(a.out / 'slot_prior.npz', allow_pickle=False) as z:
        initial = z['query_embed'].copy()
    normalize = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    initial = normalize(initial)

    def bits(v, shape):
        return np.unpackbits(v, axis=-1)[..., :int(np.prod(shape))].reshape(*v.shape[:-1], *shape).astype(bool)

    def full(mask, shape):
        return np.asarray(Image.fromarray(mask.astype('uint8')).resize(shape[::-1], Image.Resampling.NEAREST)) > 0

    def meta(z, side, t):
        box = z[side + '_box'][t].astype(float)
        c = np.clip(z[side + '_confidence'][t], 1e-6, 1-1e-6)
        v = np.column_stack([box[:, :2], np.log(np.maximum(box[:, 2:], 1e-8)), np.log(c/(1-c))])
        return normalize((v-v.mean(0)) / np.maximum(v.std(0), 1e-12))

    parent = list(range(len(episodes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    photos = {}
    for i, r in enumerate(episodes):
        for path in (r['support'], r['query']):
            if path in photos: parent[find(i)] = find(photos[path])
            else: photos[path] = i
    roots = sorted({find(i) for i in range(len(episodes))})
    group = {root: n for n, root in enumerate(roots)}
    records, close_pairs, signs, host_ius = [], [], [], []
    valid_queries = eligible_prompts = no_evidence = 0
    for ei, r in enumerate(episodes):
        truth = np.asarray(Image.open(Path(man['annotation_root']) / Path(r['query']).with_suffix('.png'))) == r['c']+1
        with np.load(a.out / r['bank'], allow_pickle=False) as z:
            canvas = tuple(r['canvas'])
            host = full(bits(z['host'], canvas), truth.shape)
            source_shape = tuple(z['source_original_shape'])
            source_truth = bits(z['source_label'], source_shape)
            I, U = int((host & truth).sum()), int((host | truth).sum())
            host_ius.append((r['c'], I, U))
            base = I/max(U, 1)
            for t in range(8):
                sm = bits(z['source_masks'][t], canvas)
                qm = bits(z['query_masks'][t], canvas)
                source_full = [full(m, source_shape) for m in sm]
                area = np.asarray([int(m.sum()) for m in source_full])
                purity = np.asarray([int((m & source_truth).sum())/max(int(m.sum()), 1) for m in source_full])
                box = z['source_box'][t]
                legal = (area > 0) & np.isfinite(box).all(1) & (box[:, 2:] > 0).all(1) & (z['source_confidence'][t] >= .5)
                P, N = np.flatnonzero(legal & (purity > .5)), np.flatnonzero(legal & (purity <= .5))
                eligible_prompts += int(len(P) > 0 and len(N) > 0)
                ss, qs = z['source_slot'][t], z['query_slot'][t]
                fs = [normalize(z['source_token'][t]), normalize(z['source_pool'][t]), meta(z, 'source', t), initial[ss]]
                fq = [normalize(z['query_token'][t]), normalize(z['query_pool'][t]), meta(z, 'query', t), initial[qs]]
                prompt_rows = []
                for k, m in enumerate(qm):
                    boxq = z['query_box'][t, k]
                    if not m.any() or not np.isfinite(boxq).all() or not (boxq[2:] > 0).all():
                        continue
                    valid_queries += 1
                    pp, nn = P[ss[P] != qs[k]], N[ss[N] != qs[k]]
                    if not len(pp) or not len(nn):
                        no_evidence += 1
                        continue
                    score = np.asarray([float(q[k] @ (s[pp].mean(0)-s[nn].mean(0))) for s, q in zip(fs, fq)])
                    B = full(m, truth.shape)
                    added_tp = int((B & ~host & truth).sum()); added_fp = int((B & ~host & ~truth).sum())
                    removed_tp = int((B & host & truth).sum()); removed_fp = int((B & host & ~truth).sum())
                    up = (I+added_tp)/max(U+added_fp, 1)-base
                    down = (I-removed_tp)/max(U-removed_fp, 1)-base
                    ar = int(B.sum())
                    rec = dict(group=group[find(ei)], episode=ei, fold=r['fold'], prompt=t, slot=int(qs[k]),
                               scores=score, q=float(z['query_confidence'][t,k]), overlap=float((B & host).sum())/max(ar,1),
                               action=int(up >= down), fixable=max(up, down) > 0,
                               purity=(added_tp+removed_tp)/max(ar,1), error_overlap=added_tp+removed_fp)
                    records.append(rec); prompt_rows.append(rec)
                    delta = up if score[0] > 0 else down if score[0] < 0 else 0
                    signs.append(dict(episode=ei, fold=r['fold'], delta=delta,
                                      corrected_fn=added_tp if score[0] > 0 else 0, new_fp=added_fp if score[0] > 0 else 0,
                                      corrected_fp=removed_fp if score[0] < 0 else 0, removed_tp=removed_tp if score[0] < 0 else 0))
                for fg in prompt_rows:
                    if fg['purity'] <= .5: continue
                    for bg in prompt_rows:
                        if bg['purity'] > .5 or abs(fg['q']-bg['q']) > .02 or not (fg['error_overlap'] or bg['error_overlap']): continue
                        close_pairs.append((fg['group'], (fg['scores'] > bg['scores']).astype(float)+.5*(fg['scores'] == bg['scores'])))

    names = ['TOKEN', 'POOL', 'META', 'INIT_DIAGNOSTIC']
    report = dict(state='SIGNAL_READ', dataset='COCO-20i', split='DEV241', seed=0, episodes=241, groups=len(roots),
                  scope='Observed identity-signal test; no combined predicted mask or method mIoU',
                  proposal_scope='Source nonempty, finite positive box, confidence>=.5; exclude actual same decoder ID; query nonempty finite box',
                  eligible_prompts=eligible_prompts, possible_prompts=241*8, valid_query_proposals=valid_queries,
                  scored_query_proposals=len(records), no_evidence=no_evidence,
                  null_scope='INIT diagnoses a fixed learned slot prior. Rank/prompt permutations are held pending interpretation; no semantic-null claim.',
                  action_signal={}, raw_sign={})
    classes = sorted({v[0] for v in host_ius})
    report['host_miou'] = 100*sum(sum(v[1] for v in host_ius if v[0]==c)/max(sum(v[2] for v in host_ius if v[0]==c),1)
                                for c in classes)/len(classes)
    report['bootstrap_draws'] = 2000
    if not records:
        report['action_signal']['reason'] = 'No legal source P/N evidence; no information ceiling inferred.'
    else:
        q = np.asarray([r['q'] for r in records]); overlap = np.asarray([r['overlap'] for r in records])
        # Equal-valued observations stay together; quantile ties may collapse bins.
        bins = np.searchsorted(np.quantile(q, np.arange(1,10)/10), q, side='right')*10
        bins += np.searchsorted(np.quantile(overlap, np.arange(1,10)/10), overlap, side='right')
        G = len(roots)
        denominator, numerator = np.zeros((G,G)), np.zeros((4,G,G))
        for b in np.unique(bins):
            ix = [i for i,r in enumerate(records) if bins[i] == b and r['fixable']]
            plus = [i for i in ix if records[i]['action']]; minus = [i for i in ix if not records[i]['action']]
            if not plus or not minus: continue
            gp = np.asarray([records[i]['group'] for i in plus]); gn = np.asarray([records[i]['group'] for i in minus])
            vp = np.asarray([records[i]['scores'] for i in plus]); vn = np.asarray([records[i]['scores'] for i in minus])
            np.add.at(denominator, (gp[:,None],gn[None,:]), 1)
            for arm in range(4):
                wins = (vp[:,arm,None] > vn[None,:,arm]).astype(float)+.5*(vp[:,arm,None] == vn[None,:,arm])
                np.add.at(numerator[arm], (gp[:,None],gn[None,:]), wins)
        rng = np.random.default_rng(0)
        weights = rng.multinomial(G, np.ones(G)/G, size=2000).astype(float)
        den = np.einsum('bi,ij,bj->b', weights, denominator, weights, optimize=True)
        boot = []
        for ai, name in enumerate(names):
            v = np.einsum('bi,ij,bj->b', weights, numerator[ai], weights, optimize=True)
            v = np.divide(v,den,out=np.full_like(v,np.nan),where=den>0); boot.append(v)
            point = float(numerator[ai].sum()/denominator.sum()) if denominator.sum() else None
            report['action_signal'][name] = dict(cauc=point, ci95=np.nanpercentile(v,[2.5,97.5]).tolist() if np.isfinite(v).any() else None,
                                                 pairs=int(denominator.sum()), usable_bootstrap_draws=int(np.isfinite(v).sum()))
        for ai,name in enumerate(names[1:],1):
            delta = boot[0]-boot[ai]
            report['action_signal']['TOKEN_minus_'+name] = dict(
                estimate=report['action_signal']['TOKEN']['cauc']-report['action_signal'][name]['cauc'] if denominator.sum() else None,
                ci95=np.nanpercentile(delta,[2.5,97.5]).tolist() if np.isfinite(delta).any() else None)
        report['action_signal']['conditioning'] = '10x10 label-free confidence/host-overlap quantiles; useful-action directions only; not safety of arbitrary proposals'
        if close_pairs:
            counts = np.zeros(G); wins = np.zeros((G,4))
            for gi,v in close_pairs: counts[gi]+=1;wins[gi]+=v
            d = weights @ counts
            report['near_confidence_pairs'] = {}
            for ai,name in enumerate(names):
                v = np.divide(weights @ wins[:,ai], d, out=np.full(2000,np.nan), where=d>0)
                report['near_confidence_pairs'][name] = dict(accuracy=float(wins[:,ai].sum()/counts.sum()), pairs=len(close_pairs),
                                                            ci95=np.nanpercentile(v,[2.5,97.5]).tolist())
    delta = [r['delta'] for r in signs]
    report['raw_sign'] = dict(beneficial=sum(x>0 for x in delta), harmful=sum(x<0 for x in delta), neutral=sum(x==0 for x in delta),
                             mean_single_proposal_iou_delta=float(np.mean(delta)) if delta else None,
                             median_single_proposal_iou_delta=float(np.median(delta)) if delta else None,
                             pixel_counts={k:sum(r[k] for r in signs) for k in ['corrected_fn','new_fp','corrected_fp','removed_tp']},
                             fold_mean_single_proposal_delta={f:float(np.mean([r['delta'] for r in signs if r['fold']==f]))
                                                             for f in range(4) if any(r['fold']==f for r in signs)},
                             scope='All legal evidence, including both-harmful/neutral proposals; counts sum isolated operations, may repeat pixels; not combined mIoU')
    if signs:
        sums, counts = np.zeros(len(roots)), np.zeros(len(roots))
        for r in signs:
            gi = group[find(r['episode'])]; sums[gi] += r['delta']; counts[gi] += 1
        weights = np.random.default_rng(0).multinomial(len(roots), np.ones(len(roots))/len(roots), size=2000)
        den = weights @ counts
        draws = np.divide(weights @ sums, den, out=np.full(2000,np.nan), where=den>0)
        report['raw_sign']['ci95_mean_single_proposal_delta'] = np.nanpercentile(draws,[2.5,97.5]).tolist()
    (a.out / 'signal.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score', action='store_true')
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--names', required=True, type=Path)
    p.add_argument('--host-names', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--stitch', required=True, type=Path)
    p.add_argument('--sam3', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--limit', type=int, default=241)
    p.add_argument('--memory-fraction', type=float, default=.4)
    a = p.parse_args()
    if a.score:
        read_signal(a)
        return
    if os.environ.get('DEMO9_CUDA_GUARD') != '1':
        raise SystemExit('Run capture only through the authorized resource guard.')

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import v2

    spec = importlib.util.spec_from_file_location('entity_probe_stitch', a.stitch)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    sys.path.insert(0, str(a.sam3))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.data_misc import FindStage

    man = json.loads(a.manifest.read_text())
    names = {tuple(r['key']): r for r in map(json.loads, a.names.read_text().splitlines())}
    old_root = a.run / 'reference_use/open_fast'
    old = {tuple(r['key']): r for r in map(json.loads, (old_root / 'variant.jsonl').read_text().splitlines())}
    host_root = a.run / 'reference_use/sem_name'
    hosts = {tuple(r['key']): r for r in map(json.loads, a.host_names.read_text().splitlines())}
    recs = [r for f in sorted(a.run.glob('predictions_shard*.jsonl'))
            for r in map(json.loads, f.read_text().splitlines()) if r.get('candidate_file')][:a.limit]
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    model = build_sam3_image_model(
        bpe_path=str(a.sam3 / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz'), device='cuda',
        checkpoint_path=str(a.checkpoint), load_from_HF=False, eval_mode=True,
        enable_inst_interactivity=False, compile=False).float().eval()
    S.configure_fp32_mlp_backend(model)
    proc = Sam3Processor(model, device='cuda', confidence_threshold=.5)
    instance_seen = {}
    def observe_instance(_module, _inputs, output):
        instance_seen['value'] = output
    model.segmentation_head.instance_seg_head.register_forward_hook(observe_instance)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'bank').mkdir(exist_ok=True)
    journal = a.out / 'observations.jsonl'
    done = {tuple(r['key']) for r in map(json.loads, journal.read_text().splitlines())} if journal.exists() else set()
    np.savez(a.out / 'slot_prior.npz',
             query_embed=model.transformer.decoder.query_embed.weight.detach().float().cpu().numpy(),
             reference_points=model.transformer.decoder.reference_points.weight.detach().sigmoid().float().cpu().numpy())

    def unpack(path, shape):
        n, h, w = shape
        with np.load(path, allow_pickle=False) as z:
            return np.unpackbits(z['proposal_query'], axis=1)[:, :h*w].reshape(n, h, w).astype(bool)

    def ground(backbone, text, text_index, canvas):
        stage = FindStage(img_ids=torch.zeros(1, device='cuda', dtype=torch.long),
                          text_ids=torch.tensor([text_index], device='cuda'),
                          input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                          input_points=None, input_points_mask=None)
        bo = {**backbone, **text}
        prompt, pm, bo = model._encode_prompt(bo, stage, model._get_dummy_prompt(1))
        bo, enc, _ = model._run_encoder(bo, stage, prompt, pm)
        out = {'encoder_hidden_states': enc['encoder_hidden_states'],
               'prev_encoder_out': {'encoder_out': enc, 'backbone_out': bo}}
        out, hs = model._run_decoder(memory=out['encoder_hidden_states'], pos_embed=enc['pos_embed'],
                                    src_mask=enc['padding_mask'], out=out, prompt=prompt,
                                    prompt_mask=pm, encoder_out=enc)
        image_ids = stage.img_ids if bo.get('id_mapping') is None else bo['id_mapping'][stage.img_ids]
        instance_seen.clear()
        model._run_segmentation_heads(out=out, backbone_out=bo, img_ids=image_ids,
                                     vis_feat_sizes=enc['vis_feat_sizes'], encoder_hidden_states=out['encoder_hidden_states'],
                                     prompt=prompt, prompt_mask=pm, hs=hs)
        # Keep the existing host's score convention; also save its two factors.
        logits = out['pred_logits'][0, :, 0]
        presence = out['presence_logit_dec'][0].reshape(-1)[0]
        prob = logits.sigmoid() * presence.sigmoid()
        assert out['queries'].shape[:2] == out['pred_masks'].shape[:2] == out['pred_logits'].shape[:2]
        slots = prob.argsort(descending=True, stable=True)[:S.KEEP]
        assert torch.isfinite(out['queries'][0, slots]).all() and torch.isfinite(prob[slots]).all()
        masks = out['pred_masks'][0, slots].float()
        binary = F.interpolate(masks[:, None], canvas, mode='bilinear', align_corners=False)[:, 0] > 0
        sem = F.interpolate(out['semantic_seg'].reshape(1, 1, *out['semantic_seg'].shape[-2:]).float(),
                            canvas, mode='bilinear', align_corners=False)[0, 0] > 0
        instance = instance_seen.pop('value')
        # The official pixel head removes the batch axis for a single image.
        if instance.ndim == 4:
            assert instance.shape[0] == 1
            instance = instance[0]
        assert instance.ndim == 3
        gh, gw = instance.shape[-2:]
        memory = instance.float().flatten(1).T
        weights = F.interpolate(binary[:, None].float(), (gh, gw), mode='area')[:, 0].flatten(1)
        dense = weights @ memory / weights.sum(1, keepdim=True).clamp(min=1e-12)
        assert torch.isfinite(dense).all()
        cpu = lambda x: x.detach().cpu().numpy()
        box = out['pred_boxes'][0, slots].float()
        xyxy = torch.cat([box[:, :2] - box[:, 2:] / 2, box[:, :2] + box[:, 2:] / 2], dim=1)
        return dict(token=cpu(out['queries'][0, slots].float()), pool=cpu(dense),
                    slot=cpu(slots), box=cpu(out['pred_boxes'][0, slots].float()),
                    box_xyxy=cpu(xyxy), empty=cpu(binary.flatten(1).sum(1) == 0),
                    logit=cpu(logits[slots]), presence=float(presence), confidence=cpu(prob[slots]),
                    logit_sigmoid=cpu(logits[slots].sigmoid()), presence_sigmoid=float(presence.sigmoid()),
                    area=cpu(binary.flatten(1).sum(1)),
                    masks=np.packbits(cpu(binary).reshape(S.KEEP, -1), axis=1),
                    semantic=np.packbits(cpu(sem).reshape(-1)))

    start = time.monotonic()
    with torch.inference_mode(), journal.open('a') as stream:
        for r in recs:
            key = tuple(r[k] for k in ('fold', 'e', 'c'))
            if key in done:
                continue
            begin = time.monotonic()
            words = [v[0] for v in names[key]['rerank'][:8]]
            assert len(words) == 8 and len(set(words)) == 8
            text = model.backbone.forward_text(words, device='cuda')
            canvas = tuple(r['proposal_shape'][1:])
            data = {}
            for side, path_key in [('source', 'support'), ('query', 'query')]:
                image = Image.open(Path(man['data_root']) / r[path_key]).convert('RGB')
                features = model.backbone.forward_image(proc.transform(v2.functional.to_image(image).to('cuda')).unsqueeze(0))
                bank = [ground(features, text, i, canvas) for i in range(8)]
                for name in bank[0]:
                    data[side + '_' + name] = np.asarray([x[name] for x in bank])
                if side == 'source':
                    mask = np.asarray(Image.open(Path(man['annotation_root']) / Path(r['support']).with_suffix('.png'))) == r['c'] + 1
                    data['source_label'] = np.packbits(mask)
                    data['source_original_shape'] = np.asarray(mask.shape)
                del features
            raw = unpack(a.run / r['candidate_file'], r['proposal_shape'])
            meta = r['proposal_metadata']
            valid = [i for i, m in enumerate(meta) if m[1] > 0]
            top = max((meta[i][0] for i in valid), default=0.)
            keep = [i for i in valid if meta[i][0] >= .7 * top]
            visual = raw[keep].any(0) if keep else np.zeros(canvas, bool)
            nr = old[key]
            named = unpack(old_root / nr['candidate_file'], nr['proposal_shape'])[nr['semantic_row']]
            data['visual'] = np.packbits(visual)
            data['legacy_host'] = np.packbits(named if nr['query_top_score'] > top else visual)
            hr = hosts[key]
            hi = max(range(len(hr['rerank'])), key=lambda i: hr['rerank'][i][4]*hr['rerank'][i][5])
            hm = unpack(host_root / hr['candidate_file'], hr['proposal_shape'])[hr['semantic_row']+hi]
            data['host'] = np.packbits(hm if hr['rerank'][hi][3] > top else visual)
            file = 'bank/%d_%d_%d.npz' % key
            np.savez_compressed(a.out / file, **data)
            row = dict(key=list(key), fold=key[0], e=key[1], c=key[2], support=r['support'], query=r['query'],
                       words=words, canvas=list(canvas), bank=file, seconds=time.monotonic()-begin,
                       peak_memory_bytes=torch.cuda.max_memory_allocated(), query_annotation_opened=False,
                       model_precision='fp32; single prompt per decoder; full image on both sides',
                       pool_field='same-pass segmentation_head.instance_seg_head output',
                       slot_field='actual decoder query ID, not top20 confidence rank',
                       host_arm='sem_name:routed_by_semantic',
                       prediction_rule_implemented=False)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            done.add(key)
            print(json.dumps(dict(episodes=len(done), key=list(key), seconds=row['seconds'],
                                  peak_memory_bytes=row['peak_memory_bytes'], query_annotation_opened=False)), flush=True)
    summary = dict(state='CAPTURE_COMPLETED', episodes=len(done), elapsed_seconds=time.monotonic()-start,
                   query_annotation_opened=False, prediction_rule_implemented=False)
    # Different completion files let the guard resume the same bank after smoke.
    (a.out / ('capture.json' if a.limit == 241 else 'smoke.json')).write_text(json.dumps(summary)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
