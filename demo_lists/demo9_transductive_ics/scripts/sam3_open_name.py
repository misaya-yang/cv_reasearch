#!/usr/bin/env python3
"""Name the reference with SAM3 itself from a general vocabulary, then segment the query with that name (GPU).

  python scripts/sam3_open_name.py --run RUN/dev --manifest SUITE/dev_episodes.json --vocabulary V.json --sam3 SRC --checkpoint CKPT --out RUN/dev/reference_use

Measured: SAM3 recognises the reference object's category with high confidence when the right name is offered (80 COCO
names: the true one is ranked first in 95% of 77 episodes, confidence 0.95 against 0.09 for the best wrong name). Here
the vocabulary is general (LVIS, 1203 names; nothing about the benchmark's classes or folds). Per episode:
  1. every name is given to SAM3 on the reference image (names batched; the image features are computed once); a name's
     score is SAM3's confidence x IoU with the reference mask, for its best proposal;
  2. the best name is given to SAM3 on the query alone, at full input size (variant `open_name`).
No class label, no query label, no training. The true class is written to the row for reporting only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--vocabulary", required=True)
    p.add_argument("--sam3", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--limit", type=int)
    p.add_argument("--k", type=int, default=5); p.add_argument("--batch", type=int, default=16); p.add_argument("--prune", type=int, default=1); p.add_argument("--pool", type=int, default=1); p.add_argument("--roi", type=int, default=0); p.add_argument("--shortlist", type=int, default=0); p.add_argument("--shortlist-pool", type=int, default=3); p.add_argument("--crop", type=int, default=0); p.add_argument("--amp", type=int, default=0); p.add_argument("--name", default="open_name"); p.add_argument("--rerank", type=int, default=1); p.add_argument("--semantic-score", type=int, default=0); p.add_argument("--skip", type=int, default=0)
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import v2
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    recs = [r for q in sorted(Path(a.run).glob("predictions_shard*.jsonl")) for r in map(json.loads, q.read_text().splitlines()) if r.get("candidate_file")][a.skip:][:a.limit]
    vocab = json.loads(Path(a.vocabulary).read_text())
    out = Path(a.out) / a.name; (out / "candidates").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.sam3))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    model = build_sam3_image_model(bpe_path=str(Path(a.sam3) / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"), device="cuda",
                                   checkpoint_path=str(a.checkpoint), load_from_HF=False,
                                   eval_mode=True, enable_inst_interactivity=False, compile=False).float().eval()
    S.configure_fp32_mlp_backend(model)
    proc = Sam3Processor(model, device="cuda", confidence_threshold=.5)
    t = S.rectangles()[1]
    from sam3.model.data_misc import FindStage
    start = time.monotonic()
    with open(out / "variant.jsonl", "w") as stream, torch.inference_mode():
        parts = [model.backbone.forward_text(vocab[i:i + 64], device="cuda") for i in range(0, len(vocab), 64)]
        keys = [k for k, v in parts[0].items() if torch.is_tensor(v)]
        print("text output:", {k: tuple(parts[0][k].shape) for k in keys}, flush=True)
        bdim = {k: (1 if parts[0][k].shape[0] != len(vocab[:64]) else 0) for k in keys}  # which axis is the batch of names
        text_all = {k: torch.cat([q[k] for q in parts], dim=bdim[k]) for k in keys}
        print("vocabulary %d names encoded, %.1f s" % (len(vocab), time.monotonic() - start), flush=True)
        feats = lambda image: model.backbone.forward_image(proc.transform(v2.functional.to_image(image).to("cuda")).unsqueeze(0))

        def pooled(b, k):
            """The naming step on a coarser grid: the backbone runs at full size (its position code is fixed to it); its
            feature maps are averaged over k x k cells before the per-name encoder and decoder, which take any grid."""
            if k == 1:
                return b
            b = dict(b)
            b["backbone_fpn"] = [F.avg_pool2d(f, k) for f in b["backbone_fpn"]]
            b["vision_pos_enc"] = [q[..., k // 2::k, k // 2::k][..., :f.shape[-2], :f.shape[-1]] for q, f in zip(b["vision_pos_enc"], b["backbone_fpn"])]
            if torch.is_tensor(b.get("vision_features")):
                b["vision_features"] = b["backbone_fpn"][-1]
            return b

        clock = dict(decode=0.0, masks=0.0, skipped=0, batches=0)

        def ground(b, idx, floor=None):
            """SAM3 with each of the names idx on one image: [names, queries] confidences; masks and semantic maps unless no
            confidence exceeds floor (a name's score is confidence x IoU <= confidence, so such names cannot win: exact pruning)."""
            n = len(idx)
            sel = torch.tensor(idx, device="cuda")
            text = {k: text_all[k].index_select(bdim[k], sel) for k in keys}
            stage = FindStage(img_ids=torch.zeros(n, device="cuda", dtype=torch.long), text_ids=torch.arange(n, device="cuda"),
                              input_boxes=None, input_boxes_mask=None, input_boxes_label=None, input_points=None, input_points_mask=None)
            t0 = time.monotonic()
            bo = {**b, **text}
            prompt, mask, bo = model._encode_prompt(bo, stage, model._get_dummy_prompt(n))
            bo, enc, _ = model._run_encoder(bo, stage, prompt, mask)
            o = {"encoder_hidden_states": enc["encoder_hidden_states"], "prev_encoder_out": {"encoder_out": enc, "backbone_out": bo}}
            o, hs = model._run_decoder(memory=o["encoder_hidden_states"], pos_embed=enc["pos_embed"], src_mask=enc["padding_mask"], out=o,
                                       prompt=prompt, prompt_mask=mask, encoder_out=enc)
            prob = (o["pred_logits"].sigmoid() * o["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
            torch.cuda.synchronize(); t1 = time.monotonic(); clock["decode"] += t1 - t0; clock["batches"] += 1
            if floor is not None and float(prob.max()) <= floor:
                clock["skipped"] += 1
                return prob, None, None
            ids = stage.img_ids
            if "id_mapping" in bo and bo["id_mapping"] is not None:
                ids = bo["id_mapping"][ids]
            model._run_segmentation_heads(out=o, backbone_out=bo, img_ids=ids, vis_feat_sizes=enc["vis_feat_sizes"],
                                          encoder_hidden_states=o["encoder_hidden_states"], prompt=prompt, prompt_mask=mask, hs=hs)
            torch.cuda.synchronize(); clock["masks"] += time.monotonic() - t1
            return prob, o["pred_masks"], o["semantic_seg"]
        for n, r in enumerate(recs):
            ref, query = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            mask = np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1  # the reference label only
            if a.crop:  # name the object on its own crop (box of the mask with a 20% margin)
                ys, xs = np.nonzero(mask)
                bw, bh = xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()
                x0, y0 = max(0, int(xs.min() - .2 * bw)), max(0, int(ys.min() - .2 * bh))
                x1, y1 = min(mask.shape[1], int(xs.max() + 1 + .2 * bw)), min(mask.shape[0], int(ys.max() + 1 + .2 * bh))
                ref, mask = ref.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):
                rb = pooled(feats(ref), a.pool)
            if a.roi:  # the naming step sees only the feature cells around the reference object (features untouched, grid kept)
                L = rb["backbone_fpn"][-1].shape[-1]
                ys, xs = np.nonzero(mask)
                H, W = mask.shape
                cx0, cx1, cy0, cy1 = xs.min() / W * L, (xs.max() + 1) / W * L, ys.min() / H * L, (ys.max() + 1) / H * L
                mx, my = max(2.0, .2 * (cx1 - cx0)), max(2.0, .2 * (cy1 - cy0))
                x0, x1, y0, y1 = max(0, int(cx0 - mx)), min(L, int(np.ceil(cx1 + mx))), max(0, int(cy0 - my)), min(L, int(np.ceil(cy1 + my)))
                rb = dict(rb)
                cut = lambda f: f[..., y0 * (f.shape[-2] // L):y1 * (f.shape[-2] // L), x0 * (f.shape[-1] // L):x1 * (f.shape[-1] // L)]
                rb["backbone_fpn"], rb["vision_pos_enc"] = [cut(f) for f in rb["backbone_fpn"]], [cut(f) for f in rb["vision_pos_enc"]]
                if torch.is_tensor(rb.get("vision_features")):
                    rb["vision_features"] = rb["backbone_fpn"][-1]
                mask = mask[int(y0 / L * H):int(np.ceil(y1 / L * H)), int(x0 / L * W):int(np.ceil(x1 / L * W))]
                cells = (y1 - y0) * (x1 - x0) / L / L
            target = None
            score, conf, ious = torch.zeros(len(vocab), device="cuda"), torch.zeros(len(vocab), device="cuda"), torch.zeros(len(vocab), device="cuda")
            # the reference label is a semantic mask (every instance of the class): a name is also scored by its semantic
            # map and by the union of its confident instances, not only by its best single instance
            topc, iou_sem, iou_uni, total = (torch.zeros(len(vocab), device="cuda") for _ in range(4))
            names = list(range(len(vocab)))
            if a.shortlist:  # coarse pass: every name on a pooled grid, confidence only (no mask head); the best names go to the exact pass
                coarse, cb = torch.zeros(len(vocab), device="cuda"), pooled(rb, a.shortlist_pool)
                for i in range(0, len(vocab), 4 * a.batch):
                    idx = list(range(i, min(i + 4 * a.batch, len(vocab))))
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):
                        coarse[idx] = ground(cb, idx, float("inf"))[0].float().max(1).values
                names = coarse.topk(a.shortlist).indices.tolist()
            for i in range(0, len(names), a.batch):
                idx = names[i:i + a.batch]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(a.amp)):
                    prob, masks, sem_r = ground(rb, idx, float(total.max()) if a.prune else None)
                prob = prob.float()
                if masks is None:
                    conf[idx] = prob.max(1).values  # an upper bound of these names' scores; all below the current best
                    continue
                if target is None:
                    target = F.interpolate(torch.from_numpy(mask).float().cuda()[None, None], masks.shape[-2:], mode="area")[0, 0] > .5
                m = masks.float() > 0
                inter = (m & target).flatten(2).sum(2).float()
                iou = inter / (m.flatten(2).sum(2) + target.sum() - inter).clamp(min=1)
                j = (prob * iou).argmax(1, keepdim=True)
                conf[idx], ious[idx] = prob.gather(1, j)[:, 0], iou.gather(1, j)[:, 0]
                score[idx] = conf[idx] * ious[idx]
                tc = prob.max(1).values
                sm = F.interpolate(sem_r.float().reshape(len(idx), 1, *sem_r.shape[-2:]), m.shape[-2:], mode="bilinear", align_corners=False)[:, 0] > 0
                um = (m & (prob >= .7 * tc[:, None])[:, :, None, None]).any(1)
                rel = lambda x: (x & target).flatten(1).sum(1).float() / ((x | target).flatten(1).sum(1).float()).clamp(min=1)
                topc[idx], iou_sem[idx], iou_uni[idx] = tc, rel(sm), rel(um)
                total[idx] = torch.maximum(score[idx], tc * torch.maximum(iou_sem[idx], iou_uni[idx]))
            order = (total if a.semantic_score else score).argsort(descending=True)[:a.k].tolist()
            best = (float(score[order[0]]), float(conf[order[0]]), float(ious[order[0]]), order[0])
            scored = [(float(score[k]), float(conf[k]), float(ious[k]), k) for k in order]
            ks = [v[3] for v in scored[:max(1, a.rerank)]]  # the best name, then the next names of the shortlist
            prob_k, masks_k, sem = ground(feats(query), ks)
            prob, masks = prob_k[0], masks_k[0]
            top = prob.argsort(descending=True, stable=True)[:S.KEEP]
            q = F.interpolate(masks[top][:, None].float(), (t[3], t[2]), mode="bilinear", align_corners=False)[:, 0] > 0
            sq = F.interpolate(sem.float().reshape(len(ks), 1, *sem.shape[-2:]), (t[3], t[2]), mode="bilinear", align_corners=False)[:, 0] > 0
            q = torch.cat([q, sq])  # after the instance masks of the best name: one semantic map per name, the best name first
            top_score = float(prob[top[0]])
            semantic_row = len(top)
            # per name on the query: its top confidence and the union of its instance masks scored >= 0.7 x that top
            inst = []
            for i in range(len(ks)):
                keep = prob_k[i] >= .7 * prob_k[i].max()
                inst.append(F.interpolate(masks_k[i][keep][:, None].float(), (t[3], t[2]), mode="bilinear", align_corners=False)[:, 0].gt(0).any(0))
            q = torch.cat([q, torch.stack(inst)])
            rerank = [[vocab[k], round(scored[i][1], 4), round(scored[i][2], 4), float(prob_k[i].max()), round(float(topc[k]), 4),
                       round(float(iou_sem[k]), 4), round(float(iou_uni[k]), 4)] for i, k in enumerate(ks)]
            prob = prob[top].float().cpu().tolist() + [-1.0] * (2 * len(ks))
            file = "candidates/%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])
            np.savez_compressed(out / file, proposal_query=np.packbits(q.cpu().numpy().reshape(len(prob), t[2] * t[3]), axis=1))
            stream.write(json.dumps(dict(key=[r["fold"], r["e"], r["c"]], boxes=0, candidate_file=file, proposal_shape=[len(prob), t[3], t[2]],
                                         name=vocab[best[3]], name_conf=best[1], name_iou=best[2], true_class=S.NAMES[r["c"]], cells=float(cells) if a.roi else 1.0, query_top_score=top_score, semantic_row=semantic_row, rerank=rerank,
                                         shortlist=[[vocab[v[3]], round(v[1], 3), round(v[2], 3)] for v in scored],
                                         correct=False, meta=[[float(prob[i]), int(q[i].sum()), 0] for i in range(len(prob))])) + "\n")
            stream.flush()
            if n % 2 == 0:
                print("%d/%d episodes, %.2f s per episode; decode %.1f s, masks %.1f s, batches skipped %d of %d; true %s -> named %s (%.2f)" % (n + 1, len(recs), (time.monotonic() - start) / (n + 1), clock["decode"], clock["masks"], clock["skipped"], clock["batches"], S.NAMES[r["c"]], vocab[best[3]], best[1]), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=len(recs), vocabulary=len(vocab), k=a.k, elapsed_s=round(time.monotonic() - start, 1), query_annotation_opened=False)))


if __name__ == "__main__":
    main()
