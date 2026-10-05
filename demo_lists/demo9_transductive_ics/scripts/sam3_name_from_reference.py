#!/usr/bin/env python3
"""Can SAM3 itself tell the concept from the reference? Names are ranked by how well they segment the reference mask (GPU).

  python scripts/sam3_name_from_reference.py --run RUN/dev --manifest SUITE/dev_episodes.json --sam3 SRC --checkpoint CKPT --out RUN/dev/reference_use

Measured on CONFIRM600: the class name beats the visual exemplar by 8.5 mIoU, and 91% of that gap sits in the episodes
where the exemplar route is unsure (top score under 0.7): the exemplar matches appearance, the name carries the category.
This diagnostic asks whether the category can be read off the reference alone: every name of a fixed vocabulary (the 80
COCO names: a closed vocabulary, so a diagnostic, not a method) is given to SAM3 on the reference image, and the names
are ranked by SAM3's own confidence times the IoU with the reference mask, for the best proposal under each name. The best name is then given to SAM3 on the query
alone (variant `retrieved_name`). The true class is used only to report how often the best name is the true one.
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
    p.add_argument("--sam3", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--limit", type=int)
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import v2
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    recs = [r for q in sorted(Path(a.run).glob("predictions_shard*.jsonl")) for r in map(json.loads, q.read_text().splitlines()) if r.get("candidate_file")][:a.limit]
    out = Path(a.out) / "retrieved_name"; (out / "candidates").mkdir(parents=True, exist_ok=True)
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
    names = list(S.NAMES)
    start = time.monotonic()
    with open(out / "variant.jsonl", "w") as stream, torch.inference_mode():
        texts = [model.backbone.forward_text([n], device="cuda") for n in names]

        def ground(b, k):
            o = model.forward_grounding(backbone_out={**b, **texts[k]}, find_input=proc.find_stage, geometric_prompt=model._get_dummy_prompt(), find_target=None)
            prob = (o["pred_logits"].sigmoid() * o["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)[0]
            return prob, o["pred_masks"][0]
        feats = lambda image: model.backbone.forward_image(proc.transform(v2.functional.to_image(image).to("cuda")).unsqueeze(0))
        for n, r in enumerate(recs):
            ref, query = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            mask = np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1  # the reference label only
            rb, target, score, iou_of, conf_of = feats(ref), None, [], [], []
            for k in range(len(names)):
                prob, masks = ground(rb, k)
                if target is None:
                    target = F.interpolate(torch.from_numpy(mask).float().cuda()[None, None], masks.shape[-2:], mode="area")[0, 0] > .5
                m = masks > 0
                inter = (m & target).flatten(1).sum(1).float()
                iou = inter / (m.flatten(1).sum(1) + target.sum() - inter).clamp(min=1)
                j = int((prob * iou).argmax())  # the proposal that is both confident under this name and on the reference mask
                score.append(float(prob[j] * iou[j])); iou_of.append(float(iou[j])); conf_of.append(float(prob[j]))
            order = np.argsort(-np.array(score))
            best = int(order[0])
            prob, masks = ground(feats(query), best)
            top = prob.argsort(descending=True, stable=True)[:S.KEEP]
            q = F.interpolate(masks[top][:, None].float(), (t[3], t[2]), mode="bilinear", align_corners=False)[:, 0] > 0
            prob = prob[top].float().cpu().tolist()
            name = "candidates/%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])
            np.savez_compressed(out / name, proposal_query=np.packbits(q.cpu().numpy().reshape(len(prob), t[2] * t[3]), axis=1))
            stream.write(json.dumps(dict(key=[r["fold"], r["e"], r["c"]], boxes=0, candidate_file=name, proposal_shape=[len(prob), t[3], t[2]],
                                         best=best, correct=best == r["c"], true_rank=int(np.where(order == r["c"])[0][0]), true_iou=iou_of[r["c"]], best_iou=iou_of[best], true_conf=conf_of[r["c"]], best_conf=conf_of[best], conf=[round(v, 4) for v in conf_of], iou=[round(v, 3) for v in iou_of],
                                         top5=[[names[int(k)], round(score[int(k)], 3)] for k in order[:5]],
                                         meta=[[float(prob[i]), int(q[i].sum()), 0] for i in range(len(prob))])) + "\n")
            stream.flush()
            if n % 10 == 0:
                print("%d/%d episodes, %.2f s per episode" % (n + 1, len(recs), (time.monotonic() - start) / (n + 1)), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=len(recs), elapsed_s=round(time.monotonic() - start, 1), vocabulary="80 COCO names (closed; diagnostic)")))


if __name__ == "__main__":
    main()
