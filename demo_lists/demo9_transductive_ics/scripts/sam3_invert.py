#!/usr/bin/env python3
"""Prompt inversion: learn, on the reference alone, the text-side prompt that makes SAM3 segment the reference mask (GPU).

  python scripts/sam3_invert.py --run RUN/dev --manifest SUITE/dev_episodes.json --sam3 SRC --checkpoint CKPT --out RUN/dev/reference_use

Measured: with the true class name SAM3 reads 76.1 on CONFIRM600 on the query alone; with the visual exemplar on the
canvas 67.6. The name is not available, but the reference image and its mask are. Per episode, the features of the
dummy text prompt are optimised (a few gradient steps, model frozen) so that SAM3, on the reference image, returns the
reference mask; the learnt prompt is then given to SAM3 on the query alone at full input size. No class name, no query
label, no base-class training. Proposals are stored like the first pass (variant `invert`).
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
    p.add_argument("--steps", type=int, default=30); p.add_argument("--lr", type=float, default=0.02); p.add_argument("--name", default="invert")
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
    for q in model.parameters():
        q.requires_grad_(False)
    proc = Sam3Processor(model, device="cuda", confidence_threshold=.5)
    t = S.rectangles()[1]
    with torch.no_grad():
        text = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in model.backbone.forward_text(["visual"], device="cuda").items()}

    def features(image):
        with torch.no_grad():
            x = proc.transform(v2.functional.to_image(image).to("cuda")).unsqueeze(0)
            return model.backbone.forward_image(x)

    def ground(b, lang):
        o = model.forward_grounding(backbone_out={**b, **text, "language_features": lang}, find_input=proc.find_stage,
                                    geometric_prompt=model._get_dummy_prompt(), find_target=None)
        prob = (o["pred_logits"].sigmoid() * o["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)[0]
        return prob, o["pred_masks"][0]

    start = time.monotonic()
    with open(out / "variant.jsonl", "w") as stream:
        for n, r in enumerate(recs):
            ref, query = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            mask = np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1  # the reference label only
            rb = features(ref)
            lang = text["language_features"].clone().requires_grad_(True)
            opt = torch.optim.Adam([lang], lr=a.lr)
            target, first, last = None, None, None
            for step in range(a.steps):
                prob, masks = ground(rb, lang)
                if target is None:
                    target = F.interpolate(torch.from_numpy(mask).float().cuda()[None, None], masks.shape[-2:], mode="area")[0, 0]
                top = prob.topk(min(50, len(prob))).indices
                m = 1 - torch.prod(1 - prob[top, None, None] * masks[top].sigmoid(), 0)
                inter = (m * target).sum()
                loss = 1 - 2 * inter / (m.sum() + target.sum() + 1) + F.binary_cross_entropy(m.clamp(1e-5, 1 - 1e-5), target)
                first = float(loss) if first is None else first
                last = float(loss)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                prob, masks = ground(features(query), lang.detach())
                top = prob.argsort(descending=True, stable=True)[:S.KEEP]
                q = F.interpolate(masks[top][:, None].float(), (t[3], t[2]), mode="bilinear", align_corners=False)[:, 0] > 0
                prob = prob[top].float().cpu().tolist()
            name = "candidates/%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])
            np.savez_compressed(out / name, proposal_query=np.packbits(q.cpu().numpy().reshape(len(prob), t[2] * t[3]), axis=1))
            stream.write(json.dumps(dict(key=[r["fold"], r["e"], r["c"]], boxes=0, candidate_file=name, proposal_shape=[len(prob), t[3], t[2]],
                                         loss=[first, last], meta=[[float(prob[i]), int(q[i].sum()), 0] for i in range(len(prob))])) + "\n")
            stream.flush()
            if n % 10 == 0:
                print("%d/%d episodes, %.2f s per episode, loss %.3f -> %.3f" % (n + 1, len(recs), (time.monotonic() - start) / (n + 1), first, last), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=len(recs), steps=a.steps, lr=a.lr, elapsed_s=round(time.monotonic() - start, 1), query_annotation_opened=False)))


if __name__ == "__main__":
    main()
