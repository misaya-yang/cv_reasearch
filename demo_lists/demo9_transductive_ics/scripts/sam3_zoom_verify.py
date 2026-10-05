#!/usr/bin/env python3
"""Scale-aligned verification of the first pass's best candidates (GPU).

  python scripts/sam3_zoom_verify.py --run RUN/dev --manifest SUITE/dev_episodes.json --sam3 SRC --checkpoint CKPT --out X/zoom_dev.jsonl

Measured on 1645 episodes: the most confident candidate is off the target in 11.8% (23% when the target covers under
1% of the query, 5% above 30%), and the target is then the 2nd or 3rd candidate in 64%. So each of the K most confident
candidates is looked at again at the reference's scale: a canvas of the reference object's crop and of the query crop
around the candidate (twice its box), prompted with the reference box. Per candidate: the highest score among the
proposals that coincide with it (IoU >= 0.5 in the crop), the best IoU, and the top score in the crop. No query label.
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
    p.add_argument("--sam3", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--limit", type=int); p.add_argument("--k", type=int, default=3)
    a = p.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    data = Path(man["data_root"])
    recs = [r for q in sorted(Path(a.run).glob("predictions_shard*.jsonl")) for r in map(json.loads, q.read_text().splitlines()) if r.get("candidate_file")][:a.limit]
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
    run = S.Runner(Sam3Processor(model, device="cuda", confidence_threshold=.5))
    t = S.rectangles()[1]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with open(out, "w") as stream, torch.inference_mode():
        for n, r in enumerate(recs):
            ref, query = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            count, h, w = r["proposal_shape"]
            with np.load(Path(a.run) / r["candidate_file"], allow_pickle=False) as z:
                raw = np.unpackbits(z["proposal_query"], axis=1)[:, :h * w].reshape(count, h, w).astype(bool)
            meta = r["proposal_metadata"]
            order = [i for i in sorted(range(count), key=lambda i: -meta[i][0]) if meta[i][1] > 0][:a.k]
            x, y, bw, bh = r["exemplar_box"]
            x0, y0 = max(0, int(x - .25 * bw)), max(0, int(y - .25 * bh))
            x1, y1 = min(ref.size[0], int(x + 1.25 * bw) + 1), min(ref.size[1], int(y + 1.25 * bh) + 1)
            ref_crop, ref_box = ref.crop((x0, y0, x1, y1)), [x - x0, y - y0, bw, bh]
            W, H = query.size
            rows = []
            for i in order:
                full = np.asarray(Image.fromarray(raw[i].astype(np.uint8)).resize((W, H), Image.NEAREST)) > 0
                ys, xs = np.nonzero(full)
                if not len(xs):
                    continue
                cx, cy, cw, ch = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2, xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()
                a0, b0, a1, b1 = max(0, int(cx - cw)), max(0, int(cy - ch)), min(W, int(cx + cw) + 1), min(H, int(cy + ch) + 1)
                canvas, box = S.stitch(ref_crop, query.crop((a0, b0, a1, b1)), ref_box)
                union, prob, masks, kept, presence = run(canvas, box, None)
                cand = torch.from_numpy(np.asarray(Image.fromarray(full[b0:b1, a0:a1].astype(np.uint8)).resize((t[2], t[3]), Image.NEAREST)) > 0).to(masks.device)
                q = masks[:, t[1]:t[1] + t[3], t[0]:t[0] + t[2]]
                area, inter = q.flatten(1).sum(1), (q & cand).flatten(1).sum(1)
                iou = (inter / (area + cand.sum() - inter).clamp(min=1)).cpu().numpy()
                inq = [j for j in range(len(prob)) if int(area[j]) > 0]
                same = [j for j in inq if iou[j] >= .5]
                rows.append([int(i), float(meta[i][0]), max([prob[j] for j in same], default=0.0), float(iou.max()),
                             prob[inq[0]] if inq else 0.0, float(iou[inq[0]]) if inq else 0.0, float(cand.float().mean())])
            stream.write(json.dumps(dict(key=[r["fold"], r["e"], r["c"]], rows=rows)) + "\n"); stream.flush()
            if n % 50 == 0:
                print("%d/%d episodes, %.2f s per episode" % (n + 1, len(recs), (time.monotonic() - start) / (n + 1)), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=len(recs), fields=["index", "first_score", "verify_score", "best_iou", "crop_top_score", "crop_top_iou", "crop_share"],
                          elapsed_s=round(time.monotonic() - start, 1), query_annotation_opened=False)))


if __name__ == "__main__":
    main()
