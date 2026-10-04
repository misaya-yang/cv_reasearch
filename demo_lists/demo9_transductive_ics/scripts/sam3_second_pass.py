#!/usr/bin/env python3
"""Second pass inside the query: SAM3 on the query image alone, prompted with the anchor found on the stitched canvas.

  GPU  python scripts/sam3_second_pass.py --run RUN/dev --manifest SUITE/dev_episodes.json --sam3 SRC --checkpoint CKPT --out RUN/dev/second
  CPU  python scripts/sam3_second_pass.py --truth --run RUN/dev --manifest SUITE/dev_episodes.json --out RUN/dev/truth.npz

The anchor is the most confident query proposal of the first pass (no label). Its box is the only prompt; the image is
the query at SAM3's full input size, not the 1008 x 404 strip of the canvas. The 20 most confident masks are stored at
the first pass's query-rectangle size, so that both passes are read by the same code. No query annotation is opened.
--truth (CPU) writes the query truths at that size for reading rules with the server off.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def records(run, name, shards):
    paths = sorted(Path(run).glob(name + "_shard*.jsonl"))
    if shards is not None:
        paths = [p for p in paths if int(p.stem.split("shard")[1]) in shards]
    return [json.loads(line) for p in paths for line in p.read_text().splitlines() if line]


def bitmaps(run, r):
    import numpy as np
    count, h, w = r["proposal_shape"]
    with np.load(Path(run) / r["candidate_file"], allow_pickle=False) as z:
        return np.unpackbits(z["proposal_query"], axis=1)[:, :h * w].reshape(count, h, w).astype(bool)


def truth(a):
    import numpy as np
    from PIL import Image
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    rect = S.rectangles()[1]
    recs = records(a.run, "predictions", a.shards)
    keys, packed = [], []
    for r in recs:
        t = np.asarray(Image.open(Path(man["annotation_root"]) / Path(r["query"]).with_suffix(".png"))) == r["c"] + 1
        keys.append([r["fold"], r["e"], r["c"]]); packed.append(np.packbits(S.on_canvas(t, rect)))
    np.savez_compressed(a.out, keys=np.array(keys), truth=np.stack(packed), shape=np.array(rect[3:1:-1]))
    print(json.dumps(dict(state="COMPLETED", episodes=len(keys))))


def work(a):
    import numpy as np
    import torch
    from PIL import Image
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    data = Path(man["data_root"])
    recs = [r for r in records(a.run, "predictions", a.shards) if r.get("candidate_file")][:a.limit]
    out = Path(a.out); (out / "candidates").mkdir(parents=True, exist_ok=True)
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
    start = time.monotonic()
    with open(out / "second.jsonl", "w") as stream, torch.inference_mode():
        for n, r in enumerate(recs):
            raw, meta = bitmaps(a.run, r), r["proposal_metadata"]
            q = [i for i in range(len(meta)) if meta[i][1] > 0]
            row = dict(key=[r["fold"], r["e"], r["c"]], anchor=None)
            if q:
                i = max(q, key=lambda j: meta[j][0])
                h, w = raw.shape[1:]
                ys, xs = np.nonzero(raw[i])
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                box = [float((x0 + x1) / 2 / w), float((y0 + y1) / 2 / h), float((x1 - x0) / w), float((y1 - y0) / h)]
                image = Image.open(data / r["query"]).convert("RGB").resize((S.CANVAS, S.CANVAS), Image.BILINEAR)
                union, prob, masks, kept, presence = run(image, box, None)
                small = torch.nn.functional.interpolate(masks[:, None].float(), (h, w), mode="nearest")[:, 0] > 0
                name = "candidates/%d_%d_%d.npz" % tuple(row["key"])
                np.savez_compressed(out / name, proposal_query=np.packbits(small.cpu().numpy().reshape(len(prob), h * w), axis=1))
                row.update(anchor=int(i), box=box, score=prob, area=[int(v) for v in small.flatten(1).sum(1)], kept=kept, presence=presence,
                           candidate_file=name, proposal_shape=[len(prob), h, w])
            stream.write(json.dumps(row) + "\n"); stream.flush()
            if n % 50 == 0:
                print("%d/%d episodes, %.2f s per episode" % (n + 1, len(recs), (time.monotonic() - start) / (n + 1)), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=len(recs), elapsed_s=round(time.monotonic() - start, 1), query_annotation_opened=False)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sam3")
    p.add_argument("--checkpoint")
    p.add_argument("--truth", action="store_true")
    p.add_argument("--shards", type=int, nargs="*")
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    truth(a) if a.truth else work(a)
