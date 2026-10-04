#!/usr/bin/env python3
"""SAM3's own semantic segmentation output, which the processor discards (GPU).

  python scripts/sam3_semantic.py --run RUN/dev --manifest SUITE/dev_episodes.json --sam3 SRC --checkpoint CKPT --out RUN/dev/reference_use

SAM3 returns instance proposals and a one-channel `semantic_seg` for the whole concept. Only the proposals were read so
far. Variants, each stored as one mask (logit > 0) at the canvas's query-rectangle size:
  semantic_canvas   the first pass unchanged (canvas, the stored reference box, no text): the query part of semantic_seg
  semantic_text     the query alone with the true class name (privileged; what this head gives when the concept is known)
No query annotation is opened.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
VARIANTS = ("semantic_canvas", "semantic_text")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--sam3", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--limit", type=int)
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    data = Path(man["data_root"])
    recs = [r for q in sorted(Path(a.run).glob("predictions_shard*.jsonl")) for r in map(json.loads, q.read_text().splitlines()) if r.get("candidate_file")][:a.limit]
    out = Path(a.out)
    for v in VARIANTS:
        (out / v / "candidates").mkdir(parents=True, exist_ok=True)
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

    def call(image, box, text):
        state = proc.set_image(image)
        proc.reset_all_prompts(state)
        seen, inner = {}, proc.model.forward_grounding

        def spy(**kw):
            seen["out"] = inner(**kw)
            return seen["out"]
        proc.model.forward_grounding = spy
        try:
            if text is not None:
                proc.set_text_prompt(text, state)
            if box is not None:
                proc.add_geometric_prompt(state=state, box=box, label=True)
        finally:
            proc.model.forward_grounding = inner
        return seen["out"]["semantic_seg"][0, 0]
    streams = {v: open(out / v / "variant.jsonl", "w") for v in VARIANTS}
    start = time.monotonic()
    with torch.inference_mode():
        for n, r in enumerate(recs):
            ref, query = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            canvas, box = S.stitch(ref, query, r["exemplar_box"])
            sem = F.interpolate(call(canvas, box, None)[None, None].float(), (S.CANVAS, S.CANVAS), mode="bilinear", align_corners=False)[0, 0]
            masks = {"semantic_canvas": sem[t[1]:t[1] + t[3], t[0]:t[0] + t[2]] > 0,
                     "semantic_text": F.interpolate(call(query, None, S.NAMES[r["c"]])[None, None].float(), (t[3], t[2]), mode="bilinear", align_corners=False)[0, 0] > 0}
            for v, m in masks.items():
                name = "candidates/%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])
                np.savez_compressed(out / v / name, proposal_query=np.packbits(m.cpu().numpy().reshape(1, t[2] * t[3]), axis=1))
                streams[v].write(json.dumps(dict(key=[r["fold"], r["e"], r["c"]], boxes=1, candidate_file=name, proposal_shape=[1, t[3], t[2]], meta=[[1.0, int(m.sum()), 0]])) + "\n")
                streams[v].flush()
            if n % 50 == 0:
                print("%d/%d episodes, %.2f s per episode" % (n + 1, len(recs), (time.monotonic() - start) / (n + 1)), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=len(recs), elapsed_s=round(time.monotonic() - start, 1), query_annotation_opened=False)))


if __name__ == "__main__":
    main()
