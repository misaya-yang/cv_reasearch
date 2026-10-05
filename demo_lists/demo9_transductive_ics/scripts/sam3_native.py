#!/usr/bin/env python3
"""What the canvas costs SAM3: the class name on the query image alone, at full input size (GPU; privileged diagnostic).

  python scripts/sam3_native.py --run RUN/dev --manifest SUITE/dev_episodes.json --sam3 SRC --checkpoint CKPT --out RUN/dev/reference_use

The stored text arm (73.4 on CONFIRM600) ran on the stitched canvas, where the query is a 1008 x 404 strip. Variant
native_text gives the same class name to SAM3 on the query alone. The difference is what the canvas removes from the
model, with the concept known. Proposals are stored like the first pass. The class name is privileged: not a method.
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
    from PIL import Image
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    data = Path(man["data_root"])
    recs = [r for q in sorted(Path(a.run).glob("predictions_shard*.jsonl")) for r in map(json.loads, q.read_text().splitlines()) if r.get("candidate_file")][:a.limit]
    out = Path(a.out) / "native_text"; (out / "candidates").mkdir(parents=True, exist_ok=True)
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
    start = time.monotonic()
    with open(out / "variant.jsonl", "w") as stream, torch.inference_mode():
        for n, r in enumerate(recs):
            state = proc.set_image(Image.open(data / r["query"]).convert("RGB"))
            proc.reset_all_prompts(state)
            seen, inner = {}, proc.model.forward_grounding

            def spy(**kw):
                seen["out"] = inner(**kw)
                return seen["out"]
            proc.model.forward_grounding = spy
            try:
                proc.set_text_prompt(S.NAMES[r["c"]], state)
            finally:
                proc.model.forward_grounding = inner
            o = seen["out"]
            prob = (o["pred_logits"].sigmoid() * o["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)[0]
            top = prob.argsort(descending=True, stable=True)[:S.KEEP]
            q = torch.nn.functional.interpolate(o["pred_masks"][0, top][:, None].float(), (t[3], t[2]), mode="bilinear", align_corners=False)[:, 0] > 0
            prob = prob[top].float().cpu().tolist()
            name = "candidates/%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])
            np.savez_compressed(out / name, proposal_query=np.packbits(q.cpu().numpy().reshape(len(prob), t[2] * t[3]), axis=1))
            stream.write(json.dumps(dict(key=[r["fold"], r["e"], r["c"]], boxes=0, candidate_file=name, proposal_shape=[len(prob), t[3], t[2]],
                                         meta=[[float(prob[i]), int(q[i].sum()), 0] for i in range(len(prob))])) + "\n")
            stream.flush()
            if n % 50 == 0:
                print("%d/%d episodes, %.2f s per episode" % (n + 1, len(recs), (time.monotonic() - start) / (n + 1)), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=len(recs), elapsed_s=round(time.monotonic() - start, 1), privileged="class name")))


if __name__ == "__main__":
    main()
