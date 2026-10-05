#!/usr/bin/env python3
"""Bank of text-head tokens for an episode list (GPU, minutes): after it, every question about the language channel
on DINOv3 is answered on a CPU by lang_gate.py. No label of a query is read here.

  python scripts/lang_bank.py --manifest M/dev_episodes.json --out B/dev --vocab V/vocabulary_clean.json \
      --repo SRC/dinov3 --weights W/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
      --backbone W/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth --bpe W/bpe_simple_vocab_16e6.txt.gz [--limit N]
  python scripts/lang_bank.py ... --tiny 1 --limit 2     # CPU plumbing with random small parts; numbers mean nothing

Per episode (float16): the query and the reference each as W (the 1024 x 1024 view FoRIS encodes, 64 x 64 tokens), S
(the official sliding-window protocol on the short-side-512 view) and B (the backbone's own tokens on the S grid); the
reference mask's coverage on both grids; the
reference object's own crop (class token, tokens, coverage). Once per bank: `text.npz`, the vocabulary's text features.
With --refs-only 1 nothing is stored per episode except the two name posteriors of the reference (`ref_names.npz`).
"""
import argparse
import json
import time
from pathlib import Path

from lang_common import COCO, STUFF, coverage, crop_view, head_tokens, load_text_model, naming, slide_tokens, text_matrix, MEAN, STD


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True); p.add_argument("--vocab", required=True)
    p.add_argument("--repo", required=True); p.add_argument("--bpe", required=True); p.add_argument("--weights"); p.add_argument("--backbone")
    p.add_argument("--limit", type=int); p.add_argument("--tiny", type=int, default=0); p.add_argument("--device", default="cuda")
    p.add_argument("--templates", type=int, default=80); p.add_argument("--report", default="report.json")
    p.add_argument("--refs-only", type=int, default=0)
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    whole, short, side, stride, long = (256, 256, 192, 96, 224) if a.tiny else (1024, 512, 384, 192, 448)
    dev = a.device
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, tokenize = load_text_model(a.repo, a.weights, a.backbone, a.bpe, dev, tiny=bool(a.tiny))
    scale = float(model.logit_scale.exp())
    began = time.monotonic()
    if not (out / "text.npz").exists():
        vocab = json.loads(Path(a.vocab).read_text())
        seen = {v.lower() for v in vocab}
        names = vocab + [s for s in STUFF if s not in seen]
        from lang_common import TEMPLATES
        e_patch, e_full = text_matrix(model, tokenize, names, dev, TEMPLATES[:a.templates])
        c_patch, c_full = text_matrix(model, tokenize, COCO, dev, TEMPLATES[:a.templates])
        np.savez(out / "text.npz", names=np.array(names), e_patch=e_patch.numpy(), e_full=e_full.numpy(),
                 coco_patch=c_patch.numpy(), coco_full=c_full.numpy(), scale=scale)
    text = np.load(out / "text.npz")
    names, e_patch, e_full = [str(v) for v in text["names"]], torch.from_numpy(text["e_patch"]).to(dev), torch.from_numpy(text["e_full"]).to(dev)
    text_s = time.monotonic() - began
    # the same tensor FoRIS encodes
    foris_view = transforms.Compose([transforms.Resize((whole, whole)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    half = lambda t: t.half().cpu().numpy()
    rows, done, began, posts = man["episodes"][:a.limit], 0, time.monotonic(), []
    with torch.inference_mode():
        for n, r in enumerate(rows):
            path = out / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"]))
            if path.exists() and not a.refs_only:
                continue
            ref, query = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            mask = torch.from_numpy((np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1).copy())  # reference label only
            if not mask.any():
                raise SystemExit("empty reference mask: %s" % r)
            g = whole // 16
            if a.refs_only:
                _, w_tok = head_tokens(model, foris_view(ref)[None].to(dev))
                c_cls, c_tok, c_cov = crop_view(model, ref, mask, dev, long)
                refd = dict(tok=F.normalize(w_tok[0].flatten(0, 1), dim=-1), cov=coverage(mask.to(dev), (whole, whole), (g, g)).flatten(),
                            c_cls=c_cls, c_tok=c_tok.flatten(0, 1), c_cov=c_cov.flatten())
                posts.append([half(naming(k, refd, e_patch, e_full, scale)) for k in ("crop_bma", "pool_bma")])
                done += 1
                continue
            _, w_tok = head_tokens(model, torch.stack([foris_view(query), foris_view(ref)]).to(dev))
            q_s, q_b, _ = slide_tokens(model, query, dev, short, side, stride)
            r_s, r_b, r_hw = slide_tokens(model, ref, dev, short, side, stride)
            c_cls, c_tok, c_cov = crop_view(model, ref, mask, dev, long)
            pack = dict(q_w=half(w_tok[0].flatten(0, 1)), r_w=half(w_tok[1].flatten(0, 1)),
                        r_cov_w=half(coverage(mask.to(dev), (whole, whole), (g, g))), q_s=half(q_s), r_s=half(r_s), q_b=half(q_b), r_b=half(r_b),
                        r_cov_s=half(coverage(mask.to(dev), r_hw, r_s.shape[:2])), c_cls=half(c_cls), c_tok=half(c_tok), c_cov=half(c_cov))
            np.savez(path, **pack)
            done += 1
            if n < 3 or n % 40 == 0:  # a reading a person can check: what the head calls the reference object
                refd = dict(tok=F.normalize(w_tok[1].flatten(0, 1), dim=-1), cov=coverage(mask.to(dev), (whole, whole), (g, g)).flatten(),
                            c_cls=c_cls, c_tok=c_tok.flatten(0, 1), c_cov=c_cov.flatten())
                top = {k: [names[i] for i in naming(k, refd, e_patch, e_full, scale).topk(3).indices.tolist()] for k in ("crop_bma", "pool_bma")}
                print("%d/%d class=%s crop=%s pool=%s %.1f s" % (n + 1, len(rows), COCO[r["c"]] if r["c"] < 80 else r["c"], top["crop_bma"],
                                                               top["pool_bma"], time.monotonic() - began), flush=True)
    spent = time.monotonic() - began
    if a.refs_only:
        np.savez(out / "ref_names.npz", keys=np.array([[r["fold"], r["e"], r["c"]] for r in rows]),
                 crop=np.stack([x[0] for x in posts]), pool=np.stack([x[1] for x in posts]))
    (out / a.report).write_text(json.dumps(dict(
        state="COMPLETED", episodes=len(rows), written=done, vocabulary=len(names), scale=scale, text_seconds=round(text_s, 1),
        seconds=round(spent, 1), seconds_per_episode=round(spent / max(done, 1), 3), tiny=bool(a.tiny))))
    print((out / a.report).read_text())


if __name__ == "__main__":
    main()
