#!/usr/bin/env python3
"""Frozen DINOv3 features of SAM3's query proposals, so that selection rules can be studied with the server off.

  python scripts/sam3_region_features.py --run RUN/dev --manifest SUITE/dev_episodes.json --out results/sam3_relative_v1/regions_dev.npz

For each episode of a `sam3_stitch.py` stage with kept proposal bitmaps (no query annotation is opened):
  pooled  [20, 1024]  mean patch feature inside each query proposal
  ref_fg, ref_bg      mean patch feature inside and outside the reference mask
  query_all           mean patch feature of the query
  dense   [20, 4]     fg_dense: mean over the proposal's patches of the best similarity to a reference-object patch
                      bg_dense: the same against reference-background patches
                      ref_votes: share of reference-object patches whose best query match lies inside the proposal
                      area: share of the query covered by the proposal
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def records(run, shards):
    paths = sorted(Path(run).glob("predictions_shard*.jsonl"))
    if shards is not None:
        paths = [p for p in paths if int(p.stem.split("shard")[1]) in shards]
    return [json.loads(line) for p in paths for line in p.read_text().splitlines() if line]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dino", default="/root/demo4_cache/models/dinov3-vitl16-timm")
    p.add_argument("--side", type=int, default=768)
    p.add_argument("--shards", type=int, nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    import numpy as np
    import timm
    import torch
    import torch.nn.functional as F
    from PIL import Image
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    recs = [r for r in records(a.run, a.shards) if r.get("candidate_file")][:a.limit]
    if not recs:
        raise SystemExit("no prediction records with proposal bitmaps under %s" % a.run)
    cfg = json.loads((Path(a.dino) / "config.json").read_text())
    model = timm.create_model(cfg["architecture"], pretrained=True, num_classes=0, dynamic_img_size=True,
                              pretrained_cfg_overlay=dict(file=str(Path(a.dino) / "model.safetensors"))).to(a.device).eval()
    mean, std = (torch.tensor(v, device=a.device).view(1, 3, 1, 1) for v in (MEAN, STD))

    def grid(image):
        """L2-normalised patch features [h, w, d] of an image resized to a multiple of 16 with long side --side."""
        w, h = image.size
        k = a.side / max(w, h)
        W, H = max(16, round(w * k / 16) * 16), max(16, round(h * k / 16) * 16)
        x = torch.from_numpy(np.asarray(image.resize((W, H), Image.BICUBIC))).to(a.device).permute(2, 0, 1)[None].float() / 255
        tokens = model.forward_features((x - mean) / std)[0, model.num_prefix_tokens:]
        assert tokens.shape[0] == (H // 16) * (W // 16), (tokens.shape, H, W)
        return F.normalize(tokens.float(), dim=-1).view(H // 16, W // 16, -1)

    def weights(mask, shape):
        """Share of each patch covered by a binary mask [n, h, w] -> [n, patches]."""
        return F.adaptive_avg_pool2d(torch.from_numpy(mask).to(a.device).float()[None], shape)[0].flatten(1)

    def pool(feat, w):
        return F.normalize(w @ feat / w.sum(1, keepdim=True).clamp(min=1e-6), dim=-1)

    keys, pooled, ref_fg, ref_bg, query_all, dense = [], [], [], [], [], []
    start = time.monotonic()
    with torch.inference_mode():
        for n, r in enumerate(recs):
            ref, query = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            ref_mask = np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1  # the reference label only
            count, h, w = r["proposal_shape"]
            with np.load(Path(a.run) / r["candidate_file"], allow_pickle=False) as z:
                raw = np.unpackbits(z["proposal_query"], axis=1)[:, :h * w].reshape(count, h, w).astype(bool)
            fr, fq = grid(ref), grid(query)
            wr = weights(ref_mask[None], fr.shape[:2])[0]
            wq = weights(raw, fq.shape[:2])
            fr, fq = fr.flatten(0, 1), fq.flatten(0, 1)
            fg = wr > .5 if bool((wr > .5).any()) else wr >= wr.max()
            bg = wr < .05 if bool((wr < .05).any()) else ~fg
            sim = fq @ fr.T
            to_fg, to_bg = sim[:, fg].max(1).values, sim[:, bg].max(1).values if bool(bg.any()) else torch.zeros(len(fq), device=a.device)
            norm = wq.sum(1).clamp(min=1e-6)
            votes = (wq[:, sim[:, fg].argmax(0)] > .5).float().mean(1)
            dense.append(torch.stack([wq @ to_fg / norm, wq @ to_bg / norm, votes, wq.mean(1)], 1).cpu().numpy())
            pooled.append(pool(fq, wq).cpu().numpy().astype(np.float16))
            ref_fg.append(pool(fr, wr[None])[0].cpu().numpy().astype(np.float16))
            ref_bg.append(pool(fr, (1 - wr)[None])[0].cpu().numpy().astype(np.float16))
            query_all.append(F.normalize(fq.mean(0), dim=-1).cpu().numpy().astype(np.float16))
            keys.append([r["fold"], r["e"], r["c"]])
            if n % 100 == 0:
                print("%d/%d episodes, %.2f s per episode" % (n + 1, len(recs), (time.monotonic() - start) / (n + 1)), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=np.array(keys), pooled=np.stack(pooled), ref_fg=np.stack(ref_fg), ref_bg=np.stack(ref_bg),
                        query_all=np.stack(query_all), dense=np.stack(dense).astype(np.float32),
                        dense_fields=np.array(["fg_dense", "bg_dense", "ref_votes", "area"]))
    print(json.dumps(dict(state="COMPLETED", episodes=len(keys), elapsed_s=round(time.monotonic() - start, 1), out=str(out),
                          query_annotation_opened=False)))


if __name__ == "__main__":
    main()
