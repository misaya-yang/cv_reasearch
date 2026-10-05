#!/usr/bin/env python3
"""Interventions on the reference: is a FoRIS failure a property of the query, of the reference, or of the pair?

  python scripts/foris_intervene.py --manifest M/dev_episodes.json --out X/intervene_dev.jsonl [--limit N] [--alts 3]

Complete public FoRIS on the same query with different references (model size 1024, after its CRF):
  native      the episode's own reference
  self        the query itself with its true mask as the reference (uses the query label: diagnostic only)
  self_flip   the mirrored query with its mirrored mask as the reference (diagnostic only)
  alt_k       the reference of another episode of the same class (another image, its own mask)
For every arm: I/U of the final mask, and of the final score field at its midpoint and at its best cut (patch level).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

EXTENT = "/root/autodl-tmp/demo9_extent"
sys.path[:0] = [EXTENT, EXTENT + "/scripts"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int); p.add_argument("--alts", type=int, default=3)
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from extent_experiment import build_host, run_foris
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root="/root/autodl-tmp/demo4"), man, "cuda")
    image = lambda rel: Image.open(data / rel).convert("RGB")
    label = lambda rel, c: np.asarray(Image.open(ann / Path(rel).with_suffix(".png"))) == c + 1
    by_class = {}
    for r in man["episodes"]:
        by_class.setdefault(r["c"], []).append(r["support"])

    def read(ref, ref_mask, query, truth):
        try:
            native, got, _, _ = run_foris(host, ref, torch.from_numpy(ref_mask.copy()), query)
        except RuntimeError as err:
            if "No foreground tokens" not in str(err):
                raise
            return None
        t = F.interpolate(torch.from_numpy(truth.copy())[None, None].float(), tuple(native.shape), mode="nearest")[0, 0].bool().to(native.device)
        s = got["score"].float()
        sn = ((s - s.min()) / (s.max() - s.min()).clamp_min(1e-6)).flatten()
        cov = F.adaptive_avg_pool2d(t[None, None].float(), tuple(s.shape))[0, 0].flatten() * (t.numel() / s.numel())
        total = float(t.sum())
        order = sn.argsort(descending=True)
        inter = cov[order].cumsum(0)
        area = torch.arange(1, len(order) + 1, device=sn.device) * (t.numel() / s.numel())
        iou = inter / (area + total - inter).clamp_min(1)
        k = int(iou.argmax())
        mid = sn > .5
        return dict(final=[int((native & t).sum()), int((native | t).sum())], area=int(native.sum()), truth=int(total),
                    mid=[float(cov[mid].sum()), float(mid.sum() * (t.numel() / s.numel()) + total - cov[mid].sum())],
                    best=float(iou[k]), best_level=float(sn[order[k]]))
    begin = time.monotonic()
    with torch.inference_mode(), open(a.out, "w") as stream:
        for n, r in enumerate(man["episodes"][:a.limit]):
            q, truth = image(r["query"]), label(r["query"], r["c"])
            arms = dict(native=read(image(r["support"]), label(r["support"], r["c"]), q, truth),
                        self=read(q, truth, q, truth),
                        self_flip=read(ImageOps.mirror(q), truth[:, ::-1], q, truth))
            others = [s for s in dict.fromkeys(by_class[r["c"]]) if s not in (r["support"], r["query"])]
            for k, s in enumerate(others[:a.alts]):
                arms["alt_%d" % k] = read(image(s), label(s, r["c"]), q, truth)
            stream.write(json.dumps(dict(fold=r["fold"], e=r["e"], c=r["c"], arms=arms)) + "\n"); stream.flush()
            if n % 20 == 0:
                print("%d/%d, %.1f s" % (n + 1, len(man["episodes"][:a.limit]), time.monotonic() - begin), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=n + 1, elapsed_s=round(time.monotonic() - begin, 1))))


if __name__ == "__main__":
    main()
