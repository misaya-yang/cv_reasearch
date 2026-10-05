#!/usr/bin/env python3
"""The arm frozen by lang_gate.py inside the complete public FoRIS (GPU): FoRIS's final score plus the language
evidence, then FoRIS's own cut and CRF, read at original resolution against FoRIS on the same episodes.

  python scripts/lang_pipe.py --manifest M/dev_episodes.json --gate R/gate_dev.json --text B/dev/text.npz --out R/pipe_dev \
      --repo SRC/dinov3 --weights W/..._vision_head_and_text_encoder-a442d8f5.pth --backbone W/..._lvd1689m-8aa4cbdd.pth --bpe W/bpe...gz
  python scripts/lang_pipe.py --manifest M/confirm_episodes.json ... --out R/pipe_confirm --after R/pipe_dev/report.json

It runs only if the gate's verdict is PASS and, with --after, only if that report says `go` (written here as: gain at
original resolution >= +1.5 with the interval above 0). Otherwise it writes a report that says why it did nothing.
The query label is opened after both masks of an episode exist.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from lang_common import (binarise, class_miou, coverage, crop_view, evidence, fuse, head_tokens, iu, load_text_model, paired,
                         photo_groups, slide_tokens, to_grid)

EXTENT = "/root/autodl-tmp/demo9_extent"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--gate", required=True); p.add_argument("--text", required=True)
    p.add_argument("--out", required=True); p.add_argument("--repo", required=True); p.add_argument("--bpe", required=True)
    p.add_argument("--weights"); p.add_argument("--backbone"); p.add_argument("--after"); p.add_argument("--limit", type=int)
    p.add_argument("--tiny", type=int, default=0); p.add_argument("--device", default="cuda"); p.add_argument("--force", type=int, default=0)
    p.add_argument("--extent", default=EXTENT)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gate = json.loads(Path(a.gate).read_text())
    why = None
    if gate["verdict"] != "PASS" and not a.force:
        why = "gate verdict %s" % gate["verdict"]
    elif a.after and not json.loads(Path(a.after).read_text()).get("go"):
        why = "previous stage did not pass its bar"
    if why:
        (out / "report.json").write_text(json.dumps(dict(state="COMPLETED", skipped=why, go=False)))
        print("skipped:", why)
        return
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path[:0] = [a.extent, a.extent + "/scripts"]
    from extent_experiment import build_host, finalise, run_foris  # FoRIS before anything that has `models` or `utils`
    man, dev = json.loads(Path(a.manifest).read_text()), a.device
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    host = build_host(SimpleNamespace(fixture=a.tiny or None, foris_root=None, demo4_root="/root/autodl-tmp/demo4"), man, dev)
    model, _ = load_text_model(a.repo, a.weights, a.backbone, a.bpe, dev, tiny=bool(a.tiny), text=False)
    text = np.load(a.text)
    e_patch, e_full, scale = torch.from_numpy(text["e_patch"]).to(dev), torch.from_numpy(text["e_full"]).to(dev), float(text["scale"])
    mode, kind, lam = gate["frozen"]["mode"], gate["frozen"]["naming"], gate["frozen"]["lam"]
    short, side, stride, long = (256, 192, 96, 224) if a.tiny else (512, 384, 192, 448)
    rows, recs, began = man["episodes"][:a.limit], [], time.monotonic()
    with torch.inference_mode(), open(out / "episodes.jsonl", "w") as stream:
        for n, r in enumerate(rows):
            sp, qp = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1).copy())
            native, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
            hw, score = tuple(native.shape), got["score"].float()
            if mode == "W":
                _, tok = head_tokens(model, torch.stack([tgt, host._transform(sp).to(dev)]))
                q, q_hw, rt = tok[0], tok.shape[1:3], tok[1]
                cov = coverage(ref_mask, hw, rt.shape[:2])
            else:  # S: the head's tokens on the sliding-window grid; B: the backbone's
                pick = 0 if mode == "S" else 1
                q = slide_tokens(model, qp, dev, short, side, stride)[pick]
                r_all = slide_tokens(model, sp, dev, short, side, stride)
                rt, q_hw = r_all[pick], q.shape[:2]
                cov = coverage(gold.to(dev), r_all[2], rt.shape[:2])
            ref = dict(tok=F.normalize(rt.flatten(0, 1), dim=-1), cov=cov.flatten())
            if kind.startswith("crop"):
                c_cls, c_tok, c_cov = crop_view(model, sp, gold, dev, long)
                ref.update(c_cls=c_cls, c_tok=c_tok.flatten(0, 1), c_cov=c_cov.flatten())
            ev = to_grid(evidence(kind, F.normalize(q.flatten(0, 1), dim=-1), ref, e_patch, e_full, scale), q_hw, score.shape)
            cut = binarise(fuse(score, ev, lam), hw)
            masks = dict(native=native, fused=finalise(host, cut, tgt))
            # labels from here on
            truth = torch.from_numpy((np.asarray(Image.open(ann / Path(r["query"]).with_suffix(".png"))) == r["c"] + 1).copy()).to(dev)
            truth_m = F.interpolate(truth[None, None].float(), hw, mode="nearest")[0, 0].bool()
            orig = lambda m: F.interpolate(m[None, None].float(), truth.shape, mode="bilinear", align_corners=False)[0, 0] > 0.5
            rec = dict(fold=r["fold"], e=r["e"], c=r["c"], support=r["support"], query=r["query"],
                       pre_iu=dict(native=iu(got["pre"], truth_m), fused=iu(cut, truth_m)),
                       iu={k: iu(v, truth_m) for k, v in masks.items()}, original_iu={k: iu(orig(v), truth) for k, v in masks.items()})
            recs.append(rec); stream.write(json.dumps(rec) + "\n"); stream.flush()
            if n % 20 == 0:
                print("%d/%d, %.1f s" % (n + 1, len(rows), time.monotonic() - began), flush=True)
    cls, fold, groups = np.array([r["c"] for r in recs]), np.array([r["fold"] for r in recs]), photo_groups(recs)
    arr = lambda level, arm: np.array([r[level][arm] for r in recs], float)
    res = paired(arr("original_iu", "fused"), arr("original_iu", "native"), cls, groups)
    pre = paired(arr("pre_iu", "fused"), arr("pre_iu", "native"), cls, groups, draws=200)
    folds = {int(f): round(class_miou(arr("original_iu", "fused")[fold == f], cls[fold == f])
                           - class_miou(arr("original_iu", "native")[fold == f], cls[fold == f]), 2) for f in np.unique(fold)}
    report = dict(state="COMPLETED", episodes=len(recs), arm=gate["frozen"], level="original resolution, after FoRIS's CRF",
                  foris=class_miou(arr("original_iu", "native"), cls), fused=res, folds=folds,
                  before_crf=dict(foris=class_miou(arr("pre_iu", "native"), cls), fused=pre["miou"], gain=pre["gain"],
                                  gate_said=gate["frozen"]["in_sample"]["gain"]),
                  go=bool(res["gain"] >= 1.5 and res["ci95"][0] > 0), seconds=round(time.monotonic() - began, 1),
                  seconds_per_episode=round((time.monotonic() - began) / max(len(recs), 1), 2))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
