#!/usr/bin/env python3
"""The boundary arm frozen by cpu/hmatte.py inside the complete public FoRIS (GPU): FoRIS runs unchanged, its cut is replaced by
the refined one (with and without FoRIS's own CRF after it), read at original resolution against FoRIS on the same episodes.

  python scripts/hires_pipe.py --manifest M/dev_episodes.json --gate cpu/hmatte_gate.json --out results/hires_pipe_dev
  python scripts/hires_pipe.py --manifest M/confirm_episodes.json --gate cpu/hmatte_gate.json --out results/hires_pipe_confirm --after results/hires_pipe_dev/report.json

It runs only if the gate's verdict is PASS and, with --after, only if that report says `go` (gain at original resolution
>= +1.5 with the interval above 0); with --after only the variant named there is read. Otherwise it writes why it did nothing.
The query label is opened after the masks of an episode exist.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

EXTENT = "/root/autodl-tmp/demo9_extent"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--gate", required=True); p.add_argument("--out", required=True)
    p.add_argument("--after"); p.add_argument("--limit", type=int); p.add_argument("--tiny", type=int, default=0)
    p.add_argument("--device", default="cuda"); p.add_argument("--force", type=int, default=0); p.add_argument("--extent", default=EXTENT)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    gate = json.loads(Path(a.gate).read_text()) if Path(a.gate).exists() else dict(verdict="NO GATE FILE")
    prev = json.loads(Path(a.after).read_text()) if a.after and Path(a.after).exists() else None
    why = None
    if gate["verdict"] != "PASS" and not a.force:
        why = "gate verdict %s" % gate["verdict"]
    elif a.after and not (prev or {}).get("go") and not a.force:
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
    from extent_experiment import build_host, finalise, run_foris
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lang_common import class_miou, iu, normalise, paired, photo_groups
    from matte_common import fine_tokens, refine
    man, dev = json.loads(Path(a.manifest).read_text()), a.device
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    host = build_host(SimpleNamespace(fixture=a.tiny or None, foris_root=None, demo4_root="/root/autodl-tmp/demo4"), man, dev)
    arm = gate.get("frozen") or dict(grid="s8", band=2, mix=1)
    off = 0 if arm["grid"] == "s8" else 4
    encode = lambda x: F.normalize(host.encoder.get_intermediate_layers(x, n=1, reshape=True)[0].float(), dim=1)
    variants = [prev["variant"]] if prev and prev.get("variant") else ["refined", "refined_crf"]
    rows, recs, began = man["episodes"][:a.limit], [], time.monotonic()
    with torch.inference_mode(), open(out / "episodes.jsonl", "w") as stream:
        for n, r in enumerate(rows):
            sp, qp = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1).copy())
            native, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
            hw = tuple(native.shape)
            deb = (lambda x: host._debias_features(x)) if getattr(host, "should_debiass", True) else (lambda x: x)
            if a.tiny:  # the stand-in works at 256: no 128 x 128 grid there, the cut passes through unchanged
                cut = got["pre"]
            else:
                q = fine_tokens(encode, deb, tgt[None], arm["grid"]).float().cpu()
                cut = refine(F.normalize(q, dim=-1), normalise(got["score"].float().cpu()), off, int(arm["band"]), int(arm["mix"])).to(native.device)
            masks = dict(native=native, refined=cut)
            if "refined_crf" in variants:
                masks["refined_crf"] = finalise(host, cut, tgt)
            # labels from here on
            truth = torch.from_numpy((np.asarray(Image.open(ann / Path(r["query"]).with_suffix(".png"))) == r["c"] + 1).copy()).to(dev)
            truth_m = F.interpolate(truth[None, None].float(), hw, mode="nearest")[0, 0].bool()
            orig = lambda m: F.interpolate(m[None, None].float(), truth.shape, mode="bilinear", align_corners=False)[0, 0] > 0.5
            rec = dict(fold=r["fold"], e=r["e"], c=r["c"], support=r["support"], query=r["query"], pre_iu=iu(got["pre"], truth_m),
                       iu={k: iu(v, truth_m) for k, v in masks.items()}, original_iu={k: iu(orig(v), truth) for k, v in masks.items()})
            recs.append(rec); stream.write(json.dumps(rec) + "\n"); stream.flush()
            if n % 20 == 0:
                print("%d/%d, %.1f s" % (n + 1, len(rows), time.monotonic() - began), flush=True)
    cls, fold, groups = np.array([r["c"] for r in recs]), np.array([r["fold"] for r in recs]), photo_groups(recs)
    arr = lambda level, k: np.array([r[level][k] for r in recs], float)
    res = {k: paired(arr("original_iu", k), arr("original_iu", "native"), cls, groups) for k in variants}
    best = max(variants, key=lambda k: res[k]["gain"])
    folds = {int(f): round(class_miou(arr("original_iu", best)[fold == f], cls[fold == f])
                           - class_miou(arr("original_iu", "native")[fold == f], cls[fold == f]), 2) for f in np.unique(fold)}
    report = dict(state="COMPLETED", episodes=len(recs), arm=arm, level="original resolution", foris=class_miou(arr("original_iu", "native"), cls),
                  variants=res, variant=best, folds=folds,
                  model_resolution=dict(foris_before_crf=class_miou(np.array([r["pre_iu"] for r in recs], float), cls), foris=class_miou(arr("iu", "native"), cls),
                                        refined=class_miou(arr("iu", "refined"), cls), gate_said=gate.get("frozen_gain")),
                  go=bool(res[best]["gain"] >= 1.5 and res[best]["ci95"][0] > 0), seconds=round(time.monotonic() - began, 1),
                  seconds_per_episode=round((time.monotonic() - began) / max(len(recs), 1), 2))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
