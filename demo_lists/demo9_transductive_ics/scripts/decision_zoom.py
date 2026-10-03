#!/usr/bin/env python3
"""A second look at matched scale, on top of a fitted read-out.

The first pass is scripts/decision_infer.py's: FoRIS, evidence, read-out. The read-out's mask then says how large
the query object is against the reference object. When they differ by 1.4x or more, the smaller one is enlarged
(the query is cropped around the read-out's largest region, or the reference around its object, enlargement at
most 4x), FoRIS and the read-out run once more on that pair, and the result is put back (outside a query crop the
first-pass mask is kept). FoRIS's refinement runs once, at the end, on the full image.

  python scripts/decision_zoom.py --models results/decision_v1/fit/models/ARM.pt --cache cache/decision_v1 \
      --manifest SUITE/dev_episodes.json --out results/decision_v1/zoom_dev --parallel 6        # under the guard
  python scripts/decision_zoom.py --fixture DIR

Arms: native (FoRIS), readout (first pass), zoom (ratio from the read-out's own mask), zoom_oracle (ratio from the
query's true mask, the crop still placed without labels: what a perfect size estimate would give).

Card, written 2026-10-03 before the run. Measured before (241 episodes, FoRIS alone): IoU falls 9.5 points per
doubling of the scale mismatch at fixed query size; a crop from the true box gains +5.96, and +18.8 where the query
object is under 0.35x of the reference object; crops placed from FoRIS's own mask gain +0.4 to +0.7.
  Assumption: FoRIS's own mask was too poor to place and size the crop; the read-out's mask is good enough.
  Prediction: zoom_oracle +1.5 to +4 over the read-out; zoom at least half of zoom_oracle; +4 or more on the
    episodes whose true mismatch is 2x or more; within 0.5 where the true ratio is inside 0.7 to 1.4.
  Match: the second look is the method's second stage; fit the read-out on evidence from both passes next.
  Mismatch: zoom_oracle under +1: scale is not what limits the read-out, drop the stage; zoom_oracle high and zoom
    low: the size estimate is the limiter, take the size from a fitted size read-out before anything else.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from decision_cache import taps  # noqa: E402
from decision_infer import evidence, symmetric  # noqa: E402
from extent_experiment import build_host, class_miou, finalise, run_foris  # noqa: E402
from scale_align_experiment import largest, plan, window  # noqa: E402

ARMS = ("native", "readout", "zoom", "zoom_oracle")
SETS = dict(score=slice(0, 1), relations=slice(0, 16), layers=slice(0, 24), features=slice(0, 40))


def work(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tics.decision_heads import Head, restore
    from tics.extent_cut import paste, to_original
    from tics.relations import near_mask
    if not a.fixture and not a.unguarded and os.environ.get("DEMO9_CUDA_GUARD") != "1":
        raise SystemExit("run under scripts/experiment_resource_guard.py, or pass --unguarded")
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stream_path = out / ("episodes_shard%d.jsonl" % a.shard[0])
    if stream_path.exists():
        raise SystemExit("fresh output required: " + str(stream_path))
    dev = "cpu" if a.fixture else "cuda"
    saved = torch.load(a.models, map_location="cpu", weights_only=False) if a.models else None
    basis = torch.load(Path(a.cache) / "pca.pt", map_location=dev, weights_only=False)
    report = dict(state="RUNNING", episodes=0, second_passes=0, arm=saved["arm"] if saved else "fixture")
    start = time.monotonic()

    def save():
        report["elapsed_s"] = time.monotonic() - start
        (out / ("report_shard%d.json" % a.shard[0])).write_text(json.dumps(report, indent=1))
    save()
    try:
        data, ann = Path(man["data_root"]), Path(man["annotation_root"])
        with torch.inference_mode():
            host = build_host(a, man, dev)
            refine, skip = host._finalize_mask, (lambda mask, image: mask)
            if saved:
                models, name = {f: restore(e, dev) for f, e in saved["models"].items()}, saved["channels"]
            else:
                net = Head("unroll", 24, 0).eval()
                models, name = {f: (lambda x, g, net=net: net(x.float(), x[:, 0].float(), g)[-1].sigmoid()) for f in range(4)}, "layers"
            near = {}

            def look(sp, gold, qp, f, refined):
                """One pass: FoRIS's mask (refined or not), the read-out's unrefined mask, FoRIS's reference mask, the image tensor."""
                host._finalize_mask = refine if refined else skip
                try:
                    with taps(host) as mid:
                        native, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
                finally:
                    host._finalize_mask = refine
                hw, (h, w) = tuple(native.shape), got["score"].shape
                if h * w not in near:
                    near[h * w] = near_mask(h * w, dev)
                ev = evidence(got, mid, ref_mask, near[h * w])
                x = torch.cat([symmetric(ev["maps"]), ev["layers"]])
                if name == "features":
                    x = torch.cat([x, ((ev["q"] - basis["mean"][f]) @ basis["V"][f]).T.reshape(-1, h, w).half().float()])
                prob = models[f](x[SETS[name]][None], (ev["idx"][None], ev["sim"][None]))[0]
                logit = torch.logit(prob.float().clamp(1e-6, 1 - 1e-6))
                cut = F.interpolate(logit[None, None], size=hw, mode="bilinear", align_corners=False)[0, 0] > 0
                return native, cut, ref_mask, tgt
            rows = [(i, r) for i, r in enumerate(man["episodes"][:a.limit])][a.shard[0]::a.shard[1]]
            with open(stream_path, "a") as stream:
                for i, row in rows:
                    c, f = row["c"], row["fold"]
                    sp, qp = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
                    gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == c + 1).copy())
                    begin = time.monotonic()
                    native, cut, ref_mask, tgt = look(sp, gold, qp, f, True)
                    hw, size = tuple(native.shape), host.image_size
                    a_ref, box_ref = largest(ref_mask)
                    a_cut, box_cut = largest(cut)
                    done = {}

                    def second(kind, win):
                        if (kind, win) in done:
                            return done[kind, win]
                        res = None
                        try:
                            if kind == "reference":
                                obox, _ = to_original(win, hw, sp.size)
                                crop = gold[obox[1]:obox[3], obox[0]:obox[2]]
                                if crop.any():
                                    res = look(sp.crop(obox), crop, qp, f, False)[1]
                            else:
                                obox, mbox = to_original(win, hw, qp.size)
                                x0, y0, x1, y1 = mbox
                                res = cut.clone()
                                res[y0:y1, x0:x1] = paste(look(sp, gold, qp.crop(obox), f, False)[1], mbox, hw)[y0:y1, x0:x1]
                            report["second_passes"] += 1
                        except RuntimeError as err:
                            if "No foreground tokens" not in str(err):
                                raise
                        done[kind, win] = res
                        return res

                    def aligned(query_area):
                        kind, zoom = plan(query_area, a_ref)
                        if kind == "reference" and box_ref is not None:
                            zoom = min(zoom, size / (1.1 * max(box_ref[2] - box_ref[0], box_ref[3] - box_ref[1])))
                            win = window(box_ref, size / zoom, hw)
                        elif kind == "query" and box_cut is not None:
                            win = window(box_cut, size / min(zoom, 4.0), hw)
                        else:
                            return cut, dict(action="none", zoom=1.0)
                        if zoom < 1.15:
                            return cut, dict(action="none", zoom=1.0)
                        res = second(kind, win)
                        return (cut, dict(action="failed", zoom=zoom)) if res is None else (res, dict(action=kind, zoom=zoom))
                    before, info = dict(readout=cut), {}
                    before["zoom"], info["zoom"] = aligned(a_cut)
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    seconds = time.monotonic() - begin
                    # Query labels are opened only here, after every label-free mask is fixed.
                    truth_o = np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == c + 1
                    truth = torch.from_numpy(truth_o.copy()).to(dev)
                    truth_m = F.interpolate(truth[None, None].float(), hw, mode="nearest")[0, 0].bool()
                    a_true, _ = largest(truth_m)
                    before["zoom_oracle"], info["zoom_oracle"] = aligned(a_true)
                    masks = dict(native=native, **{k: finalise(host, v, tgt) for k, v in before.items()})
                    iu = lambda p, t: [int((p & t).sum()), int((p | t).sum())]
                    orig = lambda p: F.interpolate(p[None, None].float(), truth_o.shape, mode="bilinear", align_corners=False)[0, 0] > 0.5
                    rec = dict(fold=f, e=row["e"], c=c, query=row["query"], support=row["support"], area=float(truth_m.float().mean()),
                               component=dict(reference=a_ref, readout=a_cut, truth=a_true), plan=info,
                               original_iu={k: iu(orig(v), truth) for k, v in masks.items()}, seconds=seconds)
                    stream.write(json.dumps(rec) + "\n")
                    stream.flush()
                    report["episodes"] += 1
                    if report["episodes"] % 20 == 0:
                        save()
        report["state"] = "COMPLETED"
        save()
        print(json.dumps(report), flush=True)
    except BaseException as err:
        report.update(state="ERROR", error=repr(err))
        save()
        raise


def merge(a):
    import math
    import numpy as np
    from analyze_extent import compare
    out = Path(a.out)
    want = len(json.loads(Path(a.manifest).read_text())["episodes"][:a.limit])
    recs = [json.loads(l) for f in sorted(out.glob("episodes_shard*.jsonl")) for l in open(f) if l.strip()]
    shards = [json.loads(f.read_text()) for f in sorted(out.glob("report_shard*.json"))]
    recs.sort(key=lambda r: (r["fold"], r["e"]))
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    ok = len(recs) == want and shards and all(s["state"] == "COMPLETED" for s in shards)
    rep = dict(state="COMPLETED" if ok else "ERROR", episodes=len(recs), expected=want, scope="class mIoU at original resolution, FoRIS's refinement",
               second_passes=sum(s["second_passes"] for s in shards), class_miou={k: class_miou(recs, k) for k in ARMS} if recs else {}, rows={})
    ratio = lambda r: math.sqrt(max(r["component"]["truth"], 1) / max(r["component"]["reference"], 1))
    groups = dict(all=recs, mismatch_2x=[r for r in recs if not 0.5 < ratio(r) < 2], matched=[r for r in recs if 0.7 <= ratio(r) <= 1.4])
    for tag, sub in groups.items():
        for k in ("zoom", "zoom_oracle"):
            if len(sub) > 1:
                rep["rows"]["%s|%s over readout" % (tag, k)] = dict(compare(sub, k, base="readout"), episodes=len(sub))
        if len(sub) > 1:
            rep["rows"]["%s|readout over native" % tag] = dict(compare(sub, "readout"), episodes=len(sub))
    for k, v in rep["rows"].items():
        print("%-34s %4d  %6.2f  %+6.2f [%+.2f, %+.2f]" % (k, v["episodes"], v["miou"], v["gain"], *v["ci95"]))
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(dict(state=rep["state"], episodes=len(recs))))
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models")
    p.add_argument("--cache")
    p.add_argument("--manifest")
    p.add_argument("--out")
    p.add_argument("--limit", type=int)
    p.add_argument("--shard", default="0/1", type=lambda s: tuple(int(v) for v in s.split("/")))
    p.add_argument("--merge", action="store_true")
    p.add_argument("--parallel", type=int)
    p.add_argument("--fixture")
    p.add_argument("--unguarded", action="store_true")
    p.add_argument("--foris-root")
    p.add_argument("--demo4-root", default=os.environ.get("DEMO4_ROOT", "/root/autodl-tmp/demo4"))
    a = p.parse_args()
    if a.fixture:
        root = Path(a.fixture)
        a.manifest, a.cache, a.out = str(root / "episodes.json"), str(root / "decision"), str(root / "zoom")
    if a.parallel:
        import subprocess
        rest = [x for i, x in enumerate(sys.argv[1:]) if x != "--parallel" and (i == 0 or sys.argv[i] != "--parallel")]
        jobs = [subprocess.Popen([sys.executable, str(Path(__file__).resolve())] + rest + ["--shard", "%d/%d" % (i, a.parallel)]) for i in range(a.parallel)]
        codes = [j.wait() for j in jobs]
        if any(codes):
            raise SystemExit("a shard failed: %s" % codes)
        a.merge = True
    merge(a) if a.merge else work(a)


if __name__ == "__main__":
    main()
