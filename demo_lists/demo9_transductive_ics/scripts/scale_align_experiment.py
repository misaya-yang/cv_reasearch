#!/usr/bin/env python3
"""Is the accuracy FoRIS loses under scale mismatch restored by comparing the two objects at the same scale?

Measured on the finished extent run (scripts/analyze_scale_mismatch.py): with the query target's size held fixed,
IoU falls by 9.5 points per doubling of the scale mismatch between the reference object and the query object
(interval -15.5 to -3.5); a third of the episodes are mismatched by 2x or more and average 51.5 against 64.9.
This run intervenes: the smaller of the two objects is enlarged until both have the same size in their frames, and
the public FoRIS is run again. Same pair, same frozen encoder, no fitting, no other images.

Arms (one complete FoRIS pass each; the scale ratio is sqrt(query object area / reference object area), areas of the
largest connected component in the 1024 frame):
  native             FoRIS as published                                                        baseline
  reference_centric  the reference always cropped to twice its object's box, no scale reasoning  naive control
  align_mask         ratio from the first-pass mask                                             method
  align_core         ratio from the first-pass confident core (score >= 0.8)                    method
  align_oracle       ratio from the query's true mask, crop placed without labels               upper bound for the ratio
When the query object is the larger one the reference is cropped around its object; when it is the smaller one the
query is cropped around the first-pass core and the result is pasted back (the first-pass mask is kept outside).

  python scripts/scale_align_experiment.py --manifest results/extent_v1/episodes.json --out results/scale_align_v0/run
  python scripts/scale_align_experiment.py --fixture DIR           # CPU end to end, synthetic
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from extent_experiment import build_host, class_miou, make_fixture, run_foris  # noqa: E402

LABEL_FREE = ("reference_centric", "align_mask", "align_core")
LEAST, MOST = 1.4, 4.0  # act only on a mismatch of at least 1.4x; never enlarge by more than 4x


def largest(mask):
    """Area and box (x0, y0, x1, y1) of the largest connected component."""
    import numpy as np
    from scipy import ndimage as ndi
    lab, k = ndi.label(mask.cpu().numpy())
    if k == 0:
        return 0, None
    sizes = np.bincount(lab.ravel())[1:]
    ys, xs = np.nonzero(lab == int(sizes.argmax()) + 1)
    return int(sizes.max()), (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def window(box, side, frame):
    """A square of the given side around the centre of `box`, moved inside the frame."""
    H, W = frame
    side = int(max(2, min(round(side), H, W)))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x, y = int(min(max(round(cx - side / 2), 0), W - side)), int(min(max(round(cy - side / 2), 0), H - side))
    return x, y, x + side, y + side


def plan(query_area, reference_area):
    """Which image to enlarge and by how much so that both objects are equally large in their frames."""
    if query_area <= 0 or reference_area <= 0:
        return None, 1.0
    rho = math.sqrt(query_area / reference_area)
    if rho >= LEAST:
        return "reference", min(rho, MOST)
    if rho <= 1 / LEAST:
        return "query", min(1 / rho, MOST)
    return None, 1.0


def episode(host, row, man, dev):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tics.extent_cut import binarise, paste, to_original
    data, ann, c = Path(man["data_root"]), Path(man["annotation_root"]), row["c"]
    sp = Image.open(data / row["support"]).convert("RGB")
    qp = Image.open(data / row["query"]).convert("RGB")
    gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == c + 1).copy())
    begin = time.monotonic()
    native, got, ref_mask, _ = run_foris(host, sp, gold, qp)
    hw, size = tuple(native.shape), host.image_size
    core = binarise(got["score"].float(), 0.8, hw)
    a_ref, box_ref = largest(ref_mask)
    a_mask, box_mask = largest(native)
    a_core, box_core = largest(core)
    anchor = box_core or box_mask  # where the query crop is placed: the confident core, never the labels
    done, none = {}, dict(action="none", zoom=1.0)

    def second(kind, win):
        """One more FoRIS pass with one image cropped to `win` (model frame); identical requests are reused."""
        if (kind, win) in done:
            return done[kind, win]
        try:
            if kind == "reference":
                obox, _ = to_original(win, hw, sp.size)
                crop = gold[obox[1]:obox[3], obox[0]:obox[2]]
                res = run_foris(host, sp.crop(obox), crop, qp)[0] if crop.any() else None
            else:
                obox, mbox = to_original(win, hw, qp.size)
                res = native.clone()
                x0, y0, x1, y1 = mbox
                res[y0:y1, x0:x1] = paste(run_foris(host, sp, gold, qp.crop(obox))[0], mbox, hw)[y0:y1, x0:x1]
        except RuntimeError as err:
            if "No foreground tokens" not in str(err):
                raise
            res = None
        done[kind, win] = res
        return res

    def aligned(query_area):
        kind, zoom = plan(query_area, a_ref)
        if kind == "reference" and box_ref is not None:
            zoom = min(zoom, size / (1.1 * max(box_ref[2] - box_ref[0], box_ref[3] - box_ref[1])))  # keep the object inside
            win = window(box_ref, size / zoom, hw)
        elif kind == "query" and anchor is not None:
            zoom = min(zoom, 4.0)
            win = window(anchor, size / zoom, hw)
        else:
            return native, none
        if zoom < 1.15:
            return native, none
        res = second(kind, win)
        return (native, dict(action="failed", zoom=zoom)) if res is None else (res, dict(action=kind, zoom=zoom, box=list(win)))
    masks, info = dict(native=native), {}
    twice = 2 * max(box_ref[2] - box_ref[0], box_ref[3] - box_ref[1]) if box_ref is not None else size
    if twice < 0.9 * size:
        res = second("reference", window(box_ref, twice, hw))
        masks["reference_centric"], info["reference_centric"] = (native, none) if res is None else (res, dict(action="reference", zoom=size / twice))
    else:
        masks["reference_centric"], info["reference_centric"] = native, none
    masks["align_mask"], info["align_mask"] = aligned(a_mask)
    masks["align_core"], info["align_core"] = aligned(a_core)
    if dev == "cuda":
        torch.cuda.synchronize()
    seconds = time.monotonic() - begin

    # Query labels are opened only here, after every label-free arm is fixed.
    truth_o = np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == c + 1
    truth = torch.from_numpy(truth_o.copy()).to(dev)
    truth_m = F.interpolate(truth[None, None].float(), hw, mode="nearest")[0, 0].bool()
    a_true, _ = largest(truth_m)
    masks["align_oracle"], info["align_oracle"] = aligned(a_true)
    iu = lambda p, t: [int((p & t).sum()), int((p | t).sum())]
    orig = lambda p: F.interpolate(p[None, None].float(), truth_o.shape, mode="bilinear", align_corners=False)[0, 0] > 0.5
    rec = dict(fold=row["fold"], e=row["e"], c=c, support=row["support"], query=row["query"], dev40=bool(row.get("dev40")),
               area=float(truth_m.float().mean()), reference_area=float(ref_mask.float().mean()),
               component=dict(reference=a_ref, mask=a_mask, core=a_core, truth=a_true), plan=info,
               iu={k: iu(v, truth_m) for k, v in masks.items()}, original_iu={k: iu(orig(v), truth) for k, v in masks.items()},
               passes=1 + len(done), seconds=seconds)
    return rec, np.packbits(native.cpu().numpy())


def run(a):
    import numpy as np
    import torch
    if a.fixture:
        a.manifest, a.out = make_fixture(Path(a.fixture))
        a.out = str(Path(a.fixture) / "scale_run")
    elif not a.unguarded and os.environ.get("DEMO9_CUDA_GUARD") != "1":
        raise SystemExit("run under scripts/experiment_resource_guard.py, or pass --unguarded")
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("fresh output directory required")
    out.mkdir(parents=True)
    dev = "cpu" if a.fixture else "cuda"
    report = dict(state="RUNNING", outcome=None, episodes=0, stored_masks_compared=0, stored_masks_identical=0,
                  query_labels_used_for_prediction=False, upper_bound_arm_uses_true_scale="align_oracle")
    recs, start = [], time.monotonic()

    def save():
        arms = sorted(recs[0]["iu"]) if recs else []
        report.update(episodes=len(recs), elapsed_s=time.monotonic() - start, class_miou={k: class_miou(recs, k) for k in arms})
        tmp = out / "report.json.tmp"
        tmp.write_text(json.dumps(report, indent=1))
        tmp.replace(out / "report.json")
    save()
    try:
        if dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        with torch.inference_mode():
            host = build_host(a, man, dev)
            with open(out / "episodes.jsonl", "a") as stream:
                for row in man["episodes"][:a.limit]:
                    rec, bits = episode(host, row, man, dev)
                    stored = Path(a.packets or "") / ("%d_%d_%d.npz" % (row["fold"], row["e"], row["c"]))
                    if a.packets and stored.is_file():  # the first pass must be the FoRIS of the finished extent run
                        differ = int(np.unpackbits(bits ^ np.load(stored)["native"]).sum())
                        if differ > 0.001 * bits.size * 8:
                            raise RuntimeError("stored mask not reproduced on %s: %d pixels differ" % (row["query"], differ))
                        report["stored_masks_compared"] += 1
                        report["stored_masks_identical"] += int(differ == 0)
                    stream.write(json.dumps(rec) + "\n")
                    stream.flush()
                    recs.append(rec)
                    save()
                    gain = {k: report["class_miou"][k] - report["class_miou"]["native"] for k in report["class_miou"]}
                    print(json.dumps(dict(n=len(recs), fold=row["fold"], c=row["c"], s=round(rec["seconds"], 1),
                                          native=round(report["class_miou"]["native"], 2),
                                          gain={k: round(v, 2) for k, v in gain.items() if k != "native"})), flush=True)
                    fresh = sum(not r["dev40"] for r in recs)
                    if fresh == a.signal_after and gain["align_oracle"] < a.signal_gain:
                        report["outcome"] = "STOPPED_NO_SIGNAL"  # even the true scale ratio does not help
                        break
        report.update(state="COMPLETED", outcome=report["outcome"] or "ALL_EPISODES",
                      peak_bytes=torch.cuda.max_memory_allocated() if dev == "cuda" else 0)
        save()
    except BaseException as err:
        report.update(state="ERROR", error=repr(err))
        save()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest")
    p.add_argument("--out")
    p.add_argument("--packets", help="packets of the finished extent run; the first pass is compared with its stored masks")
    p.add_argument("--limit", type=int)
    p.add_argument("--fixture")
    p.add_argument("--unguarded", action="store_true")
    p.add_argument("--signal-after", type=int, default=80, help="fresh episodes after which the stop rule is applied")
    p.add_argument("--signal-gain", type=float, default=0.5, help="stop when the true-scale arm gains less than this")
    p.add_argument("--foris-root")
    p.add_argument("--demo4-root", default=os.environ.get("DEMO4_ROOT", "/root/autodl-tmp/demo4"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
