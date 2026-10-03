#!/usr/bin/env python3
"""Where does the stopping information live? One pass of complete public FoRIS per episode, then every arm
re-decides only the extent. Results are streamed per episode.

Arms (all share the same FoRIS score; query labels are opened after every label-free arm is frozen):
  native        FoRIS's own cut, the midpoint of the score range                              baseline
  contrast      level with the largest score gap to a ring around it                          same-information control
  boundary      level whose boundary separates the least similar adjacent query patches       method 1, query structure
  round_trip    level whose region, sent back to the reference, best reproduces its mask      method 2, reference exclusion
  signature     level whose boundary pairs look most like reference inside/outside pairs      method 2b, reference boundary
  zoom_query    FoRIS again on a crop around its own result, reference unchanged              naive zoom
  zoom_pair     the same with the reference cropped to the matching object scale              method 3, re-observation
  oracle_cut    best level by ground truth                                                    upper bound of the cut family
  zoom_oracle   zoom_pair with the crop taken from the ground-truth box                       upper bound of zoom

  python scripts/extent_experiment.py --prepare --dev40 results/extent_v1/dev40_expected.json --per-fold 60 \
      --manifest results/extent_v1/episodes.json                                   # CPU, freezes the episode list
  python scripts/extent_experiment.py --manifest results/extent_v1/episodes.json --out results/extent_v1/run
  python scripts/extent_experiment.py --fixture /tmp/extent_fixture                 # CPU end-to-end, synthetic
"""
import argparse
import json
import os
import pickle
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
LABEL_FREE = ("contrast", "boundary", "round_trip", "signature", "zoom_query", "zoom_pair")
HOOKS = dict(_extract_features="raw", _part1_positional_debias="deb", _part2_background_suppression="s2",
             _part3_clustering="s3", _binarize_response="pre")


@contextmanager
def observe(host):
    """Read-only taps on the source stages; every original output is returned unchanged."""
    got, saved = {}, {n: (n in host.__dict__, host.__dict__.get(n)) for n in HOOKS}

    def wrap(name, fn):
        def call(*a, **k):
            out = fn(*a, **k)
            if name == "_binarize_response":
                got["score"] = a[0] if a else k["score_hw"]
            got[HOOKS[name]] = out[0] if isinstance(out, tuple) else out
            return out
        return call
    try:
        for n in HOOKS:
            setattr(host, n, wrap(n, getattr(host, n)))
        yield got
    finally:
        for n, (had, old) in saved.items():
            setattr(host, n, old) if had else delattr(host, n)


def run_foris(host, ref_img, ref_mask, tgt_img):
    """The public entry: set_reference / set_target / segment. Returns the final mask and what was observed."""
    try:
        host.set_reference(ref_img, ref_mask)
        host.set_target(tgt_img)
        mask_model, tgt = host._ref_masks[0].clone(), host._tgt_image.clone()
        with observe(host) as got:
            out = host.segment()
        return out.reshape(tgt.shape[-2:]).bool().clone(), got, mask_model, tgt
    finally:  # a failed call must not leave a reference behind for the next one
        host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None


def finalise(host, mask, tgt):
    return host._finalize_mask(mask, tgt[None]).reshape(mask.shape).bool().clone()


def zoom_pass(host, sp, gold, qp, box, ref_mask, hw, min_side):
    """FoRIS on a query crop, with the reference cropped to the same object scale when ref_mask is given."""
    from tics.extent_cut import paste, to_original, zoom_box
    obox, mbox = to_original(box, hw, qp.size)
    s_img, s_mask = sp, gold
    if ref_mask is not None:
        rb = zoom_box(ref_mask, min_side=min_side)
        if rb is not None:
            rob, _ = to_original(rb, hw, sp.size)
            crop = gold[rob[1]:rob[3], rob[0]:rob[2]]
            if crop.any():
                s_img, s_mask = sp.crop(rob), crop
    try:
        pred = run_foris(host, s_img, s_mask, qp.crop(obox))[0]
    except RuntimeError as err:
        if "No foreground tokens" not in str(err):
            raise
        return None, "no_reference_tokens"
    return paste(pred, mbox, hw), "ok"


def episode(host, row, man, dev):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tics.extent_cut import (LEVELS, binarise, choose, edge_signature, level_statistics, neighbour_affinity,
                                 normalise, zoom_box)
    data, ann, c = Path(man["data_root"]), Path(man["annotation_root"]), row["c"]
    sp = Image.open(data / row["support"]).convert("RGB")
    qp = Image.open(data / row["query"]).convert("RGB")
    gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == c + 1).copy())
    begin = time.monotonic()
    native, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
    hw, min_side = tuple(native.shape), host.image_size // 4
    score = got["score"].float()
    sn = normalise(score)
    h, w = sn.shape
    if not torch.equal(binarise(score, 0.5, hw), got["pre"]):
        raise RuntimeError("cut replica differs from the source binarisation")
    raw, deb = F.normalize(got["raw"][0], dim=1), got["deb"][0]
    fq, gq, gs = raw[-1].permute(1, 2, 0), deb[-1].permute(1, 2, 0), deb[0].permute(1, 2, 0)
    aff_r, aff_d = neighbour_affinity(fq)
    cov = F.interpolate(ref_mask[None, None].float(), (h, w), mode="area")[0, 0]
    sim = gq.flatten(0, 1) @ gs.flatten(0, 1).T
    back_idx = sim.argmax(0)
    sig = edge_signature(gq, gs, cov)
    st = level_statistics(sn, aff_r, aff_d, back_idx, cov, sig)

    masks, picks, zoom = dict(native=native), {}, {}
    for rule in ("contrast", "boundary", "round_trip", "signature"):
        k = choose(st, rule)
        picks[rule] = None if k is None else LEVELS[k]
        masks[rule] = native if k is None or LEVELS[k] == 0.5 else finalise(host, binarise(score, LEVELS[k], hw), tgt)
    kinv = choose(st, "boundary_inverted")
    picks["boundary_inverted"] = None if kinv is None else LEVELS[kinv]

    def zoom_arm(name, seed, pair, rounds):
        cur, src, info = native, seed, []
        for _ in range(rounds):
            box = zoom_box(src, min_side=min_side)
            if box is None or (info and box[2] - box[0] > 0.75 * info[-1]["side"]):
                break
            res, state = zoom_pass(host, sp, gold, qp, box, ref_mask if pair else None, hw, min_side)
            info.append(dict(box=list(box), side=box[2] - box[0], state=state))
            if res is None or not res.any():
                break
            cur = src = res
        masks[name], zoom[name] = cur, info
    zoom_arm("zoom_query", native, False, 1)
    zoom_arm("zoom_pair", native, True, 2)
    if dev == "cuda":
        torch.cuda.synchronize()
    seconds = time.monotonic() - begin

    # Query labels are opened only here, after every label-free arm is fixed.
    truth_o = np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == c + 1
    truth = torch.from_numpy(truth_o.copy()).to(dev)
    truth_m = F.interpolate(truth[None, None].float(), hw, mode="nearest")[0, 0].bool()
    up = F.interpolate(sn[None, None], size=hw, mode="bilinear", align_corners=False)[0, 0]
    lv_i = [int(((up > t) & truth_m).sum()) for t in LEVELS]
    lv_u = [int(((up > t) | truth_m).sum()) for t in LEVELS]
    best = max(range(len(LEVELS)), key=lambda k: lv_i[k] / max(lv_u[k], 1))
    picks["oracle_cut"] = LEVELS[best]
    masks["oracle_cut"] = finalise(host, binarise(score, LEVELS[best], hw), tgt)
    zoom_arm("zoom_oracle", truth_m, True, 1)

    def iu(p, t):
        return [int((p & t).sum()), int((p | t).sum())]

    def orig(p):
        return F.interpolate(p[None, None].float(), truth_o.shape, mode="bilinear", align_corners=False)[0, 0] > 0.5
    fwd_sim, fwd_idx = sim.max(1)
    fg = cov.flatten() >= 0.5
    packet = dict(score=score, s2=got["s2"].float(), s3=got["s3"].float(), aff_r=aff_r, aff_d=aff_d, cov=cov,
                  back_idx=back_idx.short(), fwd_idx=fwd_idx.short(), fwd_sim=fwd_sim,
                  fg_max=sim[:, fg].amax(1) if fg.any() else torch.zeros_like(fwd_sim),
                  bg_max=sim[:, ~fg].amax(1) if (~fg).any() else torch.zeros_like(fwd_sim))
    if sig is not None:
        packet.update({"sig_" + k: v for k, v in sig.items()})
    packet = {k: v.cpu().numpy() for k, v in packet.items()}
    packet.update(truth=np.packbits(truth_m.cpu().numpy()), native=np.packbits(native.cpu().numpy()),
                  pre=np.packbits(got["pre"].cpu().numpy()))
    levels = {k: [None if not np.isfinite(x) else float(x) for x in st[k].float().cpu().numpy()]
              for k in ("contrast", "boundary", "round_trip", "signature")}
    levels.update(area=st["area"].tolist(), valid=st["valid"].tolist(), I=lv_i, U=lv_u)
    rec = dict(fold=row["fold"], e=row["e"], c=c, support=row["support"], query=row["query"],
               dev40=bool(row.get("dev40")), area=float(truth_m.float().mean()), grid=[h, w], pick=picks, zoom=zoom,
               iu={k: iu(v, truth_m) for k, v in masks.items()},
               original_iu={k: iu(orig(v), truth) for k, v in masks.items()}, levels=levels, seconds=seconds)
    return rec, packet


def class_miou(recs, arm, key="original_iu"):
    acc = {}
    for r in recs:
        a = acc.setdefault(r["c"], [0, 0])
        a[0] += r[key][arm][0]
        a[1] += r[key][arm][1]
    return 100 * sum(i / max(u, 1) for i, u in acc.values()) / max(len(acc), 1)


def build_host(a, man, dev):
    import torch
    sys.path.insert(0, a.foris_root or man["foris_root"])
    import models.foris as foris  # FoRIS first: INSID3 has top-level packages with the same names
    if a.fixture:
        import torch.nn.functional as F
        torch.set_num_threads(1)

        class Encoder(torch.nn.Module):
            """Patch colour statistics: enough structure to drive every code path on a CPU."""
            def get_intermediate_layers(self, x, n=1, reshape=True):
                v = F.avg_pool2d(x, 16)
                e = F.avg_pool2d((x - F.interpolate(v, scale_factor=16)).abs(), 16)
                return [torch.cat([v, v.square(), v[:, :2] - v[:, 1:3], e], 1)]
        return foris.FoRIS(encoder=Encoder(), image_size=256, svd_components=2, tau=0.6, mask_refiner="bilinear",
                           resize_to_orig_size=False, device="cpu").eval().requires_grad_(False)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, a.demo4_root)
    from icx.common import TimmDINOv3
    from tics.native_assets import reuse_native_basis
    encoder = TimmDINOv3().to(dev).eval().requires_grad_(False)
    with reuse_native_basis(foris.FoRIS, man.get("projection_basis")):
        return foris.FoRIS(encoder=encoder, image_size=1024, svd_components=500, tau=0.6, mask_refiner="crf",
                           resize_to_orig_size=False, device=dev).eval().requires_grad_(False)


def run(a):
    import numpy as np
    import torch
    if a.fixture:
        a.manifest, a.out = make_fixture(Path(a.fixture))
    elif not a.unguarded and os.environ.get("DEMO9_CUDA_GUARD") != "1":
        raise SystemExit("run under scripts/experiment_resource_guard.py, or pass --unguarded")
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("fresh output directory required")
    (out / "packets").mkdir(parents=True)
    dev = "cpu" if a.fixture else "cuda"
    report = dict(state="RUNNING", outcome=None, manifest=str(a.manifest), episodes=0, dev40_checked=0,
                  dev40_exact=0, query_labels_used_for_prediction=False)
    recs, start = [], time.monotonic()

    def save():
        arms = sorted(recs[0]["iu"]) if recs else []
        report.update(episodes=len(recs), elapsed_s=time.monotonic() - start,
                      class_miou={k: class_miou(recs, k) for k in arms})
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
                    rec, packet = episode(host, row, man, dev)
                    if row.get("expect_original_iu"):
                        report["dev40_checked"] += 1
                        exp, now = row["expect_original_iu"], rec["original_iu"]["native"]
                        report["dev40_exact"] += int(exp == now)
                        if abs(exp[0] - now[0]) > 0.002 * exp[1] or abs(exp[1] - now[1]) > 0.002 * exp[1]:
                            raise RuntimeError("baseline not reproduced on %s: %s vs %s" % (row["query"], now, exp))
                    np.savez_compressed(out / "packets" / ("%d_%d_%d.npz" % (row["fold"], row["e"], row["c"])), **packet)
                    stream.write(json.dumps(rec) + "\n")
                    stream.flush()
                    recs.append(rec)
                    save()
                    gain = {k: report["class_miou"][k] - report["class_miou"]["native"] for k in report["class_miou"]}
                    print(json.dumps(dict(n=len(recs), fold=row["fold"], c=row["c"], s=round(rec["seconds"], 1),
                                          native=round(report["class_miou"]["native"], 2),
                                          gain={k: round(v, 2) for k, v in gain.items() if k != "native"})), flush=True)
                    fresh = sum(not r["dev40"] for r in recs)
                    if fresh == a.signal_after and max(gain[k] for k in LABEL_FREE) < a.signal_gain:
                        report["outcome"] = "STOPPED_NO_SIGNAL"
                        break
        report.update(state="COMPLETED", outcome=report["outcome"] or "ALL_EPISODES",
                      peak_bytes=torch.cuda.max_memory_allocated() if dev == "cuda" else 0)
        save()
    except BaseException as err:
        report.update(state="ERROR", error=repr(err))
        save()
        raise


def coco_episodes(data_root, fold, n, seed=0):
    """The episode list of INSID3's COCO-20i loader for this seed (same generator calls in the same order)."""
    import numpy as np
    meta = pickle.load(open(Path(data_root) / "splits/val" / ("fold%d.pkl" % fold), "rb"))
    ids = [fold + 4 * v for v in range(20)]
    np.random.seed(seed)
    eps = []
    for _ in range(n):
        c = np.random.choice(ids, 1, replace=False)[0]
        t = np.random.choice(meta[c], 1, replace=False)[0]
        while True:
            r = np.random.choice(meta[c], 1, replace=False)[0]
            if t != r:
                break
        eps.append((int(c), str(t), str(r)))
    return eps


def prepare(a):
    """CPU only. Freeze the list: the 40 development tasks first, then the standard episodes round-robin by fold."""
    from PIL import Image
    dev = json.loads(Path(a.dev40).read_text())
    rows = [dict(r, dev40=True) for r in dev["episodes"]]
    seen = {(r["fold"], r["e"]) for r in rows}
    lists = {f: coco_episodes(dev["data_root"], f, a.per_fold) for f in range(4)}
    for e in range(a.per_fold):
        for f in range(4):
            if (f, e) not in seen:
                c, t, r = lists[f][e]
                rows.append(dict(fold=f, e=e, c=c, support=r, query=t, dev40=False))
    for r in dev["episodes"]:  # the frozen development rows must be the standard draws
        if r["e"] < a.per_fold and (r["c"], r["query"], r["support"]) != lists[r["fold"]][r["e"]]:
            raise SystemExit("development row differs from the standard draw: %s" % r)
    for r in rows:
        for name in (r["support"], r["query"]):
            Image.open(Path(dev["data_root"]) / name).verify()
            if not (Path(dev["annotation_root"]) / Path(name).with_suffix(".png")).is_file():
                raise SystemExit("annotation missing: " + name)
    man = dict(state="PREPARED", data_root=dev["data_root"], annotation_root=dev["annotation_root"],
               foris_root=dev["foris_root"], projection_basis=dev["projection_basis"], per_fold=a.per_fold,
               seed=0, episodes=rows)
    Path(a.manifest).write_text(json.dumps(man))
    print(json.dumps(dict(state="PREPARED", episodes=len(rows), dev40=len(dev["episodes"]),
                          fresh=len(rows) - len(dev["episodes"]))))


def make_fixture(root):
    """Synthetic COCO-like data: a textured target, small in the query so the zoom path runs."""
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(0)
    (root / "data/val").mkdir(parents=True, exist_ok=True)
    (root / "ann/val").mkdir(parents=True, exist_ok=True)

    def scene(name, box, distractor):
        img = (rng.normal(110, 12, (240, 320, 3)) + np.array([0, 25, 50])).clip(0, 255)
        x0, y0, x1, y1 = distractor
        img[y0:y1, x0:x1] = rng.normal(60, 10, (y1 - y0, x1 - x0, 3)) + np.array([0, 120, 0])
        x0, y0, x1, y1 = box
        yy, xx = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = np.where(((yy // 4 + xx // 4) % 2)[..., None] == 0, [220, 40, 40], [240, 200, 40])
        mask = np.zeros((240, 320), np.uint8)
        mask[y0:y1, x0:x1] = 1
        Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(root / "data/val" / (name + ".jpg"), quality=95)
        Image.fromarray(mask).save(root / "ann/val" / (name + ".png"))
    rows = []
    for k in range(4):
        # odd episodes have a small reference object, so the reference crop path runs as well
        scene("s%d" % k, (100, 80, 150, 130) if k % 2 else (60 + 10 * k, 50, 220 + 10 * k, 190), (10, 10, 50, 60))
        scene("q%d" % k, (40 + 40 * k, 60 + 10 * k, 90 + 40 * k, 110 + 10 * k), (200, 150, 300, 230))
        rows.append(dict(fold=k % 4, e=k, c=0, support="val/s%d.jpg" % k, query="val/q%d.jpg" % k, dev40=False))
    man = dict(state="PREPARED", data_root=str(root / "data"), annotation_root=str(root / "ann"), episodes=rows)
    (root / "episodes.json").write_text(json.dumps(man))
    return str(root / "episodes.json"), str(root / "run")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--dev40")
    p.add_argument("--per-fold", type=int, default=60)
    p.add_argument("--manifest")
    p.add_argument("--out")
    p.add_argument("--limit", type=int)
    p.add_argument("--fixture")
    p.add_argument("--unguarded", action="store_true")
    p.add_argument("--signal-after", type=int, default=80, help="fresh episodes after which the stop rule is applied")
    p.add_argument("--signal-gain", type=float, default=0.5, help="stop when no label-free arm gains this much mIoU")
    p.add_argument("--foris-root")
    p.add_argument("--demo4-root", default=os.environ.get("DEMO4_ROOT", "/root/autodl-tmp/demo4"))
    a = p.parse_args()
    prepare(a) if a.prepare else run(a)


if __name__ == "__main__":
    main()
