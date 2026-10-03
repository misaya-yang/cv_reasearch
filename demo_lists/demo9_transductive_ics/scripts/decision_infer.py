#!/usr/bin/env python3
"""The fitted read-out inside the complete FoRIS pipeline: scored at original resolution with FoRIS's own refinement.

One public FoRIS pass per episode gives FoRIS's mask (the baseline) and the evidence; the read-out fitted on the
other three folds turns the evidence into a probability on the patch grid; its logit is enlarged and cut at 0 the
way FoRIS enlarges and cuts its own response (an untrained read-out gives FoRIS's mask exactly), and goes through
the same refinement.

  python scripts/decision_infer.py --models results/decision_v1/fit/models/unroll_layers.pt --cache cache/decision_v1 \
      --manifest SUITE/dev_episodes.json --packets results/extent_v1/run/packets --out results/decision_v1/infer_dev --shard 0/6
  python scripts/decision_infer.py --merge --manifest ... --out results/decision_v1/infer_dev      # CPU: the table
  python scripts/decision_infer.py --fixture DIR      # CPU; DIR holds a finished decision_cache fixture

Arms: native (FoRIS); readout; removal (read-out only inside FoRIS's unrefined mask). `pre_*` are the same masks
before refinement at model size. The first 8 development episodes must reproduce the stored FoRIS masks bit for bit,
and the evidence computed here must equal the cached evidence the read-out was fitted on.

Card, written 2026-10-03 before the run.
  Assumption: the gain of the read-out on the patch grid survives enlargement and refinement.
  Prediction: at original resolution the read-out keeps at least 70% of its patch-level gain on the same episodes;
    removal within 0.7 of the read-out; fewer than 1 episode in 20 loses more than 10 IoU points.
  Match: this is the method's score on these episodes; scale the base data and run the standard protocol.
  Mismatch: under half of the patch-level gain: refinement undoes the read-out's edits, feed the refinement the
    probability instead of the cut mask, or refine before the read-out.
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
from decision_cache import BLOCKS, K_SELF, margins, taps  # noqa: E402
from extent_experiment import build_host, class_miou, finalise, run_foris  # noqa: E402

ARMS = ("native", "readout", "removal")
CHECK = 8


def evidence(got, mid, ref_mask, near):
    """The arrays scripts/decision_cache.py stores, from one observed pass, rounded as the cache rounds them."""
    import torch
    import torch.nn.functional as F
    from tics.relations import relation_maps
    deb = got["deb"][0]
    q, r = deb[-1].flatten(1).T.half().float(), deb[0].flatten(1).T.half().float()
    h, w = got["score"].shape
    n = h * w
    cov = F.interpolate(ref_mask[None, None].float(), (h, w), mode="area")[0, 0]
    maps = relation_maps(q, r, cov, got["score"], got["s2"], got["s3"], near).view(-1, h, w)
    fgm = cov.flatten() >= 0.5
    rows = []
    for b in BLOCKS:
        if b in mid:
            t = F.normalize(mid[b][:, mid[b].shape[1] - n:].float(), dim=-1)
            rows += list(margins(t[1], t[0], fgm))
        else:
            rows += [torch.zeros(n, device=q.device)] * 2
    raw = F.normalize(got["raw"][0].float(), dim=1)
    rows += list(margins(raw[-1].flatten(1).T, raw[0].flatten(1).T, fgm))
    ns, ni = (q @ q.T).masked_fill(near, -1.0).topk(min(K_SELF, n - 1), dim=1)
    half = lambda x: x.half().float()
    return dict(maps=half(maps), layers=half(torch.stack(rows).view(-1, h, w)), q=q, idx=ni, sim=half(ns))


def symmetric(maps):
    """The two edge maps as scripts/decision_fit.py feeds them: the mean over both neighbours of each direction."""
    import torch.nn.functional as F
    maps = maps.clone()
    er, ed = maps[14].clone(), maps[15].clone()
    maps[14] = (er + F.pad(er, (1, 0))[:, :-1]) / 2
    maps[15] = (ed + F.pad(ed, (0, 0, 1, 0))[:-1]) / 2
    return maps


def work(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tics.decision_heads import agnostic, restore
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
    report = dict(state="RUNNING", episodes=0, native_checks=0, native_bit_identical=0, cache_checks=0, cache_max_difference=0.0,
                  arm=saved["arm"] if saved else "fixture")
    start = time.monotonic()

    def save():
        report["elapsed_s"] = time.monotonic() - start
        (out / ("report_shard%d.json" % a.shard[0])).write_text(json.dumps(report, indent=1))
    save()
    try:
        data, ann = Path(man["data_root"]), Path(man["annotation_root"])
        with torch.inference_mode():
            host = build_host(a, man, dev)
            if saved:
                models, name = {f: restore(e, dev) for f, e in saved["models"].items()}, saved["channels"]
            else:  # fixture: an untrained read-out, which by construction returns FoRIS's own cut
                from tics.decision_heads import Head
                net = Head("unroll", 24, 0).eval()
                models, name = {f: (lambda x, g, net=net: net(x.float(), x[:, 0].float(), g)[-1].sigmoid()) for f in range(4)}, "layers"
            near = None
            rows = [(i, r) for i, r in enumerate(man["episodes"][:a.limit])][a.shard[0]::a.shard[1]]
            with open(stream_path, "a") as stream:
                for i, row in rows:
                    c, f = row["c"], row["fold"]
                    sp, qp = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
                    gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == c + 1).copy())
                    begin = time.monotonic()
                    with taps(host) as mid:
                        native, got, ref_mask, tgt = run_foris(host, sp, gold, qp)
                    hw = tuple(native.shape)
                    h, w = got["score"].shape
                    if near is None or near.shape[0] != h * w:
                        near = near_mask(h * w, dev)
                    ev = evidence(got, mid, ref_mask, near)
                    x = torch.cat([symmetric(ev["maps"]), ev["layers"]])
                    if name == "features":
                        comp = (ev["q"] - basis["mean"][f]) @ basis["V"][f]
                        x = torch.cat([x, comp.T.reshape(-1, h, w).half().float()])
                    if name == "agnostic":
                        x = agnostic(x[None, :24], ev["idx"][None], ev["sim"][None])[0]
                    else:
                        x = x[dict(score=slice(0, 1), relations=slice(0, 16), layers=slice(0, 24), features=slice(0, 40))[name]]
                    prob = models[f](x[None], (ev["idx"][None], ev["sim"][None]))[0]
                    logit = torch.logit(prob.float().clamp(1e-6, 1 - 1e-6))  # enlarged like FoRIS enlarges its response
                    cut = F.interpolate(logit[None, None], size=hw, mode="bilinear", align_corners=False)[0, 0] > 0
                    pre = got["pre"].reshape(hw).bool()
                    masks = dict(native=native, readout=finalise(host, cut, tgt), removal=finalise(host, cut & pre, tgt))
                    before = dict(pre_native=pre, pre_readout=cut, pre_removal=cut & pre)
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    seconds = time.monotonic() - begin
                    key = "%d_%d_%d" % (f, row["e"], c)
                    stored = Path(a.packets or "") / (key + ".npz")
                    if a.packets and i < CHECK and stored.is_file():  # the complete public FoRIS of the finished extent run
                        differ = int(np.unpackbits(np.packbits(native.cpu().numpy()).reshape(-1) ^ np.load(stored)["native"].reshape(-1)).sum())
                        report["native_checks"] += 1
                        report["native_bit_identical"] += int(differ == 0)
                        if differ:
                            raise RuntimeError("stored FoRIS mask not reproduced on %s: %d pixels" % (row["query"], differ))
                    for folder in ("test", "confirm", "train"):  # the evidence must be the evidence the read-out was fitted on
                        ref = Path(a.cache) / folder / (key + ".npz")
                        if report["cache_checks"] < 16 and ref.is_file():
                            z = np.load(ref)
                            d = max(float(np.abs(z["maps"].astype(np.float32) - ev["maps"].cpu().numpy()).max()),
                                    float(np.abs(z["layers"].astype(np.float32) - ev["layers"].cpu().numpy()).max()))
                            report["cache_checks"] += 1
                            report["cache_max_difference"] = max(report["cache_max_difference"], d)
                            if d > 0.03:
                                raise RuntimeError("evidence differs from the cache on %s: %.4f" % (key, d))
                    # Query labels are opened only here, after every mask is fixed.
                    truth_o = np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == c + 1
                    truth = torch.from_numpy(truth_o.copy()).to(dev)
                    truth_m = F.interpolate(truth[None, None].float(), hw, mode="nearest")[0, 0].bool()
                    iu = lambda p, t: [int((p & t).sum()), int((p | t).sum())]
                    orig = lambda p: F.interpolate(p[None, None].float(), truth_o.shape, mode="bilinear", align_corners=False)[0, 0] > 0.5
                    rec = dict(fold=f, e=row["e"], c=c, query=row["query"], support=row["support"], area=float(truth_m.float().mean()),
                               original_iu={k: iu(orig(v), truth) for k, v in masks.items()},
                               iu={k: iu(v, truth_m) for k, v in dict(masks, **before).items()}, seconds=seconds)
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
    import numpy as np
    from analyze_extent import compare, miou
    out = Path(a.out)
    man = json.loads(Path(a.manifest).read_text())
    want = len(man["episodes"][:a.limit])
    recs = [json.loads(l) for f in sorted(out.glob("episodes_shard*.jsonl")) for l in open(f) if l.strip()]
    shards = [json.loads(f.read_text()) for f in sorted(out.glob("report_shard*.json"))]
    recs.sort(key=lambda r: (r["fold"], r["e"]))
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    ok = len(recs) == want and shards and all(s["state"] == "COMPLETED" for s in shards)
    rep = dict(state="COMPLETED" if ok else "ERROR", episodes=len(recs), expected=want, arm=shards[0]["arm"] if shards else None,
               native_checks=sum(s["native_checks"] for s in shards), native_bit_identical=sum(s["native_bit_identical"] for s in shards),
               cache_max_difference=max([s["cache_max_difference"] for s in shards] + [0.0]), scope="class mIoU at original resolution, FoRIS's refinement",
               class_miou={k: class_miou(recs, k) for k in ARMS} if recs else {}, rows={}, before_refinement={})
    cls, fold = np.array([r["c"] for r in recs]), np.array([r["fold"] for r in recs])
    for k in ARMS[1:]:
        if recs:
            v = compare(recs, k)
            g = lambda key, arm, j: np.array([r[key][arm][j] for r in recs], float)
            v["per_fold"] = [float(miou(g("original_iu", k, 0)[fold == f], g("original_iu", k, 1)[fold == f], cls[fold == f])[0]
                                   - miou(g("original_iu", "native", 0)[fold == f], g("original_iu", "native", 1)[fold == f], cls[fold == f])[0]) for f in range(4)]
            v["lose_more_than_10"] = int(sum(100 * (r["original_iu"][k][0] / max(r["original_iu"][k][1], 1) - r["original_iu"]["native"][0] / max(r["original_iu"]["native"][1], 1)) < -10 for r in recs))
            rep["rows"][k] = v
            rep["before_refinement"][k] = compare(recs, "pre_" + k, base="pre_native", key="iu")
            print("%-8s %6.2f  %+6.2f [%+.2f, %+.2f]  folds %s  up %d down %d  | before refinement %+6.2f" % (
                k, v["miou"], v["gain"], *v["ci95"], " ".join("%+.1f" % x for x in v["per_fold"]), v["up"], v["down"], rep["before_refinement"][k]["gain"]))
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(dict(state=rep["state"], episodes=len(recs), native=rep["class_miou"].get("native"))))
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models")
    p.add_argument("--cache")
    p.add_argument("--manifest")
    p.add_argument("--packets")
    p.add_argument("--out")
    p.add_argument("--limit", type=int)
    p.add_argument("--shard", default="0/1", type=lambda s: tuple(int(v) for v in s.split("/")))
    p.add_argument("--merge", action="store_true")
    p.add_argument("--parallel", type=int, help="start this many shard processes of this script, then merge")
    p.add_argument("--fixture")
    p.add_argument("--unguarded", action="store_true")
    p.add_argument("--foris-root")
    p.add_argument("--demo4-root", default=os.environ.get("DEMO4_ROOT", "/root/autodl-tmp/demo4"))
    a = p.parse_args()
    if a.fixture:
        root = Path(a.fixture)
        a.manifest, a.packets, a.cache, a.out = str(root / "episodes.json"), str(root / "run/packets"), str(root / "decision"), str(root / "infer")
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
