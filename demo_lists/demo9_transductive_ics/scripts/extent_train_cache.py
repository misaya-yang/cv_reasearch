#!/usr/bin/env python3
"""Inputs for the extent head: one public FoRIS pass per episode (refinement skipped, it does not change the score),
then the relation maps of tics/relations.py, a few principal components of the query features, and the target.

  python scripts/extent_train_cache.py --prepare --test-manifest results/extent_v1/episodes.json --per-fold 600 \
      --manifest results/extent_head_v0/train_episodes.json                       # CPU: base episodes 60..659 per fold
  python scripts/extent_train_cache.py --manifest results/extent_head_v0/train_episodes.json \
      --test-manifest results/extent_v1/episodes.json --pool cache/evidence_v1/pool.pt --out cache/extent_head_v0
  python scripts/extent_train_cache.py --replay --test-manifest results/extent_v1/episodes.json \
      --features cache/evidence_v1/feat --packets results/extent_v1/run/packets --out cache/extent_head_v0
  python scripts/extent_train_cache.py --fixture DIR      # CPU; DIR holds finished extent and evidence fixtures

The live pass also redoes the first test episodes, and the replay compares its own maps with them: both routes must
give the same inputs. Training episodes are the standard seed-0 draws after the first 60 of each fold; an episode
that shares an image with a test episode is dropped.
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
from extent_experiment import build_host, coco_episodes, run_foris  # noqa: E402

CHECK = 8  # test episodes redone live, to compare with the replay


def prepare(a):
    test = json.loads(Path(a.test_manifest).read_text())
    seen = {x for r in test["episodes"] for x in (r["support"], r["query"])}
    first = max(r["e"] for r in test["episodes"] if not r.get("dev40")) + 1
    rows, dropped = [], 0
    for f in range(4):
        draws = coco_episodes(test["data_root"], f, first + a.per_fold)
        for r in test["episodes"]:  # the standard list must be the one the test episodes came from
            if r["fold"] == f and r["e"] < len(draws) and (r["c"], r["query"], r["support"]) != draws[r["e"]]:
                raise SystemExit("episode list differs from the test manifest: %s" % r)
        for e in range(first, first + a.per_fold):
            c, t, s = draws[e]
            if s in seen or t in seen:
                dropped += 1
                continue
            for name in (s, t):
                if not (Path(test["data_root"]) / name).is_file() or not (Path(test["annotation_root"]) / Path(name).with_suffix(".png")).is_file():
                    raise SystemExit("missing file: " + name)
            rows.append(dict(fold=f, e=e, c=c, support=s, query=t))
    man = dict(state="PREPARED", episodes=rows, **{k: test[k] for k in ("data_root", "annotation_root", "foris_root", "projection_basis")})
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text(json.dumps(man))
    print(json.dumps(dict(state="PREPARED", episodes=len(rows), dropped_for_shared_images=dropped, first_index=first)))


def pca_basis(pool, k, dev):
    import torch
    x = pool["deb"].reshape(-1, pool["deb"].shape[-1])
    x = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:100000]].to(dev).float()
    mean = x.mean(0)
    k = min(k, x.shape[1])
    return mean, torch.pca_lowrank(x - mean, q=k, center=False)[2][:, :k]


def pack(maps, comp, tf, side, ref_area):
    import numpy as np
    return dict(maps=maps.view(-1, side, side).cpu().numpy().astype(np.float16), pca=comp.T.reshape(-1, side, side).cpu().numpy().astype(np.float16),
                tf=tf.cpu().numpy().astype(np.float16), ref_area=np.float32(ref_area))


def live(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tics.relations import near_mask, relation_maps
    if a.fixture:
        root = Path(a.fixture)
        a.manifest = a.test_manifest = str(root / "episodes.json")
        a.pool, a.out = str(root / "cache/pool.pt"), str(root / "head")
    elif not a.unguarded and os.environ.get("DEMO9_CUDA_GUARD") != "1":
        raise SystemExit("run under scripts/experiment_resource_guard.py, or pass --unguarded")
    man, test = json.loads(Path(a.manifest).read_text()), json.loads(Path(a.test_manifest).read_text())
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("fresh output directory required")
    (out / "train").mkdir(parents=True)
    (out / "check").mkdir()
    dev = "cpu" if a.fixture else "cuda"
    report = dict(state="RUNNING", episodes=0, skipped=0, checked=0)
    start = time.monotonic()

    def save():
        report["elapsed_s"] = time.monotonic() - start
        (out / "report.json").write_text(json.dumps(report, indent=1))
    save()
    try:
        data, ann = Path(man["data_root"]), Path(man["annotation_root"])
        with torch.inference_mode():
            host = build_host(a, man, dev)
            host._finalize_mask = lambda mask, image: mask  # the score is fixed before refinement; skip the CRF
            mean, V = pca_basis(torch.load(a.pool, map_location="cpu", weights_only=False), a.components, dev)
            torch.save(dict(mean=mean.cpu(), V=V.cpu()), out / "pca.pt")
            near = None

            def one(row):
                nonlocal near
                c = row["c"]
                sp = Image.open(data / row["support"]).convert("RGB")
                qp = Image.open(data / row["query"]).convert("RGB")
                gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == c + 1).copy())
                native, got, ref_mask, _ = run_foris(host, sp, gold, qp)
                deb = got["deb"][0]
                q, r = deb[-1].flatten(1).T.half().float(), deb[0].flatten(1).T.half().float()  # as the feature cache stores them
                h, w = got["score"].shape
                if near is None or near.shape[0] != h * w:
                    near = near_mask(h * w, dev)
                cov = F.interpolate(ref_mask[None, None].float(), (h, w), mode="area")[0, 0]
                maps = relation_maps(q, r, cov, got["score"], got["s2"], got["s3"], near)
                truth = torch.from_numpy((np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == c + 1).copy()).to(dev)
                truth = F.interpolate(truth[None, None].float(), native.shape, mode="nearest")
                tf = F.avg_pool2d(truth, native.shape[0] // h)[0, 0]
                return pack(maps, (q - mean) @ V, tf, h, float(cov.mean()))
            for row in test["episodes"][:CHECK]:
                np.savez(out / "check" / ("%d_%d_%d.npz" % (row["fold"], row["e"], row["c"])), **one(row))
                report["checked"] += 1
            for row in man["episodes"][:a.limit]:
                try:
                    np.savez(out / "train" / ("%d_%d_%d.npz" % (row["fold"], row["e"], row["c"])), **one(row))
                    report["episodes"] += 1
                except RuntimeError as err:
                    if "No foreground tokens" not in str(err):
                        raise
                    report["skipped"] += 1
                if (report["episodes"] + report["skipped"]) % 50 == 0:
                    save()
                    print(json.dumps(dict(n=report["episodes"], skipped=report["skipped"], s=round(time.monotonic() - start, 1))), flush=True)
        report["state"] = "COMPLETED"
        save()
        print(json.dumps(report), flush=True)
    except BaseException as err:
        report.update(state="ERROR", error=repr(err))
        save()
        raise


def replay(a):
    """The same inputs for the test episodes, from the cached features and the stored packets."""
    import numpy as np
    import torch
    from tics.relations import near_mask, relation_maps
    if a.fixture:
        root = Path(a.fixture)
        a.test_manifest, a.features, a.packets, a.out = (str(root / x) for x in ("episodes.json", "cache/feat", "run/packets", "head"))
    test = json.loads(Path(a.test_manifest).read_text())
    out = Path(a.out)
    (out / "test").mkdir()
    dev = "cuda" if torch.cuda.is_available() and not a.fixture else "cpu"
    basis = torch.load(out / "pca.pt", map_location=dev, weights_only=False)
    near, drift, start = None, 0.0, time.monotonic()
    with torch.no_grad():
        for row in test["episodes"]:
            name = "%d_%d_%d" % (row["fold"], row["e"], row["c"])
            feat = torch.load(Path(a.features) / (name + ".pt"), map_location="cpu", weights_only=False)
            z = np.load(Path(a.packets) / (name + ".npz"))
            q, r = feat["q"].to(dev).float(), feat["r"].to(dev).float()
            n = q.shape[0]
            side = int(round(n ** 0.5))
            if near is None or near.shape[0] != n:
                near = near_mask(n, dev)
            t = lambda k: torch.from_numpy(z[k].astype(np.float32)).to(dev)
            maps = relation_maps(q, r, t("cov"), t("score"), t("s2"), t("s3"), near)
            ps = int(round((len(z["truth"]) * 8) ** 0.5)) // side
            tf = torch.from_numpy(np.unpackbits(z["truth"])[:(side * ps) ** 2].reshape(side, ps, side, ps).mean((1, 3)).astype(np.float32))
            packed = pack(maps, (q - basis["mean"]) @ basis["V"], tf, side, float(z["cov"].mean()))
            np.savez(out / "test" / (name + ".npz"), **packed)
            ref = out / "check" / (name + ".npz")
            if ref.is_file():
                seen = np.load(ref)
                drift = max([drift] + [float(np.abs(seen[k].astype(np.float32) - packed[k].astype(np.float32)).max()) for k in ("maps", "pca", "tf")])
    rep = dict(state="COMPLETED" if drift <= 0.03 else "ERROR", episodes=len(test["episodes"]), live_against_replay_max_difference=drift,
               elapsed_s=time.monotonic() - start)
    (out / "replay_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    if rep["state"] != "COMPLETED":
        raise SystemExit("the live pass and the replay give different inputs")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--manifest")
    p.add_argument("--test-manifest")
    p.add_argument("--per-fold", type=int, default=600)
    p.add_argument("--pool")
    p.add_argument("--features")
    p.add_argument("--packets")
    p.add_argument("--out")
    p.add_argument("--components", type=int, default=16)
    p.add_argument("--limit", type=int)
    p.add_argument("--fixture")
    p.add_argument("--unguarded", action="store_true")
    p.add_argument("--foris-root")
    p.add_argument("--demo4-root", default=os.environ.get("DEMO4_ROOT", "/root/autodl-tmp/demo4"))
    a = p.parse_args()
    prepare(a) if a.prepare else replay(a) if a.replay else live(a)


if __name__ == "__main__":
    main()
