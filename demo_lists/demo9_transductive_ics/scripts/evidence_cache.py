#!/usr/bin/env python3
"""Cache, once, what every later evidence test needs: the features FoRIS matched in for each episode of a finished
extent run, and a pool of other images with their class maps. Nothing is decided here.

  python scripts/evidence_cache.py --manifest results/extent_v1/episodes.json --packets results/extent_v1/run/packets \
      --out cache/evidence_v1
  python scripts/evidence_cache.py --fixture DIR   # CPU; DIR holds a finished `extent_experiment.py --fixture DIR`

Each episode runs the public FoRIS once and must reproduce the stored mask (the run stops if more than 0.1% of
the pixels differ; exact matches are counted). Later tests replay on the
cache (scripts/evidence_audit.py) and never call the encoder again. The cache is a working file: delete it when the
audit is closed.
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
from extent_experiment import build_host, run_foris  # noqa: E402


def pool_names(man, n, seed, allow_used):
    """Images of the same split that no episode uses, drawn once with a fixed seed."""
    import numpy as np
    rows = man["episodes"]
    sub = Path(rows[0]["support"]).parent
    used = {Path(r[k]).stem for r in rows for k in ("support", "query")}
    names = sorted(p.stem for p in (Path(man["annotation_root"]) / sub).glob("*.png"))
    cand = [x for x in names if allow_used or x not in used]
    order = np.random.default_rng(seed).permutation(len(cand))
    pick = []
    for i in order:
        if (Path(man["data_root"]) / sub / (cand[i] + ".jpg")).is_file():
            pick.append(cand[i])
        if len(pick) == n:
            break
    return sub, sorted(pick)


def patch_labels(cm, h, w):
    """Majority class of every patch and the share of its pixels that carry it. cm: [S, S] class map."""
    ps = cm.shape[0] // h
    t = cm.view(h, ps, w, ps).permute(0, 2, 1, 3).reshape(h * w, ps * ps)
    lab = t.mode(1).values
    return lab, (t == lab[:, None]).float().mean(1)


def choose_patches(lab, count, gen):
    """Half annotated classes when the image has them, the rest unannotated; always `count` patches."""
    import torch
    thing, rest = (lab > 0).nonzero()[:, 0], (lab == 0).nonzero()[:, 0]
    thing, rest = thing[torch.randperm(len(thing), generator=gen)], rest[torch.randperm(len(rest), generator=gen)]
    half = min(len(thing), count // 2)
    return torch.cat([thing[:half], rest, thing[half:]])[:count].sort().values


def run(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    if a.fixture:
        a.manifest, a.packets, a.out = (str(Path(a.fixture) / x) for x in ("episodes.json", "run/packets", "cache"))
    elif not a.unguarded and os.environ.get("DEMO9_CUDA_GUARD") != "1":
        raise SystemExit("run under scripts/experiment_resource_guard.py, or pass --unguarded")
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("fresh output directory required")
    (out / "feat").mkdir(parents=True)
    dev = "cpu" if a.fixture else "cuda"
    report = dict(state="RUNNING", episodes=0, bit_identical=0, debiased=0, pool_images=0, pool_patches=0)
    start = time.monotonic()

    def save():
        report["elapsed_s"] = time.monotonic() - start
        tmp = out / "report.json.tmp"
        tmp.write_text(json.dumps(report, indent=1))
        tmp.replace(out / "report.json")
    save()
    try:
        if dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        data, ann = Path(man["data_root"]), Path(man["annotation_root"])
        with torch.inference_mode():
            host = build_host(a, man, dev)
            for row in man["episodes"][:a.limit]:
                name = "%d_%d_%d" % (row["fold"], row["e"], row["c"])
                sp = Image.open(data / row["support"]).convert("RGB")
                qp = Image.open(data / row["query"]).convert("RGB")
                gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == row["c"] + 1).copy())
                native, got, _, _ = run_foris(host, sp, gold, qp)
                stored = np.load(Path(a.packets) / (name + ".npz"))["native"]
                differ = int(np.unpackbits(np.packbits(native.cpu().numpy()) ^ stored).sum())
                if differ > 0.001 * native.numel():  # the same code on the same weights; more than this is drift
                    raise RuntimeError("stored mask not reproduced on %s: %d pixels differ" % (name, differ))
                raw, deb = F.normalize(got["raw"][0], dim=1), got["deb"][0]  # [2, C, h, w]: reference, query
                debiased = bool((raw - deb).abs().max() > 1e-4)
                tok = lambda x: x.flatten(1).T.half().cpu()  # [h*w, C], row-major like the packets
                torch.save(dict(q=tok(deb[-1]), r=tok(deb[0]), debiased=debiased), out / "feat" / (name + ".pt"))
                report["episodes"] += 1
                report["bit_identical"] += int(differ == 0)
                report["debiased"] += int(debiased)
                if report["episodes"] % 20 == 0:
                    save()
                    print(json.dumps(dict(n=report["episodes"], debiased=report["debiased"])), flush=True)
            sub, names = pool_names(man, a.pool_images, a.seed, bool(a.fixture))
            size = host.image_size
            feats = dict(raw=[], deb=[])
            labels, purity, present = [], [], []
            for i, stem in enumerate(names):
                x = host._transform(Image.open(data / sub / (stem + ".jpg")).convert("RGB")).to(dev)
                f = F.normalize(host._extract_features(x[None, None]), dim=2)  # [1, 1, C, h, w]
                h, w = f.shape[-2:]
                cm = torch.from_numpy(np.asarray(Image.open(ann / sub / (stem + ".png")).resize((size, size), Image.NEAREST)).copy()).long()
                lab, pur = patch_labels(cm, h, w)
                idx = choose_patches(lab, min(a.pool_patches, h * w), torch.Generator().manual_seed(a.seed * 100003 + i))
                feats["raw"].append(f[0, 0].flatten(1).T[idx.to(dev)].half().cpu())
                feats["deb"].append(host._debias_features(f)[0, 0].flatten(1).T[idx.to(dev)].half().cpu())
                labels.append(lab[idx].to(torch.uint8))
                purity.append(pur[idx].half())
                present.append(sorted(int(v) for v in cm.unique()))
                if (i + 1) % 100 == 0:
                    report["pool_images"] = i + 1
                    save()
                    print(json.dumps(dict(pool=i + 1)), flush=True)
            torch.save(dict(raw=torch.stack(feats["raw"]), deb=torch.stack(feats["deb"]), label=torch.stack(labels),
                            purity=torch.stack(purity), present=present, names=names, seed=a.seed), out / "pool.pt")
        report.update(state="COMPLETED", pool_images=len(names), pool_patches=int(labels[0].numel()) if labels else 0,
                      pool_reuses_episode_images=bool(a.fixture))
        save()
        print(json.dumps(report), flush=True)
    except BaseException as err:
        report.update(state="ERROR", error=repr(err))
        save()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest")
    p.add_argument("--packets")
    p.add_argument("--out")
    p.add_argument("--pool-images", type=int, default=1200)
    p.add_argument("--pool-patches", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--fixture")
    p.add_argument("--unguarded", action="store_true")
    p.add_argument("--foris-root")
    p.add_argument("--demo4-root", default=os.environ.get("DEMO4_ROOT", "/root/autodl-tmp/demo4"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
