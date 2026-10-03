#!/usr/bin/env python3
"""Inputs for the decision read-outs (tics/decision_heads.py). Every array scripts/extent_train_cache.py stores for
the extent head is kept under the same name (relation maps, per-model principal components, the embargo on
confirmation labels), so that script's consumers can read this cache too. Added per episode:
  layers   mean margin and nearest-neighbour margin against the reference's two sides at blocks 6, 12, 18 and at the
           final layer before debiasing                                                         [8, h, w]
  nbr_*    the 48 most similar query patches of every query patch, close neighbours left out      [N, 48]
  x_*      the 8 most similar reference patches of every query patch                              [N, 8]
  cov      the reference mask on the patch grid                                                  [h, w]
One public FoRIS pass per episode with refinement skipped. The development episodes are redone live (their middle
layers were never stored); the first 8 keep refinement on and must reproduce the stored masks bit for bit.

  python scripts/decision_cache.py --suite DIR --packets results/extent_v1/run/packets --out cache/decision_v1
      # DIR holds train_episodes.json, confirm_episodes.json, dev_episodes.json (image-isolated)
  python scripts/decision_cache.py --suite DIR --packets ... --out cache/decision_v1 --kinds train --manifest more.json --resume
  python scripts/decision_cache.py --fixture DIR       # CPU; DIR holds finished extent and evidence fixtures
`--kinds none` only fits the principal components; `--shard i/n --resume` lets several processes share one cache
(scripts/decision_cache_parallel.py starts them and writes the common report).
"""
import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from extent_experiment import build_host, run_foris  # noqa: E402
from extent_train_cache import CHECK, fit_bases, pack  # noqa: E402

BLOCKS = (6, 12, 18)
LAYER_NAMES = tuple("%s_%s" % (kind, where) for where in ("b6", "b12", "b18", "final") for kind in ("mean", "nn"))
K_SELF, K_CROSS = 48, 8


@contextmanager
def taps(host):
    """Outputs of the chosen encoder blocks during one pass. Read only; a host without blocks gives nothing."""
    got, handles = {}, []
    backbone = getattr(getattr(host, "encoder", None), "m", None)
    if backbone is not None and hasattr(backbone, "blocks"):
        for b in BLOCKS:
            def tap(module, args, output, b=b):
                if b in got:
                    raise RuntimeError("block %d ran twice in one pass" % b)
                got[b] = output.detach()
            handles.append(backbone.blocks[b - 1].register_forward_hook(tap))
    try:
        yield got
    finally:
        for h in handles:
            h.remove()


def margins(zq, zr, fg):
    """Unit query tokens against the reference foreground and background: by the two means, and by the nearest patch."""
    import torch.nn.functional as F
    if not bool(fg.any()) or bool(fg.all()):
        zero = zq.new_zeros(len(zq))
        return zero, zero
    mean = zq @ (F.normalize(zr[fg].mean(0), dim=0) - F.normalize(zr[~fg].mean(0), dim=0))
    sim = zq @ zr.T
    return mean, sim[:, fg].amax(1) - sim[:, ~fg].amax(1)


def build(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tics.relations import near_mask, relation_maps
    if a.fixture:
        root = Path(a.fixture)
        a.suite, a.packets, a.out = None, str(root / "run/packets"), str(root / "decision")
        sets = {k: json.loads((root / "episodes.json").read_text()) for k in ("train", "confirm", "dev")}
    else:
        if not a.unguarded and os.environ.get("DEMO9_CUDA_GUARD") != "1":
            raise SystemExit("run under scripts/experiment_resource_guard.py, or pass --unguarded")
        sets = {k: json.loads((Path(a.suite) / (k + "_episodes.json")).read_text()) for k in ("train", "confirm", "dev")}
    if a.manifest:
        sets["train"] = json.loads(Path(a.manifest).read_text())
    man = sets["train"]
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise SystemExit("fresh output directory required (or --resume)")
    for k in ("train", "confirm", "test"):
        (out / k).mkdir(parents=True, exist_ok=True)
    dev = "cpu" if a.fixture else "cuda"
    name = "report%s%s.json" % ("_" + Path(a.manifest).stem if a.manifest else "", "_shard%d" % a.shard[0] if a.shard[1] > 1 else "")
    if a.kinds == ["none"]:
        name = "report_pca_init.json"
    report = dict(state="RUNNING", train=0, confirm=0, test=0, reused=0, native_checks=0, native_bit_identical=0,
                  layers=list(LAYER_NAMES), neighbours=K_SELF, reference_neighbours=K_CROSS, middle_layers_captured=0)
    start = time.monotonic()

    def save():
        report["elapsed_s"] = time.monotonic() - start
        report["peak_bytes"] = torch.cuda.max_memory_allocated() if dev == "cuda" else 0
        tmp = out / (name + ".tmp")
        tmp.write_text(json.dumps(report, indent=1))
        tmp.replace(out / name)
    save()
    try:
        data, ann = Path(man["data_root"]), Path(man["annotation_root"])
        with torch.inference_mode():
            host = build_host(a, man, dev)
            refine, skip = host._finalize_mask, (lambda mask, image: mask)
            host._finalize_mask = skip
            if (out / "pca.pt").is_file():
                basis = torch.load(out / "pca.pt", map_location="cpu", weights_only=False)
                means, Vs = basis["mean"], basis["V"]
            else:  # the same construction as extent_train_cache.py: 32 training pairs per fold, one basis per held-out fold
                rows = [r for f in range(4) for r in [x for x in man["episodes"] if x["fold"] == f][:a.pca_prefix_per_fold]]
                prefix = []
                for row in rows:
                    sp, qp = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
                    gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == row["c"] + 1).copy())
                    states = run_foris(host, sp, gold, qp)[1]["deb"][0]
                    chunks = []
                    for state in (states[0], states[-1]):
                        x = state.flatten(1).T.half().float()
                        chunks.append(x[torch.linspace(0, len(x) - 1, min(256, len(x)), device=dev).long()].cpu())
                    prefix.append((row["fold"], torch.cat(chunks)))
                means, Vs = fit_bases(prefix, a.components, dev, bool(a.fixture))
                torch.save(dict(mean=means, V=Vs, model_axis=list(range(4)), components=a.components, training_prefix_rows=rows,
                                fitting_scope="collection_fold != model_holdfold", no_prior_pool_used=True), out / "pca.pt")
            report.update(pca_state="PCA_READY", components=int(Vs.shape[-1]), PCA_prior_pool_features_used=False)
            save()
            means, Vs = means.to(dev), Vs.to(dev)
            near = None

            def one(row, kind):
                nonlocal near
                c = row["c"]
                sp, qp = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
                gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == c + 1).copy())
                host._finalize_mask = refine if kind == "check" else skip
                try:
                    with taps(host) as mid:
                        native, got, ref_mask, _ = run_foris(host, sp, gold, qp)
                finally:
                    host._finalize_mask = skip
                if kind == "check":  # the complete public FoRIS must give the mask of the finished extent run
                    with np.load(Path(a.packets) / ("%d_%d_%d.npz" % (row["fold"], row["e"], row["c"]))) as z:
                        stored = np.asarray(z["native"], dtype=np.uint8).reshape(-1)
                    actual = np.packbits(native.cpu().numpy()).reshape(-1)
                    report["native_checks"] += 1
                    if actual.shape != stored.shape or int(np.unpackbits(actual ^ stored).sum()):
                        raise RuntimeError("complete public FoRIS mask not reproduced on %s" % row["query"])
                    report["native_bit_identical"] += 1
                deb = got["deb"][0]
                q, r = deb[-1].flatten(1).T.half().float(), deb[0].flatten(1).T.half().float()
                h, w = got["score"].shape
                n = h * w
                if near is None or near.shape[0] != n:
                    near = near_mask(n, dev)
                cov = F.interpolate(ref_mask[None, None].float(), (h, w), mode="area")[0, 0]
                maps = relation_maps(q, r, cov, got["score"], got["s2"], got["s3"], near)
                fgm = cov.flatten() >= 0.5
                rows = []
                for b in BLOCKS:
                    if b in mid:
                        t = mid[b]
                        t = F.normalize(t[:, t.shape[1] - n:].float(), dim=-1)  # prefix tokens come first; [reference, query]
                        if t.shape[0] != 2:
                            raise RuntimeError("expected one pass over the pair")
                        rows += list(margins(t[1], t[0], fgm))
                    else:
                        rows += [torch.zeros(n, device=dev)] * 2
                raw = F.normalize(got["raw"][0].float(), dim=1)
                rows += list(margins(raw[-1].flatten(1).T, raw[0].flatten(1).T, fgm))
                report["middle_layers_captured"] += int(len(mid) == len(BLOCKS))
                ns, ni = (q @ q.T).masked_fill(near, -1.0).topk(min(K_SELF, n - 1), dim=1)
                xs, xi = (q @ r.T).topk(min(K_CROSS, n), dim=1)
                tf = None  # the labels of confirmation queries are not opened while the cache is built
                if kind != "confirm":
                    truth = torch.from_numpy((np.asarray(Image.open(ann / Path(row["query"]).with_suffix(".png"))) == c + 1).copy()).to(dev)
                    truth = F.interpolate(truth[None, None].float(), native.shape, mode="nearest")
                    tf = F.avg_pool2d(truth, native.shape[0] // h)[0, 0]
                valid = [f != row["fold"] if kind == "train" else f == row["fold"] for f in range(4)]
                comp = torch.zeros(4, n, Vs.shape[-1], device=dev)
                for f, yes in enumerate(valid):
                    if yes:
                        comp[f] = (q - means[f]) @ Vs[f]
                res = pack(maps, comp, tf, h, float(cov.mean()), valid)
                f16 = lambda x: x.float().cpu().numpy().astype(np.float16)
                res.update(source_score=f16(got["score"]), model_shape=np.asarray(native.shape, dtype=np.int32),
                           query_mask_path=np.asarray(str(ann / Path(row["query"]).with_suffix(".png"))),
                           query_original_shape=np.asarray([qp.height, qp.width], dtype=np.int32),
                           layers=f16(torch.stack(rows).view(-1, h, w)), nbr_idx=ni.cpu().numpy().astype(np.int16), nbr_sim=f16(ns),
                           x_idx=xi.cpu().numpy().astype(np.int16), x_sim=f16(xs), cov=f16(cov))
                if a.lean:  # the class-free read-outs need neither the feature components nor the reference neighbours
                    for k in ("pca", "x_idx", "x_sim"):
                        del res[k]
                return res
            todo = [("train", r, "train") for r in man["episodes"]] if "train" in a.kinds else []
            if "dev" in a.kinds:
                todo += [("test", r, "check" if i < CHECK else "test") for i, r in enumerate(sets["dev"]["episodes"])]
            if "confirm" in a.kinds:
                todo += [("confirm", r, "confirm") for r in sets["confirm"]["episodes"]]
            for i, (folder, row, kind) in enumerate(todo[:a.limit][a.shard[0]::a.shard[1]]):
                path = out / folder / ("%d_%d_%d.npz" % (row["fold"], row["e"], row["c"]))
                if path.is_file():
                    report["reused"] += 1
                    continue
                tmp = path.with_suffix(".tmp.npz")
                np.savez(tmp, **one(row, kind))
                tmp.replace(path)
                report[folder] += 1
                if (i + 1) % 50 == 0:
                    save()
                    print(json.dumps(dict(n=i + 1, of=len(todo), train=report["train"], test=report["test"],
                                          confirm=report["confirm"], s=round(time.monotonic() - start, 1))), flush=True)
        report["state"] = "COMPLETED"
        save()
        print(json.dumps(report), flush=True)
    except BaseException as err:
        report.update(state="ERROR", error=repr(err))
        save()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", help="folder with train_episodes.json, confirm_episodes.json, dev_episodes.json")
    p.add_argument("--manifest", help="another training manifest to add to an existing cache (with --resume)")
    p.add_argument("--kinds", default="train,dev,confirm", type=lambda s: s.split(","))
    p.add_argument("--packets")
    p.add_argument("--out")
    p.add_argument("--components", type=int, default=16)
    p.add_argument("--pca-prefix-per-fold", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--shard", default="0/1", type=lambda s: tuple(int(v) for v in s.split("/")), help="i/n: every n-th episode from the i-th")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--lean", action="store_true", help="leave out the feature components and the reference neighbours (1.0 MB per episode instead of 1.6)")
    p.add_argument("--fixture")
    p.add_argument("--unguarded", action="store_true")
    p.add_argument("--foris-root")
    p.add_argument("--demo4-root", default=os.environ.get("DEMO4_ROOT", "/root/autodl-tmp/demo4"))
    build(p.parse_args())


if __name__ == "__main__":
    main()
