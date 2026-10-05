#!/usr/bin/env python3
"""The sweep: every proposal and every kind of label-free evidence, as additions and deletions of the host mask.

  infer  label-free. Per episode: host mask, proposal masks, cached tokens (and saved layers when given) -> the
         whole evidence library (a few hundred one-byte maps, seeded noise placebos included), sealed.
         Opens no query truth and recomputes no proposal.
  score  reads truth. One row per (proposer, add | delete, map), the universe proposer included (the map itself
         proposes, anywhere in the image). Per row: does the map rank good edit pixels above bad ones inside an
         episode, and what does the edit gain with its budget chosen on three folds and read on the fourth.
         Then: placebo calibration (what the same sweep finds in noise), family summary, the composition
         A + B + auxiliaries with the whole choice nested over folds, sealed masks and the common report.

  python scripts/run_aux_evidence.py infer --root CACHE --manifest dev241.json --out OUT --workers 6 \\
      --host RUN_D/predictions:native --proposer rcg=RUN_RCG/predictions:rcg:RUN_RCG/fields:rcg \\
      --proposer Etwo=SUITE/proposals:latent_native.two_slot_em.control [--layers RUN_D/layers --bases RUN_D/arm_bases.pt]
  python scripts/run_aux_evidence.py score --root CACHE --out OUT --workers 6

A source is MASK_DIR:ARM[:FIELD_DIR:FIELD_KEY]: packed 1024 masks in MASK_DIR/<key>.npz[ARM], an optional continuous
64x64 field in FIELD_DIR/<key>.npz[FIELD_KEY]. Any sealed run serves: `predictions/` of a cache or forward run,
`proposals/` of the fixed edit-auxiliary suite, `frozen/` of the Astra replay. Masks are checked against the seal
beside their directory when there is one. The host comes from such a directory, never from the truth-bearing packet.
`infer` uses CPU workers or one GPU process; `score` is CPU only and can run while the GPU does other work.
All data are development data; nested numbers are still read on episodes seen before.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
UNIVERSE = "all"


def source(spec, named=True):
    name, rest = spec.split("=", 1) if named else (None, spec)
    parts = rest.split(":")
    if len(parts) not in (2, 4) or not all(parts):
        raise ValueError(f"Expected MASK_DIR:ARM[:FIELD_DIR:FIELD_KEY], got {spec}")
    d = dict(masks=str(Path(parts[0]).resolve()), arm=parts[1], fields=str(Path(parts[2]).resolve()) if len(parts) == 4 else None,
             field_key=parts[3] if len(parts) == 4 else None)
    return (name, d) if named else d


def seal_of(mask_dir):
    """Digests of a mask directory's files from the seal beside it, or None when it has no seal."""
    d = Path(mask_dir)
    for name in ("sealed.json", "proposal_sealed.json", "freeze.json"):
        if (d.parent / name).exists():
            s = json.loads((d.parent / name).read_text())
            if d.name == "predictions" and isinstance(s.get("predictions"), dict): return dict(s["predictions"])
            got = {Path(k).stem: v for k, v in s.get("files", {}).items() if isinstance(v, str) and k.startswith(d.name + "/")}
            if got: return got
    return None


def read_mask(src, key, seals):
    import numpy as np
    from ics.experiment import unpack, sha
    path = Path(src["masks"]) / f"{key}.npz"
    if src["masks"] not in seals: seals[src["masks"]] = seal_of(src["masks"])
    digest = sha(path)
    if seals[src["masks"]] is not None and seals[src["masks"]].get(key) != digest:
        raise ValueError(f"{path} differs from its run's seal")
    with np.load(path, allow_pickle=False) as f:
        return unpack(f[src["arm"]]), digest


def start_worker(threads):
    import torch
    torch.set_num_threads(threads); torch.set_num_interop_threads(1); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


_FORWARD = None  # single GPU process only: the frozen host, the layer ids, the image roots and the positional bases


def layer_tokens(encoder, images, ids):
    """[len(ids)] float tensors [batch, tokens, channels] of the named blocks (1-based), encoder norm applied."""
    maps = encoder.get_intermediate_layers(images, n=[i - 1 for i in ids], reshape=True)
    if len(maps) != len(ids) or any(m.ndim != 4 for m in maps): raise ValueError("Encoder must return one NCHW map per requested block")
    return [m.permute(0, 2, 3, 1).reshape(m.shape[0], -1, m.shape[1]).float() for m in maps]


def black_bases(host, ids, rank=500):
    """Positional basis of every layer space and every consecutive difference, from the normalized black image,
    built as the host builds its own: unit tokens, centred across tokens, leading left singular vectors."""
    import torch
    import torch.nn.functional as F
    black = torch.zeros(1, 3, host.image_size, host.image_size, device=host.device)
    mean, std = black.new_tensor([.485, .456, .406])[None, :, None, None], black.new_tensor([.229, .224, .225])[None, :, None, None]
    tok = dict(zip(ids, (t[0] for t in layer_tokens(host.encoder, (black - mean) / std, ids))))
    def basis(x):
        x = F.normalize(x, dim=1).T; x = x - x.mean(1, keepdim=True)
        return torch.linalg.svd(x, full_matrices=False)[0][:, :rank].contiguous()
    out = {f"l{i}": basis(tok[i]) for i in ids}
    for lo, hi in zip(ids, ids[1:]):
        out[f"d{lo}_{hi}"] = basis(F.normalize(F.normalize(tok[hi], dim=1) - F.normalize(tok[lo], dim=1), dim=1))
    return out


def forward_layers(row):
    """One paired reference/query forward through the host's own transforms; the activations stay on the GPU."""
    import numpy as np
    import torch
    from PIL import Image
    host, ids, data, ann = (_FORWARD[k] for k in ("host", "ids", "data", "ann"))
    with Image.open(data / row["support"]) as im: support = im.convert("RGB")
    with Image.open(data / row["query"]) as im: query = im.convert("RGB")
    with Image.open(ann / Path(row["support"]).with_suffix(".png")) as im: smask = torch.from_numpy((np.asarray(im) == row["c"] + 1).copy())
    try:
        host.set_reference(support, smask); host.set_target(query)
        if host._ref_images.shape[0] != 1: raise ValueError("Exactly one labeled reference is permitted")
        toks = layer_tokens(host.encoder, torch.cat((host._ref_images, host._tgt_image[None]), 0), ids)
        return {f"{role}{i}": t[b] for i, t in zip(ids, toks) for b, role in ((0, "r"), (1, "q"))}
    finally:
        host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None


def infer_episode(task):
    """One episode's library. Label-free: the packet is opened for cov, score and the listed cached fields only."""
    import numpy as np
    import torch
    from ics.experiment import load_inputs, packet, sha
    from ics.methods import aux_evidence
    row, cfg, out = task; key, started, seals = row["key"], time.perf_counter(), {}
    (q, r, cov, score), receipt = load_inputs(cfg["root"], row)
    with np.load(packet(cfg["root"], row), allow_pickle=False) as f:
        cached = {k: f[k].copy() for k in aux_evidence.PACKET_FIELDS if k in f.files}
    hmask, receipt["host_sha256"] = read_mask(cfg["host"], key, seals)
    props, sizes = {}, {}
    for name, src in cfg["proposers"].items():
        mask, receipt[f"proposer_sha256:{name}"] = read_mask(src, key, seals)
        field = None
        if src["fields"]:
            with np.load(Path(src["fields"]) / f"{key}.npz", allow_pickle=False) as f: field = f[src["field_key"]].astype(np.float32)
        props[name] = (mask, field); sizes[name] = dict(add=int((mask & ~hmask).sum()), delete=int((hmask & ~mask).sum()))
    layers = bases = None
    if _FORWARD is not None:
        layers, bases = forward_layers(row), _FORWARD["bases"]
    elif cfg["layers"]:
        with np.load(Path(cfg["layers"]) / f"{key}.npz", allow_pickle=False) as f: layers = {k: f[k] for k in f.files}
    if cfg["bases"] and _FORWARD is None: bases = torch.load(cfg["bases"], map_location="cpu", weights_only=True)
    names, maps, info = aux_evidence.library(key, q, r, cov, score, hmask, props, packet=cached, layers=layers, bases=bases, device=cfg["device"])
    if cfg["device"] == "cuda": torch.cuda.synchronize()
    dest = Path(out) / "evidence" / f"{key}.npz"
    np.savez_compressed(dest, maps=maps)
    return dict(key=key, names=names, sha=sha(dest), receipt=receipt, sealed_sources={k: v is not None for k, v in seals.items()},
                audit=dict(key=key, seconds=time.perf_counter() - started, edit_pixels=sizes, info=info))


def pool(function, tasks, workers, threads):
    if workers <= 1:
        start_worker(threads)
        yield from map(function, tasks)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=start_worker, initargs=(threads,)) as ex:
        yield from ex.map(function, tasks, chunksize=1)


def infer(a):
    from ics.experiment import load_rows, sha
    from ics.methods import aux_evidence
    if a.out.exists(): raise FileExistsError("Use a fresh output directory")
    if a.device == "cuda" and a.workers != 1: raise ValueError("One process on the GPU; use --workers for CPU inference")
    rows = load_rows(a.manifest)
    if a.limit:  # round robin across folds, as the cache runner does for a smoke
        folds = sorted({r["fold"] for r in rows}); bins = {f: [r for r in rows if r["fold"] == f] for f in folds}; ordered = []
        while any(bins.values()):
            for f in folds:
                if bins[f]: ordered.append(bins[f].pop(0))
        rows = ordered[:a.limit]
    host, proposers = source(a.host, named=False), dict(source(s) for s in a.proposer)
    if len(proposers) != len(a.proposer) or UNIVERSE in proposers: raise ValueError(f"Proposer names must be unique and not '{UNIVERSE}'")
    (a.out / "evidence").mkdir(parents=True)
    (a.out / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    cfg = dict(root=str(a.root.resolve()), host=host, proposers=proposers, layers=str(a.layers.resolve()) if a.layers else None,
               bases=str(a.bases.resolve()) if a.bases else None, device=a.device, forward_layers=None)
    if a.forward_layers:  # more layers than were saved: one extra paired forward per episode, nothing stored but the maps
        global _FORWARD
        import torch
        from ics.foris import build_host
        ids = sorted({int(x) for x in a.forward_layers.split(",")})
        if a.device != "cuda" or a.layers or a.bases or len(ids) < 2: raise ValueError("--forward-layers needs --device cuda, at least two layers, and no --layers/--bases")
        man = json.loads(a.manifest.read_text())
        if not isinstance(man, dict) or not man.get("projection_basis"): raise ValueError("--forward-layers needs the manifest with data roots and the frozen native basis")
        fhost = build_host(man, "cuda", a.foris_root)
        with torch.inference_mode(): fbases = black_bases(fhost, ids)
        torch.save({k: v.cpu() for k, v in fbases.items()}, a.out / "bases.pt")
        _FORWARD = dict(host=fhost, ids=ids, data=Path(man["data_root"]), ann=Path(man["annotation_root"]), bases=fbases)
        cfg.update(forward_layers=ids, bases=str((a.out / "bases.pt").resolve()), encoder_instances=1, pair_forwards_per_episode=1, black_forwards=1)
    seal = dict(state="INFERRING", manifest_sha256=sha(a.out / "manifest.json"), evidence={}, inputs={})
    names, audits, checked = None, [], {}
    for got in pool(infer_episode, [(row, cfg, str(a.out)) for row in rows], a.workers, a.threads):
        if names is None: names = got["names"]
        if got["names"] != names: raise ValueError(f"{got['key']}: the library changed between episodes")
        seal["evidence"][got["key"]] = got["sha"]; seal["inputs"][got["key"]] = got["receipt"]; audits.append(got["audit"]); checked.update(got["sealed_sources"])
        print(json.dumps(dict(key=got["key"], seconds=round(got["audit"]["seconds"], 2), maps=len(names))), flush=True)
    (a.out / "audits.json").write_text(json.dumps(audits, indent=2) + "\n")
    config = dict(library=aux_evidence.CONFIG, maps=names, source_sha256=sha(aux_evidence.__file__), bases_sha256=sha(a.bases) if a.bases else None,
                  threads=a.threads, workers=a.workers, sources_checked_against_a_seal=checked,
                  parameter_selection="library constants fixed before any query scoring; budgets are chosen across folds in score", **cfg)
    (a.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    seal.update(state="ALL_EVIDENCE_SEALED", config_sha256=sha(a.out / "config.json"))
    (a.out / "sealed.json").write_text(json.dumps(seal, indent=2) + "\n")


def count_episode(task):
    """Truth is read here: per token, the good and bad pixels of every operator's edit set."""
    import numpy as np
    from ics.experiment import packet, unpack, sha
    from ics.edit_counts import OPS, blocks, edit_set
    row, root, host, proposers, evidence_sha, packet_sha, out = task; key, seals = row["key"], {}
    if sha(Path(out) / "evidence" / f"{key}.npz") != evidence_sha: raise ValueError("Evidence changed after sealing")
    if sha(packet(root, row)) != packet_sha: raise ValueError("Evaluation packet changed")
    with np.load(packet(root, row), allow_pickle=False) as f: truth, cached = unpack(f["truth"]), unpack(f["native"])
    hmask, _ = read_mask(host, key, seals)
    got = dict(key=key, base=[int((hmask & truth).sum()), int((hmask | truth).sum())], host=int(hmask.sum()),
               replay=int((hmask != cached).sum()), missed=int((truth & ~hmask).sum()), false=int((hmask & ~truth).sum()), full={}, good={}, bad={})
    for p in list(proposers) + [UNIVERSE]:
        mask = None if p == UNIVERSE else read_mask(proposers[p], key, seals)[0]
        if mask is not None: got["full"][p] = [int((mask & truth).sum()), int((mask | truth).sum())]
        for op in OPS:
            e = edit_set(hmask, mask, op); g = e & (truth if op == "add" else ~truth)
            got["good"][p, op], got["bad"][p, op] = blocks(g).astype(np.uint16), blocks(e & ~g).astype(np.uint16)
    with np.load(Path(out) / "evidence" / f"{key}.npz", allow_pickle=False) as f: got["maps"] = f["maps"]
    return got


def score(a):
    import numpy as np
    from ics.experiment import packet, unpack, sha, metric, photo_groups, summarize
    from ics.edit_counts import (BINS, OPS, SHARES, BUDGETS, edit_set, histograms, from_top, thresholds, pooled_thresholds,
                                 counts_at, compose, selectivity, accepted)
    from ics.methods.aux_evidence import is_control, is_placebo
    out = a.out; seal = json.loads((out / "sealed.json").read_text())
    if seal.get("state") != "ALL_EVIDENCE_SEALED": raise ValueError("Evidence is not sealed")
    if sha(out / "manifest.json") != seal["manifest_sha256"] or sha(out / "config.json") != seal["config_sha256"]:
        raise ValueError("Manifest or config changed after inference")
    if (out / "report.json").exists(): raise FileExistsError("This run is already scored")
    rows = json.loads((out / "manifest.json").read_text()); config = json.loads((out / "config.json").read_text())
    host, proposers, maps = config["host"], config["proposers"], config["maps"]; names = list(proposers); n, m = len(rows), len(maps)
    cls, fold, groups = np.array([r["c"] for r in rows]), np.array([r["fold"] for r in rows]), photo_groups(rows)

    # pass 1 (parallel): per-token counts of every operator, and the sealed maps
    tasks = [(row, str(a.root), host, proposers, seal["evidence"][row["key"]], seal["inputs"][row["key"]]["packet_sha256"], str(out)) for row in rows]
    operators = [(p, op) for p in names + [UNIVERSE] for op in OPS]
    base, hsize, err = np.zeros((n, 2)), np.zeros(n), np.zeros((n, 2))
    good = {o: np.zeros((n, 4096), np.float32) for o in operators}; bad = {o: np.zeros((n, 4096), np.float32) for o in operators}
    full = {p: np.zeros((n, 2)) for p in names}; bins = np.zeros((n, m, 4096), np.uint8); replay = []
    for i, got in enumerate(pool(count_episode, tasks, a.workers, 1)):
        base[i], hsize[i], err[i] = got["base"], got["host"], (got["missed"], got["false"])
        if got["replay"]: replay.append([got["key"], got["replay"]])
        for p in names: full[p][i] = got["full"][p]
        for o in operators: good[o][i], bad[o][i] = got["good"][o], got["bad"][o]
        if got["maps"].shape != (m, 4096): raise ValueError("Evidence shape differs from the sealed library")
        bins[i] = got["maps"]
        if i % 40 == 0: print(f"counted {i + 1}/{n}", flush=True)

    # class mIoU on all episodes and on each fold's fitting episodes, vectorised over candidates and bootstrap draws
    ids, inv = np.unique(cls, return_inverse=True); onehot = np.eye(len(ids))[inv]
    g = int(groups.max()) + 1; draws = np.random.RandomState(0).randint(g, size=(2000, g))
    weights = np.stack([np.bincount(d, minlength=g) for d in draws])[:, groups].astype(np.float64)
    np.seterr(divide="ignore", over="ignore", invalid="ignore")  # some BLAS builds warn inside matmul; results are checked below
    present = (weights @ onehot) > 0
    def miou(iu, ix=None):
        c = onehot if ix is None else onehot[ix][:, onehot[ix].sum(0) > 0]
        x = iu if ix is None else iu[..., ix, :]
        v = 100 * ((x[..., 0] @ c) / np.maximum(x[..., 1] @ c, 1)).mean(-1)
        if not np.isfinite(v).all(): raise FloatingPointError("Nonfinite class mIoU")
        return v
    def boot(iu):
        i, u = (weights * iu[:, 0]) @ onehot, (weights * iu[:, 1]) @ onehot
        v = 100 * ((i / np.maximum(u, 1)) * present).sum(1) / present.sum(1)
        if not np.isfinite(v).all(): raise FloatingPointError("Nonfinite bootstrap class mIoU")
        return v
    if abs(miou(base) - metric(base, cls)) > 1e-9: raise AssertionError("Vectorised class mIoU differs from the shared metric")
    native, base_boot = miou(base), boot(base)
    def ci(iu, ref=None):
        s = boot(iu) - (base_boot if ref is None else boot(ref)); return [float(v) for v in np.percentile(s, [2.5, 97.5])]
    train = {}
    for f in sorted(set(fold.tolist())):  # fit on the other folds, without episodes sharing a photograph with this fold
        test = fold == f; train[f] = np.flatnonzero((~test) & ~np.isin(groups, groups[test]))
    base_train = {f: miou(base, ix) for f, ix in train.items()}
    side = lambda op, c: compose(base, add=c) if op == "add" else compose(base, delete=c)

    # the sweep
    kinds = [("placebo" if is_placebo(x) else "control" if is_control(x) else "candidate") for x in maps]
    K = a.top; best = {(f, op, k): [] for f in train for op in OPS for k in ("none", "placebo", "control", "candidate")}
    def keep(f, op, kind, gain, rule, counts, thr):
        lst = best[f, op, kind]; lst.append((float(gain), rule, counts.astype(np.float32), np.broadcast_to(thr, (n,)).astype(np.int16)))
        lst.sort(key=lambda t: -t[0]); del lst[K:]
    table, whole, started = [], {}, time.perf_counter()
    for o in operators:
        p, op = o; universe = p == UNIVERSE
        levels = np.array(BUDGETS if universe else SHARES); ref = hsize if universe else (good[o] + bad[o]).sum(1)
        tot = np.stack([good[o].sum(1), bad[o].sum(1)], 1).astype(np.float64)
        if not universe:  # without an auxiliary: the whole edit set or none of it, chosen on the fitting folds
            iu = base.copy(); picks = {}
            for f, tr in train.items():
                gain = miou(side(op, tot), tr) - base_train[f]; take = gain > 0; picks[int(f)] = bool(take)
                if take: iu[fold == f] = side(op, tot)[fold == f]
                keep(f, op, "none", gain, (p, op, None, "all", 1.0), tot, 0)
            whole[o] = iu
        for j, x in enumerate(maps):
            hist = histograms(bins[:, j], good[o], bad[o], op); top = from_top(hist); total = top.sum(-1)
            need = levels[:, None] * ref[None]; thr_rank = thresholds(total, need); c_rank = counts_at(top, thr_rank)
            iu, picks, kept = base.copy(), {}, np.zeros(2)
            for f, tr in train.items():
                thr_abs = pooled_thresholds(total[tr], need[:, tr].sum(1)); c_abs = counts_at(top, thr_abs, common=True)
                cand = np.concatenate([c_rank, c_abs]); gains = miou(side(op, cand), tr) - base_train[f]; b = int(gains.argmax())
                mode, lv = ("rank", b) if b < len(levels) else ("absolute", b - len(levels))
                thr = thr_rank[lv] if mode == "rank" else thr_abs[lv]
                keep(f, op, kinds[j], gains[b], (p, op, j, mode, float(levels[lv])), cand[b], thr)
                if gains[b] > 0:
                    iu[fold == f] = side(op, cand[b])[fold == f]; kept += cand[b][fold == f].sum(0); picks[int(f)] = [mode, float(levels[lv])]
                else: picks[int(f)] = None
            sel = selectivity(hist); both = np.isfinite(sel)
            row = dict(proposer=p, op=op, map=x, kind=kinds[j], selectivity_within=float(sel[both].mean()) if both.any() else None,
                       selectivity_pooled=float(selectivity(hist.sum(0, keepdims=True))[0]), episodes_with_both=int(both.sum()),
                       nested_gain=float(miou(iu) - native), ci95=ci(iu), picks=picks,
                       kept_good_share=float(kept[0] / max(tot[:, 0].sum(), 1)), kept_bad_share=float(kept[1] / max(tot[:, 1].sum(), 1)))
            if not universe: row.update(minus_all_or_nothing=float(miou(iu) - miou(whole[o])), minus_all_or_nothing_ci95=ci(iu, whole[o]))
            table.append(row)
        print(f"swept {p}/{op}: {len(maps)} maps, {time.perf_counter() - started:.0f} s", flush=True)

    report = dict(n=n, classes=len(ids), photo_groups=g, host=host, proposers=proposers, native=float(native), maps=m,
                  rows=len(table), host_differs_from_cached_native=replay, levels=BINS, shares=list(SHARES), budgets_of_host_area=list(BUDGETS),
                  bootstrap={"draws": 2000, "rng": "RandomState(0)", "unit": "connected support/query photographs"},
                  exposure="development; not independent confirmation; every row was read on the same episodes")
    missed, false = err[:, 0].sum(), err[:, 1].sum(); roles = {}
    for p in names:  # (1) every proposal as an adder and as a deleter
        roles[p] = dict(complete=dict(gain=float(miou(full[p]) - native), ci95=ci(full[p])))
        for op in OPS:
            t = np.stack([good[p, op].sum(1), bad[p, op].sum(1)], 1).astype(np.float64); gs, bs = t.sum(0)
            roles[p][op] = dict(edit_pixels=int(gs + bs), good_share=float(gs / max(gs + bs, 1)), reach=float(gs / max(missed if op == "add" else false, 1)),
                                gain_now=float(miou(side(op, t)) - native), ci95=ci(side(op, t)),
                                gain_side_effect_halved=float(miou(side(op, t * [1, .5])) - native), gain_side_effect_removed=float(miou(side(op, t * [1, 0])) - native))
    report["roles"] = roles
    calibration = {}
    for o in operators:  # (2) what the sweep finds in noise, per operator, and how many real maps clear it
        key = "minus_all_or_nothing" if o[0] != UNIVERSE else "nested_gain"
        mine = [r for r in table if (r["proposer"], r["op"]) == o]; noise = [r for r in mine if r["kind"] == "placebo"]; real = [r for r in mine if r["kind"] != "placebo"]
        ng = np.array([r[key] for r in noise]); ns = np.array([r["selectivity_within"] or .5 for r in noise])
        calibration["/".join(o)] = dict(statistic=key, placebo_rows=len(noise), placebo_gain_max=float(ng.max()), placebo_gain_p95=float(np.percentile(ng, 95)),
            placebo_selectivity_max=float(ns.max()), placebo_selectivity_min=float(ns.min()),
            real_rows=len(real), real_above_placebo_gain_max=int(sum(r[key] > ng.max() for r in real)),
            real_above_placebo_selectivity_max=int(sum((r["selectivity_within"] or .5) > ns.max() for r in real)),
            real_below_placebo_selectivity_min=int(sum((r["selectivity_within"] or .5) < ns.min() for r in real)))
        for r in mine: r["clears_placebo"] = bool(r["kind"] != "placebo" and r[key] > ng.max())
    report["placebo_calibration"] = calibration
    families = {}
    for r in table:  # (3) by family and space
        if r["kind"] == "placebo": continue
        fam = r["map"].split(".")[0] + ("@" + r["map"].split("@")[1] if "@" in r["map"] else ""); key = f"{fam} | {r['op']}"
        d = families.setdefault(key, dict(rows=0, clear=0, best_gain=-1e9, best=None, sel=[]))
        d["rows"] += 1; d["clear"] += r["clears_placebo"]; d["sel"].append(r["selectivity_within"] or .5)
        if r["nested_gain"] > d["best_gain"]: d["best_gain"], d["best"] = r["nested_gain"], f"{r['proposer']}: {r['map']}"
    report["families"] = {k: dict(rows=d["rows"], rows_clearing_placebo=int(d["clear"]), best_nested_gain=d["best_gain"], best=d["best"],
                                  median_selectivity_within=float(np.median(d["sel"]))) for k, d in sorted(families.items())}
    report["sweep"] = sorted(table, key=lambda r: -r["nested_gain"])

    # (4) the composition: additions of one rule and deletions of another, the whole choice nested over folds
    arms, chosen = {}, {}
    for arm, allowed in (("edit.unfiltered.control", ("none",)), ("edit.placebo.control", ("none", "placebo")),
                         ("edit.simple_auxiliary.control", ("none", "control")), ("edit", ("none", "control", "candidate"))):
        iu, chosen[arm] = base.copy(), {}
        for f, tr in train.items():
            short = {op: sorted((t for k in allowed for t in best[f, op, k]), key=lambda t: -t[0])[:K] + [None] for op in OPS}
            def value(x, y): return miou(compose(base, add=None if x is None else x[2], delete=None if y is None else y[2]), tr)
            x, y = max(((x, y) for x in short["add"] for y in short["delete"]), key=lambda xy: value(*xy))
            iu[fold == f] = compose(base, add=None if x is None else x[2], delete=None if y is None else y[2])[fold == f]; chosen[arm][int(f)] = (x, y)
        arms[arm] = iu
    show = lambda t: None if t is None else dict(proposer=t[1][0], map=None if t[1][2] is None else maps[t[1][2]], mode=t[1][3], level=t[1][4], fit_gain=t[0])
    report["composition"] = {arm: dict(gain=float(miou(iu) - native), ci95=ci(iu), picks={f: dict(add=show(x), delete=show(y)) for f, (x, y) in chosen[arm].items()})
                             for arm, iu in arms.items()}
    for ref in ("edit.unfiltered.control", "edit.placebo.control", "edit.simple_auxiliary.control"):
        report["composition"]["edit"]["minus " + ref] = dict(gain=float(miou(arms["edit"]) - miou(arms[ref])), ci95=ci(arms["edit"], arms[ref]))

    # render the chosen masks with the same thresholds, check them against the counts, seal, write the common report
    (out / "predictions").mkdir(exist_ok=True); arrays, corrections, pseal, seals = {}, {}, {}, {}
    for i, row in enumerate(rows):
        key, f = row["key"], int(fold[i]); hmask, _ = read_mask(host, key, seals)
        masks = {p: read_mask(proposers[p], key, seals)[0] for p in names}
        with np.load(packet(a.root, row), allow_pickle=False) as fh: truth = unpack(fh["truth"])
        def take(t):
            if t is None: return np.zeros_like(hmask)
            p, op, j, _, _ = t[1]; e = edit_set(hmask, None if p == UNIVERSE else masks[p], op)
            return e if j is None else accepted(e, bins[i, j], op, int(t[3][i]))
        produced = {"native": hmask, **masks}
        for arm in arms:
            add, delete = (take(t) for t in chosen[arm][f]); produced[arm] = (hmask & ~delete) | add
            if arm == "edit": produced["edit.add_only"], produced["edit.delete_only"] = hmask | add, hmask & ~delete
            got = [int((produced[arm] & truth).sum()), int((produced[arm] | truth).sum())]
            if got != [int(round(v)) for v in arms[arm][i]]: raise AssertionError(f"{key} {arm}: rendered mask and counts disagree")
        dest = out / "predictions" / f"{key}.npz"
        np.savez_compressed(dest, **{k: np.packbits(v) for k, v in produced.items() if k.startswith("edit")}); pseal[key] = sha(dest)
        for arm, mk in produced.items():
            arrays.setdefault(arm, []).append([int((mk & truth).sum()), int((mk | truth).sum())]); add, delete = mk & ~hmask, hmask & ~mk
            corrections.setdefault(arm, []).append(dict(key=key, c=row["c"], fold=row["fold"], batch=str(row.get("batch", "unspecified")),
                add_TP=int((add & truth).sum()), delete_FP=int((delete & ~truth).sum()), delete_TP=int((delete & truth).sum()), add_FP=int((add & ~truth).sum())))
    (out / "sealed_predictions.json").write_text(json.dumps(dict(state="ALL_PREDICTIONS_SEALED", predictions=pseal,
        note="masks follow from sealed evidence and rules chosen without the held-out fold's truth"), indent=2) + "\n")
    report["complete_masks"], draws_out = summarize(rows, {k: np.asarray(v, np.int64) for k, v in arrays.items()}, corrections)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n"); np.save(out / "bootstrap_photo_draws.npy", draws_out)

    L = [f"# Sweep: {n} episodes, native {native:.2f}, {m} maps, {len(table)} rows", "", "## Proposals as operators", "",
         "| Proposal | Op | Reach | Good share | Now [95%] | Side effect halved | Removed |", "|---|---|---:|---:|---:|---:|---:|"]
    for p in names:
        for op in OPS:
            r = roles[p][op]
            L.append(f"| {p} | {op} | {100 * r['reach']:.1f}% | {100 * r['good_share']:.1f}% | {r['gain_now']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {r['gain_side_effect_halved']:+.2f} | {r['gain_side_effect_removed']:+.2f} |")
    L += ["", "## Placebo calibration (the same sweep on seeded noise maps)", "",
          "| Operator | Statistic | Placebo max / p95 | Real rows above the placebo max | Placebo selectivity min..max | Real rows outside it (above / below) |", "|---|---|---:|---:|---:|---:|"]
    for k, c in calibration.items():
        L.append(f"| {k} | {c['statistic']} | {c['placebo_gain_max']:+.2f} / {c['placebo_gain_p95']:+.2f} | {c['real_above_placebo_gain_max']} of {c['real_rows']} | "
                 f"{c['placebo_selectivity_min']:.3f}..{c['placebo_selectivity_max']:.3f} | {c['real_above_placebo_selectivity_max']} / {c['real_below_placebo_selectivity_min']} |")
    for op in OPS:
        L += ["", f"## Best rows, {op} (nested gain vs native; budget chosen on three folds)", "",
              "| Proposal | Map | Kind | Nested gain [95%] | Minus all-or-nothing | Selectivity within / pooled | Kept good / bad | Clears placebo |", "|---|---|---|---:|---:|---:|---:|---|"]
        for r in [r for r in report["sweep"] if r["op"] == op][:a.show]:
            w = "n/a" if r["selectivity_within"] is None else f"{r['selectivity_within']:.3f}"
            d = f"{r['minus_all_or_nothing']:+.2f}" if "minus_all_or_nothing" in r else "n/a"
            L.append(f"| {r['proposer']} | {r['map']} | {r['kind']} | {r['nested_gain']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {d} | {w} / {r['selectivity_pooled']:.3f} | "
                     f"{100 * r['kept_good_share']:.0f}% / {100 * r['kept_bad_share']:.0f}% | {'yes' if r['clears_placebo'] else 'no'} |")
    L += ["", "## Families (real maps only)", "", "| Family and space, op | Rows | Clearing placebo | Best nested gain | Best row | Median selectivity within |", "|---|---:|---:|---:|---|---:|"]
    for k, d in report["families"].items():
        L.append(f"| {k} | {d['rows']} | {d['rows_clearing_placebo']} | {d['best_nested_gain']:+.2f} | {d['best']} | {d['median_selectivity_within']:.3f} |")
    L += ["", "## Composition (additions of one rule, deletions of another; whole choice nested over folds)", ""]
    for arm, c in report["composition"].items():
        L.append(f"- `{arm}`: {c['gain']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] vs native")
    for ref in ("edit.unfiltered.control", "edit.placebo.control", "edit.simple_auxiliary.control"):
        d = report["composition"]["edit"]["minus " + ref]; L.append(f"- `edit` minus `{ref}`: {d['gain']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]")
    L += ["", "Picks of `edit` by fold: " + json.dumps(report["composition"]["edit"]["picks"]), ""]
    (out / "report.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["infer", "score", "all"])
    p.add_argument("--root", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path); p.add_argument("--host"); p.add_argument("--proposer", action="append", default=[])
    p.add_argument("--layers", type=Path); p.add_argument("--bases", type=Path)
    p.add_argument("--forward-layers", help="comma-separated 1-based blocks to tap in one paired forward per episode, e.g. 4,8,12,16,20,22,24")
    p.add_argument("--foris-root")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu"); p.add_argument("--threads", type=int, default=1)
    p.add_argument("--workers", type=int, default=1, help="CPU processes: episodes in infer (CPU device) and in the counting pass of score")
    p.add_argument("--limit", type=int); p.add_argument("--top", type=int, default=8, help="rules kept per fold, operator and kind for the pair search")
    p.add_argument("--show", type=int, default=25, help="rows printed per operator")
    a = p.parse_args()
    if a.stage in ("infer", "all"):
        if not (a.manifest and a.host): p.error("infer needs --manifest and --host")
        infer(a)
    if a.stage in ("score", "all"):
        score(a)


if __name__ == "__main__":
    main()
