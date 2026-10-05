#!/usr/bin/env python3
"""The raw-model origin, both public pipelines stage by stage, and the family that contains them.

infer   Label-free and CPU only (worker processes; it can run beside GPU work). For every episode it reads the
        saved final-layer tokens of the D run, the reference mask and the query image, and seals the origin
        `model.raw_nn`, `model.raw_mean`, every stage of INSID3 and of FoRIS, FoRIS with one term removed at a time,
        and the term fields (src/ics/methods/stage_bank.py). The rebuilt FoRIS stages are compared with the stage
        responses cached from the public code (packet s2, s3, score); no truth, native or pre is opened here.
        The output directory is a sealed source in the format the other runners accept (`--origin OUT:model.raw_nn`).
score   Reads truth. (1) Every sealed mask as an edit of the origin: added true / added false / deleted false /
        deleted true pixels, the class mIoU of its additions alone, of its deletions alone and of both; and the
        same counts along each pipeline's own chain. (2) The family: any weighting of the terms of both pipelines,
        scaled per image or not, cut by a threshold or by the self-consistent rule, rendered either way. FoRIS,
        INSID3 and the origin are points of it. One-at-a-time lines through those points give each component's
        response; random points mix them; the best point is chosen on three folds and read on the fourth.
        (3) The first-order value of including each score level, which is what the decision theory in
        evidence/local/research_20261005/objective.md predicts changes sign at the optimal cut.
All episodes are development data. Family masks have no CRF; complete FoRIS (`native`) has.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHAINS = {"insid3": ["model.raw_nn", "insid3.vote", "insid3.candidates", "insid3.seed", "insid3.final"],
          "foris": ["model.raw_nn", "foris.s2", "foris.s3_vote", "foris.s3", "foris.s4_penalty", "foris.pre", "native"]}
PARITY = {"s2": "foris.s2", "s3": "foris.s3", "score": "foris.pre"}
WORKER = {}


def write_json(path, value):
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(value, indent=2) + "\n"); tmp.replace(path)


def write_npz(path, values):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


# ---------------------------------------------------------------- infer

def start_worker(cfg):
    import torch
    from ics.native_basis import load_native_basis
    torch.set_num_threads(cfg["threads"])
    WORKER.update(cfg=cfg, basis=load_native_basis(cfg["projection_basis"])[0])


def infer_episode(row):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from ics.experiment import packet, sha
    from ics.methods import stage_bank as bank
    cfg, key = WORKER["cfg"], row["key"]; out = Path(cfg["out"]); started = time.monotonic()
    layer = Path(cfg["layers"]) / f"{key}.npz"
    receipt = dict(layers_sha256=sha(layer))
    if cfg["layer_seal"] is not None and cfg["layer_seal"].get(key) != receipt["layers_sha256"]:
        raise ValueError(f"Saved layers differ from their seal: {key}")
    with np.load(layer, allow_pickle=False) as z:
        q, r = z["q24"], z["r24"]
    with Image.open(Path(cfg["annotation_root"]) / Path(row["support"]).with_suffix(".png")) as im:
        mask = torch.from_numpy((np.asarray(im) == row["c"] + 1).copy())
    mask = F.interpolate(mask[None, None].float(), (1024, 1024), mode="nearest")[0, 0] > .5   # FoRIS set_reference
    with Image.open(Path(cfg["data_root"]) / row["query"]) as im:
        image = transforms.ToTensor()(transforms.Resize((1024, 1024))(im.convert("RGB")))       # FoRIS build_transform
    rgb = F.interpolate(image[None], (64, 64), mode="bilinear", align_corners=False)[0].clamp(0, 1).reshape(3, -1).T
    fields, masks, info = bank.episode(q, r, mask, rgb, WORKER["basis"], tau=cfg["tau"], never=cfg["never"])
    parity = {}
    if cfg["root"]:
        with np.load(packet(cfg["root"], row), allow_pickle=False) as p:   # cached stage responses only
            for cached, rebuilt in PARITY.items():
                parity[cached] = float(np.abs(p[cached].astype(np.float32) - fields[rebuilt]).max())
        receipt["packet_sha256"] = sha(packet(cfg["root"], row))
    write_npz(out / "fields" / f"{key}.npz", fields)
    write_npz(out / "predictions" / f"{key}.npz", {k: np.packbits(v) for k, v in masks.items()})
    return dict(key=key, info=info, parity=parity, receipt=receipt, seconds=time.monotonic() - started,
                arms=sorted(masks), field_names=sorted(fields),
                predictions=sha(out / "predictions" / f"{key}.npz"), fields=sha(out / "fields" / f"{key}.npz"))


def infer(a):
    import multiprocessing as mp
    from ics.experiment import load_rows, sha
    from ics.methods import stage_bank as bank
    manifest = json.loads(a.manifest.read_text())
    rows = load_rows(a.manifest)[:a.limit]
    if a.expected and len(rows) != a.expected:
        raise ValueError(f"Expected {a.expected} episodes, found {len(rows)}")
    if a.out.exists():
        raise FileExistsError("Choose a fresh output directory; existing outputs are never replaced")
    seal_path = a.layers.parent / "sealed.json"
    layer_seal = json.loads(seal_path.read_text()).get("layers") if seal_path.exists() else None
    for folder in ("fields", "predictions"):
        (a.out / folder).mkdir(parents=True)
    keep = ("fold", "e", "c", "key", "support", "query", "batch")
    write_json(a.out / "manifest.json", [{k: r[k] for k in keep if k in r} for r in rows])
    cfg = dict(bank.CONFIG, tau=a.tau, never=not a.no_never, threads=a.threads, out=str(a.out), layers=str(a.layers),
               root=str(a.root) if a.root else None, layer_seal=layer_seal, source_manifest=str(a.manifest),
               source_manifest_sha256=sha(a.manifest), layers_run_seal_sha256=sha(seal_path) if seal_path.exists() else None,
               token_space="encoder final layer, norm applied, L2-normalised per token, before any projection",
               **{k: manifest[k] for k in ("data_root", "annotation_root", "projection_basis")})
    write_json(a.out / "config.json", {k: v for k, v in cfg.items() if k != "layer_seal"})
    seal = dict(state="INFERRING", manifest_sha256=sha(a.out / "manifest.json"), config_sha256=sha(a.out / "config.json"),
                predictions={}, fields={}, inputs={}, query_labels_opened=False)
    audit, started = {}, time.monotonic()
    with mp.get_context("spawn").Pool(a.workers, initializer=start_worker, initargs=(cfg,)) as pool:
        for done, got in enumerate(pool.imap_unordered(infer_episode, rows), 1):
            key = got["key"]
            seal["predictions"][key], seal["fields"][key], seal["inputs"][key] = got["predictions"], got["fields"], got["receipt"]
            audit[key] = dict(info=got["info"], parity=got["parity"], seconds=got["seconds"])
            arms, names = got["arms"], got["field_names"]
            print(json.dumps(dict(completed=done, total=len(rows), key=key, seconds=round(got["seconds"], 1), parity=got["parity"])), flush=True)
    worst = {k: max((v["parity"].get(k, 0.) for v in audit.values()), default=0.) for k in PARITY}
    write_json(a.out / "audit.json", dict(episodes=audit, arms=arms, fields=names, elapsed_seconds=time.monotonic() - started,
               parity_max_abs=worst, projected=int(sum(v["info"]["foris"]["projected"] for v in audit.values())),
               parity_note="largest absolute difference between the rebuilt FoRIS stage responses and the cached ones"))
    seal.update(state="ALL_PREDICTIONS_SEALED", arms=arms, field_names=names)
    write_json(a.out / "sealed.json", seal)
    print(json.dumps(dict(state=seal["state"], episodes=len(rows), parity_max_abs=worst)))


# ---------------------------------------------------------------- score

def four_way(mask, base, truth):
    add, delete = mask & ~base, base & ~mask
    return [int((add & truth).sum()), int((add & ~truth).sum()), int((delete & ~truth).sum()), int((delete & truth).sum())]


def load_scored(a):
    """Every sealed mask against truth and against the origin; the family fields and the truth as tensors."""
    import torch
    from ics.experiment import packet, sha, unpack
    from ics.methods import stage_bank as bank
    seal = json.loads((a.out / "sealed.json").read_text())
    if seal.get("state") != "ALL_PREDICTIONS_SEALED" or sha(a.out / "manifest.json") != seal["manifest_sha256"]:
        raise ValueError("Stages are not sealed, or the manifest changed")
    rows = json.loads((a.out / "manifest.json").read_text()); origin = bank.CONFIG["origin"]
    arms = seal["arms"] + ["native"]
    iu = {arm: np.zeros((len(rows), 2), np.int64) for arm in arms}
    edits = {arm: np.zeros((len(rows), 4), np.int64) for arm in arms}
    chain = {name: np.zeros((len(rows), len(steps) - 1, 4), np.int64) for name, steps in CHAINS.items()}
    truth = torch.zeros((len(rows), 1024, 1024), dtype=torch.bool); fam = np.zeros((len(rows), len(bank.FAMILY), 4096), np.float32)
    pre_differs = np.zeros(len(rows), np.int64)
    for i, row in enumerate(rows):
        key = row["key"]
        for group in ("predictions", "fields"):
            if sha(a.out / group / f"{key}.npz") != seal[group][key]:
                raise ValueError(f"Sealed {group} changed: {key}")
        if "packet_sha256" in seal["inputs"][key] and sha(packet(a.root, row)) != seal["inputs"][key]["packet_sha256"]:
            raise ValueError(f"Evaluation packet changed: {key}")
        with np.load(a.out / "predictions" / f"{key}.npz", allow_pickle=False) as p:
            masks = {arm: unpack(p[arm]) for arm in seal["arms"]}
        with np.load(packet(a.root, row), allow_pickle=False) as p:
            t, masks["native"], pre = unpack(p["truth"]), unpack(p["native"]), unpack(p["pre"])
        with np.load(a.out / "fields" / f"{key}.npz", allow_pickle=False) as p:
            fam[i] = bank.family_fields(p)
        pre_differs[i] = int((masks["foris.pre"] != pre).sum()); truth[i] = torch.from_numpy(t)
        for arm, m in masks.items():
            iu[arm][i] = (m & t).sum(), (m | t).sum()
            edits[arm][i] = four_way(m, masks[origin], t)
        for name, steps in CHAINS.items():
            for j in range(len(steps) - 1):
                chain[name][i, j] = four_way(masks[steps[j + 1]], masks[steps[j]], t)
    return rows, seal, iu, edits, chain, truth, torch.from_numpy(fam), pre_differs


def fixed_point(p, steps=30):
    """Self-consistent cut of a posterior field p [N, T]: keep a token iff its odds exceed the expected IoU of the
    kept set. -> probability thresholds [N]"""
    import torch
    j = torch.full((p.shape[0],), .5, device=p.device)
    for _ in range(steps):
        keep = (p > (j / (1 + j))[:, None]).float()
        j = (keep * p).sum(1) / (p.sum(1) + (keep * (1 - p)).sum(1)).clamp_min(1e-6)
    return j / (1 + j)


def evaluate(points, fam, truth, device, chunk=24):
    """Exact I and U at 1024 for every point. -> int64 [P, N, 2]"""
    import torch
    import torch.nn.functional as F
    n = fam.shape[0]; fam = fam.to(device); out = np.zeros((len(points), n, 2), np.int64)
    area = truth.reshape(n, -1).sum(1).numpy()
    up = lambda x: F.interpolate(x.reshape(-1, 1, 64, 64), (1024, 1024), mode="bilinear", align_corners=False)[:, 0]
    for pi, pt in enumerate(points):
        w = torch.as_tensor(pt["w"], dtype=torch.float32, device=device)
        s = torch.einsum("k,nkt->nt", w, fam)
        if pt["scale"] == "minmax":
            s = s - s.min(1, keepdim=True).values
            s = s / s.max(1, keepdim=True).values.clamp_min(1e-6)
        if pt.get("rule") == "fixed_point":
            theta = fixed_point(torch.sigmoid(pt["a"] * (s - pt["b"])))
            t = pt["b"] + torch.logit(theta.clamp(1e-4, 1 - 1e-4)) / pt["a"]
        else:
            t = torch.full((n,), float(pt["t"]), device=device)
        for lo in range(0, n, chunk):
            sl = slice(lo, lo + chunk); tt = t[sl, None, None]
            m = up(s[sl]) > tt if pt["render"] == "field" else up((s[sl] > t[sl, None]).float()) > .5
            g = truth[sl].to(device)
            inter = (m & g).flatten(1).sum(1).cpu().numpy()
            out[pi, sl, 0] = inter; out[pi, sl, 1] = m.flatten(1).sum(1).cpu().numpy() + area[sl] - inter
    return out


def design(fam, n_random, seed):
    """Anchors, one-at-a-time lines through them, pairwise mixtures, random points. Label-free."""
    from ics.methods.stage_bank import ANCHORS, FAMILY
    K = len(FAMILY); spread = fam.std(dim=(0, 2)).numpy() + 1e-6
    vec = lambda w: [float(w.get(k, 0.)) for k in FAMILY]
    def point(name, kind, w, scale, render, t=None, **rule):
        return dict(name=name, kind=kind, w=list(map(float, w)), scale=scale, render=render, t=t, **rule)
    pts = []
    for name, a in ANCHORS.items():
        pts.append(point(name, "anchor", vec(a["w"]), a["scale"], a["render"], a["t"]))
    f, i3 = ANCHORS["foris.pre"], ANCHORS["insid3.final"]
    for k in f["w"]:
        for m in (0., .25, .5, .75, 1.5, 2., 3.):
            pts.append(point(f"foris:{k} x{m:g}", "line", vec(dict(f["w"], **{k: f["w"][k] * m})), "minmax", "field", .5))
    for t in (.3, .35, .4, .45, .55, .6, .65, .7):
        pts.append(point(f"foris:t={t:g}", "threshold", vec(f["w"]), "minmax", "field", t))
    pts.append(point("foris:render=mask", "line", vec(f["w"]), "minmax", "mask", .5))
    for a_, b_ in ((4., .5), (8., .5), (16., .5), (8., .4), (8., .6), (16., .4), (16., .6)):
        pts.append(point(f"foris:fixed_point a={a_:g} b={b_:g}", "rule", vec(f["w"]), "minmax", "field", rule="fixed_point", a=a_, b=b_))
    for k in i3["w"]:
        for m in (0., .5, 1.5, 2., 3.):
            pts.append(point(f"insid3:{k} ^{m:g}", "line", vec(dict(i3["w"], **{k: m})), "abs", "mask", i3["t"]))
    for t in (.05, .1, .15, .25, .3, .4):
        pts.append(point(f"insid3:t={t:g}", "threshold", vec(i3["w"]), "abs", "mask", math.log(t)))
    pts.append(point("insid3:render=field", "line", vec(i3["w"]), "abs", "field", i3["t"]))
    for name in ("model.raw_nn", "model.raw_mean"):
        for t in (-.1, -.05, -.02, .02, .05, .1):
            pts.append(point(f"{name}:t={t:g}", "threshold", vec(ANCHORS[name]["w"]), "abs", "mask", t))
    typical = lambda base: float(np.sqrt(sum((w * spread[FAMILY.index(k)]) ** 2 for k, w in base["w"].items())))
    sign = lambda k: -1. if k in ("foris.bg", "foris.penalty") else 1.
    for k in [k for k in FAMILY if k not in f["w"]]:
        for g in (.1, .25, .5, 1.):
            w = dict(f["w"], **{k: g * typical(f) / spread[FAMILY.index(k)]})
            pts.append(point(f"foris+{k} x{g:g}", "mixture", vec(w), "minmax", "field", .5))
    rng = np.random.RandomState(seed)
    quantiles = lambda w, q: float(np.quantile(np.einsum("k,nkt->nt", np.asarray(w, np.float32), fam.numpy())[::4, ::4], q))
    for j in range(n_random):
        base = f if rng.rand() < .7 else i3
        w = {k: v * math.exp(.5 * rng.randn()) * (rng.rand() > .15) for k, v in base["w"].items()}
        for k in FAMILY:
            if k not in base["w"] and rng.rand() < .3:
                w[k] = sign(k) * typical(base) / spread[FAMILY.index(k)] * math.exp(rng.randn() - 1.)
        w = vec(w)
        if not any(w): continue
        scale, render = ("minmax", "abs")[rng.rand() < .35], ("field", "mask")[rng.rand() < .3]
        if scale == "minmax" and render == "field" and rng.rand() < .3:
            pts.append(point(f"random{j}", "random", w, scale, render, rule="fixed_point",
                             a=float(math.exp(rng.uniform(math.log(3.), math.log(30.)))), b=float(rng.uniform(.3, .7))))
        else:
            t = float(rng.uniform(.3, .7)) if scale == "minmax" else quantiles(w, rng.uniform(.55, .97))
            pts.append(point(f"random{j}", "random", w, scale, render, t))
    return pts


def value_curve(field, base_w, truth, iu_base, classes, device, bins=20, chunk=24):
    """First-order value, in mIoU points, of the pixels at each level of a response: sum over classes of
    (true - J_c * false) / U_c, with J_c and U_c of the pipeline's own mask. Positive levels are worth including."""
    import torch
    import torch.nn.functional as F
    ids, cls = np.unique(classes, return_inverse=True); C = len(ids)
    i_c = np.bincount(cls, weights=iu_base[:, 0], minlength=C); u_c = np.maximum(np.bincount(cls, weights=iu_base[:, 1], minlength=C), 1)
    j_c = i_c / u_c; n = field.shape[0]; good = np.zeros((C, bins)); bad = np.zeros((C, bins))
    for lo in range(0, n, chunk):
        z = F.interpolate(field[lo:lo + chunk].reshape(-1, 1, 64, 64).to(device), (1024, 1024), mode="bilinear", align_corners=False)[:, 0]
        b = (z.clamp(0, 1 - 1e-6) * bins).long(); g = truth[lo:lo + chunk].to(device)
        for k in range(b.shape[0]):
            c = cls[lo + k]
            good[c] += torch.bincount(b[k][g[k]], minlength=bins).cpu().numpy()
            bad[c] += torch.bincount(b[k][~g[k]], minlength=bins).cpu().numpy()
    value = 100 * ((good - j_c[:, None] * bad) / u_c[:, None]).sum(0) / C
    return dict(levels=[(k + .5) / bins for k in range(bins)], true=good.sum(0).tolist(), false=bad.sum(0).tolist(),
                pooled_purity=(good.sum(0) / np.maximum(good.sum(0) + bad.sum(0), 1)).tolist(), value_points=value.tolist(),
                pooled_break_even=float(i_c.sum() / (i_c.sum() + u_c.sum())), note=f"levels of {base_w}; a level is worth including when its value is positive")


def score(a):
    import torch
    from ics.experiment import metric, photo_groups, summarize
    from ics.methods import stage_bank as bank
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads)
    rows, seal, iu, edits, chain, truth, fam, pre_differs = load_scored(a)
    n, classes, folds, groups = len(rows), np.array([r["c"] for r in rows]), np.array([r["fold"] for r in rows]), photo_groups(rows)
    origin = bank.CONFIG["origin"]; miou = lambda v, ix=slice(None): metric(v[ix], classes[ix])
    # (1) every sealed mask as an edit of the origin
    base = iu[origin].astype(np.float64); ledger = {}
    for arm in iu:
        e = edits[arm].astype(np.float64)
        add = base + np.stack([e[:, 0], e[:, 1]], 1); delete = base - np.stack([e[:, 3], e[:, 2]], 1)
        ledger[arm] = dict(miou=miou(iu[arm]), add_only=miou(add), delete_only=miou(delete),
                           **dict(zip(("add_TP", "add_FP", "delete_FP", "delete_TP"), map(int, edits[arm].sum(0)))))
    chains = {name: [dict(step=f"{steps[j]} -> {steps[j + 1]}", miou_from=miou(iu[steps[j]]), miou_to=miou(iu[steps[j + 1]]),
                          **dict(zip(("add_TP", "add_FP", "delete_FP", "delete_TP"), map(int, chain[name][:, j].sum(0)))))
                     for j in range(len(steps) - 1)] for name, steps in CHAINS.items()}
    # (2) the family
    points = design(fam, a.random, a.seed); table = evaluate(points, fam, truth, device)
    anchors = {p["name"]: i for i, p in enumerate(points) if p["kind"] == "anchor"}
    anchor_check = {name: int(np.abs(table[i] - iu[name]).sum()) for name, i in anchors.items()}
    scores = np.array([miou(t) for t in table])
    def nested(candidates):
        """Best candidate on three folds (episodes sharing a photograph with the read fold left out), read on the fourth."""
        out, chosen = np.zeros((n, 2), np.int64), {}
        for f in sorted(set(folds)):
            read = folds == f; fit = ~read & ~np.isin(groups, groups[read])
            best = candidates[int(np.argmax([miou(table[c], fit) for c in candidates]))]
            out[read] = table[best][read]; chosen[str(f)] = points[best]["name"]
        return out, chosen
    everything = list(range(len(points)))
    subsets = {"family.nested": everything,
               "family.threshold_only.control": [i for i, p in enumerate(points) if p["kind"] in ("anchor", "threshold") and p["name"].startswith("foris")],
               "family.lines_only.control": [i for i, p in enumerate(points) if p["kind"] != "random"],
               "family.foris_terms_only.control": [i for i, p in enumerate(points) if not any(w for k, w in zip(bank.FAMILY, p["w"]) if not k.startswith("foris"))]}
    arrays = {"native": iu["native"], "foris.pre.control": iu["foris.pre"], "insid3.final.control": iu["insid3.final"],
              "model.raw_nn.control": iu[origin]}
    selection = {}
    for name, subset in subsets.items():
        arrays[name], selection[name] = nested(subset)
    best = int(scores.argmax()); arrays["family.best_in_sample"] = table[best]
    corrections = {arm: [dict(key=r["key"], c=r["c"], fold=r["fold"], batch=str(r.get("batch", "unspecified")),
                              add_TP=0, delete_FP=0, delete_TP=0, add_FP=0) for r in rows] for arm in arrays}
    summary, _ = summarize(rows, arrays, corrections)
    for key in ("corrections_vs_native", "corrections_by_class", "corrections_by_batch"):
        summary.pop(key, None)                               # edit counts live in the ledger, relative to the origin
    lines = [dict(name=p["name"], kind=p["kind"], miou=float(scores[i]), gain_vs_foris_pre=float(scores[i] - miou(iu["foris.pre"])),
                  gain_vs_insid3=float(scores[i] - miou(iu["insid3.final"]))) for i, p in enumerate(points) if p["kind"] != "random"]
    order = np.argsort(-scores)[:a.top]
    # (3) the value of each level of each pipeline's own response
    unit = lambda s: (s - s.min(1, keepdim=True).values) / (s.max(1, keepdim=True).values - s.min(1, keepdim=True).values).clamp_min(1e-6)
    weigh = lambda name: torch.einsum("k,nkt->nt", torch.tensor([float(bank.ANCHORS[name]["w"].get(k, 0.)) for k in bank.FAMILY]), fam)
    curves = {"foris.pre": value_curve(unit(weigh("foris.pre")), "the FoRIS response scaled per image; FoRIS cuts at 0.5", truth, iu["foris.pre"], classes, device),
              "insid3.final": value_curve(torch.exp(weigh("insid3.final")), "the INSID3 aggregate; INSID3 cuts at 0.2", truth, iu["insid3.final"], classes, device),
              "model.raw_nn": value_curve(weigh("model.raw_nn") * .5 + .5, "half the nearest-token margin plus one half; the origin cuts at 0.5", truth, iu[origin], classes, device)}
    report = dict(n=n, classes=int(len(set(classes))), photo_groups=int(groups.max()) + 1, origin=origin, device=device,
                  exposure="development; not independent confirmation", ledger_vs_origin=ledger, chains=chains,
                  foris_pre_pixels_differing_from_cached=dict(total=int(pre_differs.sum()), episodes=int((pre_differs > 0).sum())),
                  family=dict(fields=list(bank.FAMILY), points=len(points), random=a.random, seed=a.seed, anchor_iu_abs_difference=anchor_check,
                              component_lines=lines, selection=selection, best_in_sample=dict(points[best], miou=float(scores[best])),
                              top=[dict(points[i], miou=float(scores[i])) for i in order],
                              note="family masks have no CRF; `native` is complete FoRIS with CRF; best_in_sample is optimistic"),
                  complete=summary, value_by_level=curves)
    write_json(a.out / "report.json", report)
    np.save(a.out / "family_iu.npy", table); write_json(a.out / "family_points.json", points)
    L = [f"# Stage bank: {n} episodes, origin {origin} {miou(iu[origin]):.2f}, INSID3 rule {miou(iu['insid3.final']):.2f}, "
         f"FoRIS pre-CRF {miou(iu['foris.pre']):.2f}, complete FoRIS {miou(iu['native']):.2f}", "",
         f"Rebuilt FoRIS pre-CRF differs from the cached one in {int(pre_differs.sum())} pixels over {int((pre_differs > 0).sum())} episodes.", "",
         "## Every mask as an edit of the origin", "", "| mask | mIoU | additions alone | deletions alone | added true | added false | deleted false | deleted true |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, r in sorted(ledger.items(), key=lambda kv: -kv[1]["miou"]):
        L.append(f"| {arm} | {r['miou']:.2f} | {r['add_only']:.2f} | {r['delete_only']:.2f} | {r['add_TP']:,} | {r['add_FP']:,} | {r['delete_FP']:,} | {r['delete_TP']:,} |")
    for name, steps in chains.items():
        L += ["", f"## Chain: {name}", "", "| step | mIoU | added true | added false | deleted false | deleted true |", "|---|---|---:|---:|---:|---:|"]
        L += [f"| {s['step']} | {s['miou_from']:.2f} -> {s['miou_to']:.2f} | {s['add_TP']:,} | {s['add_FP']:,} | {s['delete_FP']:,} | {s['delete_TP']:,} |" for s in steps]
    L += ["", "## Family, chosen on three folds and read on the fourth", ""]
    for name in subsets:
        c = summary["contrasts"][name]
        L.append(f"- `{name}` {summary['scores'][name]:.2f}: " + "; ".join(f"{c[b]['gain']:+.2f} [{c[b]['ci95'][0]:+.2f}, {c[b]['ci95'][1]:+.2f}] vs {b}" for b in c)
                 + f". Chosen: {selection[name]}")
    L += ["", f"Best point in sample (optimistic): {points[best]['name']} {scores[best]:.2f}.", "", "## Component lines (all episodes, no selection)", "",
          "| point | mIoU | vs FoRIS pre-CRF | vs INSID3 rule |", "|---|---:|---:|---:|"]
    L += [f"| {r['name']} | {r['miou']:.2f} | {r['gain_vs_foris_pre']:+.2f} | {r['gain_vs_insid3']:+.2f} |" for r in lines]
    (a.out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))


def self_check():
    """Random tensors only: every designed point evaluates, and the anchors equal the masks the stage code renders."""
    import torch
    from ics.methods import stage_bank as bank
    g = torch.Generator().manual_seed(0); n = 6
    fam = torch.randn(n, len(bank.FAMILY), 4096, generator=g) * .2
    fam[:, [bank.FAMILY.index(k) for k in bank.FAMILY if k.startswith("insid3")]] = torch.log(torch.rand(n, 3, 4096, generator=g).clamp_min(1e-3))
    truth = torch.rand(n, 1, 16, 16, generator=g).repeat_interleave(64, 2).repeat_interleave(64, 3)[:, 0] > .6
    points = design(fam, 40, 0); table = evaluate(points, fam, truth, "cpu", chunk=4)
    assert table.shape == (len(points), n, 2) and (table[:, :, 0] <= table[:, :, 1]).all()
    for i, pt in enumerate(points):
        if pt["kind"] != "anchor": continue
        a = bank.ANCHORS[pt["name"]]
        for e in range(n):
            s = sum(w * fam[e, bank.FAMILY.index(k)] for k, w in a["w"].items())
            s = bank.unit(s) if a["scale"] == "minmax" else s
            m = bank.render_field(s, a["t"]) if a["render"] == "field" else bank.render_mask(s > a["t"])
            assert abs(int((m & truth[e]).sum()) - int(table[i, e, 0])) <= 2 and abs(int((m | truth[e]).sum()) - int(table[i, e, 1])) <= 2, pt["name"]
    p = torch.rand(3, 4096, generator=g); theta = fixed_point(p); j = theta / (1 - theta)
    keep = (p > theta[:, None]).float()
    assert torch.allclose(j, (keep * p).sum(1) / (p.sum(1) + (keep * (1 - p)).sum(1)), atol=1e-4)   # a fixed point
    curve = value_curve(torch.rand(n, 4096, generator=g), "noise", truth, table[0], np.arange(n) % 3, "cpu", chunk=4)
    assert len(curve["value_points"]) == 20
    return f"run_stage_bank self-check passed: {len(points)} points"


def main():
    if sys.argv[1:] == ["self-check"]:
        print(self_check()); return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    i = sub.add_parser("infer"); s = sub.add_parser("score")
    i.add_argument("--manifest", type=Path, required=True, help="cohort with data_root, annotation_root, projection_basis")
    i.add_argument("--layers", type=Path, required=True, help="saved q24/r24 of the D run (its sealed.json is checked)")
    i.add_argument("--root", type=Path, help="cache root with the public stage responses, for the parity audit")
    i.add_argument("--tau", type=float, default=.6); i.add_argument("--no-never", action="store_true", help="skip the FoRIS chain without projection")
    i.add_argument("--workers", type=int, default=10); i.add_argument("--limit", type=int); i.add_argument("--expected", type=int)
    s.add_argument("--root", type=Path, required=True, help="cache root with truth and complete FoRIS")
    s.add_argument("--device", default="auto"); s.add_argument("--random", type=int, default=2000); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--top", type=int, default=30)
    for q in (i, s):
        q.add_argument("--out", type=Path, required=True); q.add_argument("--threads", type=int, default=1 if q is i else 8)
    a = p.parse_args()
    (infer if a.stage == "infer" else score)(a)


if __name__ == "__main__":
    main()
