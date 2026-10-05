#!/usr/bin/env python3
"""Sealed, label-isolated boundary feature diagnostic; this is not a method result.

Infer uses 80 rows (20/fold), sampled from the existing 600-row manifest with
RandomState(0), and only the existing single reference and frozen encoder.
Four reflected +/-4 pixel query shifts provide a 128x128 feature grid. All
FP16 fine features and float32 fields are retained and sealed before score
opens any query truth. Score is CPU-only and labels its GT use explicitly.

python scripts/diagnose_boundary_features.py infer \
    --root outputs/fresh600_root --run outputs/recheck_fresh600_v1 \
    --out outputs/boundary_features80_v1
python scripts/diagnose_boundary_features.py score \
    --root outputs/fresh600_root --out outputs/boundary_features80_v1

Host dependencies and PYTHONPATH are the same as run_subtoken.py. Score
requires NumPy and SciPy; it imports neither torch nor the host/backbone.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ics.experiment import load_rows, packet, photo_groups, sha, unpack

ARMS = ("fine_nn", "bilinear_feature_nn.control", "bilinear_margin.control", "rgb_nn.control")
CONFIG = dict(
    purpose="GT-labeled feature-separability diagnostic; not a deployable method or SOTA result",
    dataset="COCO-20i, historical source600 reuse; not independent confirmation",
    episodes=80, per_fold=20, source_episodes=600, sampling_rng="RandomState(0)",
    sampling="ascending folds; sample source-row indices without replacement, sort indices within fold",
    resolution=1024, coarse_grid=64, fine_grid=128, fine_cell_pixels=8,
    shifts=[[-4, -4], [-4, 4], [4, -4], [4, 4]], padding="reflect, 4 pixels",
    feature_space="same cached last-layer space; host debias only when cached debiased flag is true",
    quantization="normalize fine grid, store FP16, cast FP32 and renormalize for cosine; same for cached q/r",
    reference_roles="single supplied reference, cov>=0.5 FG; cov<0.5 BG; require both",
    margin="max FP32 cosine to reference FG minus max to reference BG; no temperature or tuning",
    coarse_feature_control="unit bilinear interpolation of unit cached q from 64 to 128, align_corners=False",
    coarse_margin_control="bilinear interpolation of cached q/r 64-grid NN margin, align_corners=False",
    rgb_control="query avg_pool8 vs original supplied reference avg_pool16, negative squared RGB distance; FGmax-BGmax",
    local_rcg_diagnostic="nearest-cosine query token among valid 5x5 coarse neighbors of nearest token center; sealed RCG value",
    local_rcg_roles="value>0.5 FG; value<=0.5 BG; true-side availability uses GT only in score",
    query_gt_in_inference=False, encoder_forwards_per_episode=4, feature_retention="retain all80 FP16 fine grids and fields",
    gt_pure_cells="8x8 truth coverage<=0.1 BG or >=0.9 FG",
    gt_boundary="mean same-side distance to opposite GT label in each 8x8 cell <=16 pixels; no band for one-label truth",
    parent_ranking="all FG/BG pure boundary-cell pairs inside each 16x16 parent; ties count 0.5; mean eligible-parent rate per episode",
    primary_statistic="macro per-episode AUROC and macro per-episode parent ranking; exclude episodes lacking both labels",
    bootstrap=dict(draws=2000, rng="RandomState(0)", unit="connected support/query photographs", paired=True),
    tf32=False, tuning=False,
)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_rows(path):
    rows = load_rows(path)
    if len(rows) != 600 or sorted({r["fold"] for r in rows}) != [0, 1, 2, 3]:
        raise ValueError("Require the existing 600-row, four-fold source manifest")
    rng = np.random.RandomState(0)
    chosen = []
    for fold in range(4):
        pool = np.array([i for i, r in enumerate(rows) if r["fold"] == fold])
        if len(pool) < 20:
            raise ValueError("Source fold contains fewer than 20 rows")
        chosen.extend(sorted(int(i) for i in rng.choice(pool, 20, replace=False)))
    return [rows[i] for i in chosen], chosen


def neighborhood():
    # Fine centers are 8*i+4; coarse centers 16*j+8, so nearest j is i//2.
    y, x = np.divmod(np.arange(128 * 128), 128)
    dy, dx = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing="ij")
    yy, xx = y[:, None] // 2 + dy.ravel(), x[:, None] // 2 + dx.ravel()
    valid = (yy >= 0) & (yy < 64) & (xx >= 0) & (xx < 64)
    return np.clip(yy, 0, 63) * 64 + np.clip(xx, 0, 63), valid


def infer(a):
    import torch
    import torch.nn.functional as F
    from PIL import Image

    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, indices = select_rows(a.run / "manifest.json")
    source_seal = json.loads((a.run / "sealed.json").read_text())
    if source_seal["state"] != "ALL_PREDICTIONS_SEALED" or sha(a.run / "manifest.json") != source_seal["manifest_sha256"]:
        raise ValueError("Require the unchanged sealed source cohort")
    if a.out.exists():
        raise FileExistsError("Never overwrite or remove retained diagnostic assets: " + str(a.out))
    ancestor = a.out.parent.resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    estimate = 80 * 128 * 128 * 1024 * 2
    if free < estimate + (1 << 30):
        raise OSError("Need 2.5 GiB for retained fine features plus 1 GiB headroom")
    a.out.mkdir(parents=True)
    (a.out / "features").mkdir()
    (a.out / "fields").mkdir()
    write(a.out / "manifest.json", rows)
    begin = time.monotonic()
    state = dict(state="RUNNING_INFERENCE", stage="infer", pid=os.getpid(),
                 started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 n=80, completed=0, query_truth_opened=False,
                 disk_free_bytes=free, estimated_fine_feature_bytes=estimate)
    write(a.out / "state.json", state)
    sys.path.insert(0, str(a.host_root))
    sys.path.insert(0, str(a.host_root / "scripts"))
    from extent_experiment import build_host

    man = json.loads(a.host_manifest.read_text())
    config = dict(CONFIG, source_code_sha256=sha(Path(__file__)),
                  experiment_sha256=sha(REPO / "src/ics/experiment.py"),
                  source_manifest_sha256=sha(a.run / "manifest.json"),
                  source_seal_sha256=sha(a.run / "sealed.json"),
                  selected_source_indices=indices,
                  source_root=str(a.root.resolve()), source_run=str(a.run.resolve()),
                  host_manifest_sha256=sha(a.host_manifest), host_root=str(a.host_root.resolve()),
                  demo4_root=a.demo4_root, chunk=a.chunk, torch_threads=a.threads,
                  torch_version=str(torch.__version__), device="cuda",
                  device_name=torch.cuda.get_device_name(0), disk_free_bytes_at_start=free)
    write(a.out / "config.json", config)
    dev = torch.device("cuda")
    nb, valid = neighborhood()
    nb = torch.from_numpy(nb).to(dev)
    valid = torch.from_numpy(valid).to(dev)
    receipts, assets = {}, {}

    def margin(x, reference, fg, colour=False):
        pieces = []
        rt = reference.T.contiguous()
        for start in range(0, len(x), a.chunk):
            chunk = x[start:start + a.chunk]
            sim = chunk @ rt
            if colour:
                sim = 2 * sim - chunk.square().sum(1, keepdim=True) - reference.square().sum(1)[None]
            pieces.append(sim[:, fg].max(1).values - sim[:, ~fg].max(1).values)
        return torch.cat(pieces)

    def local_nearest(x, q, rcg):
        pieces = []
        for start in range(0, len(x), a.chunk):
            end = min(start + a.chunk, len(x))
            sim = torch.einsum("pc,pkc->pk", x[start:end], q[nb[start:end]])
            sim = sim.masked_fill(~valid[start:end], -torch.inf)
            ids = nb[start:end].gather(1, sim.argmax(1, keepdim=True)).flatten()
            pieces.append(rcg[ids])
        return torch.cat(pieces).reshape(128, 128)

    with torch.inference_mode():
        host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), man, "cuda")
        from utils.data import denormalize

        data = Path(man["data_root"])
        for n, row in enumerate(rows):
            key = row["key"]
            fp = a.root / row.get("feature_export", f"cache/evidence_v1/feat/{key}.pt")
            cached = torch.load(fp, map_location="cpu", weights_only=True)
            if any(cached[k].shape != (4096, 1024) or cached[k].dtype != torch.float16 for k in ("q", "r")):
                raise ValueError("Require retained 64-grid FP16 q/r tokens: " + key)
            if any(not bool(torch.isfinite(cached[k]).all()) or bool((cached[k].float().norm(dim=1) == 0).any())
                   for k in ("q", "r")):
                raise ValueError("Non-finite or zero-norm cached token: " + key)
            q, r = (F.normalize(cached[k].to(dev).float(), dim=1) for k in ("q", "r"))
            pp = packet(a.root, row)
            # Intentionally index only supplied reference coverage, never query truth.
            with np.load(pp, allow_pickle=False) as z:
                cov = z["cov"].copy()
            if cov.shape != (64, 64) or not np.isfinite(cov).all() or cov.min() < 0 or cov.max() > 1:
                raise ValueError("Invalid reference coverage: " + key)
            fg = torch.from_numpy(cov.reshape(-1) >= .5).to(dev)
            if not bool(fg.any()) or bool(fg.all()):
                raise ValueError("Reference must contain both roles: " + key)
            source_field = a.run / "fields" / (key + ".npz")
            if sha(source_field) != source_seal["fields"][key]:
                raise ValueError("Changed sealed RCG field: " + key)
            with np.load(source_field, allow_pickle=False) as z:
                rcg = np.asarray(z["rcg"], np.float32).copy()
            if rcg.shape != (64, 64) or not np.isfinite(rcg).all():
                raise ValueError("Invalid existing RCG field: " + key)
            qp, rp = data / row["query"], data / row["support"]
            with Image.open(qp) as image:
                tgt = host._transform(image.convert("RGB")).to(dev)
            with Image.open(rp) as image:
                ref = host._transform(image.convert("RGB")).to(dev)
            if tgt.shape != (3, 1024, 1024) or ref.shape != tgt.shape:
                raise ValueError("Host transform must produce 1024 RGB tensors")
            pad = F.pad(tgt[None], (4, 4, 4, 4), mode="reflect")[0]
            fine = torch.empty(128, 128, q.shape[1], device=dev, dtype=torch.float32)
            for ay, ax in itertools.product((0, 1), repeat=2):
                sy, sx = (-4, 4)[ay], (-4, 4)[ax]
                shifted = pad[:, 4 + sy:4 + sy + 1024, 4 + sx:4 + sx + 1024]
                f = F.normalize(host._extract_features(shifted[None, None]), p=2, dim=2)
                if bool(cached["debiased"]):
                    f = host._debias_features(f)
                if f.shape != (1, 1, 1024, 64, 64):
                    raise ValueError("Unexpected host last-layer feature shape")
                fine[ay::2, ax::2] = F.normalize(f[0, 0].float(), dim=0).permute(1, 2, 0)
            fine_fp16 = fine.half()
            fine = F.normalize(fine_fp16.float().reshape(128 * 128, -1), dim=1)
            coarse_up = F.interpolate(q.T.reshape(1, 1024, 64, 64), (128, 128),
                                      mode="bilinear", align_corners=False)[0].flatten(1).T
            coarse_up = F.normalize(coarse_up, dim=1)
            coarse_margin = margin(q, r, fg).reshape(64, 64)
            rgb_q = F.avg_pool2d(denormalize(tgt).clamp(0, 1)[None], 8)[0].flatten(1).T
            rgb_r = F.avg_pool2d(denormalize(ref).clamp(0, 1)[None], 16)[0].flatten(1).T
            rcg_tensor = torch.from_numpy(rcg).to(dev).flatten()
            fields = dict(
                fine_nn=margin(fine, r, fg).reshape(128, 128),
                **{"bilinear_feature_nn.control": margin(coarse_up, r, fg).reshape(128, 128),
                   "bilinear_margin.control": F.interpolate(coarse_margin[None, None], (128, 128),
                        mode="bilinear", align_corners=False)[0, 0],
                   "rgb_nn.control": margin(rgb_q, rgb_r, fg, colour=True).reshape(128, 128)},
                rcg64=rcg_tensor.reshape(64, 64),
                local_rcg_fine_nn=local_nearest(fine, q, rcg_tensor),
                local_rcg_coarse_nn=local_nearest(coarse_up, q, rcg_tensor),
            )
            fields = {k: v.cpu().numpy().astype(np.float32, copy=False) for k, v in fields.items()}
            if not all(np.isfinite(v).all() for v in fields.values()) or not bool(torch.isfinite(fine_fp16).all()):
                raise ValueError("Non-finite diagnostic features or fields: " + key)
            feature_out = a.out / "features" / (key + ".npz")
            field_out = a.out / "fields" / (key + ".npz")
            np.savez(feature_out, fine=fine_fp16.cpu().numpy())
            np.savez_compressed(field_out, **fields)
            receipts[key] = dict(feature_sha256=sha(fp), packet_sha256=sha(pp),
                                 source_rcg_field_sha256=sha(source_field),
                                 query_image_sha256=sha(qp), reference_image_sha256=sha(rp),
                                 debiased=bool(cached["debiased"]),
                                 reference_fg_tokens=int(fg.sum()), reference_bg_tokens=int((~fg).sum()))
            assets[key] = dict(feature_sha256=sha(feature_out), fields_sha256=sha(field_out),
                               feature_bytes=feature_out.stat().st_size, fields_bytes=field_out.stat().st_size)
            state.update(completed=n + 1, seconds=time.monotonic() - begin)
            write(a.out / "state.json", state)
            print(json.dumps(dict(inferred=n + 1, total=80, seconds=round(state["seconds"], 2))), flush=True)
            del cached, q, r, tgt, ref, pad, f, fine, fine_fp16, coarse_up, rgb_q, rgb_r, fields
    seal = dict(state="ALL_FEATURES_AND_FIELDS_SEALED", n=80, assets=assets, inputs=receipts,
                manifest_sha256=sha(a.out / "manifest.json"), config_sha256=sha(a.out / "config.json"),
                query_truth_opened=False, seconds=time.monotonic() - begin,
                cuda_peak_bytes=torch.cuda.max_memory_allocated(),
                retained_feature_bytes=sum(v["feature_bytes"] for v in assets.values()))
    write(a.out / "sealed.json", seal)
    state.update(state=seal["state"], completed=80, seconds=seal["seconds"])
    write(a.out / "state.json", state)
    print("BOUNDARY_FEATURES80_INFERENCE_SEALED", flush=True)


def auc(values, positive):
    from scipy.stats import rankdata
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return None
    ranks = rankdata(values, method="average")
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def parent_ranking(values, selected, positive):
    reshape = lambda v: v.reshape(64, 2, 64, 2).transpose(0, 2, 1, 3).reshape(4096, 4)
    v, take, fg = reshape(values), reshape(selected), reshape(positive)
    pairs = (take & fg)[:, :, None] & (take & ~fg)[:, None, :]
    diff = v[:, :, None] - v[:, None, :]
    credit = (diff > 0).astype(np.float64) + .5 * (diff == 0)
    counts = pairs.sum((1, 2))
    eligible = counts > 0
    wins = (credit * pairs).sum((1, 2))
    result = None if not eligible.any() else float(np.mean(wins[eligible] / counts[eligible]))
    return result, dict(eligible_parents=int(eligible.sum()),
                        parents_with_pure_boundary_cells=int(take.any(1).sum()),
                        total_parents=4096, fg_bg_pairs=int(counts.sum()))


def bootstrap_summary(values, weights, mask=None):
    v = np.array([np.nan if x is None else x for x in values], dtype=np.float64)
    take = np.isfinite(v)
    if mask is not None:
        take &= mask
    if not take.any():
        return dict(mean=None, ci95=None, episodes=0, bootstrap_valid_draws=0)
    denominators = weights[:, take].sum(1)
    draws = (weights[:, take] @ v[take])[denominators > 0] / denominators[denominators > 0]
    return dict(mean=float(v[take].mean()), ci95=np.percentile(draws, [2.5, 97.5]).tolist(),
                episodes=int(take.sum()), bootstrap_valid_draws=len(draws))


def summarize_metric(rows, details, name, weights):
    values = {arm: [d["metrics"][name][arm] for d in details] for arm in ARMS}
    folds = np.array([r["fold"] for r in rows])
    result = dict(arms={arm: bootstrap_summary(v, weights) for arm, v in values.items()}, folds={}, contrasts={})
    for fold in range(4):
        result["folds"][str(fold)] = {arm: bootstrap_summary(v, weights, folds == fold) for arm, v in values.items()}
    for arm, other in itertools.combinations(ARMS, 2):
        paired = [None if x is None or y is None else x - y for x, y in zip(values[arm], values[other])]
        result["contrasts"][arm + " minus " + other] = bootstrap_summary(paired, weights)
    return result


def score(a):
    from scipy.ndimage import distance_transform_edt

    seal = json.loads((a.out / "sealed.json").read_text())
    config = json.loads((a.out / "config.json").read_text())
    rows = load_rows(a.out / "manifest.json")
    if seal["state"] != "ALL_FEATURES_AND_FIELDS_SEALED" or seal["n"] != 80 or len(rows) != 80:
        raise ValueError("Require all 80 feature grids and fields sealed before scoring")
    if sha(a.out / "manifest.json") != seal["manifest_sha256"] or sha(a.out / "config.json") != seal["config_sha256"]:
        raise ValueError("Changed sealed cohort or configuration")
    if sha(Path(__file__)) != config["source_code_sha256"] or sha(REPO / "src/ics/experiment.py") != config["experiment_sha256"]:
        raise ValueError("Changed diagnostic implementation")
    if sorted({r["fold"] for r in rows}) != [0, 1, 2, 3] or any(sum(r["fold"] == fold for r in rows) != 20 for fold in range(4)):
        raise ValueError("Require fixed 20/fold sampling")
    # Validate EVERY retained asset and packet before opening the first truth.
    for row in rows:
        key = row["key"]
        if sha(a.out / "features" / (key + ".npz")) != seal["assets"][key]["feature_sha256"]:
            raise ValueError("Changed retained fine feature grid: " + key)
        if sha(a.out / "fields" / (key + ".npz")) != seal["assets"][key]["fields_sha256"]:
            raise ValueError("Changed sealed fields: " + key)
        if sha(packet(a.root, row)) != seal["inputs"][key]["packet_sha256"]:
            raise ValueError("Changed evaluation packet: " + key)
    begin = time.monotonic()
    write(a.out / "score_state.json", dict(state="RUNNING_CPU_GT_DIAGNOSTIC", pid=os.getpid(), completed=0, n=80))
    details = []
    nb, valid = neighborhood()
    pool8 = lambda v: v.reshape(128, 8, 128, 8).mean((1, 3))
    for n, row in enumerate(rows):
        key = row["key"]
        with np.load(packet(a.root, row), allow_pickle=False) as z:
            truth = unpack(z["truth"])
        coverage = pool8(truth)
        pure = (coverage <= .1) | (coverage >= .9)
        positive = coverage >= .9
        if truth.any() and not truth.all():
            distance = np.where(truth, distance_transform_edt(truth), distance_transform_edt(~truth))
            boundary = pool8(distance) <= 16
        else:
            boundary = np.zeros((128, 128), dtype=bool)
        selected = pure & boundary
        with np.load(a.out / "fields" / (key + ".npz"), allow_pickle=False) as z:
            fields = {arm: z[arm].copy() for arm in ARMS}
            rcg = z["rcg64"].reshape(-1)
            local = {arm: z[arm].copy() for arm in ("local_rcg_fine_nn", "local_rcg_coarse_nn")}
        if any(v.shape != (128, 128) or not np.isfinite(v).all() for v in fields.values()):
            raise ValueError("Invalid diagnostic fields")
        metrics = dict(boundary_auroc={}, parent_pair_ranking={})
        for arm, field in fields.items():
            metrics["boundary_auroc"][arm] = auc(field[selected], positive[selected])
            metrics["parent_pair_ranking"][arm], parents = parent_ranking(field, selected, positive)
        neighbor_fg = ((rcg[nb] > .5) & valid).any(1).reshape(128, 128)
        neighbor_bg = ((rcg[nb] <= .5) & valid).any(1).reshape(128, 128)
        available = np.where(positive, neighbor_fg, neighbor_bg)
        count = lambda mask: int(np.count_nonzero(mask))
        availability = dict(boundary_pure_cells=count(selected),
                            true_side_available=count(selected & available),
                            true_side_unavailable=count(selected & ~available), by_role={})
        for role, role_mask in (("FG", positive), ("BG", ~positive)):
            take = selected & role_mask
            availability["by_role"][role] = dict(cells=count(take), available=count(take & available),
                                                  unavailable=count(take & ~available))
        availability["nearest_feature_diagnostic"] = {}
        for arm, field in local.items():
            correct = (field > .5) == positive
            if np.any(selected & ~available & correct):
                raise ValueError("A local match cannot be correct when no valid neighbor has its GT side")
            availability["nearest_feature_diagnostic"][arm] = dict(
                correct=count(selected & correct), wrong=count(selected & ~correct),
                available_but_wrong=count(selected & available & ~correct),
                unavailable_and_wrong=count(selected & ~available & ~correct))
        details.append(dict(key=key, fold=row["fold"], c=row["c"], support=row["support"], query=row["query"],
                            boundary_cells=count(boundary), pure_boundary_cells=count(selected),
                            pure_boundary_fg=count(selected & positive), pure_boundary_bg=count(selected & ~positive),
                            metrics=metrics, parent_coverage=parents, rcg_true_side_availability=availability))
        if (n + 1) % 10 == 0:
            write(a.out / "score_state.json", dict(state="RUNNING_CPU_GT_DIAGNOSTIC", pid=os.getpid(), completed=n + 1, n=80))
            print(json.dumps(dict(scored=n + 1, total=80, seconds=round(time.monotonic() - begin, 2))), flush=True)
    groups = photo_groups(rows)
    g = int(groups.max()) + 1
    draws = np.random.RandomState(0).randint(g, size=(2000, g))
    weights = np.stack([np.bincount(d, minlength=g) for d in draws])[:, groups]
    parents = {k: sum(d["parent_coverage"][k] for d in details)
               for k in ("eligible_parents", "parents_with_pure_boundary_cells", "total_parents", "fg_bg_pairs")}
    parents["eligible_fraction_of_all_parents"] = parents["eligible_parents"] / parents["total_parents"]
    parents["eligible_fraction_of_pure_boundary_parents"] = (parents["eligible_parents"] / parents["parents_with_pure_boundary_cells"]
                                                            if parents["parents_with_pure_boundary_cells"] else None)
    avail = {k: sum(d["rcg_true_side_availability"][k] for d in details)
             for k in ("boundary_pure_cells", "true_side_available", "true_side_unavailable")}
    avail["nearest_feature_diagnostic"] = {
        arm: {k: sum(d["rcg_true_side_availability"]["nearest_feature_diagnostic"][arm][k] for d in details)
              for k in ("correct", "wrong", "available_but_wrong", "unavailable_and_wrong")}
        for arm in ("local_rcg_fine_nn", "local_rcg_coarse_nn")}
    report = dict(config=config, n=80, folds={str(f): 20 for f in range(4)},
                  photo_groups=g, largest_photo_group=int(np.bincount(groups).max()),
                  units="AUROC/ranking in [0,1]; contrasts are absolute differences, not mIoU",
                  boundary_auroc=summarize_metric(rows, details, "boundary_auroc", weights),
                  parent_pair_ranking=summarize_metric(rows, details, "parent_pair_ranking", weights),
                  parent_coverage=parents, rcg_true_side_availability=avail,
                  inference_seal_sha256=sha(a.out / "sealed.json"), gt_use="CPU labeled diagnostic only",
                  notes=["Macro statistics weight eligible episodes equally, not pixels or classes.",
                         "Pairwise contrasts use the same eligible episodes and connected-photo bootstrap weights.",
                         "Parent coverage and availability counts are descriptive, pixel-weighted counts.",
                         "Available-but-wrong nearest-feature matches diagnose this fixed affinity readout only.",
                         "A signal or capacity diagnostic does not establish complete-mask improvement."],
                  score_seconds=time.monotonic() - begin)
    write(a.out / "episode_metrics.json", details)
    write(a.out / "report.json", report)
    np.save(a.out / "bootstrap_photo_draws.npy", draws)
    lines = ["# Boundary feature diagnostic: 80 reused episodes, GT scoring only", "",
             "| arm | boundary AUROC (95% CI) | parent pair ranking (95% CI) |", "|---|---:|---:|"]
    def fmt(stat):
        return ("undefined (n=0)" if stat["mean"] is None else
                "%.4f [%.4f, %.4f], n=%d" % (stat["mean"], *stat["ci95"], stat["episodes"]))
    for arm in ARMS:
        lines.append("| %s | %s | %s |" % (arm, fmt(report["boundary_auroc"]["arms"][arm]),
                                               fmt(report["parent_pair_ranking"]["arms"][arm])))
    lines += ["", "Paired differences (fine minus control):", "",
              "| control | boundary AUROC difference (95% CI) | parent ranking difference (95% CI) |", "|---|---:|---:|"]
    for arm in ARMS[1:]:
        contrast = "fine_nn minus " + arm
        lines.append("| %s | %s | %s |" % (arm, fmt(report["boundary_auroc"]["contrasts"][contrast]),
                                               fmt(report["parent_pair_ranking"]["contrasts"][contrast])))
    lines += ["", "Eligible parents: %d/%d total; %d with pure boundary cells. FG/BG pairs: %d." %
              (parents["eligible_parents"], parents["total_parents"], parents["parents_with_pure_boundary_cells"], parents["fg_bg_pairs"]),
              "True-side RCG value available: %d/%d pure boundary cells; unavailable: %d." %
              (avail["true_side_available"], avail["boundary_pure_cells"], avail["true_side_unavailable"]),
              "", "GT defines diagnostic cells and labels only after all 80 feature grids and fields are sealed.",
              "This cohort is reused development evidence; no complete-mask or SOTA result is asserted."]
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    write(a.out / "score_state.json", dict(state="CPU_GT_DIAGNOSTIC_COMPLETE", pid=os.getpid(), completed=80,
                                          n=80, seconds=report["score_seconds"], report_sha256=sha(a.out / "report.json")))
    print("\n".join(lines), flush=True)
    print("BOUNDARY_FEATURES80_SCORE_DONE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("infer", "score"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--host-manifest", type=Path, default=Path("outputs/claude_official/batch0.json"))
    p.add_argument("--host-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent"))
    p.add_argument("--demo4-root", default="/root/autodl-tmp/demo4")
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    if a.chunk < 1 or a.threads < 1:
        p.error("chunk and threads must be positive")
    if a.stage == "infer" and a.run is None:
        p.error("infer requires --run")
    if a.stage == "infer" and a.out.exists():
        p.error("infer never overwrites existing retained assets: " + str(a.out))
    try:
        infer(a) if a.stage == "infer" else score(a)
    except Exception as error:
        if a.out.exists() and (a.out / "state.json").exists():
            write(a.out / ("state.json" if a.stage == "infer" else "score_state.json"),
                  dict(state="FAILED", stage=a.stage, pid=os.getpid(), error=repr(error)))
        raise


if __name__ == "__main__":
    main()
