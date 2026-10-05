#!/usr/bin/env python3
"""One matched subtoken control: replace extra encoder features with coarse interpolation.

The existing fine-feature arm's recorded fold-specific sigma/tau is unchanged.
No selection or grid is rerun. Only cached q and sealed RCG fields enter the
control. Six CPU readers/two writers overlap CUDA batches of four. All600
complete masks seal before CPU scoring reads GT; source fine I/U is checked
against its original counts.npz. This is a reused DEV600 ablation, not SOTA.

python scripts/control_subtoken_coarse.py infer --root outputs/fresh600_root \
    --run outputs/recheck_fresh600_v1 --fine outputs/claude_subtoken_fresh600 \
    --out outputs/subtoken_coarse_control600_v1
python scripts/control_subtoken_coarse.py score --root outputs/fresh600_root \
    --out outputs/subtoken_coarse_control600_v1
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
from pathlib import Path
import re
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ics.experiment import load_rows, packet, sha, summarize, unpack

CONTROL = "coarse_guide.control"
FINE = "fine.selected"


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def selected_parameters(report):
    selection = report["selection"]["feat.nested"]
    if set(selection) != {"0", "1", "2", "3"}:
        raise ValueError("Require original four-fold feat.nested selections")
    parameters = {}
    for fold, name in selection.items():
        match = re.fullmatch(r"feat\[s([0-9.]+),t([0-9.]+)\]", name)
        if match is None:
            raise ValueError("Selected arm is not a source fine-feature arm: " + name)
        sigma, tau = map(float, match.groups())
        if sigma not in (.75, 1.25) or tau not in (.03, .07, .15):
            raise ValueError("Selection is outside the original run_subtoken policy")
        parameters[int(fold)] = dict(arm=name, sigma=sigma, tau=tau)
    return parameters


def infer(a):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = load_rows(a.run / "manifest.json")
    source = json.loads((a.run / "sealed.json").read_text())
    original = json.loads((a.fine / "report.json").read_text())
    parameters = selected_parameters(original)
    keys = {r["key"] for r in rows}
    if len(rows) != 600 or original["n"] != 600 or any(sum(r["fold"] == f for r in rows) != 150 for f in range(4)):
        raise ValueError("Require the unchanged complete600 cohort, 150/fold")
    if source.get("state") != "ALL_PREDICTIONS_SEALED" or sha(a.run / "manifest.json") != source["manifest_sha256"]:
        raise ValueError("Require sealed source600 manifest")
    if keys != set(source["fields"]) or keys != set(source["predictions"]):
        raise ValueError("Source seal does not contain all600 fields and masks")
    with np.load(a.fine / "counts.npz", allow_pickle=False) as z:
        for arm in {"native", "rcg"} | {v["arm"] for v in parameters.values()}:
            if z["iu:" + arm].shape != (600, 2):
                raise ValueError("Source counts must preserve all600 ordered rows")
    if a.out.exists():
        raise FileExistsError("Use a new output directory; retain existing assets")
    a.out.mkdir(parents=True)
    (a.out / "predictions").mkdir()
    (a.out / "fields").mkdir()
    write(a.out / "manifest.json", rows)
    config = dict(
        purpose="one matched readout removal of finer encoder information; not a new component or SOTA claim",
        scope="COCO-20i, reused historical DEV600, seed0, 150/fold, 1024 working pixels",
        replacement="unit(bilinear(unit(cached FP16 q cast FP32),64->128,align_corners=False))",
        unchanged="5x5 neighbors, spatial/cosine kernel, recorded per-fold sigma/tau, 128->1024 renderer >0.5",
        selection="original fine arm feat.nested choices read from report; no new fitting, sweep or selection",
        parameters={str(f): v for f, v in parameters.items()}, query_gt_in_inference=False,
        encoder_forwards=0, images_opened=0, tf32=False, feature_deletion=False,
        source_root=str(a.root.resolve()), source_run=str(a.run.resolve()), source_fine=str(a.fine.resolve()),
        source_manifest_sha256=sha(a.run / "manifest.json"), source_seal_sha256=sha(a.run / "sealed.json"),
        source_fine_report_sha256=sha(a.fine / "report.json"), source_fine_counts_sha256=sha(a.fine / "counts.npz"),
        source_code_sha256=sha(Path(__file__)), experiment_sha256=sha(REPO / "src/ics/experiment.py"),
        batch=a.batch, readers=a.readers, writers=a.writers, chunk=a.chunk, threads=a.threads,
        torch_version=str(torch.__version__), device="cuda", device_name=torch.cuda.get_device_name(0))
    write(a.out / "config.json", config)
    begin = time.monotonic()
    write(a.out / "state.json", dict(state="RUNNING_INFERENCE", pid=os.getpid(), completed=0, n=600,
                                    query_truth_opened=False))
    device = torch.device("cuda")
    # Exact geometry of source run_subtoken.py, including distances before clamping.
    fi = torch.arange(128, device=device)
    base = ((fi * 8 + 4 - 8).float() / 16).round().long()
    off = torch.arange(-2, 3, device=device)
    ti = base[:, None] + off[None]
    distance = ((ti * 16 + 8) - (fi * 8 + 4)[:, None]).float() / 16
    ok = (ti >= 0) & (ti < 64)
    ti = ti.clamp(0, 63)
    ny = ti[:, None, :, None].expand(128, 128, 5, 5)
    nx = ti[None, :, None, :].expand(128, 128, 5, 5)
    nb = (ny * 64 + nx).reshape(16384, 25)
    d2 = (distance[:, None, :, None].square() + distance[None, :, None, :].square()).reshape(16384, 25)
    valid = (ok[:, None, :, None] & ok[None, :, None, :]).reshape(16384, 25)

    def load(row):
        key = row["key"]
        fp = a.root / row.get("feature_export", f"cache/evidence_v1/feat/{key}.pt")
        field = a.run / "fields" / (key + ".npz")
        rcg_mask = a.run / "predictions" / (key + ".npz")
        fine_mask = a.fine / "predictions" / (key + ".npz")
        pp = packet(a.root, row)
        receipt = dict(feature_sha256=sha(fp), packet_sha256=sha(pp),
                       source_rcg_field_sha256=sha(field), source_rcg_prediction_sha256=sha(rcg_mask),
                       source_fine_prediction_sha256=sha(fine_mask))
        if receipt["source_rcg_field_sha256"] != source["fields"][key] or receipt["source_rcg_prediction_sha256"] != source["predictions"][key]:
            raise ValueError("Changed sealed RCG field/mask: " + key)
        if "inputs" in source and receipt["packet_sha256"] != source["inputs"][key]["packet_sha256"]:
            raise ValueError("Changed source evaluation packet: " + key)
        cached = torch.load(fp, map_location="cpu", weights_only=True)
        q = cached["q"]
        if q.shape != (4096, 1024) or q.dtype != torch.float16 or not bool(torch.isfinite(q).all()):
            raise ValueError("Require original finite FP16 cached q")
        with np.load(field, allow_pickle=False) as z:
            rcg = z["rcg"].copy()
        if rcg.shape != (64, 64) or not np.isfinite(rcg).all():
            raise ValueError("Invalid sealed RCG field")
        # No packet array, image, reference feature, or query truth is indexed.
        return q, rcg, receipt

    def save(row, field, mask, receipt):
        key = row["key"]
        pred = a.out / "predictions" / (key + ".npz")
        fld = a.out / "fields" / (key + ".npz")
        np.savez_compressed(pred, **{CONTROL: np.packbits(mask)})
        np.savez_compressed(fld, **{CONTROL: field})
        return key, sha(pred), sha(fld), receipt

    gpu_seconds = 0.
    completed = []
    with torch.inference_mode(), ThreadPoolExecutor(a.readers) as readers, ThreadPoolExecutor(a.writers) as writers:
        pending = {}
        depth = max(3 * a.batch, a.readers)
        for i in range(min(depth, len(rows))):
            pending[i] = readers.submit(load, rows[i])
        for start in range(0, len(rows), a.batch):
            ids = list(range(start, min(start + a.batch, len(rows))))
            loaded = [pending.pop(i).result() for i in ids]
            for i in ids:
                j = i + depth
                if j < len(rows):
                    pending[j] = readers.submit(load, rows[j])
            q = F.normalize(torch.stack([d[0] for d in loaded]).pin_memory().to(device, non_blocking=True).float(), dim=2)
            rcg = torch.from_numpy(np.stack([d[1] for d in loaded])).pin_memory().to(device, non_blocking=True).float().flatten(1)
            sigma = torch.tensor([parameters[rows[i]["fold"]]["sigma"] for i in ids], device=device)[:, None, None]
            tau = torch.tensor([parameters[rows[i]["fold"]]["tau"] for i in ids], device=device)[:, None, None]
            torch.cuda.synchronize()
            tick = time.monotonic()
            fine = F.interpolate(q.transpose(1, 2).reshape(len(ids), 1024, 64, 64), (128, 128),
                                 mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
            fine = F.normalize(fine, dim=2)
            pieces = []
            for cell in range(0, 16384, a.chunk):
                stop = min(cell + a.chunk, 16384)
                neighbors = nb[cell:stop]
                cos = torch.einsum("bpc,bpkc->bpk", fine[:, cell:stop], q[:, neighbors])
                w = torch.exp(-d2[None, cell:stop] / (2 * sigma.square())) * torch.exp((cos - 1) / tau) * valid[None, cell:stop]
                values = rcg[:, neighbors]
                pieces.append((w * values).sum(2) / w.sum(2).clamp_min(1e-12))
            field_tensor = torch.cat(pieces, dim=1).reshape(len(ids), 128, 128)
            masks = (F.interpolate(field_tensor[:, None], (1024, 1024),
                        mode="bilinear", align_corners=False)[:, 0] > .5).cpu().numpy()
            fields = field_tensor.cpu().numpy()
            torch.cuda.synchronize()
            gpu_seconds += time.monotonic() - tick
            if not np.isfinite(fields).all():
                raise ValueError("Nonfinite coarse guide field")
            for b, i in enumerate(ids):
                completed.append(writers.submit(save, rows[i], fields[b].copy(), masks[b].copy(), loaded[b][2]))
            if start == 0 or (start // a.batch) % 10 == 0:
                state = dict(state="RUNNING_INFERENCE", pid=os.getpid(), completed=ids[-1] + 1, n=600,
                             seconds=time.monotonic() - begin, compute_seconds=gpu_seconds, query_truth_opened=False)
                write(a.out / "state.json", state)
                print(json.dumps(state), flush=True)
            del loaded, q, fine, rcg, fields, field_tensor, masks, pieces, cos, w, values
        results = [f.result() for f in completed]
    seal = dict(state="ALL_PREDICTIONS_SEALED", n=600, arms=[CONTROL],
                predictions={k: p for k, p, _, _ in results}, fields={k: f for k, _, f, _ in results},
                inputs={k: r for k, _, _, r in results},
                manifest_sha256=sha(a.out / "manifest.json"), config_sha256=sha(a.out / "config.json"),
                seconds=time.monotonic() - begin, compute_seconds=gpu_seconds,
                cuda_peak_bytes=torch.cuda.max_memory_allocated(), query_truth_opened=False)
    write(a.out / "sealed.json", seal)
    write(a.out / "state.json", dict(state=seal["state"], pid=os.getpid(), completed=600, n=600,
                                    seconds=seal["seconds"], compute_seconds=gpu_seconds))
    print("SUBTOKEN_COARSE600_SEALED", flush=True)


def score(a):
    config = json.loads((a.out / "config.json").read_text())
    seal = json.loads((a.out / "sealed.json").read_text())
    rows = load_rows(a.out / "manifest.json")
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 600 or len(rows) != 600:
        raise ValueError("Require all600 predictions sealed before GT scoring")
    for file, key in (("manifest.json", "manifest_sha256"), ("config.json", "config_sha256")):
        if sha(a.out / file) != seal[key]:
            raise ValueError("Changed sealed " + file)
    if sha(Path(__file__)) != config["source_code_sha256"] or sha(REPO / "src/ics/experiment.py") != config["experiment_sha256"]:
        raise ValueError("Changed control implementation")
    run, fine = Path(config["source_run"]), Path(config["source_fine"])
    for path, key in ((run / "manifest.json", "source_manifest_sha256"), (run / "sealed.json", "source_seal_sha256"),
                      (fine / "report.json", "source_fine_report_sha256"), (fine / "counts.npz", "source_fine_counts_sha256")):
        if sha(path) != config[key]:
            raise ValueError("Changed source fine/RCG evidence")
    # Verify every control/source mask and packet before the first truth is opened.
    for row in rows:
        key = row["key"]
        checks = [(a.out / "predictions" / (key + ".npz"), seal["predictions"][key]),
                  (a.out / "fields" / (key + ".npz"), seal["fields"][key]),
                  (fine / "predictions" / (key + ".npz"), seal["inputs"][key]["source_fine_prediction_sha256"]),
                  (run / "predictions" / (key + ".npz"), seal["inputs"][key]["source_rcg_prediction_sha256"]),
                  (packet(a.root, row), seal["inputs"][key]["packet_sha256"])]
        if any(sha(path) != digest for path, digest in checks):
            raise ValueError("Changed control/source output or GT packet: " + key)
    with np.load(fine / "counts.npz", allow_pickle=False) as z:
        original_counts = {k: z[k].copy() for k in z.files if k.startswith("iu:")}
    arrays = {k: [] for k in ("native", "rcg", FINE, CONTROL)}
    corrections = {k: [] for k in arrays}
    details = []
    begin = time.monotonic()
    write(a.out / "score_state.json", dict(state="RUNNING_CPU_SCORE", pid=os.getpid(), completed=0, n=600))
    for n, row in enumerate(rows):
        key = row["key"]
        selected = config["parameters"][str(row["fold"])]["arm"]
        with np.load(packet(a.root, row), allow_pickle=False) as z:
            truth, native = unpack(z["truth"]), unpack(z["native"])
        with np.load(run / "predictions" / (key + ".npz"), allow_pickle=False) as z:
            rcg = unpack(z["RCG"])
        with np.load(fine / "predictions" / (key + ".npz"), allow_pickle=False) as z:
            fine_mask = unpack(z[selected])
        with np.load(a.out / "predictions" / (key + ".npz"), allow_pickle=False) as z:
            control = unpack(z[CONTROL])
        for arm, mask in {"native": native, "rcg": rcg, FINE: fine_mask, CONTROL: control}.items():
            iu = [int((mask & truth).sum()), int((mask | truth).sum())]
            if arm != CONTROL:
                original_arm = selected if arm == FINE else arm
                if not np.array_equal(iu, original_counts["iu:" + original_arm][n]):
                    raise ValueError("Original fine/native/RCG ordered I/U mismatch: " + key + " " + arm)
            add, delete = mask & ~native, native & ~mask
            rec = dict(key=key, c=row["c"], fold=row["fold"], batch=str(row.get("batch", "unspecified")),
                       add_TP=int((add & truth).sum()), add_FP=int((add & ~truth).sum()),
                       delete_TP=int((delete & truth).sum()), delete_FP=int((delete & ~truth).sum()))
            arrays[arm].append(iu)
            corrections[arm].append(rec)
            details.append(dict(rec, arm=arm, intersection=iu[0], union=iu[1]))
    arrays = {k: np.array(v, dtype=np.int64) for k, v in arrays.items()}
    report, draws = summarize(rows, arrays, corrections)
    report.update(config=config, inference_seal_sha256=sha(a.out / "sealed.json"),
                  source_counts_verified="all600 native/RCG/selected fine I/U exactly match original ordered counts",
                  runtime=dict(inference_seconds=seal["seconds"], gpu_compute_seconds=seal["compute_seconds"],
                               cuda_peak_bytes=seal["cuda_peak_bytes"], cpu_score_seconds=time.monotonic() - begin),
                  interpretation="matched fixed readout ablation; reused DEV600, not independent confirmation or SOTA")
    write(a.out / "episode_metrics.json", details)
    write(a.out / "report.json", report)
    np.savez_compressed(a.out / "counts.npz", **{"iu:" + k: v for k, v in arrays.items()})
    np.save(a.out / "bootstrap_photo_draws.npy", draws)
    contrast = report["contrasts"][FINE][CONTROL]
    lines = ["# Matched subtoken coarse-feature control: reused DEV600", "",
             "| arm | class mIoU at 1024 |", "|---|---:|"]
    lines += ["| %s | %.6f |" % (arm, report["scores"][arm]) for arm in arrays]
    lines += ["", "Fine minus coarse-guide control: %+.6f [%+.6f, %+.6f] percentage points." %
              (contrast["gain"], *contrast["ci95"]),
              "All600 source native, RCG and selected fine I/U exactly match the original counts.",
              "Recorded per-fold parameters are unchanged; no encoder, image, refit or parameter grid.",
              "Inference/compute/CPU-score seconds: %.2f / %.2f / %.2f." %
              (seal["seconds"], seal["compute_seconds"], report["runtime"]["cpu_score_seconds"])]
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    write(a.out / "score_state.json", dict(state="CPU_SCORE_COMPLETE", pid=os.getpid(), completed=600, n=600,
                                          report_sha256=sha(a.out / "report.json")))
    print("\n".join(lines), flush=True)
    print("SUBTOKEN_COARSE600_SCORE_DONE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("infer", "score"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run", type=Path, default=Path("outputs/recheck_fresh600_v1"))
    p.add_argument("--fine", type=Path, default=Path("outputs/claude_subtoken_fresh600"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--readers", type=int, default=6)
    p.add_argument("--writers", type=int, default=2)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    if any(v < 1 for v in (a.batch, a.readers, a.writers, a.chunk, a.threads)):
        p.error("execution sizes must be positive")
    if a.stage == "infer" and a.out.exists():
        p.error("infer requires a fresh output directory")
    try:
        infer(a) if a.stage == "infer" else score(a)
    except Exception as error:
        if (a.out / "config.json").exists():
            write(a.out / ("state.json" if a.stage == "infer" else "score_state.json"),
                  dict(state="FAILED", stage=a.stage, pid=os.getpid(), error=repr(error)))
        raise


if __name__ == "__main__":
    main()
