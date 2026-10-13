"""Replay six selected W02 failures unchanged and inspect their saved stages.

This is a replay of an already evaluated frozen method, not a new candidate.
All six predictions are saved before query GT is opened for replay checks and
explicit stage/oracle diagnostics. Writes only to the external case directory.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/x1012"))
import r5_candidate as rc
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = ROOT.parent / "cv_data/a"
OUT = DATA / "x1012s_case_review/w02_replay"


def metrics(pred, gt):
    i, u = int((pred & gt).sum()), int((pred | gt).sum())
    return dict(I=i, U=u, predicted=int(pred.sum()), iou=i/max(u,1),
                tp=i, fp=int((pred & ~gt).sum()), fn=int((~pred & gt).sum()))


def strict_oracle(field, gt):
    x, y = field.ravel(), gt.ravel()
    ix = np.argsort(-x, kind="stable")
    s = x[ix]
    tp = np.cumsum(y[ix], dtype=np.int64)
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    j = ends[np.argmax(tp[ends] / np.maximum(ends+1+gt.sum()-tp[ends],1))]
    threshold = float(s[j])
    m = metrics(field >= threshold, gt)
    m["threshold"] = threshold
    return m


def tint(rgb, mask, color):
    x = rgb.copy().astype(float)
    x[mask] = .55*x[mask] + .45*np.asarray(color)
    return x.astype(np.uint8)


def rgb(row, role):
    path = rc.ASSETS / row[role+"_path"]
    with Image.open(path) as im:
        im = im.convert("RGB")
        if row.get(role+"_crop") is not None:
            im = im.crop(row[role+"_crop"])
        assert rc.array_hash(np.array(im)) == row[role+"_rgb_hash"]
        return np.array(im.resize((512,512)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    probe = rc.read(DATA / "x1012s_case_review/w02_apd_branch_probe.json")
    source = {}
    for part in ("explore", "confirm"):
        for b in rc.read(DATA/f"x1012_P0/{part}.json"):
            source[str(b["index"])] = b
    cases = probe["cases"]
    rc.prepare_runtime(DATA / "x1013/w02")  # Existing lock: verify, never replace.
    rc.initialize(DATA / "x1013/w02", "cpu", 2, "midrange", False)
    started = time.monotonic()
    for case in cases:
        name = case["index"]
        archive, record = OUT/(name+".npz"), OUT/(name+".json")
        if record.exists():
            assert rc.read(record)["archive_sha256"] == rc.sha(archive)
            continue
        assert not archive.exists(), "Preserve incomplete prior output: "+name
        pred, fields, detail = rc.infer(source[name])
        np.savez_compressed(archive, **fields, **{"mask/UN_identity":pred["UN_identity"]})
        rc.write(record, dict(index=name, archive_sha256=rc.sha(archive), detail=detail))
        print("replayed", name, round(detail["seconds"],3), flush=True)
    seal = dict(state="SIX_EXISTING_METHOD_REPLAYS_SAVED_BEFORE_DIAGNOSTIC_GT",
                script_sha256=rc.sha(Path(__file__)), candidate_source_sha256=rc.sha(Path(rc.__file__)),
                runtime_lock_sha256=rc.sha(DATA/"x1013/w02/runtime/runtime.json"),
                records=[rc.read(OUT/(c["index"]+".json")) for c in cases],
                seconds=time.monotonic()-started, new_encoder_calls=0, new_candidate=False)
    rc.write(OUT/"sealed.json", seal)

    results = []
    plots = {}
    stages = ("c32", "c_mod", "primary", "VP_UN")
    for case in cases:
        name, domain = case["index"], case["dataset"]
        b = source[name]
        gt = rc.lab.mask1024(b["row"]["query_mask_path"]).numpy()
        assert not b["row"].get("query_ignore_mask_path"), "Add exact ignore handling if needed"
        with np.load(OUT/(name+".npz")) as a:
            final = rc.pc.unpack(a["mask/UN_identity"])
            final_stats = metrics(final, gt)
            expected = case["saved_W02_counts"]["un50"]
            assert all(final_stats[k] == expected[k] for k in ("I","U","predicted")), (name, final_stats, expected)
            f = rc.lab.foris_cli(b).numpy()
            fm = metrics(f, gt)
            assert all(fm[k] == case["saved_W02_counts"]["foris"][k] for k in ("I","U","predicted"))
            stats, masks, fields = {}, {}, {}
            for stage in stages:
                native = np.asarray(a[stage], dtype=np.float32).reshape(64,64)
                field = rc.lab.up(torch.from_numpy(native)).numpy()
                mask = rc.natural_readout(native, "midrange")
                stats[stage] = dict(natural=metrics(mask,gt), strict_oracle=strict_oracle(field,gt),
                                    native_min=float(native.min()), native_max=float(native.max()))
                masks[stage], fields[stage] = mask, rc.lab.minmax(torch.from_numpy(field)).numpy()
            assert np.array_equal(masks["VP_UN"], final)
            results.append(dict(index=name, episode_id=b["row"]["episode_id"], dataset=domain,
                                replay_counts_exact=True, foris=fm, stages=stats,
                                selected_modulation=rc.read(OUT/(name+".json"))["detail"].get("modulation"),
                                reference_rho=float(a["reference_rho"][0])))
        R, Q = rgb(b["row"],"reference"), rgb(b["row"],"query")
        rm = rc.lab.mask1024(b["row"]["reference_mask_path"]).numpy()[::2,::2]
        panels=[(tint(R,rm,(0,255,0)), "Reference mask"),
                (tint(Q,gt[::2,::2],(255,0,0)), "Query GT"),
                (tint(Q,f[::2,::2],(0,70,255)), f"FoRIS {fm['iou']:.3f}")]
        for stage in stages:
            panels.append((tint(Q,masks[stage][::2,::2],(160,0,230)),
                           f"{stage} {stats[stage]['natural']['iou']:.3f}"))
        plots.setdefault(domain, []).append((name, panels, fields))
    for domain, items in plots.items():
        fig, axes = plt.subplots(len(items), 7, figsize=(21,3*len(items)), squeeze=False)
        for row, (name, panels, _) in zip(axes, items):
            for col,(im,title) in enumerate(panels):
                row[col].imshow(im)
                row[col].set_title(title+("\n"+name if col==0 else ""),fontsize=8)
                row[col].axis("off")
        fig.tight_layout();fig.savefig(OUT/(domain+"_stages.png"),dpi=100);plt.close(fig)
        fig, axes = plt.subplots(len(items), 4, figsize=(12,3*len(items)), squeeze=False)
        for row, (name, _, fields) in zip(axes, items):
            for col,stage in enumerate(stages):
                row[col].imshow(fields[stage][::2,::2],vmin=0,vmax=1,cmap="viridis")
                row[col].set_title(stage+("\n"+name if col==0 else ""),fontsize=8)
                row[col].axis("off")
        fig.tight_layout();fig.savefig(OUT/(domain+"_fields.png"),dpi=100);plt.close(fig)
    rc.write(OUT/"report.json", dict(scope="Six GT-selected explanatory cases; not a new candidate or benchmark",
                                    replayed_existing_method=True, all_six_W02_counts_exact=True, cases=results))
    for r in results:
        print(r["index"], {s: {"natural":round(v["natural"]["iou"],4), "oracle":round(v["strict_oracle"]["iou"],4)}
                            for s,v in r["stages"].items()}, flush=True)


if __name__ == "__main__":
    main()
