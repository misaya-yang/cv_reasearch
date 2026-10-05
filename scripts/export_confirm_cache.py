#!/usr/bin/env python3
"""Export, for a cohort whose complete FoRIS run already exists, the inputs the cached methods read (GPU).

Runs the public FoRIS once per episode through the same host and taps as the DEV241 cache
(demo9_extent/scripts/evidence_cache.py and extent_experiment.py), checks that the stored final mask is reproduced,
and writes ROOT/cache/evidence_v1/feat/<key>.pt (q, r: the features FoRIS matched in, float16) and
ROOT/results/extent_v1/run/packets/<key>.npz (stored score, s2, s3, cov, truth, native, pre + fg_max, bg_max).
The stored FoRIS masks remain the comparator. No query truth is read here beyond copying the stored packed array.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--stored", type=Path, required=True, help="packets of the existing FoRIS run")
    p.add_argument("--out", type=Path, required=True); p.add_argument("--host-root", type=Path, default=Path("/root/autodl-tmp/demo9_extent"))
    p.add_argument("--demo4-root", default="/root/autodl-tmp/demo4"); p.add_argument("--limit", type=int)
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, str(a.host_root)); sys.path.insert(0, str(a.host_root / "scripts"))
    from extent_experiment import build_host, run_foris
    man = json.loads(a.manifest.read_text()); rows = man["episodes"][:a.limit]
    feat_dir, packet_dir = a.out / "cache/evidence_v1/feat", a.out / "results/extent_v1/run/packets"
    feat_dir.mkdir(parents=True, exist_ok=False); packet_dir.mkdir(parents=True, exist_ok=False)
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    report = dict(state="RUNNING", episodes=0, bit_identical=0, debiased=0, max_native_pixels_differing=0, max_score_abs_difference=0., max_cov_abs_difference=0.)
    begin = time.monotonic()
    with torch.inference_mode():
        host = build_host(SimpleNamespace(fixture=None, foris_root=None, demo4_root=a.demo4_root), man, "cuda")
        for n, row in enumerate(rows, 1):
            name = "%d_%d_%d" % (row["fold"], row["e"], row["c"])
            sp, qp = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == row["c"] + 1).copy())
            native, got, ref_mask, _ = run_foris(host, sp, gold, qp)
            with np.load(a.stored / (name + ".npz"), allow_pickle=False) as z:
                stored = {k: z[k].copy() for k in z.files}
            differ = int(np.unpackbits(np.packbits(native.cpu().numpy()) ^ stored["native"]).sum())
            if differ > 0.001 * native.numel():
                raise RuntimeError("stored mask not reproduced on %s: %d pixels differ" % (name, differ))
            raw, deb = F.normalize(got["raw"][0], dim=1), got["deb"][0]            # [2, C, h, w]: reference, query
            q, r = deb[-1].flatten(1).T, deb[0].flatten(1).T
            score = got["score"].float(); cov = F.interpolate(ref_mask[None, None].float(), tuple(score.shape), mode="area")[0, 0]
            sim = q @ r.T; fg = torch.from_numpy(stored["cov"]).to(sim.device).flatten() >= 0.5
            zero = torch.zeros(sim.shape[0], device=sim.device)
            fg_max = sim[:, fg].amax(1) if fg.any() else zero; bg_max = sim[:, ~fg].amax(1) if (~fg).any() else zero
            debiased = bool((raw - deb).abs().max() > 1e-4)
            torch.save(dict(q=q.half().cpu(), r=r.half().cpu(), debiased=debiased), feat_dir / (name + ".pt"))
            np.savez_compressed(packet_dir / (name + ".npz"), **stored, fg_max=fg_max.float().cpu().numpy(), bg_max=bg_max.float().cpu().numpy())
            report["episodes"] = n; report["bit_identical"] += int(differ == 0); report["debiased"] += int(debiased)
            report["max_native_pixels_differing"] = max(report["max_native_pixels_differing"], differ)
            report["max_score_abs_difference"] = max(report["max_score_abs_difference"], float(np.abs(score.cpu().numpy() - stored["score"]).max()))
            report["max_cov_abs_difference"] = max(report["max_cov_abs_difference"], float(np.abs(cov.cpu().numpy() - stored["cov"]).max()))
            if n % 25 == 0:
                print(json.dumps(dict(n=n, total=len(rows), seconds=round(time.monotonic() - begin, 1), identical=report["bit_identical"])), flush=True)
    report.update(state="COMPLETED", elapsed_s=time.monotonic() - begin)
    (a.out / "report.json").write_text(json.dumps(report, indent=1) + "\n"); print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
