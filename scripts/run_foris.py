#!/usr/bin/env python3
"""Complete public FoRIS on a manifest, keeping every stage for later readings (GPU).

  python scripts/run_foris.py --manifest M/confirm_episodes.json --out X/foris_confirm

Per episode: the stage fields s2, s3 and the final score (64 x 64), the binarised and the final mask and the truth
at model size (packed bits), the reference mask's patch coverage, and I/U at model size and at original resolution.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True); p.add_argument("--limit", type=int)
    p.add_argument("--foris-root", help="Existing FoRIS checkout; defaults to the manifest path")
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from ics.foris import build_host, run_foris
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    out = Path(a.out); (out / "packets").mkdir(parents=True, exist_ok=True)
    host = build_host(man, "cuda", a.foris_root)
    begin = time.monotonic()
    with torch.inference_mode(), open(out / "episodes.jsonl", "w") as stream:
        for n, r in enumerate(man["episodes"][:a.limit]):
            sp, qp = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            gold = torch.from_numpy((np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1).copy())
            native, got, ref_mask, _ = run_foris(host, sp, gold, qp)
            hw = tuple(native.shape)
            truth_o = torch.from_numpy((np.asarray(Image.open(ann / Path(r["query"]).with_suffix(".png"))) == r["c"] + 1).copy()).to(native.device)
            truth = F.interpolate(truth_o[None, None].float(), hw, mode="nearest")[0, 0].bool()
            native_o = F.interpolate(native[None, None].float(), tuple(truth_o.shape), mode="bilinear", align_corners=False)[0, 0] > .5
            s = got["score"].float()
            np.savez_compressed(out / "packets" / ("%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])),
                                score=s.cpu().numpy().astype(np.float32), s2=got["s2"].float().cpu().numpy().astype(np.float32),
                                s3=got["s3"].float().cpu().numpy().astype(np.float32),
                                cov=F.interpolate(ref_mask[None, None].float(), tuple(s.shape), mode="area")[0, 0].cpu().numpy(),
                                native=np.packbits(native.cpu().numpy()), pre=np.packbits(got["pre"].cpu().numpy()), truth=np.packbits(truth.cpu().numpy()))
            stream.write(json.dumps(dict(fold=r["fold"], e=r["e"], c=r["c"], support=r["support"], query=r["query"],
                                         iu=[int((native & truth).sum()), int((native | truth).sum())],
                                         original_iu=[int((native_o & truth_o).sum()), int((native_o | truth_o).sum())])) + "\n"); stream.flush()
            if n % 50 == 0:
                print("%d/%d, %.1f s" % (n + 1, len(man["episodes"][:a.limit]), time.monotonic() - begin), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=n + 1, elapsed_s=round(time.monotonic() - begin, 1))))


if __name__ == "__main__":
    main()
