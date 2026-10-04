#!/usr/bin/env python3
"""Object-level identity of FoRIS's candidate regions: the frozen encoder applied again to each region's crop.

  python scripts/object_identity.py --manifest M/dev_episodes.json --packets P/packets --dino D/dinov3-vitl16-timm \
      --prefix LOCAL_ROOT --out X/object_dev.npz

Candidates are the connected components of FoRIS's final score field (min-max normalised) at six levels. Each
candidate's crop of the query and the reference object's crop are encoded by the same frozen DINOv3; no label is used:
  cls     cosine of the two crops' class tokens
  region  cosine of the mean patch token inside the candidate and inside the reference mask (both from the crop pass)
  grey    cosine of the class tokens when everything outside the region is set to the mean colour
Stored with each candidate for the reading (labels are used only there): level, area, pixels on the target, and
FoRIS's own mean score inside it (the host's evidence, the control).
"""
import argparse
import json
import time
from pathlib import Path

LEVELS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--packets", required=True); p.add_argument("--dino", required=True)
    p.add_argument("--out", required=True); p.add_argument("--prefix", default=""); p.add_argument("--side", type=int, default=224)
    p.add_argument("--margin", type=float, default=0.15); p.add_argument("--limit", type=int); p.add_argument("--device", default="mps")
    p.add_argument("--field", default="score")
    a = p.parse_args()
    import numpy as np
    import timm
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from scipy import ndimage
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(a.prefix + man["data_root"]), Path(a.prefix + man["annotation_root"])
    cfg = json.loads((Path(a.dino) / "config.json").read_text())
    model = timm.create_model(cfg["architecture"], pretrained=True, num_classes=0, dynamic_img_size=True,
                              pretrained_cfg_overlay=dict(file=str(Path(a.dino) / "model.safetensors"))).to(a.device).eval()
    mean, std = (torch.tensor(v, device=a.device).view(1, 3, 1, 1) for v in (MEAN, STD))
    prefix = model.num_prefix_tokens

    def encode(image, mask, box):
        """Class token, region token and grey-background class token of the crop `box` of an image; mask at image size."""
        x0, y0, x1, y1 = box
        crop, m = image.crop(box), mask[y0:y1, x0:x1]
        k = a.side / max(crop.size)
        W, H = max(48, round(crop.size[0] * k / 16) * 16), max(48, round(crop.size[1] * k / 16) * 16)
        x = torch.from_numpy(np.asarray(crop.resize((W, H), Image.BICUBIC))).to(a.device).permute(2, 0, 1)[None].float() / 255
        mm = torch.from_numpy(np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR))).to(a.device).float()[None, None] / 255
        x = (x - mean) / std
        tok = model.forward_features(torch.cat([x, x * mm]))  # the grey image: outside the region the normalised pixel is 0
        w = F.adaptive_avg_pool2d(mm, (H // 16, W // 16)).flatten()
        region = (w[:, None] * tok[0, prefix:]).sum(0) / w.sum().clamp(min=1e-6)
        return [F.normalize(v.float(), dim=0).cpu().numpy() for v in (tok[0, 0], region, tok[1, 0])]

    def box_of(mask, size):
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        mx, my = a.margin * (x1 - x0), a.margin * (y1 - y0)
        return (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(size[0], int(np.ceil(x1 + mx))), min(size[1], int(np.ceil(y1 + my))))

    rows, masks, begin = [], [], time.monotonic()
    with torch.inference_mode():
        for n, r in enumerate(man["episodes"][:a.limit]):
            key = (r["fold"], r["e"], r["c"])
            z = np.load(Path(a.packets) / ("%d_%d_%d.npz" % key))
            ref, query = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            ref_mask = np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1  # the reference label only
            er = encode(ref, ref_mask, box_of(ref_mask, ref.size))
            s = np.asarray(z[a.field], np.float32)
            sn = (s - s.min()) / max(float(s.max() - s.min()), 1e-6)
            truth = np.unpackbits(z["truth"])[:1024 * 1024].reshape(1024, 1024).astype(bool)  # read only into the stored record
            cov = truth.reshape(64, 16, 64, 16).mean((1, 3))
            seen = set()
            for level in LEVELS:
                lab, count = ndimage.label(sn > level, structure=np.ones((3, 3)))
                areas = ndimage.sum(np.ones_like(lab), lab, range(1, count + 1)) if count else []
                for i in sorted(range(count), key=lambda i: -areas[i])[:8]:
                    comp = lab == i + 1
                    sig = np.packbits(comp).tobytes()
                    if areas[i] < 3 or sig in seen:
                        continue
                    seen.add(sig)
                    full = np.asarray(Image.fromarray(comp.astype(np.uint8)).resize(query.size, Image.NEAREST)) > 0
                    eq = encode(query, full, box_of(full, query.size))
                    rows.append([n, level, float(areas[i]), float(cov[comp].sum()), float(sn[comp].mean()), float(sn[comp].max())]
                                + [float(u @ v) for u, v in zip(eq, er)])
                    masks.append(np.packbits(comp))
            if n % 20 == 0:
                print("%d/%d episodes, %d candidates, %.1f s" % (n + 1, len(man["episodes"][:a.limit]), len(rows), time.monotonic() - begin), flush=True)
    keys = np.array([[r["fold"], r["e"], r["c"]] for r in man["episodes"][:a.limit]])
    np.savez_compressed(a.out, keys=keys, rows=np.array(rows, np.float32), masks=np.stack(masks),
                        columns=np.array(["episode", "level", "area", "on_target", "mean_score", "max_score", "cls", "region", "grey"]))
    print(json.dumps(dict(state="COMPLETED", episodes=len(keys), candidates=len(rows), elapsed_s=round(time.monotonic() - begin, 1))))


if __name__ == "__main__":
    main()
