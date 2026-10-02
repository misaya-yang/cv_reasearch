"""Reverse-direction falsification: whole-model cost/quality, no training.

Reuses the same development images and saved native512 task statistics.
Includes PI-resize from FlexiViT as an existing method control, not our invention.
Does not rerun any completed high-resolution arm.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import types

from probe import ResolutionProbe, stats, miou, paired_ci, write_json, ADE20K
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def pi_resize(weight, target=32):
    # Basis construction of the same linear bicubic/antialias resize operation.
    old = weight.shape[-1]
    basis = torch.eye(old * old, dtype=torch.float64).reshape(-1, 1, old, old)
    matrix = F.interpolate(basis, size=(target, target), mode="bicubic", align_corners=False,
                           antialias=True).reshape(old * old, -1).T
    inverse = torch.linalg.pinv(matrix.T)
    residual = float((matrix.T @ inverse - torch.eye(old * old, dtype=torch.float64)).abs().max())
    assert residual < 1e-10, residual
    resized = weight.detach().cpu().double().flatten(2) @ inverse.T
    return resized.reshape(*weight.shape[:2], target, target).to(weight), residual


class LowResolution(ResolutionProbe):
    def __init__(self):
        super().__init__("/root/demo2_cache/models/eomt-large-ade")
        self.pi = False
        self.dynamic_embed = self.model.embeddings.forward
        self.old_projection = self.model.embeddings.patch_embeddings.projection
        new_weight, self.pi_residual = pi_resize(self.old_projection.weight)
        self.pi_projection = torch.nn.Conv2d(3, 1024, 32, stride=32, device="cuda")
        with torch.no_grad():
            self.pi_projection.weight.copy_(new_weight)
            self.pi_projection.bias.copy_(self.old_projection.bias)
        self.pi_projection.eval()

        def pi_embeddings(emb, pixel_values):
            p = self.pi_projection(pixel_values).flatten(2).transpose(1, 2)
            pe = emb.position_embeddings.weight.T.reshape(1, -1, 32, 32)
            pe = F.interpolate(pe, size=(16, 16), mode="bicubic", align_corners=False)
            p = p + pe.flatten(2).transpose(1, 2)
            return emb.dropout(torch.cat([emb.cls_token.expand(len(pixel_values), -1, -1),
                                         emb.register_tokens.expand(len(pixel_values), -1, -1), p], 1))
        self.pi_embed = types.MethodType(pi_embeddings, self.model.embeddings)

    def set_mode(self, res, mode):
        super().set_mode(res, mode)
        # All attention logic counts tokens from equivalent patch16 resolution.
        # PI control keeps RGB at 512, changes kernel/stride to32 (16x16 tokens).
        self.model.config.patch_size = 32 if self.pi else 16
        self.model.embeddings.forward = self.pi_embed if self.pi else self.dynamic_embed
        if self.pi:
            assert res == 256
            self.seg.short = 512


def run(args):
    prior = Path(args.prior)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(state="STARTING", pid=os.getpid(), training_updates=0, downloads=0,
        hypothesis="reverse density correction at reduced encoder token counts",
        split="same development cohort as prior; not independent test", images_completed=0,
        baseline="native512 statistics reused; native low-res/temperature/PI-resize controls")
    write_json(out / "status.json", status)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        manifest = json.loads((prior / "manifest.json").read_text())
        old_rows = json.loads((prior / "per_image.json").read_text())
        assert len(old_rows) == len(manifest["images"]) == 128
        dataset, model = ADE20K(), LowResolution()
        status.update(state="RUNNING", pi_basis_identity_residual=model.pi_residual)
        write_json(out / "manifest.json", manifest)
        write_json(out / "status.json", status)
        rows, details = [], []
        start = time.monotonic()
        for n, path in enumerate(manifest["images"]):
            image = Image.open(path).convert("RGB")
            gt = dataset.load_gt(path).cuda()
            assert old_rows[n]["image"] == os.path.basename(path)
            row = {"native512": np.asarray(old_rows[n]["stats"]["native512"])}
            elapsed = {}
            model.record = n < 8
            for res, mode, use_pi in [(r, m, False) for r in (384, 256)
                                      for m in ("native", "all_prefix", "backbone", "temperature")] + \
                                     [(256, "native", True), (256, "all_prefix", True)]:
                model.pi = use_pi
                name = f"{mode}{res}" if not use_pi else f"pi32_rgb512_{mode}"
                torch.cuda.synchronize()
                t = time.monotonic()
                p = model.posterior(image, tuple(gt.shape), res, mode)
                torch.cuda.synchronize()
                elapsed[name] = time.monotonic() - t
                row[name] = stats(p.argmax(0), gt)
                del p
            del gt
            rows.append(row)
            details.append(dict(image=os.path.basename(path), stats={k:v.tolist() for k,v in row.items()},
                                elapsed_shared_gpu_s=elapsed))
            status.update(images_completed=n+1, elapsed_s=time.monotonic()-start,
                          running_miou={k:miou([r[k] for r in rows]) for k in row})
            write_json(out / "status.json", status)
            write_json(out / "per_image.json", details)
            write_json(out / "attention_mass.json", model.telemetry)
            print(json.dumps({k:status[k] for k in ("images_completed","elapsed_s","running_miou")}), flush=True)
        summary = dict(miou=status["running_miou"], images=len(rows), split="development",
                       shared_gpu_timing_only=True, comparisons={})
        for res in (384,256):
            for variant in ("all_prefix", "backbone", "temperature"):
                a,b=f"{variant}{res}", f"native{res}"
                summary["comparisons"][a+"-"+b] = dict(delta_pp=summary["miou"][a]-summary["miou"][b],
                    paired_image_bootstrap_ci95=paired_ci(rows,a,b))
        for a,b in [("all_prefix256","pi32_rgb512_native"),("all_prefix384","native512"),
                    ("all_prefix256","native512"),("pi32_rgb512_all_prefix","pi32_rgb512_native")]:
            summary["comparisons"][a+"-"+b] = dict(delta_pp=summary["miou"][a]-summary["miou"][b],
                paired_image_bootstrap_ci95=paired_ci(rows,a,b))
        write_json(out / "report.json", summary)
        status.update(state="COMPLETED", allocated_peak_gb=torch.cuda.max_memory_allocated()/2**30,
                      reserved_peak_gb=torch.cuda.max_memory_reserved()/2**30)
        write_json(out / "status.json", status)
        print(json.dumps(summary), flush=True)
    except Exception:
        status.update(state="ERROR", error=traceback.format_exc())
        write_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--prior",required=True)
    p.add_argument("--out",required=True)
    run(p.parse_args())
