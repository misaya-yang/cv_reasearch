"""Task-level falsification of resolution-dependent prefix/patch attention mass.

Reuses an existing frozen EoMT checkpoint and ADE20K. No downloads or training.
All data here are development data, previously examined in demo2.
The log-size correction is related to ToMe proportional attention; not claimed novel.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DEMO2_GPU_FRAC", "0.28")
sys.path.insert(0, "/root/autodl-tmp/demo2_where_what_decoding")
from wwd.common import ADE20K, Segmenter, IMNET_MEAN, IMNET_STD
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def stats(pred, gt, classes=150):
    valid = gt != 255
    p, y = pred[valid].long(), gt[valid].long()
    joint = torch.bincount(y * classes + p, minlength=classes * classes).reshape(classes, classes)
    return torch.stack([joint.diag(), joint.sum(0), joint.sum(1)], 0).cpu().numpy()


def miou(st):
    tp, pred, true = np.asarray(st).sum(0)
    union = pred + true - tp
    present = union > 0
    return float(100 * (tp[present] / union[present]).mean())


def paired_ci(rows, a, b, seed=2037, reps=2000):
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(reps):
        ids = rng.integers(0, len(rows), len(rows))
        deltas.append(miou([rows[i][a] for i in ids]) - miou([rows[i][b] for i in ids]))
    return [float(v) for v in np.quantile(deltas, [.025, .975])]


class ResolutionProbe:
    def __init__(self, model_dir):
        self.seg = Segmenter("eomt", model_dir, 512)
        self.model = self.seg.model
        assert self.model.config.image_size == 512 and self.model.config.patch_size == 16
        self.native_grid = 32
        self.native_patches = 1024
        self.res = 512
        self.mode = "native"
        self.telemetry = []
        self.record = False
        self.orig_embed = self.model.embeddings.forward
        original_embeddings = self.orig_embed
        probe = self

        def dynamic_embeddings(emb, pixel_values):
            if pixel_values.shape[-2:] == (512, 512):
                return original_embeddings(pixel_values)
            assert pixel_values.shape[-2] == pixel_values.shape[-1]
            assert pixel_values.shape[-1] % 16 == 0
            patches = emb.patch_embeddings(pixel_values)
            g = pixel_values.shape[-1] // 16
            pe = emb.position_embeddings.weight.T.reshape(1, -1, 32, 32)
            pe = F.interpolate(pe.float(), size=(g, g), mode="bicubic", align_corners=False)
            patches = patches + pe.flatten(2).transpose(1, 2).to(patches)
            return emb.dropout(torch.cat([
                emb.cls_token.expand(len(pixel_values), -1, -1),
                emb.register_tokens.expand(len(pixel_values), -1, -1), patches], 1))

        self.model.embeddings.forward = types.MethodType(dynamic_embeddings, self.model.embeddings)
        self.original_attention = []
        for idx, layer in enumerate(self.model.layers):
            att = layer.attention
            self.original_attention.append(att.forward)
            original = att.forward

            def corrected(att, hidden_states, attention_mask=None, _idx=idx, _original=original, **kwargs):
                B, N, D = hidden_states.shape
                patches = (probe.res // 16) ** 2
                prefix = N - patches
                nquery = prefix - 5
                assert nquery in (0, probe.model.config.num_queries)
                bias = math.log(patches / probe.native_patches)
                mode = probe.mode
                mask = attention_mask
                if mode in ("all_prefix", "queries", "backbone") and bias != 0:
                    added = hidden_states.new_zeros(1, 1, 1, N)
                    if mode == "all_prefix":
                        added[..., :prefix] = bias
                    elif mode == "queries":
                        added[..., :nquery] = bias
                    else:
                        added[..., nquery:prefix] = bias
                    mask = added if mask is None else mask + added
                old_scale = att.scale
                if mode == "temperature":
                    att.scale = old_scale * math.log(patches + prefix) / math.log(probe.native_patches + prefix)
                if probe.record and _idx in (0, 11, 19, 20, 23):
                    with torch.no_grad():
                        h = hidden_states
                        q = att.q_proj(h).view(B, N, att.num_heads, att.head_dim).transpose(1, 2)
                        k = att.k_proj(h).view(B, N, att.num_heads, att.head_dim).transpose(1, 2)
                        ids = torch.linspace(prefix, N - 1, 32, device=h.device).long()
                        score = (q[:, :, ids] @ k.transpose(-2, -1)).float() * att.scale
                        if mask is not None:
                            sampled = mask if mask.shape[-2] == 1 else mask[:, :, ids]
                            score = score + sampled
                        a = score.softmax(-1)
                        probe.telemetry.append(dict(layer=_idx, mode=mode, resolution=probe.res,
                            patch_to_query=float(a[..., :nquery].sum(-1).mean()) if nquery else 0,
                            patch_to_backbone_prefix=float(a[..., nquery:prefix].sum(-1).mean()),
                            patch_to_patch=float(a[..., prefix:].sum(-1).mean())))
                try:
                    return _original(hidden_states, mask, **kwargs)
                finally:
                    att.scale = old_scale

            att.forward = types.MethodType(corrected, att)

    def set_mode(self, res, mode):
        self.res, self.mode = res, mode
        self.seg.short = res
        self.model.grid_size = (res // 16, res // 16)

    @torch.inference_mode()
    def posterior(self, image, size, res, mode):
        self.set_mode(res, mode)
        x = self.seg.prep(image).cuda()
        out = self.seg(x)
        return self.seg.posterior(out, size)

    @torch.inference_mode()
    def native_tiles_on_highres(self, image, size):
        self.set_mode(512, "native")
        # The same high-resolution RGB input as the 1024 arm; native model windows.
        x = self.seg.prep(image, short=1024).cuda()
        _, H, W = x.shape
        hs = np.linspace(0, H - 512, math.ceil(H / 512)).round().astype(int)
        ws = np.linspace(0, W - 512, math.ceil(W / 512)).round().astype(int)
        acc = torch.zeros(150, H, W, device="cuda")
        cnt = torch.zeros(1, H, W, device="cuda")
        for h in hs:
            for w in ws:
                o = self.seg(x[:, h:h + 512, w:w + 512])
                p = self.seg.posterior(o, (512, 512))
                acc[:, h:h + 512, w:w + 512] += p
                cnt[:, h:h + 512, w:w + 512] += 1
        assert float(cnt.min()) > 0
        return F.interpolate((acc / cnt)[None], size=size, mode="bilinear", align_corners=False)[0], len(hs) * len(ws)

    @torch.inference_mode()
    def check_native(self, image):
        self.set_mode(512, "native")
        x = self.seg.prep(image).cuda()
        native = self.seg(x)
        # Compare against completely original methods, including original embedding.
        emb = self.model.embeddings
        dynamic = emb.forward
        changed = [l.attention.forward for l in self.model.layers]
        emb.forward = self.orig_embed
        for l, orig in zip(self.model.layers, self.original_attention):
            l.attention.forward = orig
        try:
            reference = self.seg(x)
        finally:
            emb.forward = dynamic
            for l, f in zip(self.model.layers, changed):
                l.attention.forward = f
        differences = {k: float((native[k] - reference[k]).abs().max()) for k in ("qcls", "qmask")}
        assert max(differences.values()) == 0, differences
        return differences


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(state="STARTING", pid=os.getpid(), boundary="quality development probe; no speed claims",
                  downloads=0, training_updates=0, dataset="ADE20K validation already used in prior research",
                  hypothesis="fixed query/register mass diluted as spatial sampling density increases",
                  correction="fixed log(N/N512) on selected prefix keys; no GT or tuning at inference",
                  prior="ToMe proportional attention/TransNeXt length scaling are direct controls, not new inventions")
    write_json(out / "status.json", status)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dataset = ADE20K()
        files = dataset.files("val")
        assert len(files) == 2000
        rng = np.random.default_rng(2037)
        files = [files[i] for i in rng.permutation(len(files))[:args.limit]]
        write_json(out / "manifest.json", dict(seed=2037, split="development", images=files))
        model = ResolutionProbe("/root/demo2_cache/models/eomt-large-ade")
        im0 = Image.open(files[0]).convert("RGB")
        status["native_wrapper_maxdiff"] = model.check_native(im0)
        status["state"] = "RUNNING"
        write_json(out / "status.json", status)
        print(json.dumps(status), flush=True)
        rows, per_image, times = [], [], []
        start = time.monotonic()
        for n, path in enumerate(files):
            image = Image.open(path).convert("RGB")
            gt = dataset.load_gt(path).cuda()
            size = tuple(gt.shape)
            image_stats, elapsed = {}, {}
            # Keep just the low-res posterior for a standard multi-scale control.
            model.record = n < 8
            torch.cuda.synchronize()
            t = time.monotonic()
            low = model.posterior(image, size, 512, "native")
            torch.cuda.synchronize()
            elapsed["native512"] = time.monotonic() - t
            image_stats["native512"] = stats(low.argmax(0), gt)
            for res, mode in [(768, "native"), (1024, "native"), (1024, "all_prefix"),
                              (1024, "queries"), (1024, "backbone"), (1024, "temperature")]:
                name = f"{mode}{res}"
                torch.cuda.synchronize()
                t = time.monotonic()
                p = model.posterior(image, size, res, mode)
                torch.cuda.synchronize()
                elapsed[name] = time.monotonic() - t
                image_stats[name] = stats(p.argmax(0), gt)
                if res == 1024 and mode == "native":
                    image_stats["multiscale512_1024"] = stats((low + p).argmax(0), gt)
                del p
            model.record = False
            torch.cuda.synchronize()
            t = time.monotonic()
            tile, nw = model.native_tiles_on_highres(image, size)
            torch.cuda.synchronize()
            elapsed["native512_tiles_on1024"] = time.monotonic() - t
            image_stats["native512_tiles_on1024"] = stats(tile.argmax(0), gt)
            del tile, low, gt
            rows.append(image_stats)
            times.append(elapsed)
            per_image.append(dict(image=os.path.basename(path), window_count=nw,
                                  stats={k: v.tolist() for k, v in image_stats.items()}, elapsed_shared_gpu_s=elapsed))
            status.update(images_completed=n + 1, elapsed_s=time.monotonic() - start,
                          running_miou={k: miou([r[k] for r in rows]) for k in image_stats})
            write_json(out / "status.json", status)
            write_json(out / "per_image.json", per_image)
            write_json(out / "attention_mass.json", model.telemetry)
            print(json.dumps({k: status[k] for k in ("images_completed", "elapsed_s", "running_miou")}), flush=True)
        summary = dict(miou=status["running_miou"], images=len(rows), split="development",
                       shared_gpu_timing_only=True, task="semantic segmentation against original-resolution labels",
                       head_fixed=True, position_interpolation="bicubic align_corners=False for 768/1024",
                       comparisons={})
        for a, b in [("all_prefix1024", "native1024"), ("queries1024", "native1024"),
                     ("backbone1024", "native1024"), ("temperature1024", "native1024"),
                     ("all_prefix1024", "native512"), ("all_prefix1024", "native512_tiles_on1024"),
                     ("all_prefix1024", "multiscale512_1024"), ("native1024", "native512")]:
            summary["comparisons"][a + "-" + b] = dict(delta_pp=summary["miou"][a] - summary["miou"][b],
                                                    paired_image_bootstrap_ci95=paired_ci(rows, a, b))
        write_json(out / "report.json", summary)
        status.update(state="COMPLETED", allocated_peak_gb=torch.cuda.max_memory_allocated() / 2**30,
                      reserved_peak_gb=torch.cuda.max_memory_reserved() / 2**30)
        write_json(out / "status.json", status)
        print(json.dumps(summary), flush=True)
    except Exception:
        status.update(state="ERROR", error=traceback.format_exc())
        write_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=128)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
