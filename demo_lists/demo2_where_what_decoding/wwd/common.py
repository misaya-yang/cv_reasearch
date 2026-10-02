"""Shared pieces of the where/what experiments: data, segmenters, feature sources, region pooling, recogniser, scoring.

Conventions
- A *source* turns an image into a list of feature maps `maps` [(d, h, w), ...] and a global vector `glob` (d_g,).
  The feature of a region is the concatenation of the mask-weighted mean of every map and `glob`.
- A *region set* is a float mask stack (K, H, W). Class regions = one mask per class of an integer label map.
- Labels use 255 for ignore.
"""
import os, sys, glob, json, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT = os.environ.get("DEMO2_ROOT", "/root/autodl-tmp/demo2_pilot")
CACHE = os.environ.get("DEMO2_CACHE", "/root/demo2_cache")
sys.path.append(os.environ.get("DEMO2_ENV", ROOT + "/env"))
import transformers

DEV = "cuda:0"
if torch.cuda.is_available():       # the GPU is shared with other projects: cap this process so cached blocks are reused, not hoarded
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("DEMO2_GPU_FRAC", 0.2)))
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ----------------------------------------------------------------------------------------------- data
class ADE20K:
    name, C = "ade20k", 150
    splits = dict(train="training", val="validation")

    def __init__(self, root=None): self.root = root or ROOT + "/data/ADEChallengeData2016"
    def files(self, split): return sorted(glob.glob(f"{self.root}/images/{self.splits[split]}/*.jpg"))

    def load_gt(self, f):
        lab = torch.from_numpy(np.asarray(Image.open(f.replace("/images/", "/annotations/").replace(".jpg", ".png"))).astype(np.int64))
        return torch.where(lab == 0, torch.full_like(lab, 255), lab - 1)


class COCOPanoptic:
    """COCO panoptic collapsed to 133 semantic classes (tools/prep_coco_panoptic.py); train = a fixed 10k subset."""
    name, C = "coco", 133

    def __init__(self, root=None): self.root = root or CACHE + "/coco"
    def files(self, split): return sorted(glob.glob(f"{self.root}/images/{split}/*.jpg"))
    def load_gt(self, f): return torch.from_numpy(np.asarray(Image.open(f.replace("/images/", "/semantic/").replace(".jpg", ".png"))).astype(np.int64))


DATASETS = dict(ade20k=ADE20K, coco=COCOPanoptic)


# ----------------------------------------------------------------------------------------- segmenters
class Segmenter:
    """Frozen semantic segmenter. `prep` runs on CPU workers, `__call__` on the GPU.
    Preprocessing follows the mmseg test pipeline: keep-ratio resize to (4*short, short), sides rounded to 32, whole-image inference."""
    FAMILIES = dict(segformer="SegformerForSemanticSegmentation", upernet="UperNetForSemanticSegmentation",
                    mask2former="Mask2FormerForUniversalSegmentation", eomt="EomtForUniversalSegmentation",
                    eomt_dinov3="EomtDinov3ForUniversalSegmentation")
    MASK_CLS = ("mask2former", "eomt", "eomt_dinov3")          # mask-classification families: outputs are (query class, query mask) pairs

    def __init__(self, family, mdir, short, dev=DEV):
        self.family, self.short, self.dev, self.tag = family, int(short), dev, os.path.basename(mdir.rstrip("/"))
        self.eomt = family.startswith("eomt")
        self.model = getattr(transformers, self.FAMILIES[family]).from_pretrained(mdir).to(dev).eval()
        self._cap = {}
        if family in ("segformer", "upernet"):      # input of the final 1x1 classifier = the feature the per-pixel decision is made from
            self.model.decode_head.classifier.register_forward_pre_hook(lambda mod, inp: self._cap.__setitem__("f", inp[0]))

    def prep(self, im, short=None):
        w, h = im.size; short = short or self.short
        if self.eomt:                 # its own protocol: shorter side to `short`, then square windows along the longer side
            nw, nh = (short, int(short * h / w)) if w < h else (int(short * w / h), short)
            x = torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BILINEAR)).copy()).permute(2, 0, 1).float() / 255
            return (x - IMNET_MEAN) / IMNET_STD
        s = min(4 * short / max(h, w), short / min(h, w))
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        x = torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BILINEAR)).copy()).permute(2, 0, 1).float() / 255
        return (x - IMNET_MEAN) / IMNET_STD

    @torch.inference_mode()
    def __call__(self, x):
        """x: (3, h, w) on device. Returns dict with the raw outputs and the segmenter's own feature maps."""
        if self.eomt: return self._eomt(x)
        x = x[None]
        if self.family == "mask2former":
            o = self.model(pixel_values=x, output_hidden_states=True)
            return dict(qcls=torch.softmax(o.class_queries_logits[0].float(), -1), qmask=o.masks_queries_logits[0].float(),
                        maps=[o.encoder_hidden_states[-1][0].float(), o.pixel_decoder_last_hidden_state[0].float()])
        if self.family == "segformer":
            o = self.model(pixel_values=x, output_hidden_states=True)
            return dict(logits=o.logits[0].float(), maps=[o.hidden_states[-1][0].float(), self._cap["f"][0].float()])
        fm = self.model.backbone.forward_with_filtered_kwargs(x).feature_maps
        logits = self.model.decode_head(fm)[0].float()
        return dict(logits=logits, maps=[fm[-1][0].float(), self._cap["f"][0].float()])

    def _eomt(self, x):
        S = self.short; _, H, W = x.shape; vertical = H > W; L = max(H, W); n = -(-L // S); ov = (n * S - L) / (n - 1) if n > 1 else 0
        starts = [int(k * (S - ov)) for k in range(n)]
        o = self.model(pixel_values=torch.stack([x[:, st:st + S] if vertical else x[:, :, st:st + S] for st in starts]))
        Q = self.model.config.num_queries; g = S // self.model.config.patch_size; mw = o.masks_queries_logits.float(); r = S // mw.shape[-1]
        tok = o.last_hidden_state[:, -g * g:].float().transpose(1, 2).reshape(n, -1, g, g)
        # canvases over the whole (resized) image: query masks at mask resolution (used only to pool features), patch tokens at 1/patch
        cm = mw.new_full((n * Q, -(-H // r), -(-W // r)), -1e4); ct = tok.new_zeros(tok.shape[1], -(-H // (S // g)), -(-W // (S // g))); cc = torch.zeros_like(ct[:1])
        for k, st in enumerate(starts):
            a, b = round(st / r), round(st / (S // g))
            if vertical: cm[k * Q:(k + 1) * Q, a:a + mw.shape[-1]] = mw[k][:, :cm.shape[1] - a]; ct[:, b:b + g] += tok[k][:, :ct.shape[1] - b]; cc[:, b:b + g] += 1
            else: cm[k * Q:(k + 1) * Q, :, a:a + mw.shape[-1]] = mw[k][:, :, :cm.shape[2] - a]; ct[:, :, b:b + g] += tok[k][:, :, :ct.shape[2] - b]; cc[:, :, b:b + g] += 1
        return dict(qcls=torch.softmax(o.class_queries_logits.float(), -1).flatten(0, 1), qmask=cm, qmask_win=mw, starts=starts, vertical=vertical,
                    hw=(H, W), maps=[ct / cc.clamp(min=1)])

    def posterior(self, out, size, qcls=None, mpow=1.0):
        """(C, H, W) per-pixel class posterior at `size`. For mask classification, `qcls` (Q, C) overrides the query class weights."""
        if self.eomt:                 # per-window scores, averaged where windows overlap, then resized (the official order)
            S = self.short; H, W = out["hw"]; n = len(out["starts"]); C = out["qcls"].shape[1] - 1
            qc = (out["qcls"][:, :-1] if qcls is None else qcls).view(n, -1, C)
            sc = torch.einsum("nqc,nqhw->nchw", qc, torch.sigmoid(F.interpolate(out["qmask_win"], size=(S, S), mode="bilinear")) ** mpow)
            agg = sc.new_zeros(C, H, W); cnt = sc.new_zeros(1, H, W)
            for k, st in enumerate(out["starts"]):
                sl = (slice(None), slice(st, st + S), slice(None)) if out["vertical"] else (slice(None), slice(None), slice(st, st + S))
                agg[sl] += sc[k]; cnt[sl] += 1
            s = F.interpolate((agg / cnt.clamp(min=1))[None], size=size, mode="bilinear", align_corners=False)[0].clamp(min=1e-12)
            return s / s.sum(0, keepdim=True)
        if self.family == "mask2former":
            m = torch.sigmoid(F.interpolate(out["qmask"][None], size=size, mode="bilinear", align_corners=False)[0]) ** mpow
            s = torch.einsum("qc,qhw->chw", out["qcls"][:, :-1] if qcls is None else qcls, m).clamp(min=1e-12)
            return s / s.sum(0, keepdim=True)
        return torch.softmax(F.interpolate(out["logits"][None], size=size, mode="bilinear", align_corners=False)[0], 0)


def own_source(out):
    """The segmenter's own features as a source: its last backbone stage and its decoder feature, global = their image means."""
    maps = out["maps"]
    return maps, torch.cat([m.mean((1, 2)) for m in maps])


class Encoder:
    """Frozen external ViT encoder used as a source: patch tokens of the last layer, global = CLS token (or the pooled output for SigLIP)."""
    def __init__(self, mdir, res=518, dev=DEV, keep_ratio=False):
        model = transformers.AutoModel.from_pretrained(mdir); self.siglip = model.config.model_type.startswith("siglip")
        if self.siglip: model = model.vision_model
        self.model = model.to(dev).eval().half()
        self.res, self.patch, self.dev, self.keep_ratio = int(res), self.model.config.patch_size, dev, keep_ratio
        self.mean, self.std = (torch.full((3, 1, 1), 0.5),) * 2 if self.siglip else (IMNET_MEAN, IMNET_STD)

    def prep(self, im):
        nw = nh = self.res
        if self.keep_ratio:                       # shorter side to `res`, longer side by aspect ratio (at most 2x), both rounded to the patch size
            w, h = im.size; s = self.res / min(w, h); P = self.patch
            nw, nh = (max(P, round(min(v * s, 2 * self.res) / P) * P) for v in (w, h))
        x = torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255
        return (x - self.mean) / self.std

    @torch.inference_mode()
    def __call__(self, x):
        o = self.model(pixel_values=x[None].half()); h = o.last_hidden_state[0].float(); gh, gw = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        return [h[-gh * gw:].T.reshape(-1, gh, gw)], (o.pooler_output[0].float() if self.siglip else h[0])


def make_segmenter(spec):
    """spec = family:dir:short (dir relative to $DEMO2_ROOT unless absolute)."""
    fam, mdir, short = spec.split(":"); return Segmenter(fam, mdir if mdir.startswith("/") else f"{ROOT}/{mdir}", short)


def make_encoder(spec):
    """spec = dir:tag[:res[:ratio]]; `ratio` keeps the aspect ratio. Returns (encoder, tag)."""
    f = spec.split(":"); mdir = f[0] if f[0].startswith("/") else f"{ROOT}/{f[0]}"
    return Encoder(mdir, res=int(f[2]) if len(f) > 2 else 518, keep_ratio=len(f) > 3 and f[3] == "ratio"), f[1]


# -------------------------------------------------------------------------------------------- regions
def class_masks(labelmap, ignore=255):
    """One float mask per class present in an integer label map. Returns ids (K,), masks (K, H, W)."""
    ids = labelmap.unique(); ids = ids[ids != ignore]
    return ids, (labelmap[None] == ids.view(-1, 1, 1)).float()


def joint_hist(a, b, valid, C):
    """Joint histogram (C, C) of two label maps over the valid pixels; rows = a, columns = b. No host synchronisation:
    the GPU is shared, and every synchronisation point costs a scheduling round-trip."""
    return torch.bincount((a.clamp(max=C - 1) * C + b.clamp(max=C - 1)).flatten(), weights=valid.flatten().double(), minlength=C * C).view(C, C)


def pool(maps, masks):
    """Mask-weighted mean of every map. maps: list of (d, h, w); masks: (K, H, W) float. Returns feats (K, sum d), ok (K,) bool."""
    out, ok = [], None
    for m in maps:
        w = F.adaptive_avg_pool2d(masks[None], m.shape[-2:])[0].flatten(1)
        a = w.sum(1, keepdim=True); ok = (a[:, 0] > 0) if ok is None else ok & (a[:, 0] > 0)
        out.append((w @ m.flatten(1).T) / a.clamp(min=1e-8))
    return torch.cat(out, 1), ok


def region_features(source, masks):
    maps, glob_ = source; f, ok = pool(maps, masks)
    return torch.cat([f, glob_[None].expand(len(f), -1)], 1), ok


def multi_region_features(sources, names, masks):
    """Region features from several sources concatenated; `names` = "a+b"."""
    fs, oks = zip(*(region_features(sources[n], masks) for n in names.split("+")))
    return torch.cat(fs, 1), torch.stack(oks).all(0)


# ----------------------------------------------------------------------------------------- recogniser
def make_mlp(d, C, hidden=2048, drop=0.3):
    return torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(drop), torch.nn.Linear(hidden, C))


class Recogniser:
    """Region classifier: standardise, MLP, softmax. `prior` is the label prior it was trained on."""
    def __init__(self, path, dev=DEV):
        ck = torch.load(path, map_location=dev); C = ck["state"][sorted(k for k in ck["state"] if k.endswith("bias"))[-1]].shape[0]
        self.net = make_mlp(ck["dim"], C).to(dev); self.net.load_state_dict(ck["state"]); self.net.eval()
        self.mu, self.sd = ck["mu"].to(dev), ck["sd"].to(dev)
        self.prior = ck["prior"].to(dev) if "prior" in ck else None; self.C = C
        dims = (ck.get("args") or {}).get("dims"); self.cols = slice(*(int(v) for v in dims.split(":"))) if dims else slice(None)

    @torch.inference_mode()
    def __call__(self, feats, ok=None):
        r = torch.softmax(self.net((feats[:, self.cols] - self.mu) / self.sd), 1)
        return r if ok is None else torch.where(ok[:, None], r, torch.full_like(r, 1.0 / self.C))


class PatchHead(torch.nn.Module):
    """Per-patch classifier on ViT tokens (the pixel-level control). `cls` appends the CLS token to every patch token."""
    def __init__(self, d, C, kind="linear", cls=False):
        super().__init__(); din = d * (2 if cls else 1); self.cls = cls
        self.net = torch.nn.Sequential(torch.nn.BatchNorm1d(din), torch.nn.Linear(din, C)) if kind == "linear" else \
            torch.nn.Sequential(torch.nn.BatchNorm1d(din), torch.nn.Linear(din, 2048), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(2048, C))

    def forward(self, h, g):
        """h: (B, tokens, d) with CLS first and the g*g patch tokens last. Returns logits (B, C, g, g)."""
        t = h[:, -g * g:]
        if self.cls: t = torch.cat([t, h[:, :1].expand(-1, g * g, -1)], 2)
        return self.net(t.reshape(len(t) * g * g, -1)).view(len(t), g, g, -1).permute(0, 3, 1, 2)

    @staticmethod
    def load(path, dev=DEV):
        ck = torch.load(path, map_location=dev); m = PatchHead(ck["d"], ck["C"], ck["kind"], ck["cls"]).to(dev); m.load_state_dict(ck["state"]); return m.eval()

    @torch.inference_mode()
    def posterior(self, source, size):
        """Per-pixel class posterior (C, H, W) from an Encoder source."""
        (tok,), cls = source; g = tok.shape[-1]; h = torch.cat([cls[None], tok.flatten(1).T])[None]
        return torch.softmax(F.interpolate(self(h, g), size=size, mode="bilinear", align_corners=False)[0], 0)


# -------------------------------------------------------------------------------------------- scoring
class Scores:
    """Confusion matrices per named variant. With `keep` (a regex), also per-image (TP, predicted, ground-truth) pixel counts per class
    for the matching variants, so that differences between variants can be bootstrapped over images."""
    def __init__(self, C, dev=DEV, keep=None):
        import re
        self.C, self.dev, self.S, self.per = C, dev, {}, {}; self.keep = re.compile(keep) if keep else None

    def add(self, name, pred, gt):
        h = joint_hist(gt, pred, gt != 255, self.C)
        self.S.setdefault(name, torch.zeros(self.C * self.C, device=self.dev, dtype=torch.float64)).add_(h.flatten())
        if self.keep is not None and self.keep.match(name):
            self.per.setdefault(name, []).append(torch.stack([h.diag(), h.sum(0), h.sum(1)]).to(torch.int32))

    @staticmethod
    def metrics(conf, C):
        c = conf.view(C, C); tp, g, p = c.diag(), c.sum(1), c.sum(0)
        return dict(mIoU=round(float(torch.nanmean(tp / (g + p - tp))) * 100, 3), mAcc=round(float(torch.nanmean(tp / g)) * 100, 3),
                    aAcc=round(float(tp.sum() / c.sum()) * 100, 3))

    def result(self): return {k: self.metrics(v, self.C) for k, v in self.S.items()}
    def class_iou(self, name):
        c = self.S[name].view(self.C, self.C); tp, g, p = c.diag(), c.sum(1), c.sum(0)
        return (tp / (g + p - tp)).tolist()

    def save_per_image(self, path):
        """Sparse per-image statistics: for each kept variant, (image, class) indices and the three counts."""
        out = {}
        for name, rows in self.per.items():
            t = torch.stack(rows).cpu(); nz = (t.sum(1) > 0).nonzero()          # (N, 3, C) -> rows (image, class) with any count
            out[name] = dict(idx=nz.to(torch.int16 if len(rows) < 32768 else torch.int32), val=t[nz[:, 0], :, nz[:, 1]])
        torch.save(dict(C=self.C, images=len(next(iter(self.per.values()))) if self.per else 0, variants=out), path)


class ImageSet(torch.utils.data.Dataset):
    """Loads an image, its label and the CPU-side preprocessing of every model, in worker processes."""
    def __init__(self, ds, files, preps): self.ds, self.fs, self.preps = ds, files, preps
    def __len__(self): return len(self.fs)

    def __getitem__(self, i):
        torch.set_num_threads(1); f = self.fs[i]; im = Image.open(f).convert("RGB")
        return [p(im) for p in self.preps], self.ds.load_gt(f).to(torch.uint8), i


def loader(ds, files, preps, workers=12):
    return torch.utils.data.DataLoader(ImageSet(ds, files, preps), batch_size=None, num_workers=workers, shuffle=False, prefetch_factor=4)


def save_json(obj, path):
    if os.path.exists(path): raise SystemExit(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True); json.dump(obj, open(path, "w"), indent=1)
