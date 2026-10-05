#!/usr/bin/env python3
"""The language channel on the frozen DINOv3 backbone: code shared by lang_bank / lang_gate / lang_pipe.

One reference fixes one point of a concept, not how far the concept reaches. FoRIS sets that reach by a constant, and
every re-reading of the same pair's similarity stopped near +1.5 (README ledger). Inside the DINOv3 column the one
source of a category prior that needs no mask, no training and no extra image is the backbone's own text head
(dino.txt): after it, patch tokens and words share a space. Here that becomes one evidence value per query patch,
added to FoRIS's score before FoRIS's own cut and CRF.
"""
import math
import sys

import torch
import torch.nn.functional as F

TEMPLATES = (  # the 80 templates of the official dino.txt segmentation notebook
    "a bad photo of a {0}.", "a photo of many {0}.", "a sculpture of a {0}.", "a photo of the hard to see {0}.",
    "a low resolution photo of the {0}.", "a rendering of a {0}.", "graffiti of a {0}.", "a bad photo of the {0}.",
    "a cropped photo of the {0}.", "a tattoo of a {0}.", "the embroidered {0}.", "a photo of a hard to see {0}.",
    "a bright photo of a {0}.", "a photo of a clean {0}.", "a photo of a dirty {0}.", "a dark photo of the {0}.",
    "a drawing of a {0}.", "a photo of my {0}.", "the plastic {0}.", "a photo of the cool {0}.",
    "a close-up photo of a {0}.", "a black and white photo of the {0}.", "a painting of the {0}.",
    "a painting of a {0}.", "a pixelated photo of the {0}.", "a sculpture of the {0}.", "a bright photo of the {0}.",
    "a cropped photo of a {0}.", "a plastic {0}.", "a photo of the dirty {0}.", "a jpeg corrupted photo of a {0}.",
    "a blurry photo of the {0}.", "a photo of the {0}.", "a good photo of the {0}.", "a rendering of the {0}.",
    "a {0} in a video game.", "a photo of one {0}.", "a doodle of a {0}.", "a close-up photo of the {0}.",
    "a photo of a {0}.", "the origami {0}.", "the {0} in a video game.", "a sketch of a {0}.", "a doodle of the {0}.",
    "a origami {0}.", "a low resolution photo of a {0}.", "the toy {0}.", "a rendition of the {0}.",
    "a photo of the clean {0}.", "a photo of a large {0}.", "a rendition of a {0}.", "a photo of a nice {0}.",
    "a photo of a weird {0}.", "a blurry photo of a {0}.", "a cartoon {0}.", "art of a {0}.", "a sketch of the {0}.",
    "a embroidered {0}.", "a pixelated photo of a {0}.", "itap of the {0}.", "a jpeg corrupted photo of the {0}.",
    "a good photo of a {0}.", "a plushie {0}.", "a photo of the nice {0}.", "a photo of the small {0}.",
    "a photo of the weird {0}.", "the cartoon {0}.", "art of the {0}.", "a drawing of the {0}.",
    "a photo of the large {0}.", "a black and white photo of a {0}.", "the plushie {0}.", "a dark photo of a {0}.",
    "itap of a {0}.", "graffiti of the {0}.", "a toy {0}.", "itap of my {0}.", "a photo of a cool {0}.",
    "a photo of a small {0}.", "a tattoo of the {0}.")
# background words: without them every background patch has to take the name of some object
STUFF = ("sky,clouds,wall,floor,ceiling,ground,grass,road,pavement,sand,snow,water,sea,river,mountain,hill,rock,tree,bush,"
         "leaves,building,house,roof,fence,window,door,curtain,carpet,dirt,mud,gravel,field,railing,stairs,bridge,"
         "platform,wood,metal,plastic,cloth,paper,fog,shadow,background").split(",")
COCO = ("person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,"
        "bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,"
        "snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,"
        "fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,"
        "potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,"
        "refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush").split(",")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
EPS = 1e-3
# W: the whole image at FoRIS's size, head tokens; S: the notebook's sliding windows, head tokens; B: the same windows,
# the backbone's own patch tokens (what the notebook's code takes; kept so that a wrong guess of the aligned space cannot fail the gate)
MODES = ("W", "S", "B")
NAMINGS = ("space", "pool_top1", "pool_bma", "pool_cos", "crop_top1", "crop_bma", "crop_cos", "like_top1")
ORACLES = ("oracle", "oracle_cos")  # the true class name: the ceiling of the channel, never a method arm
LAMBDAS = (0.25, 0.5, 1.0, 2.0)


# ---------------------------------------------------------------- model (server only: needs the dinov3 source, Python >= 3.10)
def load_text_model(repo, weights, backbone, bpe, device, tiny=False, text=True):
    """dino.txt as released: the frozen ViT-L/16, two head blocks, the text tower. `tiny` builds the same path with
    random small parts, for plumbing on a CPU; its numbers mean nothing."""
    sys.path.insert(0, repo)
    from dinov3.eval.text.dinotxt_model import DINOTxt, DINOTxtConfig
    from dinov3.eval.text.text_transformer import TextTransformer
    from dinov3.eval.text.tokenizer import get_tokenizer
    from dinov3.hub import backbones
    cfg = DINOTxtConfig(
        embed_dim=768 if tiny else 2048, vision_model_freeze_backbone=True, vision_model_train_img_size=224,
        vision_model_use_class_token=True, vision_model_use_patch_tokens=True, vision_model_num_head_blocks=2,
        vision_model_head_blocks_drop_path=0.3, vision_model_use_linear_projection=False,
        vision_model_patch_tokens_pooler_type="mean", vision_model_patch_token_layer=1, text_model_freeze_backbone=False,
        text_model_num_head_blocks=0, text_model_head_blocks_is_causal=False, text_model_head_blocks_drop_prob=0.0,
        text_model_tokens_pooler_type="argmax", text_model_use_linear_projection=True,
        init_logit_scale=math.log(1 / 0.07), init_logit_bias=None, freeze_logit_scale=False)
    vision = (backbones.dinov3_vits16 if tiny else backbones.dinov3_vitl16)(pretrained=False)
    text = TextTransformer(context_length=77, vocab_size=49408, dim=64 if tiny else 1280, num_heads=4 if tiny else 20,
                           num_layers=2 if tiny else 24, ffn_ratio=4, is_causal=True, ls_init_value=None, dropout_prob=0.0)
    model = DINOTxt(model_config=cfg, vision_backbone=vision, text_backbone=text)
    if tiny:
        torch.manual_seed(0)
        model.visual_model.init_weights(); model.text_model.init_weights()
        model.logit_scale.data.fill_(cfg.init_logit_scale)
    else:
        print("backbone", vision.load_state_dict(torch.load(backbone, map_location="cpu", weights_only=True), strict=True))
        got = model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=False)
        stray = [k for k in got.missing_keys if not k.startswith("visual_model.backbone.")]
        if stray or got.unexpected_keys:
            raise SystemExit("text head checkpoint does not fit: missing %s, unexpected %s" % (stray[:5], got.unexpected_keys[:5]))
    if not text:  # the pipeline reads the saved text features
        model.text_model = None
    return model.to(device).eval().requires_grad_(False), get_tokenizer(bpe_path_or_url=bpe).tokenize


@torch.inference_mode()
def text_matrix(model, tokenize, names, device, templates=TEMPLATES, batch=256):
    """Per name: the patch-aligned half and the full feature, each averaged over templates as the notebook does."""
    prompts = [t.format(n) for n in names for t in templates]
    with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
        f = torch.cat([model.encode_text(tokenize(prompts[i:i + batch]).to(device)).float() for i in range(0, len(prompts), batch)])
    f = f.view(len(names), len(templates), -1)
    d = f.shape[-1] // 2
    return (F.normalize(F.normalize(f[..., d:], dim=-1).mean(1), dim=-1).cpu(),
            F.normalize(F.normalize(f, dim=-1).mean(1), dim=-1).cpu())


@torch.inference_mode()
def head_tokens(model, x, backbone=False):
    """x [B, 3, H, W], normalised -> class token [B, D] and patch tokens [B, H/16, W/16, D] after the text head
    (with `backbone`, also the backbone's own patch tokens on the same grid)."""
    cls, tok, back = model.visual_model.get_class_and_patch_tokens(x)
    grid = lambda t: t.float().reshape(x.shape[0], x.shape[2] // 16, x.shape[3] // 16, -1)
    return (cls.float(), grid(tok), grid(back)) if backbone else (cls.float(), grid(tok))


def image_tensor(img, wh, device):
    import numpy as np
    from PIL import Image
    x = torch.from_numpy(np.array(img.resize(wh, Image.BICUBIC))).to(device).permute(2, 0, 1).float() / 255
    return (x - torch.tensor(MEAN, device=device).view(3, 1, 1)) / torch.tensor(STD, device=device).view(3, 1, 1)


def slide_tokens(model, img, device, short=512, side=384, stride=192):
    """The notebook's protocol (short side 512, windows of 384 every 192), with every size a multiple of 16 so that
    the windows share one patch lattice: the mean of the normalised tokens of all windows covering a patch.
    Returns the head's grid, the backbone's grid and the size the image was shown at."""
    w, h = img.size
    k = short / min(w, h)
    W, H = max(side, round(w * k / 16) * 16), max(side, round(h * k / 16) * 16)
    x = image_tensor(img, (W, H), device)
    ys = sorted(set(list(range(0, H - side + 1, stride)) + [H - side]))
    xs = sorted(set(list(range(0, W - side + 1, stride)) + [W - side]))
    wins, acc, cnt, g = [(a, b) for a in ys for b in xs], None, torch.zeros(H // 16, W // 16, 1, device=device), side // 16
    for i in range(0, len(wins), 8):
        part = wins[i:i + 8]
        _, tok, back = head_tokens(model, torch.stack([x[:, a:a + side, b:b + side] for a, b in part]), backbone=True)
        both = torch.stack([F.normalize(tok, dim=-1), F.normalize(back, dim=-1)], 1)
        if acc is None:
            acc = torch.zeros(2, H // 16, W // 16, tok.shape[-1], device=device)
        for (a, b), t in zip(part, both):
            acc[:, a // 16:a // 16 + g, b // 16:b // 16 + g] += t
            cnt[a // 16:a // 16 + g, b // 16:b // 16 + g] += 1
    acc = F.normalize(acc / cnt, dim=-1)
    return acc[0], acc[1], (H, W)


def coverage(mask, size_hw, grid_hw):
    """Share of each patch inside `mask` (a bool tensor at any size), for an image shown at `size_hw`."""
    m = F.interpolate(mask[None, None].float(), size_hw, mode="nearest")
    return F.adaptive_avg_pool2d(m, grid_hw)[0, 0]


def crop_view(model, img, mask, device, long=448, margin=0.15):
    """The reference object alone: class token, patch tokens and mask coverage of its box, long side `long`."""
    ys, xs = torch.nonzero(mask, as_tuple=True)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    box = (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(img.size[0], math.ceil(x1 + mx)), min(img.size[1], math.ceil(y1 + my)))
    crop, m = img.crop(box), mask[box[1]:box[3], box[0]:box[2]]
    k = long / max(crop.size)
    W, H = max(96, round(crop.size[0] * k / 16) * 16), max(96, round(crop.size[1] * k / 16) * 16)
    cls, tok = head_tokens(model, image_tensor(crop, (W, H), device)[None])
    return cls[0], tok[0], coverage(m.to(device), (H, W), (H // 16, W // 16))


# ---------------------------------------------------------------- evidence (pure tensor code; runs anywhere)
def naming(kind, ref, e_patch, e_full, scale):
    """Weights over the vocabulary for the reference object; one-hot for `_top1`, the model's own posterior for `_bma`.
    ref: tok [N, D] normalised, cov [N]; c_cls [D], c_tok [n, D], c_cov [n] (not normalised, as the head gives them)."""
    cov = ref["cov"]
    if kind.startswith("crop"):  # the model's trained image-level reading: class token and mean patch token, here inside the mask
        g = torch.cat([ref["c_cls"], (ref["c_cov"][:, None] * ref["c_tok"]).sum(0) / ref["c_cov"].sum().clamp_min(1e-6)])
        logit = scale * F.normalize(g, dim=0) @ e_full.T
    elif kind.startswith("pool"):
        logit = scale * F.normalize((cov[:, None] * ref["tok"]).sum(0), dim=0) @ e_patch.T
    else:  # like_top1: the name whose map on the reference explains the mask best (broad names fire outside it)
        p = (scale * ref["tok"] @ e_patch.T).softmax(1)
        inside = cov >= min(0.5, float(cov.max()))
        logit = p[inside].clamp_min(1e-6).log().mean(0)
        if (~inside).any():
            logit = logit + (1 - p[~inside]).clamp_min(1e-6).log().mean(0)
    return logit.softmax(0) if kind.endswith("bma") else F.one_hot(logit.argmax(), len(logit)).to(logit.dtype)


def evidence(kind, q_tok, ref, e_patch, e_full, scale, true_row=None, p_q=None):
    """One value per query token, higher = more likely the reference's concept.
    space: similarity to the reference region in the text-aligned space (no word involved, the same-space control);
    *_cos: plain cosine to the chosen word, as the official notebook reads it (no temperature);
    oracle*: the same two readings with the true class name (uses the label: the ceiling of the channel, not a method)."""
    if kind == "space":
        return q_tok @ F.normalize((ref["cov"][:, None] * ref["tok"]).sum(0), dim=0)
    if kind == "oracle_cos":
        return q_tok @ true_row
    if kind == "oracle":
        return ((scale * q_tok @ torch.cat([e_patch, true_row[None]]).T).softmax(1)[:, -1] + EPS).log()
    if kind.endswith("_cos"):
        return q_tok @ e_patch[naming(kind[:-4] + "_top1", ref, e_patch, e_full, scale).argmax()]
    p = (scale * q_tok @ e_patch.T).softmax(1) if p_q is None else p_q
    return (p @ naming(kind, ref, e_patch, e_full, scale) + EPS).log()


def to_grid(v, hw, grid=(64, 64)):
    """Evidence on its own patch grid -> FoRIS's score grid."""
    v = v.reshape(tuple(hw))
    return v if tuple(hw) == tuple(grid) else F.interpolate(v[None, None], tuple(grid), mode="bilinear", align_corners=False)[0, 0]


def normalise(s):
    s = s - s.min()
    return s / s.max().clamp_min(1e-6)


def fuse(score, ev, lam):
    """Two readings of the same patch, each put on its own 0..1 range, added; FoRIS's cut follows unchanged."""
    return normalise(score) + lam * normalise(ev)


def binarise(score, hw, t=0.5):
    """FoRIS's `_binarize_response`: min-max, bilinear to the image, midpoint."""
    return F.interpolate(normalise(score)[None, None], size=hw, mode="bilinear", align_corners=False)[0, 0] > t


def iu(pred, truth):
    return [int((pred & truth).sum()), int((pred | truth).sum())]


def contested_auc(field, target, native):
    """P(a target patch outranks a non-target patch) inside native mask + target; None if one side is empty."""
    area = target | native
    pos, neg = field[area & target], field[area & ~target]
    if len(pos) == 0 or len(neg) == 0:
        return None
    rank = torch.cat([pos, neg]).argsort().argsort().float()[:len(pos)] + 1
    return float((rank.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- reading
def class_miou(i_u, cls, pick=None):
    import numpy as np
    i, u, k = (i_u[:, 0], i_u[:, 1], cls) if pick is None else (i_u[pick, 0], i_u[pick, 1], cls[pick])
    return 100 * float(np.mean([i[k == c].sum() / max(u[k == c].sum(), 1) for c in np.unique(k)]))


def photo_groups(rows):
    """Episodes that share a photograph are resampled together."""
    import numpy as np
    parent, owner = list(range(len(rows))), {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for n, r in enumerate(rows):
        for key in ("support", "query"):
            if r.get(key) is None:
                continue
            if r[key] in owner:
                parent[find(n)] = find(owner[r[key]])
            else:
                owner[r[key]] = n
    groups = {}
    for n in range(len(rows)):
        groups.setdefault(find(n), []).append(n)
    return [np.asarray(g, int) for g in groups.values()]


def paired(a, b, cls, groups, draws=2000):
    """Class mIoU of a, its gain over b, and the 95% interval of the gain (photograph groups, seed 0)."""
    import numpy as np
    rng, boot = np.random.default_rng(0), []
    for _ in range(draws):
        pick = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        boot.append(class_miou(a, cls, pick) - class_miou(b, cls, pick))
    up = int(((a[:, 0] / np.maximum(a[:, 1], 1)) > (b[:, 0] / np.maximum(b[:, 1], 1)) + 1e-9).sum())
    down = int(((a[:, 0] / np.maximum(a[:, 1], 1)) < (b[:, 0] / np.maximum(b[:, 1], 1)) - 1e-9).sum())
    return dict(miou=class_miou(a, cls), gain=class_miou(a, cls) - class_miou(b, cls),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], up=up, down=down, groups=len(groups))
