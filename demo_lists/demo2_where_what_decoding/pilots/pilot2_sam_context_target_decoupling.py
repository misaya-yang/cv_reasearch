"""Pilot 2: decouple context set (object code) from target set (pointwise evaluation). stdout only."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from PIL import Image
ROOT = "/root/autodl-tmp/demo1_sam"
sys.path.insert(0, ROOT + "/assets/source/segment-anything")
from segment_anything import sam_model_registry
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = "cuda:0"
sam = sam_model_registry["vit_b"](checkpoint=ROOT + "/assets/checkpoints/sam_vit_b_01ec64.pth").to(dev).eval()
dec, penc = sam.mask_decoder, sam.prompt_encoder
L1, L2 = dec.transformer.layers; tr = dec.transformer

def tokens_init(sp):
    out = torch.cat([dec.iou_token.weight, dec.mask_tokens.weight], 0)[None].expand(sp.size(0), -1, -1)
    return torch.cat((out, sp), 1)
def tok_part(layer, q, t0, keys, pos):
    if layer.skip_first_layer_pe: q = layer.self_attn(q=q, k=q, v=q)
    else:
        a = q + t0; q = q + layer.self_attn(q=a, k=a, v=q)
    q = layer.norm1(q)
    q = layer.norm2(q + layer.cross_attn_token_to_image(q=q + t0, k=keys + pos, v=keys))
    return layer.norm3(q + layer.mlp(q))
def write(layer, q, t0, keys, pos):
    return layer.norm4(keys + layer.cross_attn_image_to_token(q=keys + pos, k=q + t0, v=q))
def final(q, t0, keys, pos):
    return tr.norm_final_attn(q + tr.final_attn_token_to_image(q=q + t0, k=keys + pos, v=keys))
def head(keys, g):
    return dec.output_upscaling(keys.transpose(1, 2).reshape(keys.size(0), 256, g, g))
def hyper(q): return torch.stack([dec.output_hypernetworks_mlps[i](q[:, 1 + i]) for i in range(4)], 1)

def code(sp, ctx, cpos):
    """Object code from a context token set (1,Nc,256)."""
    t0 = tokens_init(sp)
    q1 = tok_part(L1, t0, t0, ctx, cpos); c1 = write(L1, q1, t0, ctx, cpos)
    q2 = tok_part(L2, q1, t0, c1, cpos);  c2 = write(L2, q2, t0, c1, cpos)
    qf = final(q2, t0, c2, cpos)
    return dict(t0=t0, q1=q1, q2=q2, h=hyper(qf), iou=dec.iou_prediction_head(qf[:, 0]))
def evaluate(c, tgt, tpos, g):
    """Pointwise evaluation of the object code on any target token set arranged as a g x g grid."""
    x = write(L2, c["q2"], c["t0"], write(L1, c["q1"], c["t0"], tgt, tpos), tpos)
    up = head(x, g)
    return (c["h"] @ up.reshape(up.size(0), 32, -1)).view(-1, 4, 4 * g, 4 * g)
def iou(a, b):
    i = (a & b).flatten(-2).sum(-1).float(); u = (a | b).flatten(-2).sum(-1).float()
    return torch.where(u > 0, i / u.clamp(min=1), torch.ones_like(u))
def bband(m, r):  # inner boundary band of a bool mask (.., H, W)
    s = m.shape; x = m.reshape(-1, 1, *s[-2:]).float()
    er = -F.max_pool2d(-x, 2 * r + 1, 1, r)
    return (x - er > 0).reshape(s)
def biou(a, b, r): return iou(bband(a, r), bband(b, r))

man = json.load(open(ROOT + "/assets/coco_quality_seed2027_v1/manifest.json"))
NI = int(sys.argv[1]) if len(sys.argv) > 1 else 8
acc = {}
def add(k, v): acc.setdefault(k, []).extend(torch.as_tensor(v).flatten().float().cpu().tolist())
mean, std = sam.pixel_mean, sam.pixel_std
def encode(x):  # x: (B,3,1024,1024) uint8-range float, already padded with zeros AFTER normalisation
    return sam.image_encoder(x)
def prep(img_t):  # (3,h,w) float 0..255 -> normalised + zero-padded 1024
    x = (img_t - mean) / std
    return F.pad(x, (0, 1024 - x.shape[-1], 0, 1024 - x.shape[-2]))

with torch.inference_mode():
    for info in man["images"][:NI]:
        pil = Image.open(f"{ROOT}/assets/coco_quality_seed2027_v1/{info['image_file']}").convert("RGB")
        H, W = info["height"], info["width"]; s = 1024 / max(H, W); nh, nw = int(H * s + .5), int(W * s + .5)
        im1 = torch.from_numpy(np.asarray(pil.resize((nw, nh), Image.BILINEAR))).permute(2, 0, 1).float().to(dev)
        E = encode(prep(im1)[None])                                             # (1,256,64,64)
        nomask = penc.no_mask_embed.weight.reshape(1, 1, 256)
        ctx = E.flatten(2).permute(0, 2, 1) + nomask
        pos = penc.get_dense_pe().flatten(2).permute(0, 2, 1)
        gt = torch.from_numpy(np.load(f"{ROOT}/assets/coco_quality_seed2027_v1/{info['gt_file']}")["gt"]).to(dev)   # (K,H,W) bool
        # ---- 2x tiles: resize original to long side 2048, pad to 2048, cut 2x2 tiles, encode each
        im2 = torch.from_numpy(np.asarray(pil.resize((2 * nw, 2 * nh), Image.BICUBIC))).permute(2, 0, 1).float().to(dev)
        x2 = (im2 - mean) / std; x2 = F.pad(x2, (0, 2048 - x2.shape[-1], 0, 2048 - x2.shape[-2]))
        tiles = torch.stack([x2[:, a:a + 1024, b:b + 1024] for a in (0, 1024) for b in (0, 1024)])
        Et = torch.cat([encode(t[None]) for t in tiles])                         # (4,256,64,64)
        E2 = torch.cat([torch.cat([Et[0], Et[1]], -1), torch.cat([Et[2], Et[3]], -1)], -2)[None]   # (1,256,128,128)
        tgt2 = E2.flatten(2).permute(0, 2, 1) + nomask
        pos2 = penc.pe_layer((128, 128)).unsqueeze(0).flatten(2).permute(0, 2, 1)
        objs = info["objects"]
        pts = torch.tensor([o["central_xy"] for o in objs], device=dev).float()[:, None] * s
        box = torch.tensor([o["tight_box_xyxy"] for o in objs], device=dev).float() * s
        for regime, sp in (("point", penc(points=(pts, torch.ones(len(objs), 1, device=dev)), boxes=None, masks=None)[0]),
                           ("box", penc(points=None, boxes=box, masks=None)[0])):
            c = code(sp, ctx, pos)
            std_l = evaluate(c, ctx, pos, 64)                                   # (P,4,256,256) == official SAM
            ref = dec.predict_masks(E, penc.get_dense_pe(), sp, penc.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(len(objs), -1, 64, 64))
            add("sanity_maxerr_vs_official", (std_l - ref[0]).abs().max())
            sel = c["iou"][:, 1:].argmax(1) + 1 if regime == "point" else torch.zeros(len(objs), dtype=torch.long, device=dev)
            def to_orig(l):  # logits on padded square -> original image size, like Sam.postprocess_masks
                l = F.interpolate(l, (1024, 1024), mode="bilinear", align_corners=False)[..., :nh, :nw]
                return F.interpolate(l, (H, W), mode="bilinear", align_corners=False) > 0
            m_std = to_orig(std_l)
            pick = lambda m: m[torch.arange(len(objs)), sel]
            add(f"{regime}/std_gtIoU", iou(pick(m_std), gt)); add(f"{regime}/std_gtBIoU", biou(pick(m_std), gt, 3))
            # ---- A. context subsampling: object code from fewer context tokens, evaluated on the full grid
            idx_all = torch.arange(4096, device=dev).view(64, 64)
            for name, sub in (("stride2", idx_all[::2, ::2]), ("stride2_off", idx_all[1::2, 1::2]), ("stride4", idx_all[1::4, 1::4]), ("stride8", idx_all[3::8, 3::8])):
                ii = sub.flatten()
                cs = code(sp, ctx[:, ii], pos[:, ii]); m = to_orig(evaluate(cs, ctx, pos, 64))
                add(f"{regime}/A_ctx={name}_agree_all4", iou(m, m_std)); add(f"{regime}/A_ctx={name}_gtIoU", iou(pick(m), gt))
                add(f"{regime}/A_ctx={name}_agree_frac<0.95", (iou(m, m_std) < 0.95).float()); add(f"{regime}/A_ctx={name}_ioupred_err", (cs["iou"] - c["iou"]).abs())
            # ---- B. target super-resolution: same object code, evaluated on 2x tile tokens (128x128 grid -> 512x512 logits)
            for cname, cc in (("global", c), ("tiles", code(sp, tgt2, pos2)), ("global+tiles", code(sp, torch.cat([ctx, tgt2], 1), torch.cat([pos, pos2], 1)))):
                m = to_orig(evaluate(cc, tgt2, pos2, 128))
                selc = cc["iou"][:, 1:].argmax(1) + 1 if regime == "point" else sel
                add(f"{regime}/B_code={cname}_target=tiles_agree_all4", iou(m, m_std))
                add(f"{regime}/B_code={cname}_target=tiles_gtIoU", iou(m[torch.arange(len(objs)), selc], gt)); add(f"{regime}/B_code={cname}_target=tiles_gtBIoU", biou(m[torch.arange(len(objs)), selc], gt, 3))
            # code from tiles, evaluated on the standard grid (isolates code vs target effect)
            m = to_orig(evaluate(code(sp, tgt2, pos2), ctx, pos, 64)); add(f"{regime}/B_code=tiles_target=global_agree_all4", iou(m, m_std))
        print("done", info["image_id"], flush=True)
print(json.dumps({k: [round(float(np.mean(v)), 4), len(v)] for k, v in acc.items()}, indent=1))
