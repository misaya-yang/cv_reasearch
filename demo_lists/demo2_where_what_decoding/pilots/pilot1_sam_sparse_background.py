"""Pilot 1: is the per-prompt image-state deviation spatially sparse? stdout only."""
import sys, json, glob, numpy as np, torch, torch.nn.functional as F
ROOT = "/root/autodl-tmp/demo1_sam"
sys.path.insert(0, ROOT + "/assets/source/segment-anything")
from segment_anything import sam_model_registry
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = "cuda:0"
sam = sam_model_registry["vit_b"](checkpoint=ROOT + "/assets/checkpoints/sam_vit_b_01ec64.pth").to(dev).eval()
dec, penc = sam.mask_decoder, sam.prompt_encoder
L1, L2 = dec.transformer.layers
tr = dec.transformer

def tokens_init(sp):
    out = torch.cat([dec.iou_token.weight, dec.mask_tokens.weight], 0)[None].expand(sp.size(0), -1, -1)
    return torch.cat((out, sp), 1)
def tok_part(layer, q, t0, keys, pos):
    if layer.skip_first_layer_pe:
        q = layer.self_attn(q=q, k=q, v=q)
    else:
        a = q + t0; q = q + layer.self_attn(q=a, k=a, v=q)
    q = layer.norm1(q)
    q = layer.norm2(q + layer.cross_attn_token_to_image(q=q + t0, k=keys + pos, v=keys))
    return layer.norm3(q + layer.mlp(q))
def write(layer, q, t0, keys, pos):
    return layer.norm4(keys + layer.cross_attn_image_to_token(q=keys + pos, k=q + t0, v=q))
def final(q, t0, keys, pos):
    return tr.norm_final_attn(q + tr.final_attn_token_to_image(q=q + t0, k=keys + pos, v=keys))
def head(keys):
    return dec.output_upscaling(keys.transpose(1, 2).reshape(keys.size(0), 256, 64, 64))
def hyper(q):
    return torch.stack([dec.output_hypernetworks_mlps[i](q[:, 1 + i]) for i in range(4)], 1)
def to_logits(h, up):
    return (h @ up.reshape(up.size(0), 32, -1)).view(-1, 4, 256, 256)
def parents(mask_bool):  # (P,256,256)->(P,64,64) any child positive
    return F.max_pool2d(mask_bool.float()[:, None], 4)[:, 0] > 0
def dilate(a, r):
    return a if r == 0 else F.max_pool2d(a.float()[:, None], 2 * r + 1, 1, r)[:, 0] > 0

def exact(sp, base, pos):
    t0 = tokens_init(sp)
    q1 = tok_part(L1, t0, t0, base, pos); X1 = write(L1, q1, t0, base, pos)
    q2 = tok_part(L2, q1, t0, X1, pos);   X2 = write(L2, q2, t0, X1, pos)
    qf = final(q2, t0, X2, pos)
    return dict(t0=t0, q1=q1, X1=X1, X2=X2, qf=qf, logits=to_logits(hyper(qf), head(X2)), iou=dec.iou_prediction_head(qf[:, 0]))

def approx(e, A, R1, R2, pos):
    """A:(P,N) bool active; outside A the image state is replaced by shared reference R1/R2 (1,N,256)."""
    t0, q1 = e["t0"], e["q1"]
    X1m = torch.where(A[..., None], e["X1"], R1)
    q2 = tok_part(L2, q1, t0, X1m, pos)
    X2m = torch.where(A[..., None], write(L2, q2, t0, X1m, pos), R2)
    qf = final(q2, t0, X2m, pos)
    return to_logits(hyper(qf), head(X2m)), dec.iou_prediction_head(qf[:, 0])

def agree(la, le, A=None):
    """per-mask flips and IoU over multimask tokens 1..3; optional negative fill outside active parents"""
    ba, be = la[:, 1:] > 0, le[:, 1:] > 0
    if A is not None:
        up = A.view(-1, 1, 64, 64).float().repeat_interleave(4, 2).repeat_interleave(4, 3) > 0
        ba = ba & up
    flips = (ba != be).flatten(2).sum(-1).float()
    inter = (ba & be).flatten(2).sum(-1).float(); union = (ba | be).flatten(2).sum(-1).float()
    return flips, torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))

files = sorted(glob.glob(ROOT + "/results/takeover_20261001_v1/real_original/image_*/encoded_inputs.npz"))[:int(sys.argv[1]) if len(sys.argv) > 1 else 6]
G = 32; CH = 64
acc = {}
def add(k, v): acc.setdefault(k, []).append(float(v))
with torch.inference_mode():
    for f in files:
        z = np.load(f)
        img, pe, dense = [torch.from_numpy(z[k]).to(dev) for k in ("image_embeddings", "image_pe", "dense_nomask")]
        nh, nw = [int(v) for v in z["input_size"]]
        base = (img + dense).flatten(2).permute(0, 2, 1); pos = pe.flatten(2).permute(0, 2, 1)
        u = (torch.arange(G, device=dev) + 0.5) / G
        pts = torch.stack(torch.meshgrid(u * nw, u * nh, indexing="xy"), -1).reshape(-1, 1, 2)
        sp_all, _ = penc(points=(pts, torch.ones(len(pts), 1, device=dev)), boxes=None, masks=None)
        # sanity: manual forward == official
        e = exact(sp_all[:8], base, pos); ref = dec.predict_masks(img, pe, sp_all[:8], dense.expand(8, -1, -1, -1))
        add("sanity_manual_vs_official_maxerr", (e["logits"] - ref[0]).abs().max())
        # references: null-prompt trajectory, and cross-prompt mean (oracle-ish)
        n = exact(sp_all[:1, :0], base, pos); N1, N2 = n["X1"], n["X2"]
        M1 = torch.zeros_like(N1); M2 = torch.zeros_like(N2)
        for lo in range(0, len(sp_all), CH):
            e = exact(sp_all[lo:lo + CH], base, pos); M1 += e["X1"].sum(0, keepdim=True); M2 += e["X2"].sum(0, keepdim=True)
        M1 /= len(sp_all); M2 /= len(sp_all)
        upN, upM = head(N2), head(M2)
        for lo in range(0, len(sp_all), CH):
            sp = sp_all[lo:lo + CH]; e = exact(sp, base, pos); le = e["logits"]; P = len(sp)
            # noise floor: same prompts, different microbatch
            e2 = torch.cat([exact(sp[i:i + 16], base, pos)["logits"] for i in range(0, P, 16)])
            fl, io = agree(e2, le); add("floor_flips_per_mask", fl.mean()); add("floor_frac_iou<0.99", (io < 0.99).float().mean())
            fg = parents((le > 0).any(1)).flatten(1)                     # (P,N) union of 4 masks at parent level
            band = dilate(fg.view(P, 64, 64), 2).flatten(1) & ~fg; far = ~(fg | band)
            add("fg_parent_frac", fg.float().mean())
            for name, R in (("null", N2), ("mean", M2)):
                d = (e["X2"] - R).norm(dim=-1) / R.norm(dim=-1)
                for rn, reg in (("fg", fg), ("band", band), ("far", far)):
                    if reg.any(): add(f"reldev_X2_vs_{name}_{rn}", d[reg].mean())
            # C: oracle active set (true FG parents dilated r), shared reference elsewhere
            for name, R1, R2 in (("null", N1, N2), ("mean", M1, M2)):
                for r in (0, 1, 2, 4):
                    A = dilate(fg.view(P, 64, 64), r).flatten(1)
                    la, ia = approx(e, A, R1, R2, pos)
                    for fill, Af in (("coarse", None), ("neg", A)):
                        fl, io = agree(la, le, Af)
                        k = f"C_oracleA_ref={name}_r={r}_fill={fill}"
                        add(k + "_flips_per_mask", fl.mean()); add(k + "_meanIoU", io.mean()); add(k + "_frac_iou<0.99", (io < 0.99).float().mean())
                    add(f"C_oracleA_ref={name}_r={r}_active_frac", A.float().mean())
                    add(f"C_oracleA_ref={name}_r={r}_ioupred_abs_err", (ia - e["iou"]).abs().mean())
            # D: non-oracle active set from an all-shared pass 0
            for name, R1, R2, upR in (("null", N1, N2, upN), ("mean", M1, M2, upM)):
                q2 = tok_part(L2, e["q1"], e["t0"], R1, pos); qf = final(q2, e["t0"], R2, pos)
                coarse = to_logits(hyper(qf), upR.expand(P, -1, -1, -1))
                fl, io = agree(coarse, le); add(f"D_pass0only_ref={name}_meanIoU", io.mean())
                for tau in (0.0, 2.0, 4.0):
                    for r in (1, 2):
                        A = dilate(parents((coarse > -tau).any(1)), r).flatten(1)
                        la, ia = approx(e, A, R1, R2, pos); fl, io = agree(la, le, A)
                        k = f"D_ref={name}_tau={tau}_r={r}"
                        add(k + "_active_frac", A.float().mean()); add(k + "_recall_of_fg_parents", (A & fg).float().sum() / fg.float().sum().clamp(min=1))
                        add(k + "_flips_per_mask", fl.mean()); add(k + "_meanIoU", io.mean()); add(k + "_frac_iou<0.99", (io < 0.99).float().mean())
        print("done", f.split("/")[-2], flush=True)
print(json.dumps({k: round(float(np.mean(v)), 5) for k, v in acc.items()}, indent=1))
