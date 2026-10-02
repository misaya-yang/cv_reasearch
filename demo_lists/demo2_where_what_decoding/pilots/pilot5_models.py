"""Pilot 5: does the dataset-level plug-in decoder still help stronger models and mask-classification models?
Same protocol as pilot 3, with model adapters.  Usage: python pilot5_models.py <family> <local_model_dir> <short_side> [num_images]
family in {segformer, upernet, mask2former}
"""
import os, sys, glob, json, time, numpy as np, torch, torch.nn.functional as F
from PIL import Image
D = os.environ.get("DEMO2_ROOT", "/root/autodl-tmp/demo2_pilot"); sys.path.append(D + "/env")
import transformers
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
family, mdir, short = sys.argv[1], sys.argv[2], int(sys.argv[3]); NI = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
ITERS = int(os.environ.get("ITERS", 4)); dev = "cuda:0"; C = 150
cls = {"segformer": transformers.SegformerForSemanticSegmentation, "upernet": transformers.UperNetForSemanticSegmentation,
       "mask2former": transformers.Mask2FormerForUniversalSegmentation}[family]
model = cls.from_pretrained(mdir).to(dev).eval()
mean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(3, 1, 1); std = torch.tensor([58.395, 57.12, 57.375], device=dev).view(3, 1, 1)
imgs = sorted(glob.glob(D + "/data/ADEChallengeData2016/images/validation/*.jpg"))[:NI]
store, labels = [], []
t0 = time.time()
with torch.inference_mode():
    for f in imgs:
        im = Image.open(f).convert("RGB"); w, h = im.size
        s = min(4 * short / max(h, w), short / min(h, w))
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        x = ((torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BILINEAR)).copy()).to(dev).permute(2, 0, 1).float() - mean) / std)[None]
        if family == "segformer": store.append(model(pixel_values=x).logits[0].half().cpu())
        elif family == "upernet":
            feats = model.backbone.forward_with_filtered_kwargs(x).feature_maps
            store.append(model.decode_head(feats)[0].half().cpu())
        else:
            o = model(pixel_values=x)
            store.append((torch.softmax(o.class_queries_logits[0], -1)[:, :-1].half().cpu(), o.masks_queries_logits[0].half().cpu()))
        lab = torch.from_numpy(np.asarray(Image.open(f.replace("images", "annotations").replace(".jpg", ".png"))).astype(np.int64))
        labels.append(torch.where(lab == 0, torch.full_like(lab, 255), lab - 1).to(torch.uint8))
print(f"inference done {len(imgs)} images in {time.time()-t0:.0f}s", flush=True)

def probs(i, T=1.0):
    size = labels[i].shape
    if family == "mask2former":
        pc, ml = store[i]
        m = torch.sigmoid(F.interpolate(ml.to(dev).float()[None], size=size, mode="bilinear", align_corners=False)[0])
        s = torch.einsum("qc,qhw->chw", pc.to(dev).float(), m).clamp(min=1e-12)
        if T != 1.0: s = s ** (1.0 / T)
        return s / s.sum(0, keepdim=True)
    l = F.interpolate(store[i].to(dev).float()[None], size=size, mode="bilinear", align_corners=False)[0]
    return torch.softmax(l / T, 0)

def sweep(idx, a, b, T=1.0):
    conf = torch.zeros(C * C, device=dev, dtype=torch.float64)
    sTP = torch.zeros(C, device=dev, dtype=torch.float64); sP = torch.zeros_like(sTP); sG = torch.zeros_like(sTP); n = 0
    for i in idx:
        p = probs(i, T); pred = (a.view(C, 1, 1) * p - b.view(C, 1, 1)).argmax(0)
        gt = labels[i].to(dev).long(); v = gt != 255
        conf += torch.bincount(gt[v] * C + pred[v], minlength=C * C).double()
        sP += torch.bincount(pred.flatten(), minlength=C).double(); sG += p.sum((1, 2)).double()
        sTP.scatter_add_(0, pred.flatten(), p.gather(0, pred[None])[0].flatten().double()); n += pred.numel()
    return conf.view(C, C), (sTP, sP, sG, n)
def metrics(conf):
    TP = conf.diag(); G = conf.sum(1); P = conf.sum(0); iou = TP / (G + P - TP)
    return dict(mIoU=round(float(torch.nanmean(iou)) * 100, 3), mAcc=round(float(torch.nanmean(TP / G)) * 100, 3), aAcc=round(float(TP.sum() / conf.sum()) * 100, 3))
def params(TP, P, G, n, smooth=1e-7):
    G = G / n + smooth; P = P / n; TP = TP / n; U = (G + P - TP).clamp(min=smooth); return (TP / U).clamp(0, 1).float(), U.float()
def expected_miou(sTP, sP, sG, n):
    J, _ = params(sTP, sP, sG, n); return round(float(J.mean()) * 100, 3)
def fixed_point(idx, T, labeled):
    a, b = torch.ones(C, device=dev), torch.zeros(C, device=dev); J = U = None; traj, psi = [], []
    for k in range(ITERS + 1):
        conf, soft = sweep(idx, a, b, T); traj.append(metrics(conf)["mIoU"]); psi.append(expected_miou(*soft))
        if k == ITERS: break
        Jn, Un = params(conf.diag(), conf.sum(0), conf.sum(1), conf.sum()) if labeled else params(*soft)
        J = Jn if J is None else 0.5 * J + 0.5 * Jn; U = Un if U is None else torch.exp(0.5 * U.log() + 0.5 * Un.log())
        a, b = (1 + J) / U, J / U
    return a, b, traj, psi, conf

res = {"family": family, "model": mdir, "images": len(imgs)}
ALL = list(range(len(imgs))); F0, F1 = ALL[0::2], ALL[1::2]; one, zero = torch.ones(C, device=dev), torch.zeros(C, device=dev)
with torch.inference_mode():
    res["argmax"] = metrics(sweep(ALL, one, zero)[0]); print("argmax", res["argmax"], flush=True)
    best = None
    for T in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        nll = 0.0; m = 0
        for i in F0[::10]:
            p = probs(i, T); gt = labels[i].to(dev).long(); v = gt != 255
            nll += float(-torch.log(p.gather(0, gt.clamp(max=C - 1)[None])[0][v].clamp(min=1e-12)).sum()); m += int(v.sum())
        if best is None or nll / m < best[1]: best = (T, nll / m)
    Ts = best[0]; res["T_star"] = Ts; print("T*", best, flush=True)
    for T in sorted({1.0, Ts}):
        a, b, traj, psi, conf = fixed_point(ALL, T, labeled=False)
        res[f"plugin_labelfree_T={T}"] = metrics(conf); res[f"plugin_labelfree_T={T}_traj"] = traj; res[f"plugin_labelfree_T={T}_expected"] = psi
        print("label-free T", T, res[f"plugin_labelfree_T={T}"], "true:", traj, "expected:", psi, flush=True)
    c = torch.zeros(C, C, device=dev, dtype=torch.float64)
    for fit, app in ((F0, F1), (F1, F0)):
        a, b, _, _, _ = fixed_point(fit, 1.0, labeled=True); c += sweep(app, a, b, 1.0)[0]
    res["plugin_crossfit_labeled_T=1"] = metrics(c); print("crossfit labeled", res["plugin_crossfit_labeled_T=1"], flush=True)
print(json.dumps(res, indent=1)); json.dump(res, open(D + f"/pilot5_{os.path.basename(mdir.rstrip('/'))}.json", "w"), indent=1)
