"""Pilot 3: is argmax decoding mIoU-suboptimal?  Dataset-level plug-in decoder on ADE20K val.

Rule under test (derived in THEORY.md):  y(x) = argmax_c [ (1+J_c) p_c(x) - J_c ] / U_c
where J_c is the class IoU and U_c the class union (as a fraction of pixels) at the fixed point.
Usage: python pilot3_miou_decoding.py <hf_model_name> <short_side> [num_images]
"""
import os, sys, glob, json, time, numpy as np, torch, torch.nn.functional as F
from PIL import Image
D = os.environ.get("DEMO2_ROOT", "/root/autodl-tmp/demo2_pilot"); sys.path.append(D + "/env")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com"); os.environ.setdefault("HF_HOME", D + "/hf")
from transformers import SegformerForSemanticSegmentation
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
name, short = sys.argv[1], int(sys.argv[2]); NI = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
dev = "cuda:0"; C = 150
model = SegformerForSemanticSegmentation.from_pretrained(name).to(dev).eval()
mean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(3, 1, 1); std = torch.tensor([58.395, 57.12, 57.375], device=dev).view(3, 1, 1)
imgs = sorted(glob.glob(D + "/data/ADEChallengeData2016/images/validation/*.jpg"))[:NI]
logits, labels = [], []
t0 = time.time()
with torch.inference_mode():
    for f in imgs:
        im = Image.open(f).convert("RGB"); w, h = im.size
        s = min(4 * short / max(h, w), short / min(h, w))                     # mmseg Resize(scale=(4*short, short), keep_ratio)
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        x = torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BILINEAR))).to(dev).permute(2, 0, 1).float()
        out = model(pixel_values=((x - mean) / std)[None]).logits[0]           # (C, nh/4, nw/4)
        logits.append(out.half().cpu())
        lab = torch.from_numpy(np.asarray(Image.open(f.replace("images", "annotations").replace(".jpg", ".png"))).astype(np.int64))
        labels.append(torch.where(lab == 0, torch.full_like(lab, 255), lab - 1).to(torch.uint8))
print(f"inference done {len(imgs)} images in {time.time()-t0:.0f}s", flush=True)

def probs(i, T=1.0):
    l = F.interpolate(logits[i].to(dev).float()[None], size=labels[i].shape, mode="bilinear", align_corners=False)[0]
    return torch.softmax(l / T, 0)

def sweep(idx, a, b, T=1.0):
    """One pass: decisions under score a_c p_c - b_c; returns labeled confusion and label-free (soft) stats."""
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
    TP = conf.diag(); G = conf.sum(1); P = conf.sum(0); iou = TP / (G + P - TP); acc = TP / G
    return dict(mIoU=float(torch.nanmean(iou)) * 100, mAcc=float(torch.nanmean(acc)) * 100, aAcc=float(TP.sum() / conf.sum()) * 100), iou

def params(TP, P, G, n, smooth=1e-7):
    G = G / n + smooth; P = P / n; TP = TP / n; U = (G + P - TP).clamp(min=smooth); J = (TP / U).clamp(0, 1)
    return J.float(), U.float()

def fixed_point(idx, T, labeled, iters=int(os.environ.get("ITERS", 8)), log=None, eval_idx=None):
    one, zero = torch.ones(C, device=dev), torch.zeros(C, device=dev)
    a, b = one, zero; J = U = None; traj = []
    for k in range(iters + 1):
        conf, (sTP, sP, sG, n) = sweep(idx, a, b, T)
        if eval_idx is None: traj.append(round(metrics(conf)[0]["mIoU"], 3))
        if k == iters: break
        Jn, Un = params(conf.diag(), conf.sum(0), conf.sum(1), conf.sum()) if labeled else params(sTP, sP, sG, n)
        J = Jn if J is None else 0.5 * J + 0.5 * Jn
        U = Un if U is None else torch.exp(0.5 * U.log() + 0.5 * Un.log())
        a, b = (1 + J) / U, J / U
    return a, b, traj, conf

res = {"model": name, "images": len(imgs)}
ALL = list(range(len(imgs))); F0, F1 = ALL[0::2], ALL[1::2]
one, zero = torch.ones(C, device=dev), torch.zeros(C, device=dev)
with torch.inference_mode():
    conf0, (sTP, sP, sG, n) = sweep(ALL, one, zero)
    res["argmax"], iou0 = metrics(conf0); print("argmax", res["argmax"], flush=True)
    # temperature by NLL on fold 0 (subsampled pixels)
    best = None
    for T in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        nll = 0.0; m = 0
        for i in F0[::10]:
            p = probs(i, T); gt = labels[i].to(dev).long(); v = gt != 255
            nll += float(-torch.log(p.gather(0, gt.clamp(max=C - 1)[None])[0][v].clamp(min=1e-12)).sum()); m += int(v.sum())
        if best is None or nll / m < best[1]: best = (T, nll / m)
    Ts = best[0]; res["T_star"] = Ts; print("T*", best, flush=True)
    # baselines: prior-power reweighting p_c / pi_c^gamma  (gamma=1 is the "maximum likelihood" rule), priors from the other fold's labels
    Gtrue = conf0.sum(1)
    for g in ((0.25,) if os.environ.get("FAST") else (0.25, 0.5, 0.75, 1.0)):
        c = torch.zeros(C, C, device=dev, dtype=torch.float64)
        for fit, app in ((F0, F1), (F1, F0)):
            cf, _ = sweep(fit, one, zero); pri = (cf.sum(1) / cf.sum() + 1e-7).float()
            c += sweep(app, pri ** (-g), zero)[0]
        res[f"prior_power_gamma={g}_crossfit"] = metrics(c)[0]; print("prior power", g, res[f"prior_power_gamma={g}_crossfit"], flush=True)
    # oracle: labeled fixed point fitted and evaluated on the same images (upper bound, not deployable)
    a, b, traj, conf = fixed_point(ALL, 1.0, labeled=True); res["oracle_same_set"] = metrics(conf)[0]; res["oracle_traj"] = traj
    print("oracle", res["oracle_same_set"], traj, flush=True)
    # honest labeled: 2-fold cross-fit
    for T in sorted({1.0, Ts}):
        c = torch.zeros(C, C, device=dev, dtype=torch.float64)
        for fit, app in ((F0, F1), (F1, F0)):
            a, b, _, _ = fixed_point(fit, T, labeled=True); c += sweep(app, a, b, T)[0]
        res[f"plugin_crossfit_labeled_T={T}"] = metrics(c)[0]; print("crossfit labeled T", T, res[f"plugin_crossfit_labeled_T={T}"], flush=True)
        if T == 1.0: iou_cf = metrics(c)[1]
    # label-free transductive: expected statistics under the model's own probabilities
    for T in sorted({1.0, Ts} if os.environ.get("FAST") else {1.0, Ts, 2.0}):
        a, b, traj, conf = fixed_point(ALL, T, labeled=False)
        res[f"plugin_labelfree_T={T}"] = metrics(conf)[0]; res[f"plugin_labelfree_T={T}_traj"] = traj
        print("label-free T", T, res[f"plugin_labelfree_T={T}"], traj, flush=True)
    # where does the gain come from: classes sorted by pixel frequency, split in thirds
    order = Gtrue.argsort(); third = C // 3
    for nm, sl in (("rare", order[:third]), ("mid", order[third:2 * third]), ("frequent", order[2 * third:])):
        res[f"delta_IoU_{nm}_classes_crossfit_T=1"] = float(torch.nanmean(iou_cf[sl] - iou0[sl])) * 100
print(json.dumps(res, indent=1))
json.dump(res, open(D + f"/pilot3_{name.split('/')[-1]}.json", "w"), indent=1)
