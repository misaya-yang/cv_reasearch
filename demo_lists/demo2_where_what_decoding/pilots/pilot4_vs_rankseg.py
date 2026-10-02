"""Pilot 4: closest prior art (RankSEG, per-image Dice/IoU decoding) vs the dataset-level plug-in rule,
scored with BOTH the standard dataset-level mIoU and a per-image mIoU.
Usage: python pilot4_vs_rankseg.py <hf_model_name> <short_side> [num_images]
"""
import os, sys, glob, json, time, numpy as np, torch, torch.nn.functional as F
from PIL import Image
D = os.environ.get("DEMO2_ROOT", "/root/autodl-tmp/demo2_pilot"); sys.path.append(D + "/env")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com"); os.environ.setdefault("HF_HOME", D + "/hf")
from transformers import SegformerForSemanticSegmentation
from rankseg import RankSEG
name, short = sys.argv[1], int(sys.argv[2]); NI = int(sys.argv[3]) if len(sys.argv) > 3 else 500
dev = "cuda:0"; C = 150
model = SegformerForSemanticSegmentation.from_pretrained(name).to(dev).eval()
mean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(3, 1, 1); std = torch.tensor([58.395, 57.12, 57.375], device=dev).view(3, 1, 1)
imgs = sorted(glob.glob(D + "/data/ADEChallengeData2016/images/validation/*.jpg"))[:NI]
logits, labels = [], []
with torch.inference_mode():
    for f in imgs:
        im = Image.open(f).convert("RGB"); w, h = im.size
        s = min(4 * short / max(h, w), short / min(h, w))
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        x = torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BILINEAR)).copy()).to(dev).permute(2, 0, 1).float()
        logits.append(model(pixel_values=((x - mean) / std)[None]).logits[0].half().cpu())
        lab = torch.from_numpy(np.asarray(Image.open(f.replace("images", "annotations").replace(".jpg", ".png"))).astype(np.int64))
        labels.append(torch.where(lab == 0, torch.full_like(lab, 255), lab - 1).to(torch.uint8))
def probs(i):
    return torch.softmax(F.interpolate(logits[i].to(dev).float()[None], size=labels[i].shape, mode="bilinear", align_corners=False)[0], 0)

class Score:
    def __init__(s): s.conf = torch.zeros(C * C, device=dev, dtype=torch.float64); s.per_img = []; s.t = 0.0
    def add(s, pred, gt):
        v = gt != 255; c = torch.bincount(gt[v] * C + pred[v], minlength=C * C).double(); s.conf += c
        c = c.view(C, C); TP = c.diag(); G = c.sum(1); P = c.sum(0); present = G > 0
        s.per_img.append(float((TP[present] / (G + P - TP)[present]).mean()))
    def out(s):
        c = s.conf.view(C, C); TP = c.diag(); G = c.sum(1); P = c.sum(0)
        return dict(dataset_mIoU=round(float(torch.nanmean(TP / (G + P - TP))) * 100, 3), per_image_mIoU=round(float(np.mean(s.per_img)) * 100, 3),
                    mAcc=round(float(torch.nanmean(TP / G)) * 100, 3), aAcc=round(float(TP.sum() / c.sum()) * 100, 3), sec_per_img=round(s.t / len(imgs), 4))

methods = {"argmax": None, "rankseg_iou_rma": RankSEG(metric="IoU", output_mode="multiclass", solver="RMA"),
           "rankseg_dice_rma": RankSEG(metric="dice", output_mode="multiclass", solver="RMA")}
S = {k: Score() for k in list(methods) + ["ours_dataset_plugin_labelfree"]}
with torch.inference_mode():
    # ours: label-free fixed point over the whole evaluated set (3 sweeps), then score
    a, b = torch.ones(C, device=dev), torch.zeros(C, device=dev); J = U = None
    for k in range(3):
        sTP = torch.zeros(C, device=dev, dtype=torch.float64); sP = torch.zeros_like(sTP); sG = torch.zeros_like(sTP); n = 0
        for i in range(len(imgs)):
            p = probs(i); pred = (a.view(C, 1, 1) * p - b.view(C, 1, 1)).argmax(0)
            sP += torch.bincount(pred.flatten(), minlength=C).double(); sG += p.sum((1, 2)).double()
            sTP.scatter_add_(0, pred.flatten(), p.gather(0, pred[None])[0].flatten().double()); n += pred.numel()
        G_, P_, TP_ = sG / n + 1e-7, sP / n, sTP / n; Un = (G_ + P_ - TP_).clamp(min=1e-7).float(); Jn = (TP_.float() / Un).clamp(0, 1)
        J = Jn if J is None else 0.5 * J + 0.5 * Jn; U = Un if U is None else torch.exp(0.5 * U.log() + 0.5 * Un.log())
        a, b = (1 + J) / U, J / U
    for i in range(len(imgs)):
        p = probs(i); gt = labels[i].to(dev).long()
        t = time.time(); pred = (a.view(C, 1, 1) * p - b.view(C, 1, 1)).argmax(0); torch.cuda.synchronize(); S["ours_dataset_plugin_labelfree"].t += time.time() - t
        S["ours_dataset_plugin_labelfree"].add(pred, gt)
        for k, m in methods.items():
            t = time.time()
            pred = p.argmax(0) if m is None else m(p[None])[0].long()
            torch.cuda.synchronize(); S[k].t += time.time() - t; S[k].add(pred, gt)
        if i % 100 == 0: print(i, {k: v.out() for k, v in S.items()}, flush=True)
res = {"model": name, "images": len(imgs), **{k: v.out() for k, v in S.items()}}
print(json.dumps(res, indent=1)); json.dump(res, open(D + f"/pilot4_{name.split('/')[-1]}.json", "w"), indent=1)
