"""Pilot 6: how much of the error is "a class that is not in the image at all"?
Oracle image-level class presence (headroom) vs label-free presence estimates from the model's own posteriors.
Usage: python pilot6_presence_gating.py <local_model_dir> <short_side> [num_images] [family=segformer|mask2former|upernet]
"""
import os, sys, glob, json, time, numpy as np, torch, torch.nn.functional as F
from PIL import Image
D = os.environ.get("DEMO2_ROOT", "/root/autodl-tmp/demo2_pilot"); sys.path.append(D + "/env")
import transformers
mdir, short = sys.argv[1], int(sys.argv[2]); NI = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
family = sys.argv[4] if len(sys.argv) > 4 else "segformer"
dev = "cuda:0"; C = 150
model = {"segformer": transformers.SegformerForSemanticSegmentation, "upernet": transformers.UperNetForSemanticSegmentation,
         "mask2former": transformers.Mask2FormerForUniversalSegmentation}[family].from_pretrained(mdir).to(dev).eval()
def posterior(x, size):
    if family == "mask2former":
        o = model(pixel_values=x)
        m = torch.sigmoid(F.interpolate(o.masks_queries_logits.float(), size=size, mode="bilinear", align_corners=False)[0])
        s = torch.einsum("qc,qhw->chw", torch.softmax(o.class_queries_logits[0].float(), -1)[:, :-1], m).clamp(min=1e-12)
        return s / s.sum(0, keepdim=True)
    l = model(pixel_values=x).logits if family == "segformer" else model.decode_head(model.backbone.forward_with_filtered_kwargs(x).feature_maps)
    return torch.softmax(F.interpolate(l.float(), size=size, mode="bilinear", align_corners=False)[0], 0)
mean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(3, 1, 1); std = torch.tensor([58.395, 57.12, 57.375], device=dev).view(3, 1, 1)
imgs = sorted(glob.glob(D + "/data/ADEChallengeData2016/images/validation/*.jpg"))[:NI]
S = {}
def score(name, pred, gt):
    v = gt != 255
    S.setdefault(name, torch.zeros(C * C, device=dev, dtype=torch.float64)).add_(torch.bincount(gt[v] * C + pred[v], minlength=C * C).double())
def gated(p, present):
    q = p.clone(); q[~present] = -1; return q.argmax(0)
stats = dict(n_gt=0, n_pred=0, n_fp=0, n_fn=0, px_absent_class=0, px_wrong=0, px=0)
MASS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02); AREA = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05); MAXP = (0.5, 0.7, 0.9, 0.97)
with torch.inference_mode():
    for f in imgs:
        im = Image.open(f).convert("RGB"); w, h = im.size
        s = min(4 * short / max(h, w), short / min(h, w))
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        x = ((torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BILINEAR)).copy()).to(dev).permute(2, 0, 1).float() - mean) / std)[None]
        lab = torch.from_numpy(np.asarray(Image.open(f.replace("images", "annotations").replace(".jpg", ".png"))).astype(np.int64)).to(dev)
        gt = torch.where(lab == 0, torch.full_like(lab, 255), lab - 1)
        p = posterior(x, gt.shape)
        am = p.argmax(0); score("argmax", am, gt)
        v = gt != 255; present_gt = torch.zeros(C, dtype=torch.bool, device=dev); present_gt[gt[v].unique()] = True
        present_am = torch.zeros(C, dtype=torch.bool, device=dev); present_am[am.unique()] = True
        stats["n_gt"] += int(present_gt.sum()); stats["n_pred"] += int(present_am.sum())
        stats["n_fp"] += int((present_am & ~present_gt).sum()); stats["n_fn"] += int((present_gt & ~present_am).sum())
        wrong = v & (am != gt); stats["px_wrong"] += int(wrong.sum()); stats["px"] += int(v.sum())
        stats["px_absent_class"] += int((wrong & ~present_gt[am]).sum())      # wrong pixels whose predicted class is absent from the image
        score("oracle_presence", gated(p, present_gt), gt)                      # headroom: restrict to classes that truly occur in the image
        score("oracle_drop_absent_only", gated(p, present_gt | ~present_am), gt)  # only remove predicted-but-absent classes
        mass = p.mean((1, 2)); area = torch.bincount(am.flatten(), minlength=C).float() / am.numel(); mx = p.amax((1, 2))
        top = torch.zeros(C, dtype=torch.bool, device=dev); top[mass.argmax()] = True
        for t in MASS: score(f"mass>={t}", gated(p, (mass >= t) | top), gt)
        for t in AREA: score(f"argmax_area>={t}", gated(p, (area >= t) | top), gt)
        for t in MAXP: score(f"maxprob>={t}", gated(p, (mx >= t) | top), gt)
res = {"model": mdir, "images": len(imgs)}
for k, c in S.items():
    c = c.view(C, C); tp, g, pr = c.diag(), c.sum(1), c.sum(0)
    res[k] = dict(mIoU=round(float(torch.nanmean(tp / (g + pr - tp))) * 100, 3), mAcc=round(float(torch.nanmean(tp / g)) * 100, 3), aAcc=round(float(tp.sum() / c.sum()) * 100, 3))
res["presence_stats"] = dict(classes_per_image_gt=stats["n_gt"] / len(imgs), classes_per_image_argmax=stats["n_pred"] / len(imgs),
                             spurious_classes_per_image=stats["n_fp"] / len(imgs), missed_classes_per_image=stats["n_fn"] / len(imgs),
                             frac_wrong_pixels_predicting_absent_class=stats["px_absent_class"] / max(stats["px_wrong"], 1), pixel_error_rate=stats["px_wrong"] / stats["px"])
print(json.dumps(res, indent=1)); json.dump(res, open(D + f"/pilot6_{os.path.basename(mdir.rstrip('/'))}.json", "w"), indent=1)
