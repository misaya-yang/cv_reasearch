#!/usr/bin/env python3
"""Phase-A harness: one model x one dataset -> argmax vs metric-optimal decoding, all standard metrics.

Example (server):
  python scripts/eval_decoding.py --dataset ade20k --data-root $D/data/ADEChallengeData2016 \
      --family segformer --model-dir $D/models/segformer-b5-ade --short-side 640 --out results/ade20k/segformer-b5.json

Design notes
  * Model outputs are cached at native (1/4) resolution in host RAM as float16; every decoding variant is a
    cheap re-sweep over that cache.  Nothing large is written to disk.
  * "label-free" variants never read the ground truth to choose decisions; labels are used only to score.
  * The 2-fold "crossfit" variants fit on even-indexed images and apply to odd-indexed ones, and vice versa.
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if os.environ.get("DEMO2_ENV"): sys.path.append(os.environ["DEMO2_ENV"])       # isolated pip --target dir, appended so base packages win
from mdecode import SoftStats, decode, fit_labelfree, fit_labeled

CITY_ID2TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}


def dataset_files(name, root, limit):
    if name == "ade20k":
        imgs = sorted(glob.glob(f"{root}/images/validation/*.jpg"))
        return [(f, f.replace("/images/", "/annotations/").replace(".jpg", ".png")) for f in imgs][:limit], 150
    if name == "cityscapes":
        imgs = sorted(glob.glob(f"{root}/leftImg8bit/val/*/*_leftImg8bit.png"))
        return [(f, f.replace("/leftImg8bit/", "/gtFine/").replace("_leftImg8bit.png", "_gtFine_labelIds.png")) for f in imgs][:limit], 19
    raise ValueError(name)


def load_label(name, path):
    lab = torch.from_numpy(np.asarray(Image.open(path)).astype(np.int64))
    if name == "ade20k":                       # 0 = unlabeled -> ignore, classes shifted down by one
        return torch.where(lab == 0, torch.full_like(lab, 255), lab - 1).to(torch.uint8)
    lut = torch.full((256,), 255, dtype=torch.int64)
    for k, v in CITY_ID2TRAIN.items(): lut[k] = v
    return lut[lab].to(torch.uint8)


def input_size(name, h, w, short):
    if name == "cityscapes": return h, w       # whole-image inference at native 1024x2048
    s = min(4 * short / max(h, w), short / min(h, w))                         # mmseg Resize(scale=(4*short, short), keep_ratio=True)
    return max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("ade20k", "cityscapes")); ap.add_argument("--data-root", required=True)
    ap.add_argument("--family", required=True, choices=("segformer", "upernet", "mask2former")); ap.add_argument("--model-dir", required=True)
    ap.add_argument("--short-side", type=int, default=512); ap.add_argument("--num-images", type=int, default=10 ** 9)
    ap.add_argument("--iters", type=int, default=4); ap.add_argument("--metrics", default="miou"); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if os.path.exists(args.out): sys.exit(f"refusing to overwrite {args.out}")
    import transformers
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dev = args.device
    files, C = dataset_files(args.dataset, args.data_root, args.num_images)
    cls = {"segformer": transformers.SegformerForSemanticSegmentation, "upernet": transformers.UperNetForSemanticSegmentation,
           "mask2former": transformers.Mask2FormerForUniversalSegmentation}[args.family]
    model = cls.from_pretrained(args.model_dir).to(dev).eval()
    mean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(3, 1, 1); std = torch.tensor([58.395, 57.12, 57.375], device=dev).view(3, 1, 1)
    store, labels = [], []; t0 = time.time()
    with torch.inference_mode():
        for f, g in files:
            im = Image.open(f).convert("RGB"); w, h = im.size; nh, nw = input_size(args.dataset, h, w, args.short_side)
            if (nh, nw) != (h, w): im = im.resize((nw, nh), Image.BILINEAR)
            x = ((torch.from_numpy(np.asarray(im).copy()).to(dev).permute(2, 0, 1).float() - mean) / std)[None]
            if args.family == "segformer": store.append(model(pixel_values=x).logits[0].half().cpu())
            elif args.family == "upernet": store.append(model.decode_head(model.backbone.forward_with_filtered_kwargs(x).feature_maps)[0].half().cpu())
            else:
                o = model(pixel_values=x)
                store.append((torch.softmax(o.class_queries_logits[0], -1)[:, :-1].half().cpu(), o.masks_queries_logits[0].half().cpu()))
            labels.append(load_label(args.dataset, g))
    print(f"inference: {len(files)} images, {time.time() - t0:.0f}s", flush=True)

    def probs(i, T=1.0):
        size = labels[i].shape
        if args.family == "mask2former":       # semantic inference of mask-classification models, renormalised over classes
            pc, ml = store[i]
            m = torch.sigmoid(F.interpolate(ml.to(dev).float()[None], size=size, mode="bilinear", align_corners=False)[0])
            s = torch.einsum("qc,qhw->chw", pc.to(dev).float(), m).clamp(min=1e-12)
            if T != 1.0: s = s ** (1.0 / T)
            return s / s.sum(0, keepdim=True)
        return torch.softmax(F.interpolate(store[i].to(dev).float()[None], size=size, mode="bilinear", align_corners=False)[0] / T, 0)

    def run(idx, a, b, T, score=True):
        """One pass. Returns label-free SoftStats and, if score, the labelled confusion + per-image IoUs + changed fraction."""
        soft = SoftStats(C, dev); conf = torch.zeros(C * C, device=dev, dtype=torch.float64); per_img, changed, total = [], 0, 0
        for i in idx:
            p = probs(i, T); pred = decode(p, a, b); soft.add(p, pred)
            if score:
                gt = labels[i].to(dev).long(); v = gt != 255
                c = torch.bincount(gt[v] * C + pred[v], minlength=C * C).double(); conf += c; c = c.view(C, C)
                tp, g_, p_ = c.diag(), c.sum(1), c.sum(0); present = g_ > 0
                if present.any(): per_img.append(float((tp[present] / (g_ + p_ - tp)[present]).mean()))
                changed += int((pred != p.argmax(0)).sum()); total += pred.numel()
        return soft, conf.view(C, C), per_img, changed / max(total, 1)

    def report(conf, per_img, changed):
        tp, g_, p_ = conf.diag(), conf.sum(1), conf.sum(0); iou = tp / (g_ + p_ - tp)
        return dict(mIoU=round(float(torch.nanmean(iou)) * 100, 3), mDice=round(float(torch.nanmean(2 * tp / (g_ + p_))) * 100, 3),
                    mAcc=round(float(torch.nanmean(tp / g_)) * 100, 3), aAcc=round(float(tp.sum() / conf.sum()) * 100, 3),
                    per_image_mIoU=round(float(np.mean(per_img)) * 100, 3), changed_pixel_fraction=round(changed, 5),
                    class_iou=[None if x != x else round(x * 100, 2) for x in iou.tolist()])

    ALL = list(range(len(files))); F0, F1 = ALL[0::2], ALL[1::2]
    one, zero = torch.ones(C, device=dev), torch.zeros(C, device=dev)
    res = dict(args=vars(args), images=len(files), num_classes=C)
    with torch.inference_mode():
        _, conf, per, ch = run(ALL, one, zero, 1.0); res["argmax"] = report(conf, per, ch)
        res["class_pixel_fraction"] = (conf.sum(1) / conf.sum()).tolist()
        print("argmax", {k: v for k, v in res["argmax"].items() if k != "class_iou"}, flush=True)
        best = None                            # temperature by NLL on a 10% subsample of fold 0 (labels used: disclosed)
        for T in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
            nll = m = 0
            for i in F0[::10]:
                p = probs(i, T); gt = labels[i].to(dev).long(); v = gt != 255
                nll += float(-torch.log(p.gather(0, gt.clamp(max=C - 1)[None])[0][v].clamp(min=1e-12)).sum()); m += int(v.sum())
            if best is None or nll / m < best[1]: best = (T, nll / m)
        res["T_star"] = best[0]; print("T*", best, flush=True)
        for metric in args.metrics.split(","):
            for T in sorted({1.0, best[0]}):
                a, b, hist = fit_labelfree(lambda a, b: run(ALL, a, b, T, score=False)[0], C, metric, args.iters, device=dev)
                _, conf, per, ch = run(ALL, a, b, T); key = f"labelfree_{metric}_T={T}"
                res[key] = report(conf, per, ch); res[key]["expected_metric_history"] = [round(100 * h, 3) for h in hist]
                print(key, {k: v for k, v in res[key].items() if k != "class_iou"}, flush=True)
            conf = torch.zeros(C, C, device=dev, dtype=torch.float64); per = []
            for fit, app in ((F0, F1), (F1, F0)):                              # inductive: parameters come from the other half, with labels
                a, b, _ = fit_labeled(lambda a, b: run(fit, a, b, 1.0)[1], C, metric, args.iters, device=dev)
                _, c, p_, _ = run(app, a, b, 1.0); conf += c; per += p_
            key = f"crossfit_labeled_{metric}_T=1.0"; res[key] = report(conf, per, float("nan"))
            print(key, {k: v for k, v in res[key].items() if k != "class_iou"}, flush=True)
            conf = torch.zeros(C, C, device=dev, dtype=torch.float64); per = []
            for fit, app in ((F0, F1), (F1, F0)):                              # inductive and label-free: parameters from the other half's images only
                a, b, _ = fit_labelfree(lambda a, b: run(fit, a, b, 1.0, score=False)[0], C, metric, args.iters, device=dev)
                _, c, p_, _ = run(app, a, b, 1.0); conf += c; per += p_
            key = f"crossfit_labelfree_{metric}_T=1.0"; res[key] = report(conf, per, float("nan"))
            print(key, {k: v for k, v in res[key].items() if k != "class_iou"}, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1); print("wrote", args.out)


if __name__ == "__main__":
    main()
