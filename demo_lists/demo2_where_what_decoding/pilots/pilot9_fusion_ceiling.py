"""Pilot 9: how far does region-level what/where fusion go, and does the metric rule stack on top?
Caches segmenter outputs and DINOv2 patch tokens for ADE20K val in RAM, then sweeps decoding variants.
  python pilot9_fusion_ceiling.py <encoder_dir> <tag> <family> <segmenter_dir> <short_side>
"""
import os, sys, glob, json, time, numpy as np, torch, torch.nn.functional as F
from PIL import Image
D = os.environ.get("DEMO2_ROOT", "/root/autodl-tmp/demo2_pilot"); sys.path.append(D + "/env")
sys.path.insert(0, "/root/autodl-tmp/demo2_where_what_decoding")
import transformers
from mdecode import SoftStats, decode, fit_labelfree
dev = "cuda:0"; C = 150; R = 518; G = R // 14
ADE = D + "/data/ADEChallengeData2016"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
enc = transformers.AutoModel.from_pretrained(sys.argv[1]).to(dev).eval().half(); tag, family, mdir, short = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
ck = torch.load(f"/root/demo2_cache/regclf_{tag}.pt")
net = torch.nn.Sequential(torch.nn.Linear(ck["dim"], 2048), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(2048, C)).to(dev); net.load_state_dict(ck["state"]); net.eval()
mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
model = {"segformer": transformers.SegformerForSemanticSegmentation, "mask2former": transformers.Mask2FormerForUniversalSegmentation}[family].from_pretrained(mdir).to(dev).eval()
smean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(3, 1, 1); sstd = torch.tensor([58.395, 57.12, 57.375], device=dev).view(3, 1, 1)
fs = sorted(glob.glob(f"{ADE}/images/validation/*.jpg")); store, toks, labels = [], [], []
with torch.inference_mode():
    for f in fs:
        im = Image.open(f).convert("RGB"); w, h = im.size; s = min(4 * short / max(h, w), short / min(h, w))
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        x = ((torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BILINEAR)).copy()).to(dev).permute(2, 0, 1).float() - smean) / sstd)[None]
        if family == "segformer": store.append(model(pixel_values=x).logits[0].half().cpu())
        else:
            o = model(pixel_values=x); store.append((torch.softmax(o.class_queries_logits[0], -1)[:, :-1].half().cpu(), o.masks_queries_logits[0].half().cpu()))
        e = (torch.from_numpy(np.asarray(im.resize((R, R), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 - MEAN) / STD
        toks.append(enc(pixel_values=e[None].to(dev).half()).last_hidden_state[0].cpu())
        lab = torch.from_numpy(np.asarray(Image.open(f.replace("/images/", "/annotations/").replace(".jpg", ".png"))).astype(np.int64))
        labels.append(torch.where(lab == 0, torch.full_like(lab, 255), lab - 1).to(torch.uint8))
print("cached", len(fs), flush=True)

def seg_post(i):
    size = labels[i].shape
    if family == "mask2former":
        pc, ml = store[i]; m = torch.sigmoid(F.interpolate(ml.to(dev).float()[None], size=size, mode="bilinear", align_corners=False)[0])
        s = torch.einsum("qc,qhw->chw", pc.to(dev).float(), m).clamp(min=1e-12); return s / s.sum(0, keepdim=True)
    return torch.softmax(F.interpolate(store[i].to(dev).float()[None], size=size, mode="bilinear", align_corners=False)[0], 0)
def recog(i, labelmap):
    """Recogniser posterior (K, C) for the class-level regions of an integer label map."""
    h = toks[i].to(dev).float(); ids = labelmap.unique()
    onehot = (labelmap[None] == ids.view(-1, 1, 1)).float(); w = F.adaptive_avg_pool2d(onehot[None], (G, G))[0].flatten(1); area = w.sum(1); ok = area > 0
    feat = (w[ok] @ h[1:]) / area[ok, None]; r = torch.zeros(len(ids), C, device=dev); r[:, :] = 1.0 / C
    r[ok] = torch.softmax(net((torch.cat([feat, h[:1].expand(int(ok.sum()), -1)], 1) - mu) / sd), 1)
    return ids, r
def fused(i, g):
    p = seg_post(i); am = p.argmax(0); ids, r = recog(i, am); rmap = torch.full((C, C), 1.0 / C, device=dev); rmap[ids] = r
    q = p * rmap[am].permute(2, 0, 1).clamp(min=1e-6) ** g; return q / q.sum(0, keepdim=True)
def relabel(i, g, rounds=1):
    p = seg_post(i); lab = p.argmax(0)
    for _ in range(rounds):
        ids, r = recog(i, lab); sreg = torch.stack([p[:, lab == c].mean(1) for c in ids.tolist()])
        new = r.argmax(1) if g == float("inf") else (sreg * r.clamp(min=1e-6) ** g).argmax(1)
        lut = torch.arange(C, device=dev); lut[ids] = new; lab = lut[lab]
    return lab
S = {}
def score(name, pred, gt):
    v = gt != 255; S.setdefault(name, torch.zeros(C * C, device=dev, dtype=torch.float64)).add_(torch.bincount(gt[v] * C + pred[v], minlength=C * C).double())
def miou(c):
    c = c.view(C, C); tp, g_, pr = c.diag(), c.sum(1), c.sum(0)
    return dict(mIoU=round(float(torch.nanmean(tp / (g_ + pr - tp))) * 100, 3), mAcc=round(float(torch.nanmean(tp / g_)) * 100, 3), aAcc=round(float(tp.sum() / c.sum()) * 100, 3))
GF = (2.0, 3.0, 4.0, 6.0)
with torch.inference_mode():
    for i in range(len(fs)):
        gt = labels[i].to(dev).long(); score("argmax", seg_post(i).argmax(0), gt)
        for g in GF: score(f"pixel_fusion^{g}", fused(i, g).argmax(0), gt)
        for g in (2.0, 4.0, 8.0, float("inf")): score(f"region_relabel^{g}", relabel(i, g), gt)
        score("region_relabel^4.0_x2rounds", relabel(i, 4.0, 2), gt)
    res = {"segmenter": mdir, "recogniser": tag, **{k: miou(v) for k, v in S.items()}}
    for k, v in res.items():
        if isinstance(v, dict): print(k, v, flush=True)
    # metric-optimal rule on the segmenter alone and stacked on the fused posterior (label-free, transductive)
    for name, post in (("metric_rule_only", seg_post), ("fusion^4.0+metric_rule", lambda i: fused(i, 4.0))):
        def sweep(a, b):
            st = SoftStats(C, dev)
            for i in range(len(fs)):
                p = post(i); st.add(p, decode(p, a, b))
            return st
        a, b, hist = fit_labelfree(sweep, C, "miou", 3, device=dev); conf = torch.zeros(C * C, device=dev, dtype=torch.float64)
        for i in range(len(fs)):
            gt = labels[i].to(dev).long(); pred = decode(post(i), a, b); v = gt != 255; conf += torch.bincount(gt[v] * C + pred[v], minlength=C * C).double()
        res[name] = miou(conf); res[name]["expected_history"] = [round(100 * h, 2) for h in hist]; print(name, res[name], flush=True)
json.dump(res, open(f"{D}/pilot9_{tag}_{os.path.basename(mdir.rstrip('/'))}.json", "w"), indent=1); print("PILOT9_DONE")
