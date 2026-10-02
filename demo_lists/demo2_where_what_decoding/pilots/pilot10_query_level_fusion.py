"""Pilot 10: for mask-classification segmenters the purest regions are the query masks themselves.
Fuse the frozen recogniser at QUERY level (before semantic inference) instead of on class-level regions.
  python pilot10_query_level_fusion.py <encoder_dir> <tag> <mask2former_dir> <short_side>
"""
import os, sys, glob, json, numpy as np, torch, torch.nn.functional as F
from PIL import Image
D = os.environ.get("DEMO2_ROOT", "/root/autodl-tmp/demo2_pilot"); sys.path.append(D + "/env")
import transformers
dev = "cuda:0"; C = 150; R = 518; G = R // 14
ADE = D + "/data/ADEChallengeData2016"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
enc = transformers.AutoModel.from_pretrained(sys.argv[1]).to(dev).eval().half(); tag, mdir, short = sys.argv[2], sys.argv[3], int(sys.argv[4])
ck = torch.load(f"/root/demo2_cache/regclf_{tag}.pt")
net = torch.nn.Sequential(torch.nn.Linear(ck["dim"], 2048), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(2048, C)).to(dev); net.load_state_dict(ck["state"]); net.eval()
mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(mdir).to(dev).eval()
smean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(3, 1, 1); sstd = torch.tensor([58.395, 57.12, 57.375], device=dev).view(3, 1, 1)
S = {}
def score(name, pred, gt):
    v = gt != 255; S.setdefault(name, torch.zeros(C * C, device=dev, dtype=torch.float64)).add_(torch.bincount(gt[v] * C + pred[v], minlength=C * C).double())
def class_region_fusion(p, hid, g):
    am = p.argmax(0); ids = am.unique(); onehot = (am[None] == ids.view(-1, 1, 1)).float()
    w = F.adaptive_avg_pool2d(onehot[None], (G, G))[0].flatten(1); area = w.sum(1).clamp(min=1e-6)
    r = torch.softmax(net((torch.cat([(w @ hid[1:]) / area[:, None], hid[:1].expand(len(ids), -1)], 1) - mu) / sd), 1)
    rmap = torch.full((C, C), 1.0 / C, device=dev); rmap[ids] = r
    return (p * rmap[am].permute(2, 0, 1).clamp(min=1e-6) ** g).argmax(0)
with torch.inference_mode():
    for f in sorted(glob.glob(f"{ADE}/images/validation/*.jpg")):
        im = Image.open(f).convert("RGB"); w, h = im.size; s = min(4 * short / max(h, w), short / min(h, w))
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        x = ((torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BILINEAR)).copy()).to(dev).permute(2, 0, 1).float() - smean) / sstd)[None]
        lab = torch.from_numpy(np.asarray(Image.open(f.replace("/images/", "/annotations/").replace(".jpg", ".png"))).astype(np.int64)).to(dev)
        gt = torch.where(lab == 0, torch.full_like(lab, 255), lab - 1)
        o = model(pixel_values=x); pq = torch.softmax(o.class_queries_logits[0].float(), -1); ml = o.masks_queries_logits[0].float()   # (Q, C+1), (Q, h4, w4)
        m = torch.sigmoid(F.interpolate(ml[None], size=gt.shape, mode="bilinear", align_corners=False)[0])                             # (Q, H, W)
        e = (torch.from_numpy(np.asarray(im.resize((R, R), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 - MEAN) / STD
        hid = enc(pixel_values=e[None].to(dev).half()).last_hidden_state[0].float()
        def semseg(pc):
            sq = torch.einsum("qc,qhw->chw", pc, m).clamp(min=1e-12); return sq / sq.sum(0, keepdim=True)
        base = semseg(pq[:, :-1]); score("argmax", base.argmax(0), gt)
        for mode in ("soft", "hard"):                                          # how the query mask weights the patch tokens
            wq = F.adaptive_avg_pool2d((torch.sigmoid(ml) if mode == "soft" else (ml > 0).float())[None], (G, G))[0].flatten(1)       # (Q, G*G)
            area = wq.sum(1); ok = area > 1e-3
            r = torch.full((len(pq), C), 1.0 / C, device=dev)
            r[ok] = torch.softmax(net((torch.cat([(wq[ok] @ hid[1:]) / area[ok, None], hid[:1].expand(int(ok.sum()), -1)], 1) - mu) / sd), 1)
            obj = 1 - pq[:, -1:]                                               # keep each query's "is an object" mass, re-split it over classes
            for g in (0.5, 1.0, 2.0, 4.0):
                z = pq[:, :-1].clamp(min=1e-8) * r.clamp(min=1e-6) ** g; pc = obj * z / z.sum(1, keepdim=True)
                fused = semseg(pc); score(f"query_{mode}^{g}", fused.argmax(0), gt)
                if mode == "hard" and g in (1.0, 2.0):                          # then the class-level region pass on top
                    score(f"query_hard^{g}+class_region^2", class_region_fusion(fused, hid, 2.0), gt)
        score("class_region^2", class_region_fusion(base, hid, 2.0), gt)
res = {"segmenter": mdir, "recogniser": tag}
for k, c in S.items():
    c = c.view(C, C); tp, g_, pr = c.diag(), c.sum(1), c.sum(0)
    res[k] = dict(mIoU=round(float(torch.nanmean(tp / (g_ + pr - tp))) * 100, 3), mAcc=round(float(torch.nanmean(tp / g_)) * 100, 3), aAcc=round(float(tp.sum() / c.sum()) * 100, 3))
print(json.dumps(res, indent=1)); json.dump(res, open(f"{D}/pilot10_{tag}_{os.path.basename(mdir.rstrip('/'))}.json", "w"), indent=1); print("PILOT10_DONE")
