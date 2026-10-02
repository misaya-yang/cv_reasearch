"""Pilot 11 (control): is region-level what/where fusion more than a plain ensemble with the same frozen encoder?
Train the standard per-patch linear segmentation probe on the same DINOv2 features, then
  (a) score the probe alone, (b) pixel-level product ensemble with the segmenter, (c) region-level fusion (pilot 8/9).
  python pilot11_naive_ensemble_control.py train <encoder_dir> <tag>
  python pilot11_naive_ensemble_control.py eval  <encoder_dir> <tag> <family> <segmenter_dir> <short_side>
"""
import os, sys, glob, json, time, numpy as np, torch, torch.nn.functional as F
from PIL import Image
D = os.environ.get("DEMO2_ROOT", "/root/autodl-tmp/demo2_pilot"); sys.path.append(D + "/env")
import transformers
dev = "cuda:0"; C = 150; R = 518; G = R // 14; mode = sys.argv[1]
ADE = D + "/data/ADEChallengeData2016"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def enc_input(im): return (torch.from_numpy(np.asarray(im.resize((R, R), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 - MEAN) / STD
def load_gt(f):
    lab = torch.from_numpy(np.asarray(Image.open(f.replace("/images/", "/annotations/").replace(".jpg", ".png"))).astype(np.int64))
    return torch.where(lab == 0, torch.full_like(lab, 255), lab - 1)
class DS(torch.utils.data.Dataset):
    def __init__(s, fs): s.fs = fs
    def __len__(s): return len(s.fs)
    def __getitem__(s, i):
        torch.set_num_threads(1); f = s.fs[i]; gt = load_gt(f)
        g = F.interpolate(gt[None, None].float(), size=(G, G), mode="nearest")[0, 0].long()   # patch-grid labels
        return enc_input(Image.open(f).convert("RGB")), g
enc = transformers.AutoModel.from_pretrained(sys.argv[2]).to(dev).eval().half(); tag = sys.argv[3]
def make_head(d): return torch.nn.Sequential(torch.nn.BatchNorm1d(d), torch.nn.Linear(d, C)).to(dev)

if mode == "train":
    fs = sorted(glob.glob(f"{ADE}/images/training/*.jpg")); head = make_head(1024); E = 3
    dl = torch.utils.data.DataLoader(DS(fs), batch_size=32, num_workers=16, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=E * len(dl)); t0 = time.time()
    for ep in range(E):
        for it, (x, g) in enumerate(dl):
            with torch.inference_mode(): h = enc(pixel_values=x.to(dev).half()).last_hidden_state[:, 1:].float()
            logit = head(h.clone().reshape(-1, 1024)); loss = F.cross_entropy(logit, g.to(dev).reshape(-1), ignore_index=255)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if it % 100 == 0: print(ep, it, round(float(loss), 4), f"{time.time()-t0:.0f}s", flush=True)
    torch.save(head.state_dict(), f"/root/demo2_cache/patchhead_{tag}.pt"); print("TRAIN_DONE")
else:
    family, mdir, short = sys.argv[4], sys.argv[5], int(sys.argv[6])
    head = make_head(1024); head.load_state_dict(torch.load(f"/root/demo2_cache/patchhead_{tag}.pt")); head.eval()
    ck = torch.load(f"/root/demo2_cache/regclf_{tag}.pt")
    net = torch.nn.Sequential(torch.nn.Linear(ck["dim"], 2048), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(2048, C)).to(dev); net.load_state_dict(ck["state"]); net.eval()
    mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
    model = {"segformer": transformers.SegformerForSemanticSegmentation, "mask2former": transformers.Mask2FormerForUniversalSegmentation}[family].from_pretrained(mdir).to(dev).eval()
    smean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(3, 1, 1); sstd = torch.tensor([58.395, 57.12, 57.375], device=dev).view(3, 1, 1)
    S = {}
    def score(name, pred, gt):
        v = gt != 255; S.setdefault(name, torch.zeros(C * C, device=dev, dtype=torch.float64)).add_(torch.bincount(gt[v] * C + pred[v], minlength=C * C).double())
    with torch.inference_mode():
        for f in sorted(glob.glob(f"{ADE}/images/validation/*.jpg")):
            im = Image.open(f).convert("RGB"); w, h = im.size; s = min(4 * short / max(h, w), short / min(h, w))
            nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
            x = ((torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BILINEAR)).copy()).to(dev).permute(2, 0, 1).float() - smean) / sstd)[None]
            gt = load_gt(f).to(dev)
            if family == "mask2former":
                o = model(pixel_values=x); m = torch.sigmoid(F.interpolate(o.masks_queries_logits.float(), size=gt.shape, mode="bilinear", align_corners=False)[0])
                p = torch.einsum("qc,qhw->chw", torch.softmax(o.class_queries_logits[0].float(), -1)[:, :-1], m).clamp(min=1e-12); p = p / p.sum(0, keepdim=True)
            else: p = torch.softmax(F.interpolate(model(pixel_values=x).logits.float(), size=gt.shape, mode="bilinear", align_corners=False)[0], 0)
            hid = enc(pixel_values=enc_input(im)[None].to(dev).half()).last_hidden_state[0].float()
            d = torch.softmax(F.interpolate(head(hid[1:]).T.reshape(1, C, G, G), size=gt.shape, mode="bilinear", align_corners=False)[0], 0)   # per-pixel probe posterior
            score("segmenter_argmax", p.argmax(0), gt); score("patch_probe_alone", d.argmax(0), gt)
            for g in (0.25, 0.5, 1.0, 2.0, 4.0): score(f"pixel_ensemble^{g}", (p * d.clamp(min=1e-6) ** g).argmax(0), gt)
            am = p.argmax(0); ids = am.unique(); onehot = (am[None] == ids.view(-1, 1, 1)).float()
            wq = F.adaptive_avg_pool2d(onehot[None], (G, G))[0].flatten(1); area = wq.sum(1).clamp(min=1e-6)
            r = torch.softmax(net((torch.cat([(wq @ hid[1:]) / area[:, None], hid[:1].expand(len(ids), -1)], 1) - mu) / sd), 1)
            rmap = torch.full((C, C), 1.0 / C, device=dev); rmap[ids] = r; rpix = rmap[am].permute(2, 0, 1).clamp(min=1e-6)
            for g in (2.0, 4.0): score(f"region_fusion^{g}", (p * rpix ** g).argmax(0), gt)
            score("region^2+pixel_ensemble^1", (p * rpix ** 2 * d.clamp(min=1e-6)).argmax(0), gt)
            # the probe's own regions, relabelled by the same region recogniser: is region pooling useful even with no segmenter at all?
            dm = d.argmax(0); ids2 = dm.unique(); oh2 = (dm[None] == ids2.view(-1, 1, 1)).float(); w2 = F.adaptive_avg_pool2d(oh2[None], (G, G))[0].flatten(1); a2 = w2.sum(1).clamp(min=1e-6)
            r2 = torch.softmax(net((torch.cat([(w2 @ hid[1:]) / a2[:, None], hid[:1].expand(len(ids2), -1)], 1) - mu) / sd), 1)
            rm2 = torch.full((C, C), 1.0 / C, device=dev); rm2[ids2] = r2
            score("patch_probe+own_region_fusion^2", (d * rm2[dm].permute(2, 0, 1).clamp(min=1e-6) ** 2).argmax(0), gt)
    res = {"segmenter": mdir, "encoder": tag}
    for k, c in S.items():
        c = c.view(C, C); tp, g_, pr = c.diag(), c.sum(1), c.sum(0)
        res[k] = dict(mIoU=round(float(torch.nanmean(tp / (g_ + pr - tp))) * 100, 3), mAcc=round(float(torch.nanmean(tp / g_)) * 100, 3), aAcc=round(float(tp.sum() / c.sum()) * 100, 3))
    print(json.dumps(res, indent=1)); json.dump(res, open(f"{D}/pilot11_{tag}_{os.path.basename(mdir.rstrip('/'))}.json", "w"), indent=1); print("PILOT11_DONE")
