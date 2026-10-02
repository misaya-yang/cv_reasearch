"""Pilot 7: how much of the oracle presence headroom does a REAL presence estimator recover?
Frozen foundation encoder -> multi-label linear probe trained on ADE20K-train image-level labels
(derived from the same masks the segmenter was trained on; the segmenter itself is untouched).
Step A (this script, mode=feat): cache image-level features for train/val.
Step B (mode=probe): train the probe, report presence quality, write val presence probabilities.
Step C (mode=gate): apply presence gating to a segmenter and score mIoU.
Usage:
  python pilot7_presence_probe.py feat  <encoder_dir> <tag> [res]
  python pilot7_presence_probe.py probe <tag>
  python pilot7_presence_probe.py gate  <tag> <family> <segmenter_dir> <short_side>
"""
import os, sys, glob, json, time, numpy as np, torch, torch.nn.functional as F
from PIL import Image
D = os.environ.get("DEMO2_ROOT", "/root/autodl-tmp/demo2_pilot"); sys.path.append(D + "/env")
import transformers
dev = "cuda:0"; C = 150; mode = sys.argv[1]
ADE = D + "/data/ADEChallengeData2016"
def files(split): return sorted(glob.glob(f"{ADE}/images/{split}/*.jpg"))
def gt_of(f):
    lab = torch.from_numpy(np.asarray(Image.open(f.replace("/images/", "/annotations/").replace(".jpg", ".png"))).astype(np.int64))
    return torch.where(lab == 0, torch.full_like(lab, 255), lab - 1)
def presence_of(gt):
    y = torch.zeros(C); u = gt[gt != 255].unique(); y[u] = 1; return y

if mode == "feat":
    enc_dir, tag = sys.argv[2], sys.argv[3]; res = int(sys.argv[4]) if len(sys.argv) > 4 else 518
    enc = transformers.AutoModel.from_pretrained(enc_dir).to(dev).eval().half()
    is_clip = hasattr(enc, "vision_model")
    mean, std = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]) if is_clip else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1); std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    for split in ("training", "validation"):
        fs = files(split); feats, ys = [], []; t0 = time.time()
        with torch.inference_mode():
            for i in range(0, len(fs), 32):
                batch = fs[i:i + 32]
                x = torch.stack([torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((res, res), Image.BICUBIC)).copy()) for f in batch]).to(dev).permute(0, 3, 1, 2).float() / 255
                x = ((x - mean) / std).half()
                h = (enc.vision_model(pixel_values=x) if is_clip else enc(pixel_values=x)).last_hidden_state.float()   # (B, 1+N, d)
                feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1), h[:, 1:].amax(1)], 1).cpu())                         # cls | mean patch | max patch
                ys += [presence_of(gt_of(f)) for f in batch]
                if i % 3200 == 0: print(split, i, f"{time.time()-t0:.0f}s", flush=True)
        torch.save(dict(x=torch.cat(feats), y=torch.stack(ys), files=fs), f"{D}/feat_{tag}_{split}.pt")
    print("FEAT_DONE")

elif mode == "probe":
    tag = sys.argv[2]
    tr = torch.load(f"{D}/feat_{tag}_training.pt"); va = torch.load(f"{D}/feat_{tag}_validation.pt")
    mu, sd = tr["x"].mean(0, keepdim=True), tr["x"].std(0, keepdim=True) + 1e-6
    Xtr, Ytr, Xva, Yva = ((tr["x"] - mu) / sd).to(dev), tr["y"].to(dev), ((va["x"] - mu) / sd).to(dev), va["y"].to(dev)
    torch.manual_seed(0); n = len(Xtr); perm = torch.randperm(n, device=dev); hold = perm[:2000]; fit = perm[2000:]   # held-out part of TRAIN for model selection
    out = {}
    for name, hidden in (("linear", 0), ("mlp", 2048)):
        best = None
        for wd in (1e-4, 1e-3, 1e-2):
            net = torch.nn.Linear(Xtr.shape[1], C) if hidden == 0 else torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.5), torch.nn.Linear(hidden, C))
            net = net.to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd); E = 40
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=E * (len(fit) // 256 + 1))
            for ep in range(E):
                net.train(); p = fit[torch.randperm(len(fit), device=dev)]
                for i in range(0, len(p), 256):
                    b = p[i:i + 256]; loss = F.binary_cross_entropy_with_logits(net(Xtr[b]), Ytr[b]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            net.eval()
            with torch.no_grad(): hl = float(F.binary_cross_entropy_with_logits(net(Xtr[hold]), Ytr[hold]))
            if best is None or hl < best[0]: best = (hl, wd, net)
        net = best[2]
        with torch.no_grad(): P = torch.sigmoid(net(Xva))
        ap = []                                                                 # per-class average precision on val
        for c in range(C):
            y = Yva[:, c];
            if y.sum() == 0: continue
            o = P[:, c].argsort(descending=True); tp = y[o].cumsum(0); prec = tp / torch.arange(1, len(o) + 1, device=dev)
            ap.append(float((prec * y[o]).sum() / y.sum()))
        st = {}
        for t in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
            keep = P >= t; st[t] = dict(recall_present=round(float((keep & (Yva > 0)).sum() / Yva.sum()), 4), kept_absent_per_image=round(float((keep & (Yva == 0)).sum() / len(Yva)), 3))
        out[name] = dict(weight_decay=best[1], heldout_train_bce=round(best[0], 5), val_mAP=round(float(np.mean(ap)) * 100, 2), thresholds=st)
        torch.save(dict(P=P.cpu(), files=va["files"]), f"{D}/presence_{tag}_{name}.pt")
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(f"{D}/pilot7_probe_{tag}.json", "w"), indent=1)

elif mode == "gate":
    tag, family, mdir, short = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    pres = {k: torch.load(f"{D}/presence_{tag}_{k}.pt") for k in ("linear", "mlp")}
    model = {"segformer": transformers.SegformerForSemanticSegmentation, "upernet": transformers.UperNetForSemanticSegmentation,
             "mask2former": transformers.Mask2FormerForUniversalSegmentation}[family].from_pretrained(mdir).to(dev).eval()
    mean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(3, 1, 1); std = torch.tensor([58.395, 57.12, 57.375], device=dev).view(3, 1, 1)
    def posterior(x, size):
        if family == "mask2former":
            o = model(pixel_values=x)
            m = torch.sigmoid(F.interpolate(o.masks_queries_logits.float(), size=size, mode="bilinear", align_corners=False)[0])
            s = torch.einsum("qc,qhw->chw", torch.softmax(o.class_queries_logits[0].float(), -1)[:, :-1], m).clamp(min=1e-12)
            return s / s.sum(0, keepdim=True)
        l = model(pixel_values=x).logits if family == "segformer" else model.decode_head(model.backbone.forward_with_filtered_kwargs(x).feature_maps)
        return torch.softmax(F.interpolate(l.float(), size=size, mode="bilinear", align_corners=False)[0], 0)
    S = {}
    def score(name, pred, gt):
        v = gt != 255; S.setdefault(name, torch.zeros(C * C, device=dev, dtype=torch.float64)).add_(torch.bincount(gt[v] * C + pred[v], minlength=C * C).double())
    fs = files("validation"); assert fs == pres["linear"]["files"]
    with torch.inference_mode():
        for i, f in enumerate(fs):
            im = Image.open(f).convert("RGB"); w, h = im.size; s = min(4 * short / max(h, w), short / min(h, w))
            nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
            x = ((torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BILINEAR)).copy()).to(dev).permute(2, 0, 1).float() - mean) / std)[None]
            gt = gt_of(f).to(dev); p = posterior(x, gt.shape); score("argmax", p.argmax(0), gt)
            yp = torch.zeros(C, dtype=torch.bool, device=dev); yp[gt[gt != 255].unique()] = True
            q = p.clone(); q[~yp] = -1; score("oracle_presence", q.argmax(0), gt)
            for k, d in pres.items():
                pi = d["P"][i].to(dev)
                for t in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):                     # hard gate: drop classes the probe calls absent
                    keep = pi >= t; keep[pi.argmax()] = True; q = p.clone(); q[~keep] = -1; score(f"{k}/hard>={t}", q.argmax(0), gt)
                for g in (0.25, 0.5, 1.0, 2.0):                                # soft gate: product of experts p * pi^gamma
                    score(f"{k}/soft^{g}", (p * pi.view(C, 1, 1) ** g).argmax(0), gt)
    res = {"segmenter": mdir, "presence": tag}
    for k, c in S.items():
        c = c.view(C, C); tp, g, pr = c.diag(), c.sum(1), c.sum(0)
        res[k] = dict(mIoU=round(float(torch.nanmean(tp / (g + pr - tp))) * 100, 3), mAcc=round(float(torch.nanmean(tp / g)) * 100, 3), aAcc=round(float(tp.sum() / c.sum()) * 100, 3))
    print(json.dumps(res, indent=1)); json.dump(res, open(f"{D}/pilot7_gate_{tag}_{os.path.basename(mdir.rstrip('/'))}.json", "w"), indent=1)
