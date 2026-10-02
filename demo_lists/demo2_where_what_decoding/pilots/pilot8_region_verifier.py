"""Pilot 8: region-level verification with a frozen foundation encoder.
The segmenter says WHERE (its predicted class regions); a frozen DINOv2 with a small classifier on
region-pooled features says WHAT.  Does this recover the hallucinated-class headroom?
  python pilot8_region_verifier.py feat  <encoder_dir> <tag>            # region features from GT masks of ADE train (+ val, for recogniser accuracy)
  python pilot8_region_verifier.py train <tag>
  python pilot8_region_verifier.py gate  <encoder_dir> <tag> <family> <segmenter_dir> <short_side>
"""
import os, sys, glob, json, time, numpy as np, torch, torch.nn.functional as F
from PIL import Image
D = os.environ.get("DEMO2_ROOT", "/root/autodl-tmp/demo2_pilot"); sys.path.append(D + "/env")
import transformers
dev = "cuda:0"; C = 150; R = 518; G = R // 14; mode = sys.argv[1]
ADE = D + "/data/ADEChallengeData2016"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def files(split): return sorted(glob.glob(f"{ADE}/images/{split}/*.jpg"))
def load_gt(f):
    lab = torch.from_numpy(np.asarray(Image.open(f.replace("/images/", "/annotations/").replace(".jpg", ".png"))).astype(np.int64))
    return torch.where(lab == 0, torch.full_like(lab, 255), lab - 1)
def enc_input(im): return (torch.from_numpy(np.asarray(im.resize((R, R), Image.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 - MEAN) / STD

class DS(torch.utils.data.Dataset):
    def __init__(s, fs): s.fs = fs
    def __len__(s): return len(s.fs)
    def __getitem__(s, i):
        torch.set_num_threads(1); f = s.fs[i]
        return enc_input(Image.open(f).convert("RGB")), load_gt(f).to(torch.uint8)

def region_feats(tokens, labelmap):
    """tokens: (G*G, d) patch features; labelmap: (H, W) long with 255 = ignore. Returns class ids and one pooled feature per class region."""
    ids = labelmap[labelmap != 255].unique()
    if len(ids) == 0: return ids, tokens.new_zeros(0, tokens.shape[1]), tokens.new_zeros(0)
    onehot = (labelmap[None] == ids.view(-1, 1, 1)).float()                    # (K, H, W)
    w = F.adaptive_avg_pool2d(onehot[None], (G, G))[0].flatten(1)              # (K, G*G) fractional membership of each patch
    area = w.sum(1); keep = area > 0
    return ids[keep], (w[keep] @ tokens) / area[keep, None], onehot.flatten(1).mean(1)[keep]

if mode == "feat":
    enc = transformers.AutoModel.from_pretrained(sys.argv[2]).to(dev).eval().half(); tag = sys.argv[3]
    for split in ("training", "validation"):
        dl = torch.utils.data.DataLoader(DS(files(split)), batch_size=16, num_workers=16, collate_fn=lambda b: (torch.stack([x for x, _ in b]), [g for _, g in b]))
        X, Y, A, I = [], [], [], []; t0 = time.time(); n = 0
        with torch.inference_mode():
            for x, gts in dl:
                h = enc(pixel_values=x.to(dev).half()).last_hidden_state.float()
                for j, gt in enumerate(gts):
                    ids, f, area = region_feats(h[j, 1:], gt.to(dev).long())
                    X.append(torch.cat([f, h[j, :1].expand(len(ids), -1)], 1).half().cpu()); Y.append(ids.cpu()); A.append(area.cpu()); I += [n] * len(ids); n += 1
                if n % 3200 == 0: print(split, n, f"{time.time()-t0:.0f}s", flush=True)
        torch.save(dict(x=torch.cat(X), y=torch.cat(Y), area=torch.cat(A), img=torch.tensor(I)), f"/root/demo2_cache/regfeat_{tag}_{split}.pt")
    print("FEAT_DONE")

elif mode == "train":
    tag = sys.argv[2]
    tr = torch.load(f"/root/demo2_cache/regfeat_{tag}_training.pt"); va = torch.load(f"/root/demo2_cache/regfeat_{tag}_validation.pt")
    mu, sd = tr["x"].float().mean(0, keepdim=True), tr["x"].float().std(0, keepdim=True) + 1e-6
    Xtr, Ytr, Xva, Yva = ((tr["x"].float() - mu) / sd).to(dev), tr["y"].to(dev), ((va["x"].float() - mu) / sd).to(dev), va["y"].to(dev)
    torch.manual_seed(0); best = None
    for wd in (1e-3, 1e-2, 5e-2):
        net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 2048), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(2048, C)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd); E = 30; steps = E * (len(Xtr) // 512 + 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        hold = torch.arange(len(Xtr), device=dev) % 10 == 0                     # 10% of train regions held out for model selection
        fit = (~hold).nonzero()[:, 0]
        for ep in range(E):
            net.train(); p = fit[torch.randperm(len(fit), device=dev)]
            for i in range(0, len(p), 512):
                b = p[i:i + 512]; loss = F.cross_entropy(net(Xtr[b]), Ytr[b], label_smoothing=0.1); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad(): acc = float((net(Xtr[hold]).argmax(1) == Ytr[hold]).float().mean())
        print("wd", wd, "held-out train region acc", round(acc, 4), flush=True)
        if best is None or acc > best[0]: best = (acc, wd, net)
    net = best[2]
    with torch.no_grad(): pv = net(Xva).argmax(1)
    res = dict(weight_decay=best[1], heldout_train_region_acc=round(best[0], 4), val_gt_region_acc=round(float((pv == Yva).float().mean()), 4),
               val_gt_region_acc_area_weighted=round(float(((pv == Yva).float() * va["area"].to(dev)).sum() / va["area"].sum()), 4),
               train_regions=len(Xtr), val_regions=len(Xva))
    print(json.dumps(res)); json.dump(res, open(f"{D}/pilot8_train_{tag}.json", "w"), indent=1)
    torch.save(dict(state=net.state_dict(), mu=mu, sd=sd, dim=Xtr.shape[1]), f"/root/demo2_cache/regclf_{tag}.pt")

elif mode == "gate":
    enc = transformers.AutoModel.from_pretrained(sys.argv[2]).to(dev).eval().half(); tag, family, mdir, short = sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
    ck = torch.load(f"/root/demo2_cache/regclf_{tag}.pt")
    net = torch.nn.Sequential(torch.nn.Linear(ck["dim"], 2048), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(2048, C)).to(dev); net.load_state_dict(ck["state"]); net.eval()
    mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
    model = {"segformer": transformers.SegformerForSemanticSegmentation, "upernet": transformers.UperNetForSemanticSegmentation,
             "mask2former": transformers.Mask2FormerForUniversalSegmentation}[family].from_pretrained(mdir).to(dev).eval()
    smean = torch.tensor([123.675, 116.28, 103.53], device=dev).view(3, 1, 1); sstd = torch.tensor([58.395, 57.12, 57.375], device=dev).view(3, 1, 1)
    def posterior(x, size):
        if family == "mask2former":
            o = model(pixel_values=x)
            m = torch.sigmoid(F.interpolate(o.masks_queries_logits.float(), size=size, mode="bilinear", align_corners=False)[0])
            s = torch.einsum("qc,qhw->chw", torch.softmax(o.class_queries_logits[0].float(), -1)[:, :-1], m).clamp(min=1e-12)
            return s / s.sum(0, keepdim=True)
        l = model(pixel_values=x).logits if family == "segformer" else model.decode_head(model.backbone.forward_with_filtered_kwargs(x).feature_maps)
        return torch.softmax(F.interpolate(l.float(), size=size, mode="bilinear", align_corners=False)[0], 0)
    S = {}; ver = dict(real=[], spurious=[])
    def score(name, pred, gt):
        v = gt != 255; S.setdefault(name, torch.zeros(C * C, device=dev, dtype=torch.float64)).add_(torch.bincount(gt[v] * C + pred[v], minlength=C * C).double())
    with torch.inference_mode():
        for f in files("validation"):
            im = Image.open(f).convert("RGB"); w, h = im.size; s = min(4 * short / max(h, w), short / min(h, w))
            nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
            x = ((torch.from_numpy(np.asarray(im.resize((nw, nh), Image.BILINEAR)).copy()).to(dev).permute(2, 0, 1).float() - smean) / sstd)[None]
            gt = load_gt(f).to(dev); p = posterior(x, gt.shape); am = p.argmax(0); score("argmax", am, gt)
            present = torch.zeros(C, dtype=torch.bool, device=dev); present[gt[gt != 255].unique()] = True
            q = p.clone(); q[~present] = -1; score("oracle_presence", q.argmax(0), gt)
            hid = enc(pixel_values=enc_input(im)[None].to(dev).half()).last_hidden_state.float()[0]
            ids, feat, _ = region_feats(hid[1:], am)                             # one region per PREDICTED class
            r = torch.softmax(net((torch.cat([feat, hid[:1].expand(len(ids), -1)], 1) - mu) / sd), 1)   # (K, C) recogniser posterior per region
            own = r[torch.arange(len(ids)), ids]                                # recogniser's belief that the region is the class the segmenter gave it
            for k, c in enumerate(ids.tolist()): ver["real" if present[c] else "spurious"].append(float(own[k]))
            rmap = torch.zeros(C, C, device=dev); rmap[ids] = r                 # row c = recogniser posterior of the region the segmenter labelled c
            rpix = rmap[am].permute(2, 0, 1)                                    # (C, H, W): recogniser posterior of the region each pixel belongs to
            for g in (0.25, 0.5, 1.0, 2.0): score(f"pixel_fusion^{g}", (p * rpix.clamp(min=1e-6) ** g).argmax(0), gt)
            for t in (0.02, 0.05, 0.1, 0.2, 0.3):                               # veto: remove a predicted class whose region the recogniser rejects
                keep = torch.ones(C, dtype=torch.bool, device=dev); keep[ids[own < t]] = False
                if not keep[ids].any(): keep[ids[own.argmax()]] = True
                q = p.clone(); q[~keep] = -1; score(f"veto<{t}", q.argmax(0), gt)
            sreg = torch.stack([p[:, am == c].mean(1) for c in ids.tolist()])   # (K, C) segmenter's own mean posterior inside each region
            for g in (0.5, 1.0, 2.0):                                           # relabel each region as a whole
                new = (sreg * r.clamp(min=1e-6) ** g).argmax(1); lut = torch.arange(C, device=dev); lut[ids] = new; score(f"region_relabel^{g}", lut[am], gt)
    res = {"segmenter": mdir, "recogniser": tag}
    for k, c in S.items():
        c = c.view(C, C); tp, g_, pr = c.diag(), c.sum(1), c.sum(0)
        res[k] = dict(mIoU=round(float(torch.nanmean(tp / (g_ + pr - tp))) * 100, 3), mAcc=round(float(torch.nanmean(tp / g_)) * 100, 3), aAcc=round(float(tp.sum() / c.sum()) * 100, 3))
    a, b = torch.tensor(ver["real"]), torch.tensor(ver["spurious"])
    res["verifier"] = dict(real_regions=len(a), spurious_regions=len(b), mean_score_real=round(float(a.mean()), 4), mean_score_spurious=round(float(b.mean()), 4),
                           auc=round(float((a[:, None] > b[None, :]).float().mean() + 0.5 * (a[:, None] == b[None, :]).float().mean()), 4),
                           **{f"at_thr_{t}": dict(real_rejected=round(float((a < t).float().mean()), 4), spurious_rejected=round(float((b < t).float().mean()), 4)) for t in (0.02, 0.05, 0.1, 0.2, 0.3)})
    print(json.dumps(res, indent=1)); json.dump(res, open(f"{D}/pilot8_gate_{tag}_{os.path.basename(mdir.rstrip('/'))}.json", "w"), indent=1)
