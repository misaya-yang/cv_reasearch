"""Where/what fusion on a validation set: diagnostics, oracles and every fusion variant for any number of recognisers.

  python scripts/eval_fusion.py --seg mask2former:models/m2f-swin-large-ade:640 --encoder models/dinov2-large:dinov2l \
      --rec ext=dinov2l:regclf_dinov2l.pt --rec own=own:rec/X_own_gt.pt --rec owncf=own:rec/X_own_cf.pt.f0,rec/X_own_cf.pt.f1 \
      --pix lin=dinov2l:pix/dinov2l_mlp_cls.pt --out results/ade20k/fusion_m2f-swin-large.json
A recogniser is `name=source:ckpt`; `ckptA,ckptB` means cross-fitted: A is applied to even-indexed images, B to odd-indexed ones.
A pixel head (`--pix`) is the control: a per-patch classifier on the same encoder, ensembled pixel by pixel.
"""
import argparse, functools, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wwd import *

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="ade20k"); ap.add_argument("--split", default="val"); ap.add_argument("--seg", required=True)
ap.add_argument("--encoder", action="append", default=[]); ap.add_argument("--rec", action="append", default=[]); ap.add_argument("--out", required=True)
ap.add_argument("--pix", action="append", default=[], help="pixel-level control: name=source:patch_head_ckpt")
ap.add_argument("--seg2", action="append", default=[], help="a second segmenter used only as a per-pixel posterior: name=family:dir:short")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--start", type=int, default=0); ap.add_argument("--gammas", default="1,2,4")
ap.add_argument("--views", default="", help="extra views of the SAME segmenter used as second opinions: comma list of `flip` and input scales, e.g. flip,0.75,1.5")
ap.add_argument("--keep", default=r"^(argmax|.*fuse\^2|.*/mean|.*/alone|.*ens\^(0\.5|1)|.*region_vote/fuse\^(0\.5|1)|.*relabel.*|.*region_vote\^1|mc/.*)$",
                help="variants whose per-image statistics are saved next to the JSON (for bootstrap confidence intervals)")
ap.add_argument("--max-pixels", type=float, default=1.5e6,
                help="images with more label pixels are scored on every second row and column (on ADE20K val: one 1600x1600 image); "
                     "a 150-class posterior at that size is 1.4 GB per copy and the GPU is shared")
ap.add_argument("--tta", action="store_true", help="segmenter posterior = mean over the image and its horizontal flip (query-level variants are skipped)")
a = ap.parse_args()
if os.path.exists(a.out): raise SystemExit(f"refusing to overwrite {a.out}")
P = lambda p: p if p.startswith("/") else f"{CACHE}/{p}"
ds = DATASETS[a.dataset](); C = ds.C; fs = ds.files(a.split)[a.start:]
if a.limit: fs = fs[:a.limit]
seg = make_segmenter(a.seg); preps = [seg.prep]; encs = [make_encoder(e) for e in a.encoder]; preps += [e.prep for e, _ in encs]
recs = {}
for spec in a.rec:
    name, rest = spec.split("=", 1); src, cks = rest.split(":", 1); recs[name] = (src, [Recogniser(P(c)) for c in cks.split(",")])
pixs = {}
for spec in a.pix:
    name, rest = spec.split("=", 1); src, ck = rest.split(":", 1); pixs[name] = (src, PatchHead.load(P(ck)))
seg2 = {}
for spec in a.seg2:
    name, rest = spec.split("=", 1); seg2[name] = make_segmenter(rest); preps.append(seg2[name].prep)
FLIP = "flip" in a.views.split(","); SCALES = [float(v) for v in a.views.split(",") if v and v != "flip"]; nview0 = len(preps)
preps += [functools.partial(seg.prep, short=int(round(seg.short * sc / 32)) * 32) for sc in SCALES]
GAM = [float(g) for g in a.gammas.split(",")]; MASK_CLS = seg.family in Segmenter.MASK_CLS and not a.tta
S = Scores(C, keep=a.keep); eye = torch.eye(C, device=DEV); arange = torch.arange(C, device=DEV)
diag = torch.zeros(5, device=DEV, dtype=torch.float64)        # error pixels, of which predicting an absent class, predicted / true / spurious classes
reg = {n: [] for n in recs}                                    # per predicted region: seg ok, rec ok, fused ok, area share, own-class score, class present, valid
t0 = time.time(); nbig = 0


def fuse(p, am, ids, r, g, prior=None):
    """Rule R: every pixel is re-scored by the recogniser posterior of the region it currently belongs to."""
    rr = r if prior is None else r / prior[None]
    rmap = torch.ones(C, C, device=DEV); rmap[ids] = rr.clamp(min=1e-6) ** g
    return rmap.T[:, am].mul_(p)                              # one (C, H, W) temporary: a few validation images are very large


def second_opinions(xs, sources, size):
    """Yields (name, per-pixel posterior, number of views it averages), one at a time: a patch head on an encoder, another segmenter,
    or other views of the same segmenter (and their mean, `v:all`)."""
    npre = 1 + len(encs)
    for name, (src, head) in pixs.items(): yield name, head.posterior(sources[src], size), 1
    for k, (name, m) in enumerate(seg2.items()): yield name, m.posterior(m(xs[npre + k]), size), 1
    acc = None; nv = FLIP + len(SCALES)
    for k in range(nv):
        if FLIP and k == 0: name, d = "flip", seg.posterior(seg(xs[0].flip(-1)), size).flip(-1)
        else: sc = SCALES[k - FLIP]; name, d = f"x{sc:g}", seg.posterior(seg(xs[nview0 + k - FLIP]), size)
        if nv > 1: acc = d.clone() if acc is None else acc.add_(d)
        yield "v:" + name, d, 1
        del d
    if nv > 1: yield "v:all", acc.div_(nv), nv


with torch.inference_mode():
    for n, (xs, gt, i) in enumerate(loader(ds, fs, preps)):
        gt = gt.to(DEV).long(); xs = [x.to(DEV) for x in xs]; out = seg(xs[0])
        if gt.numel() > a.max_pixels: gt = gt[::2, ::2].contiguous(); nbig += 1
        size = tuple(gt.shape); v = gt != 255; gtc = gt.clamp(max=C - 1)
        p = seg.posterior(out, size)
        if a.tta: p = 0.5 * (p + seg.posterior(seg(xs[0].flip(-1)), size).flip(-1))
        am = p.argmax(0); S.add("argmax", am, gt)
        present = torch.bincount(gtc.flatten(), weights=v.flatten().float(), minlength=C) > 0
        ids, masks = class_masks(am)
        # ---- diagnostics and oracles
        wrong = v & (am != gt)
        diag += torch.stack([wrong.sum(), (wrong & ~present[am]).sum(), torch.tensor(len(ids), device=DEV), present.sum(), (~present[ids]).sum()])
        S.add("oracle/presence", (p + 2 * (present.float() - 1).view(-1, 1, 1)).argmax(0), gt)
        hist = joint_hist(am, gt, v, C)[ids]; nv = hist.sum(1); maj = torch.where(nv > 0, hist.argmax(1), ids)
        lut = arange.clone(); lut[ids] = maj; S.add("oracle/region_majority", lut[am], gt)
        if MASK_CLS:                                          # name every query by the majority ground-truth class under its hard mask
            qh = torch.cat([torch.zeros(len(qm), C, device=DEV).index_add_(
                1, gtc.flatten(), (F.interpolate(qm[None], size=size, mode="bilinear", align_corners=False)[0] > 0).flatten(1).float() * v.flatten())
                for qm in out["qmask"].split(25)])
            qc = torch.where(qh.sum(1, keepdim=True) > 0, eye[qh.argmax(1)] * (1 - out["qcls"][:, -1:]), out["qcls"][:, :-1])
            S.add("oracle/query_majority", seg.posterior(out, size, qc).argmax(0), gt)
            # label-free controls for query-level fusion: other ways of reading class scores out of the same (class, mask) pairs
            pq = out["qcls"][:, :-1]; obj = 1 - out["qcls"][:, -1:]
            for g in (2.0, 4.0): S.add(f"mc/sum_pow^{g:g}", seg.posterior(out, size, pq ** g, mpow=g).argmax(0), gt)   # sum_q (m p)^g: towards max over queries
            z = pq.clamp(min=1e-8) ** 3; S.add("mc/self_sharpen^2", seg.posterior(out, size, obj * z / z.sum(1, keepdim=True)).argmax(0), gt)
            for t in (0.3, 0.5): S.add(f"mc/drop_conf<{t:g}", seg.posterior(out, size, pq * (pq.max(1, keepdim=True).values >= t)).argmax(0), gt)
        # the other corner: perfect "where" (ground-truth class regions), "what" by the segmenter's own mean posterior in each region
        gids, gmasks = class_masks(gt); lutg = arange.clone()
        lutg[gids] = (p.flatten(1) @ gmasks.flatten(1).T).argmax(0); S.add("oracle/gt_regions+seg_what", lutg[gtc], gt)
        # ---- sources, the pixel-level control, recognisers
        sources = {"own": own_source(out)}
        for k, (e, etag) in enumerate(encs): sources[etag] = e(xs[1 + k])
        sreg = (p.flatten(1) @ masks.flatten(1).T).T / masks.flatten(1).sum(1, keepdim=True)   # segmenter's mean posterior in each of its regions, (K, C)
        dregs = {}
        for name, d, cnt in second_opinions(xs, sources, size):
            d = d.clamp(min=1e-6); S.add(f"{name}/alone", d.argmax(0), gt)
            S.add(f"{name}/mean", (p + cnt * d).argmax(0), gt)                # arithmetic mean = standard test-time augmentation / ensembling
            for g in (0.25, 0.5, 1.0, 2.0, 4.0): S.add(f"{name}/ens^{g:g}", (p * d ** g).argmax(0), gt)
            # classify-then-pool: the second posterior averaged over each segmenter region, used exactly like a recogniser posterior
            dreg = (d.flatten(1) @ masks.flatten(1).T).T / masks.flatten(1).sum(1, keepdim=True); dregs[name] = dreg; del d
            for g in (0.5, 1.0, 2.0): S.add(f"{name}/region_vote/fuse^{g:g}", fuse(p, am, ids, dreg, g).argmax(0), gt)
            lut = arange.clone(); lut[ids] = (sreg * dreg).argmax(1); S.add(f"{name}/region_vote/relabel", lut[am], gt)
        for name, (src, models) in recs.items():
            R = models[int(i) % len(models)]; feats, ok = multi_region_features(sources, src, masks); r = R(feats, ok)
            for g in GAM: S.add(f"{name}/fuse^{g:g}", fuse(p, am, ids, r, g).argmax(0), gt)
            if R.prior is not None: S.add(f"{name}/fuse^2_prior", fuse(p, am, ids, r, 2.0, R.prior.clamp(min=1e-4)).argmax(0), gt)
            for pn, dreg in dregs.items() if name == next(iter(recs)) else ():   # the first recogniser together with each second opinion
                S.add(f"{name}/fuse^2*{pn}/region_vote^1", fuse(p, am, ids, r.clamp(min=1e-6) ** 2 * dreg, 1.0).argmax(0), gt)
            new = (sreg * r.clamp(min=1e-6) ** 2).argmax(1)   # whole-region decision: segmenter's mean posterior in the region x recogniser
            lut = arange.clone(); lut[ids] = new; S.add(f"{name}/relabel^2", lut[am], gt)
            lut = arange.clone(); lut[ids] = r.argmax(1); S.add(f"{name}/rec_only", lut[am], gt)
            fg, okg = multi_region_features(sources, src, gmasks); lutg = arange.clone(); lutg[gids] = R(fg, okg).argmax(1); S.add(f"{name}/gt_regions", lutg[gtc], gt)
            reg[name].append(torch.stack([(ids == maj).float(), (r.argmax(1) == maj).float(), (new == maj).float(), nv / nv.sum().clamp(min=1),
                                          r[torch.arange(len(ids), device=DEV), ids], present[ids].float(), (nv > 0).float()], 1))
            if MASK_CLS:                                      # fuse at query level first (regions = hard query masks), then at class level
                fq, okq = multi_region_features(sources, src, (out["qmask"] > 0).float()); rq = R(fq, okq)
                z = out["qcls"][:, :-1].clamp(min=1e-8) * rq.clamp(min=1e-6) ** 2; qc = (1 - out["qcls"][:, -1:]) * z / z.sum(1, keepdim=True)
                p2 = seg.posterior(out, size, qc); am2 = p2.argmax(0); S.add(f"{name}/query^2", am2, gt)
                ids2, m2 = class_masks(am2); f2, ok2 = multi_region_features(sources, src, m2)
                S.add(f"{name}/query^2+fuse^2", fuse(p2, am2, ids2, R(f2, ok2), 2.0).argmax(0), gt)
        if gt.numel() > 1e6: del p; torch.cuda.empty_cache()
        if (n + 1) % 500 == 0: print(n + 1, f"{time.time() - t0:.0f}s", {k: m["mIoU"] for k, m in S.result().items()}, flush=True)

dg = diag.tolist(); N = len(fs)
res = dict(segmenter=seg.tag, dataset=a.dataset, images=N, images_scored_at_half_resolution=nbig, tta=a.tta, recognisers=a.rec, pixel_heads=a.pix, second_segmenters=a.seg2, views=a.views,
           protocol=dict(gamma=2.0, hyperparameters_selected_on="none"),
           diagnostics=dict(error_pixels_predicting_absent_class=round(dg[1] / max(dg[0], 1), 4), predicted_classes_per_image=round(dg[2] / N, 3),
                            gt_classes_per_image=round(dg[3] / N, 3), spurious_classes_per_image=round(dg[4] / N, 3)), **S.result())
for name, rows in reg.items():
    t = torch.cat(rows).cpu(); t = t[t[:, 6] > 0]; sg, rc, fu, ar, own, real = (t[:, k] for k in range(6))
    x, y = own[real > 0], own[real == 0]; x = x[torch.randperm(len(x))[:20000]]
    res[f"{name}/regions"] = dict(n=len(sg), seg_acc=round(float(sg.mean()), 4), rec_acc=round(float(rc.mean()), 4), fused_acc=round(float(fu.mean()), 4),
                                  seg_acc_area=round(float((sg * ar).sum() / ar.sum()), 4), rec_acc_area=round(float((rc * ar).sum() / ar.sum()), 4),
                                  fused_acc_area=round(float((fu * ar).sum() / ar.sum()), 4),
                                  rec_fixes_seg_wrong=round(float(rc[sg == 0].mean()), 4), rec_breaks_seg_right=round(float(1 - rc[sg == 1].mean()), 4),
                                  fused_fixes_seg_wrong=round(float(fu[sg == 0].mean()), 4), fused_breaks_seg_right=round(float(1 - fu[sg == 1].mean()), 4),
                                  verifier_auc=round(float((x[:, None] > y[None]).float().mean() + 0.5 * (x[:, None] == y[None]).float().mean()), 4))
res["class_iou_argmax"] = [round(v, 4) for v in S.class_iou("argmax")]
print(json.dumps({k: v for k, v in res.items() if k != "class_iou_argmax"}, indent=1)); save_json(res, a.out); S.save_per_image(a.out[:-5] + ".perimage.pt"); print("EVAL_DONE")
