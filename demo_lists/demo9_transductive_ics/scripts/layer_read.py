"""Reads scripts/layer_probe.py: the reference's one-vector reading on each layer / encoding, against the same reading on FoRIS's
own tokens (view l24). Fixed cut, best cut (label), all-token AUC. Verdict per view: BETTER when its best cut exceeds l24's by
>= 2.0 with the paired interval above 0.   BANK=/path/to/layers_dev python layer_read.py"""
from cl_common import *
BANK = Path(os.environ["BANK"]); LEVELS = [i / 20 for i in range(1, 20)]; R = {}; used = []
for r in rows:
    key = "%d_%d_%d" % (r["fold"], r["e"], r["c"])
    if not (BANK / (key + ".npz")).exists(): continue
    bk = np.load(BANK / (key + ".npz")); used.append(r)
    truth = torch.from_numpy(np.unpackbits(np.load(packets / (key + ".npz"))["truth"])[:1024 * 1024].reshape(1024, 1024).astype(bool))
    t64 = (F.adaptive_avg_pool2d(truth[None, None].float(), 64).flatten() >= .5).numpy()
    for k in bk.files:
        if k == "debiased": continue
        v = torch.from_numpy(bk[k].astype(np.float32))
        if v.shape != (64, 64): v = F.interpolate(v[None, None], size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
        d = R.setdefault(k, dict(cut=[], best=[], auc=[])); d["cut"].append(iu(binarise(v, (1024, 1024)), truth))
        u = F.interpolate(normalise(v)[None, None], size=(1024, 1024), mode="bilinear", align_corners=False)[0, 0]
        d["best"].append(max((iu(u > l, truth) for l in LEVELS), key=lambda z: z[0] / max(z[1], 1))); d["auc"].append(auc(v.flatten().numpy(), t64))
cls, groups = np.array([r["c"] for r in used]), photo_groups(used); out = dict(state="COMPLETED", episodes=len(used), views={})
print("episodes %d; one reference vector on each view (DEV241, model resolution)" % len(used))
for kind in ("one", "bg"):
    ref = np.array(R["l24_" + kind]["best"], float)
    for k, d in R.items():
        if not k.endswith("_" + kind): continue
        b = np.array(d["best"], float); p = paired(b, ref, cls, groups, 1000)
        verdict = "BETTER" if p["gain"] >= 2.0 and p["ci95"][0] > 0 else "no"
        out["views"][k] = dict(cut=class_miou(np.array(d["cut"], float), cls), best_cut=p["miou"], best_cut_vs_l24=p["gain"], ci95=p["ci95"], auc=float(np.nanmean(d["auc"])), verdict=verdict)
        print("%-12s cut %6.2f | best cut %6.2f, vs FoRIS's tokens %+6.2f [%+5.2f,%+5.2f] | AUC %.3f | %s" % (k, out["views"][k]["cut"], p["miou"], p["gain"], p["ci95"][0], p["ci95"][1], out["views"][k]["auc"], verdict))
(Path(os.environ.get("OUT", str(L / "layer_read.json")))).write_text(json.dumps(out, indent=1))
