"""Control for H1: a per-patch segmentation head on the SAME frozen encoder tokens the region recogniser uses.

  python scripts/train_patch_head.py --encoder models/dinov2-large:dinov2l --kind mlp --cls --epochs 10 --out pix/dinov2l_mlp_cls.pt
`--kind linear` is the standard linear probe (BatchNorm + Linear); `--kind mlp` has the same capacity as the region recogniser.
`--cls` concatenates the CLS token to every patch token, so the head sees exactly the information the recogniser sees.
The head is trained on logits upsampled x4 against the label map, with horizontal flips.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wwd import *

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="ade20k"); ap.add_argument("--encoder", required=True); ap.add_argument("--kind", default="linear")
ap.add_argument("--cls", action="store_true"); ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
out = a.out if a.out.startswith("/") else f"{CACHE}/{a.out}"
if os.path.exists(out): raise SystemExit(f"refusing to overwrite {out}")
ds = DATASETS[a.dataset](); C = ds.C; fs = ds.files("train")
if a.limit: fs = fs[:a.limit]
enc, etag = make_encoder(a.encoder); g = enc.res // enc.patch; U = 4 * g


class Train(torch.utils.data.Dataset):
    def __len__(s): return len(fs)
    def __getitem__(s, i):
        torch.set_num_threads(1); f = fs[i]; x = enc.prep(Image.open(f).convert("RGB"))
        y = F.interpolate(ds.load_gt(f)[None, None].float(), size=(U, U), mode="nearest")[0, 0].long()
        if torch.rand(()) < 0.5: x, y = x.flip(-1), y.flip(-1)
        return x, y


dl = torch.utils.data.DataLoader(Train(), batch_size=a.bs, num_workers=12, shuffle=True, drop_last=True)
d = enc.model.config.hidden_size; head = PatchHead(d, C, a.kind, a.cls).to(DEV)
opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl))
t0 = time.time()
for ep in range(a.epochs):
    for it, (x, y) in enumerate(dl):
        with torch.inference_mode(): h = enc.model(pixel_values=x.to(DEV).half()).last_hidden_state.float()
        logit = F.interpolate(head(h.clone(), g), size=(U, U), mode="bilinear", align_corners=False)
        loss = F.cross_entropy(logit, y.to(DEV), ignore_index=255); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 200 == 0: print(ep, it, round(float(loss), 4), f"{time.time() - t0:.0f}s", flush=True)
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(dict(state=head.state_dict(), d=d, C=C, kind=a.kind, cls=a.cls, args=vars(a)), out); print("TRAIN_DONE", out)
