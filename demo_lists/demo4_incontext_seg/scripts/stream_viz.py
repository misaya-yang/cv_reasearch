"""Contact sheet for one class of a cached stream set: reference with its mask, then unlabeled images with the true mask
(green) and INSID3's one-shot mask (red outline fill)."""
import argparse, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icx.fast import ClassSet, DEV
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="/root/demo4_cache/stream/f0_s0_N16"); ap.add_argument("--c", type=int, default=36); ap.add_argument("--n", type=int, default=8); ap.add_argument("--out", default="/root/demo4_cache/stream/viz.jpg")
a = ap.parse_args(); U = torch.load(f"{a.dir}/U.pt").to(DEV); cs = ClassSet(f"{a.dir}/class{a.c}.pt", U); W = 224
def tile(name, masks):
    im = np.asarray(Image.open(f"/root/demo4_cache/data/COCO2014/{name}").convert("RGB").resize((W, W))).copy().astype(np.float32)
    for m, col in masks:
        mm = np.asarray(Image.fromarray(m.reshape(cs.h, cs.h).cpu().numpy().astype(np.uint8) * 255).resize((W, W), Image.NEAREST)) > 127
        im[mm] = 0.45 * im[mm] + 0.55 * np.array(col)
    return im.astype(np.uint8)
g0 = cs.gt64[0]; top = [tile(cs.names[0], [(g0, (0, 255, 0))])]; bot = [tile(cs.names[0], [])]
for j in range(1, a.n + 1):
    p = cs.predict(j, [0], [g0]); top.append(tile(cs.names[j], [(cs.gt64[j], (0, 255, 0))])); bot.append(tile(cs.names[j], [(p, (255, 0, 0))]))
Image.fromarray(np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)).save(a.out, quality=88); print("saved", a.out)
