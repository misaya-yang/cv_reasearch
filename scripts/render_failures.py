"""Contact sheets of the episodes where the RCG field fails, from the stored token record: no model, no GPU.

Per episode: reference with its mask (green), query with the truth (green), query with the RCG mask at 0.5 (red).
Images are stretched to a square as the pipeline does. Episodes are ordered by best-cut IoU of the RCG field.

    python render_failures.py outputs/claude_order_fresh600 <episodes.json> 0.5
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

D, MAN, LIM = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), float(sys.argv[3])
NAMES = "person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier toothbrush".split()
T = np.load(D / "tokens.npz"); rows = json.loads((D / "rows.json").read_text()); root = Path(MAN["data_root"]); S = 256
t, z, ref = T["truth"].astype(float), T["rcg"].astype(float), T["ref"].astype(float)
levels = np.linspace(.1, .9, 33); iou = lambda m, y: (m * y).sum() / max(y.sum() + m.sum() - (m * y).sum(), 1e-9)
best = np.array([max(iou(z[i] > l, t[i]) for l in levels) for i in range(len(rows))]); half = np.array([iou(z[i] > .5, t[i]) for i in range(len(rows))])
order = [i for i in np.argsort(best) if best[i] < LIM]


def panel(path, m, colour):
    im = np.asarray(Image.open(root / path).convert("RGB").resize((S, S), Image.BILINEAR)).astype(float)
    a = np.asarray(Image.fromarray((m.reshape(64, 64) * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)).astype(float)[..., None] / 255
    return Image.fromarray((im * (1 - .5 * a) + np.array(colour) * .5 * a).astype(np.uint8))


info = []
for s in range(0, len(order), 12):
    part = order[s:s + 12]; sheet = Image.new("RGB", (3 * S, len(part) * (S + 14)), "white"); d = ImageDraw.Draw(sheet)
    for j, i in enumerate(part):
        r = rows[i]; y = j * (S + 14); m = z[i] > .5
        for k, im in enumerate([panel(r["support"], ref[i], (0, 255, 0)), panel(r["query"], t[i], (0, 255, 0)), panel(r["query"], m.astype(float), (255, 0, 0))]):
            sheet.paste(im, (k * S, y + 14))
        d.text((2, y + 1), "#%d %s %s  IoU@0.5 %.2f  best %.2f  object share %.3f  mask/object %.1f  ref share %.3f" % (s + j, r["key"], NAMES[r["c"]], half[i], best[i], t[i].mean(), m.sum() / max(t[i].sum(), 1e-9), ref[i].mean()), fill="black")
        info.append(dict(n=s + j, key=r["key"], name=NAMES[r["c"]], half=round(float(half[i]), 3), best=round(float(best[i]), 3), share=round(float(t[i].mean()), 4), ratio=round(float(m.sum() / max(t[i].sum(), 1e-9)), 2), ref=round(float(ref[i].mean()), 4)))
    sheet.save(D / ("fail_%02d.jpg" % (s // 12)), quality=78)
(D / "fail.json").write_text(json.dumps(info)); print(len(order), "episodes,", (len(order) + 11) // 12, "sheets")
