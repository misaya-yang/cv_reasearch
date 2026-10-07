"""Look at where the false area sits on a packed transfer set. Prediction = stored 64x64 RCG field cut at 0.5.
usage: python look.py <name> <pack_dir,...> <rcg2_dir,...> <out.png>"""
import sys, json, numpy as np
from PIL import Image
from scipy import ndimage
name, packs, outs, dst = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(','), sys.argv[4]
E = []
for p, o in zip(packs, outs):
    z = np.load(o + '/counts.npz'); rows = json.load(open(o + '/rows.json'))
    for i, r in enumerate(rows): E.append((p, r, z['fields'][i].astype(np.float32), z['iu:native'][i], z['iu:rcg'][i], z['truth'][i]))
def mask(path, n): return np.asarray(Image.open(path).convert('L').resize((n, n), Image.BILINEAR), np.float32) / 255. if True else None
S = []
for p, r, f, nat, rcg, T in E:
    t = np.asarray(Image.open(p + '/ann/' + r['query']).convert('L'), np.float32); t = (t > 0).astype(np.float32)
    t64 = np.asarray(Image.fromarray((t * 255).astype(np.uint8)).resize((64, 64), Image.BOX), np.float32) / 255 > .5
    m = f.reshape(64, 64) > .5; fp = m & ~t64; lb, n = ndimage.label(m); touch = np.zeros_like(m)
    if n:
        ids = np.unique(lb[t64 & m]); touch = np.isin(lb, ids[ids > 0])
    # bounding box of the predicted component(s) touching the truth vs the truth box
    S.append(dict(I=(m & t64).sum(), U=(m | t64).sum(), T=t64.sum(), P=m.sum(), fp=fp.sum(), fp_att=(fp & touch).sum(), fpn=int(nat[1] - T), ncomp=n))
A = {k: np.array([s[k] for s in S], float) for k in S[0]}
print(f"{name}: n={len(E)}  pooled IoU of the 64x64 cut {A['I'].sum()/A['U'].sum()*100:.1f}  predicted/true area {A['P'].sum()/A['T'].sum():.2f}")
print(f"  false area attached to a predicted component that touches the truth: {A['fp_att'].sum()/A['fp'].sum()*100:.0f}%")
r = A['P'] / np.maximum(A['T'], 1); print("  predicted/true area per episode, quartiles:", np.round(np.percentile(r, [25, 50, 75, 90]), 2), " share with ratio>2:", round((r > 2).mean(), 2), " ratio>4:", round((r > 4).mean(), 2))
order = np.argsort(-A['fp']); top = order[:12]; print(f"  12 episodes with the largest false area hold {A['fp'][top].sum()/A['fp'].sum()*100:.0f}% of all false area")
W = 200; sheet = Image.new('RGB', (W * 6, W * 6), 'white')
def tint(img, m, c, a=.55):
    x = np.asarray(img, np.float32); x[m] = x[m] * (1 - a) + np.array(c, np.float32) * a; return Image.fromarray(x.astype(np.uint8))
for k, i in enumerate(top):
    p, r, f, *_ = E[i]
    R = Image.open(p + '/data/' + r['support']).convert('RGB').resize((W, W)); Q = Image.open(p + '/data/' + r['query']).convert('RGB').resize((W, W))
    rm = np.asarray(Image.open(p + '/ann/' + r['support']).convert('L').resize((W, W), Image.NEAREST)) > 0
    tm = np.asarray(Image.open(p + '/ann/' + r['query']).convert('L').resize((W, W), Image.NEAREST)) > 0
    pm = np.asarray(Image.fromarray(((f.reshape(64, 64) > .5) * 255).astype(np.uint8)).resize((W, W), Image.NEAREST)) > 0
    a = tint(R, rm, (255, 230, 0)); b = tint(Q, tm, (0, 255, 0)); c = tint(tint(tint(Q, pm & tm, (0, 255, 0)), pm & ~tm, (255, 0, 0)), ~pm & tm, (0, 80, 255))
    x0, y0 = (k % 2) * 3 * W, (k // 2) * W
    for j, im in enumerate((a, b, c)): sheet.paste(im, (x0 + j * W, y0))
sheet.save(dst); print('saved', dst)
