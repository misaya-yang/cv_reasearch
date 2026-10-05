"""Each sealed arm read as two edit operators on native: its deletions alone and its additions alone (DEV241, truth read).
Break-even on pooled pixels: deleting helps when the false share of the deleted pixels exceeds 1/(1+IoU); adding helps
when the true share of the added pixels exceeds IoU/(1+IoU)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/Users/yang/projects/CVPR2027/src')
from ics.experiment import unpack, summarize
W = Path(__file__).parent; T = np.load(W / 'truth241.npz'); rows = json.loads((W / 'gpu_multilayer_dev241_v1/manifest.json').read_text())
ARMS = {'D (l24 + transition)': 'multilayer', 'concat l24,l16': 'multilayer.control', 'transition only': 'multilayer.delta.control', 'RCG': 'rcg'}
arr, cor, I, U = {}, {}, 0, 0
for r in rows:
    k = r['key']; t = unpack(T['truth/' + k])
    with np.load(W / 'gpu_multilayer_dev241_v1/predictions' / (k + '.npz')) as f: m = {a: unpack(f[a]) for a in f.files}
    with np.load(W / 'rcg241/predictions' / (k + '.npz')) as f: m['rcg'] = unpack(f['rcg'])
    nat = m['native']; I += int((nat & t).sum()); U += int((nat | t).sum()); out = {'native': nat}
    for name, a in ARMS.items(): out[name + ' | deletions only'] = nat & m[a]; out[name + ' | additions only'] = nat | m[a]; out[name + ' | both'] = m[a]
    for a, x in out.items():
        arr.setdefault(a, []).append([int((x & t).sum()), int((x | t).sum())]); add, de = x & ~nat, nat & ~x
        cor.setdefault(a, []).append(dict(key=k, c=r['c'], fold=r['fold'], batch=str(r.get('batch')), add_TP=int((add & t).sum()), delete_FP=int((de & ~t).sum()), delete_TP=int((de & t).sum()), add_FP=int((add & ~t).sum())))
rep, _ = summarize(rows, {a: np.asarray(v, np.int64) for a, v in arr.items()}, cor); iou = I / U
print('native pooled pixel IoU %.3f: deleting helps above a false share of %.1f%%, adding helps above a true share of %.1f%%' % (iou, 100 / (1 + iou), 100 * iou / (1 + iou)))
res = {}
for a in arr:
    if a == 'native': continue
    c, k = rep['contrasts'][a]['native'], rep['corrections_vs_native'][a]; d, ad = k['delete_FP'] + k['delete_TP'], k['add_TP'] + k['add_FP']
    res[a] = dict(gain=c['gain'], ci95=c['ci95'], up=c['up'], down=c['down'], deleted=d, false_share=k['delete_FP'] / max(d, 1), added=ad, true_share=k['add_TP'] / max(ad, 1))
    print('%-40s %+6.2f [%+6.2f, %+6.2f] %3d/%3d | deleted %8d px, %5.1f%% false | added %8d px, %5.1f%% true' % (a, c['gain'], *c['ci95'], c['up'], c['down'], d, 100 * res[a]['false_share'], ad, 100 * res[a]['true_share']))
(W / 'edit_purity.json').write_text(json.dumps(dict(native_pixel_iou=iou, arms=res), indent=1))
