"""W50: is transfer failure caused by the reference/query object-scale mismatch?  The same frozen SH with the
reference re-encoded unchanged (id) and rescaled so its object area fraction matches the query GT fraction (match,
oracle).  S half only.  Plan and readings: docs/research/x1012_tasks/W50_scale_matching_PLAN.md.

    nice -n 19 python scripts/x1012/w50_scale_matching.py produce
    python scripts/x1012/w50_scale_matching.py report
"""
import gzip, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c
from w45_solvability import best_cut

OUT = c.task_path('W50')
WEIGHTS = ROOT.parent / 'cv_data/demo4_cache/models/dinov3-vitl16-timm'
ARMS = ('id', 'match')
CANVAS = (124, 116, 104)


class Encoder:
    def __init__(self):
        import torch, timm
        from safetensors.torch import load_file
        import importlib.util
        self.torch = torch
        self.dev = torch.device('mps')
        m = timm.create_model('vit_large_patch16_dinov3', pretrained=False, num_classes=0)
        m.load_state_dict(load_file(str(WEIGHTS / 'model.safetensors')), strict=True)
        self.m, self.npre = m.eval().to(self.dev), m.num_prefix_tokens
        spec = importlib.util.spec_from_file_location('foris_data', c.ASSETS / 'third_party/foris_official/utils/data.py')
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.tf = mod.build_transform(1024)

    def __call__(self, img):
        with self.torch.inference_mode():
            x = self.tf(img)[None].to(self.dev)
            o = self.m.forward_features(x)[0, self.npre:].float().cpu()
        assert o.shape == (4096, 1024)
        return o


def reference(row):
    """Official reference RGB (with crop) as PIL and its mask as a bool array in the same frame."""
    from PIL import Image
    from ics.official_data import decoded_rgb
    img = decoded_rgb(c.ASSETS / row['reference_path'], row.get('reference_crop'))
    m = np.array(Image.open(row['reference_mask_path']).convert('L')) > 0
    return img, m


def rescale(img, m, s):
    """Zoom in (s >= 1) by a centroid window of 1/s per axis at native resolution; zoom out (s < 1) by shrinking the
    1024 frame into a centred canvas.  Returns (PIL image, mask1024 bool tensor)."""
    import torch
    from PIL import Image
    if s >= 1:
        W, H = img.size; h, w = m.shape
        ys, xs = np.nonzero(m)
        cy, cx = (ys.mean() + .5) / h, (xs.mean() + .5) / w             # normalised centroid
        half = .5 / s
        y0 = min(max(cy - half, 0.), 1 - 2 * half); x0 = min(max(cx - half, 0.), 1 - 2 * half)
        box = lambda X, Y: (int(round(x0 * X)), int(round(y0 * Y)), max(int(round((x0 + 2 * half) * X)), int(round(x0 * X)) + 1),
                            max(int(round((y0 + 2 * half) * Y)), int(round(y0 * Y)) + 1))
        im2 = img.crop(box(W, H))
        mi = Image.fromarray(m.astype(np.uint8) * 255).crop(box(w, h)).resize((1024, 1024), Image.NEAREST)
        return im2, torch.from_numpy(np.array(mi) > 0)
    k = max(int(round(1024 * s)), 1); o = (1024 - k) // 2
    im2 = Image.new('RGB', (1024, 1024), CANVAS)
    im2.paste(img.resize((1024, 1024), Image.BILINEAR).resize((k, k), Image.BILINEAR), (o, o))
    m1024 = Image.fromarray(m.astype(np.uint8) * 255).resize((1024, 1024), Image.NEAREST).resize((k, k), Image.NEAREST)
    mm = np.zeros((1024, 1024), bool); mm[o:o + k, o:o + k] = np.array(m1024) > 0
    return im2, torch.from_numpy(mm)


def run_sh(b, feat, mask):
    """SH (pre-CRF) with the reference features and mask replaced in memory."""
    import w42_ordered_cut as w42
    rc, lab = c.rc, c.rc.lab
    oraw, omask = rc.raw, lab.mask1024
    b2 = dict(b, fit=dict(b['fit'], reference_raw={'w50': 1}), row=dict(b['row'], reference_mask_path='W50::reference'))
    rc.raw = lambda meta, counters: feat.clone() if meta == {'w50': 1} else oraw(meta, counters)
    lab.mask1024 = lambda p: mask.clone() if p == 'W50::reference' else omask(p)
    try:
        F, A0, U, M, rho = w42.capture(b2)
    finally:
        rc.raw, lab.mask1024 = oraw, omask
    return F, A0, rho, U is None


def one(b, enc):
    import torch
    import w31_demonstration as w31
    began = time.monotonic()
    img, m = reference(b['row'])
    stored = c.rc.raw(b['fit']['reference_raw'], {})
    f_id = enc(img)
    cos = torch.nn.functional.cosine_similarity(f_id, stored, dim=1)
    gt, valid = c.truth(b)
    q = float(gt.mean())                                          # the only query-GT quantity used before scoring
    m1024 = c.rc.lab.mask1024(b['row']['reference_mask_path'])
    r_ = float(m1024.float().mean())
    s = float(np.clip(np.sqrt(max(q, 1e-9) / max(r_, 1e-9)), 1 / 8, 8))
    out = {'id': run_sh(b, f_id, m1024)}
    im2, mm = rescale(img, m, s)
    if mm.any():
        out['match'] = run_sh(b, enc(im2), mm)
    else:                                                          # zoomed window lost the object: keep id
        out['match'] = out['id']
    r = dict(w31.meta(b), gt=int(gt.sum()), q=q, r=r_, s=s, r_after=float(mm.float().mean()), empty_match=not bool(mm.any()),
             cos_median=float(cos.median()), cos_min=float(cos.min()))
    for a, (F, A0, rho, fb) in out.items():
        r[a] = c.iu(A0, gt, valid); r['cut_' + a] = best_cut(F, gt); r['fb_' + a] = fb
    r['seconds'] = round(time.monotonic() - began, 2)
    return r


def produce_all():
    import torch
    import w31_demonstration as w31
    c.initialize()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in w31.rows() if str(b['index']) not in done]
    print('todo', len(todo), flush=True)
    enc = Encoder()
    with path.open('a') as fh:
        for n, b in enumerate(todo, 1):
            r = one(b, enc)
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 50 == 0:
                print(n, len(todo), r['seconds'], flush=True)
    print('done', flush=True)


def report():
    import score as sc
    import w37_coverage as w37
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    W48 = {r['index']: r for r in map(json.loads, (c.task_path('W48') / 'episodes.jsonl').read_text().splitlines())}
    S0 = {r['index']: r for r in map(json.loads, (c.task_path('W48') / 'episodes_self0.jsonl').read_text().splitlines())}
    j = lambda v: v[0] / max(v[1], 1)
    for r in R:
        r['true'] = W48[r['index']]['true']; r['self'] = S0[r['index']]['self0']
        r['lr'] = float(np.log2(max(r['q'], 1e-9) / max(r['r'], 1e-9)))
    close = np.mean([abs(j(r['id']) - j(r['true'])) < .01 for r in R])
    L = [f'n {len(R)}  check: median token cosine (median over episodes) {np.median([r["cos_median"] for r in R]):.5f}, '
         f'min {min(r["cos_min"] for r in R):.4f}; |IoU(id) - IoU(W48 true)| < .01 in {close:.1%} (need median cos > .999 and >= 95%)',
         f'zoom window lost the object in {sum(r["empty_match"] for r in R)} episodes (match = id there)', '']
    doms = list(c.DOMAINS)
    by = {d: [r for r in R if r['dataset'] == d] for d in doms}
    ci = np.percentile(np.mean([w37._boot(sc, by[d], 'match', 'id') for d in doms], axis=0), [2.5, 97.5])
    L.append('class mIoU ten domains: id %.2f  match %.2f  true(W48) %.2f  self0 %.2f;  match - id %+.2f [%+.2f,%+.2f]' % (
        *[np.mean([sc.miou(by[d], a) for d in doms]) for a in ('id', 'match', 'true', 'self')],
        np.mean([sc.miou(by[d], 'match') - sc.miou(by[d], 'id') for d in doms]), *ci))
    L.append('  per domain match - id: ' + '  '.join(f'{d[:6]} {sc.miou(by[d], "match") - sc.miou(by[d], "id"):+.2f}' for d in doms))
    L.append('')
    L.append(f'{"subset":28s}{"n":>5s}{"self0":>7s}{"id":>7s}{"match":>7s}{"cut id":>8s}{"cut m":>7s}{"R":>8s}')
    rows = []
    for tag, sel in (('|lr| >= 2 (main)', [r for r in R if abs(r['lr']) >= 2]),
                     ('  zoom in  (q > r, lr >= 2)', [r for r in R if r['lr'] >= 2]),
                     ('  zoom out (q < r, lr <= -2)', [r for r in R if r['lr'] <= -2]),
                     ('|lr| >= 4', [r for r in R if abs(r['lr']) >= 4]),
                     ('1 <= |lr| < 2', [r for r in R if 1 <= abs(r['lr']) < 2]),
                     ('|lr| < 1 (control)', [r for r in R if abs(r['lr']) < 1]),
                     ('  control, zoom out side', [r for r in R if -1 < r['lr'] < 0]),
                     ('  control, zoom in side', [r for r in R if 0 <= r['lr'] < 1])):
        if not sel:
            continue
        a = {k: np.mean([j(r[k]) for r in sel]) for k in ('self', 'id', 'match')}
        R_ = (a['match'] - a['id']) / (a['self'] - a['id']) if a['self'] != a['id'] else float('nan')
        L.append(f'{tag:28s}{len(sel):5d}{a["self"]:7.3f}{a["id"]:7.3f}{a["match"]:7.3f}{np.mean([r["cut_id"] for r in sel]):8.3f}'
                 f'{np.mean([r["cut_match"] for r in sel]):7.3f}{R_:+8.1%}')
    L.append(f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}')
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    keys = ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'q', 'r', 's', 'r_after', 'empty_match', 'cos_median', 'cos_min',
            'id', 'match', 'cut_id', 'cut_match', 'fb_id', 'fb_match')
    with gzip.open(ROOT / 'evidence/research/w50_scale_matching_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in keys}) + '\n')


if __name__ == '__main__':
    produce_all() if sys.argv[1] == 'produce' else report()
