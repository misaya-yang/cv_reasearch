"""W56: contrast boards -- per domain, the S episodes FoRIS wins most, SH+ wins most, and two where both fail.
Row = reference + mask | query + GT | query + FoRIS | query + SH+.  For looking only (hypotheses, no method).
Plan: docs/research/x1012_tasks/W56_contrast_boards_PLAN.md (fixed before looking).

    nice -n 19 python scripts/x1012/w56_contrast_boards.py [workers]
"""
import gzip, json, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

OUT = c.task_path('W56')
CELL = 256
COL = {'ref': (255, 200, 0), 'gt': (0, 220, 0), 'foris': (255, 60, 60), 'sh': (60, 140, 255)}


def iou(v):
    return v[0] / max(v[1], 1)


def select(rows):
    L = lambda f: {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research' / f, 'rt'))}
    base, w23 = L('x1013_E1_iu.jsonl.gz'), L('w23_simple_pipeline_S.jsonl.gz')
    rng = np.random.default_rng(2026101503)
    pick = []
    for d in c.DOMAINS:
        R = [b for b in rows if b['dataset'] == d]
        diff = {str(b['index']): iou(base[str(b['index'])]['foris']) - iou(w23[str(b['index'])]['SH+']) for b in R}
        order = sorted(R, key=lambda b: diff[str(b['index'])])
        sh_win, f_win = order[:4], order[::-1][:4]
        used = {str(b['index']) for b in sh_win + f_win}
        both = [b for b in R if str(b['index']) not in used and iou(base[str(b['index'])]['foris']) < .3
                and iou(w23[str(b['index'])]['SH+']) < .3]
        both = [both[i] for i in sorted(rng.choice(len(both), min(2, len(both)), replace=False))] if both else []
        pick += [(d, 'FoRIS wins', b) for b in f_win] + [(d, 'SH+ wins', b) for b in sh_win] + [(d, 'both fail', b) for b in both]
    return pick


def tint(rgb, mask, color):
    from scipy import ndimage
    x = rgb.copy()
    x[mask] = .55 * x[mask] + .45 * np.array(color, np.float64) / 255
    edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
    x[edge] = np.array(color, np.float64) / 255
    return x


def row(job):
    import w23_simple_pipeline as sh
    import w42_ordered_cut as w42
    from PIL import Image, ImageDraw
    d, kind, b = job
    lab = c.rc.lab
    _, shp, rho = sh.simple_head(b, crf=True)
    fo = lab.arm_cli(b, 'B').numpy()
    gt, valid = c.truth(b)
    rm = lab.mask1024(b['row']['reference_mask_path']).numpy() > .5
    rr = w42.rgb(b['row']['reference_path'], b['row'].get('reference_crop'))
    qq = w42.rgb(b['row']['query_path'], b['row'].get('query_crop'))
    cells = [tint(rr, rm, COL['ref']), tint(qq, gt, COL['gt']), tint(qq, fo, COL['foris']), tint(qq, shp, COL['sh'])]
    img = np.concatenate([np.asarray(Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).resize((CELL, CELL), Image.BILINEAR))
                          for x in cells], 1)
    i_f, i_s = c.iu(fo, gt, valid), c.iu(shp, gt, valid)
    head = Image.new('RGB', (4 * CELL, 30), (255, 255, 255))
    t = (f"{d} | {kind} | idx {b['index']} | IoU FoRIS {100 * iou(i_f):.0f}  SH+ {100 * iou(i_s):.0f} | "
         f"area% ref {100 * rm.mean():.1f} GT {100 * gt.mean():.1f} FoRIS {100 * fo.mean():.1f} SH+ {100 * shp.mean():.1f} | rho {rho:.2f}")
    ImageDraw.Draw(head).text((4, 8), t, fill=(0, 0, 0))
    meta = dict(domain=d, kind=kind, index=str(b['index']), iou_foris=iou(i_f), iou_sh=iou(i_s), ref_area=float(rm.mean()),
                gt_area=float(gt.mean()), foris_area=float(fo.mean()), sh_area=float(shp.mean()), rho=rho)
    return np.concatenate([np.asarray(head), img], 0), meta


def main(workers):
    import w31_demonstration as w31
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = select(w31.rows())
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=c.initialize) as pool:
        out = list(pool.map(row, jobs))
    metas = []
    for d in c.DOMAINS:
        R = [(img, m) for img, m in out if m['domain'] == d]
        for k in range(0, len(R), 5):
            Image.fromarray(np.concatenate([img for img, _ in R[k:k + 5]], 0)).save(OUT / f'{d}_{k // 5 + 1}.png')
        metas += [m for _, m in R]
    (OUT / 'selection.json').write_text(json.dumps(metas, indent=1))
    print('boards', len(out), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
