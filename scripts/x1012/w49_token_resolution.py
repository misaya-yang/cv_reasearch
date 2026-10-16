"""W49: can the token grid express the target at all?  Best-cut IoU of the bilinearly upsampled TRUE coverage on
64/128/256 grids (no features).  Plan and readings: docs/research/x1012_tasks/W49_token_resolution_PLAN.md.

    nice -n 19 python scripts/x1012/w49_token_resolution.py produce [workers]
    python scripts/x1012/w49_token_resolution.py report
"""
import gzip, json, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c
from diagnostic_metrics import exact_best_iou, classify_diagnostic, DIAGNOSTIC_CLASSES, SCORING_VERSION

OUT = c.task_path('W49_exact')  # W58 repair: retain the original W49 ledger verbatim.
GRIDS = (64, 128, 256)
CHECK = []


def fast_cut(f, gt, valid=None):
    """Exact oracle for this coverage field only, not a grid-family upper bound."""
    return exact_best_iou(f, gt, valid)


def one(b):
    import torch
    import torch.nn.functional as F
    import w31_demonstration as w31
    began = time.monotonic()
    gt, valid = c.truth(b)
    r = dict(w31.meta(b), gt=int(gt.sum()), scoring=SCORING_VERSION)
    t = torch.from_numpy(gt.astype(np.float32))
    for g in GRIDS:
        k = 1024 // g
        y = t.reshape(g, k, g, k).mean((1, 3))
        f = F.interpolate(y[None, None], size=(1024, 1024), mode='bilinear', align_corners=False)[0, 0].numpy().astype(np.float64)
        r[f'C{g}'] = fast_cut(f, gt, valid)
        r[f'C{g}_all'] = r[f'C{g}'] if valid.all() else fast_cut(f, gt)
        if g == 64:
            r['N64'] = c.iu(f > .5, gt, valid)
            assert r['C64'] + 1e-12 >= r['N64'][0] / max(r['N64'][1], 1)
            if len(CHECK) < 20:                 # first 20 episodes of each worker
                from w45_solvability import best_cut
                CHECK.append(abs(best_cut(f, gt, valid) - r['C64']))
                r['check'] = CHECK[-1]
    r['seconds'] = round(time.monotonic() - began, 2)
    return r



def init():
    c.initialize()
    import torch
    torch.set_num_threads(1)


def produce_all(workers):
    import w31_demonstration as w31
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in w31.rows() if str(b['index']) not in done]
    print('todo', len(todo), flush=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=init) as pool, \
            path.open('a') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=4), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 200 == 0:
                print(n, len(todo), r['seconds'], flush=True)
    print('done', flush=True)


def report():
    import score as sc
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    assert len(R) == 1285 and len({r['index'] for r in R}) == 1285
    assert all(r['scoring'] == SCORING_VERSION for r in R)
    w27 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt'))}
    w45 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w45_solvability_S_exact.jsonl.gz', 'rt'))}
    chk = [r['check'] for r in R if 'check' in r]
    for r in R:
        e = w45[r['index']]
        r['cat'] = w27[r['index']]['SH+'][0] / max(w27[r['index']]['SH+'][1], 1) < .3
        r['w45'] = classify_diagnostic(e)
        r['PQ'], r['F'] = e['cut_PQ'], e['cut_F']
    doms = list(c.DOMAINS)
    L = [f'n {len(R)}  shared scorer wrapper agreement on {len(chk)} episodes, max diff {max(chk) if chk else float("nan"):.2e}; independent brute-force tests in test_diagnostic_metrics.py', '']
    m = lambda S, k: np.mean([r[k] for r in S]) if S else float('nan')
    for tag, sel in (('catastrophic', [r for r in R if r['cat']]), ('non-catastrophic', [r for r in R if not r['cat']]), ('all', R)):
        L.append(f'## {tag} n {len(sel)}: C64 {m(sel, "C64"):.3f} C128 {m(sel, "C128"):.3f} C256 {m(sel, "C256"):.3f} | P_Q {m(sel, "PQ"):.3f} F {m(sel, "F"):.3f}; '
                 f'C64 < .5 in {np.mean([r["C64"] < .5 for r in sel]):.1%}, C128 < .5 {np.mean([r["C128"] < .5 for r in sel]):.1%}, C256 < .5 {np.mean([r["C256"] < .5 for r in sel]):.1%}')
        for k in DIAGNOSTIC_CLASSES:
            S = [r for r in sel if r['w45'] == k]
            if S:
                L.append(f'   W45 {k:14s} n {len(S):4d}: C64 {m(S, "C64"):.3f} C128 {m(S, "C128"):.3f} C256 {m(S, "C256"):.3f} P_Q {m(S, "PQ"):.3f}; '
                         f'C64 < .5 {np.mean([r["C64"] < .5 for r in S]):.1%}, C128 < .5 {np.mean([r["C128"] < .5 for r in S]):.1%}')
        L.append('')
    rep = [r for r in R if r['cat'] and r['w45'] == 'unresolved_by_probes']
    res = [r for r in rep if r['C64'] < .5]
    L.append(f'catastrophic unresolved-by-probes n {len(rep)}: low coverage reconstruction (C64 < .5) {len(res)} ({len(res) / max(len(rep), 1):.1%}), '
             f'coverage witness >= .5 in {len(rep) - len(res)}; among low C64 cases, C128 >= .5 in {np.mean([r["C128"] >= .5 for r in res]) if res else float("nan"):.1%}, '
             f'C256 >= .5 in {np.mean([r["C256"] >= .5 for r in res]) if res else float("nan"):.1%}')
    L.append('')
    L.append(f'{"domain":12s}{"n":>5s}{"C64":>7s}{"C128":>7s}{"C256":>7s}{"<.5@64":>8s}{"unres.":>8s}{"lowC64":>8s}{"N64 mIoU":>10s}')
    for d in doms:
        D = [r for r in R if r['dataset'] == d]; rp = [r for r in D if r['cat'] and r['w45'] == 'unresolved_by_probes']
        L.append(f'{d[:11]:12s}{len(D):5d}{m(D, "C64"):7.3f}{m(D, "C128"):7.3f}{m(D, "C256"):7.3f}{np.mean([r["C64"] < .5 for r in D]):8.1%}'
                 f'{len(rp):8d}{sum(r["C64"] < .5 for r in rp):8d}{sc.miou(D, "N64"):10.2f}')
    L.append(f'ten-domain mIoU of the natural (.5) threshold on true 64-grid coverage: {np.mean([sc.miou([r for r in R if r["dataset"] == d], "N64") for d in doms]):.2f}')
    L.append(f'seconds per episode median {np.median([r["seconds"] for r in R]):.2f}')
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    keys = ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'C64', 'C128', 'C256', 'N64', 'C64_all', 'C128_all', 'C256_all', 'scoring')
    with gzip.open(ROOT / 'evidence/research/w49_token_resolution_S_exact.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in keys}) + '\n')


if __name__ == '__main__':
    produce_all(int(sys.argv[2]) if len(sys.argv) > 2 else 2) if sys.argv[1] == 'produce' else report()
