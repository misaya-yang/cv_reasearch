"""W48: how much of SH's output comes from the reference's concept?  The same frozen SH with three references:
the query itself with its GT (oracle, diagnostic), the true reference, and a mismatched-class reference from the same
domain.  S half only.  Plan and readings: docs/research/x1012_tasks/W48_reference_specificity_PLAN.md.

    nice -n 19 python scripts/x1012/w48_reference_specificity.py produce [workers]
    nice -n 19 python scripts/x1012/w48_reference_specificity.py self0 [workers]   # corrected self arm (plan §3c)
    python scripts/x1012/w48_reference_specificity.py report
"""
import copy, gzip, json, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c
from w45_solvability import best_cut
from w47_concept_consistency import pairs

OUT = c.task_path('W48')
REF = ('path', 'crop', 'mask_path', 'mask_hash', 'rgb_hash', 'photo_id', 'size_hw', 'mask_size_hw', 'object_id')
ARMS = ('self', 'true', 'mis')


def with_reference(b, row, raw, role):
    """Copy of binding b whose reference (image, mask, crop, features) is taken from `row` / `raw` with prefix `role`."""
    b2 = copy.deepcopy(b)
    for k in REF:
        b2['row']['reference_' + k] = row.get(role + '_' + k)
    b2['fit']['reference_raw'] = copy.deepcopy(raw)
    return b2


def one(pair):
    import w31_demonstration as w31
    import w42_ordered_cut as w42
    b, other = pair
    began = time.monotonic()
    variants = {'self': with_reference(b, b['row'], b['fit']['query_raw_identity'], 'query'), 'true': b}
    if other is not None:                       # single-class domains (Lung, Fundus, Road) have no mismatched class
        variants['mis'] = with_reference(b, other['row'], other['fit']['reference_raw'], 'reference')
    out = {}
    for a, bb in variants.items():
        F, A0, U, M, rho = w42.capture(bb)
        out[a] = (F, A0, rho, U is None)
    gt, valid = c.truth(b)                      # read only after every output is fixed
    r = dict(w31.meta(b), gt=int(gt.sum()), other=str(other['index']) if other is not None else None)
    for a, (F, A0, rho, fb) in out.items():
        r[a] = c.iu(A0, gt, valid); r['cut_' + a] = best_cut(F, gt); r['rho_' + a] = rho; r['fb_' + a] = fb
    r['seconds'] = round(time.monotonic() - began, 2)
    return r


def one_self0(b):
    """Self arm with the degenerate novelty handled by its limit (plan §3c): when sim_max is constant, novelty^s is a
    constant factor that the min-max inside solve removes, so c_mod = c.  Never triggers on a different reference."""
    import torch
    import w42_ordered_cut as w42
    began = time.monotonic()
    om, hit = c.rc.modulate, []

    def modulate(cc, labels, ind0, indn, sim_max, back_f):
        sm = np.asarray(sim_max, np.float32)
        if float(sm.max() - sm.min()) == 0.:
            hit.append(1)
            return cc, {'selected_s': 0., 'J_grid': []}
        return om(cc, labels, ind0, indn, sim_max, back_f)
    c.rc.modulate = modulate
    try:
        F, A0, U, M, rho = w42.capture(with_reference(b, b['row'], b['fit']['query_raw_identity'], 'query'))
    finally:
        c.rc.modulate = om
    gt, valid = c.truth(b)
    return {'index': str(b['index']), 'self0': c.iu(A0, gt, valid), 'cut_self0': best_cut(F, gt), 'fb_self0': U is None,
            'degenerate': bool(hit), 'seconds': round(time.monotonic() - began, 2)}


def produce_self0(workers):
    import w31_demonstration as w31
    path = OUT / 'episodes_self0.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in w31.rows() if str(b['index']) not in done]
    print('todo', len(todo), flush=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=init) as pool, \
            path.open('a') as fh:
        for n, r in enumerate(pool.map(one_self0, todo, chunksize=2), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 100 == 0:
                print(n, len(todo), r['seconds'], flush=True)
    print('done', flush=True)


def init():
    c.initialize()
    import torch
    torch.set_num_threads(1)


def produce_all(workers):
    import w31_demonstration as w31
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [p for p in pairs(w31.rows()) if str(p[0]['index']) not in done]
    print('todo', len(todo), flush=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'), initializer=init) as pool, \
            path.open('a') as fh:
        for n, r in enumerate(pool.map(one, todo, chunksize=2), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 50 == 0:
                print(n, len(todo), r['seconds'], flush=True)
    print('done', flush=True)


def report():
    import score as sc
    import w37_coverage as w37
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    S0 = {r['index']: r for r in map(json.loads, (OUT / 'episodes_self0.jsonl').read_text().splitlines())}
    raw = {'n': len(R), 'empty': np.mean([r['self'][2] == 0 for r in R])}
    for r in R:                                 # plan §3c: the self arm is replaced by its corrected limit self0
        s0 = S0[r['index']]
        r['self_raw'], r['cut_self_raw'] = r['self'], r['cut_self']
        r['self'], r['cut_self'], r['fb_self'], r['degenerate'] = s0['self0'], s0['cut_self0'], s0['fb_self0'], s0['degenerate']
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    w27 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt'))}
    w45 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w45_solvability_S.jsonl.gz', 'rt'))}
    same = sum(r['true'][:2] == w23[r['index']]['SH'][:2] for r in R)
    j = lambda r, a: r[a][0] / max(r[a][1], 1)
    for r in R:
        r['cat'] = w27[r['index']]['SH+'][0] / max(w27[r['index']]['SH+'][1], 1) < .3
    doms = list(c.DOMAINS)
    by = {d: [r for r in R if r['dataset'] == d] for d in doms}
    mdoms = [d for d in doms if all('mis' in r for r in by[d])]
    L = [f'self arm = corrected self0 (plan 3c); raw self arm had {raw["empty"]:.1%} empty outputs, now '
         f'{np.mean([r["self"][2] == 0 for r in R]):.1%}; degenerate novelty in {np.mean([r["degenerate"] for r in R]):.1%}; '
         f'raw self ten-domain mIoU {np.mean([sc.miou([r for r in R if r["dataset"] == d], "self_raw") for d in c.DOMAINS]):.2f} (invalid)',
         f'mismatched reference available in {len(mdoms)} domains: {mdoms}', f'n {len(R)}  check: true == W23 SH {same}/{len(R)} (need >= 1280); fallback rate self {np.mean([r["fb_self"] for r in R]):.1%}, '
         f'true {np.mean([r["fb_true"] for r in R]):.1%}, mismatched {np.mean([r["fb_mis"] for r in R if "fb_mis" in r]):.1%}', '',
         '## class mIoU (pre-CRF)', f'{"":12s}{"self":>8s}{"true":>8s}{"mis":>8s}{"transfer":>10s}{"concept":>9s}']
    for d in doms:
        m = {a: sc.miou(by[d], a) if d in mdoms or a != 'mis' else float('nan') for a in ARMS}
        L.append(f'{d[:11]:12s}' + ''.join(f'{m[a]:8.2f}' for a in ARMS) + f'{m["self"] - m["true"]:+10.2f}{m["true"] - m["mis"]:+9.2f}')
    ci = lambda a, b_, D: np.percentile(np.mean([w37._boot(sc, by[d], a, b_) for d in D], axis=0), [2.5, 97.5])
    T = np.mean([sc.miou(by[d], 'self') - sc.miou(by[d], 'true') for d in doms]); t_lo, t_hi = ci('self', 'true', doms)
    C = np.mean([sc.miou(by[d], 'true') - sc.miou(by[d], 'mis') for d in mdoms]); c_lo, c_hi = ci('true', 'mis', mdoms)
    L.append(f'{"means":12s}self {np.mean([sc.miou(by[d], "self") for d in doms]):.2f} true {np.mean([sc.miou(by[d], "true") for d in doms]):.2f} (ten domains); '
             f'true {np.mean([sc.miou(by[d], "true") for d in mdoms]):.2f} mis {np.mean([sc.miou(by[d], "mis") for d in mdoms]):.2f} ({len(mdoms)} domains)')
    L += [f'transfer loss (self - true, ten domains) {T:+.2f} [{t_lo:+.2f},{t_hi:+.2f}]; concept contribution (true - mis, {len(mdoms)} domains) '
          f'{C:+.2f} [{c_lo:+.2f},{c_hi:+.2f}]', '']

    def w45class(e):
        return 'representation' if e['cut_PQ'] < .5 else 'readout' if e['cut_F'] >= .5 else 'transfer' if e['cut_PR'] < .3 else 'other'
    for tag, sel in (('catastrophic (domains with a mismatched class)', [r for r in R if r['cat'] and 'mis' in r]),
                     ('non-catastrophic (domains with a mismatched class)', [r for r in R if not r['cat'] and 'mis' in r]),
                     ('catastrophic, single-class domains (self vs true only)', [r for r in R if r['cat'] and 'mis' not in r])):
        if not sel:
            continue
        if 'mis' not in sel[0]:
            L.append(f'## {tag} (n {len(sel)}): mean IoU self {np.mean([j(r, "self") for r in sel]):.3f}, true {np.mean([j(r, "true") for r in sel]):.3f}; self < .5 in {np.mean([j(r, "self") < .5 for r in sel]):.1%}')
            L.append('')
            continue
        L.append(f'## {tag} (n {len(sel)}): mean episode IoU self {np.mean([j(r, "self") for r in sel]):.3f}, true {np.mean([j(r, "true") for r in sel]):.3f}, '
                 f'mis {np.mean([j(r, "mis") for r in sel]):.3f}; best-cut self {np.mean([r["cut_self"] for r in sel]):.3f}, true {np.mean([r["cut_true"] for r in sel]):.3f}, mis {np.mean([r["cut_mis"] for r in sel]):.3f}')
        L.append(f'   mismatched IoU >= true IoU in {np.mean([j(r, "mis") >= j(r, "true") for r in sel]):.1%} of episodes; self IoU < .5 in {np.mean([j(r, "self") < .5 for r in sel]):.1%}')
        for k in ('representation', 'readout', 'transfer', 'other'):
            S = [r for r in sel if w45class(w45[r['index']]) == k]
            if S:
                L.append(f'   W45 {k:14s} n {len(S):3d}: IoU self {np.mean([j(r, "self") for r in S]):.3f} true {np.mean([j(r, "true") for r in S]):.3f} mis {np.mean([j(r, "mis") for r in S]):.3f}; mis >= true {np.mean([j(r, "mis") >= j(r, "true") for r in S]):.0%}')
        L.append('')
    cat = [r for r in R if r['cat']]; catm = [r for r in cat if 'mis' in r]
    L += [f'reading 3: mismatched >= true in {np.mean([j(r, "mis") >= j(r, "true") for r in catm]):.1%} of catastrophic episodes with a mismatched class (>= 40% -> reference concept not acting); '
          f'self mean IoU over all catastrophic {np.mean([j(r, "self") for r in cat]):.3f} (< .5 -> SH machinery fails even without transfer)',
          f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}']
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    keys = ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'other', 'self_raw', 'cut_self_raw', 'degenerate') + ARMS + tuple('cut_' + a for a in ARMS) + \
        tuple('rho_' + a for a in ARMS) + tuple('fb_' + a for a in ARMS)
    with gzip.open(ROOT / 'evidence/research/w48_reference_specificity_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in keys if k in r}) + '\n')


if __name__ == '__main__':
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    produce_all(w) if sys.argv[1] == 'produce' else produce_self0(w) if sys.argv[1] == 'self0' else report()
