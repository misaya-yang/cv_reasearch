"""W44: reference relation forest (W39 B section 5; W34 A8 section 6.2-6.4; W34 A2 section 10.3-10.4).  S half only.
Plan and readings: docs/research/x1012_tasks/W44_relation_forest_PLAN.md.  One process: frozen encoder on MPS for the
last-block pre-RoPE keys of reference and query (kept in memory only), SH capture and the forest on CPU.

    nice -n 19 python scripts/x1012/w44_relation_forest.py produce
    python scripts/x1012/w44_relation_forest.py report
"""
import gzip, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
import common as c

OUT = c.task_path('W44')
WEIGHTS = ROOT.parent / 'cv_data/demo4_cache/models/dinov3-vitl16-timm'
ARMS = ('A0', 'F5', 'F5lit', 'FC', 'FD', 'AC')
REG = ('C0', 'C1', 'D', 'fb')
G = 64
_ii, _jj = np.meshgrid(np.arange(G), np.arange(G), indexing='ij')
_id = (_ii * G + _jj)
EP = np.concatenate([_id[:, :-1].ravel(), _id[:-1, :].ravel()])            # 8064 4-neighbour token edges
EQ = np.concatenate([_id[:, 1:].ravel(), _id[1:, :].ravel()])


def relations(o, k, heads):
    """A8 eq. (3): [(1 - o_i.o_j)/2, {d_h / (d_h + 2 v_h)}_h] for the 8064 token edges."""
    z0 = (1 - (o[EP] * o[EQ]).sum(1)) / 2
    v = ((k - k.mean(1, keepdims=True)) ** 2).sum(2).mean(1)
    d = ((k[:, EP] - k[:, EQ]) ** 2).sum(2)
    zk = d / (d + 2 * v[:, None])
    return np.concatenate([z0[:, None], zk[heads].T], 1)


def dist(a, b):
    """Root mean squared coordinate difference between rows of a and rows of b (float32)."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    sq = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(sq, 0) / a.shape[1])


def support(DQJ, DQC, DJC, xiJ, xiC, visJ, visC):
    """A8 eq. (4) for one reference version (visible J / C reference edges).  Returns (s_J, s_C) or None."""
    if not visJ.any() or not visC.any():
        return None
    D = DJC[np.ix_(visJ, visC)]
    out = []
    for DQ, xi, vis, delta in ((DQJ, xiJ, visJ, D.min(1)), (DQC, xiC, visC, D.min(0))):
        ok = (delta > 0) & (xi[vis] > 0)
        s = np.zeros(DQ.shape[0])
        if ok.any():
            cols = np.flatnonzero(vis)[ok]
            x, dl = xi[cols].astype(np.float32), delta[ok].astype(np.float32)
            for i in range(0, DQ.shape[0], 2048):
                s[i:i + 2048] = np.maximum((x[None, :] - 2 * DQ[i:i + 2048][:, cols] / dl[None, :]).max(1), 0)
        out.append(s)
    return out[0], out[1]


def terminals(sim, y, vis_tok):
    """A2 10.3: mutual nearest neighbours to visible pure reference tokens with strict class dominance."""
    sim = sim.astype(np.float32)
    s = np.where(vis_tok[None, :], sim, -np.inf)
    j = s.argmax(1)
    i_of = s.argmax(0)
    mutual = i_of[j] == np.arange(len(j))
    pos, neg = vis_tok & (y == 1), vis_tok & (y == 0)
    if not pos.any() or not neg.any():
        return None
    mp, mn = np.where(pos[None, :], sim, -np.inf).max(1), np.where(neg[None, :], sim, -np.inf).max(1)
    tp = mutual & (y[j] == 1) & (mp > mn)
    tn = mutual & (y[j] == 0) & (mn > mp)
    return tp, tn


def forest(sJ, sC, tp, tn):
    """A8 6.4 Kruskal with cut-path rejection, and A2 eq. (16) bottlenecks.  Returns labels (+1/-1/0), forest edges."""
    n = G * G
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    members = {i: [i] for i in range(n)}
    cut = {i: set() for i in range(n)}
    for e in np.flatnonzero(sC > 0):
        cut[EP[e]].add(EQ[e]); cut[EQ[e]].add(EP[e])
    hp, hn = tp.copy(), tn.copy()                              # per root: component has a +/- terminal
    kp, kn = tp.astype(np.float64), tn.astype(np.float64)
    order = sorted(np.flatnonzero(sJ > 0), key=lambda e: (-sJ[e], min(EP[e], EQ[e]), max(EP[e], EQ[e])))
    used = []
    for e in order:
        a, b = find(EP[e]), find(EQ[e])
        if a == b:
            continue
        if len(cut[a]) > len(cut[b]):
            a, b = b, a
        if any(find(t) == b for t in cut[a]):
            continue
        w = sJ[e]
        if hp[a] and not hp[b]:
            kp[members[b]] = w
        elif hp[b] and not hp[a]:
            kp[members[a]] = w
        if hn[a] and not hn[b]:
            kn[members[b]] = w
        elif hn[b] and not hn[a]:
            kn[members[a]] = w
        if len(members[a]) > len(members[b]):
            a, b = b, a
        parent[a] = b
        members[b] += members.pop(a)
        cut[b] |= cut.pop(a)
        hp[b] |= hp[a]; hn[b] |= hn[a]
        used.append(e)
    lab = np.where(kp > kn, 1, np.where(kn > kp, -1, 0))
    return lab, np.array(used, int)


def blow(t):
    return np.kron(t.reshape(G, G), np.ones((16, 16), bool))


def edit(A0, lab, literal=False):
    n = A0.reshape(G, 16, G, 16).mean((1, 3)).ravel()
    add, rem = (lab == 1), (lab == -1)
    if not literal:
        add, rem = add & (n < .5), rem & (n >= .5)
    return (A0 | blow(add)) & ~blow(rem)


def budget_control(A0, F, out):
    """Same number of added / removed pixels as out, chosen by F margin (add highest F outside, remove lowest F inside)."""
    na, nr = int((out & ~A0).sum()), int((A0 & ~out).sum())
    m = A0.copy().ravel(); f = F.ravel()
    if na:
        cand = np.flatnonzero(~A0.ravel()); m[cand[np.argsort(-f[cand], kind='stable')[:na]]] = True
    if nr:
        cand = np.flatnonzero(A0.ravel()); m[cand[np.argsort(f[cand], kind='stable')[:nr]]] = False
    return m.reshape(A0.shape)


class Encoder:
    def __init__(self):
        import torch, timm
        from safetensors.torch import load_file
        self.torch = torch
        self.dev = torch.device('mps')
        m = timm.create_model('vit_large_patch16_dinov3', pretrained=False, num_classes=0)
        m.load_state_dict(load_file(str(WEIGHTS / 'model.safetensors')), strict=True)
        self.m = m.eval().to(self.dev)
        self.npre = m.num_prefix_tokens
        self.grab = {}
        m.blocks[-1].attn.k_norm.register_forward_hook(lambda mod, inp, out: self.grab.__setitem__('k', out))
        self.host, _ = c.rc.pc.host_cpu()

    def keys(self, row):
        from ics.official_data import decoded_rgb
        x = [self.host._transform(decoded_rgb(c.ASSETS / row[r + '_path'], row.get(r + '_crop'))) for r in ('reference', 'query')]
        with self.torch.inference_mode():
            self.m.forward_features(self.torch.stack(x).to(self.dev))
        k = self.grab.pop('k')[:, :, self.npre:, :].float().cpu().numpy().astype(np.float64)   # (2, 16, 4096, 64)
        return k[0], k[1]


def one(b, enc):
    import torch
    import produce
    import w31_demonstration as w31
    import w37_coverage as w37
    import w42_ordered_cut as w42
    began = time.monotonic()
    got, oi = [], c.rc.independent

    def keep(*a, **k):
        v = oi(*a, **k); got.append(v); return v
    c.rc.independent = keep
    try:
        F, A0, U, M, rho = w42.capture(b)
    finally:
        c.rc.independent = oi
    r_, q_, _, y, _ = got[0]
    fb = U is None
    r = dict(w31.meta(b), rho=rho, fallback=fb)
    out = {a: A0 for a in ARMS}
    status = 'fallback' if fb else 'ok'
    diag = {}
    if not fb:
        lab_ = c.rc.lab
        o_r, o_q = lab_.unit(r_).numpy().astype(np.float64), lab_.unit(q_).numpy().astype(np.float64)
        mu = .5 * (r_.mean(0) + q_.mean(0))
        sim = (lab_.unit(q_ - mu) @ lab_.unit(r_ - mu).T).numpy().astype(np.float64)
        y = y.numpy().astype(np.float64)
        kR, kQ = enc.keys(b['row'])
        vR = ((kR - kR.mean(1, keepdims=True)) ** 2).sum(2).mean(1)
        vQ = ((kQ - kQ.mean(1, keepdims=True)) ** 2).sum(2).mean(1)
        heads = (vR > 0) & (vQ > 0)
        r['heads'] = int(heads.sum())
        if not heads.any():
            status = 'no_heads'
        else:
            zR, zQ = relations(o_r, kR, heads), relations(o_q, kQ, heads)
            l = y >= .5
            Je, Ce = l[EP] & l[EQ], l[EP] != l[EQ]                 # reference J (FG-FG) and C (FG-BG) edges
            xi = np.minimum(np.abs(2 * y[EP] - 1), np.abs(2 * y[EQ] - 1))
            DQJ, DQC, DJC = dist(zQ, zR[Je]), dist(zQ, zR[Ce]), dist(zR[Je], zR[Ce])
            # reference versions: full, then each quadrant removed with a one-token guard band
            vis_tok = [np.ones(G * G, bool)]
            for qi in (0, 1):
                for qj in (0, 1):
                    gone = np.zeros((G, G), bool)
                    gone[max(32 * qi - 1, 0):32 * qi + 33, max(32 * qj - 1, 0):32 * qj + 33] = True
                    vis_tok.append(~gone.ravel())
            labs = []
            for v, vt in enumerate(vis_tok):
                ve = vt[EP] & vt[EQ]
                sup = support(DQJ, DQC, DJC, xi[Je], xi[Ce], ve[Je], ve[Ce]) if Je.any() and Ce.any() else None
                term = terminals(sim, y, vt)
                if sup is None or term is None:
                    labs.append(None)
                    if v == 0:
                        status = 'ref_empty'
                    continue
                sJ, sC = sup
                lab, used = forest(sJ, sC, *term)
                labs.append(lab)
                if v == 0:
                    diag = dict(sJ=sJ, sC=sC, tp=term[0], tn=term[1], used=used, lab=lab)
            if status == 'ok':
                out['F5'] = edit(A0, labs[0])
                out['F5lit'] = edit(A0, labs[0], literal=True)
                if all(x is not None for x in labs):
                    agree = np.all([x == labs[0] for x in labs[1:]], axis=0)
                    out['FC'] = edit(A0, np.where(agree, labs[0], 0))
                else:
                    r['fc_fallback'] = True
                C0, C1 = (U <= .5) & (M <= .5), (U > .5) & (M > .5)
                Dreg = ~(C0 | C1)
                out['FD'] = np.where(Dreg, out['F5'], A0)
                out['AC'] = budget_control(A0, F, out['F5'])
                r['both_support_edges'] = int(((diag['sJ'] > 0) & (diag['sC'] > 0)).sum())
                r['edges'] = [int((diag['sJ'] > 0).sum()), int((diag['sC'] > 0).sum()), len(EP)]
                r['forest_edges'] = int(len(diag['used']))
                r['terminals'] = [int(diag['tp'].sum()), int(diag['tn'].sum())]
                r['labels'] = [int((diag['lab'] == 1).sum()), int((diag['lab'] == -1).sum())]
    r['status'] = status
    if fb:
        out['F5+'] = A0
    else:
        with c.rc.pc.inference_only(b):
            out['F5+'] = produce.crf(b, out['F5']) if status == 'ok' else produce.crf(b, A0)
        torch.set_num_threads(2)
    gt, valid = c.truth(b)                      # read only after every label-free output is fixed
    r['gt'] = int(gt.sum())
    for a in ARMS + ('F5+',):
        r[a] = c.iu(out[a], gt, valid)
    if fb:
        reg = {'C0': np.zeros_like(A0), 'C1': np.zeros_like(A0), 'D': np.zeros_like(A0), 'fb': np.ones_like(A0)}
    else:
        C0, C1 = (U <= .5) & (M <= .5), (U > .5) & (M > .5)
        reg = {'C0': C0, 'C1': C1, 'D': ~(C0 | C1), 'fb': np.zeros_like(A0)}
    Av = A0 & valid
    for a in ('F5', 'FC', 'AC'):
        X = out[a] & valid
        add, rem = X & ~Av, Av & ~X
        r['edit_' + a] = {k: [int((add & gt & R_).sum()), int((add & ~gt & R_).sum()),
                              int((rem & gt & R_).sum()), int((rem & ~gt & R_).sum())] for k, R_ in reg.items()}
    if diag:
        t = gt.reshape(G, 16, G, 16).mean((1, 3)).ravel() >= .5
        u = diag['used']
        wj = diag['sJ'][u]
        r['wrong_join'] = float((wj * (t[EP[u]] != t[EQ[u]])).sum() / max(wj.sum(), 1e-12)) if len(u) else None
        r['term_acc'] = [int((diag['tp'] & t).sum()), int(diag['tp'].sum()), int((diag['tn'] & ~t).sum()), int(diag['tn'].sum())]
        r['reach'] = [int((t & (diag['lab'] == 1)).sum()), int(t.sum())]
        band = w37.band(w37.pool16(A0.astype(np.float64))).ravel()
        cfn = t & (C0.reshape(G, 16, G, 16).mean((1, 3)).ravel() >= .5) & ~band & (A0.reshape(G, 16, G, 16).mean((1, 3)).ravel() < .5)
        r['cfn_tokens'] = [int((cfn & (diag['lab'] == 1)).sum()), int(cfn.sum())]
    r['seconds'] = round(time.monotonic() - began, 2)
    return r


def produce_all():
    import torch
    import w31_demonstration as w31
    c.initialize()
    torch.set_num_threads(2)
    enc = Encoder()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'episodes.jsonl'
    done = {json.loads(l)['index'] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [b for b in w31.rows() if str(b['index']) not in done]
    print('todo', len(todo), flush=True)
    with path.open('a') as fh:
        for n, b in enumerate(todo, 1):
            r = one(b, enc)
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 25 == 0:
                print(n, len(todo), r['seconds'], flush=True)
    print('done', flush=True)


def attribution(D, arm):
    slots = {}
    for r in D:
        s = slots.setdefault((r['fold'], r['cls']), {'I': 0, 'U': 0, **{k: np.zeros(4) for k in REG}})
        s['I'] += r['A0'][0]; s['U'] += r['A0'][1]
        for k in REG:
            s[k] += r['edit_' + arm][k]
    out = {k: [] for k in REG}
    for s in slots.values():
        J = s['I'] / max(s['U'], 1)
        den = max(s['U'] + sum(s[k][1] - s[k][3] for k in REG), 1)
        for k in REG:
            a, b_, c_, d = s[k]
            out[k].append(100 * (a - c_ - J * (b_ - d)) / den)
    return {k: float(np.mean(v)) for k, v in out.items()}


def report():
    import score as sc
    import w37_coverage as w37
    R = [json.loads(l) for l in (OUT / 'episodes.jsonl').read_text().splitlines()]
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    w27 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w27_sh_single_line_S.jsonl.gz', 'rt'))}
    for r in R:
        r['SH+'] = w27[r['index']]['SH+']
    same = sum(r['A0'][:2] == w23[r['index']]['SH'][:2] for r in R)
    fbsame = all(r[a] == r['A0'] for r in R if r['fallback'] for a in ARMS)
    both = sum(r.get('both_support_edges', 0) for r in R)
    ok = [r for r in R if r['status'] == 'ok']
    doms = list(c.DOMAINS)
    by = {d: [r for r in R if r['dataset'] == d] for d in doms}
    st = {s: sum(r['status'] == s for r in R) for s in dict.fromkeys(r['status'] for r in R)}
    e = np.sum([r['edges'] for r in ok], axis=0)
    ta = np.sum([r['term_acc'] for r in ok], axis=0)
    rc = np.sum([r['reach'] for r in ok], axis=0); cf = np.sum([r['cfn_tokens'] for r in ok], axis=0)
    wj = [r['wrong_join'] for r in ok if r['wrong_join'] is not None]
    L = [f'n {len(R)}  checks: A0 == W23 SH {same}/{len(R)} (need >= 1280); fallback arms == A0 {fbsame}; '
         f'edges with both s_J > 0 and s_C > 0: {both} (need 0)',
         f'status {st}; FC whole-episode fallback {sum(bool(r.get("fc_fallback")) for r in ok)}; valid heads median {np.median([r["heads"] for r in ok]):.0f}',
         f'query edges: join {e[0] / e[2]:.1%}, cut {e[1] / e[2]:.1%}, unknown {1 - (e[0] + e[1]) / e[2]:.1%}; forest edges per episode median {np.median([r["forest_edges"] for r in ok]):.0f}',
         f'terminals per episode median +{np.median([r["terminals"][0] for r in ok]):.0f} / -{np.median([r["terminals"][1] for r in ok]):.0f}; '
         f'accuracy + {ta[0] / max(ta[1], 1):.1%}, - {ta[2] / max(ta[3], 1):.1%}',
         f'weighted wrong-join fraction: median {np.median(wj):.3f}, pooled mean {np.mean(wj):.3f}',
         f'reach: GT-foreground tokens labelled + {rc[0] / max(rc[1], 1):.1%}; off-band common-FN tokens labelled + {cf[0] / max(cf[1], 1):.1%} (n {cf[1]})',
         f'labels per episode median + {np.median([r["labels"][0] for r in ok]):.0f}, - {np.median([r["labels"][1] for r in ok]):.0f}']

    def ci(a, b_):
        v = np.mean([sc.miou(by[d], a) - sc.miou(by[d], b_) for d in doms])
        lo, hi = np.percentile(np.mean([w37._boot(sc, by[d], a, b_) for d in doms], axis=0), [2.5, 97.5])
        return v, lo, hi, sum(sc.miou(by[d], a) - sc.miou(by[d], b_) >= 0 for d in doms)
    cols = ('F5', 'F5lit', 'FC', 'FD', 'AC')
    L += ['', '## class mIoU minus A0 (pre-CRF); last column F5+ - SH+ (post-CRF)', f'{"":12s}' + ''.join(f'{a:>8s}' for a in cols) + '  F5+-SH+']
    for d in doms:
        L.append(f'{d[:11]:12s}' + ''.join(f'{sc.miou(by[d], a) - sc.miou(by[d], "A0"):+8.2f}' for a in cols) +
                 f'{sc.miou(by[d], "F5+") - sc.miou(by[d], "SH+"):+8.2f}')
    L.append(f'{"ten-domain":12s}' + ''.join(f'{np.mean([sc.miou(by[d], a) - sc.miou(by[d], "A0") for d in doms]):+8.2f}' for a in cols) +
             f'{np.mean([sc.miou(by[d], "F5+") - sc.miou(by[d], "SH+") for d in doms]):+8.2f}')
    L += ['', '## attribution by region (W39 A eq. 6), ten-domain', f'{"":6s}' + ''.join(f'{k:>8s}' for k in REG)]
    for a in ('F5', 'FC', 'AC'):
        at = [attribution(by[d], a) for d in doms]
        L.append(f'{a:6s}' + ''.join(f'{np.mean([x[k] for x in at]):+8.2f}' for k in REG))
    fmt = lambda t, v: f'{t} {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}] ({v[3]}/10 >= 0)'
    m = ci('F5+', 'SH+')
    at = [attribution(by[d], 'F5') for d in doms]
    common = np.mean([x['C0'] + x['C1'] for x in at])
    L += ['', 'reading 2: ' + fmt('F5+ - SH+', m) + f' -> method validation plan: {bool(m[0] >= 1 and m[1] > 0 and m[3] >= 7)}',
          'secondary: ' + '; '.join(fmt(t, ci(a, b_)) for t, a, b_ in (('FC - A0', 'FC', 'A0'), ('F5lit - A0', 'F5lit', 'A0'))),
          'reading 3: ' + '; '.join(fmt(t, ci(a, b_)) for t, a, b_ in (('F5 - A0', 'F5', 'A0'), ('F5 - FD', 'F5', 'FD'), ('F5 - AC', 'F5', 'AC')))
          + f'; common-region (C0 + C1) attribution of F5 {common:+.2f}',
          f'seconds per episode median {np.median([r["seconds"] for r in R]):.1f}']
    text = '\n'.join(L)
    (OUT / 'report.txt').write_text(text + '\n')
    print(text)
    keys = ('index', 'dataset', 'fold', 'cls', 'photo', 'gt', 'rho', 'fallback', 'status', 'heads', 'edges', 'forest_edges',
            'terminals', 'labels', 'both_support_edges', 'fc_fallback', 'wrong_join', 'term_acc', 'reach', 'cfn_tokens') \
        + ARMS + ('F5+', 'edit_F5', 'edit_FC', 'edit_AC')
    with gzip.open(ROOT / 'evidence/research/w44_relation_forest_S.jsonl.gz', 'wt') as f:
        for r in R:
            f.write(json.dumps({k: r[k] for k in keys if k in r}) + '\n')


if __name__ == '__main__':
    produce_all() if sys.argv[1] == 'produce' else report()
