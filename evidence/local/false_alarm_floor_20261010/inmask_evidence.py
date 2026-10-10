"""Which evidence separates target from false alarm inside the complete FoRIS mask, and does one rule hold on every dataset?

The false-alarm floor (floor_law.py) says class mIoU moves only if false-alarm area falls at nearly unchanged recall,
and that a rule which keeps a share h of the hits and k of the false alarms helps a dataset only when
(1 - h) / (1 - k) is below that dataset's false-alarm share of the union, sum F / (sum g + sum F).
This instrument measures exactly that for a fixed list of evidence fields, on stored tokens, before any pipeline is built:

  separation   inside the sealed complete FoRIS mask: AUC, and the false-alarm mass kept when 95% of the hits are kept
               (kept95 <= 0.5 halves the false alarms for five points of recall; about AUC 0.88)
  rule         the mask "FoRIS restricted to tokens whose evidence is above its level-free point", scored exactly in
               the stored frame as class mIoU against complete FoRIS, with an episode bootstrap
  the floor    the (target area, recall, false alarm) row of floor_law.py for the same episodes
  misses       whether the same field ranks missed target above true background outside the mask

Every field uses reference tokens, the complete reference mask, query tokens and the sealed FoRIS mask only. All
fields of a source are written and sealed before a query mask is opened. Nothing is encoded.

  python3 evidence/local/false_alarm_floor_20261010/inmask_evidence.py selftest
  ... probe    --cohort a/<cohort>                       what was detected for the first episodes; opens no query mask
  ... fields   --cohort a/<cohort> [--foris a/<folder with the FoRIS masks>] [--limit 600] [--shard 0/6]
  ... evaluate --cohort a/<cohort> [--foris ...]
  ... summary  --design <name> <name> --confirm <name> <name>              (names are folders under inmask/)

--cohort is a manifest cohort: tokens come from the input-addressed raw cache (O/24, 4096 x 1024, FP32) through the
repository's own RawFeatureCache and load_inputs; masks and score fields are read from where the cohort stored them.
--run reads an output folder of scripts/run_m4_baselines.py infer instead (features/ per episode).
Both readers were run end to end on the writing machine against fabricated folders in the real layouts, the cohort
one through the real RawFeatureCache and load_inputs; neither has met the real cache yet, so run probe first.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '2')
from collections import defaultdict
from contextlib import contextmanager
import argparse
import builtins
import hashlib
import json
import io
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SHARED_PROFILE = 'a/paco_mean200_20261008/raw_cache/0a555915a7972480b74fcce11c876a79c4e776c93db8573d31f8b0e01c1c6158/profile.json'
ZERO = lambda name: .5 if 'back_auc' in name else 0.          # the level-free decision point of a field
# Not eligible as a rule: a foreign reference is outside the task, FoRIS's score and one-sided similarity have no zero.
DIAGNOSTIC = ('diag.', 'foris_score', 'fg_peak')


def read(path): return json.loads(Path(path).read_text())
def lines(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = Path(str(path) + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def token_mass(binary, grid=64):
    """Pixel count of a mask in each of the grid x grid tokens of the squashed square input, for any stored frame."""
    h, w = binary.shape
    index = (np.arange(h) * grid // h)[:, None] * grid + (np.arange(w) * grid // w)[None, :]
    return np.bincount(index.ravel(), weights=binary.ravel().astype(np.float64), minlength=grid * grid)


def reference_coverage(mask, grid=64, side=1024):
    """Lawful reference labels in the actual square encoder's16pixel footprints.

    Original-frame token_mass is appropriate for scoring stored pixels, but is
    not the encoder's supervision when the source is smaller than the grid.
    """
    assert side % grid == 0
    tensor = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, None]
    canvas = F.interpolate(tensor, (side, side), mode='nearest')[0, 0]
    cell = side // grid
    return canvas.reshape(grid, cell, grid, cell).mean((1, 3)).reshape(-1)


@contextmanager
def no_query_truth(source):
    """Enforce the plan's prediction/probe boundary, including indirect opens."""
    denied = set()
    for row in source.rows:
        if source.kind == 'run':
            denied.add(str((source.assets / 'episodes/claude_packs' / row['pack'] / 'ann' / row['query']).resolve()))
        for key in ('query_mask_path', 'query_ignore_mask_path'):
            if key in row:
                path = Path(row[key])
                denied.update((str(path.resolve()), str((source.assets / path).resolve())))
    original = builtins.open, io.open
    def guarded(opener):
        def opened(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and str(Path(os.fsdecode(file)).resolve()) in denied:
                raise PermissionError('Query truth is forbidden during probe/field generation')
            return opener(file, *args, **kwargs)
        return opened
    builtins.open, io.open = (guarded(opener) for opener in original)
    try:
        yield
    finally:
        builtins.open, io.open = original


def official_transform(assets, side=1024):
    """The official preprocessing, loaded by path so that no other module named utils can shadow it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('_foris_official_data', Path(assets) / 'third_party/foris_official/utils/data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_transform(side)


# ---------------------------------------------------------------- evidence: no query label below this line

def minmax(x): return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-12)


def target_tokens(coverage):
    target = coverage >= .5
    if target.sum() < 4:
        target = torch.zeros_like(target)
        target[coverage.topk(4).indices] = True
    return target


def ridge(R, positive, negative, lam=.01, cap=256):
    """Role-balanced ridge between two sets of reference tokens (dual form, as in study.py)."""
    pick = lambda s: torch.nonzero(s)[:, 0][torch.linspace(0, int(s.sum()) - 1, min(int(s.sum()), cap)).round().long()]
    p, n = pick(positive), pick(negative)
    x = torch.cat((R[p], R[n])).double()
    y = torch.cat((torch.ones(len(p)), -torch.ones(len(n)))).double()
    w = torch.cat((torch.full((len(p),), .5 / len(p)), torch.full((len(n),), .5 / len(n)))).double()
    mx, my = (w[:, None] * x).sum(0), (w * y).sum()
    xc = x - mx
    coefficient = xc.T @ torch.linalg.solve(xc @ xc.T + torch.diag(lam / w), y - my)
    return coefficient, my - mx @ coefficient


def back_auc(S, positive, negative):
    """Each query token as a probe on the labelled image: does it rank the reference target above the negatives?"""
    p, n = int(positive.sum()), int(negative.sum())
    assert p > 0 and n > 0
    result = torch.empty(S.shape[1], dtype=torch.float64)
    # Independent query columns; chunking bounds temporary sorting memory.
    for begin in range(0, S.shape[1], 512):
        selected = torch.cat((S[positive, begin:begin+512], S[negative, begin:begin+512]))
        value, order = selected.sort(dim=0)
        rows = torch.arange(1, p+n+1)[:, None].expand_as(order)
        first = torch.cat((torch.ones_like(value[:1], dtype=torch.bool), value[1:] != value[:-1]))
        last = torch.cat((value[:-1] != value[1:], torch.ones_like(value[:1], dtype=torch.bool)))
        low = torch.where(first, rows, 0).cummax(dim=0).values
        high = torch.where(last, rows, p+n+1).flip(0).cummin(dim=0).values.flip(0)
        mean_rank = (low+high).double()*.5
        sum_positive_rank = (mean_rank*(order < p)).sum(0)
        result[begin:begin+512] = (sum_positive_rank-p*(p+1)/2)/(p*n)
    return result


def clusters(X, k=8, rounds=10):
    """Deterministic cosine k-means of unit rows, farthest-point start."""
    centre = [F.normalize(X.mean(0), dim=0)]
    for _ in range(k - 1):
        centre.append(X[(X @ torch.stack(centre).T).max(1).values.argmin()])
    C = torch.stack(centre)
    for _ in range(rounds):
        label = (X @ C.T).argmax(1)
        C = torch.stack([F.normalize(X[label == j].mean(0), dim=0) if (label == j).any() else C[j] for j in range(k)])
    return (X @ C.T).argmax(1)


def evidence(R, Q, coverage, candidate, raw=None, foreign=None, extra=None, temperature=.07):
    """R, Q: unit token rows of reference and query. coverage: reference mask share per token. candidate: FoRIS mask tokens.

    raw: the same two images before positional debiasing, used only with `foreign`, the raw reference tokens and
    coverage of another episode of a different class (a diagnostic of what lights up for any concept; not a method).
    """
    target = target_tokens(coverage)
    background = (coverage < .05) & ~target
    info = dict(target=int(target.sum()), background=int(background.sum()), candidate=int(candidate.sum()))
    if background.sum() < 4 or candidate.sum() < 1:
        return {}, dict(info, skipped=True)
    S = R @ Q.T
    peak = S[target].max(0).values
    out = {'fg_peak': peak, 'two_sided': peak - S[background].max(0).values}
    for name, value in (extra or {}).items():
        out[name] = value

    # FoRIS's own negatives: the fifth of the background most similar to the target mean.
    similarity = R @ F.normalize((R * coverage[:, None]).sum(0), dim=0)
    hard = background & (similarity >= similarity[background].topk(max(4, int(.2 * background.sum()))).values[-1])
    # Negatives found on the labelled side: background tokens that the query candidate explains at least as well as
    # it explains the median target token. Empty when the candidate confuses nothing in the reference (open world).
    explained = S[:, candidate].max(1).values
    confusable = background & (explained >= explained[target].median())
    info.update(hard=int(hard.sum()), confusable=int(confusable.sum()), skipped=False)

    out['two_sided_hard'] = peak - S[hard].max(0).values
    out['back_auc'] = back_auc(S, target, background)
    share = torch.softmax(S / temperature, dim=1)              # every reference token spends one unit on the query
    out['reverse'] = torch.log(share[target].mean(0) + len(Q) ** -2.) - torch.log(share[background].mean(0) + len(Q) ** -2.)
    for name, negative in (('all', background), ('hard', hard)):
        coefficient, bias = ridge(R, target, negative)
        out['ridge_' + name] = Q.double() @ coefficient + bias

    # Decoy concepts from the same reference photo: each background cluster is matched like a target and read in its
    # own relative scale on this query, as FoRIS reads the target. Kept: tokens that answer the target more.
    index = torch.nonzero(background)[:, 0]
    label = clusters(R[index])
    decoy = [minmax(S[index[label == j]].max(0).values) for j in range(8) if (label == j).sum() >= 4]
    if decoy:
        out['decoy_relative'] = minmax(peak) - torch.stack(decoy).max(0).values
    # The query's own structure: side of the candidate centroid on which its best-matched tokens lie.
    inside = torch.nonzero(candidate)[:, 0]
    seeds = inside[peak[inside].topk(min(len(inside), max(4, len(inside) // 10))).indices]
    Qc = F.normalize(Q - Q[candidate].mean(0), dim=1)
    out['seed_centred'] = Qc @ F.normalize(Qc[seeds].mean(0), dim=0)

    if confusable.sum() >= 4:
        out['two_sided_back'] = peak - S[confusable].max(0).values
        out['back_auc_hard'] = back_auc(S, target, confusable)
        coefficient, bias = ridge(R, target, confusable)
        out['ridge_back'] = Q.double() @ coefficient + bias
        # The same three in a frame centred on what is being told apart: target and confusables in the reference,
        # the candidate in the query. The shared host component is removed in each image by its own tokens.
        Rc = F.normalize(R - .5 * (R[target].mean(0) + R[confusable].mean(0)), dim=1)
        Sc = Rc @ Qc.T
        out['centred.two_sided_back'] = Sc[target].max(0).values - Sc[confusable].max(0).values
        out['centred.back_auc_hard'] = back_auc(Sc, target, confusable)
        coefficient, bias = ridge(Rc, target, confusable)
        out['centred.ridge_back'] = Qc.double() @ coefficient + bias

    if foreign is not None:
        Rr, Qr = raw if raw is not None else (R, Q)
        other = minmax((foreign[0][target_tokens(foreign[1])] @ Qr.T).max(0).values)
        out['diag.foreign_contrast'] = minmax((Rr[target] @ Qr.T).max(0).values) - other
        out['aux.foreign_response'] = other
    out['aux.own_response'] = minmax(peak)
    return {k: v.double().numpy() for k, v in out.items()}, info


# ---------------------------------------------------------------- instrument: query label enters here only

def curve(score, positive, negative):
    """Cumulative positive and negative mass at the end of every tie group, highest score first."""
    keep = positive + negative > 0
    s, p, n = score[keep], positive[keep], negative[keep]
    order = np.argsort(-s, kind='stable')
    s, p, n = s[order], p[order], n[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1] if len(s) else np.zeros(0, int)
    return np.cumsum(p)[ends], np.cumsum(n)[ends], float(p.sum()), float(n.sum())


def area(tp, fp, P, N):
    dp, dn = np.diff(np.r_[0., tp]), np.diff(np.r_[0., fp])
    return float(np.sum(dp * (N - fp + .5 * dn)) / (P * N))


def separation(score, hit, false_alarm):
    """Inside the mask, weighted by pixel mass. kept95: false-alarm mass kept at the level keeping >= 95% of the hits."""
    tp, fp, P, N = curve(score, hit, false_alarm)
    if not P or not N:
        return dict(auc=None)
    at = lambda level: float(fp[np.searchsorted(tp, level * P - 1e-9)])
    return dict(auc=area(tp, fp, P, N), kept95=at(.95) / N, kept90=at(.90) / N, kept95_mass=at(.95), kept90_mass=at(.90), false_alarm=N)


def recovery(score, missed, rest):
    """Outside the mask: missed target against true background; share of the misses above the level admitting 1% of the rest."""
    tp, fp, P, N = curve(score, missed, rest)
    if not P or not N:
        return dict(miss_auc=None)
    return dict(miss_auc=area(tp, fp, P, N), miss_at_1pct=float(np.max(np.r_[0., tp[fp <= .01 * N]])), missed=P)


def rule(score, zero, predicted):
    """Tokens kept by the level-free rule. A rule that would leave under a tenth of the mask abstains."""
    keep = score > zero
    return keep if predicted[keep].sum() >= .1 * predicted.sum() else np.ones_like(keep)


def macro(I, U, cls):
    index = np.unique(cls, return_inverse=True)[1]
    return float(100 * np.mean(np.bincount(index, I) / np.maximum(np.bincount(index, U), 1)))


def interval(I0, U0, I1, U1, cls, draws=1000, seed=0):
    """Episode bootstrap of the class mIoU difference (rule minus FoRIS)."""
    rs, index = np.random.RandomState(seed), np.unique(cls, return_inverse=True)[1]
    k, values = index.max() + 1, []
    for _ in range(draws):
        s = rs.randint(0, len(cls), len(cls))
        c = index[s]
        seen = np.bincount(c, minlength=k) > 0
        ratio = lambda I, U: (np.bincount(c, I[s], k)[seen] / np.maximum(np.bincount(c, U[s], k)[seen], 1)).mean()
        values.append(100 * (ratio(I1, U1) - ratio(I0, U0)))
    return [float(v) for v in np.quantile(values, [.025, .975])]


def rank_correlation(a, b):
    try:
        from scipy.stats import spearmanr
        value = float(spearmanr(a, b)[0])
        return value if np.isfinite(value) else None
    except ImportError:
        return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def fixtures():
    one = separation(np.array([3., 2., 1., 0.]), np.array([1., 1., 0., 0.]), np.array([0., 0., 1., 1.]))
    assert one['auc'] == 1 and one['kept95'] == 0
    assert separation(np.array([1., 1.]), np.array([3., 0.]), np.array([0., 7.])) ['auc'] == .5
    half = separation(np.array([4., 3., 2., 1.]), np.array([1., 0., 1., 0.]), np.array([0., 1., 0., 1.]))
    assert half['auc'] == .75 and half['kept95'] == .5
    assert separation(np.array([1.]), np.array([0.]), np.array([2.]))['auc'] is None
    found = recovery(np.arange(200.)[::-1], np.r_[1., np.zeros(199)], np.r_[0., np.ones(199)])
    assert found['miss_auc'] == 1 and found['miss_at_1pct'] == 1
    assert rule(np.array([1., -1.]), 0., np.array([5., 5.])).tolist() == [True, False]
    assert rule(np.array([1., -1.]), 0., np.array([0., 9.])).tolist() == [True, True]          # would empty the mask: abstain
    picture = np.zeros((128, 192), bool); picture[:2, :3] = True                                # one token is 2 x 3 pixels here
    assert token_mass(picture)[0] == 6 and token_mass(picture).sum() == 6 and token_mass(np.ones((50, 70), bool)).sum() == 3500
    assert token_mass(np.ones((96, 96), bool), 48).tolist() == [4.] * 2304
    assert abs(macro(np.array([1., 1., 2.]), np.array([2., 2., 2.]), np.array(['a', 'a', 'b'])) - 75) < 1e-9
    low, high = interval(*(np.array(v, float) for v in ([1, 1, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2])), np.array(['a', 'a', 'b']), draws=50)
    assert 0 <= low <= high <= 50


def selftest():
    """Planted scene: a part (target) on a host, the host's other part in both images, a look-alike only in the query."""
    fixtures()
    g = torch.Generator().manual_seed(0)
    unit = lambda: F.normalize(torch.randn(1024, generator=g), dim=0)
    host, part, other, alike, ground, ground_q, shift, elsewhere = (unit() for _ in range(8))
    def tokens(n, *directions):                               # token noise as large as the signal; the host dominates the part
        return sum(directions) + .9 * torch.randn(n, 1024, generator=g) / 32
    R = F.normalize(torch.cat((tokens(120, host, .3 * part), tokens(400, host, .3 * other), tokens(3576, ground))), dim=1)
    coverage = torch.cat((torch.ones(120), torch.zeros(3976)))
    Q = torch.cat((tokens(100, host, .3 * part), tokens(300, host, .3 * other), tokens(150, host, .2 * part, .3 * alike), tokens(3546, ground_q)))
    Q = F.normalize(Q + .4 * shift, dim=1)                    # the query photo has its own offset
    candidate = torch.zeros(4096, dtype=torch.bool); candidate[:550] = True
    foreign = (F.normalize(tokens(4096, elsewhere), dim=1), torch.cat((torch.ones(200), torch.zeros(3896))))
    fields, info = evidence(R, Q, coverage, candidate, foreign=foreign, extra={'foris_score': Q @ R[:120].mean(0)})
    assert not info['skipped'] and info['target'] == 120 and info['confusable'] >= 4, info
    assert all(np.isfinite(v).all() and v.shape == (4096,) for v in fields.values())
    assert len(set(clusters(R[120:]).tolist())) >= 2
    hit = np.r_[np.full(100, 256.), np.zeros(3996)]
    parent, lookalike = (np.zeros(4096) for _ in range(2)); parent[100:400] = 256; lookalike[400:550] = 256
    predicted = hit + parent + lookalike
    print(f"   reference tokens: target {info['target']}, hard {info['hard']}, confusable {info['confusable']} (planted host part: 400)")
    table = {}
    for k, v in fields.items():
        if k.startswith('aux.'):
            continue
        keep = rule(v, ZERO(k), predicted)
        table[k] = a, b = separation(v, hit, parent), separation(v, hit, lookalike)
        a.update(hits=hit[keep].sum() / hit.sum(), parents=parent[keep].sum() / parent.sum())
        print(f"   {k:24s} host part: AUC {a['auc']:.3f} kept95 {a['kept95']:.2f}, rule keeps {a['hits']:.2f} of hits and {a['parents']:.2f} of false alarms"
              f" | look-alike absent from the reference: AUC {b['auc']:.3f} kept95 {b['kept95']:.2f}")
    # The host's other part lies in the reference background, so reference negatives must tell it apart better than
    # one-sided similarity does; the look-alike has no reference negative.
    assert table['fg_peak'][0]['auc'] <= table['two_sided_back'][0]['auc'] <= table['ridge_back'][0]['auc']
    assert table['ridge_back'][0]['kept95'] < .1 and table['centred.ridge_back'][0]['kept95'] < .1
    for k in ('two_sided_back', 'back_auc_hard', 'centred.back_auc_hard'):     # the level-free rule removes the host part
        assert table[k][0]['hits'] > .95 and table[k][0]['parents'] < .1, k
    print('selftest passed')


# ---------------------------------------------------------------- sources

class Run:
    """Output folder of scripts/run_m4_baselines.py infer: manifest.json, predictions/ and features/ (q, r, cov; q_raw, r_raw, score if saved)."""
    kind = 'run'

    def __init__(self, args):
        self.folder, self.assets, self.dataset = Path(args.run), Path(args.assets), args.dataset
        self.where, self.rows = str(self.folder), read(self.folder / 'manifest.json')

    def describe(self, row):
        return dict(id=row['pack'] + '__' + row['key'], cls=f"{row['pack']}:{row['c']}",
                    dataset=self.dataset or re.sub(r'_f\d+$', '', row['pack']))

    def _load(self, row):
        with np.load(self.folder / 'features' / (self.describe(row)['id'] + '.npz'), allow_pickle=False) as z:
            unit = lambda k, other: F.normalize(torch.from_numpy(z[k if k in z.files else other].astype(np.float32)), dim=1)
            score = z['score'].astype(np.float32).reshape(-1) if 'score' in z.files else np.zeros(0, np.float32)
            return unit('r', 'r_raw'), unit('q', 'q_raw'), unit('r_raw', 'r'), unit('q_raw', 'q'), torch.from_numpy(z['cov'].astype(np.float32).reshape(-1)), score

    def tokens(self, row):
        R, Q, Rr, Qr, coverage, score = self._load(row)
        return R, Q, Rr, Qr, coverage, ({'foris_score': torch.from_numpy(score)} if score.size == len(Q) else {})

    def reference(self, row):
        _, _, Rr, _, coverage, _ = self._load(row)
        return Rr, coverage

    def prediction(self, row):
        with np.load(self.folder / 'predictions' / (self.describe(row)['id'] + '.npz'), allow_pickle=False) as z:
            h, w = (int(v) for v in z['original_hw'])
            key = next((k for k in ('foris.crf', 'foris.bilinear') if k in z.files), None)
            assert key, f'No complete FoRIS mask among {z.files}'
            return np.unpackbits(z[key], count=h * w).reshape(h, w).astype(bool)

    def truth(self, row, shape):
        from PIL import Image
        with Image.open(self.assets / 'episodes/claude_packs' / row['pack'] / 'ann' / row['query']) as im:
            value = np.asarray(im.convert('L')) > 0
        assert value.shape == tuple(shape), 'Annotation and stored prediction differ in size'
        return value, None

    def images(self, row):
        """Reference photograph, its complete mask, query photograph (the three legal inputs)."""
        from PIL import Image
        base = self.assets / 'episodes/claude_packs' / row['pack']
        with Image.open(base / 'data' / row['support']) as im: reference = im.convert('RGB')
        with Image.open(base / 'data' / row['query']) as im: query = im.convert('RGB')
        with Image.open(base / 'ann' / row['support']) as im: mask = np.asarray(im.convert('L')) > 0
        return reference, mask, query

    def inspect(self):
        with np.load(self.folder / 'features' / (self.describe(self.rows[0])['id'] + '.npz'), allow_pickle=False) as z:
            features = {k: list(z[k].shape) for k in z.files}
        with np.load(self.folder / 'predictions' / (self.describe(self.rows[0])['id'] + '.npz'), allow_pickle=False) as z:
            return dict(features=features, prediction_keys=list(z.files))


class Cohort:
    """Manifest cohort: tokens from the input-addressed raw cache, complete FoRIS masks from a prediction folder.

    Layouts met so far, all detected:
      folder       the cohort folder itself or its run/ subfolder
      index        inference.jsonl (lines) or inference.json (list); file name under 'filename' or 'prediction_file'
      mask         'original/foris.crf' (size from the file or the row), else 'cli/foris.crf' or 'cli1024/foris.crf' (1024 canvas)
      raw profile  --profile, else config.json raw_profile | profile_path, else raw_cache.json profile_path, else the shared one
      score field  <foris>/fields/<file> 'score', else source_score_index.json; optional, only a comparison field
      query mask   query_mask_path, nearest-resized to the prediction frame; query_ignore_mask_path is excluded from counts
    --foris names another folder when the FoRIS masks of these episodes were written by a different run.
    """
    kind = 'cohort'

    def __init__(self, args):
        self.assets, self.dataset, self.branch = Path(args.assets), args.dataset, args.branch
        self.folder = self._root(args.cohort, 'manifest.json')
        self.foris = self._root(args.foris, 'predictions') if args.foris else self.folder
        self.where, self.rows = str(self.folder), read(self.folder / 'manifest.json')
        if (self.foris / 'inference.jsonl').exists():
            records = lines(self.foris / 'inference.jsonl')
        elif (self.foris / 'inference.json').exists():
            records = read(self.foris / 'inference.json')
        else:
            raise FileNotFoundError(f'No inference.jsonl or inference.json in {self.foris}; name the folder of the FoRIS masks with --foris')
        self.index = {r['episode_id']: r for r in records}
        assert len(self.index) == len(records), 'Duplicate episode IDs in the FoRIS index'
        absent = [r['episode_id'] for r in self.rows if r['episode_id'] not in self.index]
        assert not absent, f'{len(absent)} manifest episodes have no FoRIS mask in {self.foris} (first {absent[:3]})'
        self.profile = self._profile(args.profile)
        if (self.foris / 'manifest.json').exists():
            paired = {r['episode_id']: r for r in read(self.foris / 'manifest.json')}
            for row in self.rows:
                for key in ('reference_rgb_hash', 'query_rgb_hash', 'reference_mask_hash', 'query_mask_hash'):
                    assert row[key] == paired[row['episode_id']][key], f'Cross-run input differs: {row["episode_id"]}/{key}'
        self.scores = read(self.foris / 'source_score_index.json') if (self.foris / 'source_score_index.json').exists() else {}
        self.ready, self.memo = False, {}
        self.frame_by_id = {}

    def _root(self, name, marker):
        root = Path(name) if Path(name).is_absolute() else self.assets / name
        return root / 'run' if not (root / marker).exists() and (root / 'run' / marker).exists() else root

    def _profile(self, given):
        found = [given]
        for folder in (self.folder, self.foris, self.folder.parent, self.foris.parent):
            for name, keys in (('config.json', ('raw_profile', 'profile_path')), ('raw_cache.json', ('profile_path',))):
                if (folder / name).exists():
                    document = read(folder / name)
                    found += [document.get(k) for k in keys] if isinstance(document, dict) else []
        found.append(self.assets / SHARED_PROFILE)
        for path in found:
            if path:
                options = [Path(path)] if Path(path).is_absolute() else [self.assets / path, Path(path)]
                for option in options:
                    if option.is_file():
                        return option.resolve()
        raise FileNotFoundError('No raw-cache profile.json found for this cohort; pass --profile')

    def _prepare(self):
        if self.ready:
            return
        sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
        from ics.official_data import load_inputs, decoded_rgb, array_hash
        from raw_feature_cache import RawFeatureCache
        self.load_inputs, self.decoded_rgb, self.array_hash = load_inputs, decoded_rgb, array_hash
        self.cache = RawFeatureCache(self.profile.parent.parent, read(self.profile))
        assert self.cache.folder == self.profile.parent, 'profile.json does not sit in the folder named by its own hash'
        profile = self.cache.profile
        assert profile['model'] == 'DINOv3-L/16' and profile['storage_dtype'] == 'float32'
        assert profile['preprocessing']['resize'] == [1024, 1024]
        assert sha(self.assets / 'third_party/foris_official/utils/data.py') == profile['preprocessing']['source_sha256']
        for folder in (self.folder, self.foris, self.folder.parent, self.foris.parent):
            if (folder / 'config.json').exists():
                config = read(folder / 'config.json')
                expected = config.get('weights_sha256')
                assert expected is None or expected == profile['weights_sha256'], 'Baseline/cache model identity differs'
                expected = config.get('raw_profile_sha256', config.get('profile_sha256'))
                assert expected is None or expected == sha(self.profile), 'Declared/shared cache profile differs'
        self.transform = official_transform(self.assets)
        if self.branch == 'O/24':
            basis = torch.load(self.assets / 'native_assets/positional_basis.pt', map_location='cpu', weights_only=True)['basis'].float()
            self.projection = torch.eye(1024) - basis @ basis.T
        self.ready = True

    def describe(self, row):
        return dict(id=row['episode_id'], cls=f"{row.get('fold')}:{row.get('loader_class_id', row.get('class_id'))}",
                    dataset=self.dataset or row.get('dataset', self.folder.name))

    def _raw(self, image, identity):
        """Unit rows of the cached branch for one transformed input; raises when the input was never encoded."""
        if identity not in self.memo:
            x = self.transform(image).numpy()
            assert (self.cache.folder / self.cache.key(x) / 'entry.json').exists(), 'This input has no entry in the raw cache; the instrument adds no encoding'
            array = self.cache.read(x, (self.branch,))[self.branch]
            assert array.ndim == 2 and len(array) == 4096 and array.dtype == np.float32, (array.shape, array.dtype)
            if len(self.memo) >= 6:
                self.memo.pop(next(iter(self.memo)))
            self.memo[identity] = F.normalize(torch.from_numpy(array.astype(np.float32)), dim=1)
        return self.memo[identity]

    def _mask(self, row, role):
        from PIL import Image
        path = Path(row[role + '_mask_path'])
        with Image.open(path if path.exists() else self.assets / path) as im:
            value = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
        if role + '_mask_hash' in row:                        # the same identity check as the repository's loaders
            h = hashlib.sha256(json.dumps([list(value.shape), value.dtype.str]).encode())
            h.update(np.ascontiguousarray(value).tobytes())
            assert h.hexdigest() == row[role + '_mask_hash'], f'{role} mask changed'
        return value > 0

    def _coverage(self, row):
        value = self._mask(row, 'reference')
        return reference_coverage(value)

    def _file(self, row):
        record = self.index[row['episode_id']]
        name = record.get('filename', record.get('prediction_file'))
        assert name, f'The index record names no prediction file: {sorted(record)}'
        path = self.foris / 'predictions' / name
        assert 'prediction_sha256' not in record or sha(path) == record['prediction_sha256'], 'Prediction changed after sealing'
        return path

    def _score(self, row):
        """FoRIS's own score field where the cohort stored one (64 x 64); None otherwise."""
        try:
            item = self.scores.get(row['episode_id'])
            path, key = (Path(item['path']), item.get('score_key', 'score')) if item else (self.foris / 'fields' / self._file(row).name, 'score')
            with np.load(path, allow_pickle=False) as z:
                value = z[key].astype(np.float32).reshape(-1)
            return torch.from_numpy(value) if value.size == 4096 else None
        except (OSError, KeyError, ValueError):
            return None

    def tokens(self, row):
        self._prepare()
        reference, _, query = self.load_inputs(row, self.assets)
        Rr, Qr = self._raw(reference, row['reference_rgb_hash']), self._raw(query, row['query_rgb_hash'])
        coverage, R, Q = self._coverage(row), Rr, Qr
        if self.branch == 'O/24':                             # the official positional-debias branch of FoRIS Part 1
            inside = F.interpolate(torch.from_numpy(self._mask(row, 'reference').astype(np.float32))[None, None], (64, 64), mode='nearest')[0, 0].reshape(-1) > .5
            mean_q = Qr.mean(0)
            semantic = float(F.normalize(Rr[inside].mean(0), dim=0) @ mean_q / (mean_q.norm() + 1e-6)) if inside.any() else None
            if semantic is None or semantic < .8:
                R, Q = F.normalize(Rr @ self.projection.T, dim=1), F.normalize(Qr @ self.projection.T, dim=1)
        score = self._score(row)
        return R, Q, Rr, Qr, coverage, ({} if score is None else {'foris_score': score})

    def reference(self, row):
        self._prepare()
        if row['reference_rgb_hash'] not in self.memo:
            image = self.decoded_rgb(self.assets / row['reference_path'], row.get('reference_crop'))
            assert self.array_hash(np.asarray(image)) == row['reference_rgb_hash'], 'Frozen RGB input changed'
            self._raw(image, row['reference_rgb_hash'])
        return self.memo[row['reference_rgb_hash']], self._coverage(row)

    def images(self, row):
        self._prepare()
        reference, mask, query = self.load_inputs(row, self.assets)
        return reference, np.asarray(mask, dtype=bool), query

    def prediction(self, row):
        with np.load(self._file(row), allow_pickle=False) as z:
            if 'original/foris.crf' in z.files:
                self.frame_by_id[row['episode_id']] = 'original'
                h, w = (int(v) for v in (z['original_hw'] if 'original_hw' in z.files else row['query_size_hw']))
                return np.unpackbits(z['original/foris.crf'], count=h * w).reshape(h, w).astype(bool)
            key = next((k for k in ('cli/foris.crf', 'cli1024/foris.crf') if k in z.files), None)
            assert key, f'No complete FoRIS mask among {z.files}'
            self.frame_by_id[row['episode_id']] = 'cli1024'
            return np.unpackbits(z[key], count=1024 ** 2).reshape(1024, 1024).astype(bool)

    def truth(self, row, shape):
        def frozen(role):                                     # direct nearest resize, as load_query_ground_truth
            value = self._mask(row, role)
            if value.shape != tuple(shape):
                value = F.interpolate(torch.from_numpy(value.astype(np.float32))[None, None], tuple(shape), mode='nearest')[0, 0].numpy() > .5
            return value
        return frozen('query'), frozen('query_ignore') if 'query_ignore_mask_path' in row else None

    def inspect(self):
        self._prepare()
        first = []
        for row in self.rows[:3]:
            reference, _, query = self.load_inputs(row, self.assets)
            item = dict(episode_id=row['episode_id'])
            for role, image in (('reference', reference), ('query', query)):
                entry = self.cache.folder / self.cache.key(self.transform(image).numpy()) / 'entry.json'
                item[role + '_cached'] = sorted(read(entry)['features']) if entry.exists() else 'NOT IN THE RAW CACHE'
            with np.load(self._file(row), allow_pickle=False) as z:
                item['prediction_keys'] = list(z.files)
            item.update(prediction_frame=list(self.prediction(row).shape), score_field=self._score(row) is not None)
            first.append(item)
        datasets = defaultdict(int)
        for row in self.rows:
            datasets[self.describe(row)['dataset']] += 1
        return dict(foris=str(self.foris), profile=str(self.profile), branch=self.branch, datasets=dict(datasets),
                    index_keys=sorted(next(iter(self.index.values()))), first=first)


def opened(args):
    assert bool(args.run) != bool(args.cohort), 'Give exactly one of --run and --cohort'
    source = Run(args) if args.run else Cohort(args)
    return source, ROOT / 'inmask' / (args.name or Path(source.where).name)


def probe(args):
    """What this source looks like to the readers, for the first episodes. Opens no query mask and writes nothing."""
    source, out = opened(args)
    with no_query_truth(source):
        print(json.dumps(dict(kind=source.kind, where=source.where, output=str(out), episodes=len(source.rows),
                              row_keys=sorted(source.rows[0]), **source.inspect()), indent=2, ensure_ascii=False))


def standard_fields(source, rows, described, i):
    R, Q, Rr, Qr, coverage, extra = source.tokens(rows[i])
    other = next((j for j in ((i + s) % len(rows) for s in range(1, len(rows))) if described[j]['cls'] != described[i]['cls']), None)
    fields, note = evidence(R, Q, coverage, candidate_tokens(source.prediction(rows[i])), raw=(Rr, Qr), extra=extra,
                            foreign=source.reference(rows[other]) if other is not None else None)
    return fields, dict(note, foreign=None if other is None else described[other]['id'])


def candidate_tokens(mask, grid=64):
    """Tokens at least half inside the stored FoRIS mask (any touched token if none is)."""
    share = token_mass(mask, grid) / np.maximum(token_mass(np.ones(mask.shape, bool), grid), 1)
    return torch.from_numpy(share >= .5 if (share >= .5).any() else share > 0)


def make_fields(args, compute=standard_fields, grid=64):
    source, out = opened(args)
    (out / 'fields').mkdir(parents=True, exist_ok=True); (out / 'records').mkdir(exist_ok=True)
    assert not (out / 'sealed.json').exists(), 'Sealed fields are not extended; use --name for a new folder'
    rows = source.rows[:args.limit] if args.limit else source.rows
    described = [source.describe(r) for r in rows]
    plan = dict(kind=source.kind, where=source.where, branch=getattr(source, 'branch', 'O/24 after FoRIS Part 1'), grid=grid, ids=[d['id'] for d in described],
                instrument_sha256=sha(__file__), generator_sha256=sha(sys.argv[0]),
                generator_path=str(Path(sys.argv[0]).resolve()), threads=args.threads,
                exposure='Previously exposed development queues; confirm means development transfer verification, not independent confirmation')
    if source.kind == 'cohort':
        plan.update(foris=str(source.foris), profile=str(source.profile), profile_sha256=sha(source.profile),
                    manifest_sha256=sha(source.folder / 'manifest.json'),
                    foris_index_sha256=sha(source.foris / ('inference.jsonl' if (source.foris / 'inference.jsonl').exists() else 'inference.json')),
                    dependency_sha256={rel: sha(REPO / rel) for rel in ('scripts/raw_feature_cache.py', 'src/ics/official_data.py')},
                    transform_sha256=sha(source.assets / 'third_party/foris_official/utils/data.py'),
                    basis_sha256=sha(source.assets / 'native_assets/positional_basis.pt'))
    if (out / 'config.json').exists():
        assert read(out / 'config.json') == plan, 'This folder was started with other episodes; use --name'
    else:
        write(out / 'config.json', plan)
    shard, of = (int(v) for v in args.shard.split('/'))
    version, begin, done = sha(sys.argv[0]), time.monotonic(), 0
    for i, (row, info) in enumerate(zip(rows, described)):
        record = out / 'records' / f'{i:06d}.json'
        if i % of != shard or record.exists():
            if record.exists():
                prior = read(record)
                assert prior['id'] == info['id'] and prior['script_sha256'] == version
                assert sha(out / 'fields' / f'{i:06d}.npz') == prior['sha256']
            continue
        started = time.monotonic()
        with no_query_truth(source):
            fields, note = compute(source, rows, described, i)
        if source.kind == 'cohort':
            note['prediction_frame'] = source.frame_by_id[row['episode_id']]
        else:
            note['prediction_frame'] = 'original'
        np.savez_compressed(out / 'fields' / f'{i:06d}.npz', **{k: v.astype(np.float32) for k, v in fields.items()})
        write(record, dict(index=i, **info, sha256=sha(out / 'fields' / f'{i:06d}.npz'), script_sha256=version,
                           seconds=round(time.monotonic() - started, 2), **note))
        done += 1
        if done % 20 == 0:
            print(json.dumps(dict(shard=args.shard, completed=done, last=i, total=len(rows), seconds=round(time.monotonic() - begin, 1))), flush=True)
    print(json.dumps(dict(shard=args.shard, finished=done, seconds=round(time.monotonic() - begin, 1))), flush=True)


def evaluate(args):
    fixtures()
    source, out = opened(args)
    plan = read(out / 'config.json')
    assert plan['instrument_sha256'] == sha(__file__), 'Instrument changed after field freeze'
    assert sha(plan['generator_path']) == plan['generator_sha256'], 'Generator changed after field freeze'
    if source.kind == 'cohort':
        assert plan['where'] == str(source.folder) and plan['foris'] == str(source.foris)
        assert plan['profile'] == str(source.profile) and plan['profile_sha256'] == sha(source.profile)
        assert plan['manifest_sha256'] == sha(source.folder / 'manifest.json')
        assert plan['foris_index_sha256'] == sha(source.foris / ('inference.jsonl' if (source.foris / 'inference.jsonl').exists() else 'inference.json'))
        for rel, digest in plan['dependency_sha256'].items():
            assert sha(REPO / rel) == digest
        assert plan['transform_sha256'] == sha(source.assets / 'third_party/foris_official/utils/data.py')
        assert plan['basis_sha256'] == sha(source.assets / 'native_assets/positional_basis.pt')
    rows, grid = source.rows[:len(plan['ids'])], plan.get('grid', 64)
    missing = [i for i in range(len(rows)) if not (out / 'records' / f'{i:06d}.json').exists()]
    assert not missing, f'{len(missing)} episodes have no fields yet (first {missing[:5]}); finish every shard first'
    records = [read(out / 'records' / f'{i:06d}.json') for i in range(len(rows))]
    assert [r['id'] for r in records] == plan['ids'] == [source.describe(r)['id'] for r in rows]
    assert all(r['script_sha256'] == plan['generator_sha256'] for r in records), 'Mixed generator source versions'
    frames = sorted({r['prediction_frame'] for r in records})
    assert len(frames) == 1, 'Mixed original/CLI predictions must be scored as separate sources'
    for i, rec in enumerate(records):
        assert sha(out / 'fields' / f'{i:06d}.npz') == rec['sha256']
    if not (out / 'sealed.json').exists():                    # every field is fixed before a query mask is opened
        write(out / 'sealed.json', dict(n=len(records), fields={r['id']: r['sha256'] for r in records}, query_GT_read=False,
                                        encoder_calls=sum(r.get('encoder_calls', 0) for r in records),
                                        config_sha256=sha(out / 'config.json'), script_sha256=[plan['generator_sha256']]))
    seal = read(out / 'sealed.json')
    assert seal['config_sha256'] == sha(out / 'config.json')
    episodes = []
    for i, (row, rec) in enumerate(zip(rows, records)):
        path = out / 'fields' / f'{i:06d}.npz'
        assert sha(path) == seal['fields'][rec['id']]
        with np.load(path, allow_pickle=False) as z:
            fields = {k: z[k].astype(np.float64) for k in z.files}
        mask = source.prediction(row)
        predicted_for_rule = token_mass(mask, grid)
        truth, ignore = source.truth(row, mask.shape)
        valid = np.ones(mask.shape, bool) if ignore is None else ~ignore       # ignored pixels count on neither side
        mask, truth = mask & valid, truth & valid
        predicted, hit = token_mass(mask, grid), token_mass(mask & truth, grid)
        false_alarm, missed, rest = predicted - hit, token_mass(truth & ~mask, grid), token_mass(~truth & ~mask & valid, grid)
        item = dict(id=rec['id'], cls=rec['cls'], dataset=rec['dataset'], g=float(truth.sum()), hit=float(hit.sum()),
                    false_alarm=float(false_alarm.sum()), confusable=rec.get('confusable'), fields={})
        for k, v in fields.items():
            if k.startswith('aux.'):
                continue
            keep = rule(v, ZERO(k), predicted_for_rule)
            item['fields'][k] = dict(**separation(v, hit, false_alarm), **recovery(v, missed, rest), rule_hit=float(hit[keep].sum()),
                                     rule_false_alarm=float(false_alarm[keep].sum()), changed=bool((predicted[~keep] > 0).any()))
        if 'aux.foreign_response' in fields:                  # where another concept's upper half falls
            upper, own = fields['aux.foreign_response'] > .5, fields['aux.own_response'] > .5
            item['foreign'] = dict(hit=float(hit[upper].sum()), false_alarm=float(false_alarm[upper].sum()), tokens=int(upper.sum()), own_tokens=int(own.sum()))
        episodes.append(item)
    (out / 'episodes.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in episodes))

    report = {}
    for dataset in sorted({e['dataset'] for e in episodes}):
        part = [e for e in episodes if e['dataset'] == dataset]
        cls = np.array([e['cls'] for e in part])
        g, I, Fa = (np.array([e[k] for e in part]) for k in ('g', 'hit', 'false_alarm'))
        ok = g > 0
        base = macro(I, g + Fa, cls)
        floor = dict(episodes=len(part), observed_fold_class_slots=len(np.unique(cls)),
                     class_metric='observed fold/class pooled I/U; absent benchmark slots not added',
                     foris=base, recall=float(I.sum() / g.sum()), false_alarm_per_target=float(Fa.sum() / g.sum()),
                     false_alarm_share_of_union=float(Fa.sum() / (g.sum() + Fa.sum())), no_false_alarm=macro(I, g, cls), no_miss=macro(g, g + Fa, cls),
                     rho_F_g=rank_correlation(Fa[ok], g[ok]), rho_r_g=rank_correlation(I[ok] / g[ok], g[ok]), rho_F_r=rank_correlation(Fa[ok], I[ok] / g[ok]),
                     confusable_found=float(np.mean([(e['confusable'] or 0) >= 4 for e in part])))
        foreign = [e['foreign'] for e in part if 'foreign' in e]
        if foreign:
            total = lambda key: float(sum(f[key] for f in foreign))
            floor['foreign_upper_half'] = dict(false_alarm_inside=total('false_alarm') / max(float(sum(e['false_alarm'] for e in part if 'foreign' in e)), 1),
                                               hit_inside=total('hit') / max(float(sum(e['hit'] for e in part if 'foreign' in e)), 1),
                                               tokens_against_own=total('tokens') / max(total('own_tokens'), 1))
        table = {}
        for k in sorted({k for e in part for k in e['fields']}):
            have = [e['fields'][k] for e in part if k in e['fields'] and e['fields'][k]['auc'] is not None]
            out_side = [e['fields'][k] for e in part if k in e['fields'] and e['fields'][k]['miss_auc'] is not None]
            I1 = np.array([e['fields'][k]['rule_hit'] if k in e['fields'] else e['hit'] for e in part])
            F1 = np.array([e['fields'][k]['rule_false_alarm'] if k in e['fields'] else e['false_alarm'] for e in part])
            h, kept = float(I1.sum() / max(I.sum(), 1)), float(F1.sum() / max(Fa.sum(), 1))
            total = lambda rows_, key: float(sum(x[key] for x in rows_))
            table[k] = dict(
                present=float(np.mean([k in e['fields'] for e in part])), separable=len(have),
                auc=float(np.mean([x['auc'] for x in have])) if have else None,
                kept95=total(have, 'kept95_mass') / total(have, 'false_alarm') if have else None,
                kept90=total(have, 'kept90_mass') / total(have, 'false_alarm') if have else None,
                hits_kept=h, false_alarms_kept=kept, exchange=(1 - h) / (1 - kept) if kept < 1 else None,
                changed=float(np.mean([e['fields'][k]['changed'] if k in e['fields'] else False for e in part])),
                rule=macro(I1, g + F1, cls), delta=macro(I1, g + F1, cls) - base, interval=interval(I, g + Fa, I1, g + F1, cls),
                miss_auc=float(np.mean([x['miss_auc'] for x in out_side])) if out_side else None,
                miss_at_1pct=total(out_side, 'miss_at_1pct') / total(out_side, 'missed') if out_side else None)
        report[dataset] = dict(floor=floor, fields=table)
        f = floor
        rho = lambda value: f'{value:+.2f}' if value is not None else 'N/A'
        print(f"\n== {out.name}/{dataset}  n={f['episodes']}  FoRIS {f['foris']:.2f} | recall {100 * f['recall']:.1f}% | sum F / sum g {f['false_alarm_per_target']:.3f}"
              f" | false-alarm share of union {f['false_alarm_share_of_union']:.3f} | no false alarm {f['no_false_alarm']:.2f} | no miss {f['no_miss']:.2f}"
              f" | rho(F,g) {rho(f['rho_F_g'])} rho(r,g) {rho(f['rho_r_g'])} rho(F,r) {rho(f['rho_F_r'])} | confusables found in {100 * f['confusable_found']:.0f}%")
        if foreign:
            x = f['foreign_upper_half']
            print(f"   another concept's upper half holds {100 * x['false_alarm_inside']:.1f}% of FoRIS false alarms and {100 * x['hit_inside']:.1f}% of its hits; its size is {x['tokens_against_own']:.2f} of the own one")
        print('   field                     present  AUC   kept95 kept90 | rule: hits  false alarms  exchange  changed | class mIoU  delta [95%]            | misses: AUC  found at 1%')
        show = lambda v, form: format(v, form) if v is not None else '  -  '
        for k, v in table.items():
            print(f"   {k:25s} {v['present']:5.2f}   {show(v['auc'], '.3f')}  {show(v['kept95'], '.2f')}   {show(v['kept90'], '.2f')}  |      {v['hits_kept']:.3f}  {v['false_alarms_kept']:.3f}"
                  f"        {show(v['exchange'], '.3f')}     {v['changed']:.2f}   |  {v['rule']:6.2f}    {v['delta']:+5.2f} [{v['interval'][0]:+.2f}, {v['interval'][1]:+.2f}]   |"
                  f"   {show(v['miss_auc'], '.3f')}    {show(v['miss_at_1pct'], '.3f')}")
    write(out / 'report.json', dict(frame=frames[0], frame_note='Tokens are the grid of the squashed input; one stored prediction frame per source', grid=grid,
                                    exposure=plan['exposure'], interval_scope='Paired episode bootstrap over exposed class/photo composition; not independent confirmation',
                                    source=plan['where'], datasets=report, seal_sha256=sha(out / 'sealed.json'), evaluator_sha256=sha(__file__)))


def summary(args):
    """One rule for all: chosen on the design sets by its worst case, then read on the confirm sets without change."""
    load = lambda names: {f'{n}/{d}': v for n in names for d, v in read(ROOT / 'inmask' / n / 'report.json')['datasets'].items()}
    design, confirm = load(args.design), load(args.confirm)
    fields = sorted({k for v in list(design.values()) + list(confirm.values()) for k in v['fields']})
    eligible = [k for k in fields if not any(tag in k for tag in DIAGNOSTIC) and all(k in v['fields'] for v in design.values())]
    width = max(len(k) for k in list(design) + list(confirm))
    print('class mIoU of "FoRIS restricted to tokens above the level-free point" minus complete FoRIS')
    print(f"{'':{width}s}  FoRIS  share  " + ' '.join(f'{k[:16]:>16s}' for k in fields))
    for tag, group in (('design', design), ('confirm', confirm)):
        for name, v in group.items():
            print(f"{name:{width}s} {v['floor']['foris']:6.2f}  {v['floor']['false_alarm_share_of_union']:.3f}  "
                  + ' '.join(f"{v['fields'][k]['delta']:+16.2f}" if k in v['fields'] else f"{'-':>16s}" for k in fields) + f'   {tag}')
    worst = {k: min(v['fields'][k]['delta'] for v in design.values()) for k in eligible}
    chosen = max(worst, key=worst.get) if worst else None
    if chosen is None or worst[chosen] <= 0:
        print(f"\nNo eligible field is positive on every design set (best worst case: {chosen} {worst.get(chosen, float('nan')):+.2f}). Nothing is carried to the confirm sets.")
        return
    print(f"\nChosen on the design sets by worst case: {chosen} ({worst[chosen]:+.2f}). On the confirm sets, unchanged:")
    for name, v in confirm.items():
        x = v['fields'].get(chosen)
        if x is None:
            print(f"   {name}: field absent"); continue
        print(f"   {name}: {x['delta']:+.2f} [{x['interval'][0]:+.2f}, {x['interval'][1]:+.2f}] | hits kept {x['hits_kept']:.3f}, false alarms kept {x['false_alarms_kept']:.3f},"
              f" exchange {x['exchange'] if x['exchange'] is None else round(x['exchange'], 3)} against share {v['floor']['false_alarm_share_of_union']:.3f} | kept95 {x['kept95']}")
    holds = [v['fields'][chosen] for v in confirm.values() if chosen in v['fields']]
    print(f"   positive on {sum(x['delta'] > 0 for x in holds)}/{len(holds)} confirm sets; interval above -0.3 on {sum(x['interval'][0] > -.3 for x in holds)}/{len(holds)};"
          f" at least +1.0 on {sum(x['delta'] >= 1 for x in holds)}/{len(holds)}")


def arguments(stages):
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=stages)
    parser.add_argument('--run', help='output folder of scripts/run_m4_baselines.py infer (has features/)')
    parser.add_argument('--cohort', help='manifest cohort folder (absolute or relative to the assets), tokens read through the raw cache')
    parser.add_argument('--foris', help='folder with the complete FoRIS masks of these episodes when it is not the cohort folder')
    parser.add_argument('--assets', default=str(REPO.parent / 'cv_data'))
    parser.add_argument('--name', help='folder under inmask/ for this source; default is the source folder name')
    parser.add_argument('--dataset', help='report the whole source as this one dataset')
    parser.add_argument('--limit', type=int, help='first N episodes of the manifest')
    parser.add_argument('--shard', default='0/1', help='k/n: this process handles episodes with index % n == k')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--profile', help='profile.json of the raw cache; default is the one the cohort recorded, else the shared one')
    parser.add_argument('--branch', default='O/24', help='cached branch read by --cohort, e.g. O/24, O/12, QK/24, V/24')
    parser.add_argument('--design', nargs='*', default=[])
    parser.add_argument('--confirm', nargs='*', default=[])
    return parser


if __name__ == '__main__':
    args = arguments(['selftest', 'probe', 'fields', 'evaluate', 'summary']).parse_args()
    torch.set_num_threads(args.threads)
    {'selftest': selftest, 'probe': lambda: probe(args), 'fields': lambda: make_fields(args), 'evaluate': lambda: evaluate(args),
     'summary': lambda: summary(args)}[args.stage]()
