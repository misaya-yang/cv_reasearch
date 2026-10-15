"""W30: the frozen SH+ (w23_simple_pipeline.simple_head) on the new-photo holdout (E1, scored once) and on the
official protocol (E2), on this Mac.  No FoRIS run: E1 pairs with the sealed FoRIS predictions of the holdout,
E2 compares with FoRIS's published numbers (and the sealed FoRIS reproduction on ISIC/Lung/Fundus/Road).
Plan and readings: docs/research/x1012_tasks/W30_mac_confirmation_PLAN.md (fixed before this file was written).

The head is not reimplemented: features are encoded here (same frozen TimmDINOv3, same FoRIS transform and loader
as W12 / the holdout) and handed to simple_head by replacing r5_candidate.raw for that call only.

    python scripts/x1012/w30_confirm.py check          # P1 (cached features) and P2 (fresh MPS encoding), 40 S episodes
    python scripts/x1012/w30_confirm.py run            # E1 then E2 in the planned order; resumable; stops at 10-19 08:00
    python scripts/x1012/w30_confirm.py score NAME     # NAME = E1 or an E2 dataset; only after that part is sealed
"""
import hashlib, json, multiprocessing, sys, time
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1013'))
sys.path.insert(0, str(ROOT / 'scripts/x1012'))
sys.path.insert(0, str(ROOT / 'scripts/x1012s'))
import common as c

A = ROOT.parent / 'cv_data/a'
OUT = A / 'x1013_W30'
MANIFEST = {'E1': A / 'x1013_photo_holdout/manifest.json', 'E2': A / 'x1013_official/manifest.json'}
ORDER = ('E1', 'isic', 'lung', 'fundus', 'deepglobe_road', 'coco', 'suim', 'isaid', 'pascal_part', 'paco_part', 'lvis')
STOP = datetime(2026, 10, 19, 8, 0)
FULL = (datetime(2026, 10, 16, 22, 0), STOP)   # user-approved full-power window
CACHE = 96                                     # in-memory O24 (16 MB each); see W12
METHOD = ['scripts/x1012/' + n for n in ('w23_simple_pipeline.py', 'r5_candidate.py', 'w19_simple_head.py',
                                         'w09_integration_lambda.py', 'lab.py', 'plan_common.py')]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
safe = lambda i: str(i).replace('/', '_')


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.tmp'); t.write_text(json.dumps(v, indent=1)); t.replace(p)


# ---------------------------------------------------------------- head (worker processes)
def worker_init():
    c.initialize()


def head(job):
    """job = (part, binding, {'reference': tensor, 'query': tensor}, metadata) -> record."""
    import torch
    import w23_simple_pipeline as sh
    part, b, feats, meta = job
    b = dict(b, fit={'reference_raw': {'role': 'reference'}, 'query_raw_identity': {'role': 'query'}})
    original = c.rc.raw
    c.rc.raw = lambda metadata, counters: feats[metadata['role']]
    began = time.monotonic()
    try:
        pre, final, rho = sh.simple_head(b, crf=True)
    finally:
        c.rc.raw = original
    path = OUT / 'predictions' / part / (safe(b['index']) + '.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, SH=c.pack(pre), **{'SH+': c.pack(final)}); tmp.replace(path)
    rec = dict(index=str(b['index']), dataset=b['dataset'], part=part, rho=rho, prediction_path=str(path),
               prediction_sha256=sha(path), head_seconds=time.monotonic() - began, query_GT_reads=0, **meta)
    if not part.startswith('check'):
        write(OUT / 'records' / part / (safe(b['index']) + '.json'), rec)
    return rec


# ---------------------------------------------------------------- encoder side (main process)
class Encoder:
    def __init__(self):
        import torch
        import photo_holdout as ph
        self.ph, self.torch = ph, torch
        self.task, self.rt = ph.runtime()
        self.model = ph.module('w30_encoder', A / 'x1013_official/frozen/encoder.py').TimmDINOv3(
            str(ph.WEIGHTS)).to('mps').eval().requires_grad_(False)
        self.cache = OrderedDict()
        self.new = 0

    def inputs(self, b):
        reference, _, query = self.rt.load_inputs(b['row'], self.ph.ASSETS)
        x = {'reference': self.rt.host._transform(reference), 'query': self.rt.host._transform(query)}
        return x, {role: self.task.array_hash(v.numpy()) for role, v in x.items()}

    def features(self, x, hashes):
        torch = self.torch
        for h in hashes.values():
            if h in self.cache:
                self.cache.move_to_end(h)
        missing = [r for r, h in hashes.items() if h not in self.cache]
        if missing:
            with torch.inference_mode():
                maps = self.model.get_intermediate_layers(torch.stack([x[r] for r in missing]).to('mps'), n=1, reshape=True)[0]
                torch.mps.synchronize()
                for r, raw in zip(missing, maps.cpu().permute(0, 2, 3, 1).reshape(len(missing), 4096, 1024).contiguous().unbind(0)):
                    assert raw.dtype == torch.float32 and torch.isfinite(raw).all()
                    self.cache[hashes[r]] = raw
            self.new += len(missing)
            if self.new % 64 < len(missing):
                torch.mps.empty_cache()
        feats = {r: self.cache[h] for r, h in hashes.items()}
        while len(self.cache) > CACHE:
            self.cache.popitem(last=False)
        return feats, len(missing)


# ---------------------------------------------------------------- P1 / P2
def check():
    """Heads run in spawned workers exactly as in `run` (the holdout runtime's `ics` must not shadow src/ics)."""
    import gzip
    import torch
    import w23_simple_pipeline as sh
    c.initialize()
    w23 = {r['index']: r for r in map(json.loads, gzip.open(ROOT / 'evidence/research/w23_simple_pipeline_S.jsonl.gz', 'rt'))}
    rows = sorted(sh.rows_for('S'), key=lambda b: str(b['index']))
    pick = [b for d in c.DOMAINS for b in [x for x in rows if x['dataset'] == d][:4]]
    assert len(pick) == 40
    enc = Encoder()
    jobs, coss = [], {}
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context('spawn'), initializer=worker_init) as pool:
        for b in pick:
            cached = {role: c.rc.raw(b['fit'][k], {}) for role, k in (('reference', 'reference_raw'), ('query', 'query_raw_identity'))}
            x, hashes = enc.inputs(b)
            fresh, _ = enc.features(x, hashes)
            coss[str(b['index'])] = {r: float(torch.nn.functional.cosine_similarity(fresh[r], cached[r], dim=1).median()) for r in cached}
            jobs.append((b, pool.submit(head, ('check', b, cached, {})), pool.submit(head, ('check_fresh', b, fresh, {}))))
        iou = lambda x: x[0] / max(x[1], 1)
        out, p1, p2 = [], 0, 0
        for b, f1, f2 in jobs:
            gt, valid = c.truth(b)
            ius = []
            for f in (f1, f2):
                with np.load(f.result()['prediction_path']) as z:
                    ius.append(c.iu(c.unpack(z['SH+']), gt, valid))
            ref = w23[str(b['index'])]['SH+']
            p1 += ius[0][:2] == ref[:2]
            p2 += abs(iou(ius[1]) - iou(ref)) < .01
            out.append(dict(index=str(b['index']), dataset=b['dataset'], cos=coss[str(b['index'])], P1_equal=ius[0][:2] == ref[:2],
                            dIoU_fresh=100 * (iou(ius[1]) - iou(ref))))
    allcos = [v for d in coss.values() for v in d.values()]
    res = dict(P1=f'{p1}/40', P1_pass=p1 >= 39, P2_cos_median=float(np.median(allcos)), P2_cos_min=float(np.min(allcos)),
               P2_within_1pt=f'{p2}/40', P2_pass=bool(np.median(allcos) >= .999 and p2 >= 38), episodes=out)
    write(OUT / 'check.json', res)
    print(json.dumps({k: v for k, v in res.items() if k != 'episodes'}, indent=1), flush=True)


# ---------------------------------------------------------------- run
def evaluation():
    import photo_holdout as ph
    rec = dict(method='w23_simple_pipeline.simple_head (SH+)', method_sha256={m: sha(ROOT / m) for m in METHOD},
               runner_sha256=sha(__file__), plan_sha256=sha(ROOT / 'docs/research/x1012_tasks/W30_mac_confirmation_PLAN.md'),
               manifests={k: sha(v) for k, v in MANIFEST.items()}, protocol_sha256=sha(A / 'x1013_official/protocol.json'),
               encoder_sha256=sha(A / 'x1013_official/frozen/encoder.py'), weights_sha256=sha(ph.WEIGHTS / 'model.safetensors'),
               positional_basis_sha256=sha(c.rc.lab.BASIS), order=ORDER, stop=STOP.isoformat(),
               full_power_window=[t.isoformat() for t in FULL], foris_run=False, feature_cache_on_disk=False)
    p = OUT / 'evaluation.json'
    if p.exists():
        old = read(p)
        for k in ('method_sha256', 'manifests', 'protocol_sha256', 'encoder_sha256', 'weights_sha256', 'positional_basis_sha256'):
            assert old[k] == rec[k], f'frozen input changed: {k}'
    else:
        write(p, dict(rec, frozen_at=datetime.now().isoformat()))


def parts():
    e1 = read(MANIFEST['E1']); e2 = read(MANIFEST['E2'])
    return {name: (e1 if name == 'E1' else [b for b in e2 if b['dataset'] == name]) for name in ORDER}


def resource():
    import psutil
    return psutil.cpu_percent(interval=.2), psutil.virtual_memory().available / 1024 ** 3


def run():
    import w12_official as w12
    c.initialize()
    evaluation()
    enc = Encoder()
    started = time.monotonic()
    log = (OUT / 'run.log').open('a')
    say = lambda **k: (log.write(json.dumps(dict(time=datetime.now().isoformat(timespec='seconds'), **k)) + '\n'), log.flush(),
                       print(json.dumps(k), flush=True))
    for name, rows in parts().items():
        if (OUT / 'sealed' / (name + '.json')).exists():
            continue
        if datetime.now() >= STOP:
            say(state='STOPPED_AT_DEADLINE', next=name); return
        rows = w12.bfs_order(rows)
        todo = [b for b in rows if not (OUT / 'records' / name / (safe(b['index']) + '.json')).exists()]
        workers = 6 if FULL[0] <= datetime.now() < FULL[1] else 4
        say(state='START', part=name, n=len(rows), todo=len(todo), workers=workers)
        with ThreadPoolExecutor(3) as decode, ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                                                  initializer=worker_init) as pool:
            ahead = deque(decode.submit(enc.inputs, b) for b in todo[:16])
            pending, done, cpu, avail = set(), len(rows) - len(todo), None, 99.
            for n, b in enumerate(todo):
                t0 = time.monotonic()
                x, hashes = ahead.popleft().result()
                if n + 16 < len(todo):
                    ahead.append(decode.submit(enc.inputs, todo[n + 16]))
                feats, new = enc.features(x, hashes)
                pending.add(pool.submit(head, (name, b, feats, dict(new_encoded_images=new, encode_seconds=time.monotonic() - t0))))
                if n % 25 == 0:
                    cpu, avail = resource()
                limit = workers + 2 if avail >= 4 else max(1, workers // 2)
                while len(pending) >= limit:
                    fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in fin:
                        f.result(); done += 1
                if n % 100 == 0:
                    say(part=name, done=done, n=len(rows), cpu=cpu, available_gib=round(avail, 1),
                        hours=round((time.monotonic() - started) / 3600, 2), new_encodes=enc.new)
            for f in pending:
                f.result()
        recs = [read(OUT / 'records' / name / (safe(b['index']) + '.json')) for b in rows]
        assert all(sha(r['prediction_path']) == r['prediction_sha256'] for r in recs)
        write(OUT / 'sealed' / (name + '.json'), dict(part=name, n=len(recs), sealed_at=datetime.now().isoformat(),
                                                      evaluation_sha256=sha(OUT / 'evaluation.json'),
                                                      predictions={r['index']: r['prediction_sha256'] for r in recs}))
        say(state='SEALED', part=name, n=len(recs), hours=round((time.monotonic() - started) / 3600, 2))
    say(state='ALL_SEALED')


# ---------------------------------------------------------------- score (only a sealed part; one reading)
def _truth(row):
    import torch
    from PIL import Image

    def mask(path):
        with Image.open(path) as image:
            x = torch.from_numpy((np.asarray(image.convert('L')) > 0).astype(np.float32))
        return torch.nn.functional.interpolate(x[None, None], (1024, 1024), mode='nearest')[0, 0].numpy() > .5

    gt = mask(row['query_mask_path'])
    valid = ~mask(row['query_ignore_mask_path']) if row.get('query_ignore_mask_path') else np.ones_like(gt)
    return gt & valid, valid


def _iu(args):
    """Same GT alignment as photo_holdout.score and w12_official._iu (nearest resize to the 1024 frame)."""
    name, b = args
    rec = read(OUT / 'records' / name / (safe(b['index']) + '.json'))
    assert sha(rec['prediction_path']) == rec['prediction_sha256']
    gt, valid = _truth(b['row'])
    out = dict(index=str(b['index']), dataset=b['dataset'], fold=b['row']['fold'], cls=b['row']['loader_class_id'],
               class_id=b['row'].get('global_class_id', b['row']['loader_class_id']), photo=b['row']['query_photo_id'],
               rho=rec['rho'], gt=int(gt.sum()), valid=int(valid.sum()))
    with np.load(rec['prediction_path'], allow_pickle=False) as z:
        for arm in ('SH', 'SH+'):
            m = c.unpack(z[arm]) & valid
            i, n = int((m & gt).sum()), int(m.sum())
            out[arm] = [i, n + out['gt'] - i, n]
    return out


def _score_rows(name, rows):
    import gzip
    seal = read(OUT / 'sealed' / (name + '.json'))
    assert seal['n'] == len(rows) and seal['evaluation_sha256'] == sha(OUT / 'evaluation.json')
    write(OUT / 'scores' / (name + '_started.json'), dict(time=datetime.now().isoformat(), seal_sha256=sha(OUT / 'sealed' / (name + '.json'))))
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        R = list(pool.map(_iu, [(name, b) for b in rows], chunksize=16))
    with gzip.open(OUT / 'scores' / (name + '.jsonl.gz'), 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in R)
    return R


def _metric(iu, gt, valid):
    """x1012s2_confirm_score.packed_metric from [I, U, predicted]."""
    i, u, n = iu
    err = float(abs(np.log2(n / gt))) if n and gt else (0. if not n and not gt else None)
    return dict(I=i, U=u, TP=i, FP=n - i, FN=gt - i, GT=gt, predicted=n, IoU=100 * i / max(u, 1), target_fraction=gt / max(valid, 1),
                abs_log2_area_error=err, infinite_area_error=err is None, zero_predicted_area=n == 0)


def score_e1():
    """Plan 6/E1: photo_holdout statistics; main = SH+, paired with the sealed FoRIS and MFU holdout predictions."""
    import photo_holdout as ph
    helper = ph.module('holdout_score_helpers', ROOT / 'scripts/x1012s2_confirm_score.py')
    rows = read(MANIFEST['E1'])
    R = _score_rows('E1', rows)
    holdout = {r['index']: r for r in read(A / 'x1013_photo_holdout/scores/records.json')}
    seal = read(A / 'x1013_photo_holdout/sealed.json')
    assert sha(A / 'x1013_photo_holdout/scores/records.json') and len(holdout) == len(rows) == seal['n']
    canon = {str(b['index']): b for b in rows}
    records = []
    for r in R:
        h, b = holdout[r['index']], canon[r['index']]
        metric = lambda iu: _metric(iu, r['gt'], r['valid'])
        assert h['metrics']['FoRIS']['GT'] == r['gt']
        records.append(dict(index=r['index'], dataset=r['dataset'], fold=h['fold'], class_id=h['class_id'],
                            query_photo=b['canonical_query_photo'], reference_photo=b['canonical_reference_photo'],
                            metrics={'main': metric(r['SH+']), 'SH': metric(r['SH']), 'FoRIS': h['metrics']['FoRIS'],
                                     'MFU': h['metrics']['main']}))
    arms = ('main', 'SH', 'FoRIS', 'MFU')
    summary = helper.point_summary(records, arms)
    paired = helper.global_photo_bootstrap(records, arms, summary, repeats=2000, seed=2026101303)
    macro = {a: float(np.mean([d[a]['class_miou'] for d in summary.values()])) for a in arms}
    report = dict(part='E1', n=len(records), main='SH+', macro=macro, paired=paired,
                  per_domain={d: {a: v[a]['class_miou'] for a in arms} for d, v in summary.items()})
    write(OUT / 'scores/E1_report.json', report)
    print(json.dumps(dict(macro=macro, SHplus_minus_FoRIS=paired['macro10']['FoRIS'], SHplus_minus_MFU=paired['macro10']['MFU'])), flush=True)


def score_e2(name):
    """Plan 6/E2: w12_official metric; 1,000 query-photo bootstraps; FoRIS reproduction pairing where it exists."""
    import gzip
    import w12_official as w12
    protocol = read(A / 'x1013_official/protocol.json')
    rows = [b for b in read(MANIFEST['E2']) if b['dataset'] == name]
    R = _score_rows(name, rows)
    repro = A / 'x1013_official/scores' / (name + '.jsonl.gz')
    if repro.exists():
        old = {r['index']: r for r in map(json.loads, gzip.open(repro, 'rt'))}
        for r in R:
            assert old[r['index']]['gt'] == r['gt']
            r['FoRIS'], r['MFU'] = old[r['index']]['FoRIS'], old[r['index']]['main']
    arms = [a for a in ('SH+', 'SH', 'FoRIS', 'MFU') if a in R[0]]
    layout = w12.slots(R, {k: v for k, v in protocol['folds'].items() if k.split('/')[0] == name})
    point = {a: w12.official(R, a, layout) for a in arms}
    photos = {p: i for i, p in enumerate(sorted({r['photo'] for r in R}))}
    pi = np.array([photos[r['photo']] for r in R])
    rng = np.random.default_rng(2026101303)
    boot = {a: [] for a in arms}
    for _ in range(1000):
        w = np.bincount(rng.integers(0, len(photos), len(photos)), minlength=len(photos))[pi]
        for a in arms:
            boot[a].append(w12.official(R, a, layout, w))
    boot = {a: np.array(v) for a, v in boot.items()}
    ci = lambda v: np.percentile(v, [2.5, 97.5]).tolist()
    published = w12.PUBLISHED.get(name)
    res = dict(part=name, n=len(R), photos=len(photos), official_miou=point, ci95={a: ci(boot[a]) for a in arms},
               published_foris=published,
               SHplus_minus={a: dict(point=point['SH+'] - point[a], ci95=ci(boot['SH+'] - boot[a])) for a in arms if a != 'SH+'})
    if published is not None:
        lo, hi = res['ci95']['SH+']
        res['vs_published'] = 'significantly higher' if lo > published else 'significantly lower' if hi < published else \
            'not lower' if point['SH+'] >= published else 'lower, not significant'
    summary = read(OUT / 'scores/E2_summary.json') if (OUT / 'scores/E2_summary.json').exists() else {}
    summary[name] = res
    write(OUT / 'scores/E2_summary.json', summary)
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'check':
        check()
    elif cmd == 'run':
        run()
    elif cmd == 'score':
        c.initialize()
        score_e1() if sys.argv[2] == 'E1' else score_e2(sys.argv[2])
    else:
        raise SystemExit(__doc__)
