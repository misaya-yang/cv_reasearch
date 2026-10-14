"""W19 runtime: complete FoRIS vs MFU vs the FoRIS-free head, same harness, same process, same inputs.

Protocol (as CONFIRM1000's head timing): the frozen x1012s2 runtime, CPU with 2 torch threads, O24 features of
reference and query supplied in memory (no encoder, no feature I/O), RGB and reference mask loaded and
preprocessed inside each call, official CRF where the method has one.  Two warm-up episodes are discarded,
and the FoRIS / MFU order alternates per episode.  27 non-fallback episodes, 3 per domain (W15 sample).
  FoRIS : run(providers=('B',))                 complete FoRIS head incl. CRF
  MFU   : run(candidate='MFU', boundary='full_crf')
  W19   : the MFU call minus its two FoRIS-dependent stages ('FoRIS_semantic_core_without_CRF' and
          'U_independent_VP'), plus the measured cost of the cheap score (p + k-means-64 region mean) on
          the same tensors.  All other stages of the W19 head are the very stages the MFU call runs.
Each FoRIS-internal stage is also timed by wrapping the host's methods.

    python scripts/x1012/w19_time.py
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/x1012s'))
import photo_holdout as ph

OUT = ROOT.parent / 'cv_data/a/x1013_W19'
STAGES = ('_part1_positional_debias', '_part2_background_suppression', '_locate_candidates',
          '_build_seed_cluster_prior', '_part4_semantic_consistency_correction', '_binarize_response', '_finalize_mask')


def raw_features(binding):
    out = {}
    for role, name in (('reference', 'reference_raw'), ('query', 'query_raw_identity')):
        import torch
        with np.load(binding['fit'][name]['payload_path'], allow_pickle=False) as z:
            out[role] = torch.from_numpy(z['O/24'].copy())
    return out


def cheap_score(rt, features, binding):
    """Time only the score p + region mean.  The debiased tensors and the k-means labels are produced by stages
    the W19 head keeps ('shared_M_U_T_unit_projection_coverage', U's kernel stage), so they are built untimed."""
    import torch
    import torch.nn.functional as F
    lab = rt.lab
    r_unit, q_unit = lab.unit(features['reference']), lab.unit(features['query'])
    r = lab.unit(r_unit - (r_unit @ rt.basis) @ rt.basis.T)
    q = lab.unit(q_unit - (q_unit @ rt.basis) @ rt.basis.T)
    cov = lab.coverage64(lab.mask1024(binding['row']['reference_mask_path']).float()).flatten().numpy()
    labels = np.random.default_rng(0).integers(0, 64, 4096)  # bincount cost does not depend on the label values
    t = time.perf_counter()
    fg = torch.from_numpy(cov >= .5 if (cov >= .5).any() else cov == cov.max())
    mu_fg = F.normalize(r[fg].mean(0), dim=0)
    bg = r[~fg]
    if len(bg):
        hard = bg[(bg @ mu_fg).topk(max(1, int(.2 * len(bg)))).indices]
        mu_bg = F.normalize(hard.mean(0), dim=0); mu_bg = mu_bg - (mu_bg @ mu_fg) * mu_fg; mu_bg = mu_bg / mu_bg.norm().clamp_min(1e-8)
    else:
        mu_bg = torch.zeros_like(mu_fg)
    p = (q @ mu_fg - .55 * (q @ mu_bg)).numpy()
    region = (np.bincount(labels, p, 64) / np.maximum(np.bincount(labels, None, 64), 1))[labels]
    _ = p + region
    return time.perf_counter() - t


def main():
    import torch
    rows = json.loads((OUT / 'time_bindings.json').read_text())
    task, rt = ph.runtime()
    torch.set_num_threads(2)
    timers = defaultdict(float)
    for name in STAGES:
        original = getattr(rt.host, name)
        def timed(*a, _f=original, _n=name, **k):
            t = time.perf_counter()
            try:
                return _f(*a, **k)
            finally:
                timers[_n] += time.perf_counter() - t
        setattr(rt.host, name, timed)
    records = []
    for i, b in enumerate([rows[0], rows[1]] + rows):
        features = raw_features(b)
        out = {'index': str(b['index']), 'dataset': b['dataset']}
        order = ('FoRIS', 'MFU') if i % 2 == 0 else ('MFU', 'FoRIS')
        with task.inference_files(b):
            for m in order:
                if m == 'FoRIS':
                    timers.clear()
                    *_, d = rt.run(b, providers=('B',), provided_raw=features)
                    out['FoRIS'] = d['complete_cached_seconds']
                    out['FoRIS_stages'] = dict(timers)
                else:
                    *_, d = rt.run(b, candidate='MFU', boundary='full_crf', provided_raw=features)
                    out['MFU'] = d['complete_cached_seconds']
                    out['MFU_stages'] = d['stage_seconds']
                    out['gate'] = d['reference_gate_true']
        out['cheap_score'] = cheap_score(rt, features, b)
        s = out['MFU_stages']
        out['W19'] = out['MFU'] - s.get('FoRIS_semantic_core_without_CRF', 0) - s.get('U_independent_VP', 0) + out['cheap_score']
        if i >= 2:
            records.append(out)
            print(f"{out['dataset']:15s} FoRIS {out['FoRIS']:.3f}  MFU {out['MFU']:.3f}  W19 {out['W19']:.3f}", flush=True)
    (OUT / 'timing.json').write_text(json.dumps(records, indent=1))
    agg = lambda k: np.array([r[k] for r in records])
    print(f"\nmean seconds  FoRIS {agg('FoRIS').mean():.3f}  MFU {agg('MFU').mean():.3f}  W19 {agg('W19').mean():.3f}"
          f"   (median {np.median(agg('FoRIS')):.3f} / {np.median(agg('MFU')):.3f} / {np.median(agg('W19')):.3f})")
    print('FoRIS internal stages (mean s):', {k: round(float(np.mean([r['FoRIS_stages'].get(k, 0) for r in records])), 3) for k in STAGES})
    keys = sorted({k for r in records for k in r['MFU_stages']})
    print('MFU stages (mean s):', {k: round(float(np.mean([r['MFU_stages'].get(k, 0) for r in records])), 3) for k in keys})
    print('cheap score (mean s):', round(float(agg('cheap_score').mean()), 4))


if __name__ == '__main__':
    main()
