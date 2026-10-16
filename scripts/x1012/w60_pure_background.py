"""W60: remove mixed reference tokens from the negative kernel, preserving its mass.

    python -B scripts/x1012/w60_pure_background.py produce 2
    python -B scripts/x1012/w60_pure_background.py score
"""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/x1012'), str(ROOT/'scripts/x1013')]
import common as c
import w59_normalization_causal as run

OUT = c.task_path('W60')
ARMS = ('A0', 'A0+', 'PureBG', 'PureBG+')


def changed_fields(r, q, y, fields):
    import torch
    import torch.nn.functional as F
    lab = c.rc.lab
    mu = .5 * (r.mean(0) + q.mean(0))
    sim = lab.unit(q - mu) @ lab.unit(r - mu).T
    maximum = sim.max()
    kernel = torch.exp((sim - maximum) / .05)
    df, db = kernel @ y, kernel @ (1-y)
    original = df / (df+db).clamp_min(1e-30)
    error = float(np.abs(original.numpy() - fields['c32']).max())
    assert error < 1e-6
    pure = (y == 0).float()
    n_pure, nb = float(pure.sum()), float((1-y).sum())
    if n_pure:
        db_new = (kernel @ pure) * (nb / n_pure)
        cc = df / (df + db_new).clamp_min(1e-30)
    else:
        db_new, cc = db, original
    labels = torch.from_numpy(fields['regions64'].astype(np.int64))
    onehot = F.one_hot(labels, 64).float()
    region = lambda z: onehot @ ((onehot.T @ z) / onehot.sum(0).clamp_min(1))
    sim_max = sim.max(1).values
    result = dict(fields, c32=cc.numpy(), ind0_32=region(cc).numpy(),
                  indn_32=region(cc * lab.minmax(sim_max)).numpy())
    # Offset common to all query tokens is retained; raw F is identical across arms.
    fields_out = dict(F_score=(.05*df.clamp_min(1e-30).log()+maximum).numpy(),
                      B_score=(.05*db.clamp_min(1e-30).log()+maximum).numpy(),
                      B_pure=(.05*db_new.clamp_min(1e-30).log()+maximum).numpy(),
                      c_original=original.numpy(), c_pure=cc.numpy())
    extra = dict(c_reproduction_max_error=error, pure_background_tokens=int(n_pure),
                 no_pure_background=not bool(n_pure), background_mass=nb,
                 mixed_background_mass_fraction=float(((1-y)*(y>0)).sum()/max(nb,1e-6)),
                 reference_fraction=float(y.mean()))
    return result, fields_out, extra


def one(b):
    if time.time() >= run.DEADLINE:
        raise TimeoutError('Authorized window ended; no further inference')
    import w23_simple_pipeline as sh
    start = time.monotonic()
    original_ind = c.rc.independent
    masks = {}
    with c.rc.pc.inference_only(b):
        r, q, mask, y, fields = original_ind(b, {})
        cleaned, values, extra = changed_fields(r, q, y, fields)
        for arm, variant in (('A0', fields), ('PureBG', cleaned)):
            c.rc.independent = lambda *args, _f=variant, **kwargs: (r, q, mask, y, _f)
            try:
                pre, final, rho = sh.simple_head(b, crf=True)
            finally:
                c.rc.independent = original_ind
            masks[arm], masks[arm+'+'] = pre, final
    extra.update(rho=rho, fallback=rho>.9)
    return run.save_episode(OUT, b, masks, extra, time.monotonic()-start, fields=values)


def auc(labels, scores):
    from scipy.stats import rankdata
    n1 = int(labels.sum()); n0 = len(labels)-n1
    if not n1 or not n0:
        return None
    return float((rankdata(scores)[labels].sum() - n1*(n1+1)/2)/(n1*n0))


def score_extra(z, gt, valid):
    coverage = gt.reshape(64,16,64,16).mean((1,3)).ravel()
    valid_fraction = valid.reshape(64,16,64,16).mean((1,3)).ravel()
    selected = ((coverage>=.9)|(coverage<=.1)) & (valid_fraction == 1)
    labels = coverage[selected] >= .9
    return dict(auc_F=auc(labels, z['F_score'][selected]),
                auc_B=auc(labels, -z['B_score'][selected]),
                auc_B_pure=auc(labels, -z['B_pure'][selected]),
                auc_c=auc(labels, z['c_original'][selected]),
                auc_c_pure=auc(labels, z['c_pure'][selected]), pure_query_tokens=int(selected.sum()))


def report_mechanism(records, summary):
    lines, table = [], {}
    for d in c.DOMAINS:
        sel = [r for r in records if r['dataset']==d and r['auc_B'] is not None]
        if not sel:
            table[d] = dict(n=0)
            continue
        x = dict(n=len(sel), median_B=float(np.median([r['auc_B'] for r in sel])),
                 median_B_pure=float(np.median([r['auc_B_pure'] for r in sel])),
                 mean_B_delta=float(np.mean([r['auc_B_pure']-r['auc_B'] for r in sel])),
                 median_c=float(np.median([r['auc_c'] for r in sel])),
                 median_c_pure=float(np.median([r['auc_c_pure'] for r in sel])))
        table[d]=x
        lines.append(f'{d}: n {len(sel)}  median A_B {x["median_B"]:.3f}->{x["median_B_pure"]:.3f}  mean delta {x["mean_B_delta"]:+.3f}; median A_c {x["median_c"]:.3f}->{x["median_c_pure"]:.3f}')
    summary['mechanism'] = table
    summary['no_pure_background'] = sum(r['no_pure_background'] for r in records)
    summary['max_c_reproduction_error'] = max(r['c_reproduction_max_error'] for r in records)
    run.write(OUT/'summary.json', summary)
    with (OUT/'report.txt').open('a') as f:
        f.write('\nMechanism: pure query tokens, same F and reference class masses\n'+'\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if sys.argv[1]=='produce':
        run.run_produce(OUT, ARMS, one, int(sys.argv[2]) if len(sys.argv)>2 else 2)
    elif sys.argv[1]=='score':
        rows, summary = run.score_run(OUT, ARMS, 'w60_pure_background_S.jsonl.gz', score_extra)
        report_mechanism(rows, summary)
