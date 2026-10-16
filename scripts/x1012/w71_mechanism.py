"""Fixed 2x2x2 causal contrasts, without training or selecting a winning arm."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import time
import types
import numpy as np
import w71_runtime as rt

LAYERS = (23, 19)
# 1 is the current SH treatment: soft labels, always debias, joint centering.
ARMS = tuple(''.join(map(str,x)) for x in itertools.product((0,1), repeat=3))
RECALLS = (.25, .5, .75)


def render(field):
    import common as c
    import torch
    return c.rc.lab.minmax(c.rc.lab.up(torch.from_numpy(field.reshape(64,64)))).numpy()


def generate(binding, features):
    import torch
    import common as c
    from models.foris import FoRIS
    from utils.data import downsample_mask
    lab = c.rc.lab
    saved = torch.load(rt.ASSETS/'a/fundus_layer_resolution50_20261013/basis/1024.pt', map_location='cpu', weights_only=True)
    bases = {23:lab.basis().cpu(), 19:saved['bases'][19].float()}
    fields = []; masks = []; decisions = {}
    with c.rc.pc.inference_only(binding), torch.inference_mode():
        rt.check_mask(binding,'reference')
        mask = lab.mask1024(binding['row']['reference_mask_path']).float()
        soft = lab.coverage64(mask).flatten(); hard = downsample_mask(mask[None,None],64,64).flatten().float()
        rho = float(1-(soft*(soft>=.9)).sum()/soft.sum().clamp_min(1e-9))
        assert rho > .9, 'Fixed cohort must remain on the original fallback branch'
        for layer in LAYERS:
            r, q = (lab.unit(features[role][layer]) for role in ('reference','query'))
            basis = bases[layer]
            packed = torch.stack([r.T.reshape(1024,64,64),q.T.reshape(1024,64,64)])[None]
            host = types.SimpleNamespace(positional_basis=basis, should_debiass=True)
            apd = FoRIS._should_apply_positional_debias(host, packed, mask[None,None], 1)
            decisions[str(layer)] = bool(apd)
            transformed = {}
            for always, center in itertools.product((0,1),repeat=2):
                if always or apd:
                    rr = lab.unit(r-(r@basis)@basis.T); qq = lab.unit(q-(q@basis)@basis.T)
                else: rr, qq = r, q
                if center:
                    mean=.5*(rr.mean(0)+qq.mean(0)); rr,qq=lab.unit(rr-mean),lab.unit(qq-mean)
                sim=qq@rr.T; kernel=torch.exp((sim-sim.max())/.05)
                # Preserve the frozen SH denominator and operation order.
                for label,y in ((0,hard),(1,soft)):
                    fg,bg=kernel@y,kernel@(1-y)
                    transformed[f'{label}{always}{center}']=(fg/(fg+bg).clamp_min(1e-30)).numpy().reshape(64,64)
                del sim,kernel
            for arm in ARMS:
                f=transformed[arm]; assert np.isfinite(f).all()
                fields.append(f); masks.append(c.pack(render(f)>.5))
    return dict(fields=np.stack(fields).reshape(2,8,64,64), masks=np.stack(masks).reshape(2,8,-1)), dict(rho=rho, apd=decisions)


def save_prediction(path, arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**arrays); stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)


def produce():
    rt.initialize(); manifest=rt.read(rt.OUT/'manifest.json'); rows=manifest['mechanism']
    out=rt.OUT/'mechanism'; seal=out/'sealed.json'
    if seal.exists(): return
    enc=rt.Encoder(LAYERS, cache_limit=12)
    records=[]
    for n,item in enumerate(rows):
        rt.checkpoint(); rid=item['id']; path=out/'records'/f'{rid}.json'
        record=rt.valid_record(path)
        if record is None:
            tick=time.monotonic(); features,trace=enc.features(item['binding'])
            arrays,detail=generate(item['binding'],features)
            prediction=out/'predictions'/f'{rid}.npz'; save_prediction(prediction,arrays)
            record=dict(id=rid,cohort=item['cohort'],prediction_path=str(prediction),prediction_sha256=rt.sha(prediction),
                        seconds=time.monotonic()-tick,query_GT_reads=0,trace=trace,**detail)
            rt.write(path,record)
        records.append(record)
        rt.heartbeat('mechanism_predict',n+1,len(rows),encoder_calls=enc.calls)
        if (n+1)%10==0: rt.event('mechanism_progress',done=n+1,total=len(rows))
    rt.write(seal,dict(n=len(records),manifest_sha256=rt.sha(rt.OUT/'manifest.json'),
                      predictions={r['id']:r['prediction_sha256'] for r in records},utc=rt.utc()))


def exact_curve(field, gt, valid):
    """Exhaustive tied score boundaries, no sparse threshold approximation."""
    x=field[valid].ravel(); y=gt[valid].ravel(); g=int(y.sum())
    order=np.argsort(-x,kind='stable'); ranked=x[order]
    ends=np.r_[np.flatnonzero(ranked[:-1]!=ranked[1:]),len(ranked)-1]
    tp=np.cumsum(y[order],dtype=np.int64)[ends]; predicted=ends+1
    fp=predicted-tp; union=g+fp; iou=tp/np.maximum(union,1)
    best=int(np.argmax(iou))
    fixed={}
    for recall in RECALLS:
        index=int(np.searchsorted(tp,np.ceil(recall*g)))
        fixed[str(recall)]=dict(FP=int(fp[index]),TP=int(tp[index]),recall=float(tp[index]/max(g,1)),FP_per_GT=float(fp[index]/max(g,1)))
    return dict(best_threshold_iou=100*float(iou[best]),at_recall=fixed)


def score_one(item):
    import common as c
    out=rt.OUT/'mechanism'; rid=item['id']; target=out/'scores'/f'{rid}.json'
    rec=rt.valid_record(out/'records'/f'{rid}.json')
    if target.exists():
        result=rt.read(target); assert result['prediction_sha256']==rec['prediction_sha256']; return result
    rt.check_mask(item['binding'],'query')
    rt.check_mask(item['binding'],'query_ignore')
    gt,valid=c.rc.pc.truth(item['binding']); g=int(gt.sum()); arms={}
    with np.load(rec['prediction_path'],allow_pickle=False) as z:
        for li,layer in enumerate(LAYERS):
            for ai,arm in enumerate(ARMS):
                rt.checkpoint(); field=render(z['fields'][li,ai]); pred=(field>.5)&valid
                assert np.array_equal(c.pack(field>.5),z['masks'][li,ai])
                i=int((pred&gt).sum()); p=int(pred.sum()); u=p+g-i
                arms[f'L{layer}_{arm}']=dict(I=i,U=u,TP=i,FP=p-i,FN=g-i,GT=g,predicted=p,
                    iou=100*i/max(u,1),**exact_curve(field,gt,valid))
    result=dict(id=rid,cohort=item['cohort'],qimg=item['qimg'],rimg=item['rimg'],apd=rec['apd'],
                prediction_sha256=rec['prediction_sha256'],arms=arms)
    rt.write(target,result); return result


def bootstrap(rows, seed, two_way=False, repeats=5000):
    names=list(rows[0]['arms']); I=np.array([[r['arms'][a]['I'] for a in names] for r in rows],float)
    U=np.array([[r['arms'][a]['U'] for a in names] for r in rows],float)
    photos=sorted({r['qimg'] for r in rows}|({r['rimg'] for r in rows} if two_way else set()))
    ids={p:i for i,p in enumerate(photos)}; qi=np.array([ids[r['qimg']] for r in rows])
    ri=np.array([ids[r['rimg']] for r in rows]) if two_way else None
    rng=np.random.default_rng(seed); output=np.empty((repeats,len(names)))
    for n in range(repeats):
        w=np.bincount(rng.integers(len(photos),size=len(photos)),minlength=len(photos))
        weights=(w[qi]*w[ri]) if two_way else w[qi]
        while not weights.any():
            w=np.bincount(rng.integers(len(photos),size=len(photos)),minlength=len(photos))
            weights=(w[qi]*w[ri]) if two_way else w[qi]
        output[n]=100*(weights@I)/np.maximum(weights@U,1)
    return names,output


def summarize(rows):
    report=dict(n=len(rows),layers=list(LAYERS),arms=list(ARMS),primary_family='3 single-factor effects x 3 cohorts at layer23',
                primary_interval_level=1-.05/9,cohorts={})
    for cohort in ('S','AVSeg','FIVES_train'):
        group=[r for r in rows if r['cohort']==cohort]
        names,photo=bootstrap(group,2026101675); _,image=bootstrap(group,2026101676,True)
        point={a:100*sum(r['arms'][a]['I'] for r in group)/max(sum(r['arms'][a]['U'] for r in group),1) for a in names}
        output=dict(n=len(group),query_photos=len({r['qimg'] for r in group}),point=point,
                    conditional_debias_applied={str(l):sum(r['apd'][str(l)] for r in group) for l in LAYERS},
                    single_factor_effects={},factorial={},metrics={})
        for a in names:
            totals={k:sum(r['arms'][a][k] for r in group) for k in ('TP','FP','FN','GT','predicted')}
            output['metrics'][a]=dict(**totals,precision=totals['TP']/max(totals['predicted'],1),recall=totals['TP']/max(totals['GT'],1),
                mean_episode_best_threshold_iou=float(np.mean([r['arms'][a]['best_threshold_iou'] for r in group])),
                mean_FP_per_GT_at_recall={str(t):float(np.mean([r['arms'][a]['at_recall'][str(t)]['FP_per_GT'] for r in group])) for t in RECALLS})
        for layer in LAYERS:
            anchor=f'L{layer}_111'; output['single_factor_effects'][str(layer)]={}
            for factor,control in (('soft_label','011'),('always_debias','101'),('joint_center','110')):
                a,b=names.index(anchor),names.index(f'L{layer}_{control}'); pa=photo[:,a]-photo[:,b]; im=image[:,a]-image[:,b]
                cuts=[100*.05/18,100*(1-.05/18)]
                output['single_factor_effects'][str(layer)][factor]=dict(current_minus_control_pp=point[anchor]-point[names[b]],
                    ci95_photo=np.percentile(pa,[2.5,97.5]).tolist(),ci95_image2way=np.percentile(im,[2.5,97.5]).tolist(),
                    family_ci_photo=np.percentile(pa,cuts).tolist(),family_ci_image2way=np.percentile(im,cuts).tolist())
            # Factorial coefficients use +/-1 coding; effects are 2*coefficient.
            for degree in (1,2,3):
                for subset in itertools.combinations(range(3),degree):
                    signs=np.array([np.prod([2*int(arm[i])-1 for i in subset]) for arm in ARMS],float)/4
                    cols=[names.index(f'L{layer}_{arm}') for arm in ARMS]
                    effects=photo[:,cols]@signs; effects2=image[:,cols]@signs
                    output['factorial'][f'L{layer}_'+','.join(map(str,subset))]=dict(effect_pp=float(np.array([point[names[i]] for i in cols])@signs),
                          ci95_photo=np.percentile(effects,[2.5,97.5]).tolist(),ci95_image2way=np.percentile(effects2,[2.5,97.5]).tolist())
        report['cohorts'][cohort]=output
    return report


def score():
    rt.initialize(); out=rt.OUT/'mechanism'; seal=rt.read(out/'sealed.json')
    assert seal['manifest_sha256']==rt.sha(rt.OUT/'manifest.json')
    rows=rt.read(rt.OUT/'manifest.json')['mechanism']; assert seal['n']==len(rows)
    if (out/'scored.json').exists():
        assert rt.sha(out/'summary.json')==rt.read(out/'scored.json')['summary_sha256']; return
    for item in rows:
        r=rt.valid_record(out/'records'/f'{item["id"]}.json'); assert r['prediction_sha256']==seal['predictions'][item['id']]
    result=[]
    for n,item in enumerate(rows):
        rt.checkpoint(); result.append(score_one(item)); rt.heartbeat('mechanism_score',n+1,len(rows))
    summary=summarize(result); rt.write(out/'summary.json',summary)
    rt.write(out/'scored.json',dict(n=len(rows),summary_sha256=rt.sha(out/'summary.json'),utc=rt.utc()))
