"""W62 paired point/prototype negative-dictionary interventions; W60 K is reused."""
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/x1012'),str(ROOT/'scripts/x1013')]
import common as c
import w59_normalization_causal as run
import w60_pure_background as kernel_control

OUT=c.task_path('W62')
ARMS=tuple(a+s for a in ('A0','K','Mclean','Joint') for s in ('','+'))
K_OUT=c.task_path('W60')


def clean_prototype(r,q,y):
    import torch
    import torch.nn.functional as F
    cov=y.numpy();qn,rn=F.normalize(q.float(),dim=1),F.normalize(r.float(),dim=1)
    fg=torch.from_numpy(cov>=.5 if (cov>=.5).any() else cov==cov.max())
    mf=F.normalize(rn[fg].mean(0),dim=0)
    bg_ids=torch.nonzero(~fg,as_tuple=True)[0];pure_ids=torch.nonzero(y==0,as_tuple=True)[0]
    original_k=max(1,int(.2*len(bg_ids)))
    if len(bg_ids):
        old_hard=bg_ids[(rn[bg_ids]@mf).topk(min(original_k,len(bg_ids))).indices]
        mixed=float((y[old_hard]>0).float().mean())
        old_bg=F.normalize(rn[old_hard].mean(0),dim=0)
        old_residual=float((old_bg-(old_bg@mf)*mf).norm())
    else:mixed=0.;old_residual=0.
    extras=dict(original_background_tokens=len(bg_ids),pure_background_tokens=len(pure_ids),
                original_hard_count=original_k if len(bg_ids) else 0,
                cleaned_hard_count=min(original_k,len(pure_ids)),mixed_share_original_hard=mixed,
                no_pure_background=not bool(len(pure_ids)),original_orthogonal_norm=old_residual)
    if not len(pure_ids):return None,extras
    hard=rn[pure_ids][(rn[pure_ids]@mf).topk(min(original_k,len(pure_ids))).indices]
    mb=F.normalize(hard.mean(0),dim=0);mb=mb-(mb@mf)*mf
    extras['cleaned_orthogonal_norm']=float(mb.norm())
    mb=mb/mb.norm().clamp_min(1e-8)
    return (qn@mf-.55*(qn@mb)).numpy().reshape(64,64),extras


def one(b):
    if time.time()>=run.DEADLINE:raise TimeoutError('Authorized window ended')
    import w23_simple_pipeline as sh
    import w19_simple_head as w19
    start=time.monotonic();oi,om=c.rc.independent,w19.mean_solve
    source=K_OUT/'predictions'/(c.key(b)+'.npz')
    with np.load(source,allow_pickle=False) as z:
        old={a:c.unpack(z[a]) for a in ('A0','A0+','PureBG','PureBG+')}
    masks={'K':old['PureBG'],'K+':old['PureBG+']}
    with c.rc.pc.inference_only(b):
        r,q,mask,y,fields=oi(b,{})
        cleaned,_,kmeta=kernel_control.changed_fields(r,q,y,fields)
        pclean,extra=clean_prototype(r,q,y)
        for a,ff,change_m in (('A0',fields,False),('Mclean',fields,True),('Joint',cleaned,True)):
            c.rc.independent=lambda *args,_f=ff,**kwargs:(r,q,mask,y,_f)
            w19.mean_solve=(lambda score,guide,lap:om(pclean if pclean is not None else score,guide,lap)) if change_m else om
            try:pre,post,rho=sh.simple_head(b,crf=True)
            finally:c.rc.independent,w19.mean_solve=oi,om
            masks[a],masks[a+'+']=pre,post
    assert np.array_equal(masks['A0'],old['A0']) and np.array_equal(masks['A0+'],old['A0+'])
    if rho>.9:
        assert np.array_equal(masks['Mclean'],masks['A0']) and np.array_equal(masks['Joint'],masks['K'])
        assert np.array_equal(masks['Mclean+'],masks['A0+']) and np.array_equal(masks['Joint+'],masks['K+'])
    extra.update(rho=rho,fallback=rho>.9,K_source_sha256=run.sha(source),kernel_c_error=kmeta['c_reproduction_max_error'])
    return run.save_episode(OUT,b,masks,extra,time.monotonic()-start)


def dependency_check():
    complete=json.loads((K_OUT/'complete.json').read_text())
    assert complete['n']==1285 and run.sha(K_OUT/'episodes.jsonl')==complete['episodes_sha256']
    records=list(map(json.loads,(K_OUT/'episodes.jsonl').read_text().splitlines()))
    assert all(run.sha(r['prediction'])==r['prediction_sha256'] for r in records)
    OUT.mkdir(exist_ok=True,parents=True)
    dep=dict(K_complete_sha256=run.sha(K_OUT/'complete.json'),K_source_sha256=run.sha(Path(kernel_control.__file__)),
             plan_sha256=run.sha(ROOT/'docs/research/x1012_tasks/W62_background_channels_PLAN.md'))
    p=OUT/'dependencies.json'
    if p.exists():assert json.loads(p.read_text())==dep
    else:run.write(p,dep)


if __name__=='__main__':
    if sys.argv[1]=='produce':
        dependency_check();run.run_produce(OUT,ARMS,one,int(sys.argv[2]) if len(sys.argv)>2 else 2)
    elif sys.argv[1]=='score':
        dependency_check();rows,summary=run.score_run(OUT,ARMS,'w62_background_channels_S.jsonl.gz')
        import score
        extra={}
        for post in ('','+'):
            extra[post or 'pre']={}
            for a,b in (('Joint','K'),('Joint','Mclean')):
                extra[post or 'pre'][a+'-'+b]=float(np.mean([
                    score.miou([r for r in rows if r['dataset']==d],a+post)-score.miou([r for r in rows if r['dataset']==d],b+post)
                    for d in c.DOMAINS]))
        summary['channel_comparisons']=extra
        summary['mean_mixed_share_in_original_hard_negative']=float(np.mean([r['mixed_share_original_hard'] for r in rows if not r['fallback']]))
        run.write(OUT/'summary.json',summary)
        print(json.dumps(extra,indent=2))
