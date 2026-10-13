"""Single-reference pixel residual, preserving every 16x16 prediction area.

Only the supplied reference mask fits a twelve-slope pairwise classifier and
selects its strength on a spatial reference holdout. No DINO/other model runs.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import types

for _env in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_env]='2'
import numpy as np
import torch
from scipy.special import expit
import lab
import plan_common as pc
import plan_report as pr

OUT=lab.ASSETS/'a/x1012_R2'
B4=lab.ASSETS/'a/x1012_B4/seed150'
MOD=lab.ASSETS/'a/x1012_B4_modulated/region/seed150'
ASSEMBLY=lab.ASSETS/'a/x1012_B4_assembly/seed150'
PIXEL_ORIGIN=lab.ROOT/'evidence/local/autonomous_20261010/physical_pixel_evidence/frozen/diagnose.py'
PIXEL_SOURCE=OUT/'frozen/4_diagnose.py'
ALPHAS=(0.,.25,1.)
ARMS=('VP_identity','pixel','pixel_fixed1','pixel_permuted','patch_constant')


def initialize():
    global pixel
    pc.setup('cpu');torch.set_num_threads(2)
    # The concurrent repository cleanup removed the legacy source. Its exact
    # pre-run sealed copy remains available; keep its original relative paths.
    pixel=types.ModuleType('r2_existing_pixel_descriptor')
    pixel.__file__=str(PIXEL_ORIGIN)
    exec(compile(PIXEL_SOURCE.read_text(),str(PIXEL_SOURCE),'exec'),pixel.__dict__)


def blocks(array):
    tail=array.shape[2:]
    return array.reshape(64,16,64,16,*tail).transpose(0,2,1,3,*range(4,4+len(tail))).reshape(4096,256,*tail)


def unblocks(array):
    return array.reshape(64,64,16,16).transpose(0,2,1,3).reshape(1024,1024)


def standardize_within(array):
    x=blocks(array).astype(np.float32)
    return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)


def pairs(descriptor,label,allowed):
    x,y,keep=blocks(descriptor),blocks(label),blocks(allowed)
    differences=[]
    for i in range(4096):
        fg=np.flatnonzero(y[i]&keep[i]);bg=np.flatnonzero(~y[i]&keep[i])
        if not len(fg) or not len(bg):continue
        fi=fg[((np.arange(8)+.5)*len(fg)/8).astype(int)]
        bi=bg[((np.arange(8)+.5)*len(bg)/8).astype(int)]
        differences.append(x[i,fi]-x[i,bi])
    return np.concatenate(differences).astype(np.float64) if differences else np.empty((0,12))


def fit_pairwise(differences):
    if len(differences)<16:
        return np.zeros(12),{'pairs':len(differences),'fallback':'insufficient mixed reference blocks'}
    scale=np.maximum(np.sqrt(np.mean(differences**2,axis=0)),1e-3)
    x=differences/scale
    beta=np.zeros(12);ridge=.01
    def loss(v):return float(np.logaddexp(0,-x@v).mean()+.5*ridge*(v@v))
    for step in range(40):
        p=expit(x@beta)
        grad=x.T@(p-1)/len(x)+ridge*beta
        if np.max(np.abs(grad))<1e-8:break
        h=(x.T*(p*(1-p)))@x/len(x)+ridge*np.eye(12)
        direction=np.linalg.solve(h,grad);eta=1.;before=loss(beta)
        for _ in range(30):
            candidate=beta-eta*direction
            if loss(candidate)<=before:
                beta=candidate;break
            eta*=.5
        else:raise RuntimeError('Reference pairwise fit failed line search')
    residual=float(np.max(np.abs(x.T@(expit(x@beta)-1)/len(x)+ridge*beta)))
    assert residual<1e-7
    return beta/scale,{'pairs':len(x),'steps':step+1,'gradient_max_abs':residual,'objective':loss(beta)}


def select_alpha(reference_semantic,residual,label,holdout):
    semantic=standardize_within(reference_semantic)
    detail=standardize_within(residual)
    y,allowed=blocks(label),blocks(holdout)
    eligible=[]
    for i in range(4096):
        ids=np.flatnonzero(allowed[i]);k=int(y[i,ids].sum())
        if 0<k<len(ids):eligible.append((i,ids,k))
    totals=[]
    for alpha in ALPHAS:
        tp=0
        for i,ids,k in eligible:
            score=semantic[i,ids]+alpha*detail[i,ids]
            selected=ids[np.argsort(-score,kind='stable')[:k]]
            tp+=int(y[i,selected].sum())
        totals.append(tp)
    selected=int(np.argmax(totals))
    return ALPHAS[selected],{'alphas':ALPHAS,'heldout_mixed_blocks':len(eligible),
                             'reference_heldout_TP_at_true_block_area':totals,'selected':ALPHAS[selected]}


def rerank(semantic,pixel_residual,counts,alpha,baseline=None):
    if alpha == 0 and baseline is not None:
        assert np.array_equal(blocks(baseline).sum(1),counts)
        return baseline.copy()
    score=semantic+alpha*pixel_residual
    order=np.argsort(-score,axis=1,kind='stable')
    pred=np.zeros((4096,256),bool)
    pred[np.arange(4096)[:,None],order]=np.arange(256)[None,:]<counts[:,None]
    assert np.array_equal(pred.sum(1),counts)
    return unblocks(pred)


def produce_one(binding):
    name=pc.key(binding);record_path=OUT/'records'/f'{name}.json'
    if record_path.exists():
        r=pc.read(record_path);assert pc.sha(r['prediction_path'])==r['prediction_sha256'];return r
    started=time.monotonic()
    with pc.inference_only(binding):
        with np.load(lab.ASSETS/'a/x1012_P0/fields/data'/f'{name}.npz',allow_pickle=False) as z:
            rho=float(z['ref_stats'][2]);field=z['c32'].reshape(64,64).copy() if rho>.9 else None
        if field is None:
            with np.load(MOD/'fields'/f'{name}.npz',allow_pickle=False) as z:field=z['field_VP'].copy()
        continuous=lab.up(torch.as_tensor(field)).numpy()
        rendered=pc.readout(field)
        with np.load(ASSEMBLY/'predictions'/f'{name}.npz',allow_pickle=False) as z:
            baseline=pc.unpack(z['VP_thin09'])
        render_xor=int(np.count_nonzero(rendered!=baseline))
        counts=blocks(baseline).sum(1)
        semantic=standardize_within(continuous)
        identity=rerank(semantic,np.zeros_like(semantic),counts,0.,baseline)
        assert np.array_equal(identity,baseline),(name,'same-area identity')
        reference_rgb=pixel.rgb(binding['row'],'reference')
        label=lab.mask1024(binding['row']['reference_mask_path']).numpy()
        train,holdout=pixel.source_split()
        descriptor=pixel.descriptor(reference_rgb)
        beta,fit=fit_pairwise(pairs(descriptor,label,train))
        reference_semantic=lab.up(lab.coverage64(torch.from_numpy(label))).numpy()
        alpha,selection=select_alpha(reference_semantic,descriptor@beta.astype(np.float32),label,holdout)
        rng=np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:16],16))
        label_blocks=blocks(label)
        shuffled=np.take_along_axis(label_blocks,np.argsort(rng.random((4096,256)),axis=1),axis=1)
        shuffled=unblocks(shuffled)
        beta_perm,fit_perm=fit_pairwise(pairs(descriptor,shuffled,train))
        alpha_perm,selection_perm=select_alpha(reference_semantic,descriptor@beta_perm.astype(np.float32),label,holdout)
        del descriptor
        query_descriptor=pixel.descriptor(pixel.rgb(binding['row'],'query'))
        residual=standardize_within(query_descriptor@beta.astype(np.float32))
        residual_perm=standardize_within(query_descriptor@beta_perm.astype(np.float32))
        predictions={'VP_identity':identity,'pixel':rerank(semantic,residual,counts,alpha,baseline),
                     'pixel_fixed1':rerank(semantic,residual,counts,1.),
                     'pixel_permuted':rerank(semantic,residual_perm,counts,alpha_perm,baseline),
                     'patch_constant':identity.copy()}
        assert all(np.array_equal(blocks(p).sum(1),counts) for p in predictions.values())
        output=OUT/'predictions'/f'{name}.npz'
        assert not output.exists()
        np.savez_compressed(output,**{a:pc.pack(p) for a,p in predictions.items()})
    r={'index':name,'dataset':binding['dataset'],'episode_id':binding['row']['episode_id'],
       'prediction_path':str(output),'prediction_sha256':pc.sha(output),'seconds':time.monotonic()-started,
       'beta':beta.tolist(),'beta_permuted':beta_perm.tolist(),'fit':fit,'fit_permuted':fit_perm,
       'selection':selection,'selection_permuted':selection_perm,'mixed_query_blocks':int(((counts>0)&(counts<256)).sum()),
       'same_area_identity_xor':0,'CPU_vs_saved_field_render_xor':render_xor,'all_patch_areas_preserved':True,'reference_count':1,
       'new_encoder_calls':0,'inference_query_GT_reads':0,'FoRIS_score_consumed':False}
    pc.write(record_path,r);return r


def prepare():
    for name in ('records','predictions','frozen'):(OUT/name).mkdir(parents=True,exist_ok=True)
    path=OUT/'config.json'
    if not path.exists():
        sources=(Path(__file__),Path(pc.__file__),Path(pr.__file__),Path(lab.__file__),PIXEL_SOURCE)
        pc.write(path,{'state':'FROZEN_BEFORE_NATURAL_OUTPUTS','n':1360,'arms':ARMS,'reference_count':1,
                       'fit':'12-slope pairwise logistic, ridge.01, no intercept;8 deterministic FG/BG pairs per mixed block',
                       'reference_split':'original64tile checkerboard interior with8pixel guard',
                       'selection':{'alphas':ALPHAS,'criterion':'reference spatial-holdout TP at true within-block count; lower alpha wins ties'},
                       'output':'same per16x16 block foreground count; rank local-standardized semantic+alpha*local-standardized pixel residual',
                       'control':'shuffle reference labels within each16x16 block; same fit and real-reference holdout selection',
                       'iSAID':'pixel counts preserved before ignore filtering; valid-area metrics need not be equal',
                       'source_sha256':{str(p):pc.sha(p) for p in sources},'new_encoder_calls':0})
        for i,p in enumerate(sources):shutil.copyfile(p,OUT/'frozen'/f'{i}_{p.name}')
        shutil.copyfile(B4/'manifest.json',OUT/'manifest.json')
    config=pc.read(path);assert all(pc.sha(p)==h for p,h in config['source_sha256'].items()),'R2 source changed'
    return pc.read(OUT/'manifest.json')


def produce(workers):
    rows=prepare();start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for i,r in enumerate(pool.map(produce_one,rows,chunksize=1),1):
            records.append(r)
            if i<=3 or i%50==0:print({'phase':'R2-produce','completed':i,'n':len(rows),'seconds':round(time.monotonic()-start,1)},flush=True)
    pc.write(OUT/'sealed.json',{'state':'ALL_NATURAL_OUTPUTS_SEALED','n':len(rows),'records':records,
                              'manifest_sha256':pc.sha(OUT/'manifest.json'),'config_sha256':pc.sha(OUT/'config.json'),
                              'wall_seconds':time.monotonic()-start,'case_seconds_sum':sum(r['seconds'] for r in records),
                              'workers':workers,'threads_per_worker':2,'new_encoder_calls':0,'inference_query_GT_reads':0})


def score_one(job):
    binding,receipt=job
    with np.load(receipt['prediction_path'],allow_pickle=False) as z:predictions={a:pc.unpack(z[a]) for a in ARMS}
    record=pr.score_record(binding,predictions);gt,valid=pc.truth(binding)
    baseline=predictions['VP_identity']
    counts=blocks(baseline).sum(1);g=blocks(gt).sum(1);ignored=blocks(~valid).sum(1)
    tp=int(np.minimum(counts,g).sum());fp=int(np.maximum(0,counts-g-ignored).sum());fn=int(g.sum())-tp
    oracle=dict(record['metrics']['VP_identity']);oracle.update(I=tp,U=tp+fp+fn,TP=tp,FP=fp,FN=fn,predicted=tp+fp,IoU=100*tp/max(tp+fp+fn,1))
    oracle['abs_log2_area_error']=float(abs(np.log2((tp+fp)/max(int(g.sum()),1)))) if tp+fp else None
    oracle['zero_predicted_area']=tp+fp==0;oracle['infinite_area_error']=tp+fp==0
    record['metrics']['fixed_block_area_oracle']=oracle
    record['edits']={a:{'add_TP':int((p&~baseline&gt&valid).sum()),'add_FP':int((p&~baseline&~gt&valid).sum()),
                        'delete_TP':int((~p&baseline&gt&valid).sum()),'delete_FP':int((~p&baseline&~gt&valid).sum())} for a,p in predictions.items()}
    record['alpha']=receipt['selection']['selected'];record['alpha_permuted']=receipt['selection_permuted']['selected']
    record['all_patch_areas_preserved']=all(np.array_equal(blocks(p).sum(1),counts) for p in predictions.values())
    return record


def score(workers):
    initialize();rows=prepare();seal=pc.read(OUT/'sealed.json')
    assert seal['state']=='ALL_NATURAL_OUTPUTS_SEALED' and seal['n']==len(rows)
    assert pc.sha(OUT/'config.json')==seal['config_sha256'] and pc.sha(OUT/'manifest.json')==seal['manifest_sha256']
    receipts={r['index']:r for r in seal['records']}
    for r in receipts.values():assert pc.sha(r['prediction_path'])==r['prediction_sha256']
    start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for i,r in enumerate(pool.map(score_one,((b,receipts[pc.key(b)]) for b in rows),chunksize=1),1):
            records.append(r)
            if i<=3 or i%100==0:print({'phase':'R2-score','completed':i,'n':len(rows),'seconds':round(time.monotonic()-start,1)},flush=True)
    controls={r['index']:r for r in pc.read(B4/'report.json')['records']}
    prior={r['index']:r for r in pc.read(ASSEMBLY/'report.json')['records']}
    for r in records:
        assert r['all_patch_areas_preserved']
        assert r['metrics']['VP_identity']==prior[r['index']]['metrics']['VP_thin09']
        for a in ('FoRIS','MEAN'):r['metrics'][a]=controls[r['index']]['metrics'][a]
    arms=('FoRIS','MEAN')+ARMS+('fixed_block_area_oracle',)
    report=pr.report('R2',OUT,records,arms,[
        '单参考混合块内pairwise logistic，只有参考Mask参与拟合和空间留出强度选择；无新编码/无跨任务训练。',
        '每个16x16块严格保留旧VP_thin09前景数，identity逐位一致；GT仅在完整自然封存后评分。',
        '真实像素、固定alpha1、参考标签块内置换及块常数对照完整报告；oracle只诊断固定面积下位置潜力。',
        '全图各块面积恒等；iSAID跨ignore位置的编辑可能改变有效像素面积，不能据此称范围改善。'],
        extra={'production':{k:v for k,v in seal.items() if k!='records'},'score_seconds':time.monotonic()-start})
    contrasts={}
    for baseline in ('VP_identity','pixel_permuted','pixel_fixed1'):
        rows2=[dict(r,metrics={'FoRIS':r['metrics'][baseline],'pixel':r['metrics']['pixel']}) for r in records]
        contrasts['pixel_minus_'+baseline]=pr.summary(rows2,('FoRIS','pixel'))
    pc.write(OUT/'mechanism_contrasts.json',contrasts)
    passing=[a for a in ('pixel','pixel_fixed1') if all(t[a]['delta_to_foris_pp']>=-.5 for t in report['summary'].values())]
    pc.write(OUT/'completion.json',{'state':'COMPLETE','n':len(rows),'ten_domain_gate_passing':passing,
                                  'all_patch_areas_preserved':True,'new_encoder_calls':0,'total_score_seconds':time.monotonic()-start})
    print({'phase':'R2-complete','passing':passing},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('produce','score'))
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    produce(args.workers) if args.mode=='produce' else score(args.workers)
