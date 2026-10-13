"""Standalone reference-conditioned finite tangent-kernel segmentation on60 pairs."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import task_transfer_identifiability as common
import prepare_task_tangent_features as producer

np,torch,F,base=common.np,common.torch,common.F,common.base
OUT=base.ASSETS/'a/task_tangent_kernel60_20261010'
MODULE=base.REPO/'src/ics/methods/task_tangent_kernel.py'
ARMS=('o24_ridge','o24_position_ridge','hidden_ridge','task_tangent','mean_tangent')


def prepare(out):
    if (out/'config.json').exists():return
    complete=producer.read(producer.OUT/'COMPLETE.json');producer.verify(producer.OUT)
    assert complete['state']=='FROZEN_LAST_MLP_FEATURES_COMPLETE'
    assert all(r['old_O24_bit_exact'] and not r['query_label_access_attempts'] for r in complete['records']), 'Refresh external native controls if O24 is not identical'
    assert producer.read(producer.OUT/'jacobian_verification.json')['state']=='PASSED'
    out.mkdir(parents=True,exist_ok=True)
    for folder in ('fields','predictions','records','frozen'):(out/folder).mkdir(exist_ok=True)
    files=[Path(__file__),MODULE,Path(common.__file__),Path(producer.__file__)]
    for p in files:shutil.copyfile(p,out/'frozen'/p.name)
    shutil.copyfile(producer.OUT/'pairs.json',out/'pairs.json')
    traces={r['id']:r for r in map(json.loads,(base.OUT/'inference.jsonl').read_text().splitlines())}
    base.write(out/'native_controls.json',traces)
    base.write(out/'config.json',dict(n_pairs=60,variants=['A','B'],arms=ARMS,primary='task_tangent',
        source_sha256={str(p):base.sha(p) for p in files},pairs_sha256=base.sha(out/'pairs.json'),
        controls_sha256=base.sha(out/'native_controls.json'),feature_complete_sha256=base.sha(producer.OUT/'COMPLETE.json'),
        parameters_sha256=base.sha(producer.OUT/'parameters.npz'),basis_sha256=base.sha(base.ASSETS/'native_assets/positional_basis.pt'),
        formula='normalized finite kernel of last fc1/fc2 weights+bias for a fixed mask-defined unit-output scalar task',
        cotangents='task: fullR area-weighted FG-BG unit-output mean contrast; control: wholeR mean; stop-gradient through cotangent choice',
        supervision='same128 role-area occurrences per FG/BG, duplicated positions collapsed, centered weighted kernel ridge lambda=.01',
        controls='same-forward rawO24, position-projectedO24, newMLP hidden feature, same-cost mask-free tangent',
        renderer='signed64 bilinear1024 strict>0; direct complete mask, no CRF; original binary bilinear>0.5',
        positions='only o24_position_ridge projects the existing frozen500-dimensional positional basis; tangent is exact raw-output task differential',
        no_FoRIS_input=True,query_GT_in_inference=False,encoder_updates=0,
        old_O24_bit_exact_for_all_inputs=True,
        comparisons=['task_tangent - hidden_ridge','task_tangent - mean_tangent','task_tangent - o24_position_ridge','task_tangent - o24_ridge','task_tangent - complete native FoRIS'],
        exposure='same60 already-exposed task-switch development pairs, not official full protocol or untouched confirmation',
        statistic='mean of the two task IoUs per query photo; bootstrap linked reference/query-photo components',
        novelty='finite tangent transfer is prior art; mask-defined kernel requires both matched mechanism and complete performance evidence'))


def verify(out):
    cfg=producer.read(out/'config.json')
    for p,digest in cfg['source_sha256'].items():assert base.sha(p)==digest==base.sha(out/'frozen'/Path(p).name)
    for file,key in [('pairs.json','pairs_sha256'),('native_controls.json','controls_sha256')]:assert base.sha(out/file)==cfg[key]
    assert base.sha(producer.OUT/'COMPLETE.json')==cfg['feature_complete_sha256']
    assert base.sha(producer.OUT/'parameters.npz')==cfg['parameters_sha256']
    assert base.sha(base.ASSETS/'native_assets/positional_basis.pt')==cfg['basis_sha256']
    return cfg


def method(out):
    spec=importlib.util.spec_from_file_location('frozen_task_tangent_kernel',out/'frozen/task_tangent_kernel.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def load_data(key,bindings):
    record=bindings[key];path=producer.OUT/'raw'/record['file'];assert base.sha(path)==record['sha256']
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    for name,a in data.items():assert common.tensor_hash(a)==record['tensors'][name]['sha256']
    return data


def selected(data,ids):return {k:np.ascontiguousarray(v[ids]) for k,v in data.items()}


def run(out):
    prepare(out);verify(out)
    if (out/'sealed.json').exists():return
    model=method(out);pairs=producer.read(out/'pairs.json')
    bindings={r['key']:r for r in producer.read(producer.OUT/'COMPLETE.json')['records']}
    with np.load(producer.OUT/'parameters.npz') as z:params={k:z[k].copy() for k in z.files}
    basis=torch.load(base.ASSETS/'native_assets/positional_basis.pt',map_location='cpu',weights_only=True)['basis'].float()
    records=[];begin=time.monotonic()
    for i,pair in enumerate(pairs):
        rp=out/'records'/f'{i:03d}.json'
        if rp.exists():
            rec=producer.read(rp)
            for folder,key in [('fields','fields_sha256'),('predictions','prediction_sha256')]:assert base.sha(out/folder/rec['file'])==rec[key]
            records.append(rec);continue
        start=time.monotonic();attempts=[]
        with base.deny_labels(base.OUT,attempts),torch.inference_mode():
            data_r=load_data(pair['raw_binding']['reference']['key'],bindings)
            data_q=load_data(pair['raw_binding']['query']['key'],bindings);io_seconds=time.monotonic()-start
            er=F.normalize(torch.from_numpy(data_r['O24']),dim=1);eq=F.normalize(torch.from_numpy(data_q['O24']),dim=1)
            hr=F.normalize(torch.from_numpy(data_r['mlp_hidden']),dim=1);hq=F.normalize(torch.from_numpy(data_q['mlp_hidden']),dim=1)
            pr=F.normalize(er-(er@basis)@basis.T,dim=1);pq=F.normalize(eq-(eq@basis)@basis.T,dim=1)
            representations=dict(o24_ridge=(er,eq),o24_position_ridge=(pr,pq),hidden_ridge=(hr,hq))
            mean_axis=model.task_axis(data_r['O24'],None,'whole_reference')
            mean_q=model.factors(data_q,mean_axis,params)
            fields={};predictions={};fits={};axes={}
            for v in ('A','B'):
                ref=pair['variants'][v];coverage=common.coverage(ref['path'],ref['file_sha256'])
                sample=common.sample(er,coverage,np.ones(len(er),bool));assert sample is not None
                _,target,weights,ids=sample
                task_axis=model.task_axis(data_r['O24'],coverage)
                axes[v]=dict(task=task_axis.numpy().tolist(),whole_reference=mean_axis.numpy().tolist(),
                    direction_cosine=float(torch.dot(task_axis,mean_axis)),sample_ids=ids.tolist(),
                    weights=weights.numpy().tolist(),target=target.numpy().tolist())
                task_r=model.factors(selected(data_r,ids),task_axis,params)
                task_q=model.factors(data_q,task_axis,params)
                mean_r=model.factors(selected(data_r,ids),mean_axis,params)
                for arm in ARMS:
                    tick=time.monotonic()
                    if arm in representations:
                        r,q=representations[arm];rg=r[ids].to('mps')
                        gram=(rg@rg.T).cpu();qgram=(q.to('mps')@rg.T).cpu()
                    else:
                        r,q=(task_r,task_q) if arm=='task_tangent' else (mean_r,mean_q)
                        gram=model.kernel(r,r).cpu();qgram=model.kernel(q,r).cpu()
                    score,info=model.kernel_ridge(gram,qgram,target,weights)
                    assert np.isfinite(score).all()
                    fields[v+'/'+arm]=score.reshape(64,64);fields[v+'/'+arm+'/reference_gram']=gram.numpy()
                    fields[v+'/'+arm+'/query_gram_probe']=qgram.numpy()[[0,1365,2730,4095]]
                    pixels=F.interpolate(torch.from_numpy(score).reshape(1,1,64,64),(1024,1024),mode='bilinear',align_corners=False)[0,0]>0
                    for frame,hw in [('cli1024',(1024,1024)),('original',tuple(pair['row']['query_size_hw']))]:
                        mask=pixels if hw==(1024,1024) else F.interpolate(pixels.float()[None,None],hw,mode='bilinear',align_corners=False)[0,0]>.5
                        predictions[frame+'/'+v+'/'+arm]=np.packbits(mask.numpy())
                    fits[v+'/'+arm]=info|dict(seconds=time.monotonic()-tick,gram_diagonal_min=float(gram.diagonal().min()),
                        gram_diagonal_max=float(gram.diagonal().max()),gram_symmetry_max=float((gram-gram.T).abs().max()))
                del task_r,task_q,mean_r
            file=f'{i:03d}.npz';np.savez_compressed(out/'fields'/file,**fields);np.savez_compressed(out/'predictions'/file,**predictions)
            assert not attempts;driver=torch.mps.driver_allocated_memory();assert driver<20e9
            del mean_q
        rec=dict(index=i,id=pair['id'],dataset=pair['dataset'],file=file,fields_sha256=base.sha(out/'fields'/file),
            prediction_sha256=base.sha(out/'predictions'/file),axes=axes,fits=fits,io_seconds=io_seconds,
            seconds=time.monotonic()-start,mps_driver_bytes=driver,query_GT_reads=0,encoder_updates=0)
        base.write(rp,rec);records.append(rec);torch.mps.empty_cache()
        base.write(out/'activity.json',dict(state='INFERENCE',pid=__import__('os').getpid(),completed=i+1,total=60))
        print(json.dumps(dict(completed=i+1,seconds=rec['seconds'],driver_gib=driver/2**30)),flush=True)
    verify(out);base.write(out/'sealed.json',dict(state='ALL120_TASKS_AND600_COMPLETE_MASKS_SEALED',n_pairs=60,records=records,
        config_sha256=base.sha(out/'config.json'),seconds=time.monotonic()-begin,query_GT_reads=0,encoder_updates=0))


def photo_components(pairs):
    parent={}
    def root(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=root(parent[x])
        return parent[x]
    for p in pairs:
        a,b=(p['dataset']+'/'+str(p['row'][role+'_photo_id']) for role in ('reference','query'))
        parent[root(a)]=root(b)
    return {p['id']:root(p['dataset']+'/'+str(p['row']['query_photo_id'])) for p in pairs}


def group_bootstrap(rows,values,groups):
    ids=sorted({groups[r['id']] for r in rows});buckets=[np.array([i for i,r in enumerate(rows) if groups[r['id']]==g]) for g in ids]
    value=np.asarray(values,dtype=float);rng=np.random.default_rng(0);boot=[]
    for _ in range(2000):
        chosen=np.concatenate([buckets[i] for i in rng.integers(0,len(ids),len(ids))]);boot.append(value[chosen].mean())
    return dict(mean=float(value.mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist(),n_pairs=len(rows),photo_components=len(ids))


def score(out):
    verify(out);seal=producer.read(out/'sealed.json');assert seal['config_sha256']==base.sha(out/'config.json')
    pairs=producer.read(out/'pairs.json');truth=producer.read(base.OUT/'query_truth.json');native=producer.read(out/'native_controls.json')
    records=[];names=(*ARMS,'foris');groups=photo_components(pairs)
    for pair,rec in zip(pairs,seal['records']):
        assert pair['id']==rec['id'];parent=native[pair['id']]
        assert base.sha(parent['file'])==parent['sha256']
        for folder,key in [('fields','fields_sha256'),('predictions','prediction_sha256')]:assert base.sha(out/folder/rec['file'])==rec[key]
        tasks=[]
        with np.load(out/'predictions'/rec['file']) as masks,np.load(out/'fields'/rec['file']) as fields,np.load(parent['file']) as old:
            for v in ('A','B'):
                qb=truth[pair['id']]['masks'][v];assert base.sha(qb['path'])==qb['sha256']
                gt0=base.read_mask(qb['path']).astype('float32');baseline=np.unpackbits(old[v+'/crf'],count=1024**2).reshape(1024,1024).astype(bool)
                metrics={};ranks={}
                for frame,hw in [('cli1024',(1024,1024)),('original',tuple(pair['row']['query_size_hw']))]:
                    gt=F.interpolate(torch.from_numpy(gt0)[None,None],hw,mode='nearest')[0,0].numpy().astype(bool);metrics[frame]={}
                    for name in names:
                        if name=='foris':pred=baseline if hw==(1024,1024) else F.interpolate(torch.from_numpy(baseline).float()[None,None],hw,mode='bilinear',align_corners=False)[0,0].numpy()>.5
                        else:pred=np.unpackbits(masks[frame+'/'+v+'/'+name],count=np.prod(hw)).reshape(hw).astype(bool)
                        I=int((pred&gt).sum());U=int((pred|gt).sum())
                        metrics[frame][name]=dict(I=I,U=U,FP=int((pred&~gt).sum()),FN=int((~pred&gt).sum()),iou=100*I/max(U,1))
                    if frame=='cli1024':
                        for roi_name,roi in [('whole',np.ones_like(gt)),('inside_external_foris',baseline),('outside_external_foris',~baseline)]:
                            f=(gt&roi).reshape(64,16,64,16).sum((1,3)).ravel();b=(~gt&roi).reshape(64,16,64,16).sum((1,3)).ravel()
                            ranks[roi_name]=None if not f.sum() or not b.sum() else {a:base.weighted_auc(fields[v+'/'+a].ravel(),f,b) for a in ARMS}
                tasks.append(dict(variant=v,metrics=metrics,ranking=ranks))
        records.append(dict(id=pair['id'],dataset=pair['dataset'],tasks=tasks))
    summary={}
    for dataset in sorted({r['dataset'] for r in records}):
        rows=[r for r in records if r['dataset']==dataset];summary[dataset]=dict(n_pairs=len(rows),frames={},ranking={},photo_components=len({groups[r['id']] for r in rows}))
        for frame in ('cli1024','original'):
            values={a:[float(np.mean([t['metrics'][frame][a]['iou'] for t in r['tasks']])) for r in rows] for a in names}
            summary[dataset]['frames'][frame]=dict(paired_photo_mean_iou={a:float(np.mean(v)) for a,v in values.items()},
                task_tangent_minus={control:group_bootstrap(rows,np.asarray(values['task_tangent'])-values[control],groups) for control in names if control!='task_tangent'},
                variants={v:{a:float(np.mean([next(t for t in r['tasks'] if t['variant']==v)['metrics'][frame][a]['iou'] for r in rows])) for a in names} for v in ('A','B')})
        for roi in ('whole','inside_external_foris','outside_external_foris'):
            valid=[r for r in rows if all(t['ranking'][roi] is not None for t in r['tasks'])]
            means={a:[float(np.mean([t['ranking'][roi][a] for t in r['tasks']])) for r in valid] for a in ARMS}
            summary[dataset]['ranking'][roi]=dict(n_pairs=len(valid),auc={a:float(np.mean(v)) for a,v in means.items()},
                task_tangent_minus={control:group_bootstrap(valid,np.asarray(means['task_tangent'])-means[control],groups) for control in ARMS if control!='task_tangent'})
    base.write(out/'report.json',dict(state='COMPLETE',datasets=summary,records=records,
        statistic='per-photo mean of two task IoUs; not official class mIoU. Paired bootstrap connected reference/query-photo components.',
        seal_sha256=base.sha(out/'sealed.json'),config_sha256=base.sha(out/'config.json'),encoder_updates=0))
    base.write(out/'activity.json',dict(state='COMPLETE',n_pairs=60))
    print(json.dumps({d:{'cli':s['frames']['cli1024']['paired_photo_mean_iou'],'ranking':{roi:x['auc'] for roi,x in s['ranking'].items()}} for d,s in summary.items()}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','score','all']);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    torch.mps.set_per_process_memory_fraction(20e9/torch.mps.recommended_max_memory())
    if a.mode=='prepare':prepare(a.out)
    if a.mode in ('run','all'):run(a.out)
    if a.mode in ('score','all'):score(a.out)


if __name__=='__main__':main()
