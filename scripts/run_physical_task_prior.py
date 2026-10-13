"""Frozen40 standalone task-prior candidate and matched covariance controls."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task_view_compatibility as parent
import task_transfer_identifiability as common

np,torch,F,base=parent.np,parent.torch,parent.F,parent.base
OUT=base.ASSETS/'a/physical_task_prior40_20261010'
MODULE=base.REPO/'src/ics/methods/physical_task_prior.py'
ARMS=('reference_only','query_variance','physical_prior','pairing_null')


def prepare(out):
    if (out/'config.json').exists():return
    assert (parent.OUT/'report.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    for name in ('fields','predictions','records','frozen'):(out/name).mkdir(exist_ok=True)
    for p in (Path(__file__),MODULE,Path(parent.__file__),Path(common.__file__)):
        shutil.copyfile(p,out/'frozen'/p.name)
    shutil.copyfile(parent.OUT/'tasks.json',out/'tasks.json')
    base.write(out/'config.json',dict(n=40,arms=ARMS,primary='physical_prior/otsu',
        mechanism='physical-view agreement as an unlabeled prior on the reference task functional; external benchmark only, no FoRIS inputs',
        spectrum='positive generalized cross-covariance relative to exact independent-pair second moment, ridge=trace(B)/D',
        fitting='source physical area regression; all4096 legal reference labels; ridge=trace(C_R)/D; no global learned model',
        output='whole/fine least-squares latent field; exact query two-means split primary; coverage0.5 as declared calibration control',
        calibration_assumption='two score populations; not a proven posterior or area estimate',
        comparisons=['physical_prior - reference_only','physical_prior - query_variance','physical_prior - pairing_null'],
        source_sha256={str(p):base.sha(p) for p in (Path(__file__),MODULE,Path(parent.__file__),Path(common.__file__))},
        tasks_sha256=base.sha(out/'tasks.json'),diagnostic_report_sha256=base.sha(parent.OUT/'report.json'),
        query_GT_in_inference=False,new_encodings=0,device='MPS covariance, CPU FP64 eigensystems/solves',
        exposure='same40 hypothesis-development images; not independent confirmation',
        originality='candidate mechanism, no originality or performance claim before matched tests and literature scrutiny'))


def verify(out):
    c=common.read(out/'config.json')
    for p,digest in c['source_sha256'].items():assert base.sha(p)==digest==base.sha(out/'frozen'/Path(p).name)
    assert base.sha(out/'tasks.json')==c['tasks_sha256']
    return c


def module(out):
    spec=importlib.util.spec_from_file_location('frozen_physical_task_prior',out/'frozen/physical_task_prior.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def run(out):
    prepare(out);verify(out)
    if (out/'sealed.json').exists():return
    method=module(out);tasks=common.read(out/'tasks.json');denied=[t['row']['query_mask_path'] for t in tasks]
    records=[];start=time.monotonic()
    for i,t in enumerate(tasks):
        row,task=t['row'],t['task'];rp=out/'records'/f'{i:03d}.json'
        if rp.exists():
            rec=common.read(rp)
            assert base.sha(out/'predictions'/rec['file'])==rec['prediction_sha256'];records.append(rec);continue
        tick=time.monotonic()
        with parent.binding.deny_paths(denied+[t['task']['baseline']['path'] for t in tasks]) as guard,torch.inference_mode():
            r,q,native,_=parent.raw_inputs(task);c=parent.coverage(row['reference_mask_path'],64)
            loaded=time.monotonic();value=method.predict(r,c,q,native)
            computed=time.monotonic();predictions={};fields={}
            for arm in ARMS:
                for name,field in value['fields'][arm].items():fields[arm+'/'+name]=field
                for name,a in value['heads'][arm].items():fields[arm+'/'+name]=a
                latent=torch.from_numpy(value['fields'][arm]['latent'])
                pixels=F.interpolate(latent[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]
                for renderer,threshold in [('otsu',value['fits'][arm]['otsu_threshold']),('half',.5)]:
                    mask=pixels>threshold
                    for frame,hw in [('cli1024',(1024,1024)),('original',tuple(row['query_size_hw']))]:
                        binary=mask if hw==(1024,1024) else F.interpolate(mask.float()[None,None],hw,mode='bilinear',align_corners=False)[0,0]>.5
                        predictions[frame+'/'+arm+'/'+renderer]=np.packbits(binary.numpy())
            file=f'{i:03d}.npz';np.savez_compressed(out/'fields'/file,**fields);np.savez_compressed(out/'predictions'/file,**predictions)
            assert guard['attempts']==0
            driver=torch.mps.driver_allocated_memory();assert driver<20e9
        rec=dict(index=i,id=row['episode_id'],dataset=row['dataset'],file=file,
            prediction_sha256=base.sha(out/'predictions'/file),fields_sha256=base.sha(out/'fields'/file),
            fits=value['fits'],prior=value['prior'],io_seconds=loaded-tick,model_seconds=computed-loaded,
            seconds=time.monotonic()-tick,mps_driver_bytes=driver,query_label_access_attempts=guard['attempts'],new_encodings=0)
        base.write(rp,rec);records.append(rec);torch.mps.empty_cache()
        base.write(out/'activity.json',dict(state='INFERENCE',controller_pid=__import__('os').getpid(),completed=i+1,n=40,seconds=time.monotonic()-start))
        print(json.dumps(dict(stage='PREDICT',n=i+1,seconds=rec['seconds'])),flush=True)
    verify(out);base.write(out/'sealed.json',dict(state='ALL40_COMPLETE_PREDICTIONS_SEALED',n=40,records=records,
        seconds=time.monotonic()-start,config_sha256=base.sha(out/'config.json'),query_GT_reads=0,new_encodings=0))


def aggregate(records,frame,names):
    grouped={name:{} for name in names}
    for r in records:
        for name in names:
            slot=r['fold'],r['class_id'];old=grouped[name].setdefault(slot,np.zeros(2,dtype=np.int64))
            old+=r['metrics'][frame][name]['I'],r['metrics'][frame][name]['U']
    result={}
    for name,slots in grouped.items():
        folds={}
        for (f,c),(I,U) in slots.items():folds.setdefault(f,[]).append(100*I/U if U else 0)
        result[name]=float(np.mean([np.mean(v) for v in folds.values()]))
    return result


def score(out):
    verify(out);seal=common.read(out/'sealed.json');assert seal['config_sha256']==base.sha(out/'config.json')
    tasks=common.read(out/'tasks.json');records=[];names=[a+'/'+r for a in ARMS for r in ('otsu','half')]+['foris','mean','region.fast']
    for t,rec in zip(tasks,seal['records']):
        row,task=t['row'],t['task'];assert rec['id']==row['episode_id']
        assert base.sha(out/'predictions'/rec['file'])==rec['prediction_sha256']
        assert base.sha(task['baseline']['path'])==task['baseline']['sha256']
        truth=base.read_mask(row['query_mask_path']).astype('float32');metrics={}
        with np.load(out/'predictions'/rec['file']) as z,np.load(task['baseline']['path']) as old:
            for frame,hw in [('cli1024',(1024,1024)),('original',tuple(row['query_size_hw']))]:
                gt=F.interpolate(torch.from_numpy(truth)[None,None],hw,mode='nearest')[0,0].numpy().astype(bool)
                metrics[frame]={}
                for name in names:
                    if name=='foris':a=old[frame+'/foris.crf']
                    elif name=='mean':a=old[frame+'/mean']
                    elif name=='region.fast':a=old[frame+'/region.fast']
                    else:a=z[frame+'/'+name]
                    pred=np.unpackbits(a,count=np.prod(hw)).reshape(hw).astype(bool)
                    I=int((pred&gt).sum());U=int((pred|gt).sum())
                    metrics[frame][name]=dict(I=I,U=U,FP=int((pred&~gt).sum()),FN=int((~pred&gt).sum()),iou=100*I/max(U,1))
        records.append(dict(id=rec['id'],dataset=row['dataset'],fold=row['fold'],class_id=row['loader_class_id'],
            query_photo_id=row['query_photo_id'],reference_photo_id=row['reference_photo_id'],metrics=metrics))
    summary={}
    for d in sorted({r['dataset'] for r in records}):
        sub=[r for r in records if r['dataset']==d]
        summary[d]=dict(n=len(sub),query_photos=len({r['query_photo_id'] for r in sub}),
            observed_class_slots=len({(r['fold'],r['class_id']) for r in sub}),frames={})
        for frame in ('cli1024','original'):
            summary[d]['frames'][frame]=dict(miou=aggregate(sub,frame,names),
                episode_iou={name:float(np.mean([r['metrics'][frame][name]['iou'] for r in sub])) for name in names},
                paired_episode_differences={name:base.bootstrap([r['metrics'][frame]['physical_prior/otsu']['iou']-r['metrics'][frame][name]['iou'] for r in sub])
                    for name in ('reference_only/otsu','query_variance/otsu','pairing_null/otsu','foris')})
    base.write(out/'report.json',dict(state='COMPLETE',datasets=summary,records=records,
        statistic='Deep pooled foreground I/U; PACO observed fold/class pooled I/U then fold means; paired intervals are episode-IoU diagnostic, not class-mIoU uncertainty',
        new_encodings=0,seal_sha256=base.sha(out/'sealed.json'),config_sha256=base.sha(out/'config.json')))
    base.write(out/'activity.json',dict(state='COMPLETE',n=40))
    print(json.dumps({d:{f:x['miou'] for f,x in s['frames'].items()} for d,s in summary.items()}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','score','all']);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    torch.mps.set_per_process_memory_fraction(20e9/torch.mps.recommended_max_memory())
    if a.mode=='prepare':prepare(a.out)
    if a.mode in ('run','all'):run(a.out)
    if a.mode in ('score','all'):score(a.out)


if __name__=='__main__':main()
