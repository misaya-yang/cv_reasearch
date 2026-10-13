#!/usr/bin/env python3
"""Frozen supported-MEAN additions to existing full consensus on 2380 pairs."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,resource,shutil,sys,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(k,'2')
import numpy as np
import torch
import torch.nn.functional as F

REPO=Path(__file__).resolve().parents[1]
DATA=REPO.parent/'cv_data'
ROOT=REPO/'evidence/local/insid3_foris_generalization_20261010'
PROTOCOL=ROOT/'consensus_supported_foreground_protocol.json'
DUAL=DATA/'a/consensus_dual_frame1020_v1_20261011'
LVIS=DATA/'a/consensus_lvis1400_dual_frame_v1_20261011'
LVIS_BIND=ROOT/'lvis1400_binding/manifest.json'
MEAN_BIND=ROOT/'mean_continuous1020_complete_binding/manifest.json'
ATOMIC=DATA/'a/lvis_atomic1400_20261009/tasks.json'
OUT=DATA/'a/consensus_supported_foreground2380_v2_20261011'
FRAMES=('cli','original')
ARMS=('supported_mean','own_margin_sameK')
EXPECTED_COUNTS={'deepglobe_road':100,'coco':200,'paco_part':600,'lvis':1400,'pascal_part':40,'suim':40}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=Path(str(p)+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    t.replace(p)
def unpack(a,shape):return np.unpackbits(a,count=shape[0]*shape[1]).reshape(shape).astype(bool)
def fp32_bilinear(a,shape):
    x=torch.from_numpy(np.asarray(a,dtype=np.float32))[None,None]
    return F.interpolate(x,size=shape,mode='bilinear',align_corners=False)[0,0].numpy()
def inherited_bilinear(a,shape):
    x=torch.from_numpy(np.asarray(a,dtype=np.float64))[None,None]
    return F.interpolate(x,size=shape,mode='bilinear',align_corners=False)[0,0].numpy()
def own_margin_choice(eligible,mean64,shape,K):
    """Stable descending own-MEAN margin; flat input index supplies row-major ties."""
    assert eligible.shape==shape and 0<=K<=int(eligible.sum())
    result=np.zeros(shape,bool)
    if K==0:return result,dict(eligible=int(eligible.sum()),selected=0,cut_tie_pixels=0,cut_tie_selected=0,cut_score=None)
    margin=np.abs(np.float32(2)*fp32_bilinear(mean64,shape)-np.float32(1)).astype(np.float32)
    ids=np.flatnonzero(eligible.reshape(-1))
    order=np.argsort(-margin.reshape(-1)[ids],kind='stable')
    chosen=ids[order[:K]];result.reshape(-1)[chosen]=True
    cut=float(margin.reshape(-1)[chosen[-1]])
    ties=int(np.count_nonzero(margin.reshape(-1)[ids]==cut))
    used=int(np.count_nonzero(margin.reshape(-1)[chosen]==cut))
    return result,dict(eligible=len(ids),selected=K,cut_tie_pixels=ties,cut_tie_selected=used,cut_score=cut)

def expected_classes():
    a=read(DUAL/'expected_classes.json');b=read(LVIS/'expected_classes.json')
    assert set(b)=={'lvis'} and set(a)==set(EXPECTED_COUNTS)
    a['lvis']=b['lvis']
    return a

def prepare():
    protocol=read(PROTOCOL)
    assert protocol['state']=='FROZEN_COMPLETE_COMPOSITION_READY_FOR_DESIGNATED_SOL'
    assert sha(PROTOCOL)=='5fd1f4be5abc6473af2b59c60d75a521fb10211269820c093e2949893166f2e2'
    for obj in protocol['parents'].values():assert sha(obj['path'])==obj['sha256']
    dual,lv,means,tasks=[read(p) for p in (DUAL/'manifest.json',LVIS_BIND,MEAN_BIND,ATOMIC)]
    dual_seal,lvis_seal=read(DUAL/'sealed.json'),read(LVIS/'sealed.json')
    assert len(dual)==len(means)==1020 and len(lv)==len(tasks)==1400
    assert len(dual_seal['records'])==1020 and len(lvis_seal['records'])==1400
    assert all(s['row']['episode_id']==m['episode_id'] for s,m in zip(dual,means))
    assert all(s['row']['episode_id']==t['row']['episode_id'] for s,t in zip(lv,tasks))
    assert [x['row']['dataset'] for x in dual[:900]]==(['deepglobe_road']*100+['coco']*200+['paco_part']*600)
    assert [x['row']['dataset'] for x in dual[900:]]==(['lvis']*40+['pascal_part']*40+['suim']*40)
    old_lvis={x['row']['episode_id'] for x in dual[900:940]}
    assert len(old_lvis)==40 and old_lvis.issubset({x['row']['episode_id'] for x in lv})
    ordered=[]
    for i in list(range(900))+list(range(940,1020)):
        s,m,rec=dual[i],means[i],dual_seal['records'][i]
        assert s['row']['episode_id']==rec['episode_id']
        ordered.append(dict(row=s['row'],kind='dual1020',parent_index=i,parent_prediction=str(DUAL/'predictions'/rec['file']),
            parent_prediction_sha256=rec['sha256'],baseline=s['baseline'],baseline_sha256=s['baseline_sha256'],
            baseline_keys=s['baseline_keys'],mean=s['mean'],mean_sha256=s['mean_sha256'],mean_keys=s['mean_keys'],
            mean64_path=m['field_path'],mean64_sha256=m['field_sha256'],
            evidence_path=s['field'],evidence_sha256=s['field_sha256'],evidence_key=s['field_key'],calibration=s['calibration'],
            parent_frame_shapes={f:rec['frames'][f]['shape'] for f in FRAMES}))
    # Insert LVIS1400 between PACO600 and the remaining PASCAL/SUIM80.
    lvis_rows=[]
    for i,(s,t,rec) in enumerate(zip(lv,tasks,lvis_seal['records'])):
        assert s['row']['episode_id']==rec['episode_id']
        mean64=Path(t['source'])/'fields'/t['filename']
        lvis_rows.append(dict(row=s['row'],kind='lvis1400',parent_index=i,
            parent_prediction=str(LVIS/'predictions'/rec['file']),parent_prediction_sha256=rec['prediction_sha256'],
            baseline=s['baseline'],baseline_sha256=s['baseline_sha256'],baseline_keys=s['baseline_keys'],
            mean=s['mean'],mean_sha256=s['mean_sha256'],mean_keys=s['mean_keys'],
            mean64_path=str(mean64),mean64_sha256=t['source_field_sha256'],
            evidence_path=rec['field_path'],evidence_sha256=rec['field_sha256'],evidence_key='evidence_global',
            calibration=rec['calibration'],origin=rec['origin'],
            parent_frame_shapes={f:rec['frames'][f]['shape'] for f in FRAMES}))
    ordered=ordered[:900]+lvis_rows+ordered[900:]
    assert len(ordered)==2380 and len({s['row']['episode_id'] for s in ordered})==2380
    from collections import Counter
    assert dict(Counter(s['row']['dataset'] for s in ordered))==EXPECTED_COUNTS
    if not OUT.exists():OUT.mkdir()
    for sub in ('predictions','records','frozen'):(OUT/sub).mkdir(exist_ok=True)
    binding=OUT/'manifest.json';expected=OUT/'expected_classes.json';cfgpath=OUT/'config.json'
    if binding.exists():assert read(binding)==ordered
    else:write(binding,ordered)
    ex=expected_classes()
    if expected.exists():assert read(expected)==ex
    else:write(expected,ex)
    cfg=dict(state='FROZEN_SUPPORTED_FOREGROUND2380',n=2380,frames=list(FRAMES),arms=list(ARMS),
        protocol_sha256=sha(PROTOCOL),manifest_sha256=sha(binding),expected_classes_sha256=sha(expected),
        parent_dual_seal_sha256=sha(DUAL/'sealed.json'),parent_lvis_seal_sha256=sha(LVIS/'sealed.json'),
        mean_binding_sha256=sha(MEAN_BIND),lvis_binding_sha256=sha(LVIS_BIND),lvis_atomic_tasks_sha256=sha(ATOMIC),
        source_sha256=sha(__file__),new_encoder_calls=0,new_raw_reads=0,new_fits=0,new_graphs=0,new_CRF_calls=0,
        query_GT_in_prediction=False,producer_lock='nonblocking flock',
        render='MEAN mean64 FP32 direct bilinear to CLI >.5; native G from FP32 bilinear of binary CLI >.5',
        control='U sorted by descending abs(2*FP32_bilinear(mean64,frame)-1), stable ties row-major',
        no_old_mutable_core_restriction=True,exposure='Existing exposed DEV cohorts, LVIS40 overlap removed')
    if cfgpath.exists():assert read(cfgpath)==cfg
    else:write(cfgpath,cfg)
    frozen=OUT/'frozen'/Path(__file__).name
    if frozen.exists():assert sha(frozen)==cfg['source_sha256']
    else:frozen.write_bytes(Path(__file__).read_bytes())
    return cfg

def run():
    cfg=prepare()
    if (OUT/'sealed.json').exists():return
    import sol_bg_admission_composition as host
    manifest=read(OUT/'manifest.json');records=[];checks=0;attempts=[];start=time.monotonic()
    lock=(OUT/'.producer.lock').open('a+')
    try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another producer owns the supported-foreground output lock')
    try:
        with host.deny_query_masks(manifest,attempts):
            for i,s in enumerate(manifest):
                eid=s['row']['episode_id'];rp=OUT/'records'/f'{i:06}.json';tick=time.monotonic()
                if rp.exists():
                    rec=read(rp);assert rec['episode_id']==eid
                    assert sha(OUT/'predictions'/rec['file'])==rec['sha256']
                    records.append(rec);continue
                for key in ('parent_prediction','baseline','mean','mean64_path','evidence_path'):
                    assert sha(s[key])==s[{'mean64_path':'mean64_sha256','evidence_path':'evidence_sha256'}.get(key,key+'_sha256')],(eid,key)
                    checks+=1
                with np.load(s['mean64_path'],allow_pickle=False) as z:mean64=z['mean'].copy()
                assert mean64.shape==(64,64) and mean64.dtype==np.float32 and np.isfinite(mean64).all()
                with np.load(s['evidence_path'],allow_pickle=False) as z:field=z[s['evidence_key']].copy()
                assert field.shape==(128,128) and np.isfinite(field).all()
                if s['kind']=='dual1020':
                    cal=s['calibration'];scale=float(cal['separation'])
                    h=np.clip((field-float(cal['threshold']))/scale,-1,1) if np.isfinite(scale) and scale>np.finfo(float).eps else np.zeros_like(field)
                else:
                    h=field
                    if s.get('origin')=='SOURCE_ROLE_DEGENERATE_ZERO_EVIDENCE':assert np.array_equal(h,np.zeros_like(h))
                with np.load(s['parent_prediction'],allow_pickle=False) as z:parent={k:z[k].copy() for f in FRAMES for k in (f+'/anchor.global',f+'/shared_edits_complete')}
                with np.load(s['baseline'],allow_pickle=False) as z:base={f:z[s['baseline_keys'][f]].copy() for f in FRAMES}
                with np.load(s['mean'],allow_pickle=False) as z:archived_mean={f:z[s['mean_keys'][f]].copy() for f in FRAMES}
                cli_mean=fp32_bilinear(mean64,(1024,1024))>.5
                pred={};frames={}
                for frame in FRAMES:
                    shape=(1024,1024) if frame=='cli' else tuple(s['row']['query_size_hw'])
                    assert list(shape)==s['parent_frame_shapes'][frame]
                    B=unpack(base[frame],shape);R=unpack(parent[frame+'/anchor.global'],shape)
                    C=unpack(parent[frame+'/shared_edits_complete'],shape)
                    G=unpack(archived_mean[frame],shape)
                    replay_G=cli_mean if frame=='cli' else fp32_bilinear(cli_mean.astype(np.float32),shape)>.5
                    assert np.array_equal(G,replay_G),(eid,frame,'mean renderer')
                    assert np.array_equal(C,(B&R)|(B&G)|(R&G)),(eid,frame,'parent consensus')
                    H=inherited_bilinear(h,shape)
                    U=(~B)&(~R)&G
                    A=U&(H>0)
                    N=C|A
                    K=int(A.sum())
                    control_add,ties=own_margin_choice(U,mean64,shape,K)
                    control=C|control_add
                    assert np.array_equal(N&~C,A)
                    assert not np.any(A&(~U)) and not np.any((N^C)&B)
                    assert np.array_equal(N&C,C) and np.array_equal(control&C,C)
                    assert int(control_add.sum())==K and not np.any(control_add&~U)
                    assert np.array_equal(control&~C,control_add)
                    if not K or not np.any(h):assert np.array_equal(N,C)
                    if s.get('origin')=='SOURCE_ROLE_DEGENERATE_ZERO_EVIDENCE':
                        assert np.array_equal(R,B) and np.array_equal(C,B) and K==0 and np.array_equal(control,B)
                    pred[frame+'/supported_mean']=np.packbits(N)
                    pred[frame+'/own_margin_sameK']=np.packbits(control)
                    frames[frame]=dict(shape=list(shape),K=K,U=int(U.sum()),
                        changed_N_from_C=int(np.count_nonzero(N!=C)),changed_control_from_C=int(np.count_nonzero(control!=C)),
                        C_deletions_from_B=int(np.count_nonzero(B&~C)),
                        N_deletions_from_B=int(np.count_nonzero(B&~N)),
                        control_deletions_from_B=int(np.count_nonzero(B&~control)),ties=ties)
                    assert frames[frame]['C_deletions_from_B']==frames[frame]['N_deletions_from_B']==frames[frame]['control_deletions_from_B']
                    checks+=12
                name=f'{i:06}.npz';pp=OUT/'predictions'/name
                np.savez_compressed(pp,**pred)
                rec=dict(index=i,episode_id=eid,dataset=s['row']['dataset'],file=name,sha256=sha(pp),frames=frames,
                         parent_prediction_sha256=s['parent_prediction_sha256'],mean64_sha256=s['mean64_sha256'],
                         seconds=time.monotonic()-tick,new_encoder_calls=0,new_raw_reads=0,new_fits=0)
                write(rp,rec);records.append(rec)
                if (i+1)%50==0 or i+1==len(manifest):
                    write(OUT/'activity.json',dict(state='PRODUCING',pid=os.getpid(),completed=i+1,total=len(manifest),seconds=time.monotonic()-start))
                    print(json.dumps(dict(completed=i+1,total=len(manifest),seconds=round(time.monotonic()-start,2))),flush=True)
        assert len(records)==2380 and not attempts
        write(OUT/'sealed.json',dict(state='ALL9520_SUPPORTED_FOREGROUND_MASKS_SEALED',n=2380,frames=list(FRAMES),arms=list(ARMS),
            records=records,config_sha256=sha(OUT/'config.json'),manifest_sha256=sha(OUT/'manifest.json'),
            expected_classes_sha256=sha(OUT/'expected_classes.json'),checks=checks,
            query_gt_access_attempts=attempts,new_complete_masks=9520,
            new_encoder_calls=0,new_raw_reads=0,new_fits=0,new_CRF_calls=0,
            seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write(OUT/'activity.json',dict(state='SEALED',completed=2380,total=2380))
    finally:
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','run'));a=p.parse_args()
    {'prepare':prepare,'run':run}[a.phase]()
