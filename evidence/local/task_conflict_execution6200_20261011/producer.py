"""Sole complete6200 producer: frozen dense cue, bound cache IO, per-frame host.

No model construction, cache writes, query labels, old score replay or tuning.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='4'
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from dense_core import compute_dense

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
PRIOR=REPO/'evidence/local/nine_window_execution_20261011'
sys.path.insert(0,str(PRIOR))
from truth_guard import QueryTruthGuard
ARMS=('anchor_relative_complete','reference_relative_complete','same_edit_budget_foris_margin')
PROTOCOL_SHA='4fd6b2cb3a9ff3ef7b56eeb22833f16c2f6d051f296df9db81d83d5134a7e7cd'


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.pending')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):h.update(part)
    return h.hexdigest()
def tensor_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(json.dumps([list(a.shape),a.dtype.str]).encode());h.update(a.tobytes());return h.hexdigest()
def resize(a,shape,mode='bilinear'):
    a=torch.as_tensor(np.ascontiguousarray(a),dtype=torch.float64,device='cpu')[None,None]
    return F.interpolate(a,tuple(shape),mode=mode,**({} if mode=='nearest' else {'align_corners':False}))[0,0].numpy()
def unpack(z,key,shape):
    raw=z[key];n=int(np.prod(shape));assert raw.dtype==np.uint8 and raw.ndim==1 and raw.size==(n+7)//8
    bits=np.unpackbits(raw);assert not bits[n:].any();return bits[:n].reshape(shape).astype(bool)


class RawReader:
    def __init__(self):
        self.meta=read(HERE/'raw_metadata.json');self.cost=Counter();self.used=set()
    def load(self,request):
        tick,cpu=time.monotonic(),time.process_time();bound=self.meta[request['entry_path']]
        assert bound['available'];entry=Path(request['entry_path']);payload=Path(request['payload_path'])
        if not entry.is_file() or not payload.is_file():raise FileNotFoundError('Missing bound native cache; encoder fallback forbidden: '+request['key'])
        assert sha(entry)==request['entry_sha256'];info=read(entry)
        assert info['input_tensor_hash']==request['input_sha256']
        assert info['file_sha256']==request['payload_sha256'] and payload.name==info['file']
        assert sha(payload)==request['payload_sha256']
        with np.load(payload,allow_pickle=False) as z:raw=z['O/24'].copy()
        assert raw.shape==(4096,1024) and raw.dtype==np.float32
        assert np.isfinite(raw).all() and tensor_sha(raw)==request['tensor_sha256']==info['features']['O/24']['tensor_sha256']
        self.cost['raw_loads']+=1;self.cost['payload_container_bytes_requested']+=payload.stat().st_size
        self.cost['IO_wall_seconds']+=time.monotonic()-tick;self.cost['IO_CPU_seconds']+=time.process_time()-cpu
        self.used.add(request['key']);return raw


def masks_and_score(s):
    groups={};out={f:{} for f in ('cli','original')};hw=s['row']['query_size_hw']
    for name,binding in s['masks'].items():groups.setdefault(binding['path'],[]).append((name,binding))
    for path,requests in groups.items():
        assert sha(path)==requests[0][1]['sha256']
        with np.load(path,allow_pickle=False) as z:
            for name,binding in requests:
                for frame in out:
                    out[frame][name]=unpack(z,binding[frame+'_key'],(1024,1024) if frame=='cli' else hw)
    source=s['source_score64'];assert sha(source['path'])==source['sha256']
    with np.load(source['path'],allow_pickle=False) as z:score=z[source['key']].copy().reshape(64,64).astype(np.float64)
    assert np.isfinite(score).all();return out,score


def reference_coverage(s):
    row=s['row']
    with Image.open(row['reference_mask_path']) as im:a=(np.asarray(im.convert('L'))>0).astype(np.uint8)
    assert list(a.shape)==row['reference_mask_size_hw'] and tensor_sha(a)==row['reference_mask_hash']
    canonical=resize(a,(1024,1024),'nearest')
    return canonical.reshape(64,16,64,16).mean((1,3)).ravel()


def select(eligible,margin,K,descending):
    ids=np.flatnonzero(eligible.ravel());assert 0<=K<=len(ids);chosen=np.zeros(eligible.size,bool)
    if not K:return chosen.reshape(eligible.shape),dict(K=0,eligible=len(ids),cutoff=None,ties=0,selected_ties=0)
    values=margin.ravel()[ids];order=np.lexsort((ids,-values if descending else values));selected=ids[order[:K]];chosen[selected]=True
    cut=float(values[order[K-1]])
    return chosen.reshape(eligible.shape),dict(K=K,eligible=len(ids),cutoff=cut,ties=int((values==cut).sum()),selected_ties=int((values[order[:K]]==cut).sum()),last_selected_index=int(selected[-1]))


def frame_outputs(masks,score,h_anchor,h_reference):
    B,G,R=(masks[k] for k in ('B','G','R0'));shape=B.shape;span=float(np.ptp(score))
    normalized=(score-score.min())/max(span,1e-6);dB=2*resize(normalized,shape)-1
    conf=np.abs(dB) if span else np.zeros(shape);mB=(2*B.astype(float)-1)*conf
    Dadd=(~B)&(G|R);Ddel=B&((~G)|(~R));agreed=(B==G)&(G==R)
    output={};cert={};rendered={}
    for name,h in zip(ARMS[:2],(h_anchor,h_reference)):
        field=np.zeros(shape) if h is None else np.clip(resize(h,shape),-1,1)
        z=mB+.5*field;add=Dadd&(z>0);delete=Ddel&(z<0);prediction=(B&~delete)|add
        # Independent algebraic formulation, no query labels.
        expected=B.copy();expected[(~B)&Dadd&(field>2*conf)]=True;expected[B&Ddel&(field < -2*conf)]=False
        assert np.array_equal(prediction,expected)
        assert np.array_equal(prediction[agreed],B[agreed])
        assert not np.any((prediction!=B)&(conf>=.5))
        if h is None:assert np.array_equal(prediction,B)
        output[name]=prediction;rendered[name]=field;cert[name]=dict(add=int(add.sum()),delete=int(delete.sum()))
    main=output[ARMS[0]];Kadd=int((main&~B).sum());Kdel=int((~main&B).sum())
    EA=Dadd&(conf<.5);ED=Ddel&(conf<.5)
    add,ca=select(EA,dB,Kadd,True);delete,cd=select(ED,dB,Kdel,False)
    budget=(B|add)&~delete
    # Stable-sort independent tie check uses row-major eligible input order.
    check=B.copy()
    for eligible,K,desc,value in ((EA,Kadd,True,True),(ED,Kdel,False,False)):
        ids=np.flatnonzero(eligible.ravel());rank=np.argsort(-dB.ravel()[ids] if desc else dB.ravel()[ids],kind='stable')
        check.ravel()[ids[rank[:K]]]=value
    assert np.array_equal(check,budget) and np.array_equal(budget[agreed],B[agreed])
    assert int((budget&~B).sum())==Kadd and int((~budget&B).sum())==Kdel
    output[ARMS[2]]=budget;cert[ARMS[2]]=dict(add=Kadd,delete=Kdel,add_selection=ca,delete_selection=cd)
    return output,dict(edit_counts=cert,D_add=int(Dadd.sum()),D_delete=int(Ddel.sum()),agreed_pixels=int(agreed.sum()),
                       formula_reconstructions=3,old_R0_equals_B=bool(np.array_equal(R,B)))


def diagnostic_difference(actual,reference):
    result={}
    for k in ('anchor_h','reference_h'):
        a,b=actual[k],reference[k]
        result[k]=dict(actual_available=a is not None,reference_available=b is not None)
        if a is not None and b is not None:
            result[k].update(max_abs=float(np.max(np.abs(a-b))),rms=float(np.sqrt(np.mean((a-b)**2))),native_sign_difference=int(np.count_nonzero((a>0)!=(b>0))))
    for k in ('source_scale_squared','S_A','anchor_centre_delta_norm'):
        a,b=actual['stats'].get(k),reference['stats'].get(k)
        if isinstance(a,(float,int)) and isinstance(b,(float,int)) and np.isfinite(a) and np.isfinite(b):result[k+'_absolute_difference']=abs(a-b)
    result['fallback_reason_agreement']={k:actual['stats'].get(k)==reference['stats'].get(k)
                                       for k in ('reference_reason','anchor_reason')}
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);args=parser.parse_args()
    assert sha(args.protocol)==PROTOCOL_SHA
    protocol=read(args.protocol);assert protocol['state']=='FROZEN_COMPLETE_MIXED_PRECISION_COMPOSITION_BEFORE_ANY_NEW_CUE_OR_MASK'
    assert tuple(protocol['new_arms'])==ARMS
    preflight=read(HERE/'preflight_receipt.json');assert not preflight['missing']
    for name,key in (('inputs.jsonl','inputs_sha256'),('score_bindings.jsonl','score_bindings_sha256'),('raw_metadata.json','raw_metadata_sha256')):assert sha(HERE/name)==preflight[key]
    rows=lines(HERE/'inputs.jsonl');scoring=lines(HERE/'score_bindings.jsonl');assert len(rows)==len(scoring)==6200
    guard=QueryTruthGuard([s['row'] for s in scoring]);reader=RawReader();torch.set_num_threads(4)
    with (HERE/'.producer.lock').open('a+') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/'sealed.json').exists():raise RuntimeError('Already sealed: never repeat producer')
        for folder in ('predictions','records'):(HERE/folder).mkdir(exist_ok=True)
        config=dict(state='FROZEN_UNIFIED6200_SOLE_PRODUCER',protocol_path=str(args.protocol.resolve()),protocol_sha256=PROTOCOL_SHA,
            input_receipt_sha256=sha(HERE/'preflight_receipt.json'),inputs_sha256=preflight['inputs_sha256'],score_bindings_sha256=preflight['score_bindings_sha256'],
            source_sha256={str(p):sha(p) for p in (Path(__file__),HERE/'dense_core.py',PRIOR/'truth_guard.py')},n=6200,arms=list(ARMS),CPU_threads=4,workers=1,query_chunk_rows=1024)
        if (HERE/'config.json').exists():assert read(HERE/'config.json')==config
        else:write(HERE/'config.json',config)
        (HERE/'protocol.frozen.json').write_bytes(args.protocol.read_bytes())
        started,cpu=time.monotonic(),time.process_time();totals=Counter();records=[];fallbacks=Counter()
        write(HERE/'activity.json',dict(state='RUNNING_FULL6200',pid=os.getpid(),completed=0,total=6200,protocol_sha256=PROTOCOL_SHA))
        print(json.dumps(dict(stage='started',pid=os.getpid(),n=6200,protocol_sha256=PROTOCOL_SHA)),flush=True)
        with guard:
            for i,s in enumerate(rows):
                record_path=HERE/'records'/f'{i:06}.json';dest=HERE/'predictions'/f'{i:06}.npz'
                if record_path.exists():
                    rec=read(record_path);assert rec['episode_id']==s['row']['episode_id'] and sha(dest)==rec['prediction_sha256'];records.append(rec);continue
                if dest.exists():raise RuntimeError('Orphan final mask: preserve and inspect before resume '+str(dest))
                begin=time.monotonic();phase={};tick=time.monotonic()
                masks,score=masks_and_score(s);p=reference_coverage(s)
                a=(masks['cli']['B']&masks['cli']['G']&masks['cli']['R0']).reshape(64,16,64,16).mean((1,3)).ravel()
                phase['source_masks_margin_reference_seconds']=time.monotonic()-tick
                tick=time.monotonic();rr=reader.load(s['raw']['reference_full']);rq=rr if s['raw']['reference_full']['key']==s['raw']['query_full']['key'] else reader.load(s['raw']['query_full'])
                phase['raw_read_seconds']=time.monotonic()-tick;tick=time.monotonic();runtime='mps_float32';failure=None;failed_seconds=0.
                try:
                    dense=compute_dense(rr,rq,p,a,device='mps',dtype=torch.float32,chunk_rows=1024);torch.mps.synchronize()
                except RuntimeError as exc:
                    failed_seconds=time.monotonic()-tick;failure=str(exc)
                    try:torch.mps.synchronize()
                    except RuntimeError:pass
                    torch.mps.empty_cache();tick=time.monotonic();runtime='cpu_float32_backend_fallback'
                    dense=compute_dense(rr,rq,p,a,device='cpu',dtype=torch.float32,chunk_rows=1024)
                phase['dense_cues_seconds']=time.monotonic()-tick;phase['failed_MPS_seconds']=failed_seconds
                reference_check=None
                if i<3:
                    tick=time.monotonic();reference=compute_dense(rr,rq,p,a,device='cpu',dtype=torch.float64,chunk_rows=1024)
                    reference_check=diagnostic_difference(dense,reference);phase['first3_CPU64_reference_seconds']=time.monotonic()-tick
                    del reference
                del rr,rq;p_sha=tensor_sha(p);a_sha=tensor_sha(a);del p,a
                tick=time.monotonic();payload={'original_hw':np.asarray(s['row']['query_size_hw'])};frames={}
                for frame in ('cli','original'):
                    outputs,certificate=frame_outputs(masks[frame],score,dense['anchor_h'],dense['reference_h']);frames[frame]=certificate
                    for arm,mask in outputs.items():payload[frame+'/'+arm]=np.packbits(mask)
                    del outputs
                phase['frame_compose_verify_budget_seconds']=time.monotonic()-tick;tick=time.monotonic()
                temp=dest.with_suffix('.pending.npz');np.savez_compressed(temp,**payload);temp.replace(dest)
                rec=dict(index=i,binding_index=s['binding_index'],episode_id=s['row']['episode_id'],dataset=s['row']['dataset'],filename=dest.name,
                    prediction_sha256=sha(dest),dense_stats=dense['stats'],dense_runtime=runtime,MPS_failure=failure,
                    reference_coverage_sha256=p_sha,anchor_coverage_sha256=a_sha,frames=frames,first3_CPU64_numerical_diagnostic=reference_check,
                    source_bindings=s['masks'],source_score64=s['source_score64'],raw_keys=[s['raw'][r]['key'] for r in ('reference_full','query_full')],
                    source_APD_diagnostic=s['source_apd_applied'],costs=phase,query_GT_reads=0,new_encoder_calls=0,new_CRF_calls=0)
                write(record_path,rec);phase['mask_record_write_seconds']=time.monotonic()-tick;rec['total_seconds']=time.monotonic()-begin
                write(record_path,rec);records.append(rec);totals.update(phase)
                fallbacks[runtime]+=1
                del masks,score,dense,payload,frames
                write(HERE/'activity.json',dict(state='RUNNING_FULL6200',pid=os.getpid(),completed=i+1,total=6200,last_episode_id=s['row']['episode_id'],runtime=runtime,protocol_sha256=PROTOCOL_SHA))
                if (i+1)%10==0 or i<3:print(json.dumps(dict(stage='produce',completed=i+1,total=6200,seconds=time.monotonic()-started,last_seconds=rec['total_seconds'],runtime=runtime)),flush=True)
            assert len(records)==6200 and not guard.attempts
            seal=dict(state='ALL6200_THREE_COMPLETE_ARMS_TWO_FRAMES_SEALED_BEFORE_GT',n=6200,mask_count=37200,
                protocol_sha256=PROTOCOL_SHA,config_sha256=sha(HERE/'config.json'),inputs_sha256=preflight['inputs_sha256'],
                records=[{k:r[k] for k in ('index','binding_index','episode_id','filename','prediction_sha256')} for r in records],
                query_GT_guard=guard.receipt(),new_encoder_calls=0,new_CRF_calls=0,new_dense_archives=0)
            write(HERE/'sealed.json',seal)
        receipt=dict(state=seal['state'],pid=os.getpid(),n=6200,mask_count=37200,CPU_threads=4,workers=1,
            wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            phase_seconds=totals,raw_reader=dict(reader.cost,unique_keys=len(reader.used)),runtime_counts=fallbacks,
            guard=guard.receipt(),seal_sha256=sha(HERE/'sealed.json'),new_encoder_calls=0,new_CRF_calls=0,new_dense_archives=0)
        write(HERE/'production_receipt.json',receipt);write(HERE/'activity.json',dict(state=seal['state'],pid=os.getpid(),completed=6200,total=6200,seal_sha256=receipt['seal_sha256']))
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
