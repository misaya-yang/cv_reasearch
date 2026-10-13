"""One streaming producer; no query GT, encoder or old heads are invoked."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='4'
import argparse
from collections import Counter
import fcntl
import importlib.util
from pathlib import Path
import resource
import sys
import time

import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
OLD=BASE/'evidence/local/task_conflict_execution6200_20261011'
ALIGN=BASE/'evidence/local/strong600_conflict_alignment_20261011'
STAGE=BASE/'evidence/local/strong_stage600_20261011'
sys.path.insert(0,str(BASE/'src'))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


host=module('constellation_cached_host',BASE/'evidence/local/reference_background_conflict6200_20261011/producer.py')
core=module('constellation_fixed_core',HERE/'core.py')
legacy=host.legacy
read,write,sha,lines=host.read,host.write,host.sha,host.lines
WINDOWS=[[x,y,x+512,y+512] for y in (0,256,512) for x in (0,256,512)]


def accepted_support(entry,receipt):
    from scipy.ndimage import label
    result=np.zeros((1024,1024),bool)
    assert sha(entry['assets']['windows']['path'])==entry['assets']['windows']['sha256']
    with np.load(entry['assets']['windows']['path'],allow_pickle=False) as z:
        for i,(x0,y0,x1,y1) in enumerate(WINDOWS):
            local=legacy.unpack(z,'window.native/'+str(i),(512,512))
            ids,n=label(local,np.ones((3,3),bool))
            rec=receipt['regions'][i]
            assert rec['components']==n and rec['component_sizes']==np.bincount(ids.ravel(),minlength=n+1)[1:].tolist()
            take=rec['selections']['component.query_context']
            result[y0:y1,x0:x1]|=np.asarray([False]+take,bool)[ids]
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True,type=Path);parser.add_argument('--protocol-sha',required=True);args=parser.parse_args()
    assert sha(args.protocol)==args.protocol_sha
    protocol=read(args.protocol)
    assert tuple(protocol['arms'])==core.ARMS
    for p,digest in protocol['source_sha256'].items():assert sha(p)==digest,p
    members=read(ALIGN/'membership.json');inventory=read(STAGE/'inventory.json')['records']
    sources,scoring=lines(OLD/'inputs.jsonl'),lines(OLD/'score_bindings.jsonl')
    receipts={c:{r['episode_id']:r for r in lines(Path(c)/'inference.jsonl')} for c in {e['cohort'] for e in inventory}}
    guard=host.QueryTruthGuard([scoring[m['source_index']]['row'] for m in members]);reader=legacy.RawReader()
    assert len(members)==len(inventory)==600
    torch.set_num_threads(4)
    with (HERE/'.producer.lock').open('a+') as lock,threadpool_limits(limits=4):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (HERE/'config.json').exists() and not (HERE/'sealed.json').exists(),'Preserve attempts; do not restart'
        for name in ('predictions','records'):
            (HERE/name).mkdir(exist_ok=True);assert not any((HERE/name).iterdir())
        config=dict(state='FIXED600_TASK_CONSTELLATION_COMPLETE',n=600,protocol_path=str(args.protocol.resolve()),protocol_sha256=args.protocol_sha,
                    arms=list(core.ARMS),source_sha256=protocol['source_sha256'],membership_sha256=sha(ALIGN/'membership.json'),workers=1,CPU_threads=4)
        write(HERE/'config.json',config);(HERE/'protocol.frozen.json').write_bytes(args.protocol.read_bytes())
        started,cpu=time.monotonic(),time.process_time();costs,runtimes=Counter(),Counter();records=[]
        max_allocated=max_driver=0
        with guard:
            for i,(member,entry) in enumerate(zip(members,inventory)):
                begin=time.monotonic();phase={}
                source,score=sources[member['source_index']],scoring[member['source_index']]
                assert source['row']['episode_id']==score['row']['episode_id']==member['episode_id']==entry['episode_id']
                assert entry['index']==i and source['binding_index']==member['binding_index']
                tick=time.monotonic();masks,source_score=host.load_masks(source,score)
                path=entry['assets']['region']['path'];assert sha(path)==entry['assets']['region']['sha256']
                with np.load(path,allow_pickle=False) as z:
                    for frame in ('cli','original'):
                        key=('canvas' if frame=='cli' else 'original')+'/component.query_context'
                        masks[frame]['S']=legacy.unpack(z,key,masks[frame]['B'].shape)
                any_accepted=accepted_support(entry,receipts[entry['cohort']][entry['episode_id']])
                row=source['row']
                with Image.open(row['reference_mask_path']) as im:rm=(np.asarray(im.convert('L'))>0).astype(np.uint8)
                assert list(rm.shape)==row['reference_mask_size_hw'] and legacy.tensor_sha(rm)==row['reference_mask_hash']
                canonical=core.resize(rm,(1024,1024),'nearest')>.5
                p=canonical.reshape(64,16,64,16).mean((1,3)).ravel()
                s=masks['cli']['S'].reshape(64,16,64,16).mean((1,3)).ravel()
                phase['saved_stage_masks_reference_roles_seconds']=time.monotonic()-tick
                tick=time.monotonic();rr=reader.load(source['raw']['reference_full']);rq=reader.load(source['raw']['query_full'])
                phase['old_R_Q_raw_load_seconds']=time.monotonic()-tick
                tick=time.monotonic();failure=None;runtime='MPS32_feature_cost_CPU64_assignment_pose'
                phase['failed_MPS_seconds']=0.
                try:
                    dense=core.compute(rr,rq,p,s,row['reference_size_hw'],row['query_size_hw'],device='mps',validate=i<3);torch.mps.synchronize()
                except RuntimeError as exc:
                    failure=str(exc);phase['failed_MPS_seconds']=time.monotonic()-tick
                    try:torch.mps.synchronize()
                    except RuntimeError:pass
                    torch.mps.empty_cache();tick=time.monotonic();runtime='CPU32_feature_cost_CPU64_assignment_pose_actual_fallback'
                    dense=core.compute(rr,rq,p,s,row['reference_size_hw'],row['query_size_hw'],device='cpu',validate=i<3)
                phase['contrastive_cost_four_registrations_seconds']=time.monotonic()-tick
                max_allocated=max(max_allocated,torch.mps.current_allocated_memory());max_driver=max(max_driver,torch.mps.driver_allocated_memory())
                del rr,rq,p,s
                tick=time.monotonic();payload={'original_hw':np.asarray(row['query_size_hw'])};frames={}
                for frame in ('cli','original'):
                    outputs,frames[frame]=core.compose_frame(masks[frame],source_score,dense,canonical,any_accepted)
                    for arm,value in outputs.items():payload[frame+'/'+arm]=np.packbits(value)
                    del outputs
                phase['two_frame_warp_compose_budget_seconds']=time.monotonic()-tick
                tick=time.monotonic();path=HERE/'predictions'/f'{i:06}.npz';tmp=path.with_suffix('.pending.npz')
                np.savez_compressed(tmp,**payload);tmp.replace(path)
                record=dict(index=i,source_index=member['source_index'],binding_index=member['binding_index'],episode_id=member['episode_id'],filename=path.name,
                            prediction_sha256=sha(path),registration_stats=dense['stats'],frames=frames,runtime=runtime,MPS_failure=failure,
                            original_S=entry['assets']['region'],window_mask_source=entry['assets']['windows'],accepted_union_sha256=core.hash_array(any_accepted),
                            canonical_reference_mask_sha256=core.hash_array(canonical),query_GT_reads=0,new_encoder_calls=0,new_CRF_calls=0)
                phase['packed_output_write_seconds']=time.monotonic()-tick
                record.update(costs=phase,total_seconds=time.monotonic()-begin);write(HERE/'records'/f'{i:06}.json',record)
                costs.update(phase);runtimes[runtime]+=1
                records.append({k:record[k] for k in ('index','source_index','binding_index','episode_id','filename','prediction_sha256')})
                del masks,source_score,dense,payload,frames,canonical,any_accepted
                write(HERE/'activity.json',dict(state='RUNNING_FIXED600_TASK_CONSTELLATION',pid=os.getpid(),completed=i+1,total=600))
                if i<3 or (i+1)%50==0:print(dict(completed=i+1,total=600,seconds=time.monotonic()-started),flush=True)
            assert len(records)==600 and not guard.attempts
            seal=dict(state='ALL600_SIX_COMPLETE_ARMS_TWO_FRAMES_SEALED_BEFORE_GT',n=600,mask_count=7200,protocol_sha256=args.protocol_sha,
                      config_sha256=sha(HERE/'config.json'),records=records,query_GT_guard=guard.receipt(),new_encoder_calls=0,new_CRF_calls=0,new_dense_archives=0)
            write(HERE/'sealed.json',seal)
        production=dict(state=seal['state'],n=600,mask_count=7200,wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
                        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,phase_seconds=dict(costs),raw_reader=dict(reader.cost,unique_keys=len(reader.used)),
                        runtime_counts=dict(runtimes),guard=guard.receipt(),seal_sha256=sha(HERE/'sealed.json'),new_encoder_calls=0,new_CRF_calls=0,new_dense_archives=0,
                        query_RGB_reads=0,reference_RGB_reads=0,new_encoded_views=0,reference_raw_requests=600,query_raw_requests=600,
                        MPS_sampled_allocated_max_bytes=max_allocated,MPS_sampled_driver_max_bytes=max_driver,MPS_memory_sampling='return_time_after_compute_not_true_peak_or_utilization',
                        raw_IO_measurement='requested_container_bytes_not_physical_IO',resumed=False)
        write(HERE/'production_receipt.json',production);write(HERE/'activity.json',dict(state=seal['state'],pid=None,completed=600,total=600))
        print(production,flush=True)


if __name__=='__main__':main()
