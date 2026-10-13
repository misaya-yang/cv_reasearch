"""Reference/query task discriminant around exact strong600; no encoder update."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
import argparse
from collections import Counter
import fcntl
import importlib.util
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
OLD = BASE/'evidence/local/task_conflict_execution6200_20261011'
BG = BASE/'evidence/local/reference_background_conflict6200_20261011'
ALIGN = BASE/'evidence/local/strong600_conflict_alignment_20261011'
sys.path.insert(0, str(BASE/'src'))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


host = module('completed_background_producer', BG/'producer.py')
core = module('fixed600_task_discriminant_core', HERE/'core.py')
read, lines, write, sha = host.read, host.lines, host.write, host.sha
legacy = host.legacy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    assert sha(args.protocol) == args.protocol_sha
    protocol = read(args.protocol)
    assert tuple(protocol['arms']) == tuple(core.ARMS)
    for path, digest in protocol['source_sha256'].items():assert sha(path) == digest, path
    membership = read(ALIGN/'membership.json')
    assert len(membership) == 600 and [r['subset_index'] for r in membership] == list(range(600))
    sources, scoring = lines(OLD/'inputs.jsonl'), lines(OLD/'score_bindings.jsonl')
    scores = [scoring[r['source_index']] for r in membership]
    guard, reader = host.QueryTruthGuard([r['row'] for r in scores]), legacy.RawReader()
    torch.set_num_threads(4)
    with (HERE/'.producer.lock').open('a+') as lock, threadpool_limits(limits=4):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (HERE/'sealed.json').exists(), 'Never repeat completed production'
        assert not (HERE/'config.json').exists(), 'Preserve any attempt before restarting'
        for name in ('predictions','records'):
            (HERE/name).mkdir(exist_ok=True)
            assert not any((HERE/name).iterdir())
        config = dict(state='FIXED_EXISTING600_REFERENCE_QUERY_TASK_DISCRIMINANT_FULL_COMPOSITION',n=600,
                      protocol_sha256=args.protocol_sha,protocol_path=str(args.protocol.resolve()),
                      arms=list(core.ARMS),source_sha256=protocol['source_sha256'],
                      membership_sha256=sha(ALIGN/'membership.json'),workers=1,CPU_threads=4)
        write(HERE/'config.json',config)
        (HERE/'protocol.frozen.json').write_bytes(args.protocol.read_bytes())
        started, cpu = time.monotonic(),time.process_time()
        costs, runtimes, records = Counter(),Counter(),[]
        max_mps_allocated, max_mps_driver = 0, 0
        write(HERE/'activity.json',dict(state='RUNNING_FIXED600',pid=os.getpid(),completed=0,total=600))
        with guard:
            for i, binding in enumerate(membership):
                source, score_binding = sources[binding['source_index']], scores[i]
                assert source['row']['episode_id'] == score_binding['row']['episode_id'] == binding['episode_id']
                assert source['binding_index'] == score_binding['binding_index'] == binding['binding_index']
                one, begin = {},time.monotonic()
                masks, source_score = host.load_masks(source,score_binding)
                sb = binding['old_region']
                assert sha(sb['old_region_path']) == sb['old_region_current_sha256']
                with np.load(sb['old_region_path'],allow_pickle=False) as z:
                    for frame in ('cli','original'):
                        key=('canvas' if frame=='cli' else 'original')+'/component.query_context'
                        masks[frame]['S']=legacy.unpack(z,key,masks[frame]['B'].shape)
                s,b,g,r=(masks['cli'][k] for k in ('S','B','G','R0'))
                union=b|g|r
                foreground=(s&b&g&r).reshape(64,16,64,16).mean((1,3)).ravel()
                background=(~s&~union).reshape(64,16,64,16).mean((1,3)).ravel()
                p=legacy.reference_coverage(source)
                one['saved_masks_reference_query_seed_seconds']=time.monotonic()-begin
                tick=time.monotonic();rr=reader.load(source['raw']['reference_full'])
                rq=rr if source['raw']['reference_full']['key']==source['raw']['query_full']['key'] else reader.load(source['raw']['query_full'])
                one['old_reference_query_raw_IO_seconds']=time.monotonic()-tick
                tick=time.monotonic();failure=None;runtime='mps_float32_Gram_readout_cpu64_ridge';one['failed_MPS_seconds']=0.
                try:
                    dense=core.compute(rr,rq,p,foreground,background,device='mps',validate=i<3)
                    torch.mps.synchronize()
                except RuntimeError as exc:
                    failure=str(exc);one['failed_MPS_seconds']=time.monotonic()-tick
                    try:torch.mps.synchronize()
                    except RuntimeError:pass
                    torch.mps.empty_cache();tick=time.monotonic();runtime='cpu_float32_actual_backend_fallback_cpu64_ridge'
                    dense=core.compute(rr,rq,p,foreground,background,device='cpu',validate=i<3)
                one['three_task_discriminant_fields_seconds']=time.monotonic()-tick
                max_mps_allocated=max(max_mps_allocated,torch.mps.current_allocated_memory())
                max_mps_driver=max(max_mps_driver,torch.mps.driver_allocated_memory())
                seed_sha={'legal_reference':legacy.tensor_sha(p),'query_foreground':legacy.tensor_sha(foreground),'query_background':legacy.tensor_sha(background)}
                del rr,rq,p,foreground,background
                tick=time.monotonic();payload={'original_hw':np.asarray(source['row']['query_size_hw'])};frames={}
                for frame in ('cli','original'):
                    outputs,frames[frame]=core.compose_frame(masks[frame],source_score,dense)
                    assert tuple(outputs)==tuple(core.ARMS)
                    for arm,value in outputs.items():payload[frame+'/'+arm]=np.packbits(value)
                    del outputs
                one['two_frame_complete_readout_budget_seconds']=time.monotonic()-tick
                tick=time.monotonic();path=HERE/'predictions'/f'{i:06}.npz'
                temporary=path.with_suffix('.pending.npz');np.savez_compressed(temporary,**payload);temporary.replace(path)
                record=dict(index=i,source_index=source['index'],binding_index=source['binding_index'],
                            episode_id=binding['episode_id'],dataset='lvis',prediction_sha256=sha(path),
                            filename=path.name,task_fit_stats=dense['stats'],dense_runtime=runtime,MPS_failure=failure,
                            seed_sha256=seed_sha,frames=frames,query_GT_reads=0,new_encoder_calls=0,new_CRF_calls=0)
                write(HERE/'records'/f'{i:06}.json',record)
                one['packed_output_record_write_seconds']=time.monotonic()-tick
                record.update(costs=one,total_seconds=time.monotonic()-begin)
                write(HERE/'records'/f'{i:06}.json',record)
                costs.update(one);runtimes[runtime]+=1
                records.append({k:record[k] for k in ('index','source_index','binding_index','episode_id','filename','prediction_sha256')})
                del masks,source_score,dense,payload,frames
                write(HERE/'activity.json',dict(state='RUNNING_FIXED600',pid=os.getpid(),completed=i+1,total=600,protocol_sha256=args.protocol_sha))
                if i<3 or (i+1)%50==0:print(dict(completed=i+1,total=600,seconds=time.monotonic()-started),flush=True)
            assert len(records)==600 and not guard.attempts
            seal=dict(state='ALL600_FOUR_COMPLETE_ARMS_TWO_FRAMES_SEALED_BEFORE_GT',n=600,mask_count=4800,
                      protocol_sha256=args.protocol_sha,config_sha256=sha(HERE/'config.json'),records=records,
                      query_GT_guard=guard.receipt(),new_encoder_calls=0,new_CRF_calls=0,new_dense_archives=0)
            write(HERE/'sealed.json',seal)
        receipt=dict(state=seal['state'],n=600,mask_count=4800,wall_seconds=time.monotonic()-started,
                     CPU_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     phase_seconds=dict(costs),raw_reader=dict(reader.cost,unique_keys=len(reader.used)),
                     runtime_counts=dict(runtimes),guard=guard.receipt(),seal_sha256=sha(HERE/'sealed.json'),
                     new_encoder_calls=0,new_CRF_calls=0,new_dense_archives=0,resumed=False,query_RGB_reads=0,
                     reference_raw_requests=600,query_raw_requests=600,source_raw_reads=600,
                     MPS_observed_allocated_max_bytes=max_mps_allocated,MPS_observed_driver_max_bytes=max_mps_driver,
                     MPS_memory_sampling='after_each_episode_compute_not_true_allocation_peak',
                     raw_IO_measurement='requested_container_bytes_and_load_time_not_physical_disk_IO')
        write(HERE/'production_receipt.json',receipt)
        write(HERE/'activity.json',dict(state=seal['state'],pid=None,completed=600,total=600))
        print(receipt,flush=True)


if __name__=='__main__':main()
