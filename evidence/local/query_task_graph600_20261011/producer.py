"""Task-seeded query graph around exact existing strong600; no new encoder."""
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
from PIL import Image
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
core = module('fixed600_query_graph_core', HERE/'core.py')
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
        config = dict(state='FIXED_EXISTING600_TASK_SEEDED_QUERY_GRAPH_FULL_COMPOSITION',n=600,
                      protocol_sha256=args.protocol_sha,protocol_path=str(args.protocol.resolve()),
                      arms=list(core.ARMS),source_sha256=protocol['source_sha256'],
                      membership_sha256=sha(ALIGN/'membership.json'),workers=1,CPU_threads=4)
        write(HERE/'config.json',config)
        (HERE/'protocol.frozen.json').write_bytes(args.protocol.read_bytes())
        started, cpu = time.monotonic(),time.process_time()
        costs, runtimes, records = Counter(),Counter(),[]
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
                row=score_binding['row'];image=BASE.parent/'cv_data'/row['query_path']
                assert sha(image)==row['source_files'][row['query_path']]
                with Image.open(image) as im:
                    qrgb=np.asarray(im.convert('RGB'))
                if row['query_crop'] is not None:
                    x0,y0,x1,y1=row['query_crop'];qrgb=qrgb[y0:y1,x0:x1]
                assert list(qrgb.shape[:2])==row['query_size_hw']
                from ics.official_data import array_hash
                assert array_hash(qrgb)==row['query_rgb_hash']
                can=np.asarray(Image.fromarray(qrgb).resize((1024,1024),Image.Resampling.BILINEAR)).astype(np.float64)/255
                rgb64=can.reshape(64,16,64,16,3).mean((1,3))
                one['saved_mask_queryRGB_seed_seconds']=time.monotonic()-begin
                tick=time.monotonic();rq=reader.load(source['raw']['query_full'])
                one['old_query_raw_IO_seconds']=time.monotonic()-tick
                tick=time.monotonic();failure=None;runtime='mps_float32_edges_cpu64_CG';one['failed_MPS_seconds']=0.
                try:
                    dense=core.compute(rq,rgb64,foreground,background,device='mps')
                    torch.mps.synchronize()
                except RuntimeError as exc:
                    failure=str(exc);one['failed_MPS_seconds']=time.monotonic()-tick
                    try:torch.mps.synchronize()
                    except RuntimeError:pass
                    torch.mps.empty_cache();tick=time.monotonic();runtime='cpu_float32_edges_actual_backend_fallback_cpu64_CG'
                    dense=core.compute(rq,rgb64,foreground,background,device='cpu')
                one['query_edges_three_CG_fields_seconds']=time.monotonic()-tick
                seed_sha={'foreground':legacy.tensor_sha(foreground),'background':legacy.tensor_sha(background),'query_RGB64':legacy.tensor_sha(rgb64)}
                del rq,qrgb,can,rgb64,foreground,background
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
                            filename=path.name,graph_stats=dense['stats'],dense_runtime=runtime,MPS_failure=failure,
                            seed_sha256=seed_sha,frames=frames,query_GT_reads=0,new_encoder_calls=0,new_CRF_calls=0,
                            query_RGB_file_sha256=sha(image))
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
                     new_encoder_calls=0,new_CRF_calls=0,new_dense_archives=0,resumed=False,query_RGB_reads=600,source_raw_reads=0)
        write(HERE/'production_receipt.json',receipt)
        write(HERE/'activity.json',dict(state=seal['state'],pid=None,completed=600,total=600))
        print(receipt,flush=True)


if __name__=='__main__':main()
