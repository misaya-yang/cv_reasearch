"""Bounded actual CPU audit worker; runs only in its own Pro M5 namespace."""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'
os.environ['OMP_MAX_ACTIVE_LEVELS']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import hashlib
import json
import threading
import time
import traceback
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case-index',type=int,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.code/'src'))
    started=time.perf_counter()
    status={'state':'INITIALIZING','pid':os.getpid(),'threads':4,'RSS_limit_bytes':8<<30,
            'started_UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'episodes':[]}
    lock=threading.Lock()

    def write(name,data):
        path=args.out/name
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        temporary.replace(path)

    def progress(state,**values):
        with lock:
            status.update(state=state,elapsed_seconds=time.perf_counter()-started,**values)
            write('state.json',status)
        print(json.dumps({'state':state,'elapsed_seconds':status['elapsed_seconds'],**values}),flush=True)

    def guard():
        while True:
            rss=0
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):rss=int(line.split()[1])*1024
            with lock:
                status['peak_RSS_bytes']=max(status.get('peak_RSS_bytes',0),rss)
                if rss>8<<30 or time.perf_counter()-started>900:
                    status.update(state='RESOURCE_BUDGET_STOP',rss_bytes=rss,
                                  elapsed_seconds=time.perf_counter()-started,
                                  reason='RSS>8GiB' if rss>8<<30 else 'first_bounded_run>900s')
                    write('state.json',status)
                    os._exit(137)
            time.sleep(.5)
    threading.Thread(target=guard,daemon=True).start()
    try:
        import torch
        import numpy as np
        from PIL import Image
        from ics.methods import pro_message_extrapolation as method
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if torch.cuda.is_available():raise RuntimeError('Expected CPU-only authorized server')
        rows=json.loads((args.root/'bound600_v2/smoke4.json').read_text())
        inference=json.loads((args.root/'runs/nine_public600_v2/inference_manifest.json').read_text())
        assets={
            'weights_path':'/root/demo4_cache/models/dinov3-vitl16-timm/model.safetensors',
            'weights_sha256':'45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941',
            'source_path':'/root/demo4_cache/env/timm/models/eva.py',
            'source_sha256':'a8c9807ef5e8dabc725c1e2a89439760a900d6af6cdafefea18fc121173643a2',
            'model_config_path':'/root/demo4_cache/models/dinov3-vitl16-timm/config.json',
            'model_config_sha256':'a71f705b0074e173540d0bdbd3aa940fa8d7d3c6c7f020a683004c46ca605b24',
            'basis_path':'/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt',
            'basis_sha256':'9b9b20755a796cbda11bb7220d246ee540106e40cb024e249a5f63f884b6a116'}
        first_feature=torch.load(rows[args.case_index]['feature_export'],map_location='cpu',weights_only=True)
        assets['projection_enabled']=bool(first_feature['debiased']);del first_feature
        write('assets_first.json',assets)
        progress('MODEL_LOADING',fixed_first_gate=assets['projection_enabled'])
        load_started=time.perf_counter()
        pipeline=method.load_local_eva_pipeline(assets,torch.device('cpu'))
        model_load_seconds=time.perf_counter()-load_started
        model=pipeline['model']
        basis=torch.load(assets['basis_path'],map_location='cpu',weights_only=True)['basis']
        progress('MODEL_READY',model_load_seconds=model_load_seconds)
        rgb_root=Path('/root/demo4_cache/data/COCO2014')
        annotation_root=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')
        normalization_mean=torch.tensor([.485,.456,.406])[:,None,None]
        normalization_std=torch.tensor([.229,.224,.225])[:,None,None]
        def image(path):
            with Image.open(path) as im:
                im=im.convert('RGB');hw=(im.height,im.width)
                a=np.asarray(im.resize((1024,1024),Image.Resampling.BILINEAR)).copy()
            x=torch.from_numpy(a).permute(2,0,1).float()/255
            return (x-normalization_mean)/normalization_std,hw

        for index,row in [(args.case_index,rows[args.case_index])]:
            case_started=time.perf_counter()
            out=args.out/f'episode_{index:02d}_{row["key"]}'
            out.mkdir(exist_ok=False)
            feature=torch.load(row['feature_export'],map_location='cpu',weights_only=True)
            projection=bool(feature['debiased']);old_dtype=str(feature['q'].dtype);del feature
            project=(lambda x:x-(x@basis)@basis.T) if projection else (lambda x:x)
            rr=next(r for r in inference if r['key']==row['key'])
            occurrence=rr['occurrence_id']
            field_path=args.root/'runs/nine_public600_v2/fields'/f'{occurrence}.npz'
            with np.load(field_path,allow_pickle=False) as packet:base=packet['mean.control'].copy()
            support_path,query_path=rgb_root/row['support'],rgb_root/row['query']
            reference,reference_hw=image(support_path);query,query_hw=image(query_path)
            reference_annotation=annotation_root/Path(row['support']).with_suffix('.png')
            with Image.open(reference_annotation) as im:raw=np.asarray(im).copy()
            mask=raw==int(row['c'])+1
            if not mask.any():raise ValueError('Bound reference foreground unexpectedly empty')
            canonical=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((1024,1024),Image.Resampling.NEAREST)).copy()
            coverage=torch.nn.functional.interpolate(torch.from_numpy(canonical.astype(np.float32))[None,None],
                       (64,64),mode='area')[0,0].numpy()
            with np.load(row['packet_export'],allow_pickle=False) as packet:old_coverage=packet['cov'].copy()
            producer={'native_source_sha256':assets['source_sha256'],'weights_sha256':assets['weights_sha256'],
                      'H20':'actual_after_block20','dtype':'float32','projection_gate':projection,
                      'gate_source':row['feature_export'],'gate_feature_SHA256':sha(row['feature_export']),
                      'input_key':row['key'],'host_variant':'existing_FP16_conditional_cache_MEAN_graph_float64',
                      'host_field_sha256':sha(field_path),'old_cache_dtype':old_dtype,
                      'projection_gate_recomputed':False}
            write(str(out.relative_to(args.out)/'binding.json'),dict(row=row,producer=producer,
                  image_transform='PIL_RGB_bilinear1024_ImageNet_FP32',reference_HW=reference_hw,query_HW=query_hw,
                  reference_sha256=sha(support_path),query_sha256=sha(query_path),
                  reference_annotation_sha256=sha(reference_annotation),
                  coverage_max_difference_from_old_packet=float(np.max(np.abs(coverage-old_coverage))),
                  query_GT_opened=False))
            progress('NATIVE_PAIR_CAPTURE',case=index,key=row['key'],projection_enabled=projection)
            with torch.inference_mode(),torch.autocast(device_type='cpu',enabled=False):
                r_cache,q_cache,capture=method.capture_native_pair(model,torch.stack((reference,query)),project,
                                                          producer=producer)
                write(str(out.relative_to(args.out)/'capture.json'),capture)
                saved={}
                for label,cache in (('reference',r_cache),('query',q_cache)):
                    saved[label]={'h20':cache.h20,'patch_ids':cache.patch_ids,'qkv':cache.qkv,
                                  'native_states':cache.native_states,'projected_native':cache.projected_native,
                                  'rope':tuple(adapter.rope for adapter in cache.adapters),'producer':cache.producer}
                torch.save(saved,out/'native_pair.pt')
                progress('REAL_NATIVE_AUDIT',case=index,capture_seconds=capture['paired_capture_seconds'])
                probe=[0,1023,2048,4095]
                audits={}
                for label,cache in (('reference',r_cache),('query',q_cache)):
                    _,audits[label]=method.replay_region(cache,probe,1.,audit_native=True)
                write(str(out.relative_to(args.out)/'native_audit.json'),audits)
                progress('COMPLETE_PREDICTION',case=index,native_probe_audits=audits)
                prediction_started=time.perf_counter()
                # Record bounded progress without changing any numerical operation.
                original_replay=method.replay_region
                replay_counter={'n':0}
                def logged_replay(*a,**k):
                    answer=original_replay(*a,**k);replay_counter['n']+=1
                    if replay_counter['n']%3==0:
                        progress('COMPLETE_PREDICTION',case=index,completed_region_triplets=replay_counter['n']//3,
                                 prediction_elapsed_seconds=time.perf_counter()-prediction_started)
                    return answer
                method.replay_region=logged_replay
                try:
                    result=method.predict(q_cache,r_cache,q_cache.projected_native.cpu().numpy(),
                           r_cache.projected_native.cpu().numpy(),coverage,base,query_hw,
                           reference_has_foreground=True)
                finally:method.replay_region=original_replay
            prediction_seconds=time.perf_counter()-prediction_started
            np.savez_compressed(out/'fields.npz',**result['fields'])
            np.savez_compressed(out/'masks.npz',**{k+'.work':v for k,v in result['work_masks'].items()},
                                                **{k+'.original':v for k,v in result['original_masks'].items()})
            all_native=[x['native'] for x in result['info']['reference_audits']]
            all_native += [x['native'] for partition in result['info']['query_audits'] for x in partition['regions']]
            max_errors={str(block):max(x['maximum_absolute_error'] for audit in all_native
                             for x in audit['native_layer_errors'] if x['block']==block) for block in range(21,25)}
            receipt={'state':'COMPLETE_PREDICTIONS_SEALED_UNSCORED','episode':row['key'],'producer':producer,
                      'capture':capture,'native_probe_audits':audits,'all_ROI_native_layer_maxabs':max_errors,
                      'beta1_all_ROI_count':len(all_native),'prediction_seconds':prediction_seconds,
                      'episode_complete_seconds':time.perf_counter()-case_started,'model_load_seconds':model_load_seconds,
                      'original_HW':query_hw,'work_HW':[1024,1024],
                      'arms':list(result['fields']),'query_GT_opened':False,
                      'zero_field_exact':bool(np.array_equal(result['fields']['zero'],base.astype(np.float32))),
                      'mask_foreground_pixels':{k:{'work':int(result['work_masks'][k].sum()),
                                                  'original':int(result['original_masks'][k].sum())} for k in result['fields']},
                      'fields_sha256':sha(out/'fields.npz'),'masks_sha256':sha(out/'masks.npz'),
                      'method_source_sha256':sha(args.code/'src/ics/methods/pro_message_extrapolation.py'),
                      'peak_RSS_bytes':status.get('peak_RSS_bytes',0),'quality':'unscored',
                      'historical_baseline_parity':'not_asserted'}
            write(str(out.relative_to(args.out)/'inference_diagnostics.json'),result['info'])
            write(str(out.relative_to(args.out)/'receipt.json'),receipt)
            status['episodes'].append(receipt)
            progress('EPISODE_COMPLETE',case=index,episode_seconds=receipt['episode_complete_seconds'],
                     prediction_seconds=prediction_seconds,native_layer_maxabs=max_errors)
            del r_cache,q_cache,result
            if False:  # fixed smoke4 already explicitly authorized; no automatic expansion
                progress('STOPPED_AT_MEASURED_COST_GATE',completed=len(status['episodes']),
                         reason='complete first episode >=60s; no remaining4 or24 expansion')
                return
        progress('ASSIGNED_SINGLE_CASE_COMPLETED',completed=len(status['episodes']))
    except BaseException as error:
        progress('FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
