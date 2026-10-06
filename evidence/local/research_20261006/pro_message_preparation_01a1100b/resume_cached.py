"""Complete one actual M5 episode using immutable native cache; no encoder/GT."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
os.environ.update(HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_MAX_ACTIVE_LEVELS='1')
import argparse,json,time,threading,sys,hashlib,traceback


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(exist_ok=False)
    sys.path.insert(0,str(args.code/'src'))
    started=time.perf_counter();state={'state':'LOADING_IMMUTABLE_NATIVE_CACHE','pid':os.getpid(),'threads':4,
                                     'new_encoder_calls':0,'query_GT_read':False}
    lock=threading.Lock()
    def write(name,data):
        target=args.out/name;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(target)
    def progress(label,**values):
        with lock:
            state.update(state=label,elapsed_seconds=time.perf_counter()-started,**values);write('state.json',state)
        print(json.dumps({'state':label,'elapsed_seconds':state['elapsed_seconds'],**values}),flush=True)
    def guard():
        while True:
            rss=next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
            with lock:
                state['peak_RSS_bytes']=max(state.get('peak_RSS_bytes',0),rss)
                if rss>8<<30 or time.perf_counter()-started>900:
                    state.update(state='RESOURCE_BUDGET_STOP',elapsed_seconds=time.perf_counter()-started);write('state.json',state);os._exit(137)
            time.sleep(.5)
    threading.Thread(target=guard,daemon=True).start()
    try:
        import torch,numpy as np
        from PIL import Image
        import torch.nn.functional as F
        from ics.methods import pro_message_extrapolation as m
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        ep=args.run/'outputs_v2/episode_00_0_0_72'
        binding=json.loads((ep/'binding.json').read_text());row=binding['row']
        assets=json.loads((args.run/'outputs_v2/assets_first.json').read_text())
        pipe=m.load_local_eva_pipeline(assets,torch.device('cpu'));model=pipe['model']
        data=torch.load(ep/'native_pair.pt',map_location='cpu',weights_only=True)
        native_hash=sha(ep/'native_pair.pt')
        caches={}
        for label in ('reference','query'):
            saved=data[label]
            adapters=[]
            for layer,b in enumerate(model.blocks[20:24]):
                adapter=m.EvaBlockAdapter(b,rope=saved['rope'][layer],captured_only=True)
                adapters.append(adapter)
            caches[label]=m.NativeCache(saved['h20'],saved['patch_ids'],saved['qkv'],saved['native_states'],
                                       tuple(adapters),model.norm,pipe['project'],saved['projected_native'],saved['producer'])
        ann=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')/Path(row['support']).with_suffix('.png')
        with Image.open(ann) as im:mask=np.asarray(im)==row['c']+1
        canonical=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((1024,1024),Image.Resampling.NEAREST)).copy()
        cov=F.interpolate(torch.from_numpy(canonical.astype(np.float32))[None,None],(64,64),mode='area')[0,0].numpy()
        root=args.run.parents[1]
        inference=json.loads((root/'runs/nine_public600_v2/inference_manifest.json').read_text())
        occurrence=next(r['occurrence_id'] for r in inference if r['key']==row['key'])
        host_path=root/'runs/nine_public600_v2/fields'/f'{occurrence}.npz'
        with np.load(host_path,allow_pickle=False) as a:base=a['mean.control'].copy()
        progress('ALL_ROI_STRICT_NATIVE_AUDIT_AND_COMPLETE_BRANCHES',source_native_cache_SHA256=native_hash,
                 host_variant=binding['producer']['host_variant'],cold_load_seconds=time.perf_counter()-started)
        inference_started=time.perf_counter();original=m.replay_region;counter={'n':0}
        def logged(*args,**kwargs):
            result=original(*args,**kwargs);counter['n']+=1
            if counter['n']%3==0:progress('ALL_ROI_STRICT_NATIVE_AUDIT_AND_COMPLETE_BRANCHES',
                         completed_region_triplets=counter['n']//3,inference_seconds=time.perf_counter()-inference_started)
            return result
        m.replay_region=logged
        try:
            r,q=caches['reference'],caches['query']
            result=m.predict(q,r,q.projected_native.numpy(),r.projected_native.numpy(),cov,base,
                              tuple(row['query_image_hw']),reference_has_foreground=bool(mask.any()))
        finally:m.replay_region=original
        inference_seconds=time.perf_counter()-inference_started
        native=[a['native'] for a in result['info']['reference_audits']]
        native += [a['native'] for p in result['info']['query_audits'] for a in p['regions']]
        errors={str(block):max(e['maximum_absolute_error'] for a in native for e in a['native_layer_errors']
                             if e['block']==block) for block in range(21,25)}
        if len(native)!=69 or not all(e['passed'] for a in native for e in a['native_layer_errors']):
            raise RuntimeError('Expected strict passing 60Q+9R native ROI audits')
        np.savez_compressed(args.out/'fields.npz',**result['fields'])
        np.savez_compressed(args.out/'masks.npz',**{k+'.work':v for k,v in result['work_masks'].items()},
                                                  **{k+'.original':v for k,v in result['original_masks'].items()})
        write('inference_diagnostics.json',result['info'])
        host_work,host_original=m.render(base,tuple(row['query_image_hw']))
        if not np.array_equal(result['work_masks']['zero'],host_work) or not np.array_equal(result['original_masks']['zero'],host_original):
            raise RuntimeError('Zero-control whole mask differs from exact host renderer')
        unchanged=sha(ep/'native_pair.pt')==native_hash
        if not unchanged:raise RuntimeError('Immutable native cache changed')
        receipt={'state':'PREDICTIONS_SEALED','episode':row['key'],'new_encoder_calls':0,'query_GT_read':False,
                 'native_all_ROI_count':len(native),'native_all_ROI_layer_maxabs':errors,
                 'native_atol':5e-5,'native_rtol':5e-5,'inference_seconds':inference_seconds,
                 'complete_cached_entry_seconds':time.perf_counter()-started,'peak_RSS_bytes':state.get('peak_RSS_bytes',0),
                 'host_variant':binding['producer']['host_variant'],'host_field_sha256':sha(host_path),
                 'native_cache_path':str(ep/'native_pair.pt'),'native_cache_sha256':native_hash,'native_cache_unchanged':unchanged,
                 'model_source_sha256':assets['source_sha256'],'weights_sha256':assets['weights_sha256'],
                 'method_sha256':sha(args.code/'src/ics/methods/pro_message_extrapolation.py'),
                 'fields_sha256':sha(args.out/'fields.npz'),'masks_sha256':sha(args.out/'masks.npz'),
                 'query_HW':row['query_image_hw'],'work_HW':[1024,1024],
                 'arms':list(result['fields']),'all_masks_bool':True,'zero_field_and_both_masks_exact':True,
                 'quality':'unscored','binding_path':str(ep/'binding.json')}
        write('sealed.json',receipt)
        progress('PREDICTIONS_SEALED',native_all_ROI_layer_maxabs=errors,inference_seconds=inference_seconds)
    except BaseException as error:
        progress('FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
