"""Owned bounded CPU audit/continuation runner; no downloads or query GT reads."""
import argparse, hashlib, json, os, resource, sys, threading, time, traceback
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code',required=True);p.add_argument('--rows',required=True)
    p.add_argument('--out',required=True);p.add_argument('--phase',choices=['audit','predict'],default='audit')
    p.add_argument('--count',type=int,default=1)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('New output directory required')
    out.mkdir(parents=True)
    sys.path.insert(0,str(Path(args.code)/'src'))
    peak={'rss_bytes':0}
    def guard():
        while True:
            try:
                lines=Path('/proc/self/status').read_text().splitlines()
                rss=int(next(line.split()[1] for line in lines if line.startswith('VmRSS:')))*1024
                peak['rss_bytes']=max(peak['rss_bytes'],rss)
                if rss>8*1024**3:
                    (out/'guard_failure.json').write_text(json.dumps({'RSS_bytes':rss,'limit_bytes':8*1024**3}))
                    os._exit(88)
            except (StopIteration,FileNotFoundError):pass
            time.sleep(.2)
    threading.Thread(target=guard,daemon=True).start()
    started=time.perf_counter()
    try:
        import numpy as np
        import torch
        from PIL import Image
        import torch.nn.functional as F
        from ics.data import TimmDINOv3
        from ics.methods.pro_common_context import Config,capture_native,audit_native,predict
        from ics.native_basis import load_native_basis
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        rows=json.loads(Path(args.rows).read_text())[:args.count]
        weights=Path('/root/demo4_cache/models/dinov3-vitl16-timm')
        image_root=Path('/root/demo4_cache/data/COCO2014')
        annotation_root=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')
        basis_path=Path('/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt')
        init=time.perf_counter()
        model=TimmDINOv3(str(weights)).to('cpu').float().eval().requires_grad_(False).m
        model_seconds=time.perf_counter()-init
        def rgb(path):
            with Image.open(path) as im:
                im=im.convert('RGB');hw=(im.height,im.width)
                array=np.asarray(im.resize((1024,1024),Image.Resampling.BILINEAR)).copy()
            t=torch.from_numpy(array).permute(2,0,1).float()/255
            return (t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None],hw
        results=[]
        for occurrence,row in enumerate(rows):
            episode_start=time.perf_counter()
            r,rhw=rgb(image_root/row['support']);q,qhw=rgb(image_root/row['query'])
            images=torch.stack((r,q))
            print(json.dumps({'stage':'native_start','occurrence':occurrence,'key':row['key'],'elapsed':time.perf_counter()-started}),flush=True)
            if args.phase=='audit':
                t=time.perf_counter();cache=capture_native(model,images,Config());capture_seconds=time.perf_counter()-t
                t=time.perf_counter();audit=audit_native(model,cache,Config());audit_seconds=time.perf_counter()-t
                result={'key':row['key'],'reference':row['support'],'query':row['query'],'capture_seconds':capture_seconds,
                        'native_audit_suffix_seconds':audit_seconds,'audit':audit,'source':cache['source'],
                        'actual_shapes':{'raw_H20':list(cache['raw_h20'].shape),'query_KV':list(cache['layers'][20]['k'].shape)},
                        'raw_H20_origin':'actual_block21_pre_hook','cache_final_bytes':cache['final'].numel()*4,
                        'query_GT_read':False,'RSS_peak_bytes':peak['rss_bytes']}
                del cache
            else:
                ann_path=annotation_root/Path(row['support']).with_suffix('.png')
                with Image.open(ann_path) as im:
                    mask=Image.fromarray((np.asarray(im)==int(row['c'])+1).astype(np.uint8)*255)
                    binary=np.asarray(mask.resize((1024,1024),Image.Resampling.NEAREST)).copy()>0
                coverage=F.interpolate(torch.from_numpy(binary.astype(np.float32))[None,None],(64,64),mode='area')[0,0].numpy()
                with np.load(row['packet_export'],allow_pickle=False) as packet:score=packet['score'].copy()
                exported=torch.load(row['feature_export'],map_location='cpu',weights_only=True)
                gate=bool(exported['debiased']);del exported
                basis,basis_meta=load_native_basis(basis_path)
                prediction=predict(model,images,coverage,score,qhw,Config(),basis=basis if gate else None)
                target=out/f'{occurrence:04d}_{row["key"]}';target.mkdir()
                np.savez_compressed(target/'fields.npz',**prediction['fields'])
                np.savez_compressed(target/'masks.npz',**{f'{arm}_1024':np.packbits(ms[0]) for arm,ms in prediction['masks'].items()},
                                    **{f'{arm}_original':np.packbits(ms[1]) for arm,ms in prediction['masks'].items()})
                result={'key':row['key'],'info':prediction['info'],'query_original_hw':qhw,'reference_original_hw':rhw,
                        'gate_from_existing_feature_cache':gate,'native_basis':basis_meta,'query_GT_read':False,
                        'variant':'cachehost_new_FP32_native_graph_not_historical_MEAN_parity',
                        'host_packet_path':row['packet_export'],'host_packet_sha256':sha(row['packet_export']),
                        'old_feature_path':row['feature_export'],'feature_cache_sha256':sha(row['feature_export']),
                        'reference_rgb_sha256':sha(image_root/row['support']),'query_rgb_sha256':sha(image_root/row['query']),
                        'support_annotation_sha256':sha(ann_path),'score_keys_read':['score'],
                        'working_foreground_pixels':{arm:int(ms[0].sum()) for arm,ms in prediction['masks'].items()},
                        'complete_source_producer_is_historical_cache':True,'RSS_peak_bytes':peak['rss_bytes']}
                (target/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
                del prediction,basis
            result['episode_wall_seconds']=time.perf_counter()-episode_start
            results.append(result)
            (out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({'stage':'episode_done','key':row['key'],'wall_seconds':result['episode_wall_seconds'],'rss_peak_bytes':peak['rss_bytes']}),flush=True)
        receipt={'state':'REAL_NATIVE_AUDIT_PASSED' if args.phase=='audit' else 'FOUR_ARM_PREDICTIONS_SEALED_UNSCORED',
                 'phase':args.phase,'threads':4,'RSS_limit_bytes':8*1024**3,'RSS_peak_bytes':peak['rss_bytes'],
                 'ru_maxrss_linux_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':time.perf_counter()-started,
                 'model_load_seconds':model_seconds,'results':results,'query_GT_read':False,'weights_sha256':sha(weights/'model.safetensors'),
                 'model_config_sha256':sha(weights/'config.json'),'rows_sha256':sha(args.rows),
                 'source_sha256':{'module':sha(Path(args.code)/'src/ics/methods/pro_common_context.py'),'runner':sha(__file__)},
                 'hostname':os.uname().nodename,'cuda_used':False,'actual_encoder':'timm Eva'}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt,indent=2),flush=True)
    except Exception as exc:
        failure={'state':'FAILED','error':str(exc),'traceback':traceback.format_exc(),'wall_seconds':time.perf_counter()-started,
                 'RSS_peak_bytes':peak['rss_bytes'],'ru_maxrss_linux_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        (out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        print(json.dumps(failure,indent=2),flush=True);raise


if __name__=='__main__':main()
