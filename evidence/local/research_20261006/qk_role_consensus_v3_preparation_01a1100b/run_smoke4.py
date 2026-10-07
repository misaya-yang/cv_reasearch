"""Fixed QK v3 remaining3 cache inference, first-case reuse, unified seal/no GT."""
import argparse, gc, hashlib, json, os, resource, shutil, sys, threading, time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('code','root','out'):parser.add_argument('--'+name,required=True,type=Path)
    a=parser.parse_args();a.out.mkdir(exist_ok=False);started=time.perf_counter();peak=[0]
    def guard():
        while True:
            rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
            peak[0]=max(peak[0],rss)
            if rss>4<<30 or time.perf_counter()-started>180:
                (a.out/'guard_failure.json').write_text(json.dumps({'rss':rss,'elapsed':time.perf_counter()-started}));os._exit(88)
            time.sleep(.2)
    threading.Thread(target=guard,daemon=True).start()
    sys.path.insert(0,str(a.code/'src'))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import timm
    from safetensors import safe_open
    from PIL import Image
    from ics.methods.qk_role_consensus_v3 import Config,extract_pre_rope,predict
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows_path=a.root/'bound600_v2/smoke4.json';rows=json.loads(rows_path.read_text())
    m5seal_path=a.root/'runs/pro_message_audit_v1/smoke4_fixed_v1/merged_sealed.json'
    m5seal=json.loads(m5seal_path.read_text())
    if sha(rows_path)!=m5seal['source_rows_sha256'] or len(rows)!=4:raise ValueError('smoke4 metadata mismatch')
    first=a.root/'runs/qk_role_v3_feasibility_first_v1';first_receipt=json.loads((first/'receipt.json').read_text())
    method_sha=sha(a.code/'src/ics/methods/qk_role_consensus_v3.py')
    if first_receipt['source_sha256']['method']!=method_sha or first_receipt['info']['config']!=Config().__dict__:
        raise ValueError('Frozen first-case implementation/config changed')
    model_dir=Path('/root/demo4_cache/models/dinov3-vitl16-timm')
    config_path=model_dir/'config.json';weights_path=model_dir/'model.safetensors'
    weight_sha=sha(weights_path);config_sha=sha(config_path)
    if weight_sha!=first_receipt['actual_weights_sha256'] or config_sha!=first_receipt['model_config_sha256']:
        raise ValueError('Weight/config producer changed')
    init=time.perf_counter()
    model=timm.create_model(json.loads(config_path.read_text())['architecture'],pretrained=False,num_classes=0,device='meta')
    block=model.blocks[20].to_empty(device='cpu').float().eval().requires_grad_(False)
    with safe_open(weights_path,framework='pt',device='cpu') as weights:
        state={key[10:]:weights.get_tensor(key) for key in weights.keys() if key.startswith('blocks.20.')}
        block.load_state_dict(state,strict=True)
    load_seconds=time.perf_counter()-init
    evaluation_path=a.root/'runs/nine_public600_v2/evaluation_manifest.json'
    evaluation=json.loads(evaluation_path.read_text())
    complete=[]
    for index,row in enumerate(rows):
        source=m5seal['cases'][index]
        if source['key']!=row['key']:raise ValueError('M5 cache case ordering mismatch')
        binding_path=Path(source['binding_path'])
        if sha(binding_path)!=source['binding_sha256']:raise ValueError('M5 native binding changed')
        binding=json.loads(binding_path.read_text());producer=binding['producer']
        if (producer['weights_sha256']!=weight_sha or producer['native_source_sha256']!='a8c9807ef5e8dabc725c1e2a89439760a900d6af6cdafefea18fc121173643a2'
                or producer['H20']!='actual_after_block20' or producer['dtype']!='float32'
                or binding['image_transform']!='PIL_RGB_bilinear1024_ImageNet_FP32'):
            raise ValueError('Native producer mismatch; do not synthesize/cache-mix')
        case=a.out/f'case{index}';case.mkdir()
        er=next(r for r in evaluation if r['key']==row['key'])
        host_path=a.root/'runs/nine_public600_v2/fields'/f'{er["occurrence_id"]}.npz'
        if sha(host_path)!=producer['host_field_sha256']:raise ValueError('Complete MEAN host field differs')
        if index==0:
            for name in ('fields.npz','masks.npz'):
                if sha(first/name)!=first_receipt['files_sha256'][name]:raise ValueError('First-case sealed prediction changed')
                shutil.copy2(first/name,case/name)
            record={'index':index,'key':row['key'],'reused_prediction':True,'result':first_receipt,
                    'native_pair_path':str(a.root/'runs/pro_message_audit_v1/outputs_v2/episode_00_0_0_72/native_pair.pt'),
                    'native_pair_sha256':first_receipt['native_pair_sha256'],'extraction_seconds':first_receipt['actual_projection_extract_seconds'],
                    'prediction_seconds':first_receipt['evidence_and_complete_controls_seconds'],'new_processing_seconds':0.}
        else:
            case_start=time.perf_counter();pair_path=Path(source['directory'])/'native_pair.pt'
            cache_start=time.perf_counter();pair_sha=sha(pair_path)
            pair=torch.load(pair_path,map_location='cpu',weights_only=True,mmap=True)
            cache_seconds=time.perf_counter()-cache_start
            for role in ('reference','query'):
                cache=pair[role]
                if (cache['h20'].shape!=(4101,1024) or cache['h20'].dtype!=torch.float32
                        or not torch.equal(cache['patch_ids'],torch.arange(5,4101))
                        or cache['projected_native'].shape!=(4096,1024) or cache['projected_native'].dtype!=torch.float32
                        or cache['producer']!=producer):raise ValueError('Native cache role/shape/index/producer mismatch')
            extract_start=time.perf_counter()
            with torch.inference_mode(),torch.autocast(device_type='cpu',enabled=False):
                q,k,teacher=extract_pre_rope(block,torch.stack((pair['reference']['h20'],pair['query']['h20'])),prefix=5)
            extraction_seconds=time.perf_counter()-extract_start
            with Image.open(Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')/Path(row['support']).with_suffix('.png')) as im:
                mask=np.asarray(im)==row['c']+1
            mask=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((1024,1024),Image.Resampling.NEAREST)).copy()
            cov=F.interpolate(torch.from_numpy(mask.astype(np.float32))[None,None],(64,64),mode='area')[0,0].numpy()
            with np.load(host_path,allow_pickle=False) as packet:base=packet['mean.control'].copy()
            prediction_start=time.perf_counter()
            result=predict(q[0],k[0],q[1],k[1],cov,base,pair['reference']['h20'][pair['reference']['patch_ids']],
                           pair['query']['h20'][pair['query']['patch_ids']],pair['reference']['projected_native'],
                           pair['query']['projected_native'],binding['query_HW'],Config())
            prediction_seconds=time.perf_counter()-prediction_start
            np.savez_compressed(case/'fields.npz',**result['fields'])
            np.savez_compressed(case/'masks.npz',**{arm+'.work':np.packbits(ms[0]) for arm,ms in result['masks'].items()},
                                **{arm+'.original':np.packbits(ms[1]) for arm,ms in result['masks'].items()})
            record={'index':index,'key':row['key'],'reused_prediction':False,'info':result['info'],'teacher':teacher,
                    'native_pair_path':str(pair_path),'native_pair_sha256':pair_sha,'cache_load_hash_seconds':cache_seconds,
                    'extraction_seconds':extraction_seconds,'prediction_seconds':prediction_seconds,
                    'new_processing_seconds':time.perf_counter()-case_start,'native_pair_unchanged':sha(pair_path)==pair_sha}
            del pair,q,k,result;gc.collect()
        record.update(binding_path=str(binding_path),binding_sha256=sha(binding_path),native_producer=producer,
                      original_hw=binding['query_HW'],original_GT_mapping=source['original_GT_mapping'],
                      host_field_path=str(host_path),host_sha256=sha(host_path),packet_path=row['packet_export'],
                      packet_sha256=sha(row['packet_export']),fields_sha256=sha(case/'fields.npz'),masks_sha256=sha(case/'masks.npz'),
                      prior_native_capture_seconds=source['capture_seconds'],prior_capture_threads=4,
                      method_sha256=method_sha,query_GT_read=False)
        (case/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');complete.append(record)
        print(json.dumps({'case':row['key'],'reuse':index==0,'new_processing_seconds':record['new_processing_seconds']}),flush=True)
    files={str(f.relative_to(a.out)):sha(f) for f in a.out.rglob('*') if f.is_file()}
    seal={'state':'ALL4_FIXED_QK_V3_PREDICTIONS_SEALED_BEFORE_QUERY_GT','cases':complete,'files_sha256':files,
          'n':4,'query_GT_read':False,'independent_method_increment':0,'family_revision':'QK v3','threads':1,'RSS_limit_bytes':4<<30,
          'peak_sample_RSS_bytes':peak[0],'ru_maxrss_linux_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'model_block_load_seconds':load_seconds,'new_batch_wall_seconds':time.perf_counter()-started,
          'new_encoding_performed':False,'deployment_encoding_cost_not_claimed_zero':True,
          'M5_native_index_sha256':sha(m5seal_path),'evaluation_manifest_sha256':sha(evaluation_path),'rows_sha256':sha(rows_path),
          'actual_weight_sha256':weight_sha,'actual_config_sha256':config_sha,'runner_sha256':sha(__file__)}
    (a.out/'sealed.json').write_text(json.dumps(seal,indent=2)+'\n');print(json.dumps({'state':seal['state'],'wall':seal['new_batch_wall_seconds'],'rss':peak[0]}))


if __name__=='__main__':main()
