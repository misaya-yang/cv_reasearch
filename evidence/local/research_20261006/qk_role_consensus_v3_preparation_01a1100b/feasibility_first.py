"""One CPU, one existing native case; locked QK v3 feasibility only, no GT."""
import argparse, hashlib, json, os, resource, sys, threading, time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('code','pair','binding','host','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(exist_ok=False);started=time.perf_counter();peak=[0]
    def guard():
        while True:
            rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
            peak[0]=max(peak[0],rss)
            if rss>4<<30 or time.perf_counter()-started>180:
                (a.out/'guard_failure.json').write_text(json.dumps({'RSS_bytes':rss,'elapsed':time.perf_counter()-started}))
                os._exit(88)
            time.sleep(.2)
    threading.Thread(target=guard,daemon=True).start()
    sys.path.insert(0,str(a.code/'src'))
    import numpy as np
    import torch
    import timm
    import torch.nn.functional as F
    from safetensors import safe_open
    from PIL import Image
    from ics.methods.qk_role_consensus_v3 import Config,extract_pre_rope,predict
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    input_hash=sha(a.pair)
    pair=torch.load(a.pair,map_location='cpu',weights_only=True,mmap=True)
    binding=json.loads(a.binding.read_text());row=binding['row']
    config_path=Path('/root/demo4_cache/models/dinov3-vitl16-timm/config.json')
    weight_path=config_path.with_name('model.safetensors')
    config=json.loads(config_path.read_text())
    init=time.perf_counter()
    meta=timm.create_model(config['architecture'],pretrained=False,num_classes=0,device='meta')
    block=meta.blocks[20].to_empty(device='cpu').float().eval().requires_grad_(False)
    with safe_open(weight_path,framework='pt',device='cpu') as weights:
        state={key[len('blocks.20.'):]:weights.get_tensor(key) for key in weights.keys() if key.startswith('blocks.20.')}
        block.load_state_dict(state,strict=True)
    load_seconds=time.perf_counter()-init
    extraction_start=time.perf_counter()
    with torch.inference_mode(),torch.autocast(device_type='cpu',enabled=False):
        q,k,producer=extract_pre_rope(block,torch.stack((pair['reference']['h20'],pair['query']['h20'])),prefix=5)
    extraction_seconds=time.perf_counter()-extraction_start
    with Image.open(Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')/Path(row['support']).with_suffix('.png')) as im:
        mask=np.asarray(im)==row['c']+1
    mask=np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((1024,1024),Image.Resampling.NEAREST)).copy()
    cov=F.interpolate(torch.from_numpy(mask.astype(np.float32))[None,None],(64,64),mode='area')[0,0].numpy()
    with np.load(a.host,allow_pickle=False) as packet:base=packet['mean.control'].copy()
    raw_r=pair['reference']['h20'][pair['reference']['patch_ids']]
    raw_q=pair['query']['h20'][pair['query']['patch_ids']]
    final_r=pair['reference']['projected_native'];final_q=pair['query']['projected_native']
    inference_start=time.perf_counter()
    result=predict(q[0],k[0],q[1],k[1],cov,base,raw_r,raw_q,final_r,final_q,binding['query_HW'],Config())
    prediction_seconds=time.perf_counter()-inference_start
    np.savez_compressed(a.out/'fields.npz',**result['fields'])
    np.savez_compressed(a.out/'masks.npz',**{name+'.work':np.packbits(ms[0]) for name,ms in result['masks'].items()},
                        **{name+'.original':np.packbits(ms[1]) for name,ms in result['masks'].items()})
    report={'state':'ACTUAL_PRE_ROPE_QK_V3_ONE_CASE_FEASIBILITY_SEALED_UNSCORED','case':row['key'],
            'family_revision':'QK v3','independent_method_increment':0,'threads':1,
            'source':producer,'actual_teacher_block_load_seconds':load_seconds,'actual_projection_extract_seconds':extraction_seconds,
            'evidence_and_complete_controls_seconds':prediction_seconds,'info':result['info'],
            'quality':'unmeasured; no query labels read in this run; case was exposed in prior tasks',
            'native_producer':binding['producer'],'host_variant':'existing_FP16_conditional_cache_MEAN_graph_float64',
            'host_field_path':str(a.host),'host_sha256':sha(a.host),'native_pair_sha256':input_hash,
            'native_pair_unchanged':sha(a.pair)==input_hash,'native_encoding_performed_here':False,
            'standalone_deployment_cost':'must include real R/Q rawH20 encoding, producer/code/basis and full host; cache feasibility does not establish encoder0 deployment',
            'model_config_sha256':sha(config_path),'actual_weights_sha256':sha(weight_path),
            'source_sha256':{'method':sha(a.code/'src/ics/methods/qk_role_consensus_v3.py'),'runner':sha(__file__)},
            'stage_exposure':'same first exposed case; not independent confirmation','query_GT_read':False,
            'head_count':16,'changed_work_pixels_vs_host':{},'wall_seconds':time.perf_counter()-started,
            'ru_maxrss_linux_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_sample_RSS_bytes':peak[0],
            'files_sha256':{name:sha(a.out/name) for name in ('fields.npz','masks.npz')}}
    baseline=result['masks']['mean.control'][0]
    for arm,(work,original) in result['masks'].items():
        report['changed_work_pixels_vs_host'][arm]=int((work!=baseline).sum())
    (a.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
