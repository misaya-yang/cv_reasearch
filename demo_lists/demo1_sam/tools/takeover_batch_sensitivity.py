"""Deterministic repeat and batching sensitivity of the official FP32 decoder."""
from datetime import datetime,timezone
import argparse
import json
from types import SimpleNamespace
import numpy as np
from takeover_common import *


def collect(model,image,pe,dense,sparse,mb):
    outputs=[official_predict(model,image,pe,dense,sparse[i:i+mb]) for i in range(0,len(sparse),mb)]
    return tuple(torch.cat([o[k] for o in outputs]) for k in (0,1))


def delta(a,b,info):
    from segment_anything.modeling.sam import Sam
    proxy=SimpleNamespace(image_encoder=SimpleNamespace(img_size=1024))
    full_a=Sam.postprocess_masks(proxy,a[0],info['input_size'],info['original_size'])>info['threshold']
    full_b=Sam.postprocess_masks(proxy,b[0],info['input_size'],info['original_size'])>info['threshold']
    return {'mask_logit_max_abs':float((a[0]-b[0]).abs().max()),
            'iou_prediction_max_abs':float((a[1]-b[1]).abs().max()),
            'full_resolution_binary_flips':int((full_a!=full_b).sum()),
            'iou_head_argmax_changes':int((a[1][:,1:].argmax(-1)!=b[1][:,1:].argmax(-1)).sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--p128-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    configure();sys.path.insert(0,str(ROOT/'assets/source/segment-anything'))
    source=json.loads((a.run_dir/'report.json').read_text())
    state=torch.load(a.run_dir/'mask_decoder_state.pt',map_location='cpu',weights_only=True)
    model=make_fixture(grid=4,batch=1)[0];model.load_state_dict(state['state_dict'],strict=True);model=model.to('cuda:0').eval()
    report={'status':'RUNNING','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'precision':'float32','tf32':False,
            'original_gate_unchanged':5e-5,'contract':'Same official weights, encoded inputs, GPU and FP32 precision. Repeat comparison holds batch fixed; batching comparison changes only microbatch. No timing/SOTA inference.',
            'cases':[]}
    with torch.inference_mode():
        for item in source['images']:
            with np.load(a.run_dir/item['encoded_inputs']['npz'],allow_pickle=False) as arr:
                image,pe,dense=[torch.from_numpy(arr[k]).to('cuda:0') for k in ('image_embeddings','image_pe','dense_nomask')]
                info={'input_size':tuple(arr['input_size'].tolist()),'original_size':tuple(arr['original_size'].tolist()),'threshold':float(arr['mask_threshold'])}
                for regime in ('central','near_boundary','box'):
                    sparse=torch.from_numpy(arr['sparse_'+regime]).to('cuda:0');mb=len(sparse)
                    ref=collect(model,image,pe,dense,sparse,mb)
                    repeat=collect(model,image,pe,dense,sparse,mb)
                    split=collect(model,image,pe,dense,sparse,1)
                    report['cases'].append({'image_id':item['image_id'],'regime':regime,'prompts':mb,
                                            'same_batch_repeat':delta(ref,repeat,info),'full_batch_vs_batch1':delta(ref,split,info)})
                    del ref,repeat,split
            atomic_json(a.output,report)
        with np.load(a.p128_dir/'encoded_inputs.npz',allow_pickle=False) as arr:
            image,pe,dense=[torch.from_numpy(arr[k]).to('cuda:0') for k in ('image_embeddings','image_pe','dense_nomask')]
            sparse=torch.from_numpy(arr['sparse_perf']).to('cuda:0')
            info={'input_size':tuple(arr['input_size'].tolist()),'original_size':tuple(arr['original_size'].tolist()),'threshold':float(arr['mask_threshold'])}
            ref=collect(model,image,pe,dense,sparse,128)
            repeat=collect(model,image,pe,dense,sparse,128)
            split=collect(model,image,pe,dense,sparse,64)
            report['p128']={'same_batch_repeat':delta(ref,repeat,info),'batch128_vs_batch64':delta(ref,split,info)}
    report['summary']={'cases':len(report['cases']),
                       'same_batch_repeat_max_error':max(x['same_batch_repeat']['mask_logit_max_abs'] for x in report['cases']),
                       'full_vs_batch1_max_error':max(x['full_batch_vs_batch1']['mask_logit_max_abs'] for x in report['cases']),
                       'full_vs_batch1_over_original_gate':sum(x['full_batch_vs_batch1']['mask_logit_max_abs']>5e-5 for x in report['cases']),
                       'full_vs_batch1_total_binary_flips':sum(x['full_batch_vs_batch1']['full_resolution_binary_flips'] for x in report['cases'])}
    report['status']='COMPLETED_OFFICIAL_BATCHING_SENSITIVITY';report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(a.output,report);print(json.dumps({'summary':report['summary'],'p128':report['p128']},indent=2))


if __name__=='__main__':main()
