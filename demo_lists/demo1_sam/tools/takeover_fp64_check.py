"""High-precision diagnosis of real decoder rewrites; does not change FP32 gate."""
from datetime import datetime,timezone
import argparse
import json
import numpy as np
from takeover_common import *


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    configure();source=json.loads((a.run_dir/'report.json').read_text())
    bundle=torch.load(a.run_dir/'mask_decoder_state.pt',map_location='cpu',weights_only=True)
    model=make_fixture(grid=4,batch=1)[0];model.load_state_dict(bundle['state_dict'],strict=True)
    model=model.to(device='cuda:0',dtype=torch.float64).eval()
    first=source['images'][0]
    report={'status':'RUNNING','started_at_utc':datetime.now(timezone.utc).isoformat(),'image_id':first['image_id'],
            'scope':'FP32 pretrained weights and saved encoded tensors converted exactly to FP64; complete decoder only. Not a double-precision image encoder, speed result, or FP32 gate pass.',
            'dtype':'float64','fp64_tolerance':1e-10,'fp32_tolerance_unchanged':5e-5,'records':[]}
    with np.load(a.run_dir/first['encoded_inputs']['npz'],allow_pickle=False) as encoded,torch.inference_mode():
        image,pe,dense=[torch.from_numpy(encoded[k]).to(device='cuda:0',dtype=torch.float64) for k in ('image_embeddings','image_pe','dense_nomask')]
        for regime in ('central','near_boundary','box'):
            sparse=torch.from_numpy(encoded['sparse_'+regime]).to(device='cuda:0',dtype=torch.float64)
            reference=official_predict(model,image,pe,dense,sparse)
            for arm in ARMS:
                build,forward,plans=arm_functions(arm,model,image,pe,dense,sparse.shape[1]+5)
                cache=build();output=forward(sparse,cache);check=compare(reference,output,1e-10)
                report['records'].append({'regime':regime,'arm':arm[0],**check})
                atomic_json(a.output,report);print(regime,arm[0],check['masks']['max_abs'],check['passed'],flush=True)
                del cache,output
    report['all_fp64_passed']=all(x['passed'] for x in report['records'])
    report['status']='COMPLETED_FP64_DIAGNOSIS';report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    atomic_json(a.output,report)


if __name__=='__main__':main()
