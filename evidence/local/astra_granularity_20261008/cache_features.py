"""Cache exactly the part_probe.py features; only PACO-Part and COCO fresh600."""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
B = Path('/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9')
O = B / 'outputs'
sys.path.insert(0, str(B / 'src'))
from ics.data import TimmDINOv3

torch.set_num_threads(2)
torch.manual_seed(0)
dev = 'cuda'
net = TimmDINOv3().to(dev).eval()
basis = torch.load('/root/autodl-tmp/demo9_transductive_ics/results/native_runtime_v1/positional_basis.pt', map_location=dev)
if isinstance(basis, dict):
    basis = [v for v in basis.values() if torch.is_tensor(v) and v.ndim == 2][0]
basis = basis.float()
basis = basis if basis.shape[0] == 1024 else basis.T
MEAN = torch.tensor([.485, .456, .406], device=dev).view(3, 1, 1)
STD = torch.tensor([.229, .224, .225], device=dev).view(3, 1, 1)


@torch.no_grad()
def enc(path):
    im = Image.open(path).convert('RGB').resize((1024, 1024), Image.BICUBIC)
    x = torch.from_numpy(np.array(im)).to(dev).permute(2, 0, 1).float() / 255
    with torch.autocast('cuda', dtype=torch.bfloat16):
        t = net.get_intermediate_layers(((x-MEAN)/STD)[None], n=1, reshape=False)[0][0]
    t = F.normalize(t.float(), dim=1)
    return F.normalize(t-(t @ basis) @ basis.T, dim=1).half().cpu().numpy()


def coverage(mask):
    return np.asarray(Image.fromarray((mask*255).astype(np.uint8)).resize((64, 64), Image.BOX), np.float32).reshape(-1)/255


def main():
    cache = ROOT / 'cache'
    cache.mkdir(exist_ok=True)
    packs = [f'paco_part_f{i}' for i in range(4)] + ['coco_fresh600']
    manifests = {}
    for pack in packs:
        folder = O / ('claude_order_fresh600' if pack == 'coco_fresh600' else 'claude_rcg2_'+pack)
        manifests[pack] = json.loads((folder/'rows.json').read_text())
    required = sum(map(len, manifests.values())) * 2 * 4096 * 1024 * 2
    free = shutil.disk_usage(ROOT).free
    if free < required + 2*1024**3:
        raise RuntimeError(f'Insufficient disk: {free=} {required=}')
    (ROOT/'cache_contract.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        basis_shape=list(basis.shape), packs={p:len(r) for p,r in manifests.items()},
        required_feature_bytes=required, free_bytes_before=free,
        encoder='part_probe.py exact path: 1024 BICUBIC, BF16, unit, subtract basis, unit, FP16 cache',
        truth_usage='evaluation arrays only; never passed to candidate',
        previous_holdout_exposure='all packs previously evaluated; newly frozen candidate evaluation is one read, not never-seen data'),indent=2)+'\n')
    start=time.monotonic()
    total=0
    for pack,rows in manifests.items():
        dest=cache/pack
        dest.mkdir(exist_ok=False)
        n=len(rows)
        feat=np.lib.format.open_memmap(dest/'features.npy',mode='w+',dtype=np.float16,shape=(n,2,4096,1024))
        cov=np.zeros((n,4096),np.float32)
        field=np.zeros((n,4096),np.float16)
        truth=np.zeros((n,4096),bool)
        if pack=='coco_fresh600':
            z=np.load(O/'claude_order_fresh600/tokens.npz')
            data=Path('/root/demo4_cache/data/COCO2014')
        else:
            z=np.load(O/('claude_rcg2_'+pack)/'counts.npz')
            data=O/'claude_packs'/pack
        for i,row in enumerate(rows):
            if pack=='coco_fresh600':
                rp,qp=data/row['support'],data/row['query']
                rm=np.asarray(Image.open(data/'annotations'/Path(row['support']).with_suffix('.png'))) == row['c']+1
                field[i]=z['rcg'][i].reshape(-1)
            else:
                rp,qp=data/'data'/row['support'],data/'data'/row['query']
                rm=np.asarray(Image.open(data/'ann'/row['support']).convert('L'))>0
                field[i]=z['fields'][i].reshape(-1)
            feat[i,0]=enc(qp)
            feat[i,1]=enc(rp)
            cov[i]=coverage(rm.astype(np.float32))
            # Evaluation is stored separately from inference-available arrays.
            if pack=='coco_fresh600':
                truth[i]=z['truth'][i]>.5
            else:
                tm=np.asarray(Image.open(data/'ann'/row['query']).convert('L'))>0
                truth[i]=coverage(tm.astype(np.float32))>.5
            total+=1
            if (i+1)%25==0 or i+1==n:
                feat.flush()
                state=dict(pack=pack,n=i+1,total=total,elapsed_s=time.monotonic()-start,free_bytes=shutil.disk_usage(ROOT).free)
                (ROOT/'cache_progress.json').write_text(json.dumps(state)+'\n')
                print(json.dumps(state),flush=True)
        feat.flush()
        np.savez(dest/'inference.npz',cov=cov,field=field)
        np.savez(dest/'evaluation.npz',truth=truth)
        (dest/'rows.json').write_text(json.dumps(rows)+'\n')
        (dest/'complete.json').write_text(json.dumps(dict(n=n,feature_bytes=(dest/'features.npy').stat().st_size))+'\n')
        z.close()
        del feat
    (ROOT/'cache_receipt.json').write_text(json.dumps(dict(state='COMPLETE',n=total,seconds=time.monotonic()-start,
        free_bytes_after=shutil.disk_usage(ROOT).free),indent=2)+'\n')
    print('CACHE COMPLETE',total,flush=True)


if __name__=='__main__':
    main()
