"""A controlled information intervention, NOT an information-loss conclusion.

Keep all 4096 query tokens and all 1024 channels. The same input/encoder pass
provides blocks 12 and 24; no PCA, learned summaries or query-GT token selection.
Store full vectors in FP16 (a numerical cast, NOT lossless original tensors),
with a common legal support bank (128 FG, 128 BG actual tokens). Raw-pixel
patches are read from existing photos during training.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['DEMO4_CACHE'] = '/root/demo8_dependency_pins'
sys.path[:0] = ['/root/demo4_cache/env', '/root/autodl-tmp/demo4']
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from icx.common import TimmDINOv3
from utils.data import build_transform, downsample_mask, load_mask
from train import save


def stored(x):
    a=x.float().half().cpu().numpy()
    if not np.isfinite(a).all(): raise ValueError('FP16 serialization overflow')
    return a


@torch.inference_mode()
def run(args):
    root = Path(args.root); out = root/'information_v1'; out.mkdir(exist_ok=True)
    cache = Path('/root/demo8_dependency_pins')
    meta = json.loads((root/'pixel_data_v1/metadata.json').read_text())['rows']
    status = dict(state='STARTING', pid=os.getpid(), completed=0, total=len(meta),
                  input='same 1024 square RGB for all arms', layers=[12,24],
                  query='4096 unpooled tokens, full 1024 channels',
                  support='fixed 128 FG + 128 BG actual tokens, legal support labels only',
                  serialization='full-channel FP16 numerical cast, no PCA/projection; storage drift measured',
                  inference_gt='query GT excluded; candidate contract reused unchanged',
                  cohort='old development only; fresh800 remains unopened',
                  failure_scope='negative probe cannot prove irreversible loss or missing external evidence')
    save(out/'status.json', status)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    enc = TimmDINOv3().m.to('cuda').eval()
    for p in enc.parameters(): p.requires_grad_(False)
    assert len(enc.blocks) == 24 and enc.num_features == 1024
    transform = build_transform(1024)
    start = time.monotonic(); verified = False
    for i, r in enumerate(meta):
        dest = out/f"f{r['fold']}_e{r['e']:04d}.npz"
        if dest.exists(): status['completed'] += 1; continue
        if shutil.disk_usage(out).free < 4*1024**3:
            raise RuntimeError('Below 4GiB disk reserve; preserve completed feature files, do not delete other jobs')
        images = [transform(Image.open(cache/f"data/COCO2014/{r[k]}").convert('RGB'))[None].cuda()
                  for k in ['query', 'reference']]
        features = []
        for x in images:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = enc.forward_intermediates(x, indices=[11,23], norm=True,
                      output_fmt='NLC', intermediates_only=True)
            assert all(v.shape == (1,4096,1024) for v in z)
            if not verified:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    zz = enc.forward_features(x)[:, enc.num_prefix_tokens:]
                drift = float((zz.float()-z[1].float()).abs().max())
                checks=[]
                for v in z:
                    half=v.half().float();a=F.normalize(v.float(),dim=-1);b=F.normalize(half,dim=-1)
                    checks.append({'source_dtype':str(v.dtype),'storage_max_abs':float((v.float()-half).abs().max()),
                                   'storage_max_normalized_vector_l2':float((a-b).norm(dim=-1).max()),
                                   'storage_sample_similarity_max_abs':float((a[:,:256]@a[:,:256].transpose(-1,-2)-b[:,:256]@b[:,:256].transpose(-1,-2)).abs().max())})
                save(out/'interface.json', {'last_layer_vs_original_max_abs':drift,
                     'storage_checks':checks,'storage':'FP16 rounding, not lossless original FP32 outputs',
                     'layers_chosen_before_results':[12,24]})
                if drift != 0: raise RuntimeError(f'Last-layer interface differs: {drift}')
                verified = True
            features.append(z)
        # Reference labels are the supplied task input, never query labels.
        support = np.asarray(Image.open(cache/f"data/COCO2014/annotations/{r['reference'][:-4]}.png")) == r['c']+1
        sm = load_mask(torch.from_numpy(support.copy()),1024,'cpu')
        sf = downsample_mask(sm[None],64,64).flatten().numpy()
        rng = np.random.default_rng(int(hashlib.sha256(f"{r['fold']}/{r['e']}/2042".encode()).hexdigest()[:8],16))
        bank = np.concatenate([rng.choice(np.where(sf==v)[0],128,replace=(sf==v).sum()<128)
                               for v in [True,False]])
        assert sf[bank[:128]].all() and not sf[bank[128:]].any()
        q = np.stack([stored(z[0]) for z in features[0]])
        s = np.stack([stored(z[0,bank]) for z in features[1]])
        # Query labels are loaded ONLY after input tensors/bank are finalized.
        gt = torch.from_numpy((np.asarray(Image.open(cache/f"data/COCO2014/annotations/{r['query'][:-4]}.png")) == r['c']+1).copy())
        y = F.interpolate(gt[None,None].float(), (1024,1024),mode='nearest')
        y = F.avg_pool2d(y,16,16).flatten().numpy()
        tmp = dest.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, query=q, support=s, support_indices=bank,
                            foreground_target=y.astype(np.float32))
        tmp.replace(dest)
        del images, features, q, s
        status.update(state='EXTRACTING',completed=status['completed']+1,
                      elapsed_seconds=time.monotonic()-start)
        save(out/'status.json',status)
        if status['completed']%10 == 0: print(json.dumps(status),flush=True)
    save(out/'metadata.json', {'state':'COMPLETED','rows':meta,'contract':status})
    status.update(state='COMPLETED'); save(out/'status.json',status)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); args=ap.parse_args()
    try: run(args)
    except Exception:
        out=Path(args.root)/'information_v1'; out.mkdir(parents=True,exist_ok=True)
        save(out/'error.json', {'state':'ERROR','traceback':traceback.format_exc()}); raise
