#!/usr/bin/env python3
"""Complete 1024 FoRIS + historical MEAN/RCG on frozen binary-mask packs.

Encode on MPS, run the unchanged FoRIS stages/CRF and graph controls on CPU.
Inference opens reference labels only. Scoring is a separate sealed phase.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '2')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def rows(a):
    found = []
    for pack in a.packs:
        source = a.assets / 'cache/astra_granularity_20261008' / pack / 'rows.json'
        if not source.exists():
            raise FileNotFoundError('Original frozen rows are required: ' + str(source))
        for row in json.loads(source.read_text()):
            r = dict(row, pack=pack)
            if 'key' not in r:
                r['key'] = f"{r['fold']}_{r['e']}_{r['c']}"
            if a.key is None or r['key'] == a.key:
                found.append(r)
    if a.limit is not None:
        found = found[:a.limit]
    if not found:
        raise ValueError('No matching frozen episodes')
    if len({(r['pack'],r['key']) for r in found}) != len(found):
        raise ValueError('Duplicate frozen episode keys')
    return found


def paths(assets, row):
    root = assets / 'episodes/claude_packs' / row['pack']
    return {name: root / folder / row[role] for name,folder,role in
            [('support','data','support'),('query','data','query'),
             ('support_mask','ann','support'),('query_mask','ann','query')]}


def render(mask, size):
    import torch
    import torch.nn.functional as F
    m = torch.as_tensor(mask).float()
    return F.interpolate(m[None,None], size=size, mode='bilinear', align_corners=False)[0,0].numpy()>.5


def infer(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from ics.experiment import sha
    from ics.foris import build_host, run_foris
    from ics.methods import mean_control, raw_reference_origin, rcg

    manifest = rows(a)
    if a.out.exists():
        raise FileExistsError('Use a fresh run directory: ' + str(a.out))
    for row in manifest:
        for role,p in paths(a.assets,row).items():
            if role != 'query_mask' and not p.is_file():
                raise FileNotFoundError('Input has not arrived: ' + str(p))
    if a.encoder_device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; choose CPU explicitly')
    a.out.mkdir(parents=True); (a.out/'predictions').mkdir(); (a.out/'features').mkdir()
    write(a.out/'manifest.json',manifest)
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    model = a.assets/'demo4_cache/models/dinov3-vitl16-timm'
    basis = a.assets/'native_assets/positional_basis.pt'
    foris = a.assets/'third_party/foris'
    source_paths = [Path(__file__),REPO/'src/ics/data.py',REPO/'src/ics/foris.py',
        REPO/'src/ics/native_basis.py',REPO/'src/ics/m4_crf.py',
        REPO/'src/ics/methods/rcg.py',REPO/'src/ics/methods/mean_control.py',
        REPO/'src/ics/methods/raw_reference_origin.py']
    source_paths += sorted(foris.rglob('*.py'))
    config = dict(branch=git('branch','--show-current'),commit=git('rev-parse','HEAD'),
        source_sha256={str(p):sha(p) for p in source_paths},assets=str(a.assets.resolve()),
        weights_sha256=sha(model/'model.safetensors'),basis_sha256=sha(basis),
        encoder_device=a.encoder_device,encoder_dtype='float32',paired_encoder_batch=2,
        postprocess_device='cpu',threads=a.threads,resolution=1024,refiner=a.refiner,
        rcg_config=rcg.CONFIG,mean='historical MEAN_a0.25_l16 exact function',
        graph_input='FoRIS Part1 output, channel-normalized then FP16 rounded; controls normalize again',
        render='source-native 1024 binary mask -> bilinear original size -> >0.5',
        query_labels_in_inference=False,exposure='frozen reused development packs',
        cuda_parity='unverified; MPS FP32 encoder + CPU pipeline compared within this run')
    if a.refiner == 'crf':
        from ics.m4_crf import install
        _, config['crf_backend'] = install(a.assets/'third_party/crf_source',a.assets/'runtime/macos/crf')
        crf_files = sorted((a.assets/'third_party/crf_source/src/CRF').glob('*.py'))
        config['source_sha256'].update({str(p):sha(p) for p in crf_files})
    write(a.out/'config.json',config)
    host = build_host({'projection_basis':str(basis)}, 'cpu', str(foris),
                      weights=str(model),mask_refiner=a.refiner)
    torch.set_num_threads(a.threads)
    host.encoder.to(a.encoder_device)
    encoder_times = []

    def extract(imgs):
        started = time.monotonic()
        b,t = imgs.shape[:2]
        x = imgs.reshape(b*t,*imgs.shape[2:]).to(a.encoder_device)
        maps = host.encoder.get_intermediate_layers(x,n=1,reshape=True)[0]
        if a.encoder_device == 'mps': torch.mps.synchronize()
        result = maps.cpu().reshape(b,t,*maps.shape[1:])
        encoder_times.append(time.monotonic()-started)
        return result

    # This adapter changes only the encoding device. FoRIS's actual stage
    # methods, tensor layout, precision and paired extraction are preserved.
    host._extract_features = extract
    hashes,inputs,timings = {},{},{}
    began = time.monotonic()
    with torch.inference_mode():
        for n,row in enumerate(manifest,1):
            ident = row['pack']+'__'+row['key']; source = paths(a.assets,row)
            start = time.monotonic(); encoder_times.clear()
            with Image.open(source['support']) as im: support = im.convert('RGB')
            with Image.open(source['query']) as im: query = im.convert('RGB')
            with Image.open(source['support_mask']) as im:
                gold = torch.from_numpy((np.asarray(im.convert('L'))>0).copy())
            inputs[ident] = {k:dict(path=str(p),sha256=sha(p)) for k,p in source.items() if k!='query_mask'}
            original_hw = (query.height,query.width)
            native,got,mask,_ = run_foris(host,support,gold,query)
            native_seconds = time.monotonic()-start
            cov = F.interpolate(mask[None,None].float(),(64,64),mode='area')[0,0].numpy()
            raw = F.normalize(got['raw'][0].float(),dim=1)
            q0,r0 = (raw[i].flatten(1).T.contiguous() for i in (1,0))
            processed = F.normalize(got['deb'][0].float(),dim=1)
            q,r = (processed[i].flatten(1).T.half().contiguous() for i in (1,0))
            score = got['score'].float().numpy()
            masks = {'foris.'+a.refiner: native.numpy(), 'foris.pre':got['pre'].bool().numpy()}
            start = time.monotonic()
            token_masks,_,raw_info = raw_reference_origin.predict(q0,r0,cov)
            masks.update({k:render(m,(1024,1024)) for k,m in token_masks.items()})
            raw_seconds = time.monotonic()-start
            start = time.monotonic(); field,rg = rcg.predict(q,r,cov,score); rcg_seconds = time.monotonic()-start
            start = time.monotonic(); mf,mi = mean_control.predict(q,r,cov,score); mean_seconds = time.monotonic()-start
            masks['rcg'],masks['mean'] = rcg.mask_from_field(field),rcg.mask_from_field(mf)
            dest = a.out/'predictions'/(ident+'.npz')
            np.savez_compressed(dest,original_hw=np.array(original_hw),
                **{k:np.packbits(render(m,original_hw)) for k,m in masks.items()})
            feature = a.out/'features'/(ident+'.npz')
            np.savez_compressed(feature,q=q.numpy(),r=r.numpy(),q_raw=q0.numpy(),r_raw=r0.numpy(),
                cov=cov,score=score,s2=got['s2'].float().numpy(),s3=got['s3'].float().numpy(),
                rcg=field,mean=mf)
            hashes[ident] = dict(predictions=sha(dest),features=sha(feature))
            timings[ident] = dict(foris_seconds=native_seconds,encoder_seconds=sum(encoder_times),
                rcg_after_encoder_seconds=rcg_seconds,mean_after_encoder_seconds=mean_seconds,
                raw_after_encoder_seconds=raw_seconds,rcg_solver=rg,mean_solver=mi,raw=raw_info)
            write(a.out/'progress.json',dict(n=n,total=len(manifest),last=ident,seconds=time.monotonic()-began))
            print(json.dumps(dict(n=n,total=len(manifest),key=ident,foris_seconds=native_seconds,
                                 encoder_seconds=sum(encoder_times))),flush=True)
    write(a.out/'timings.json',timings)
    write(a.out/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',
        manifest_sha256=sha(a.out/'manifest.json'),config_sha256=sha(a.out/'config.json'),
        hashes=hashes,inputs=inputs,query_labels_opened=False,seconds=time.monotonic()-began))


def score(a):
    import numpy as np
    from PIL import Image
    from ics.experiment import sha
    seal = json.loads((a.out/'sealed.json').read_text())
    if seal['state']!='ALL_PREDICTIONS_SEALED': raise ValueError('Inference incomplete')
    for key,name in [('manifest_sha256','manifest.json'),('config_sha256','config.json')]:
        if sha(a.out/name)!=seal[key]: raise ValueError('Changed scoring input: '+name)
    config = json.loads((a.out/'config.json').read_text())
    assets = Path(config['assets']); totals,details,edits = {},[],{}
    for row in json.loads((a.out/'manifest.json').read_text()):
        ident = row['pack']+'__'+row['key']; path = a.out/'predictions'/(ident+'.npz')
        if sha(path)!=seal['hashes'][ident]['predictions']: raise ValueError('Changed prediction')
        gt_path = paths(assets,row)['query_mask']
        with Image.open(gt_path) as im: truth = np.asarray(im.convert('L'))>0
        with np.load(path,allow_pickle=False) as z:
            shape = tuple(z['original_hw'])
            if truth.shape!=shape: raise ValueError('Original annotation/image shape mismatch')
            masks = {k:np.unpackbits(z[k],count=truth.size).reshape(shape).astype(bool)
                     for k in z.files if k!='original_hw'}
        base = masks['foris.'+config['refiner']]; record = dict(key=ident,class_key=row['pack']+':'+str(row['c']),iu={})
        for arm,mask in masks.items():
            i,u = int((mask&truth).sum()),int((mask|truth).sum())
            record['iu'][arm] = [i,u]
            acc = totals.setdefault(arm,{}).setdefault(record['class_key'],[0,0]); acc[0]+=i;acc[1]+=u
            add,delete = mask&~base,base&~mask
            e=[int((add&truth).sum()),int((add&~truth).sum()),int((delete&truth).sum()),int((delete&~truth).sum())]
            record.setdefault('edits_vs_foris',{})[arm]=e
            edits.setdefault(arm,[0,0,0,0]);edits[arm]=[x+y for x,y in zip(edits[arm],e)]
        record['truth_sha256']=sha(gt_path); details.append(record)
    report = dict(n=len(details),classes=len(next(iter(totals.values()))),
        miou={arm:float(100*np.mean([i/max(u,1) for i,u in values.values()])) for arm,values in totals.items()},
        edits_vs_foris=edits,edit_order=['add_TP','add_FP','delete_TP','delete_FP'],
        metric='original-resolution per-class pooled I/U, then macro mean',
        exposure='reused development; limited runs are implementation checks, not dataset estimates',
        intervals='not computed; original photograph identities must be bound before photo-group bootstrap',
        sealed_sha256=sha(a.out/'sealed.json'))
    write(a.out/'report.json',report);write(a.out/'episode_metrics.json',details)
    print(json.dumps(report),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['infer','score'])
    p.add_argument('--assets',type=Path,default=REPO.parent/'cv_data')
    p.add_argument('--packs',nargs='+',default=['paco_part_f0'])
    p.add_argument('--key');p.add_argument('--limit',type=int)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--encoder-device',choices=['mps','cpu'],default='mps')
    p.add_argument('--refiner',choices=['crf','bilinear'],default='crf')
    p.add_argument('--threads',type=int,default=2)
    a = p.parse_args()
    if a.limit is not None and a.limit<1: p.error('--limit must be positive')
    (infer if a.mode=='infer' else score)(a)


if __name__=='__main__':
    main()
