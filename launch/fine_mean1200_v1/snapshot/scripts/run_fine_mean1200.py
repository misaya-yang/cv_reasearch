#!/usr/bin/env python3
"""One fixed fine.mean16.control on exposed1200; cache the existing fine kernel.

prepare: recover the ORIGINAL components.mean_control continuous field from
retained q/r, require exact original MEAN mask parity, seal all1200 fields.
infer: four source-identical FP32 shifted grids, exact fine16 field/mask replay,
same readout on MEAN; retain normalized FP32[16384,25] kernels and one index.
score: open GT only after every mask/kernel is sealed; keep fine64 as primary.

The original source exports MEAN masks but does not retain its continuous field.
No replacement formula, refit/grid, extra reference, CRF arm, or asset deletion.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ics.experiment import render, sha, summarize, unpack
from run_frozen_subtoken1200 import ARMS, FINE16, FINE64, manifest, parameters, verify_code, write

MEAN_FINE = "fine.mean16.control"
ALL_ARMS = (*ARMS, MEAN_FINE)


def base_metadata(base):
    config = json.loads((base / "config.json").read_text())
    seal = json.loads((base / "sealed.json").read_text())
    if seal["state"] != "ALL_PREDICTIONS_SEALED" or seal["n"] != 1200:
        raise ValueError("Require the original full1200 six-arm seal")
    for filename, key in (("manifest.json", "manifest_sha256"), ("config.json", "config_sha256")):
        if sha(base / filename) != seal[key]:
            raise ValueError("Changed original1200 metadata")
    verify_code(config)
    return manifest(base / "manifest.json"), config, seal


def prepare_one(job):
    import torch
    torch.set_num_threads(1)
    row, base, prepared, source, expected, base_seal = job
    sys.path.insert(0, source)
    import components
    key = row["key"]
    pp = Path(row["recheck_run"]) / "predictions" / (row.get("source_key", key) + ".npz")
    fp = Path(row["recheck_run"]) / "fields" / (row.get("source_key", key) + ".npz")
    receipt = dict(feature_sha256=sha(row["feature_export"]), packet_sha256=sha(row["packet_export"]),
                   provider_predictions_sha256=sha(pp), provider_fields_sha256=sha(fp),
                   base_prediction_sha256=sha(Path(base) / "predictions" / (key + ".npz")),
                   base_fields_sha256=sha(Path(base) / "fields" / (key + ".npz")))
    if receipt["provider_predictions_sha256"] != expected["predictions"] or receipt["provider_fields_sha256"] != expected["fields"]:
        raise ValueError("Changed provider outputs: " + key)
    for value, digest in ((receipt["feature_sha256"], base_seal["inputs"][key]["feature_sha256"]),
                          (receipt["packet_sha256"], base_seal["inputs"][key]["packet_sha256"]),
                          (receipt["base_prediction_sha256"], base_seal["predictions"][key]),
                          (receipt["base_fields_sha256"], base_seal["fields"][key])):
        if value != digest:
            raise ValueError("Original1200 input/output changed: " + key)
    cached = torch.load(row["feature_export"], map_location="cpu", weights_only=True)
    if any(cached[k].shape != (4096, 1024) or cached[k].dtype != torch.float16 or not bool(torch.isfinite(cached[k]).all()) for k in ("q", "r")):
        raise ValueError("Require original finite FP16 q/r headers")
    with np.load(row["packet_export"], allow_pickle=False) as z:
        cov, score = z["cov"].copy(), z["score"].copy()  # No query truth.
    with np.load(fp, allow_pickle=False) as z:
        rcg = z["rcg"].copy()
    with np.load(pp, allow_pickle=False) as z:
        original_mean = z["MEAN_CONTROL"].copy()
    with np.load(Path(base) / "predictions" / (key + ".npz"), allow_pickle=False) as z:
        if not np.array_equal(z["mean.control"], original_mean):
            raise ValueError("Original1200/provider MEAN masks differ")
    if any(v.shape != (64, 64) or v.dtype != np.float32 or not np.isfinite(v).all() for v in (cov, score, rcg)):
        raise ValueError("Require original aligned finite FP32 source grids")
    start = time.monotonic()
    # Exact hash-matched ORIGINAL source function; not a new mean_graph recipe.
    mean, solver = components.mean_control(cached["q"].float().numpy(), cached["r"].float().numpy(), cov, score, components.rcg)
    mismatch = int(np.count_nonzero(render(mean) != unpack(original_mean)))
    if mean.shape != (64, 64) or mean.dtype != np.float32 or not np.isfinite(mean).all() or mismatch:
        raise RuntimeError("Original continuous MEAN/render parity failed: " + key + " pixels=" + str(mismatch))
    destination = Path(prepared) / "fields" / (key + ".npz")
    np.savez_compressed(destination, rcg=rcg, **{"mean.control": mean})
    receipt.update(debiased=bool(cached["debiased"]), mean_render_mismatched_pixels=mismatch,
                   solver=solver, cpu_seconds=time.monotonic() - start)
    return key, sha(destination), receipt


def prepare(a):
    rows, base_config, base_seal = base_metadata(a.base)
    frozen = parameters(a.parameter_report)
    if frozen != base_config["parameters"]:
        raise ValueError("Frozen fold parameters changed")
    if a.prepared.exists():
        raise FileExistsError("Use a fresh CPU prepared directory")
    providers, jobs = {}, []
    for row in rows:
        run = Path(row["recheck_run"])
        if str(run) not in providers:
            seal = json.loads((run / "sealed.json").read_text())
            if seal["state"] != "ALL_PREDICTIONS_SEALED" or sha(run / "manifest.json") != seal["manifest_sha256"]:
                raise ValueError("Provider manifest/seal changed")
            data = json.loads((run / "manifest.json").read_text())
            providers[str(run)] = dict(seal=seal, rows={r["key"]: r for r in data}, seal_sha256=sha(run / "sealed.json"))
            for name in ("components.py", "rcg_readout.py", "hypothesis_source_contrast.py"):
                if sha(a.source / name) != seal["astra_sources"][name]:
                    raise ValueError("Original MEAN source hash differs: " + name)
        provider = providers[str(run)]; key = row.get("source_key", row["key"]); old = provider["rows"][key]
        if int(old["c"]) != row["c"] or int(old["fold"]) != row["fold"] or any(Path(old[k]).name != Path(row[k]).name for k in ("support", "query")):
            raise ValueError("Provider class/photo/fold identity differs")
        row_seal = {name: {row["key"]: base_seal[name][row["key"]]} for name in ("inputs", "predictions", "fields")}
        jobs.append((row, str(a.base), str(a.prepared), str(a.source),
                     dict(fields=provider["seal"]["fields"][key], predictions=provider["seal"]["predictions"][key]), row_seal))
    a.prepared.mkdir(parents=True); (a.prepared / "fields").mkdir()
    write(a.prepared / "manifest.json", rows)
    files = [Path(__file__).resolve(), REPO / "scripts/run_frozen_subtoken1200.py", REPO / "src/ics/experiment.py"]
    files += [a.source / name for name in ("components.py", "rcg_readout.py", "hypothesis_source_contrast.py")]
    config = dict(n=1200, stage="CPU_ORIGINAL_MEAN_FIELD_PARITY", parameters=frozen,
                  base=str(a.base.resolve()), base_seal_sha256=sha(a.base / "sealed.json"),
                  source_code_sha256={str(p): sha(p) for p in files}, workers=a.workers,
                  source="original components.mean_control: alpha0.25,lambda16, exact stored MEAN mask parity",
                  continuous_mean_was_retained=False, query_truth_opened=False)
    write(a.prepared / "config.json", config)
    start = time.monotonic(); results = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(prepare_one, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 25 == 0:
                state = dict(state="CPU_MEAN_FIELD_PREPARATION", completed=len(results), n=1200, seconds=time.monotonic() - start)
                write(a.prepared / "state.json", state); print(json.dumps(state), flush=True)
    seal = dict(state="ALL1200_ORIGINAL_MEAN_FIELDS_PARITY_SEALED", n=1200,
                fields={k: h for k, h, _ in results}, inputs={k: r for k, _, r in results},
                manifest_sha256=sha(a.prepared / "manifest.json"), config_sha256=sha(a.prepared / "config.json"),
                query_truth_opened=False, seconds=time.monotonic() - start)
    write(a.prepared / "sealed.json", seal); write(a.prepared / "state.json", dict(state=seal["state"], n=1200, completed=1200))
    print("ORIGINAL_MEAN1200_CPU_PARITY_SEALED", flush=True)


def infer(a):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    rows, base_config, base_seal = base_metadata(a.base)
    prep = json.loads((a.prepared / "sealed.json").read_text()); prep_config = json.loads((a.prepared / "config.json").read_text())
    if prep["state"] != "ALL1200_ORIGINAL_MEAN_FIELDS_PARITY_SEALED" or prep["n"] != 1200:
        raise ValueError("Require all1200 original MEAN field/render parities before GPU")
    for filename, field in (("manifest.json", "manifest_sha256"), ("config.json", "config_sha256")):
        if sha(a.prepared / filename) != prep[field]:
            raise ValueError("Changed prepared MEAN metadata")
    verify_code(prep_config)
    if prep["manifest_sha256"] != base_seal["manifest_sha256"] or prep_config["parameters"] != base_config["parameters"]:
        raise ValueError("Prepared/base identity or frozen policy changed")
    ancestor = a.out.parent.resolve()
    while not ancestor.exists(): ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free < (3 << 30):
        raise OSError("Require3GiB free for retained~2GiB kernels plus fields/headroom")
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ("fields","predictions","kernels"): (a.out / name).mkdir()
    write(a.out / "manifest.json", rows)
    sys.path.insert(0,str(a.host_root));sys.path.insert(0,str(a.host_root/'scripts'))
    import extent_experiment as producer
    man=json.loads(a.host_manifest.read_text())
    if sha(a.host_manifest)!=base_config["host_manifest_sha256"] or sha(producer.__file__)!=base_config["host_builder_sha256"]:
        raise ValueError("Original fine1200 host/config changed")
    config=dict(n=1200,primary=FINE64,new_control=MEAN_FINE,arms=list(ALL_ARMS),parameters=base_config["parameters"],
        base=str(a.base.resolve()),base_seal_sha256=sha(a.base/'sealed.json'),prepared=str(a.prepared.resolve()),prepared_seal_sha256=sha(a.prepared/'sealed.json'),
        source_code_sha256={str(Path(__file__).resolve()):sha(Path(__file__)),str(REPO/'scripts/run_frozen_subtoken1200.py'):sha(REPO/'scripts/run_frozen_subtoken1200.py'),str(REPO/'src/ics/experiment.py'):sha(REPO/'src/ics/experiment.py')},
        exposure="existing exposed DEV1200, no confirmation/SOTA",query_truth_opened=False,extra_encoder_forwards=4800,
        renderer="unchanged CUDA bilinear128->1024 align_corners=False >0.5",division_order="original(sum unnormalized_w*z)/sumw for exact replay/newcontrol",
        kernel_cache="normalized FP32[16384,25], one shared neighbor map; future normalized dot products may have FP32 rounding differences",
        readers=a.readers,writers=a.writers,prefetch=a.prefetch,chunk=a.chunk)
    write(a.out/'config.json',config)
    dev=torch.device('cuda');fi=torch.arange(128,device=dev);base=((fi*8+4-8).float()/16).round().long();off=torch.arange(-2,3,device=dev)
    ti=base[:,None]+off[None];distance=((ti*16+8)-(fi*8+4)[:,None]).float()/16;ok=(ti>=0)&(ti<64);ti=ti.clamp(0,63)
    ny=ti[:,None,:,None].expand(128,128,5,5);nx=ti[None,:,None,:].expand(128,128,5,5);nb=(ny*64+nx).reshape(16384,25)
    d2=(distance[:,None,:,None].square()+distance[None,:,None,:].square()).reshape(16384,25);valid=(ok[:,None,:,None]&ok[None,:,None,:]).reshape(16384,25)
    np.save(a.out/'neighbors.npy',nb.cpu().numpy().astype(np.uint16));np.save(a.out/'valid_neighbors.npy',valid.cpu().numpy())
    def load(row):
        key=row['key'];receipt=dict(prep['inputs'][key]);fp=a.prepared/'fields'/(key+'.npz');bp=a.base/'predictions'/(key+'.npz');bf=a.base/'fields'/(key+'.npz')
        for path,digest in ((Path(row['feature_export']),receipt['feature_sha256']),(fp,prep['fields'][key]),(bp,base_seal['predictions'][key]),(bf,base_seal['fields'][key])):
            if sha(path)!=digest:raise ValueError('Changed checked input '+key)
        cached=torch.load(row['feature_export'],map_location='cpu',weights_only=True)
        with np.load(fp,allow_pickle=False) as z:coarse={FINE16:z['rcg'].copy(),MEAN_FINE:z['mean.control'].copy()}
        with np.load(bp,allow_pickle=False) as z:masks={arm:z[arm].copy() for arm in ARMS}
        with np.load(bf,allow_pickle=False) as z:expected=z[FINE16].copy()
        image_path=Path(man['data_root'])/row['query']
        with Image.open(image_path) as im:rgb=np.asarray(im.convert('RGB')).copy()
        receipt.update(query_image_sha256=sha(image_path),prepared_field_sha256=prep['fields'][key])
        if receipt['query_image_sha256']!=base_seal['inputs'][key]['query_image_sha256']:raise ValueError('Original query RGB changed')
        return cached['q'],bool(cached['debiased']),rgb,coarse,masks,expected,receipt
    def save(row,fields,masks,kernel,receipt):
        key=row['key'];fp=a.out/'fields'/(key+'.npz');pp=a.out/'predictions'/(key+'.npz');kp=a.out/'kernels'/(key+'.npy')
        np.savez_compressed(fp,**fields);np.savez_compressed(pp,**masks);np.save(kp,kernel)
        return key,sha(fp),sha(pp),sha(kp),receipt
    start=time.monotonic();results=[];jobs=[];gpu_seconds=0.
    with torch.inference_mode(),ThreadPoolExecutor(a.readers) as readers,ThreadPoolExecutor(a.writers) as writers:
        host=producer.build_host(SimpleNamespace(fixture=None,foris_root=None,demo4_root=a.demo4_root),man,'cuda')
        pending={i:readers.submit(load,rows[i]) for i in range(min(a.prefetch,1200))}
        for n,row in enumerate(rows):
            cached_q,debiased,rgb,coarse,masks,expected,receipt=pending.pop(n).result()
            later=n+a.prefetch
            if later<1200:pending[later]=readers.submit(load,rows[later])
            target=host._transform(Image.fromarray(rgb)).to(dev);q=F.normalize(cached_q.to(dev).float(),dim=1);pad=F.pad(target[None],(4,4,4,4),mode='reflect')[0]
            torch.cuda.synchronize();tick=time.monotonic();fine=torch.zeros(128,128,1024,device=dev,dtype=torch.float32)
            for ay,ax in itertools.product((0,1),repeat=2):
                sy,sx=(-4,4)[ay],(-4,4)[ax];shifted=pad[:,4+sy:4+sy+1024,4+sx:4+sx+1024]
                f=F.normalize(host._extract_features(shifted[None,None]),p=2,dim=2)
                if debiased:f=host._debias_features(f)
                fine[ay::2,ax::2]=F.normalize(f[0,0],dim=0).permute(1,2,0)
            fine=fine.reshape(16384,1024);params=config['parameters'][str(row['fold'])];scalar={name:torch.from_numpy(v).to(dev).float().flatten() for name,v in coarse.items()}
            pieces={name:[] for name in scalar};normalized=[];cache_replay=[]
            for cell in range(0,16384,a.chunk):
                stop=min(cell+a.chunk,16384);ids=nb[cell:stop];cos=torch.einsum('pc,pkc->pk',fine[cell:stop],q[ids])
                w=torch.exp(-d2[cell:stop]/(2*params['sigma']**2))*torch.exp((cos-1)/params['tau'])*valid[cell:stop];denominator=w.sum(1).clamp_min(1e-12)
                normalized.append(w/denominator[:,None]);cache_replay.append((normalized[-1]*scalar[FINE16][ids]).sum(1))
                for name,v in scalar.items():pieces[name].append((w*v[ids]).sum(1)/denominator)
            fields={name:torch.cat(v).reshape(128,128) for name,v in pieces.items()}
            replay=fields[FINE16].cpu().numpy();nfield=int(np.count_nonzero(replay!=expected))
            replay_mask=(F.interpolate(fields[FINE16][None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]>.5).cpu().numpy();nmask=int(np.count_nonzero(replay_mask!=unpack(masks[FINE16])))
            receipt.update(fine16_field_mismatches=nfield,fine16_mask_mismatches=nmask,fine16_field_maxdiff=float(np.abs(replay-expected).max()))
            if nfield or nmask:
                write(a.out/'producer_failure.json',dict(key=row['key'],receipt=receipt));raise RuntimeError('Exact original fine16 producer replay failed')
            field=fields[MEAN_FINE];masks[MEAN_FINE]=np.packbits((F.interpolate(field[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]>.5).cpu().numpy())
            kernel=torch.cat(normalized).cpu().numpy();cached_replay=torch.cat(cache_replay).reshape(128,128).cpu().numpy()
            receipt.update(normalized_cache_fine16_field_maxdiff=float(np.abs(cached_replay-expected).max()),kernel_row_sum_maxdiff=float(np.abs(kernel.sum(1)-1).max()))
            output={name:v.cpu().numpy() for name,v in fields.items()};torch.cuda.synchronize();gpu_seconds+=time.monotonic()-tick
            if not np.isfinite(kernel).all() or not all(np.isfinite(v).all() for v in output.values()):raise RuntimeError('Nonfinite fine fields/kernel')
            jobs.append(writers.submit(save,row,output,masks,kernel,receipt))
            if len(jobs)>=2*a.writers:results.append(jobs.pop(0).result())
            if n==0 or (n+1)%10==0:
                state=dict(state='GPU_FIXED_FINE_MEAN_INFERENCE',completed=n+1,n=1200,seconds=time.monotonic()-start,gpu_seconds=gpu_seconds,query_truth_opened=False)
                write(a.out/'state.json',state);print(json.dumps(state),flush=True)
            del cached_q,rgb,target,q,pad,fine,f,scalar,pieces,normalized,cache_replay,fields,field,kernel,output,cos,w
        results.extend(job.result() for job in jobs)
    seal=dict(state='ALL_PREDICTIONS_AND_KERNELS_SEALED',n=1200,primary=FINE64,new_control=MEAN_FINE,
        manifest_sha256=sha(a.out/'manifest.json'),config_sha256=sha(a.out/'config.json'),fields={k:f for k,f,_,_,_ in results},predictions={k:p for k,_,p,_,_ in results},kernels={k:h for k,_,_,h,_ in results},inputs={k:r for k,_,_,_,r in results},
        neighbors_sha256=sha(a.out/'neighbors.npy'),valid_neighbors_sha256=sha(a.out/'valid_neighbors.npy'),seconds=time.monotonic()-start,gpu_seconds=gpu_seconds,query_truth_opened=False,
        kernel_bytes=sum((a.out/'kernels'/(k+'.npy')).stat().st_size for k,_,_,_,_ in results),cuda_peak_bytes=torch.cuda.max_memory_allocated())
    write(a.out/'sealed.json',seal);write(a.out/'state.json',dict(state=seal['state'],n=1200,completed=1200));print('FINE_MEAN1200_MASKS_KERNELS_SEALED',flush=True)


def score(a):
    seal=json.loads((a.out/'sealed.json').read_text());config=json.loads((a.out/'config.json').read_text());rows=manifest(a.out/'manifest.json')
    if seal['state']!='ALL_PREDICTIONS_AND_KERNELS_SEALED' or seal['n']!=1200:raise ValueError('All1200 masks/kernels must seal before GT')
    for file,key in (('manifest.json','manifest_sha256'),('config.json','config_sha256'),('neighbors.npy','neighbors_sha256'),('valid_neighbors.npy','valid_neighbors_sha256')):
        if sha(a.out/file)!=seal[key]:raise ValueError('Changed sealed metadata')
    verify_code(config)
    prior_path=Path(config['base'])/'episodes.jsonl';prior={r['key']:r for r in [json.loads(line) for line in prior_path.read_text().splitlines()]}
    if set(prior)!={r['key'] for r in rows}:raise ValueError('Original1200 I/U cohort differs')
    for row in rows:
        key=row['key']
        for folder,extension in (('predictions','.npz'),('fields','.npz'),('kernels','.npy')):
            if sha(a.out/folder/(key+extension))!=seal[folder][key]:raise ValueError('Changed sealed output '+key)
        if sha(row['packet_export'])!=seal['inputs'][key]['packet_sha256']:raise ValueError('Changed packet')
    start=time.monotonic();arrays={arm:[] for arm in ALL_ARMS};episodes=[]
    for row in rows:
        key=row['key']
        with np.load(row['packet_export'],allow_pickle=False) as z:truth=unpack(z['truth'])
        with np.load(a.out/'predictions'/(key+'.npz'),allow_pickle=False) as z:masks={arm:unpack(z[arm]) for arm in ALL_ARMS}
        record=dict(row,iu={})
        for arm,mask in masks.items():
            iu=[int((mask&truth).sum()),int((mask|truth).sum())]
            if arm in ARMS and iu!=prior[key]['iu'][arm]:raise ValueError('Original1200 I/U changed '+key+' '+arm)
            arrays[arm].append(iu);record['iu'][arm]=iu
        episodes.append(record)
    arrays={arm:np.asarray(v,np.int64) for arm,v in arrays.items()};report,draws=summarize(rows,arrays,{})
    for name in ('corrections_vs_native','corrections_by_class','corrections_by_batch'):report.pop(name,None)
    report.update(config=config,primary=FINE64,new_strong_simple_control=MEAN_FINE,parameter_updates=False,source_seal_sha256=sha(a.out/'sealed.json'),
        prior_six_arm_IU_parity=1200,raw_origin_edits='not exported; missing, not substituted',INSID3='original logic only DEV241; full1200 absent',
        runtime=dict(gpu_stage_seconds=seal['seconds'],gpu_compute_seconds=seal['gpu_seconds'],cpu_score_seconds=time.monotonic()-start,cold_continuous_field_producer='separate CPU reconstruction, not included in GPU readout timing'),
        kernel_cache=dict(episodes=1200,bytes=seal['kernel_bytes'],dtype='float32',shape=[16384,25],neighbors='shared neighbors.npy'))
    write(a.out/'report.json',report);np.save(a.out/'bootstrap_photo_draws.npy',draws);np.savez_compressed(a.out/'counts.npz',**{'iu:'+arm:v for arm,v in arrays.items()})
    (a.out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in episodes));write(a.out/'score_state.json',dict(state='CPU_SCORE_COMPLETE',n=1200,primary=FINE64,new_control=MEAN_FINE));print(json.dumps(report['scores']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','infer','score'))
    p.add_argument('--base',type=Path,default=Path('outputs/frozen_subtoken1200_v1'));p.add_argument('--prepared',type=Path,default=Path('outputs/fine_mean1200_prepared_v1'));p.add_argument('--out',type=Path,default=Path('outputs/fine_mean1200_v1'))
    p.add_argument('--source',type=Path,default=Path('external/astra_emd/external_mean_delete600_modules'));p.add_argument('--parameter-report',type=Path,default=Path('outputs/claude_subtoken_fresh600/report.json'))
    p.add_argument('--host-manifest',type=Path,default=Path('outputs/claude_official/batch0.json'));p.add_argument('--host-root',type=Path,default=Path('/root/autodl-tmp/demo9_extent'));p.add_argument('--demo4-root',default='/root/autodl-tmp/demo4')
    p.add_argument('--workers',type=int,default=6);p.add_argument('--readers',type=int,default=6);p.add_argument('--writers',type=int,default=2);p.add_argument('--prefetch',type=int,default=12);p.add_argument('--chunk',type=int,default=512)
    a=p.parse_args()
    if any(v<1 for v in (a.workers,a.readers,a.writers,a.prefetch,a.chunk)):p.error('Execution sizes must be positive')
    if a.stage=='prepare' and a.prepared.exists():p.error('Fresh prepare output required')
    if a.stage=='infer' and a.out.exists():p.error('Fresh infer output required')
    try:{'prepare':prepare,'infer':infer,'score':score}[a.stage](a)
    except Exception as error:
        out=a.prepared if a.stage=='prepare' else a.out
        if out.exists():write(out/'state.json',dict(state='FAILED',stage=a.stage,error=repr(error)))
        raise


if __name__=='__main__':main()
