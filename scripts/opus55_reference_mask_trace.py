#!/usr/bin/env python3
"""Fixed RGB pair, real task-mask switches, exact cached FoRIS source traces.

Preparation may inspect annotations to establish two real tasks on the same
images. Inference receives no query labels, has no encoder, and seals every
field before diagnosis. Shift controls preserve native mask area exactly.
"""
import argparse
import builtins
from collections import Counter, defaultdict
import concurrent.futures
from contextlib import contextmanager
import hashlib
import io
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO.parent / 'cv_data'
OUT = ASSETS / 'a/reference_mask_task_trace_20261010'
PROFILE = ASSETS / 'a/paco_mean200_20261008/raw_cache/0a555915a7972480b74fcce11c876a79c4e776c93db8573d31f8b0e01c1c6158/profile.json'
sys.path[:0] = [str(REPO/'scripts'), str(REPO/'src'), str(ASSETS/'third_party/foris_official')]
for env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(env, '2')

CHAIN = ['raw.fg', 'raw.bg', 'apd.fg', 'apd.bg', 'vote', 'prior', 'penalty', 'score']
VARIANTS = ['A', 'B', 'shiftA', 'shiftB']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_mask(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert('L')) > 0


def image_masks(row, role, annotations, suim_index):
    import numpy as np
    from PIL import Image
    from ics.official_data import array_hash, decoded_rgb
    from utils.data import polygons_to_bitmask
    import pycocotools.mask as coco_mask
    rgb = decoded_rgb(ASSETS/row[role+'_path'], row.get(role+'_crop'))
    assert array_hash(np.asarray(rgb)) == row[role+'_rgb_hash']
    dataset = row['dataset']; bindings = {}
    if dataset == 'coco':
        photo = row[role+'_photo_id']
        ann = ASSETS/'demo4_cache/data/COCO2014/annotations'/Path(photo).with_suffix('.png')
        labels = np.asarray(Image.open(ann))
        masks = {int(c)-1: labels == c for c in np.unique(labels) if 1 <= int(c) <= 80}
        bindings[str(ann)] = sha(ann)
    elif dataset == 'paco_part':
        image_id = int(Path(row[role+'_path']).stem)
        object_id = row[role+'_object_id']; crop = row[role+'_crop']
        full = Image.open(ASSETS/row[role+'_path']); masks = {}
        for ann in annotations['img2anno'][image_id]:
            if ann['obj_ann_id'] != object_id:
                continue
            seg = ann['segmentation']
            if isinstance(seg, list):
                mask = polygons_to_bitmask(seg, full.height, full.width)
            elif isinstance(seg, dict):
                mask = coco_mask.decode(seg).astype(bool)
            else:
                mask = np.asarray(seg).astype(bool)
            if crop is not None:
                x0,y0,x1,y1 = crop; mask = mask[y0:y1,x0:x1]
            key = int(ann['category_id'])
            masks[key] = masks.get(key, np.zeros(mask.shape, bool)) | mask
        bindings['paco_val_annotations'] = annotations['_sha256']
    else:
        original = suim_index.get(row[role+'_rgb_hash'])
        if original is None:
            return None, {'unmatched_RGB': row[role+'_rgb_hash']}
        masks = {}
        for category in ['FV','HD','PF','RI','RO','SR','WR']:
            files = sorted((ASSETS/'datasets/ics/SUIM/masks'/category).glob(original.stem+'.*'))
            if not files:
                continue
            assert len(files) == 1
            with Image.open(files[0]) as image:
                mask = np.asarray(image.convert('L')) >= 128
            # The legacy pack caps RGB and labels together at 1600 pixels.
            if mask.shape != (rgb.height,rgb.width):
                mask = np.asarray(Image.fromarray(mask).resize(rgb.size, Image.Resampling.NEAREST)).astype(bool)
            if mask.any():
                masks[category] = mask
            bindings[str(files[0])] = sha(files[0])
        bindings['matched_original_RGB'] = str(original)
        bindings['matched_original_file_sha256'] = sha(original)
    original_key = row['global_class_id']
    if original_key not in masks:
        return None, {'original_class_missing': original_key}
    expected = read_mask(row[role+'_mask_path'])
    assert np.array_equal(masks[original_key], expected), (dataset,role,row['episode_id'],'original mask mismatch')
    assert all(m.shape == (rgb.height,rgb.width) for m in masks.values())
    return masks, bindings


def prepare(out, per_dataset):
    import numpy as np
    import torch
    from PIL import Image
    from ics.official_data import array_hash, decoded_rgb, legacy_cap
    from utils.data import build_transform
    from raw_feature_cache import RawFeatureCache, tensor_hash
    torch.set_num_threads(2)
    assert not (out/'config.json').exists(), 'Preserve the frozen study; use a fresh output for another cohort'
    sources = {
        'coco': [ASSETS/'a/coco_role_competition200_20261010'],
        'paco_part': [ASSETS/'a'/name/'run' for name in ['paco_mean200_20261008','paco_mean200_new_20261008','paco_mean200_batch3_20261008']],
        'suim': [ASSETS/'a/zoom_crossdataset40_20261009']}
    annotation_path = ASSETS/'datasets/ics/PACO-Part/paco/paco_part_val.pkl'
    annotations = pickle.loads(annotation_path.read_bytes()); annotations['_sha256'] = sha(annotation_path)
    suim_index = {}; cap,_ = legacy_cap(str(ASSETS))
    for file in sorted((ASSETS/'datasets/ics/SUIM/images').glob('*')):
        if not file.is_file(): continue
        with Image.open(file) as image:
            rgb = image.convert('RGB')
            rgb,_,_ = cap(rgb, Image.new('L',rgb.size), 1600)
            digest = array_hash(np.asarray(rgb))
        if digest in suim_index:
            assert sha(file) == sha(suim_index[digest]), 'Ambiguous SUIM image binding'
        suim_index[digest] = file
    cache = RawFeatureCache(PROFILE.parent.parent, json.loads(PROFILE.read_text()))
    transform = build_transform(1024)
    pairs = []; truth = {}; inventory = {}; source_hashes = {}; excluded = Counter()
    for dataset,folders in sources.items():
        selected = 0; seen_query = set(); examined = 0
        for folder in folders:
            manifest = folder/'manifest.json'; source_hashes[str(manifest)] = sha(manifest)
            if (folder/'sealed.json').exists(): source_hashes[str(folder/'sealed.json')] = sha(folder/'sealed.json')
            rows = json.loads(manifest.read_text())
            index = {r['episode_id']:r for r in map(json.loads,(folder/'inference.jsonl').read_text().splitlines())}
            for row in rows:
                if row['dataset'] != dataset or selected >= per_dataset: continue
                examined += 1
                if row['query_rgb_hash'] in seen_query:
                    excluded[dataset+'/duplicate_query'] += 1; continue
                rmask, rbinding = image_masks(row,'reference',annotations,suim_index)
                qmask, qbinding = image_masks(row,'query',annotations,suim_index)
                if rmask is None or qmask is None:
                    excluded[dataset+'/source_unmatched'] += 1; continue
                key_a = row['global_class_id']; choices = []
                for key_b in set(rmask)&set(qmask)-{key_a}:
                    if min(int(rmask[key_a].sum()),int(rmask[key_b].sum())) < 16: continue
                    if min(int((qmask[key_a]&~qmask[key_b]).sum()),int((qmask[key_b]&~qmask[key_a]).sum())) < 16: continue
                    distance = abs(float(np.log(rmask[key_b].sum()/rmask[key_a].sum())))
                    choices.append((distance,str(key_b),key_b))
                if not choices:
                    excluded[dataset+'/no_two_exclusive_tasks'] += 1; continue
                key_b = min(choices)[2]
                # Actual RGB input keys, never a cache-hit replacement cohort.
                raw_binding = {}
                for role in ('reference','query'):
                    rgb = decoded_rgb(ASSETS/row[role+'_path'],row.get(role+'_crop'))
                    transformed = transform(rgb).numpy(); key = cache.key(transformed)
                    entry = cache.folder/key/'entry.json'; info = json.loads(entry.read_text())
                    assert info['input_tensor_hash'] == tensor_hash(transformed) and 'O/24' in info['features']
                    raw_binding[role] = dict(key=key,entry=str(entry),entry_sha256=sha(entry),input_tensor_hash=info['input_tensor_hash'])
                pair_id = f'{dataset}_{selected:03d}'
                frozen = {k:v for k,v in row.items() if not k.startswith('query_mask') and k not in ['source_files','query_annotation_path','query_ignore_mask_path']}
                variants = {}
                for name,mask in [('A',rmask[key_a]),('B',rmask[key_b])]:
                    path = out/'reference_masks'/f'{pair_id}_{name}.png'; path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(mask.astype('uint8')*255).save(path)
                    shift = np.roll(mask,(mask.shape[0]//2,mask.shape[1]//2),(0,1))
                    shift_path = path.with_name(f'{pair_id}_shift{name}.png'); Image.fromarray(shift.astype('uint8')*255).save(shift_path)
                    for label,binary,file in [(name,mask,path),('shift'+name,shift,shift_path)]:
                        variants[label] = dict(path=str(file),file_sha256=sha(file),array_hash=array_hash(binary.astype('uint8')),native_area=int(binary.sum()))
                    assert variants[name]['native_area'] == variants['shift'+name]['native_area']
                item = index[row['episode_id']]; filename = item.get('filename',item.get('prediction_file'))
                baseline = folder/'fields'/filename
                baseline_binding = None
                if baseline.exists():
                    with np.load(baseline) as stored:
                        if 'score' in stored:
                            baseline_binding = dict(path=str(baseline),sha256=sha(baseline))
                pairs.append(dict(id=pair_id,dataset=dataset,row=frozen,classes=[key_a,key_b],variants=variants,
                                  raw_binding=raw_binding,baseline=baseline_binding,reference_annotations=rbinding))
                truth[pair_id] = dict(classes=[key_a,key_b],source=qbinding,source_episode_id=row['episode_id'],masks={})
                for name,key in [('A',key_a),('B',key_b)]:
                    path = out/'query_labels'/f'{pair_id}_{name}.png'; path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(qmask[key].astype('uint8')*255).save(path)
                    truth[pair_id]['masks'][name] = dict(path=str(path),sha256=sha(path),array_hash=array_hash(qmask[key].astype('uint8')))
                selected += 1; seen_query.add(row['query_rgb_hash'])
        inventory[dataset] = dict(selected=selected,examined=examined)
        print(json.dumps(dict(dataset=dataset,**inventory[dataset])),flush=True)
    assert all(v['selected'] > 0 for v in inventory.values())
    source_files = [Path(__file__), REPO/'src/ics/foris.py',REPO/'scripts/cached_dino.py',REPO/'scripts/raw_feature_cache.py',
                    REPO/'scripts/lvis_atomic_study.py', ASSETS/'third_party/foris_official/models/foris.py',
                    ASSETS/'third_party/foris_official/utils/data.py',ASSETS/'third_party/foris_official/utils/clustering.py', PROFILE,
                    ASSETS/'native_assets/positional_basis.pt']
    config = dict(kind='Fixed R RGB and Q RGB, switch two real class or sibling-part masks',n_pairs=len(pairs),inventory=inventory,
        per_dataset=per_dataset,variants=VARIANTS,chain=CHAIN,new_DINO_encodings=0,query_GT_in_inference=False,
        selection='Existing manifest order; first eligible distinct query RGB per dataset; B has closest reference log-area to A among shared real labels; no outcome selection',
        controls='Cyclic half-height/half-width native-mask translation, exactly equal native area; artificial negative controls have no alternative-task GT',
        primary_metric='Average of A-vs-B exclusive-region weighted AUC for field A and B-vs-A AUC for field B; identical scores across tasks imply 0.5',
        exposure='Existing exposed development photographs; alternate GT used only for preparation presence/exclusivity and post-seal diagnosis; not confirmation or official benchmark',
        scope='Exact source continuous prefixes; raw comparison is same source classifier before APD; no causal claim from stage association',
        source_sha256={str(p):sha(p) for p in source_files},parent_sha256=source_hashes,excluded=dict(excluded),profile=str(PROFILE))
    snapshot=out/'frozen/trace_source_v1.py';snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    write(out/'pairs.json',pairs);write(out/'query_truth.json',truth);write(out/'config.json',config)
    write(out/'prepared.json',dict(config_sha256=sha(out/'config.json'),pairs_sha256=sha(out/'pairs.json'),query_truth_sha256=sha(out/'query_truth.json')))
    write(out/'activity.json',dict(state='PREPARED',pairs=len(pairs),encoder_calls=0))


@contextmanager
def deny_labels(out, attempts):
    """Block all query labels in this study without loading their contents."""
    from PIL import Image
    saved = builtins.open, io.open, Image.open
    def wrap(original):
        def opened(path,*args,**kwargs):
            if isinstance(path,(str,bytes,os.PathLike)):
                resolved = Path(os.fsdecode(path)).resolve()
                if resolved == out/'query_truth.json' or out/'query_labels' in resolved.parents:
                    attempts.append(str(path));raise RuntimeError('Query label access denied')
            return original(path,*args,**kwargs)
        return opened
    builtins.open,io.open,Image.open = map(wrap,saved)
    try: yield
    finally: builtins.open,io.open,Image.open = saved


def initialize_worker(out):
    global NP,TORCH,F,HOST,ADAPTER,ROOT,RUN,ARRAY_HASH,DOWNSAMPLE
    import numpy as np
    import torch
    import torch.nn.functional as f
    from cached_dino import CachedDINO,cache_host
    from raw_feature_cache import RawFeatureCache
    from ics.foris import run_foris
    from ics.official_data import array_hash
    from utils.data import downsample_mask
    torch.set_num_threads(2);torch.manual_seed(0)
    cache = RawFeatureCache(PROFILE.parent.parent,json.loads(PROFILE.read_text()))
    class PairReplay(CachedDINO):
        def __init__(self):
            super().__init__(cache);self.memo={}
        @torch.inference_mode()
        def get_intermediate_layers(self,inputs,n=1,reshape=True):
            assert n == 1 and reshape and self.encoder is None
            values=[]
            for x in inputs.numpy():
                key=self.cache.key(x)
                if key not in self.memo:
                    self.memo[key]=self.cache.read(x,('O/24',))['O/24'];self.cache_reads+=1
                values.append(self.memo[key])
            return [torch.from_numpy(np.stack(values)).reshape(len(values),64,64,1024).permute(0,3,1,2).contiguous()]
    adapter=PairReplay();host=cache_host(ASSETS,adapter,mask_refiner='crf')
    NP,TORCH,F,HOST,ADAPTER,ROOT,RUN,ARRAY_HASH,DOWNSAMPLE = np,torch,f,host,adapter,Path(out),run_foris,array_hash,downsample_mask


def infer_pair(pair):
    from lvis_atomic_study import return_taps
    from ics.official_data import load_inputs
    start=time.monotonic();ADAPTER.memo.clear(); fields={};meta={};attempts=[]
    for role,binding in pair['raw_binding'].items():
        assert sha(binding['entry']) == binding['entry_sha256']
    with deny_labels(ROOT,attempts),TORCH.inference_mode():
        for variant in VARIANTS:
            binding=pair['variants'][variant];assert sha(binding['path'])==binding['file_sha256']
            row=pair['row']|dict(reference_mask_path=binding['path'],reference_mask_hash=binding['array_hash'])
            rgb,mask,query=load_inputs(row,ASSETS)
            functions=dict(stage2=(HOST._part2_stage2_contrastive_score,['sim_fg_hw','sb','sf','sbn','mu_fg','mu_bg','fg_protos','dot','n']),
                gate=(HOST._part2_stage1_feature_gating,['gate','mask_ds']),
                stage3=(HOST._part3_clustering,['vote_soft','prior','cand']),
                seed=(HOST._build_seed_cluster_prior,['labels','seed_cluster','area_w','cross_sim','combined']),
                penalty=(HOST._semantic_disagreement_penalty,['penalty','uncertainty']),
                delta=(HOST._semantic_cluster_reweight_map,['labels_t','delta_cluster']),
                apd=(HOST._should_apply_positional_debias,['s_sem_last']))
            with return_taps(functions) as taps:
                final,got,transformed,_=RUN(HOST,rgb,mask,query)
            stage2=taps['stage2'];s2,s3=got['s2'],got['s3']
            vote=taps['stage3']['vote_soft'];prior=taps['stage3']['prior']
            vote_field=s2+HOST.candidate_boost*(vote-.5)
            assert TORCH.equal(vote_field+HOST.seed_cluster_boost*(prior-.5),s3)
            penalty=taps['penalty']['penalty'].clamp_max(HOST.semantic_penalty_max)
            delta=taps['delta']['delta_cluster'][taps['delta']['labels_t']].reshape(64,64)
            assert TORCH.equal(s3-penalty+delta,got['score'])
            actual_fg=got['reference_fg'][0]
            normalized=F.normalize(got['raw'],dim=2)
            raw_score,_,_,_=HOST._part2_stage2_contrastive_score(normalized[:,:1],normalized[:,1],transformed[None,None],1,64,64,normalized[:,1])
            # Independent raw-prefix foreground LSE, with the actual source prototype builder.
            raw_stats=HOST._reference_contrastive_prototypes(normalized[:,:1],actual_fg[None],1,mu_fg_per_reference=True)
            raw_mu,_,raw_protos=raw_stats;t=max(1e-4,HOST.cluster_logsumexp_temp)
            raw_fg=t*TORCH.logsumexp(TORCH.einsum('chw,kc->khw',normalized[0,1],raw_protos)/t,dim=0)
            raw_mean=TORCH.einsum('chw,c->hw',normalized[0,1],raw_mu)
            ref_map=got['deb'][0,0];self_fg=t*TORCH.logsumexp(TORCH.einsum('chw,kc->khw',ref_map,stage2['fg_protos'])/t,dim=0)
            self_bg=TORCH.einsum('chw,c->hw',ref_map,stage2['mu_bg'])
            for name,value in dict(**{'raw.fg':raw_fg,'raw.bg':raw_score,'raw.mean':raw_mean,'apd.fg':stage2['sim_fg_hw'],'apd.bg':s2},
                    vote=vote_field,prior=s3,penalty=s3-penalty,score=got['score'],vote_only=vote,prior_only=prior,
                    ref_self_fg=self_fg,ref_self_bg=self_fg-HOST.dino_bg_weight*self_bg,mu_fg=stage2['mu_fg'],mu_bg=stage2['mu_bg'],
                    fg_protos=stage2['fg_protos'],nearest_mask=taps['gate']['mask_ds'][0],actual_mask=actual_fg,
                    seed_clusters=taps['seed']['labels'].reshape(64,64),semantic_clusters=taps['delta']['labels_t'].reshape(64,64),
                    penalty_only=penalty,delta_only=delta,gate_query=taps['gate']['gate'][0,1]).items():
                fields[variant+'/'+name]=value.cpu().numpy()
            fields[variant+'/reference_coverage']=F.avg_pool2d(transformed[None,None].float(),16)[0,0].numpy()
            fields[variant+'/pre']=NP.packbits(got['pre'].cpu().numpy());fields[variant+'/crf']=NP.packbits(final.cpu().numpy())
            sem=taps['apd'].get('s_sem_last')
            meta[variant]=dict(apd_applied=bool(sem is None or sem.item()<.8),s_sem=None if sem is None else sem.item(),
                n_fg_tokens=int(actual_fg.sum()),n_nearest_tokens=int(taps['gate']['mask_ds'].sum()),n_fg_prototypes=len(stage2['fg_protos']),
                bg_fg_dot=stage2['dot'].item(),orthogonal_bg_norm=stage2['n'].item(),seed_cluster=int(taps['seed']['seed_cluster']),
                score_min=float(got['score'].min()),score_max=float(got['score'].max()),threshold=float((got['score'].max()+got['score'].min())/2))
            if variant=='A' and pair['baseline']:
                assert sha(pair['baseline']['path'])==pair['baseline']['sha256']
                with NP.load(pair['baseline']['path']) as saved:
                    assert NP.array_equal(saved['score'],fields['A/score']), 'Original task source score must replay bit exactly'
    assert not attempts and ADAPTER.encoder is None and ADAPTER.encoder_calls==0
    path=ROOT/'traces'/f"{pair['id']}.npz";path.parent.mkdir(parents=True,exist_ok=True);NP.savez_compressed(path,**fields)
    return dict(id=pair['id'],dataset=pair['dataset'],file=str(path),sha256=sha(path),variants=meta,encoder_calls=0,verified_raw_reads=ADAPTER.cache_reads,
                baseline_score_bit_exact=pair['baseline'] is not None,query_label_access_attempts=attempts,seconds=time.monotonic()-start,worker_pid=os.getpid())


def infer(out,workers):
    prepared=json.loads((out/'prepared.json').read_text());config=json.loads((out/'config.json').read_text())
    assert sha(out/'config.json')==prepared['config_sha256'] and sha(out/'pairs.json')==prepared['pairs_sha256']
    if (out/'sealed.json').exists():
        seal=json.loads((out/'sealed.json').read_text())
        assert sha(out/'inference.jsonl')==seal['inference_sha256']
        return
    for file,digest in config['source_sha256'].items():assert sha(file)==digest, 'Frozen source changed: '+file
    pairs=json.loads((out/'pairs.json').read_text());done={};ledger=out/'inference.jsonl'
    if ledger.exists():
        for record in map(json.loads,ledger.read_text().splitlines()):
            assert sha(record['file'])==record['sha256'];done[record['id']]=record
    start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker,initargs=(str(out),)) as pool,ledger.open('a',buffering=1) as handle:
        futures=[pool.submit(infer_pair,p) for p in pairs if p['id'] not in done]
        for future in concurrent.futures.as_completed(futures):
            item=future.result();handle.write(json.dumps(item)+'\n');done[item['id']]=item
            write(out/'activity.json',dict(state='TRACING',completed_pairs=len(done),n_pairs=len(pairs),controller_pid=os.getpid(),
                  worker_pids=[p.pid for p in pool._processes.values()],elapsed_seconds=time.monotonic()-start,new_encoder_calls=0))
            print(json.dumps(dict(id=item['id'],completed=len(done),n=len(pairs),seconds=item['seconds'])),flush=True)
    assert len(done)==len(pairs)
    write(out/'sealed.json',dict(state='ALL_TRACES_SEALED',n_pairs=len(pairs),variants=VARIANTS,config_sha256=sha(out/'config.json'),
        pairs_sha256=sha(out/'pairs.json'),inference_sha256=sha(ledger),new_encoder_calls=0,query_GT_access_attempts=0))
    write(out/'activity.json',dict(state='TRACES_SEALED',completed_pairs=len(done),n_pairs=len(pairs),new_encoder_calls=0))


def weighted_auc(score,positive,negative):
    import numpy as np
    score=np.asarray(score).ravel();p=np.asarray(positive,dtype=float).ravel();n=np.asarray(negative,dtype=float).ravel()
    if p.sum()==0 or n.sum()==0:return None
    order=np.argsort(score,kind='stable');s=score[order];p=p[order];n=n[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1];pp=np.add.reduceat(p,starts);nn=np.add.reduceat(n,starts)
    return float(np.sum(pp*(np.cumsum(nn)-nn+.5*nn))/(p.sum()*n.sum()))


def bootstrap(values):
    import numpy as np
    values=np.asarray([v for v in values if v is not None],dtype=float);rng=np.random.default_rng(0)
    if not len(values): return dict(mean=None,ci95=None,n=0)
    means=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n=len(values))


def diagnose(out):
    import numpy as np
    import torch
    import torch.nn.functional as f
    from ics.official_data import array_hash
    seal=json.loads((out/'sealed.json').read_text());prepared=json.loads((out/'prepared.json').read_text())
    for name,key in [('config.json','config_sha256'),('pairs.json','pairs_sha256'),('inference.jsonl','inference_sha256')]:assert sha(out/name)==seal[key]
    assert sha(out/'query_truth.json')==prepared['query_truth_sha256']
    pairs=json.loads((out/'pairs.json').read_text());truth=json.loads((out/'query_truth.json').read_text())
    index={r['id']:r for r in map(json.loads,(out/'inference.jsonl').read_text().splitlines())};records=[]
    def canonical(mask):return f.interpolate(torch.from_numpy(mask.astype('float32'))[None,None],size=(1024,1024),mode='nearest')[0,0].numpy().astype(bool)
    for pair in pairs:
        rec=index[pair['id']];assert sha(rec['file'])==rec['sha256'];targets=[]
        for v in ['A','B']:
            entry=truth[pair['id']]['masks'][v];assert sha(entry['path'])==entry['sha256']
            m=read_mask(entry['path']);assert array_hash(m.astype('uint8'))==entry['array_hash'];targets.append(canonical(m))
        a,b=targets;ex=[a&~b,b&~a];mass=[e.reshape(64,16,64,16).sum((1,3)) for e in ex]
        scores={};ious={};controls={}
        with np.load(rec['file']) as z:
            for name in CHAIN+['raw.mean','vote_only','prior_only']:
                auc=[weighted_auc(z[v+'/'+name],mass[i],mass[1-i]) for i,v in enumerate(['A','B'])]
                scores[name]=dict(switch_auc=float(np.mean(auc)),per_task_auc=auc,
                    field_rms_change=float(np.sqrt(np.mean((z['A/'+name]-z['B/'+name])**2))))
                shifted=[weighted_auc(z[v+'/'+name],mass[i],mass[1-i]) for i,v in enumerate(['shiftA','shiftB'])]
                controls[name]=dict(shift_auc=float(np.mean(shifted)),true_minus_shift=float(np.mean(auc)-np.mean(shifted)))
            for name in ['pre','crf']:
                masks=[np.unpackbits(z[v+'/'+name],count=1024*1024).reshape(1024,1024).astype(bool) for v in ['A','B']]
                auc=[weighted_auc(masks[i],ex[i],ex[1-i]) for i in (0,1)]
                scores[name]=dict(switch_auc=float(np.mean(auc)),per_task_auc=auc,changed_pixel_fraction=float(np.mean(masks[0]!=masks[1])))
                ious[name]=[dict(I=int((masks[i]&targets[i]).sum()),U=int((masks[i]|targets[i]).sum())) for i in (0,1)]
            rc=[z[v+'/reference_coverage'] for v in ['A','B']];r_ex=[np.maximum(rc[0]-rc[1],0),np.maximum(rc[1]-rc[0],0)]
            reference_self={}
            for name in ['ref_self_fg','ref_self_bg']:
                values=[weighted_auc(z[v+'/'+name],r_ex[i],r_ex[1-i]) for i,v in enumerate(['A','B'])]
                reference_self[name]=None if any(v is None for v in values) else float(np.mean(values))
            stats=dict(nearest_mask_equal=bool(np.array_equal(z['A/nearest_mask'],z['B/nearest_mask'])),
                stage2_mask_equal=bool(np.array_equal(z['A/actual_mask'],z['B/actual_mask'])),
                downsampling_disagreement=[float(np.mean(z[v+'/nearest_mask']!=z[v+'/actual_mask'])) for v in ['A','B']],
                foreground_prototype_cosine=float(np.dot(z['A/mu_fg'],z['B/mu_fg'])),
                query_seed_clusters_equal=bool(np.array_equal(z['A/seed_clusters'],z['B/seed_clusters'])),
                query_semantic_clusters_equal=bool(np.array_equal(z['A/semantic_clusters'],z['B/semantic_clusters'])),
                apd_gate_switch=rec['variants']['A']['apd_applied']!=rec['variants']['B']['apd_applied'])
        records.append(dict(id=pair['id'],dataset=pair['dataset'],classes=pair['classes'],query_rgb_hash=pair['row']['query_rgb_hash'],
            scores=scores,controls=controls,reference_self=reference_self,stats=stats,full_mask_IU=ious,variants=rec['variants']))
    report={}
    for dataset in sorted({r['dataset'] for r in records}):
        subset=[r for r in records if r['dataset']==dataset];stages=CHAIN+['pre','crf']
        edges=[('raw.fg','raw.bg'),('raw.fg','apd.fg'),('raw.bg','apd.bg'),('apd.fg','apd.bg'),
               ('apd.bg','vote'),('vote','prior'),('prior','penalty'),('penalty','score'),('score','pre'),('pre','crf')]
        report[dataset]=dict(n=len(subset),stage_switch_auc={s:bootstrap([r['scores'][s]['switch_auc'] for r in subset]) for s in stages},
            stage_paired_deltas={s+' -> '+t:bootstrap([r['scores'][t]['switch_auc']-r['scores'][s]['switch_auc'] for r in subset]) for s,t in edges},
            true_vs_area_matched_shift={s:bootstrap([r['controls'][s]['true_minus_shift'] for r in subset]) for s in CHAIN},
            reference_self={s:bootstrap([r['reference_self'][s] for r in subset]) for s in ['ref_self_fg','ref_self_bg']},
            collision_counts={s:sum(r['stats'][s] for r in subset) for s in ['nearest_mask_equal','stage2_mask_equal','query_seed_clusters_equal','query_semantic_clusters_equal','apd_gate_switch']},
            mean_prototype_cosine=float(np.mean([r['stats']['foreground_prototype_cosine'] for r in subset])))
    write(out/'diagnosis.json',dict(state='POST_SEAL_DIAGNOSIS',datasets=report,records=records,new_encodings=0,
        diagnosis_source_sha256=sha(Path(__file__)),inference_source_sha256=sha(out/'frozen/trace_source_v1.py'),
        caveats=['Exposed development images, 20 or fewer unique query photos per domain, not official benchmark scores',
                 'Raw stage contrasts and prefix association do not establish causality; use paired targeted intervention',
                 'AUC compares the two exclusive tasks only; background precision and full IoU retained separately',
                 'Mask reduction is bilinear threshold with fallback in stage2/voting, nearest in gating/APD; no false assumption that both are nearest']))
    lines=['# 固定 RGB、真实参考 Mask 切换：任务信息追踪','',
        f"已完成 {len(pairs)} 对同图任务、{len(pairs)*4} 个完整 FoRIS 轨迹；新增 DINO 编码 0。所有场先封存，随后才读取 query GT。",'',
        '每对固定参考 RGB 与查询 RGB，A/B 是两种真实类别或同一对象的兄弟部件；shiftA/B 保持各自原始面积，仅平移参考 mask。',
        'Switch AUC 是两个正确任务方向的平均排序 AUC；同一场无论 mask 怎么换都等于 0.5。它只回答两任务可区分性，不能替代完整 mIoU。','',
        '| 数据集 | n | raw FG | raw FG/BG | APD FG | APD FG/BG | 投票前缀 | 聚类前缀 | 语义修正 | 二值化 | CRF |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset,r in report.items():
        stages=['raw.fg','raw.bg','apd.fg','apd.bg','vote','prior','score','pre','crf']
        lines.append('| '+dataset+' | '+str(r['n'])+' | '+' | '.join(f"{r['stage_switch_auc'][s]['mean']:.4f}" for s in stages)+' |')
    lines+=['','所有逐例场、真实 label 身份、两种 mask token 化、FG/BG 原型、query cluster、面积对照和配对区间见 `diagnosis.json`。',
            '当前为机制探索。首次数值下降不等于首次不可恢复的信息损失；后续应验证对应算子的定点干预，并在独立样本确认。','',
            '资产入口：`config.json` 固定源码和 profile；`pairs.json` 只含推理合法信息；`query_truth.json` 独立保存评分标注；`traces/` 保存全部轨迹；`sealed.json` 绑定封存 hash。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({d:{s:round(r['stage_switch_auc'][s]['mean'],4) for s in CHAIN+['pre','crf']} for d,r in report.items()}),flush=True)
    write(out/'activity.json',dict(state='DIAGNOSED',n_pairs=len(pairs),new_encodings=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','infer','diagnose','run']);parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--per-dataset',type=int,default=20);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();out=args.out.resolve()
    if args.phase=='prepare':prepare(out,args.per_dataset)
    if args.phase in ['infer','run']:infer(out,args.workers)
    if args.phase in ['diagnose','run']:diagnose(out)


if __name__=='__main__':main()
