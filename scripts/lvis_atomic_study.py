#!/usr/bin/env python3
"""Exact source-stage LVIS1400 ledger and a separately frozen 100-case pilot.

Inference replays input-addressed FP32 O24 features without an encoder. Read-only
Python return-frame taps expose the actual official function locals; no rebuilt
FoRIS implementation is used. Query annotations are opened only by diagnose.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import concurrent.futures
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO.parent / 'cv_data'
PARENT = ASSETS / 'a/lvis_foris1400_score_reuse_20261009'
DEFAULT_OUT = ASSETS / 'a/lvis_atomic1400_20261009'
MEAN_RUNS = [ASSETS / 'a' / name / 'run' for name in
             ('lvis_mean200_20261008', 'lvis_mean600_new_20261008',
              'lvis_mean600_batch3_20261009')]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '2')

FORIS_CHAIN = ['foris.fg', 'foris.bg', 'foris.vote', 'foris.prior',
               'foris.penalty', 'foris.pre', 'foris.crf']
MEAN_CHAIN = ['foris.pre', 'mean.rank', 'mean']
DIAGNOSTIC_CHAIN = ['diag.raw_nn', 'diag.apd_nn']
ARMS = DIAGNOSTIC_CHAIN + FORIS_CHAIN + ['mean.rank', 'mean', 'mean.graph_only']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_index(path):
    seal = json.loads((path / 'sealed.json').read_text())
    assert seal['state'] == 'ALL_PREDICTIONS_SEALED'
    for name, key in [('manifest.json', 'manifest_sha256'), ('config.json', 'config_sha256'),
                      ('inference.jsonl', 'inference_index_sha256')]:
        assert sha(path / name) == seal[key]
    rows = json.loads((path / 'manifest.json').read_text())
    index = {r['episode_id']: r for r in map(json.loads, (path / 'inference.jsonl').read_text().splitlines())}
    assert len(index) == len(rows) == seal['n']
    return rows, index


def prepare(out):
    rows, index = read_index(PARENT / 'run')
    _, mean_index = read_index(PARENT / 'mean1400')
    assert len(rows) == len({r['query_photo_id'] for r in rows}) == 1400
    pilot = []
    for fold in range(10):
        pilot.extend([r['episode_id'] for r in rows if r['fold'] == fold][:10])
    assert len(set(pilot)) == 100
    tasks = []
    for source in MEAN_RUNS:
        source_rows, source_index = read_index(source)
        for row in source_rows:
            item = source_index[row['episode_id']]
            assert row == next(r for r in rows if r['episode_id'] == row['episode_id'])
            profile = json.loads((source / 'raw_cache.json').read_text())['profile_path']
            task = dict(row=row, source=str(source), filename=item['prediction_file'],
                        source_field_sha256=item['fields_sha256'],
                        foris_prediction_sha256=index[row['episode_id']]['prediction_sha256'],
                        mean_prediction_sha256=mean_index[row['episode_id']]['prediction_sha256'],
                        profile=profile, pilot=row['episode_id'] in pilot)
            tasks.append(task)
    assert len(tasks) == 1400
    config = dict(n=1400, pilot_n=100, pilot_selection='first10 per fold in the sealed1400 manifest; no score selection',
                  pilot_ids=pilot, hypothesis_design_ids=[r['episode_id'] for r in rows if r['episode_id'] not in pilot],
                  arms=ARMS, chains=dict(foris=FORIS_CHAIN, mean=MEAN_CHAIN, apd_diagnostic=DIAGNOSTIC_CHAIN),
                  query_GT_in_inference=False, new_encoder_calls=0,
                  source_sha256={str(p): sha(p) for p in [Path(__file__), REPO/'src/ics/foris.py',
                      REPO/'src/ics/methods/mean_control.py', ASSETS/'third_party/foris_official/models/foris.py']},
                  parent_seals={str(p): sha(p/'sealed.json') for p in [PARENT/'run', PARENT/'mean1400', *MEAN_RUNS]},
                  metric='class pooled I/U -> class mean -> fold mean; CLI1024 primary, original secondary',
                  scope='1400 source-step diagnosis, candidate preliminary comparison only100; no new images or full run',
                  exposure='baseline labels/scores already exposed; candidate pilot is exploratory, not untouched confirmation',
                  apd_scope='same nearest-reference classifier before/after actual Part1; not an FoRIS intermediate mask',
                  prefix_scope='each actual continuous prefix has its own source minmax/bilinear/>.5 readout; not raw additive causality')
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text()) == config
        return
    write(out/'config.json', config); write(out/'manifest.json', rows); write(out/'tasks.json', tasks)
    (out/'fields').mkdir(); (out/'predictions').mkdir(); (out/'pilot_features').mkdir()
    write(out/'activity.json', dict(state='PREPARED', n=1400, pilot_n=100))


@contextmanager
def return_taps(functions):
    """Capture chosen locals at return without replacing computations or outputs."""
    codes = {inspect.unwrap(fn).__code__: (name, names) for name, (fn, names) in functions.items()}
    captured = {}
    old = sys.getprofile()
    assert old is None, 'Do not override another profiler'
    def tap(frame, event, arg):
        if event == 'return' and frame.f_code in codes:
            name, names = codes[frame.f_code]
            captured[name] = {key: frame.f_locals[key] for key in names if key in frame.f_locals}
    sys.setprofile(tap)
    try:
        yield captured
    finally:
        sys.setprofile(old)


def initialize_worker(out):
    global NP, TORCH, F, IMAGE, HOST, ADAPTER, ROOT, MEAN, RCG, RUN_FORIS, LOAD_INPUTS, RENDER, CG
    ROOT = Path(out)
    import numpy as np
    import torch
    import torch.nn.functional as f
    from PIL import Image
    from cached_dino import CachedDINO, cache_host
    from raw_feature_cache import RawFeatureCache
    from ics.foris import run_foris
    from ics.official_data import load_inputs
    from ics.methods import mean_control, rcg
    from run_m4_baselines import render
    from scipy.sparse.linalg import cg
    torch.set_num_threads(2); torch.manual_seed(0)
    tasks = json.loads((ROOT/'tasks.json').read_text())
    profiles = {t['profile'] for t in tasks}; assert len(profiles) == 1
    profile = Path(profiles.pop())
    cache = RawFeatureCache(profile.parent.parent, json.loads(profile.read_text()))
    # CachedDINO normally pre-reads then reads again to support live branch filling.
    # Here no live encoder exists: one fully verified read suffices for each input.
    class SingleReadReplay(CachedDINO):
        @torch.inference_mode()
        def raw(self, inputs, branches, provenance=None):
            assert self.encoder is None and inputs.dtype == torch.float32
            assert inputs.ndim == 4 and tuple(inputs.shape[1:]) == (3,1024,1024)
            branches = tuple(dict.fromkeys(('O/24',) + tuple(branches)))
            values = [self.cache.read(x, branches) for x in inputs.numpy()]
            self.cache_reads += len(values)
            self.last_live_maps = None
            return {key: torch.from_numpy(np.stack([v[key] for v in values])) for key in branches}
    adapter = SingleReadReplay(cache)
    host = cache_host(ASSETS, adapter, mask_refiner='bilinear')
    NP,TORCH,F,IMAGE,HOST,ADAPTER,MEAN,RCG,RUN_FORIS,LOAD_INPUTS,RENDER,CG = (
        np,torch,f,Image,host,adapter,mean_control,rcg,run_foris,load_inputs,render,cg)


def read_mask(path, digest, frame, arm, shape):
    assert sha(path) == digest
    with NP.load(path) as saved:
        return NP.unpackbits(saved[frame+'/'+arm], count=int(NP.prod(shape))).reshape(shape).astype(bool)


def infer_one(task):
    began = time.monotonic(); row = task['row']; name = task['filename']
    source = Path(task['source']); source_field = source/'fields'/name
    assert sha(source_field) == task['source_field_sha256']
    with NP.load(source_field) as saved:
        saved_score, saved_mean = saved['score'].copy(), saved['mean'].copy()
    rgb, reference_mask, query = LOAD_INPUTS(row, ASSETS)
    functions = dict(
        stage2=(HOST._part2_stage2_contrastive_score, ['sim_fg_hw','sb','sf','sbn']),
        stage3=(HOST._part3_clustering, ['vote_soft','prior']),
        penalty=(HOST._semantic_disagreement_penalty, ['penalty','uncertainty']),
        delta=(HOST._semantic_cluster_reweight_map, ['labels_t','delta_cluster']),
        apd=(HOST._should_apply_positional_debias, ['s_sem_last']),
        mean=(MEAN.mean_control, ['guide','s','y','w','a','H','z']))
    with TORCH.inference_mode(), return_taps(functions) as taps:
        _, got, transformed, _ = RUN_FORIS(HOST, rgb, reference_mask, query)
        assert NP.array_equal(got['score'].numpy(), saved_score), 'Actual source FoRIS score differs'
        processed = F.normalize(got['deb'][0].float(), dim=1)
        q,r = (processed[i].flatten(1).T.half().contiguous() for i in (1,0))
        cov = F.interpolate(transformed[None,None].float(), (64,64), mode='area')[0,0].numpy()
        mean, info = MEAN.predict(q,r,cov,saved_score)
        assert NP.array_equal(mean, saved_mean), 'Actual source MEAN field differs'
        s2, s3 = got['s2'], got['s3']
        vote, prior = taps['stage3']['vote_soft'], taps['stage3']['prior']
        vote_field = s2 + HOST.candidate_boost*(vote-.5)
        assert TORCH.equal(vote_field+HOST.seed_cluster_boost*(prior-.5), s3)
        penalty = taps['penalty']['penalty'].clamp_max(HOST.semantic_penalty_max)
        labels = taps['delta']['labels_t']
        delta = taps['delta']['delta_cluster'][labels].reshape(64,64)
        assert TORCH.equal(s3-penalty+delta, got['score'])
        fields = dict(fg=taps['stage2']['sim_fg_hw'].numpy(), bg=taps['stage2']['sb'].numpy(),
                      sf=taps['stage2']['sf'].numpy(), sbn=taps['stage2']['sbn'].numpy(),
                      score=saved_score, mean=mean, vote=vote.numpy(), prior=prior.numpy(),
                      penalty=penalty.numpy(), delta=delta.numpy(),
                      mean_guide=taps['mean']['guide'].reshape(64,64),
                      mean_unary=taps['mean']['y'].reshape(64,64).astype(NP.float32),
                      query_clusters=labels.numpy().reshape(64,64))
        graph_z,status = CG(taps['mean']['H'], taps['mean']['a']*taps['mean']['s'],
                          x0=taps['mean']['s'].astype(NP.float64), rtol=1e-7,atol=1e-9,maxiter=300)
        assert status == 0
        fields['graph_only'] = graph_z.reshape(64,64).astype(NP.float32)
        prefixes = {'foris.fg': taps['stage2']['sim_fg_hw'], 'foris.bg': s2,
                    'foris.vote': vote_field, 'foris.prior': s3,
                    'foris.penalty': s3-penalty, 'foris.pre': got['score']}
        masks = {arm: HOST._binarize_response(value,target_hw=(1024,1024)).numpy()
                 for arm,value in prefixes.items()}
        assert NP.array_equal(masks['foris.pre'],got['pre'].numpy())
        fg_ref = got['reference_fg'][0].reshape(-1)
        assert bool(fg_ref.any())
        for stage, fmap in [('diag.raw_nn', F.normalize(got['raw'][0].float(),dim=1)),
                            ('diag.apd_nn', processed)]:
            rq,rr = (fmap[i].flatten(1).T for i in (1,0))
            sim = rq@rr.T
            token_mask = fg_ref[sim.argmax(1)].reshape(64,64)
            masks[stage] = RENDER(token_mask.numpy(),(1024,1024)); del sim
        for arm,key in [('mean.rank','mean_unary'),('mean','mean'),('mean.graph_only','graph_only')]:
            masks[arm] = RCG.mask_from_field(fields[key])
    packed = dict(original_hw=NP.asarray(row['query_size_hw']))
    for arm,mask in masks.items():
        packed['cli/'+arm] = NP.packbits(mask)
        packed['original/'+arm] = NP.packbits(RENDER(mask,tuple(row['query_size_hw'])))
    for frame,shape in [('cli',(1024,1024)),('original',tuple(row['query_size_hw']))]:
        crf = read_mask(PARENT/'run/predictions'/name,task['foris_prediction_sha256'],frame,'foris.crf',shape)
        packed[frame+'/foris.crf'] = NP.packbits(crf)
        expected_mean = read_mask(PARENT/'mean1400/predictions'/name,task['mean_prediction_sha256'],frame,'mean',shape)
        assert NP.array_equal(packed[frame+'/mean'],NP.packbits(expected_mean))
    field_path = ROOT/'fields'/name; pred_path = ROOT/'predictions'/name
    NP.savez_compressed(field_path,**fields); NP.savez_compressed(pred_path,**packed)
    feature_sha = None
    if task['pilot']:
        feature_path = ROOT/'pilot_features'/name
        NP.savez_compressed(feature_path,q=q.numpy(),r=r.numpy(),cov=cov)
        feature_sha = sha(feature_path)
    assert ADAPTER.encoder is None and ADAPTER.encoder_calls == 0
    sem = taps['apd'].get('s_sem_last')
    return dict(episode_id=row['episode_id'],filename=name,pilot=task['pilot'],
                field_sha256=sha(field_path),prediction_sha256=sha(pred_path),feature_sha256=feature_sha,
                source_score_bit_exact=True,source_mean_bit_exact=True,final_mean_masks_bit_exact=True,
                apd_applied=bool(sem is None or sem.item()<.8),semantic_similarity=None if sem is None else sem.item(),
                graph=info,encoder_calls=0,worker_pid=os.getpid(),seconds=time.monotonic()-began)


def infer(out,workers,resume):
    config=json.loads((out/'config.json').read_text())
    assert config['source_sha256'][str(Path(__file__))] == sha(Path(__file__))
    if (out/'sealed.json').exists():
        print('Atomic1400 already sealed; not restarting',flush=True); return
    ledger=out/'inference.jsonl'; done={}
    if ledger.exists():
        assert resume,'Use --resume to preserve completed source traces'
        for item in map(json.loads,ledger.read_text().splitlines()):
            assert sha(out/'predictions'/item['filename'])==item['prediction_sha256']
            assert sha(out/'fields'/item['filename'])==item['field_sha256']
            done[item['episode_id']]=item
    tasks=json.loads((out/'tasks.json').read_text()); started=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
             initializer=initialize_worker,initargs=(str(out),)) as pool, ledger.open('a',buffering=1) as log:
        pending=[pool.submit(infer_one,t) for t in tasks if t['row']['episode_id'] not in done]
        for future in concurrent.futures.as_completed(pending):
            item=future.result();log.write(json.dumps(item)+'\n');done[item['episode_id']]=item
            write(out/'activity.json',dict(state='SOURCE_TRACE',completed=len(done),n=1400,
                controller_pid=os.getpid(),worker_pids=[p.pid for p in pool._processes.values()],
                elapsed_seconds=time.monotonic()-started,workers=workers,encoder_calls=0))
            if len(done)%25==0: print(json.dumps(dict(completed=len(done),n=1400,seconds=time.monotonic()-started)),flush=True)
    assert len(done)==1400
    write(out/'sealed.json',dict(state='ALL_SOURCE_TRACES_SEALED',n=1400,
         config_sha256=sha(out/'config.json'),manifest_sha256=sha(out/'manifest.json'),
         inference_index_sha256=sha(ledger),query_GT_in_inference=False,
         all_source_score_and_mean_fields_bit_exact=True,all_final_mean_masks_bit_exact=True,
         encoder_calls=0,crf_predictions_reused=1400))
    write(out/'activity.json',dict(state='TRACE_COMPLETE',completed=1400,worker_pids=[],encoder_calls=0))


def point(records, arms):
    import numpy as np
    groups=defaultdict(lambda:np.zeros((len(arms),2),dtype=np.int64))
    for row in records:
        groups[(row['fold'],row['class_id'])] += np.asarray([row['iu'][a] for a in arms])
    folds=defaultdict(list)
    for (fold,cls),values in groups.items():
        folds[fold].append(100*values[:,0]/np.maximum(values[:,1],1))
    mean=np.mean([np.mean(x,axis=0) for x in folds.values()],axis=0)
    return dict(zip(arms,mean.tolist()))


def summarize_ledger(records, arms, chains):
    import numpy as np
    points=point(records,arms); transitions={}
    for chain_name,chain in chains.items():
        entries=[]
        for base,arm in zip(chain,chain[1:]):
            pixels=np.sum([r['edits'][arm][base] for r in records],axis=0).tolist()
            delta=[100*(r['iu'][arm][0]/max(r['iu'][arm][1],1)-r['iu'][base][0]/max(r['iu'][base][1],1)) for r in records]
            entries.append(dict(base=base,arm=arm,base_miou=points[base],miou=points[arm],delta_pp=points[arm]-points[base],
                edit_order=['add_TP','add_FP','delete_TP','delete_FP'],pixels=pixels,
                corrected_pixels=pixels[0]+pixels[3],harmed_pixels=pixels[1]+pixels[2],
                cases_up=sum(d>1e-10 for d in delta),cases_down=sum(d< -1e-10 for d in delta),
                cases_equal=sum(abs(d)<=1e-10 for d in delta)))
        transitions[chain_name]=entries
    return dict(n=len(records),miou=points,transitions=transitions)


def diagnose(out):
    import numpy as np
    import torch
    import torch.nn.functional as f
    from PIL import Image
    from ics.official_data import array_hash
    from ics.metrics import counts,gross_edits
    torch.set_num_threads(2)
    seal=json.loads((out/'sealed.json').read_text()); assert seal['n']==1400
    for name,key in [('config.json','config_sha256'),('manifest.json','manifest_sha256'),('inference.jsonl','inference_index_sha256')]:
        assert sha(out/name)==seal[key]
    config=json.loads((out/'config.json').read_text());pilot=set(config['pilot_ids'])
    rows=json.loads((out/'manifest.json').read_text())
    index={r['episode_id']:r for r in map(json.loads,(out/'inference.jsonl').read_text().splitlines())}
    pairs=[(base,arm) for chain in config['chains'].values() for base,arm in zip(chain,chain[1:])]
    pairs += [('foris.pre','mean.graph_only'),('mean.graph_only','mean'),('foris.crf','mean')]
    records=[]
    with (out/'episode_metrics.jsonl').open('w') as ledger:
        for row in rows:
            rec=index[row['episode_id']];path=out/'predictions'/rec['filename']
            assert sha(path)==rec['prediction_sha256']
            with Image.open(row['query_mask_path']) as image: raw=(np.asarray(image.convert('L'))>0).astype(np.uint8)
            assert array_hash(raw)==row['query_mask_hash']
            item=dict(episode_id=row['episode_id'],fold=row['fold'],class_id=row['loader_class_id'],
                      query_photo_id=row['query_photo_id'],pilot=row['episode_id'] in pilot,frames={})
            with np.load(path) as saved:
                for frame,shape in [('cli',(1024,1024)),('original',tuple(row['query_size_hw']))]:
                    truth=f.interpolate(torch.from_numpy(raw)[None,None].float(),shape,mode='nearest')[0,0].numpy()>.5
                    masks={arm:np.unpackbits(saved[frame+'/'+arm],count=int(np.prod(shape))).reshape(shape).astype(bool) for arm in ARMS}
                    iu={arm:counts(mask,truth) for arm,mask in masks.items()};edits=defaultdict(dict)
                    for base,arm in pairs:edits[arm][base]=gross_edits(masks[arm],masks[base],truth)
                    item['frames'][frame]=dict(iu=iu,edits=dict(edits),truth_pixels=int(truth.sum()))
            records.append(item);ledger.write(json.dumps(item)+'\n')
    chains=dict(config['chains'],graph_control=['foris.pre','mean.graph_only','mean'],comparison=['foris.crf','mean'])
    report=dict(n=1400,encoder_calls=0,identity=seal,frames={},scope=config['exposure'])
    for frame in ('cli','original'):
        flat=[dict(r,**r['frames'][frame]) for r in records]
        report['frames'][frame]={name:summarize_ledger(part,ARMS,chains) for name,part in
            [('all1400',flat),('design1300',[r for r in flat if not r['pilot']]),('pilot100',[r for r in flat if r['pilot']])]}
        # Independent original source endpoints already have separately scored I/U.
        parent=json.loads((PARENT/'summary.json').read_text())['frames'][frame]
        for arm,key in [('foris.crf','foris'),('mean','mean')]:
            assert abs(report['frames'][frame]['all1400']['miou'][arm]-parent[key])<1e-10
    write(out/'report.json',report)
    write(out/'activity.json',dict(state='DIAGNOSIS_COMPLETE',completed=1400,pilot_n=100,worker_pids=[]))
    print(json.dumps({f:report['frames'][f]['all1400']['miou'] for f in report['frames']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','infer','diagnose'])
    p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.mode=='prepare':prepare(a.out)
    elif a.mode=='infer':infer(a.out,a.workers,a.resume)
    else:diagnose(a.out)


if __name__=='__main__':main()
