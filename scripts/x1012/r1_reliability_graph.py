"""R1: cache-only causal comparison of explanation as veto or unary reliability.

Inference has one reference, no query annotation, no encoder, no FoRIS output.
Production seals every natural field/mask before a separate scoring invocation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil
import time

for _env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_env] = '4'

import numpy as np
import torch
from scipy import sparse
from scipy.special import expit
from scipy.sparse.linalg import cg
import lab
import plan_common as pc
import plan_report as pr
from b1_kshot import threshold_oracle

OUT = lab.ASSETS / 'a/x1012_R1'
P0 = lab.ASSETS / 'a/x1012_P0/fields'
B4 = lab.ASSETS / 'a/x1012_B4/seed150'
MOD = lab.ASSETS / 'a/x1012_B4_modulated/region/seed150'
FAMILIES = ('raw', 'VP')
MODES = ('G0', 'BG', 'UN', 'PERM')
ARMS = tuple(f'{family}_{mode}' for family in FAMILIES for mode in MODES)


def initialize():
    pc.setup('cpu')


def graph(q, neighbors):
    """Exactly the A7 reciprocal20 graph, including its normalization."""
    q = torch.nn.functional.normalize(q, dim=1)
    ids = torch.as_tensor(neighbors.astype(np.int64))
    values = torch.empty((4096, 20), dtype=torch.float32)
    for start in range(0, 4096, 256):
        values[start:start+256] = (q[start:start+256, None] * q[ids[start:start+256]]).sum(-1)
    distance = (1 - values).clamp_min(0)
    weights = torch.exp(-distance / distance[:, -1:].clamp_min(1e-6)).numpy().ravel()
    w = sparse.csr_matrix((weights, (np.repeat(np.arange(4096), 20), neighbors.ravel())), shape=(4096, 4096))
    w = w.multiply(w.T)
    w.data = np.sqrt(w.data)
    degree = np.asarray(w.sum(1)).ravel()
    w = w / max(float(degree.mean()), 1e-8)
    return w, sparse.diags(np.asarray(w.sum(1)).ravel()) - w


def solve(field, explanation, laplacian, mode, permutation):
    s = lab.minmax(torch.as_tensor(field, dtype=torch.float32).reshape(-1)).numpy()
    a0 = .1 + np.abs(2*s - 1)
    y = s.astype(np.float64)
    a = a0.copy()
    if mode == 'BG':
        y *= explanation
    elif mode == 'UN':
        a *= explanation
    elif mode == 'PERM':
        a *= explanation[permutation]
    a = (a / a.mean()).astype(np.float64)
    h = sparse.diags(a) + 16*laplacian
    rhs = a*y
    iterations = [0]
    def callback(_):
        iterations[0] += 1
    z, status = cg(h, rhs, x0=y, rtol=1e-7, atol=1e-9, maxiter=300, callback=callback)
    if status:
        raise RuntimeError(f'CG failed: {mode}, status={status}')
    residual = float(np.linalg.norm(h@z-rhs)/max(np.linalg.norm(rhs), 1e-12))
    assert np.isfinite(z).all() and residual < 2e-7
    return z.reshape(64, 64).astype(np.float32), {'iterations': iterations[0], 'relative_residual': residual}


def produce_one(binding):
    name = pc.key(binding)
    path = OUT/'records'/f'{name}.json'
    if path.exists():
        old = pc.read(path)
        for kind in ('prediction', 'field'):
            assert pc.sha(old[kind+'_path']) == old[kind+'_sha256']
        return old
    started = time.monotonic()
    with pc.inference_only(binding):
        source = pc.read(P0/'records'/f'{name}.json')
        assert pc.sha(source['path']) == source['sha256']
        with np.load(source['path'], allow_pickle=False) as saved:
            f = {k:saved[k].copy() for k in ('c32','log_e','query_neighbors20','ref_stats','mask/IND0','mask/T')}
        raw = binding['fit']['query_raw_identity']
        expected = raw.get('payload_sha256')
        if expected is None:
            expected = pc.read(Path(raw['payload_path']).parent/'entry.json')['file_sha256']
        assert pc.sha(raw['payload_path']) == expected
        q = lab.debias(lab.feats(binding, 'query'))
        w, laplacian = graph(q, f['query_neighbors20'])
        log_e = f['log_e'].astype(np.float64)
        explanation = expit(log_e - np.median(log_e)).astype(np.float32)
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:16], 16)
        permutation = np.random.default_rng(seed).permutation(4096)
        with np.load(B4/'fields'/f'{name}.npz', allow_pickle=False) as z:
            additions = z['addition_V'] + z['addition_P']
        with np.load(MOD/'fields'/f'{name}.npz', allow_pickle=False) as z:
            vp = z['c_mod'] + additions
        rho = float(f['ref_stats'][2])
        fields, predictions, info = {}, {'IND0': f['mask/IND0'], 'T': f['mask/T']}, {}
        for family, primary in (('raw', f['c32']), ('VP', vp)):
            for mode in MODES:
                arm = f'{family}_{mode}'
                if family == 'VP' and rho > .9:
                    result, detail = f['c32'].reshape(64,64), {'raw_T_fallback': True}
                else:
                    result, detail = solve(primary, explanation, laplacian, mode, permutation)
                fields[arm] = result
                predictions[arm] = pc.pack(pc.readout(result))
                info[arm] = detail
        # Reused A7 is an exact implementation control, not a scientific gate.
        old = lab.ASSETS/'a/x1012_A7/seed150/predictions'/f'{name}.npz'
        with np.load(old, allow_pickle=False) as z:
            identity_xor = int(np.count_nonzero(pc.unpack(z['IND_graph']) ^ pc.unpack(predictions['raw_G0'])))
        assert identity_xor == 0, (name, 'A7 reproduction', identity_xor)
        fields.update(explanation=explanation, reference_rho=np.asarray([rho],np.float32))
        # Store only sparse graph arrays for scoring-only neighbor diagnostics.
        fields.update(graph_data=w.data, graph_indices=w.indices, graph_indptr=w.indptr,
                      raw_s=lab.minmax(torch.as_tensor(f['c32'])).numpy())
        pred_path, field_path = OUT/'predictions'/f'{name}.npz', OUT/'fields'/f'{name}.npz'
        assert not pred_path.exists() and not field_path.exists()
        np.savez_compressed(pred_path, **predictions)
        np.savez_compressed(field_path, **fields)
    record = {'index':name, 'dataset':binding['dataset'], 'episode_id':binding['row']['episode_id'],
              'prediction_path':str(pred_path),'prediction_sha256':pc.sha(pred_path),
              'field_path':str(field_path),'field_sha256':pc.sha(field_path),
              'P0_field_sha256':source['sha256'], 'A7_mask_xor':identity_xor,
              'reference_rho':rho,'solver':info,'seconds':time.monotonic()-started,
              'new_encoder_calls':0,'inference_query_GT_reads':0,'FoRIS_score_consumed':False}
    pc.write(path,record)
    return record


def prepare():
    for folder in ('records','predictions','fields','frozen'):
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    if not (OUT/'config.json').exists():
        sources = [Path(__file__), Path(pc.__file__), Path(pr.__file__), Path(lab.__file__),
                   Path(__file__).with_name('b1_kshot.py')]
        method = {'state':'FROZEN_BEFORE_NATURAL_OUTPUTS','n':1360,'arms':ARMS,
                  'explanation':'sigmoid(log_e - median(log_e)); P0 FP16 log_e cast to FP64',
                  'a0':'.1+abs(2*minmax(primary)-1)','graph':'unchanged A7 reciprocal20, lambda16',
                  'readout':'bilinear1024, minmax, strict>.5','VP_rho_gate':.9,
                  'new_encoder_calls':0,'reference_count':1,'query_GT_in_inference':False,
                  'source_sha256':{str(p):pc.sha(p) for p in sources},
                  'dependency_seal_sha256':{str(p):pc.sha(p) for p in [P0/'sealed.json', B4/'sealed.json', MOD/'sealed.json']}}
        for i,p in enumerate(sources):shutil.copyfile(p,OUT/'frozen'/f'{i}_{p.name}')
        pc.write(OUT/'config.json',method)
        shutil.copyfile(B4/'manifest.json',OUT/'manifest.json')
    config=pc.read(OUT/'config.json')
    assert all(pc.sha(p)==h for p,h in config['source_sha256'].items()),'R1 source changed'
    return pc.read(OUT/'manifest.json')


def produce(workers):
    rows=prepare();assert len(rows)==1360
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        records=[]
        for i,r in enumerate(pool.map(produce_one,rows,chunksize=1),1):
            records.append(r)
            if i<=3 or i%50==0:print({'phase':'R1-produce','completed':i,'n':len(rows),'seconds':round(time.monotonic()-started,1)},flush=True)
    pc.write(OUT/'sealed.json',{'state':'ALL_NATURAL_OUTPUTS_SEALED','n':len(rows),'records':records,
                              'manifest_sha256':pc.sha(OUT/'manifest.json'),'config_sha256':pc.sha(OUT/'config.json'),
                              'wall_seconds':time.monotonic()-started,'case_seconds_sum':sum(r['seconds'] for r in records),
                              'workers':workers,'threads_per_worker':4,'new_encoder_calls':0,'inference_query_GT_reads':0})


def score_one(job):
    binding,receipt=job
    with np.load(receipt['prediction_path'],allow_pickle=False) as z:
        predictions={a:pc.unpack(z[a]) for a in z.files}
    record=pr.score_record(binding,predictions)
    gt,valid=pc.truth(binding)
    with np.load(receipt['field_path'],allow_pickle=False) as z:
        for arm in ARMS:
            oracle,_=threshold_oracle(z[arm],gt,valid)
            record['metrics'][arm+'/oracle']=pr.metric(oracle,gt,valid)
        coverage=lab.coverage64(torch.from_numpy(gt)).numpy().ravel()
        w=sparse.csr_matrix((z['graph_data'],z['graph_indices'],z['graph_indptr']),shape=(4096,4096))
        e=z['explanation'];a=.1+np.abs(2*z['raw_s']-1)
        den=w@(a*e)
        neighborhood=np.divide(w@(a*e*coverage),den,out=coverage.copy(),where=den>0)
        low=e<=np.quantile(e,.25)
        fn=low&(z['raw_s']<=.5)&(coverage>=.5)
        fp=low&(z['raw_s']>.5)&(coverage<.5)
        # Diagnostic GT labels are used only after the whole natural run is sealed.
        record['neighbors_diagnostic']={
            'lowE_FN_n':int(fn.sum()),'lowE_FN_neighbor_gt_sum':float(neighborhood[fn].sum()),
            'lowE_FP_n':int(fp.sum()),'lowE_FP_neighbor_gt_sum':float(neighborhood[fp].sum()),
            'label_disagreement_weight':float(np.sum(w.data*np.abs(coverage[np.repeat(np.arange(4096),np.diff(w.indptr))]-coverage[w.indices]))),
            'edge_weight':float(w.data.sum())}
    return record


def score(workers):
    initialize();rows=prepare();seal=pc.read(OUT/'sealed.json')
    assert seal['state']=='ALL_NATURAL_OUTPUTS_SEALED' and seal['n']==len(rows)
    assert pc.sha(OUT/'config.json')==seal['config_sha256'] and pc.sha(OUT/'manifest.json')==seal['manifest_sha256']
    receipts={r['index']:r for r in seal['records']}
    for r in receipts.values():
        for kind in ('prediction','field'):assert pc.sha(r[kind+'_path'])==r[kind+'_sha256']
    start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for i,r in enumerate(pool.map(score_one,((b,receipts[pc.key(b)]) for b in rows),chunksize=1),1):
            records.append(r)
            if i<=3 or i%100==0:print({'phase':'R1-score','completed':i,'n':len(rows),'seconds':round(time.monotonic()-start,1)},flush=True)
    controls={r['index']:r for r in pc.read(B4/'report.json')['records']}
    prior={r['index']:r for r in pc.read(lab.ASSETS/'a/x1012_B4_assembly/seed150/report.json')['records']}
    for r in records:
        old=controls[r['index']]
        assert old['query_photo']==r['query_photo']
        for arm in ('FoRIS','MEAN'):r['metrics'][arm]=old['metrics'][arm]
        r['metrics']['VP_previous']=prior[r['index']]['metrics']['VP_thin09']
    arms=('FoRIS','MEAN','VP_previous','IND0','T')+ARMS+tuple(a+'/oracle' for a in ARMS)
    report=pr.report('R1',OUT,records,arms,[
        '固定1360单参考开发任务，使用已有O24/P0字段；无新编码、无FoRIS输出推理、无域分支。',
        '八臂全部自然输出先封存；GT仅在本评分阶段读取。G0逐例与原A7掩码逐位一致。',
        'raw/VP两组各比较G0、陌生作为背景BG、陌生作为未知UN和可靠度置换PERM；参数与读出相同。',
        'VP沿用旧max-J+V/P证据及固定rho>.9原T回退。oracle仅作诊断，不参与自然阈值选择。'],
        extra={'production':{k:v for k,v in seal.items() if k!='records'},'score_seconds':time.monotonic()-start})
    contrasts={}
    for family in FAMILIES:
        for baseline in ('G0','BG','PERM'):
            projected=[dict(r,metrics={'FoRIS':r['metrics'][family+'_'+baseline], 'UN':r['metrics'][family+'_UN']}) for r in records]
            contrasts[family+'/UN_minus_'+baseline]=pr.summary(projected,('FoRIS','UN'))
    pc.write(OUT/'mechanism_contrasts.json',contrasts)
    passing=[a for a in ARMS if all(t[a]['delta_to_foris_pp']>=-.5 for t in report['summary'].values())]
    rank=sorted(ARMS,key=lambda a:(min(t[a]['delta_to_foris_pp'] for t in report['summary'].values()),np.mean([t[a]['delta_to_foris_pp'] for t in report['summary'].values()])),reverse=True)
    pc.write(OUT/'completion.json',{'state':'COMPLETE','n':len(rows),'candidate_rank':rank,
                                  'ten_domain_gate_passing':passing,'new_encoder_calls':0,
                                  'total_score_seconds':time.monotonic()-start})
    print({'phase':'R1-complete','passing':passing,'rank':rank},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('produce','score'))
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    produce(args.workers) if args.mode=='produce' else score(args.workers)
