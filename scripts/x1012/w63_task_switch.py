"""Fixed-RGB legal task switching: metadata prepare, sealed prediction, then scoring."""
import fcntl
import gzip
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/x1012'), str(ROOT/'scripts/x1013')]
import common as c
import w59_normalization_causal as run

OUT = c.task_path('W63')
SEED = 2026101608


def image_key(b, side):
    r = b['row']
    return (b['dataset'], str(r['fold']), r[side+'_path'],
            json.dumps(r.get(side+'_crop')), r[side+'_rgb_hash'])


def prepare():
    from w48_reference_specificity import with_reference
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'manifest.json').exists():
        print('Retain prepared manifest', run.sha(OUT/'manifest.json')); return
    groups = {s: defaultdict(dict) for s in ('reference', 'query')}
    for b in sorted(c.rows(), key=lambda b: str(b['index'])):
        if c.half(b) != 'S': continue
        for side in groups:
            raw = b['fit']['reference_raw' if side == 'reference' else 'query_raw_identity']
            if Path(raw['payload_path']).is_file():
                groups[side][image_key(b, side)].setdefault(str(b['row']['global_class_id']), b)
    reference_pairs = defaultdict(list)
    for k, v in sorted(groups['reference'].items()):
        for pair in itertools.combinations(sorted(v), 2):
            reference_pairs[k[0], k[1], pair].append(k)
    eligible = defaultdict(list)
    for k, v in sorted(groups['query'].items()):
        choices = []
        for pair in itertools.combinations(sorted(v), 2):
            refs = [r for r in reference_pairs[k[0], k[1], pair] if r[2] != k[2] and r[4] != k[4]]
            if refs: choices.append((pair, refs))
        if choices: eligible[k[0]].append((k, choices))
    rng = np.random.default_rng(SEED); items = []; counts = {}
    for d in c.DOMAINS:
        candidates = eligible[d]; used_rgb = set(); count = 0
        for i in rng.permutation(len(candidates)):
            qkey, choices = candidates[i]
            if (qkey[4], qkey[3]) in used_rgb: continue
            pair, refs = choices[int(rng.integers(len(choices)))]; rkey = refs[int(rng.integers(len(refs)))]
            bindings = []; sources = []
            for cls in pair:
                qb, rb = groups['query'][qkey][cls], groups['reference'][rkey][cls]
                assert qb['row']['global_class_id'] == rb['row']['global_class_id']
                b = with_reference(qb, rb['row'], rb['fit']['reference_raw'], 'reference')
                assert c.half(b) == 'S' and image_key(b, 'query') == qkey and image_key(b, 'reference') == rkey
                bindings.append(b); sources.append(dict(query_index=str(qb['index']), reference_index=str(rb['index'])))
            items.append(dict(pair_id=f'{d}_{count:03d}', dataset=d, photo=c.photo(bindings[0]),
                              classes=pair, source_indices=sources, bindings=bindings))
            used_rgb.add((qkey[4], qkey[3])); count += 1
            if count == 30: break
        counts[d] = dict(eligible_queries=len(candidates), selected=count)
    assert items
    run.write(OUT/'manifest.json', dict(seed=SEED, per_domain_cap=30, n=len(items), counts=counts,
                                       query_GT_reads=0, episodes=items))
    print(json.dumps(dict(n=len(items), counts=counts), indent=2))


def one(item, verify=False):
    if time.time() >= run.DEADLINE: raise TimeoutError('Authorized inference window ended')
    import torch
    import w42_ordered_cut as w42
    import w23_simple_pipeline as sh
    import produce as prod
    start = time.monotonic(); arrays = {}; traces = []; oi = c.rc.independent
    for label, b in zip(('A', 'B'), item['bindings']):
        got = []
        def capture(*args, **kwargs):
            value = oi(*args, **kwargs); got.append(value); return value
        c.rc.independent = capture
        try:
            with c.rc.pc.inference_only(b):
                field, pre, _, _, rho = w42.capture(b)
                post = pre if rho > .9 else prod.crf(b, pre)
        finally:
            c.rc.independent = oi
        assert np.array_equal(field > .5, pre)
        r, q, mask, y, fields = got[0]
        traces.append((r, q, mask, rho))
        arrays[label] = c.pack(pre); arrays[label+'+'] = c.pack(post); arrays['field_'+label] = field
        if verify:
            with c.rc.pc.inference_only(b):
                p0, p1, _ = sh.simple_head(b, crf=True)
            assert np.array_equal(p0, pre) and np.array_equal(p1, post)
    assert torch.equal(traces[0][0], traces[1][0]) and torch.equal(traces[0][1], traces[1][1])
    path = OUT/'predictions'/(item['pair_id']+'.npz'); c.save_npz(path, **arrays)
    return dict(pair_id=item['pair_id'], dataset=item['dataset'], photo=item['photo'], classes=item['classes'],
                source_indices=item['source_indices'], prediction=str(path), prediction_sha256=run.sha(path),
                same_RQ_features=True, same_reference_mask=torch.equal(traces[0][2], traces[1][2]),
                rho=[v[3] for v in traces], query_GT_reads=0, seconds=time.monotonic()-start)


def smoke():
    run.init(); item=json.loads((OUT/'manifest.json').read_text())['episodes'][0]
    r=one(item, verify=True)
    run.write(OUT/'smoke.json', dict(pair_id=r['pair_id'], standalone_SH_pixel_identity=True,
                                   same_RQ_features=True, query_GT_reads=0))
    path=OUT/'episodes.jsonl'
    assert not path.exists()
    path.write_text(json.dumps(r)+'\n'); print('Smoke PASS', r['pair_id'])


def produce(workers):
    assert workers in (1, 2)
    manifest=json.loads((OUT/'manifest.json').read_text()); items=manifest['episodes']
    lock=(OUT/'run.lock').open('a+')
    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: print('Already running'); return
    paths=[Path(__file__), Path(run.__file__), ROOT/'scripts/x1012/w42_ordered_cut.py',
           ROOT/'scripts/x1012/w23_simple_pipeline.py', ROOT/'scripts/x1012/r5_candidate.py',
           ROOT/'scripts/x1012/diagnostic_metrics.py', ROOT/'docs/research/x1012_tasks/W63_task_switch_PLAN.md']
    config=dict(n=len(items), manifest_sha256=run.sha(OUT/'manifest.json'), workers=workers,
                sources={str(p.relative_to(ROOT)):run.sha(p) for p in paths}, mps=False)
    cp=OUT/'config.json'
    if cp.exists(): assert json.loads(cp.read_text())==config
    else: run.write(cp,config)
    import psutil
    run.write(OUT/'execution.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time(),script=str(Path(__file__)),
                                       status='running',workers=workers,started_utc=datetime.now(timezone.utc).isoformat()))
    path=OUT/'episodes.jsonl'; records=list(map(json.loads,path.read_text().splitlines())) if path.exists() else []
    done={r['pair_id'] for r in records}; todo=[x for x in items if x['pair_id'] not in done]
    try:
        with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),initializer=run.init) as pool,path.open('a') as f:
            for r in pool.map(one,todo,chunksize=1):
                records.append(r);f.write(json.dumps(r)+'\n');f.flush()
                if len(records)%10==0: print(len(records),len(items),'pair seconds',round(r['seconds'],2),flush=True)
        assert len(records)==len(items) and len({r['pair_id'] for r in records})==len(items)
        assert all(run.sha(r['prediction'])==r['prediction_sha256'] for r in records)
        run.write(OUT/'complete.json',dict(n=len(items),episodes_sha256=run.sha(path),config_sha256=run.sha(cp),query_GT_reads=0,utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as e:
        run.write(OUT/'failure.json',dict(type=type(e).__name__,message=str(e),records=len(records)));raise
    finally:
        e=json.loads((OUT/'execution.json').read_text());e['status']='complete' if (OUT/'complete.json').exists() else 'partial_or_failed'
        run.write(OUT/'execution.json',e);lock.close()


def metrics(pa,pb,ga,gb,va,vb):
    ius=[[c.iu(p,g,v) for g,v in ((ga,va),(gb,vb))] for p in (pa,pb)]
    j=np.array([[x[0]/max(x[1],1) for x in row] for row in ius])
    common=va&vb; diff=(ga^gb)&common; commonfg=ga&gb&common; commonbg=~(ga|gb)&common
    direction=(gb.astype(np.int8)-ga.astype(np.int8))*(pb.astype(np.int8)-pa.astype(np.int8))
    dm=float((j[0,0]+j[1,1]-j[0,1]-j[1,0])/2)
    if np.array_equal(pa,pb): assert abs(dm)<1e-12
    return dict(iu=ius,matched_mean=float((j[0,0]+j[1,1])/2),matched_min=float(min(j[0,0],j[1,1])),
                both_matched_ge_half=bool(min(j[0,0],j[1,1])>=.5),D_match=dm,
                T_dir=float(direction[diff].mean()) if diff.any() else None,
                difference_pixels=int(diff.sum()),difference_excluded_by_ignore=int(((ga^gb)&~common).sum()),
                common_fg=int(commonfg.sum()),common_bg=int(commonbg.sum()),
                common_fg_destroy=int((pa&~pb&commonfg).sum()),common_fg_repair=int((~pa&pb&commonfg).sum()),
                common_bg_destroy=int((~pa&pb&commonbg).sum()),common_bg_repair=int((pa&~pb&commonbg).sum()))


def score():
    from diagnostic_metrics import exact_best_iou
    import plan_common as pc
    complete=json.loads((OUT/'complete.json').read_text())
    assert run.sha(OUT/'episodes.jsonl')==complete['episodes_sha256'] and run.sha(OUT/'config.json')==complete['config_sha256']
    manifest=json.loads((OUT/'manifest.json').read_text()); assert complete['n']==manifest['n']
    meta={r['pair_id']:r for r in map(json.loads,(OUT/'episodes.jsonl').read_text().splitlines())}; rows=[]
    for item in manifest['episodes']:
        row=dict(meta[item['pair_id']]);assert run.sha(row['prediction'])==row['prediction_sha256']
        (ga,va),(gb,vb)=[pc.truth(b) for b in item['bindings']]
        common=va&vb; aa,bb=ga&common,gb&common
        relation='equal' if np.array_equal(aa,bb) else 'disjoint' if not (aa&bb).any() else 'nested' if not (aa&~bb).any() or not (bb&~aa).any() else 'overlap'
        row['query_relation']=relation;row['query_gt_nonempty']=[bool(ga.any()),bool(gb.any())]
        with np.load(row['prediction'],allow_pickle=False) as z:
            for post in ('','+'):
                row['post' if post else 'pre']=metrics(c.unpack(z['A'+post]),c.unpack(z['B'+post]),ga,gb,va,vb)
            oracle=[[exact_best_iou(z['field_'+a],g,v) for g,v in ((ga,va),(gb,vb))] for a in ('A','B')]
        row['oracle']=oracle
        for i in range(2):
            for j in range(2):
                u=row['pre']['iu'][i][j];assert oracle[i][j]+1e-12>=u[0]/max(u[1],1)
        rows.append(row)
    summary=dict(n=len(rows),source_scope='fixed R/Q RGB legal S task pairs; pair equal, not official ten-domain mIoU',
                 limitations='query-pair bootstrap conditions on selected references; reference reuse and cross-role dependence not resolved',
                 same_reference_mask=sum(r['same_reference_mask'] for r in rows),relations=dict(Counter(r['query_relation'] for r in rows)),groups={})
    rng=np.random.default_rng(SEED)
    for name,sel in [('all',rows)]+[(d,[r for r in rows if r['dataset']==d]) for d in c.DOMAINS if any(r['dataset']==d for r in rows)]:
        stats=dict(n=len(sel),relations=dict(Counter(r['query_relation'] for r in sel)),natural_failed_tasks=0,failed_tasks_oracle_ge_half=0)
        for r in sel:
            for i in range(2):
                u=r['pre']['iu'][i][i];fail=u[0]/max(u[1],1)<.5
                stats['natural_failed_tasks']+=fail;stats['failed_tasks_oracle_ge_half']+=bool(fail and r['oracle'][i][i]>=.5)
        for post in ('pre','post'):
            s={}; draw=rng.integers(len(sel),size=(500,len(sel)))
            for k in ('matched_mean','matched_min','D_match','T_dir'):
                x=np.array([r[post][k] if r[post][k] is not None else np.nan for r in sel],float)
                s[k]=dict(mean=float(np.nanmean(x)),ci=np.nanpercentile(np.nanmean(x[draw],axis=1),[2.5,97.5]).tolist()) if np.isfinite(x).any() else None
            s['both_matched_ge_half']=sum(r[post]['both_matched_ge_half'] for r in sel)
            s['D_positive']=sum(r[post]['D_match']>0 for r in sel)
            for k in ('common_fg','common_bg','common_fg_destroy','common_fg_repair','common_bg_destroy','common_bg_repair','difference_pixels','difference_excluded_by_ignore'):
                s[k]=sum(r[post][k] for r in sel)
            stats[post]=s
        summary['groups'][name]=stats
    with gzip.open(ROOT/'evidence/research/w63_task_switch_S.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    run.write(OUT/'summary.json',summary);run.write(ROOT/'evidence/research/w63_task_switch_summary.json',summary)
    run.write(OUT/'scored.json',dict(n=len(rows),scorer_sha256=run.sha(Path(__file__)),utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='smoke':smoke()
    elif sys.argv[1]=='produce':produce(int(sys.argv[2]) if len(sys.argv)>2 else 2)
    elif sys.argv[1]=='score':
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):score()
