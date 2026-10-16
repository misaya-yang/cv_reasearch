"""W71: prepare/check/start/status/stop a fixed, offline, resumable 48h queue.

No start is implicit. `start` requires a successful check of the exact package.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import w71_runtime as rt

DOMAINS=('isaid','pascal_part','paco_part','lvis')
W30=rt.ASSETS/'a/x1013_W30'


def records(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as stream: return {str(r['index']):r for r in map(json.loads,stream)}


def identify(binding,role):
    row=binding['row']
    return row.get(role+'_rgb_hash') or str((rt.ASSETS/row[role+'_path']).resolve())+'|'+json.dumps(row.get(role+'_crop'))


def select_manifest():
    import common as c
    binds={str(b['index']):b for b in c.rows()}
    metrics=records(rt.ROOT/'evidence/research/w27_sh_single_line_S.jsonl.gz')
    preds=records(rt.ASSETS/'a/x1013_W27/predictions.jsonl')
    mechanism=[]; checks=[]
    def make(binding,cohort,baseline):
        return dict(id=cohort+'_'+str(binding['index']).replace('/','_'),cohort=cohort,binding=binding,
                    qimg=identify(binding,'query'),rimg=identify(binding,'reference'),baseline=baseline)
    for key,r in metrics.items():
        b=binds[key]; p=preds[key]
        baseline=dict(path=p['prediction'],sha256=p['payload_sha256'],pre='SH',final='SH+')
        if r['dataset']=='fundus':
            assert c.half(b)=='S' and r['rho']>.9
            mechanism.append(make(b,'S',baseline))
    assert len(mechanism)==108
    for pool,cohort in (('A','AVSeg'),('B','FIVES_train')):
        for b in rt.read(rt.ASSETS/f'a/x1013_W54/manifest_{pool}.json'):
            old=rt.read(rt.ASSETS/f'a/x1013_W54/sh_{pool}/records/{b["index"]}.json')
            assert old['rho']>.9
            mechanism.append(make(b,cohort,dict(path=old['prediction_path'],sha256=old['prediction_sha256'],pre='SH',final='SH+')))
    assert len(mechanism)==408
    order=lambda item:hashlib.sha256(('2026101677/'+item['id']).encode()).hexdigest()
    mechanism.sort(key=lambda r:(('S','AVSeg','FIVES_train').index(r['cohort']),order(r)))
    for domain in DOMAINS:
        choices=[(k,r) for k,r in metrics.items() if r['dataset']==domain]
        choices.sort(key=lambda x:hashlib.sha256(('2026101678/'+x[0]).encode()).hexdigest())
        for key,r in choices[:2]:
            p=preds[key]; checks.append(make(binds[key],domain,dict(path=p['prediction'],sha256=p['payload_sha256'],pre='SH',final='SH+')))
    official=rt.read(rt.ASSETS/'a/x1013_official/manifest.json')
    return dict(mechanism=mechanism,checks=checks,official={d:[b for b in official if b['dataset']==d] for d in DOMAINS},
                statistics=dict(photo_seed=2026101675,image_seed=2026101676,repeats=5000,primary_tests=9,
                                primary_layer=23,secondary_layer=19,recalls=[.25,.5,.75]),
                consistency=dict(cached_head_xor_pixels=0,fresh_max_xor_fraction=.001,min_raw_cosine=.9999,max_raw_relative_l2=.001))


def dependency_files():
    files=[rt.ASSETS/'demo4_cache/models/dinov3-vitl16-timm/model.safetensors',
           rt.ASSETS/'demo4_cache/models/dinov3-vitl16-timm/config.json',rt.ASSETS/'native_assets/positional_basis.pt',
           rt.ASSETS/'a/fundus_layer_resolution50_20261013/basis/1024.pt',rt.ASSETS/'a/x1013_official/frozen/encoder.py',
           rt.ASSETS/'a/x1013_official/manifest.json',rt.ASSETS/'a/x1013_official/protocol.json',W30/'evaluation.json']
    for sub in ('a/x1013/w02/runtime','a/x1012s2_runtime/mfu_boundary/frozen','a/x1012s2_runtime/mfu_boundary/runtime'):
        files.extend((rt.ASSETS/sub).rglob('*.py'))
    return sorted(set(files))


def prepare():
    rt.offline(); rt.OUT.mkdir(parents=True,exist_ok=True)
    lock=(rt.OUT/'package.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (rt.OUT/'session.json').exists(): raise RuntimeError('An execution session exists; do not replace its package')
    manifest=select_manifest()
    old=rt.read(W30/'evaluation.json')
    assert all(rt.sha(rt.ROOT/k)==v for k,v in old['method_sha256'].items()), 'W30 frozen method changed'
    paths=set()
    all_rows=[r['binding'] for r in manifest['mechanism']+manifest['checks']]
    all_rows += [b for group in manifest['official'].values() for b in group]
    for b in all_rows:
        for k,value in b['row'].items():
            if k.endswith('_path') and isinstance(value,str) and value:
                paths.add((rt.ASSETS/value).resolve())
    inventory={}
    for p in sorted(paths):
        s=p.stat(); assert p.is_file(),p
        inventory[str(p)]=dict(size=s.st_size,mtime_ns=s.st_mtime_ns)
    rt.write(rt.OUT/'input_inventory.json',inventory)
    rt.write(rt.OUT/'manifest.json',manifest)
    # Snapshot Python source, including original heads. Nothing is moved/deleted.
    sources=list((rt.ROOT/'scripts').rglob('*.py'))+list((rt.ROOT/'src').rglob('*.py'))
    hashes={str(p.relative_to(rt.ROOT)):rt.sha(p) for p in sources}
    protocol=rt.ROOT/'docs/research/x1012_tasks/W71_weekend_offline_PLAN.md'
    digest=hashlib.sha256(json.dumps(dict(sources=hashes,protocol=rt.sha(protocol)),sort_keys=True).encode()).hexdigest()[:16]
    root=rt.OUT/'packages'/digest/'repo'
    for p in sources:
        dest=root/p.relative_to(rt.ROOT); dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copy2(p,dest)
        assert rt.sha(dest)==hashes[str(p.relative_to(rt.ROOT))]
    assets=root.parent/'cv_data'
    if not assets.exists(): assets.symlink_to(rt.ASSETS,target_is_directory=True)
    # A few old modules refer to evidence by project-root path; read-only alias.
    evidence=root/'evidence'
    if not evidence.exists(): evidence.symlink_to(rt.ROOT/'evidence',target_is_directory=True)
    protocol_copy=root.parent/'protocol.md'
    if not protocol_copy.exists(): shutil.copy2(protocol,protocol_copy)
    config=dict(root=str(root),source_sha256=hashes,dependencies={str(p):rt.sha(p) for p in dependency_files()},
                manifest_sha256=rt.sha(rt.OUT/'manifest.json'),inventory_sha256=rt.sha(rt.OUT/'input_inventory.json'),
                protocol_sha256=rt.sha(protocol_copy),hours=48,workers=4,threads_per_process=2,mps_duty=.8,
                min_available_gib=6,memory_budget_gib=25,retries_per_stage=2,stage_timeout_seconds=1800,created_utc=rt.utc())
    rt.write(rt.OUT/'package.json',config)
    lock.close()
    print(json.dumps(dict(package=str(rt.OUT/'package.json'),snapshot=str(root),input_files=len(inventory),
                          mechanism=len(manifest['mechanism']),official={d:len(r) for d,r in manifest['official'].items()}),indent=2))


def verify_package(inputs=False):
    cfg=rt.read(rt.OUT/'package.json'); root=Path(cfg['root'])
    assert rt.sha(rt.OUT/'manifest.json')==cfg['manifest_sha256']
    assert rt.sha(rt.OUT/'input_inventory.json')==cfg['inventory_sha256']
    assert rt.sha(root.parent/'protocol.md')==cfg['protocol_sha256']
    if rt.ROOT != root:
        for name in ('w71_runtime.py','w71_offline.py','w71_mechanism.py'):
            rel='scripts/x1012/'+name
            assert rt.sha(rt.ROOT/rel)==cfg['source_sha256'][rel], 'Code was edited; run prepare and check for the reviewed version'
        assert rt.sha(rt.ROOT/'docs/research/x1012_tasks/W71_weekend_offline_PLAN.md')==cfg['protocol_sha256'], 'Protocol edited; prepare and check again'
    for path,value in cfg['source_sha256'].items(): assert rt.sha(root/path)==value,path
    for path,value in cfg['dependencies'].items(): assert rt.sha(path)==value,path
    if inputs:
        for name,stat in rt.read(rt.OUT/'input_inventory.json').items():
            now=Path(name).stat()
            assert now.st_size==stat['size'] and now.st_mtime_ns==stat['mtime_ns'],f'Input changed: {name}'
    return cfg


def official_init(output=None):
    rt.initialize()
    if output is not None:
        import w30_confirm as w30
        w30.OUT=Path(output)


def official_head(job):
    import common as c
    import w30_confirm as w30
    with c.rc.pc.inference_only(job[1]):
        rt.check_mask(job[1],'reference')
        return w30.head(job)


def official_predict(domain):
    import w12_official as w12
    import w30_confirm as w30
    rt.initialize(); rows=rt.read(rt.OUT/'manifest.json')['official'][domain]
    seal=W30/'sealed'/f'{domain}.json'
    if seal.exists():
        saved=rt.read(seal); assert saved['n']==len(rows)
        for b in rows:
            r=rt.valid_record(W30/'records'/domain/f'{w30.safe(b["index"])}.json')
            assert saved['predictions'][str(b['index'])]==r['prediction_sha256']
        return
    todo=[]
    for b in w12.bfs_order(rows):
        if rt.valid_record(W30/'records'/domain/f'{w30.safe(b["index"])}.json') is None: todo.append(b)
    cfg=rt.read(rt.OUT/'package.json'); workers=int(os.environ.get('W71_WORKERS',cfg['workers']))
    enc=rt.Encoder(); done=len(rows)-len(todo); pending=set()
    if todo:
        import psutil
        enc.features(todo[0])  # Warm MPS allocations before choosing CPU pool size.
        rt.capacity(minimum_gib=8)
        workers=rt.worker_budget(workers,psutil.virtual_memory().available/2**30,rt.job_memory_gib())
        rt.event('worker_budget',domain=domain,workers=workers,available_gib=psutil.virtual_memory().available/2**30)
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),initializer=official_init) as pool:
        for b in todo:
            rt.checkpoint(); tick=time.monotonic(); maps,trace=enc.features(b)
            feats={role:value[23] for role,value in maps.items()}
            pending.add(pool.submit(official_head,(domain,b,feats,dict(new_encoded_images=sum(x['fresh'] for x in trace),
                                encode_seconds=time.monotonic()-tick,offline_package_sha256=rt.sha(rt.OUT/'package.json')))))
            while len(pending)>=workers:
                finished,pending=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                rt.checkpoint()
                for future in finished: future.result(); done+=1
                rt.heartbeat('official_'+domain,done,len(rows),encoder_calls=enc.calls,workers=workers)
            if done%100==0: rt.event('official_progress',domain=domain,done=done,total=len(rows))
        for future in pending: future.result(); done+=1
    recs=[rt.valid_record(W30/'records'/domain/f'{w30.safe(b["index"])}.json') for b in rows]
    assert len(recs)==len(rows) and all(recs)
    rt.write(seal,dict(part=domain,n=len(recs),sealed_at=rt.utc(),evaluation_sha256=rt.sha(W30/'evaluation.json'),
                       predictions={r['index']:r['prediction_sha256'] for r in recs}))


def cached_official_scores(domain,rows):
    """Deterministic recovery after interruption; each completed GT row is reused."""
    import w30_confirm as w30
    seal=rt.read(W30/'sealed'/f'{domain}.json')
    assert seal['n']==len(rows) and seal['evaluation_sha256']==rt.sha(W30/'evaluation.json')
    result=[]
    for n,b in enumerate(rows):
        rt.checkpoint(); target=rt.OUT/'official_scores'/domain/f'{w30.safe(b["index"])}.json'
        pred=rt.valid_record(W30/'records'/domain/f'{w30.safe(b["index"])}.json')
        assert seal['predictions'][str(b['index'])]==pred['prediction_sha256']
        if target.exists(): row=rt.read(target); assert row['_prediction_sha256']==pred['prediction_sha256']
        else:
            rt.check_mask(b,'query')
            rt.check_mask(b,'query_ignore')
            row=w30._iu((domain,b)); row['_prediction_sha256']=pred['prediction_sha256']; rt.write(target,row)
        result.append(row)
        if n%100==0: rt.heartbeat('score_'+domain,n,len(rows))
    target=W30/'scores'/f'{domain}.jsonl.gz'; target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.tmp')
    with gzip.open(temp,'wt') as stream:
        for row in result: stream.write(json.dumps(row)+'\n')
    temp.replace(target); return result


def official_score(domain):
    rt.initialize(); import w30_confirm as w30
    receipt=rt.OUT/'official_scores'/f'{domain}_complete.json'
    if receipt.exists():
        saved=rt.read(receipt); assert rt.sha(W30/'scores'/f'{domain}.jsonl.gz')==saved['rows_sha256']; return
    w30._score_rows=cached_official_scores
    w30.score_e2(domain)
    rt.write(receipt,dict(rows_sha256=rt.sha(W30/'scores'/f'{domain}.jsonl.gz'),utc=rt.utc()))


def check_worker():
    import numpy as np
    import torch
    import common as c
    import w71_mechanism as mech
    rt.initialize(); cfg=verify_package(inputs=True); manifest=rt.read(rt.OUT/'manifest.json')
    out=rt.OUT/'check'/rt.sha(rt.OUT/'package.json')[:16]; out.mkdir(parents=True,exist_ok=True)
    enc=rt.Encoder((23,19)); results=[]
    chosen=[]
    for cohort in ('S','AVSeg','FIVES_train'):
        chosen.extend([x for x in manifest['mechanism'] if x['cohort']==cohort][:2])
    for item in chosen:
        tick=time.monotonic()
        features,_=enc.features(item['binding']); arrays,detail=mech.generate(item['binding'],features)
        old=item['baseline']; assert rt.sha(old['path'])==old['sha256']
        with np.load(old['path'],allow_pickle=False) as z: baseline=c.unpack(z[old['pre']])
        fresh=c.unpack(arrays['masks'][0,mech.ARMS.index('111')]); drift=float(np.mean(fresh!=baseline))
        assert drift<=manifest['consistency']['fresh_max_xor_fraction'],(item['id'],drift)
        results.append(dict(id=item['id'],anchor_xor_fraction=drift,rho=detail['rho'],seconds=time.monotonic()-tick))
        rt.heartbeat('preflight_mechanism',len(results),len(chosen))
    # Official wrapped head must exactly reproduce old results with old features.
    with ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn'),initializer=official_init,initargs=(str(out),)) as pool:
        for item in manifest['checks']:
            b=item['binding']; old=item['baseline']; assert rt.sha(old['path'])==old['sha256']
            cached={role:c.rc.raw(b['fit'][key],{}) for role,key in (('reference','reference_raw'),('query','query_raw_identity'))}
            features,_=enc.features(b); fresh={role:value[23] for role,value in features.items()}
            cos={role:float(torch.nn.functional.cosine_similarity(cached[role],fresh[role],dim=1).min()) for role in fresh}
            rel={role:float((cached[role]-fresh[role]).norm()/cached[role].norm()) for role in fresh}
            assert min(cos.values())>=manifest['consistency']['min_raw_cosine']
            assert max(rel.values())<=manifest['consistency']['max_raw_relative_l2']
            a=pool.submit(official_head,('cached',b,cached,{})).result()
            z=pool.submit(official_head,('fresh',b,fresh,{})).result()
            with np.load(old['path'],allow_pickle=False) as reference, np.load(a['prediction_path'],allow_pickle=False) as aa, np.load(z['prediction_path'],allow_pickle=False) as zz:
                for key in ('SH','SH+'):
                    assert np.array_equal(reference[key],aa[key]),('cached head differs',item['id'],key)
                    assert np.mean(c.unpack(reference[key])!=c.unpack(zz[key]))<=manifest['consistency']['fresh_max_xor_fraction']
            results.append(dict(id=item['id'],cached_exact=True,min_cosine=min(cos.values()),max_relative_l2=max(rel.values()),
                                fresh_head_seconds=z['head_seconds']))
            rt.heartbeat('preflight_official',len(results)-len(chosen),len(manifest['checks']))
    # Verify the offline guard rather than assuming environment flags suffice.
    import socket
    try: socket.getaddrinfo('example.com',443)
    except RuntimeError as exc: assert 'W71_OFFLINE' in str(exc)
    else: raise AssertionError('Network guard did not block DNS')
    report=dict(status='PASS',package_sha256=rt.sha(rt.OUT/'package.json'),cases=results,utc=rt.utc(),
                mechanism_cases=len(chosen),official_cases=len(manifest['checks']),query_GT_reads=0,network_guard=True,
                fresh_encoder_calls=enc.calls,encoder_active_seconds=enc.encode_seconds,mps_duty=enc.duty)
    rt.write(out/'result.json',report); rt.write(rt.OUT/'ready.json',report)
    rt.event('preflight_pass',mechanism=len(chosen),official=len(manifest['checks']),package=report['package_sha256'])


def stage_command(stage,cfg):
    return [str(rt.PYTHON),'-B',str(Path(cfg['root'])/'scripts/x1012/w71_offline.py'),'worker',stage]


def terminate_group(process):
    # A dead coordinator can leave pool workers alive in its own process group.
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL); process.wait()
    try: os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError: pass


def supervise():
    rt.offline(); cfg=verify_package(inputs=True)
    lock=(rt.OUT/'run.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    session=rt.read(rt.OUT/'session.json'); stages=['mechanism-predict','mechanism-score']
    stages += [part for d in DOMAINS for part in ('official-predict:'+d,'official-score:'+d)]
    failed=[]; current=None; status='RUNNING'
    try:
        for stage in stages:
            rt.checkpoint()
            if stage=='mechanism-score' and 'mechanism-predict' in failed:
                failed.append(stage); continue
            if stage.startswith('official-score:') and stage.replace('score','predict') in failed:
                failed.append(stage); continue
            target=rt.OUT/'stages'/f'{stage.replace(":","_")}.json'
            previous=rt.read(target) if target.exists() else {}
            if previous.get('status')=='complete': continue
            if previous.get('status')=='running' and previous.get('process_created'):
                import psutil
                try:
                    orphan=psutil.Process(previous['pid'])
                    if abs(orphan.create_time()-previous['process_created'])<1:
                        os.killpg(orphan.pid,signal.SIGTERM)
                        try: orphan.wait(timeout=10)
                        except psutil.TimeoutExpired: os.killpg(orphan.pid,signal.SIGKILL)
                except psutil.NoSuchProcess: pass
            success=False
            first_attempt=previous.get('attempt',-1)+1
            for attempt in range(first_attempt,cfg['retries_per_stage']+1):
                rt.checkpoint(); rt.heartbeat(stage)
                log=rt.OUT/'logs'/f'{stage.replace(":","_")}_{attempt}.log'; log.parent.mkdir(exist_ok=True)
                env=dict(os.environ,CV_RESEARCH_ROOT=cfg['root'],W71_WORKERS=str(max(1,4//2**attempt)),PYTHONDONTWRITEBYTECODE='1')
                with log.open('ab') as stream:
                    current=subprocess.Popen(stage_command(stage,cfg),cwd=cfg['root'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    import psutil
                    rt.write(target,dict(status='running',pid=current.pid,process_created=psutil.Process(current.pid).create_time(),
                                         attempt=attempt,stage=stage,log=str(log),utc=rt.utc()))
                    stalled=False
                    while current.poll() is None:
                        rt.checkpoint(); time.sleep(2)
                        progress=rt.read(rt.OUT/'progress.json')
                        if time.time()-progress['epoch']>cfg['stage_timeout_seconds']:
                            stalled=True; terminate_group(current); break
                    code=current.wait()
                    terminate_group(current)
                rt.event('stage_exit',stage=stage,attempt=attempt,exit_code=code,stalled=stalled)
                if code==0:
                    rt.write(target,dict(status='complete',stage=stage,utc=rt.utc())); success=True; break
                rt.write(target,dict(status='failed_attempt',stage=stage,attempt=attempt,exit_code=code,stalled=stalled,log=str(log),utc=rt.utc()))
                if attempt<cfg['retries_per_stage']: time.sleep(10)
            if not success: failed.append(stage)
        status='COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES'
    except rt.StopRun:
        status='DEADLINE_REACHED' if time.time()>=session['deadline_epoch'] else 'STOPPED'
    except BaseException:
        status='FAILED'; traceback.print_exc()
    finally:
        if current is not None: terminate_group(current)
        result=dict(status=status,failed_stages=failed,utc=rt.utc(),elapsed_hours=(time.time()-session['started_epoch'])/3600)
        rt.write(rt.OUT/'result.json',result); write_report(result)
        rt.event('queue_finished',**result)
        lock.close()


def write_report(result):
    lines=['# W71离线执行结果','',f'状态：{result["status"]}；已用{result["elapsed_hours"]:.2f}小时。','',
           '固定实验队列；没有根据成绩修改方法或增加任务。失败/未封存部分不作完整科学结论。','']
    mp=rt.OUT/'mechanism/summary.json'
    if (rt.OUT/'mechanism/scored.json').exists():
        summary=rt.read(mp); lines+=['## 血管机制对照','',f'完整任务：{summary["n"]}。效应为当前处理减去单因素对照，单位pp；以下为主层23的同时区间。','',
                                   '|来源|因素|效应|照片同时区间|双角色同时区间|','|---|---|---:|---|---|']
        for name,cohort in summary['cohorts'].items():
            for factor,v in cohort['single_factor_effects']['23'].items():
                ci=lambda x:', '.join(f'{a:+.3f}' for a in x)
                lines.append(f'|{name}|{factor}|{v["current_minus_control_pp"]:+.3f}|{ci(v["family_ci_photo"])}|{ci(v["family_ci_image2way"])}|')
        lines+=['','这些是SH回退的因素对照，不是FoRIS总差距的百分比分摊；不选最佳组合。','']
    else: lines+=['血管机制任务未完整评分；检查stages与logs。','']
    lines+=['## 官方确认','', '|域|已有记录|总数|封存|本次评分完成|','|---|---:|---:|---|---|']
    manifest=rt.read(rt.OUT/'manifest.json')
    for d,rows in manifest['official'].items():
        n=len(list((W30/'records'/d).glob('*.json')))
        lines.append(f'|{d}|{n}|{len(rows)}|{(W30/"sealed"/f"{d}.json").exists()}|{(rt.OUT/"official_scores"/f"{d}_complete.json").exists()}|')
    scores=W30/'scores/E2_summary.json'
    if scores.exists():
        for d,value in rt.read(scores).items():
            if d in DOMAINS and (rt.OUT/'official_scores'/f'{d}_complete.json').exists():
                lines+=['',f'{d}: SH+={value["official_miou"]["SH+"]:.3f}; 照片区间={value["ci95"]["SH+"]}; 双角色区间={value["ci95_image2way"]["SH+"]}。']
    lines+=['','详细记录：mechanism/summary.json、official_scores/、stages/、logs/。']
    (rt.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def start():
    rt.offline(); cfg=verify_package(inputs=True)
    ready=rt.read(rt.OUT/'ready.json'); assert ready['status']=='PASS' and ready['package_sha256']==rt.sha(rt.OUT/'package.json'),'Run check for this package first'
    lock=(rt.OUT/'launch.lock').open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import psutil
    if (rt.OUT/'result.json').exists() and rt.read(rt.OUT/'result.json')['status']=='COMPLETE':
        print('Fixed queue already complete; no repeat experiment started.'); return
    pidfile=rt.OUT/'supervisor.json'
    if pidfile.exists():
        old=rt.read(pidfile)
        if psutil.pid_exists(old['pid']) and abs(psutil.Process(old['pid']).create_time()-old['created'])<1:
            print('Already running; duplicate launch refused.',old['pid']); return
    session=rt.OUT/'session.json'
    if not session.exists():
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now=time.time(); zone=ZoneInfo('America/New_York')
        rt.write(session,dict(started_epoch=now,deadline_epoch=now+48*3600,utc=rt.utc(),
                 started_new_york=datetime.fromtimestamp(now,zone).isoformat(),
                 deadline_new_york=datetime.fromtimestamp(now+48*3600,zone).isoformat()))
    rt.checkpoint()  # Restarts never reset the 48-hour allowance.
    log=(rt.OUT/'supervisor.log').open('ab')
    env=dict(os.environ,CV_RESEARCH_ROOT=cfg['root'],PYTHONDONTWRITEBYTECODE='1')
    cmd=['/usr/bin/caffeinate','-is',str(rt.PYTHON),'-B',str(Path(cfg['root'])/'scripts/x1012/w71_offline.py'),'supervise']
    process=subprocess.Popen(cmd,cwd=cfg['root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rt.write(pidfile,dict(pid=process.pid,created=psutil.Process(process.pid).create_time(),utc=rt.utc()))
    print('Started local offline queue. Deadline (New York):',rt.read(session)['deadline_new_york'])
    print('Status:',rt.OUT/'progress.json','Report:',rt.OUT/'REPORT.md')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','check','check-worker','start','resume','status','stop','supervise','worker'))
    parser.add_argument('stage',nargs='?'); args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='check':
        cfg=verify_package(inputs=True)
        rt.write(rt.OUT/'ready.json',dict(status='CHECKING',package_sha256=rt.sha(rt.OUT/'package.json'),utc=rt.utc()))
        env=dict(os.environ,CV_RESEARCH_ROOT=cfg['root'])
        code=subprocess.call([str(rt.PYTHON),'-B',str(Path(cfg['root'])/'scripts/x1012/w71_offline.py'),'check-worker'],cwd=cfg['root'],env=env)
        if code:
            rt.write(rt.OUT/'ready.json',dict(status='FAILED',package_sha256=rt.sha(rt.OUT/'package.json'),exit_code=code,utc=rt.utc()))
        raise SystemExit(code)
    elif args.command=='check-worker':
        try: check_worker()
        except BaseException as exc:
            rt.write(rt.OUT/'check'/rt.sha(rt.OUT/'package.json')[:16]/'failed.json',
                     dict(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),utc=rt.utc()))
            raise
    elif args.command=='start': start()
    elif args.command=='resume':
        rt.write(rt.OUT/'STOP_REQUESTED',dict(active=False,utc=rt.utc()))
        start()
    elif args.command=='supervise': supervise()
    elif args.command=='stop':
        rt.write(rt.OUT/'STOP_REQUESTED',dict(active=True,utc=rt.utc()))
        print('Stop requested; completed records will be retained. Resume does not extend the deadline.')
    elif args.command=='status':
        for name in ('session.json','progress.json','result.json'):
            p=rt.OUT/name
            if p.exists(): print(name,json.dumps(rt.read(p),ensure_ascii=False))
        if not (rt.OUT/'session.json').exists(): print('NOT STARTED: no 48-hour deadline has been created.')
    elif args.command=='worker':
        rt.offline(); verify_package(); rt.heartbeat(args.stage)
        if args.stage=='mechanism-predict':
            import w71_mechanism as m; m.produce()
        elif args.stage=='mechanism-score':
            import w71_mechanism as m; m.score()
        elif args.stage.startswith('official-predict:'): official_predict(args.stage.split(':')[1])
        elif args.stage.startswith('official-score:'): official_score(args.stage.split(':')[1])
        else: raise ValueError(args.stage)


if __name__=='__main__': main()
