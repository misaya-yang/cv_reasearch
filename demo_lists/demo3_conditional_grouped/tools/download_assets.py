#!/usr/bin/env python3
"""Public asset ranges into one resumable sparse file; no Torch/GPU imports."""
import argparse, concurrent.futures, hashlib, json, os, pathlib, threading, time, urllib.request

def fetch(a, root, workers):
    path=root/a['path'];path.parent.mkdir(parents=True,exist_ok=True)
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return h.hexdigest()
    if path.exists():
        if path.stat().st_size!=a['bytes'] or sha(path)!=a['sha256']:raise ValueError('Existing asset mismatch: '+a['path'])
        return
    partial=path.with_name(path.name+'.partial');state=path.with_name(path.name+'.ranges.json')
    step=8*1024*1024; n=(a['bytes']+step-1)//step
    saved=json.loads(state.read_text()) if state.exists() else {'sha256':a['sha256'],'bytes':a['bytes'],'chunk_bytes':step,'complete':[]}
    assert (saved['sha256'],saved['bytes'],saved['chunk_bytes'])==(a['sha256'],a['bytes'],step)
    if saved['complete'] and not partial.exists():raise ValueError('Range state exists but partial file is missing')
    done=set(saved['complete']);lock=threading.Lock()
    fd=os.open(partial,os.O_RDWR|os.O_CREAT,0o600);os.ftruncate(fd,a['bytes'])
    def part(i):
        lo=i*step;hi=min(a['bytes'],lo+step)-1
        for attempt in range(5):
            try:
                req=urllib.request.Request(a['resolved_url'],headers={'Range':f'bytes={lo}-{hi}','Accept-Encoding':'identity'})
                with urllib.request.urlopen(req,timeout=40) as r:
                    whole=(lo==0 and hi==a['bytes']-1)
                    if not (r.status==200 and whole) and (r.status!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{a["bytes"]}'):raise ValueError('Range contract rejected')
                    pos=lo
                    while pos<=hi:
                        b=r.read(min(65536,hi-pos+1))
                        if not b:raise EOFError('Short range')
                        offset=0
                        while offset<len(b):offset+=os.pwrite(fd,b[offset:],pos+offset)
                        pos+=len(b)
                with lock:
                    os.fsync(fd)
                    done.add(i);saved['complete']=sorted(done)
                    temp=state.with_suffix('.tmp');temp.write_text(json.dumps(saved));temp.replace(state)
                    if len(done)%8==0 or len(done)==n:print(json.dumps({'path':a['path'],'chunks':len(done),'total':n}),flush=True)
                return
            except Exception as e:
                if attempt==4:raise RuntimeError(f'Resumable range failure {a["path"]} chunk {i}: {type(e).__name__}') from None
                time.sleep(attempt+1)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(part,[i for i in range(n) if i not in done]))
        os.fsync(fd)
    finally:os.close(fd)
    if sha(partial)!=a['sha256']:raise ValueError('Final SHA mismatch; partial retained')
    partial.replace(path);print(json.dumps({'path':a['path'],'status':'SHA_VERIFIED'}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=pathlib.Path,required=True);p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    assets=json.loads(args.manifest.read_text())
    for a in assets:
        rel=pathlib.PurePosixPath(a['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid asset path')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fetch,a,args.root,args.workers) for a in assets]
        for f in futures:f.result()
if __name__=='__main__':main()
