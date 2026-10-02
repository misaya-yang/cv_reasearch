"""Lossless ZIP compression and verified logical deduplication of completed NPZs.

Array member bytes, names and ordering are preserved. Changes are atomic and
recorded; active/newly modified files and non-NPZ artifacts are left alone.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def compact(path):
    stat=path.stat()
    if time.time()-stat.st_mtime<120:return {'path':str(path),'status':'SKIP_RECENT'}
    with zipfile.ZipFile(path) as source:
        infos=source.infolist()
        if not any(i.compress_type==zipfile.ZIP_STORED for i in infos):return {'path':str(path),'status':'ALREADY_COMPRESSED'}
        if len({i.filename for i in infos})!=len(infos):raise ValueError('duplicate ZIP members')
        original_hash=digest(path)
        temp=path.with_name(path.name+'.compact.tmp')
        if temp.exists():raise FileExistsError(temp)
        hashes={}
        try:
            with zipfile.ZipFile(temp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as target:
                for info in infos:
                    # Keep .npy payload, filename, timestamps and member ordering.
                    entry=zipfile.ZipInfo(info.filename,info.date_time);entry.compress_type=zipfile.ZIP_DEFLATED
                    entry.external_attr=info.external_attr
                    h=hashlib.sha256()
                    with source.open(info) as inp,target.open(entry,'w',force_zip64=True) as out:
                        for chunk in iter(lambda:inp.read(8<<20),b''):h.update(chunk);out.write(chunk)
                    hashes[info.filename]=h.hexdigest()
            with zipfile.ZipFile(temp) as check:
                if [(i.filename,i.file_size,i.CRC) for i in check.infolist()]!=[(i.filename,i.file_size,i.CRC) for i in infos]:raise ValueError('member contract changed')
                for info in check.infolist():
                    h=hashlib.sha256()
                    with check.open(info) as f:
                        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
                    if h.hexdigest()!=hashes[info.filename]:raise ValueError('array payload mismatch')
            current=path.stat()
            if (current.st_size,current.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):raise RuntimeError('original changed during compaction')
            if temp.stat().st_size>=stat.st_size:
                temp.unlink();return {'path':str(path),'status':'NO_SAVING'}
            after_hash=digest(temp);after_bytes=temp.stat().st_size
            with temp.open('rb') as f:os.fsync(f.fileno())
            os.replace(temp,path)
            return {'path':str(path),'status':'COMPRESSED_VERIFIED','before_bytes':stat.st_size,'after_bytes':after_bytes,'freed_bytes':stat.st_size-after_bytes,'before_sha256':original_hash,'after_sha256':after_hash,'member_sha256':hashes,'array_payloads_identical':True}
        finally:
            if temp.exists():temp.unlink()


def logical_signature(path):
    with zipfile.ZipFile(path) as z:return tuple(sorted((i.filename,i.file_size,i.CRC) for i in z.infolist()))


def same_arrays(a,b):
    with zipfile.ZipFile(a) as x,zipfile.ZipFile(b) as y:
        for name in x.namelist():
            with x.open(name) as xx,y.open(name) as yy:
                while True:
                    bx=xx.read(8<<20);by=yy.read(8<<20)
                    if bx!=by:return False
                    if not bx:break
    return True


def main(a):
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    files=sorted(a.root.rglob('*.npz'),key=lambda p:p.stat().st_size,reverse=True)
    rows=[];errors=[];started=time.time()
    with a.receipt.open('x') as log:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(compact,p):p for p in files}
            for n,future in enumerate(as_completed(futures),1):
                try:row=future.result();rows.append(row)
                except Exception as e:row={'path':str(futures[future]),'status':'ERROR','error':str(e)};errors.append(row)
                log.write(json.dumps(row)+'\n');log.flush()
                if n%20==0:print(json.dumps({'processed':n,'files':len(files),'freed_GiB':sum(r.get('freed_bytes',0) for r in rows)/2**30,'errors':len(errors)}),flush=True)
        # Same members are compared byte-for-byte before physical deduplication.
        buckets={};dedup_freed=0
        eligible={r['path'] for r in rows if r.get('status') in ('COMPRESSED_VERIFIED','ALREADY_COMPRESSED','NO_SAVING')}
        for path in files:
            if str(path) not in eligible:continue
            key=logical_signature(path)
            if key not in buckets:buckets[key]=path;continue
            canonical=buckets[key]
            if path.stat().st_ino==canonical.stat().st_ino:continue
            if same_arrays(canonical,path):
                size=path.stat().st_size;before=digest(path);tmp=path.with_name(path.name+'.dedup.tmp')
                os.link(canonical,tmp);os.replace(tmp,path);dedup_freed+=size
                row={'status':'DEDUPLICATED_VERIFIED','path':str(path),'canonical':str(canonical),'freed_bytes':size,'before_sha256':before,'after_sha256':digest(canonical),'same_array_payloads':True};log.write(json.dumps(row)+'\n');log.flush()
    summary={'status':'COMPLETED' if not errors else 'COMPLETED_WITH_ERRORS','files':len(files),'compressed':sum(r.get('status')=='COMPRESSED_VERIFIED' for r in rows),'compression_freed_bytes':sum(r.get('freed_bytes',0) for r in rows),'dedup_freed_bytes':dedup_freed,'errors':errors,'wall_seconds':time.time()-started,'important_results_deleted':False}
    a.receipt.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--workers',type=int,default=2);main(p.parse_args())
