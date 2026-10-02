"""Bounded author precomputed correspondences; no image/weight downloads."""
import argparse,json,time,urllib.request
from pathlib import Path


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    targets=[('scannet1500_roma.h5',322764856),('megadepth1500_roma.h5',332175160)]
    for name,size in targets:
        final=root/name;part=final.with_suffix('.part')
        if final.exists():
            if final.stat().st_size!=size:raise RuntimeError('Existing completed cache has unexpected size')
            continue
        url='https://vision.maths.lth.se/viktor/posebench/relative/'+name
        for attempt in range(3):
            try:
                offset=part.stat().st_size if part.exists() else 0
                if offset>size:raise RuntimeError('Partial file exceeds bounded input size')
                headers={'Range':f'bytes={offset}-'} if offset else {}
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as response:
                    resume=offset>0 and response.status==206
                    if offset and not resume:offset=0
                    with part.open('ab' if resume else 'wb') as f:
                        count=offset
                        while True:
                            block=response.read(1024*1024)
                            if not block:break
                            count+=len(block)
                            if count>size:raise RuntimeError('Input exceeds declared byte budget')
                            f.write(block)
                            if count//(16*1024*1024)!=(count-len(block))//(16*1024*1024):
                                (root/'download_status.json').write_text(json.dumps(dict(state='DOWNLOADING',name=name,bytes=count,total=size)))
                if part.stat().st_size!=size:raise RuntimeError('Incomplete transfer')
                part.replace(final);break
            except Exception as exc:
                (root/'download_error_attempt.json').write_text(json.dumps(dict(name=name,attempt=attempt,error=str(exc))))
                if attempt==2:raise
                time.sleep(2*(attempt+1))
        print(json.dumps(dict(completed=name,bytes=size,source=url)),flush=True)
    (root/'input_manifest.json').write_text(json.dumps(dict(state='COMPLETED',inputs=targets,total_bytes=sum(x[1] for x in targets),
          source='official DMS author correspondence caches; original RoMa, not RoMaV2',images_or_weights=False),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);run(ap.parse_args().root)
