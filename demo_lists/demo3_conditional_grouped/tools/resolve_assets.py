import pathlib,urllib.request,json,hashlib,time
out=[];public=[]
for repo,folder,names in [('MCG-NJU/PixNerd-XXL-P16-T2I','pixnerd',{'model.ckpt'}),('Qwen/Qwen3-1.7B','qwen3',None)]:
 d=json.loads(pathlib.Path('assets/manifests/'+folder+'_source.json').read_text());rev=d['sha']
 for x in d['siblings']:
  name=x['rfilename']
  if names is not None and name not in names:continue
  if names is None and not name.endswith(('.json','.safetensors','.txt','.jinja')):continue
  url='https://huggingface.co/'+repo+'/resolve/'+rev+'/'+name
  if not x.get('lfs'):
   target=pathlib.Path('assets/checkpoints')/folder/name;target.parent.mkdir(parents=True,exist_ok=True)
   for attempt in range(5):
    try:
     if not target.exists():
      with urllib.request.urlopen(url,timeout=60) as r,target.with_suffix(target.suffix+'.tmp').open('wb') as f:
       while True:
        b=r.read(65536)
        if not b:break
        f.write(b)
      target.with_suffix(target.suffix+'.tmp').replace(target)
     data=target.read_bytes()
     if len(data)!=x['size']:raise ValueError('Short metadata')
     blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
     if x.get('blobId') and blob!=x['blobId']:raise ValueError('Git blob mismatch')
     public.append({'path':folder+'/'+name,'bytes':len(data),'url':url,'sha256':hashlib.sha256(data).hexdigest(),'local_small_asset':True});break
    except Exception:
     if target.exists():target.unlink()
     if attempt==4:raise
     time.sleep(2)
   continue
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=60) as r:resolved=r.geturl()
  a={'path':folder+'/'+name,'bytes':x['size'],'sha256':x['lfs']['sha256'],'url':url,'resolved_url':resolved};out.append(a);public.append({k:v for k,v in a.items() if k!='resolved_url'})
  print(a['path'],a['bytes'],flush=True)
p=pathlib.Path('assets/manifests/download.private.json');p.write_text(json.dumps(out,indent=2));p.chmod(0o600)
pathlib.Path('assets/manifests/download_public.json').write_text(json.dumps(public,indent=2));print('Total',sum(a['bytes'] for a in public),flush=True)
