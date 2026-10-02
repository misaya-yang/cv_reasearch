"""User-authorized removal of redundant archives after exact extracted CRC checks."""
import pathlib,zipfile,zlib,hashlib,json
root=pathlib.Path('/root/autodl-tmp/demo1_sam/assets');records=[]
for archive,parent,chosen in [(root/'datasets/archives/DAVIS-2017-trainval-480p.zip',root/'datasets',None),(root/'checkpoints/efficient_sam_vits.pt.zip',root/'checkpoints',{'efficient_sam_vits.pt'})]:
 if not archive.exists():continue
 count=0
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():
   if info.is_dir() or info.filename.startswith('__MACOSX/'):continue
   if chosen is not None and info.filename not in chosen:continue
   f=parent/info.filename
   if not f.exists() or f.stat().st_size!=info.file_size:raise RuntimeError('Retained extracted asset missing/different')
   crc=0
   with f.open('rb') as stream:
    for b in iter(lambda:stream.read(1024*1024),b''):crc=zlib.crc32(b,crc)
   if crc&0xffffffff!=info.CRC:raise RuntimeError('Retained extracted asset CRC mismatch')
   count+=1
 if not count:raise RuntimeError('No verified members')
 size=archive.stat().st_size;h=hashlib.sha256()
 with archive.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 records.append({'removed_archive':str(archive),'sha256':h.hexdigest(),'verified_retained_files':count,'retained_root':str(parent),'freed_bytes':size});archive.unlink()
report={'scope':'Redundant archive copies; extracted bytes CRC checked before removal','entries':records,'freed_bytes':sum(v['freed_bytes'] for v in records)}
(root/'manifests/archive_dedup_20261001.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
