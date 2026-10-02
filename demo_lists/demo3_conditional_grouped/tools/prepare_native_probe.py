"""Predeclare16 native1024 crops; shared official archive, empty native Qwen conditioning."""
import argparse,hashlib,io,json,os,pathlib,zipfile
from PIL import Image
import torch
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=pathlib.Path,default=pathlib.Path('/root/autodl-pub/DIV2K/HighResolution/DIV2K_valid_HR.zip'));p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('assets/data/native_probe_v1'));a=p.parse_args()
 if os.statvfs(a.out.parent).f_bavail*os.statvfs(a.out.parent).f_frsize<6000000000:raise RuntimeError('Keep data-disk reserve >=5GB')
 a.out.mkdir(parents=True,exist_ok=True);(a.out/'val').mkdir(exist_ok=True);cache=a.out/'text_cache';cache.mkdir(exist_ok=True)
 empty=torch.load('assets/data/coco_pilot_v1/text_cache/empty_caption.pt',weights_only=True,map_location='cpu');assert empty['caption']==''
 common=cache/'empty_common.pt'
 if not common.exists():torch.save({'embedding':empty['embedding'].clone(),'caption_sha256':hashlib.sha256(b'').hexdigest(),'source':'Official Qwen3 empty-caption cache used by the vendor sampler'},common)
 rows=[];rejected=[]
 with zipfile.ZipFile(a.archive) as z:
  bad=z.testzip()
  if bad:raise ValueError(f'Archive CRC failure: {bad}')
  names=sorted((n for n in z.namelist() if n.endswith('.png')),key=lambda n:hashlib.sha256(('cgd-native-v1:'+n).encode()).digest())
  for name in names:
   source=z.read(name)
   with Image.open(io.BytesIO(source)) as im:
    w,h=im.size
    if min(w,h)<1024:rejected.append(name);continue
    left=(w-1024)//2;top=(h-1024)//2;crop=im.convert('RGB').crop((left,top,left+1024,top+1024));dst=a.out/'val'/pathlib.Path(name).name
    if dst.exists():
     with Image.open(dst) as old:
      if old.tobytes()!=crop.tobytes():raise ValueError('Existing crop differs')
    else:crop.save(dst)
   image_id=1000000000+int(pathlib.Path(name).stem);link=cache/f'{image_id}.pt'
   if not link.exists():os.link(common,link)
   rows.append({'split':'val','image_id':image_id,'path':str(dst.relative_to(a.out)),'caption':'','source':name,'native_width':w,'native_height':h,'crop_box':[left,top,left+1024,top+1024],'source_png_sha256':hashlib.sha256(source).hexdigest(),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
   if len(rows)==16:break
 assert len(rows)==16
 manifest={'rows':rows,'archive':str(a.archive),'archive_bytes':a.archive.stat().st_size,'archive_crc':'PASS_ALL_ENTRIES','source':'https://data.vision.ee.ethz.ch/cvl/DIV2K/','license':'Academic research only; cite Agustsson and Timofte CVPRW2017 plus NTIRE2017 report. Original image rights retained by owners.','selection':'SHA256 cgd-native-v1 filename; first16 with both native dimensions >=1024; center crop without resizing','skipped_small_images':rejected,'limits':'Frozen-weight native-photo velocity/reconstruction diagnostic with identical empty text. Not super-resolution evaluation or T2I generation/SOTA.'}
 target=a.out/'manifest.json';text=json.dumps(manifest,indent=2)+'\n'
 if target.exists() and target.read_text()!=text:raise ValueError('Refuse to replace a different manifest')
 target.write_text(text)
 # Remove only our aborted duplicate download, after the canonical ZIP CRC passes.
 partial=pathlib.Path('assets/data_archives/DIV2K_valid_HR.zip.partial');freed=0
 if partial.exists() and partial.stat().st_size<a.archive.stat().st_size:
  with partial.open('rb') as f,a.archive.open('rb') as g:
   if f.read(524288)!=g.read(524288):raise ValueError('Partial download differs from shared archive')
  freed=partial.stat().st_size;partial.unlink()
 pathlib.Path('results/native_asset_receipt.json').write_text(json.dumps({'shared_archive':str(a.archive),'crc':'PASS_ALL_ENTRIES','new_download_canceled_as_redundant':True,'own_partial_removed_bytes':freed,'native_crops':len(rows),'manifest_sha256':hashlib.sha256(text.encode()).hexdigest()},indent=2))
 print('Prepared16 native1024 crops; SHA256',hashlib.sha256(text.encode()).hexdigest(),flush=True)
if __name__=='__main__':main()
