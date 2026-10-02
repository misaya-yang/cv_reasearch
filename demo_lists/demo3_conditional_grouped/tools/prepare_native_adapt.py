"""Small mixed-domain train set and fresh native validation, selected before scoring."""
import hashlib,io,json,os,pathlib,zipfile
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'assets/data/native_adapt_v1';out.mkdir(parents=True,exist_ok=True);(out/'text_cache').mkdir(exist_ok=True)
 if os.statvfs(out).f_bavail*os.statvfs(out).f_frsize<6500000000:raise RuntimeError('Insufficient reserve')
 old=json.loads((ROOT/'assets/data/native_probe_v1/manifest.json').read_text());exclude={r['source'] for r in old['rows']};rows=[]
 common=ROOT/'assets/data/native_probe_v1/text_cache/empty_common.pt'
 for split,count,archive in [('train',64,'DIV2K_train_HR.zip'),('val',32,'DIV2K_valid_HR.zip')]:
  with zipfile.ZipFile(pathlib.Path('/root/autodl-pub/DIV2K/HighResolution')/archive) as z:
   names=sorted((n for n in z.namelist() if n.endswith('.png') and n not in exclude),key=lambda n:hashlib.sha256(f'cgd-native-adapt-v1:{split}:{n}'.encode()).digest());taken=0
   for name in names:
    data=z.read(name)  # validates CRC of each selected source; no repeated full-archive scan
    with Image.open(io.BytesIO(data)) as im:
     if min(im.size)<1024:continue
     w,h=im.size;left=(w-1024)//2;top=(h-1024)//2;crop=im.convert('RGB').crop((left,top,left+1024,top+1024));folder=out/(split+'_native');folder.mkdir(exist_ok=True);dst=folder/pathlib.Path(name).name
     if not dst.exists():crop.save(dst)
     else:
      with Image.open(dst) as previous:
       if previous.tobytes()!=crop.tobytes():raise ValueError('Existing crop differs')
    uid=(2000000000 if split=='train' else 1000000000)+int(pathlib.Path(name).stem);cache=out/'text_cache'/f'{uid}.pt'
    if not cache.exists():os.link(common,cache)
    rows.append({'split':split,'dataset':'DIV2K_native','image_id':uid,'path':str(dst.relative_to(out)),'caption':'','source':name,'source_archive':archive,'source_png_sha256':hashlib.sha256(data).hexdigest(),'native_width':w,'native_height':h,'crop_box':[left,top,left+1024,top+1024],'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()});taken+=1
    if taken==count:break
   assert taken==count
 coco=ROOT/'assets/data/coco_pilot_v1';manifest=json.loads((coco/'manifest.json').read_text());folder=out/'coco_replay';folder.mkdir(exist_ok=True)
 for split,count in [('train',31),('val',16)]:
  candidates=sorted((r for r in manifest['rows'] if r['split']==split),key=lambda r:hashlib.sha256(f'cgd-native-replay-v1:{r["image_id"]}'.encode()).digest())[:count]
  for original in candidates:
   source=coco/original['path'];dst=folder/source.name
   if not dst.exists():dst.symlink_to(source)
   if dst.resolve()!=source.resolve():raise ValueError('Conflicting replay link')
   cache=out/'text_cache'/f'{original["image_id"]}.pt'
   if not cache.exists():os.link(coco/'text_cache'/cache.name,cache)
   row=dict(original);row.update(dataset='COCO_replay_dev',path=str(dst.relative_to(out)));rows.append(row)
 rows.sort(key=lambda r:(r['split'],hashlib.sha256(f'cgd-mixed-order-v1:{r["image_id"]}'.encode()).digest()))
 train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val'];assert len(train)==95 and len(val)==48
 assert not {r['image_id'] for r in train}.intersection(r['image_id'] for r in val)
 assert not {r['image_id'] for r in old['rows']}.intersection(r['image_id'] for r in val if r['dataset']=='DIV2K_native')
 d={'rows':rows,'selection':'SHA256 filename/ID before scoring; native64 train + COCO31 replay; fresh native32 val excludes previous native16; COCO16 dev sentinel','protocol':'Native pixels cropped1024 without resizing; official COCO Resize1024+center crop; same noise/time/weights/FP32 across arms','reason_for_train_count95':'Not divisible by3: images move across stratified-time bands on repeated traversals','limits':'Matched small native-domain decoder adaptation, empty text for native photos. New native32 are heldout; COCO16 already development-used. Not standard denoising/SR or T2I SOTA.'}
 target=out/'manifest.json';text=json.dumps(d,indent=2)+'\n'
 if target.exists() and target.read_text()!=text:raise ValueError('Refuse to change fixed manifest')
 target.write_text(text);print('Prepared95 train /48 val; SHA256',hashlib.sha256(text.encode()).hexdigest(),flush=True)
if __name__=='__main__':main()
