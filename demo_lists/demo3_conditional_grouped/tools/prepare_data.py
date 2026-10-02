import argparse,hashlib,json,pathlib,zipfile
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('--coco',type=pathlib.Path,default=pathlib.Path('/root/autodl-pub/COCO2017'));p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('assets/data/coco_pilot_v1'));p.add_argument('--train',type=int,default=256);p.add_argument('--val',type=int,default=64);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 exclusion=pathlib.Path('/root/autodl-tmp/demo1_sam/assets/coco_quality_seed2027_v1/manifest.json');excluded=set()
 if exclusion.exists():
  d=json.loads(exclusion.read_text());excluded={x['image_id'] for x in d.get('images',[]) if isinstance(x,dict) and 'image_id' in x}
 rows=[]
 with zipfile.ZipFile(a.coco/'annotations_trainval2017.zip') as annotations:
  for split,n in [('train',a.train),('val',a.val)]:
   d=json.loads(annotations.read('annotations/captions_'+split+'2017.json'));caps={}
   for c in sorted(d['annotations'],key=lambda c:c['id']):caps.setdefault(c['image_id'],c)
   imgs=[im for im in d['images'] if im['id'] in caps and im['id'] not in excluded]
   imgs.sort(key=lambda im:hashlib.sha256(f'cgd2027:{split}:{im["id"]}'.encode()).digest())
   with zipfile.ZipFile(a.coco/(split+'2017.zip')) as z:
    for im in imgs[:n]:
     data=z.read(split+'2017/'+im['file_name']);dst=a.out/split/im['file_name'];dst.parent.mkdir(exist_ok=True)
     if dst.exists() and dst.read_bytes()!=data:raise ValueError('Existing image differs')
     dst.write_bytes(data);c=caps[im['id']]
     rows.append({'split':split,'image_id':im['id'],'path':str(dst.relative_to(a.out)),'caption_id':c['id'],'caption':c['caption'],'sha256':hashlib.sha256(data).hexdigest(),'native_width':im['width'],'native_height':im['height']})
 assert not ({r['image_id'] for r in rows if r['split']=='train'} & {r['image_id'] for r in rows if r['split']=='val'})
 out={'seed':2027,'rows':rows,'protocol':'Official Resize(1024) bilinear + CenterCrop(1024), RGB /127.5-1; all methods share input/noise/time/caption.','limits':'COCO usually has lower native resolution: this is an upsampled pilot, not proof on native 1024 detail. Official train/val image split; excludes demo1 image IDs where manifest available.'}
 (a.out/'manifest.json').write_text(json.dumps(out,indent=2)+'\n');print('Prepared',len(rows),'image/caption pairs')
if __name__=='__main__':main()
