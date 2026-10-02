"""Select fresh official COCO val image IDs before looking at method scores."""
import argparse,hashlib,json,pathlib,zipfile
def main():
 p=argparse.ArgumentParser();p.add_argument('--coco',type=pathlib.Path,default=pathlib.Path('/root/autodl-pub/COCO2017'));p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('assets/data/coco_unseen_v1'));p.add_argument('--exclude',type=pathlib.Path,default=pathlib.Path('assets/data/coco_pilot_v1/manifest.json'));a=p.parse_args()
 old=json.loads(a.exclude.read_text());excluded={r['image_id'] for r in old['rows']};a.out.mkdir(parents=True,exist_ok=True)
 demo=pathlib.Path('/root/autodl-tmp/demo1_sam/assets/coco_quality_seed2027_v1/manifest.json')
 if demo.exists():excluded.update(r['image_id'] for r in json.loads(demo.read_text()).get('images',[]) if isinstance(r,dict) and 'image_id' in r)
 with zipfile.ZipFile(a.coco/'annotations_trainval2017.zip') as z:d=json.loads(z.read('annotations/captions_val2017.json'))
 caps={}
 for c in sorted(d['annotations'],key=lambda r:r['id']):caps.setdefault(c['image_id'],c)
 images=sorted((i for i in d['images'] if i['id'] not in excluded and i['id'] in caps),key=lambda i:hashlib.sha256(f'cgd-unseen-v1:{i["id"]}'.encode()).digest())[:64]
 rows=[]
 with zipfile.ZipFile(a.coco/'val2017.zip') as z:
  for im in images:
   data=z.read('val2017/'+im['file_name']);dst=a.out/'val'/im['file_name'];dst.parent.mkdir(exist_ok=True)
   if dst.exists() and dst.read_bytes()!=data:raise ValueError('Existing data differs')
   dst.write_bytes(data);c=caps[im['id']];rows.append({'split':'val','image_id':im['id'],'path':str(dst.relative_to(a.out)),'caption_id':c['id'],'caption':c['caption'],'sha256':hashlib.sha256(data).hexdigest(),'native_width':im['width'],'native_height':im['height']})
 assert not excluded.intersection(r['image_id'] for r in rows)
 manifest={'rows':rows,'excluded_manifest_sha256':hashlib.sha256(a.exclude.read_bytes()).hexdigest(),'selection':'SHA256 cgd-unseen-v1 image ID, lowest caption ID; selected before scoring','limits':'Official COCO val, usually upsampled to1024; independent of pilot selection, not native1024 quality/SOTA.'}
 target=a.out/'manifest.json';text=json.dumps(manifest,indent=2)+'\n'
 if target.exists() and target.read_text()!=text:raise ValueError('Refuse to overwrite a different split')
 target.write_text(text);print('Prepared',len(rows),'fresh val images; manifest SHA256',hashlib.sha256(text.encode()).hexdigest())
if __name__=='__main__':main()
