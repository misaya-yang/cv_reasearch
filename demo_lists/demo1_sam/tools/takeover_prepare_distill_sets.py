"""Disjoint COCO test images and official DAVIS validation first frames."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/quality_mechanisms'))
from prepare_coco_subset import prepare,prompts_from_mask


def main():
    out=ROOT/'results/takeover_20261001_v1/distill_v1'
    excluded=set()
    for name in ('coco_quality_seed2027_v1','coco_flip_validation_seed2028_v1','coco_prompt_holdout_seed2029_v1'):
        m=json.loads((ROOT/'assets'/name/'manifest.json').read_text());excluded.update(x['image_id'] for x in m['images'])
    assert len(excluded)==408
    protocol={'status':'FIXED_BEFORE_NEW_STUDENT_TEST_RESULTS','train':'previous256 cohort, relabeled as training for this new learned method','validation':'previous128 cohort, development only','test':'new512 seed2030, exclude all prior408 images','external_test':'official DAVIS2017 val split, one first annotated frame per sequence; no DAVIS training labels',
              'target':'distill prompt-response consistency into a single original-decoder selection head','controls':['largest original mask','original IoU head','fixed mask-token choices','equal-architecture GT-supervised head','geometry-only head'],
              'inference_GT':False,'student_epoch_selection':'teacher cross-entropy for distilled models; validation meanIoU for supervised controls','primary_endpoint':'new COCO central IoU and paired delta against largest mask','secondary':['near-boundary IoU','box control','BoundaryIoU','DAVIS first-frame generalization','complete inference cost'],'test_output_metrics_not_for_tuning':True}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    target=ROOT/'assets/coco_distill_test_seed2030_v1'
    if target.exists():raise FileExistsError(target)
    m=prepare(Path('/root/autodl-pub/COCO2017/annotations_trainval2017.zip'),Path('/root/autodl-pub/COCO2017/val2017.zip'),target,2030,512,4,1024,excluded_image_ids=excluded)
    assert not excluded.intersection(x['image_id'] for x in m['images'])
    print(json.dumps({'COCO_images':512,'objects':m['num_instances']}),flush=True)
    davis=ROOT/'assets/datasets/DAVIS';dest=ROOT/'assets/davis_val_firstframe_v1';dest.mkdir(exist_ok=False);(dest/'ground_truth').mkdir()
    sequences=[x.strip() for x in (davis/'ImageSets/2017/val.txt').read_text().splitlines() if x.strip()]
    images=[];rng=np.random.default_rng(2030)
    for index,seq in enumerate(sequences):
        ann=sorted((davis/'Annotations/480p'/seq).glob('*.png'))[0];jpg=davis/'JPEGImages/480p'/seq/(ann.stem+'.jpg')
        labels=np.array(Image.open(ann));h,w=labels.shape;objects=[];masks=[]
        for obj in sorted(set(np.unique(labels))-{0,255}):
            mask=labels==obj;central,near,box,depth=prompts_from_mask(mask,rng);aid=index*100+int(obj)
            masks.append(mask);objects.append({'annotation_id':aid,'central_xy':central,'near_boundary_xy':near,'tight_box_xyxy':box,'gt_index':len(masks)-1})
        image_id=100000000+index
        gp=dest/'ground_truth'/f'{image_id}.npz';np.savez_compressed(gp,gt=np.stack(masks),valid=labels!=255,annotation_ids=np.array([r['annotation_id'] for r in objects]))
        images.append({'image_id':image_id,'sequence':seq,'frame':ann.stem,'image_file':str(jpg),'gt_file':str(gp),'height':h,'width':w,'objects':objects})
    (dest/'manifest.json').write_text(json.dumps({'dataset':'DAVIS2017 val first frames only, no temporal inference','images':images,'num_images':len(images),'num_instances':sum(len(x['objects']) for x in images),'seed':2030},indent=2)+'\n')
    print(json.dumps({'DAVIS_sequences':len(images),'objects':sum(len(x['objects']) for x in images)}),flush=True)


if __name__=='__main__':main()
