"""Prospective disjoint-image validation of the unchanged flip rules."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/takeover_20261001_v1'
OUT=R/'flip_validation_128'
SUBSET=ROOT/'assets/coco_flip_validation_seed2028_v1'


def save(data):
    p=OUT/'queue_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)


def main():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((SUBSET/'manifest.json').read_text())
    development=json.loads((ROOT/'assets/coco_quality_seed2027_v1/manifest.json').read_text())
    assert len(manifest['images'])==128 and not {x['image_id'] for x in manifest['images']} & {x['image_id'] for x in development['images']}
    protocol={'status':'RULES_FROZEN_BEFORE_NEW_MODEL_OUTPUTS','seed':2028,'images':128,'instances':manifest['num_instances'],
              'excluded_development_images':[x['image_id'] for x in development['images']],
              'fixed_rules':['original_iou_head','matched_mean_iou_head','matched_consistency_only'],
              'selection_uses_GT':False,'matching':'original multimask tokens 1..3; maximum sum IoU over all 6 permutations',
              'primary_endpoints':['central mean mask IoU','near_boundary mean mask IoU'],
              'controls':['box IoU','Boundary IoU','per-object improvements/losses','image-cluster bootstrap'],
              'score_source_sha256':hashlib.sha256((ROOT/'tools/takeover_flip_score.py').read_bytes()).hexdigest()}
    (OUT/'validation_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    report={'status':'RUNNING','started_at_utc':datetime.now(timezone.utc).isoformat(),'stages':[]}
    jobs=[('original_official',[sys.executable,'tools/takeover_official_export.py','--subset-dir',str(SUBSET),'--checkpoint',str(ROOT/'assets/checkpoints/sam_vit_b_01ec64.pth'),'--output-dir',str(OUT/'original'),'--device','cuda:0','--limit-images','128','--microbatch','4','--defer-quality','--npz-compression','none']),
          ('flipped_official',[sys.executable,'tools/takeover_flip.py','--subset-dir',str(SUBSET),'--output-dir',str(OUT/'flip'),'--device','cuda:0']),
          ('fixed_rule_score_CPU',[sys.executable,'tools/takeover_flip_score.py','--original-run',str(OUT/'original'),'--flipped-run',str(OUT/'flip/official_view'),'--output',str(OUT/'flip_rank_score.json')])]
    save(report)
    try:
        for name,command in jobs:
            row={'name':name,'status':'RUNNING','command':command};report['stages'].append(row);save(report)
            print('START '+name,flush=True);started=time.monotonic()
            with (OUT/(name+'.log')).open('x') as log:
                p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);row['pid']=p.pid;save(report);code=p.wait()
            row.update(status='DONE' if code==0 else 'FAILED',returncode=code,wall_seconds=time.monotonic()-started);save(report)
            if code:raise RuntimeError(name+' failed')
            print('DONE '+name,flush=True)
        report['status']='VALIDATION_COMPLETE';report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        report.update(status='ERROR',error=str(error));raise
    finally:save(report)


if __name__=='__main__':main()
