"""Compact receipts for completed quality comparisons and their inference costs."""
import argparse
import json
from pathlib import Path
import statistics
from takeover_flip_score import bootstrap_image_groups


def run(root):
    def load(name):
        d=json.loads((root/name).read_text())
        if not d["status"].startswith(("COMPLETED","PASS_")):raise ValueError(name+" incomplete")
        return d
    holdout=load("holdout_256/prompt_stability_256.json")
    out={"status":"COMPLETED","heldout_images":len({r['image_id'] for r in holdout['rows']}),
         "heldout_objects_per_regime":holdout['summary']['central']['prompts'],
         "cohort":"seed2029, exclude all prior152 images; same COCO distribution, not cross-dataset",
         "holdout_quality":holdout['summary'],"paired_boundary_vs_largest":{},
         "virtual_flip_128":load("virtual_flip_score_128.json")['summary'],
         "RGB_flip_logit_fusion_128":load("flip_fusion_128.json")['summary'],
         "standard_mask_feedback_128":load("self_refine_128.json")['summary'],
         "GPU_matching_equivalence":load("gpu_matching_256.json"),"latency":{}}
    for regime in ('central','near_boundary','box'):
        rows=[dict(r) for r in holdout['rows'] if r['regime']==regime]
        for r in rows:r['delta']=r['perturb_consistency_boundary_iou']-r['largest_original_mask_boundary_iou']
        out['paired_boundary_vs_largest'][regime]={"mean_delta":statistics.mean(r['delta'] for r in rows),"image_cluster_95":bootstrap_image_groups(rows,'delta')}
    for file in ('quality_latency.json','quality_latency_gpu.json'):
        timing=load(file)
        groups={}
        for cold in (False,True):
            g={}
            for name in sorted({r['arm'] for r in timing['records']}):
                rows=[r for r in timing['records'] if r['arm']==name and r['prepared_image_with_encoder']==cold]
                g[name]={"mean_round_image_median_ms":statistics.mean(r['median_ms'] for r in rows),
                         "round_image_medians_ms":[r['median_ms'] for r in rows]}
            groups['prepared_image_with_encoder' if cold else 'cached_original_embedding']=g
        out['latency'][file]={"scope":timing['scope'],"conditions":groups}
    (root/'quality_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k in ('heldout_images','heldout_objects_per_regime','GPU_matching_equivalence')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);run(p.parse_args().root)
