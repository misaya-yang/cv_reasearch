#!/usr/bin/env python3
"""CPU-only frozen-CF gold-condition use diagnostic of existing checkpoints.

No training and no CUDA. Natural paired conditions change gold and its derived
scalar scores. The additional paired intervention changes ONLY reference_mask
while scalar scores, physical images, candidate/P1 union and provenance stay
fixed. Score sensitivity, hard choices and correct flips are separate outcomes.
These are eight already-seen train-only CF tasks, never independent method gain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from conditional_ranker_train import ConditionalRanker, model_input, safe_load, validate_record


@torch.no_grad()
def diagnose(cf_path, checkpoints):
    document=json.loads(cf_path.read_text())
    if document.get('state')!='COMPLETED':raise ValueError('Completed frozen CF availability report required')
    frozen=document['frozen_valid_training_records']
    if len(frozen)!=8:raise ValueError('Exactly frozen8 required; no search/replacement')
    groups={};shared_cache={}
    for entry in frozen:
        path=Path(entry['path']);task=safe_load(Path(entry['overlay_path']))
        if path not in shared_cache:
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=entry['source_shared_sha256']:raise ValueError('Shared source SHA mismatch')
            shared_cache[path]=(safe_load(path),digest)
        shared,digest=shared_cache[path]
        if task['source_shared_sha256']!=digest:raise ValueError('Task source SHA mismatch')
        if any(k in task for k in ('reference_tokens','query_tokens','donor_tokens','donor_masks','candidate_masks','candidate_provenance')):
            raise ValueError('Task changes physical evidence/candidate union')
        row=dict(shared);row.update(task);row['e']=10000+int(entry['dataset_index'])
        validate_record(row)
        groups.setdefault(int(entry['counterfactual_pair_index']),[]).append(row)
    result=dict(scope='Frozen four train-only CF pairs, gold-condition interface diagnostic; not held-out improvement or encoder-cause evidence',
        source_cf_sha256=hashlib.sha256(cf_path.read_bytes()).hexdigest(),pairs=4,tasks=8,models={})
    for tag,path in checkpoints.items():
        checkpoint=safe_load(path)
        model=ConditionalRanker(1024,8,use_donors=True).eval()
        model.load_state_dict(checkpoint['model'],strict=True)
        mean=checkpoint['scalar_mean'].float();std=checkpoint['scalar_std'].float()
        rows=[]
        for pair_index,tasks in sorted(groups.items()):
            if len(tasks)!=2:raise ValueError('Need both paired conditions')
            tasks=sorted(tasks,key=lambda row:row['task_tag'])
            if torch.equal(tasks[0]['reference_mask'],tasks[1]['reference_mask']):raise ValueError('Gold mask failed to change')
            natural=[];mask_only=[]
            for i,row in enumerate(tasks):
                natural.append(model(model_input(row,torch.device('cpu'),mean,std,True)).float())
                swapped=dict(row,reference_mask=tasks[1-i]['reference_mask'])
                mask_only.append(model(model_input(swapped,torch.device('cpu'),mean,std,True)).float())
            choices=[int(score.argmax()) for score in natural]
            best=[set(torch.where(torch.isclose(row['candidate_iou'],row['candidate_iou'].max(),atol=1e-8,rtol=0))[0].tolist()) for row in tasks]
            rows.append(dict(pair_index=pair_index,classes=[int(row['c']) for row in tasks],
                natural_choices=choices,natural_scores=[score.tolist() for score in natural],
                natural_choice_flip=choices[0]!=choices[1],both_choices_in_GT_best=all(chosen in oracle for chosen,oracle in zip(choices,best)),
                gold_only_swap_choices=[int(score.argmax()) for score in mask_only],
                gold_only_swap_changes_choice=[int(score.argmax())!=chosen for score,chosen in zip(mask_only,choices)],
                gold_only_swap_score_max_abs=[float((score-before).abs().max()) for score,before in zip(mask_only,natural)],
                native_regret=[float(row['candidate_iou'].max()-row['candidate_iou'][chosen]) for row,chosen in zip(tasks,choices)],
                GT_best_indices=[sorted(value) for value in best],checkpoint_epoch=int(checkpoint['epoch'])))
            print(json.dumps(dict(event='gold_pair',model=tag,**rows[-1])),flush=True)
        result['models'][tag]=dict(checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            natural_choice_flip_pairs=sum(r['natural_choice_flip'] for r in rows),
            both_correct_flip_pairs=sum(r['both_choices_in_GT_best'] for r in rows),
            gold_only_argmax_change_tasks=sum(sum(r['gold_only_swap_changes_choice']) for r in rows),
            gold_only_score_changes_tasks=sum(sum(value>0 for value in r['gold_only_swap_score_max_abs']) for r in rows),
            mean_train_CF_native_regret=sum(sum(r['native_regret']) for r in rows)/8,records=rows)
        del model,checkpoint
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cf-report',type=Path,required=True)
    ap.add_argument('--checkpoint',action='append',required=True,help='tag=/path/checkpoint.pt')
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();torch.set_num_threads(2)
    checkpoints={tag:Path(path) for tag,path in (value.split('=',1) for value in a.checkpoint)}
    start=time.monotonic();result=diagnose(a.cf_report,checkpoints);result['elapsed_s']=time.monotonic()-start
    a.out.parent.mkdir(parents=True,exist_ok=True);tmp=a.out.with_suffix('.tmp');tmp.write_text(json.dumps(result,allow_nan=False));tmp.replace(a.out)
    print(json.dumps(dict(event='gold_diagnostic_complete',elapsed_s=result['elapsed_s'],models={k:{key:r[key] for key in ('natural_choice_flip_pairs','both_correct_flip_pairs','gold_only_argmax_change_tasks','gold_only_score_changes_tasks','mean_train_CF_native_regret')} for k,r in result['models'].items()})),flush=True)


if __name__=='__main__':main()
