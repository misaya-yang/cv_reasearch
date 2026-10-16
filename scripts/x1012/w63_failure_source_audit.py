"""Read-only source identity / geometry audit motivated by W63 failure boards."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/x1012'),str(ROOT/'scripts/x1013')]
import common as c
import w31_demonstration as w31
import w59_normalization_causal as run


def flags(b):
    r=b['row']
    return {role:r[role+'_size_hw']!=r[role+'_mask_size_hw'] for role in ('reference','query')}


def main():
    out=c.task_path('W63')/'source_audit';out.mkdir(exist_ok=True)
    with gzip.open(ROOT/'evidence/research/w58_score_repair_cases.jsonl.gz','rt') as f:
        old={r['index']:r for r in map(json.loads,f)}
    original=[]
    for b in w31.rows():
        x=flags(b);r=old[str(b['index'])]
        original.append(dict(index=r['index'],dataset=b['dataset'],photo=c.photo(b),**x,catastrophic=r['catastrophic']))
    counts={}
    for d in c.DOMAINS:
        rows=[r for r in original if r['dataset']==d]
        counts[d]=dict(n=len(rows),reference_mismatch=sum(r['reference'] for r in rows),query_mismatch=sum(r['query'] for r in rows),
                       either=sum(r['reference'] or r['query'] for r in rows),
                       catastrophic_total=sum(r['catastrophic'] for r in rows),
                       catastrophic_with_either=sum(r['catastrophic'] and (r['reference'] or r['query']) for r in rows))
    manifest=json.loads((c.task_path('W63')/'manifest.json').read_text())
    with gzip.open(ROOT/'evidence/research/w63_task_switch_S.jsonl.gz','rt') as f:
        scored={r['pair_id']:r for r in map(json.loads,f)}
    task_rows=[]
    for item in manifest['episodes']:
        for i,b in enumerate(item['bindings']):
            task_rows.append(dict(pair_id=item['pair_id'],task=('A','B')[i],dataset=item['dataset'],**flags(b),
                                  field_oracle_below_half=scored[item['pair_id']]['oracle'][i][i]<.5))
    source_inventory=json.loads((c.ASSETS/'setup/manifests/ics_datasets.json').read_text())
    expected={r['path']:r for r in source_inventory['files']}; examples=[]
    for pair in ('suim_026','suim_028'):
        item=next(x for x in manifest['episodes'] if x['pair_id']==pair);b=item['bindings'][1];r=b['row']
        relative=f"SUIM/masks/{r['global_class_id']}/{Path(r['query_path']).stem}.bmp"
        p=c.ASSETS/'datasets/ics'/relative
        source=np.asarray(Image.open(p).convert('L'));saved=np.asarray(Image.open(r['query_mask_path']).convert('L'))
        source_bin=source>=128;saved_bin=saved>0
        assert np.array_equal(source_bin,saved_bin)
        assert run.sha(p)==expected[relative]['sha256']
        values,n=np.unique(source,return_counts=True)
        examples.append(dict(pair_id=pair,task='B',class_id=r['global_class_id'],query_photo=c.photo(b),
                             RGB_hw=r['query_size_hw'],mask_hw=list(source.shape),
                             source_mask=str(p),frozen_mask=r['query_mask_path'],source_sha256=run.sha(p),
                             transfer_inventory_SHA_match=True,source_and_frozen_pixel_identity=True,
                             values={str(int(v)):int(k) for v,k in zip(values,n)},foreground_pixels=int(source_bin.sum()),
                             foreground_fraction=float(source_bin.mean()),
                             interpretation='Source-pool label / geometry concern from visual inspection; not a proven conversion bug or corrected annotation'))
    result=dict(original_S=counts,W63=dict(n_tasks=len(task_rows),
                 reference_mismatch=sum(r['reference'] for r in task_rows),query_mismatch=sum(r['query'] for r in task_rows),
                 mismatch_tasks=[r for r in task_rows if r['reference'] or r['query']],
                 field_failures_with_mismatch=sum(r['field_oracle_below_half'] and (r['reference'] or r['query']) for r in task_rows)),
                examples=examples,changes_to_labels_or_scores=False,
                limitations='Inventory SHA proves transfer identity only, not semantic annotation correctness. Upstream original RGB annotations/conversion not independently recovered.')
    run.write(out/'original_S_flags.json',original);run.write(out/'W63_task_flags.json',task_rows)
    run.write(out/'summary.json',result);run.write(ROOT/'evidence/research/w63_source_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
