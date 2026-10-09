#!/usr/bin/env python3
"""Label-side normalization bridges, error lifecycles and reference coverage.

Uses sealed source fields/masks only; no candidate prediction or model call.
"""
import concurrent.futures
from collections import defaultdict
import json
import multiprocessing
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from lvis_atomic_study import DEFAULT_OUT, FORIS_CHAIN, point, sha, write


def initialize(out):
    global ROOT, NP, TORCH, F, IMAGE, ARRAY_HASH, COUNTS, GROSS
    import numpy as np
    import torch
    import torch.nn.functional as f
    from PIL import Image
    from ics.official_data import array_hash
    from ics.metrics import counts,gross_edits
    torch.set_num_threads(1)
    ROOT,NP,TORCH,F,IMAGE,ARRAY_HASH,COUNTS,GROSS=Path(out),np,torch,f,Image,array_hash,counts,gross_edits


def render_field(value, parent, shape):
    scaled=value-parent.min()
    denominator=(parent-parent.min()).max().clamp_min(1e-6)
    cli=F.interpolate((scaled/denominator)[None,None],(1024,1024),mode='bilinear',align_corners=False)[0,0]>.5
    if shape!=(1024,1024):
        cli=F.interpolate(cli[None,None].float(),shape,mode='bilinear',align_corners=False)[0,0]>.5
    return cli.numpy()


def lifecycle(chain,masks,truth):
    final=masks[chain[-1]]
    ever=NP.logical_or.reduce([masks[a] for a in chain])
    always=NP.logical_and.reduce([masks[a] for a in chain])
    fn=truth&~final;fp=~truth&final
    categories=dict(fn_never_selected=int((fn&~ever).sum()),fn_selected_then_lost=int((fn&ever).sum()),
        fp_always_selected=int((fp&always).sum()),fp_excluded_then_readded=int((fp&~always).sum()))
    assert categories['fn_never_selected']+categories['fn_selected_then_lost']==int(fn.sum())
    assert categories['fp_always_selected']+categories['fp_excluded_then_readded']==int(fp.sum())
    last=NP.zeros(truth.shape,dtype=NP.uint8)
    retained={}
    for index,(base,arm) in enumerate(zip(chain,chain[1:]),1):
        previous,current=masks[base],masks[arm]
        corrected=(current!=previous)&(current==truth)
        harmed=(current!=previous)&(current!=truth)
        last[harmed]=index
        retained[arm]=dict(corrected=int(corrected.sum()),corrected_retained=int((corrected&(final==truth)).sum()),
            corrected_cancelled=int((corrected&(final!=truth)).sum()))
    attribution={arm:dict(fn_last_deleted=int((fn&ever&(last==index)).sum()),
                         fp_last_added=int((fp&~always&(last==index)).sum())) for index,arm in enumerate(chain[1:],1)}
    return dict(categories=categories,last_error_transition=attribution,correct_edit_retention=retained,
                fn_pixels=int(fn.sum()),fp_pixels=int(fp.sum()),gt_pixels=int(truth.sum()))


def one(task):
    row,rec=task;name=rec['filename']
    with IMAGE.open(row['query_mask_path']) as image:raw=(NP.asarray(image.convert('L'))>0).astype(NP.uint8)
    assert ARRAY_HASH(raw)==row['query_mask_hash']
    with IMAGE.open(row['reference_mask_path']) as image:ref=(NP.asarray(image.convert('L'))>0).astype(NP.uint8)
    assert ARRAY_HASH(ref)==row['reference_mask_hash']
    transformed=F.interpolate(TORCH.from_numpy(ref)[None,None].float(),(1024,1024),mode='nearest')>.5
    cov=F.interpolate(transformed.float(),(64,64),mode='area')[0,0].numpy()
    reference=dict(n_cov_ge_09=int((cov>=.9).sum()),max_cov=float(cov.max()),mean_cov=float(cov.mean()),
                   fallback=bool(not (cov>=.9).any()))
    with NP.load(ROOT/'fields'/name) as stored:
        assert sha(ROOT/'fields'/name)==rec['field_sha256']
        fg,bg,vote,prior,penalty,delta=(TORCH.from_numpy(stored[k].copy()) for k in ('fg','bg','vote','prior','penalty','delta'))
        values=[fg,fg-.55*bg]
        values.append(values[-1]+.20*(vote-.5));values.append(values[-1]+.25*(prior-.5))
        values.append(values[-1]-penalty);values.append(values[-1]+delta)
        assert NP.array_equal(values[-1].numpy(),stored['score'])
    result=dict(episode_id=row['episode_id'],fold=row['fold'],class_id=row['loader_class_id'],pilot=rec['pilot'],
                reference=reference,frames={})
    with NP.load(ROOT/'predictions'/name) as saved:
        for frame,shape in [('cli',(1024,1024)),('original',tuple(row['query_size_hw']))]:
            truth=F.interpolate(TORCH.from_numpy(raw)[None,None].float(),shape,mode='nearest')[0,0].numpy()>.5
            arms=FORIS_CHAIN+['mean.rank','mean','mean.graph_only']
            masks={a:NP.unpackbits(saved[frame+'/'+a],count=int(NP.prod(shape))).reshape(shape).astype(bool) for a in arms}
            bridges={}
            for index,(parent,child) in enumerate(zip(values,values[1:])):
                base,arm=FORIS_CHAIN[index:index+2]
                assert NP.array_equal(render_field(parent,parent,shape),masks[base])
                assert NP.array_equal(render_field(child,child,shape),masks[arm])
                bridge=render_field(child,parent,shape)
                bridges[arm]=dict(iu=COUNTS(bridge,truth),parent_to_bridge=GROSS(bridge,masks[base],truth),
                                 bridge_to_child=GROSS(masks[arm],bridge,truth))
            size=float(truth.mean());group='lt1pct' if size<.01 else ('1to20pct' if size<=.2 else 'gt20pct')
            life={label:lifecycle(chain,masks,truth) for label,chain in
                [('foris',FORIS_CHAIN),('mean',FORIS_CHAIN[:-1]+['mean.rank','mean'])]}
            final_iu={a:COUNTS(masks[a],truth) for a in ['foris.pre','foris.crf','mean.rank','mean.graph_only','mean']}
            # Fixed GT masks are a diagnostic; the groups are not inference rules.
            found={a:('missed' if iu[0]==0 else ('low_recall' if iu[0]<.5*truth.sum() else 'found'))
                   for a,iu in final_iu.items()}
            result['frames'][frame]=dict(bridges=bridges,lifecycle=life,query_area_fraction=size,
                                         query_area_group=group,iu=final_iu,found=found)
    return result


def total_life(rows,frame,branch):
    templates=rows[0]['frames'][frame]['lifecycle'][branch]
    result=dict(n=len(rows),categories={},last_error_transition={},correct_edit_retention={})
    for key in templates['categories']:
        result['categories'][key]=sum(r['frames'][frame]['lifecycle'][branch]['categories'][key] for r in rows)
    for block in ['last_error_transition','correct_edit_retention']:
        result[block]={arm:{k:sum(r['frames'][frame]['lifecycle'][branch][block][arm][k] for r in rows)
                           for k in values} for arm,values in templates[block].items()}
    result['fn_pixels']=sum(r['frames'][frame]['lifecycle'][branch]['fn_pixels'] for r in rows)
    result['fp_pixels']=sum(r['frames'][frame]['lifecycle'][branch]['fp_pixels'] for r in rows)
    result['gt_pixels']=sum(r['frames'][frame]['lifecycle'][branch]['gt_pixels'] for r in rows)
    return result


def main():
    out=DEFAULT_OUT;start=time.monotonic()
    if (out/'supplemental_report.json').exists():
        print('Supplement already completed');return
    rows=json.loads((out/'manifest.json').read_text())
    index={r['episode_id']:r for r in map(json.loads,(out/'inference.jsonl').read_text().splitlines())}
    assert json.loads((out/'sealed.json').read_text())['n']==1400
    tasks=[(r,index[r['episode_id']]) for r in rows]
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),
          initializer=initialize,initargs=(str(out),)) as pool,(out/'supplemental_episode_metrics.jsonl').open('w') as log:
        for r in pool.map(one,tasks,chunksize=4):
            results.append(r);log.write(json.dumps(r)+'\n')
    original=[json.loads(line) for line in (out/'episode_metrics.jsonl').read_text().splitlines()]
    old={r['episode_id']:r for r in original};report=dict(n=1400,frames={},seconds=time.monotonic()-start,
        source_sha256=sha(Path(__file__)),encoder_calls=0,GT_use='label-side diagnostics only')
    for frame in ['cli','original']:
        summaries={}
        for subset,rr in [('all1400',results),('design1300',[r for r in results if not r['pilot']])]:
            bridges=[]
            for i,(base,arm) in enumerate(zip(FORIS_CHAIN[:-2],FORIS_CHAIN[1:-1])):
                metric_rows=[dict(fold=r['fold'],class_id=r['class_id'],iu={
                    'parent':old[r['episode_id']]['frames'][frame]['iu'][base],
                    'bridge':r['frames'][frame]['bridges'][arm]['iu'],
                    'child':old[r['episode_id']]['frames'][frame]['iu'][arm]}) for r in rr]
                p=point(metric_rows,['parent','bridge','child'])
                bridges.append(dict(base=base,arm=arm,miou=p,raw_change_pp=p['bridge']-p['parent'],
                                    normalization_pp=p['child']-p['bridge'],total_delta_pp=p['child']-p['parent']))
            condition={}
            grouped=defaultdict(list)
            for r in rr:
                grouped['query/'+r['frames'][frame]['query_area_group']].append(r)
                grouped['reference/'+('fallback' if r['reference']['fallback'] else 'pure_present')].append(r)
            for key,g in grouped.items():
                arms=list(g[0]['frames'][frame]['iu'])
                p=point([dict(fold=r['fold'],class_id=r['class_id'],iu=r['frames'][frame]['iu']) for r in g],arms)
                condition[key]=dict(n=len(g),miou=p,class_count=len({(r['fold'],r['class_id']) for r in g}),
                    missed_cases={a:sum(r['frames'][frame]['found'][a]=='missed' for r in g) for a in arms},
                    low_recall_cases={a:sum(r['frames'][frame]['found'][a]=='low_recall' for r in g) for a in arms})
            summaries[subset]=dict(normalization_bridges=bridges,
                lifecycle={branch:total_life(rr,frame,branch) for branch in ['foris','mean']},conditions=condition,
                reference_fallback_count=sum(r['reference']['fallback'] for r in rr),
                reference_max_cov_min=min(r['reference']['max_cov'] for r in rr))
        report['frames'][frame]=summaries
    write(out/'supplemental_report.json',report)
    print(json.dumps(dict(n=1400,seconds=report['seconds'],reference_fallback_count=report['frames']['cli']['all1400']['reference_fallback_count'])))


if __name__=='__main__':main()
