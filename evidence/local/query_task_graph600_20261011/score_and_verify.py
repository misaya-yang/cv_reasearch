"""Task-query graph: all4800 masks gated before GT and direct600 verification."""
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
OLD=BASE/'evidence/local/task_conflict_execution6200_20261011'
ALIGN=BASE/'evidence/local/strong600_conflict_alignment_20261011'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


helper=module('archived_background_scorer',BASE/'evidence/local/reference_background_conflict6200_20261011/score_and_verify.py')
core=module('fixed600_graph_core',HERE/'core.py')
read,lines,write,sha=helper.read,helper.lines,helper.write,helper.sha
ACTS=('add_TP','add_FP','delete_TP','delete_FP')


def pool(rows,arm,expected):
    totals=defaultdict(lambda:[0,0])
    for r in rows:
        pair=r['iu'][arm];key=str(r['fold']),str(r['class_id'])
        totals[key][0]+=pair[0];totals[key][1]+=pair[1]
    folds=[sum(100*totals[f,c][0]/max(totals[f,c][1],1) for c in cs)/len(cs) for f,cs in expected.items()]
    return sum(folds)/len(folds)


def gross(a,b,target):
    add,delete=a&~b,b&~a
    return dict(zip(ACTS,map(int,((add&target).sum(),(add&~target).sum(),(delete&target).sum(),(delete&~target).sum()))))


def main():
    assert not (HERE/'report.json').exists(),'Never repeat completed scoring'
    tick,cpu=time.monotonic(),time.process_time();torch.set_num_threads(4)
    seal=read(HERE/'sealed.json');config=read(HERE/'config.json');protocol=read(HERE/'protocol.json')
    assert seal['n']==config['n']==600 and seal['mask_count']==4800
    assert sha(HERE/'config.json')==seal['config_sha256']
    assert sha(HERE/'protocol.json')==seal['protocol_sha256']==config['protocol_sha256']
    assert tuple(config['arms'])==tuple(core.ARMS)==tuple(protocol['arms'])
    for p,d in protocol['source_sha256'].items():assert sha(p)==d,p
    checks=Counter();members=read(ALIGN/'membership.json');prior=lines(ALIGN/'episodes.jsonl')
    scores=lines(OLD/'score_bindings.jsonl');sources=lines(OLD/'inputs.jsonl')
    expected=helper.expected_classes()['lvis']
    # Every packet, all four arms and both frame schemas before first GT read.
    for index,rec in enumerate(seal['records']):
        member=members[index];row=scores[member['source_index']]['row']
        assert rec['index']==index and rec['episode_id']==member['episode_id']==row['episode_id']
        assert rec['source_index']==member['source_index']
        p=HERE/'predictions'/rec['filename'];assert sha(p)==rec['prediction_sha256']
        with np.load(p,allow_pickle=False) as z:
            assert set(z.files)=={'original_hw'}|{f+'/'+a for f in ('cli','original') for a in core.ARMS}
            assert z['original_hw'].tolist()==row['query_size_hw']
            for frame in ('cli','original'):
                shape=(1024,1024) if frame=='cli' else tuple(row['query_size_hw'])
                for arm in core.ARMS:helper.unpack(z,frame+'/'+arm,shape,checks)
        checks['complete_packet_SHA_identity_preGT']+=1
    assert checks['packed_schema_and_padding']==4800
    pregt=time.monotonic()-tick;records=[];hashed=set()
    for index,(member,old) in enumerate(zip(members,prior)):
        scoring=scores[member['source_index']];row=scoring['row'];rec=read(HERE/'records'/f'{index:06}.json')
        assert member['episode_id']==old['episode_id']==rec['episode_id']==row['episode_id']
        sb=member['old_region'];assert sha(sb['old_region_path'])==sb['old_region_current_sha256']
        truth=helper.truth(row,checks)
        out=dict(index=index,source_index=member['source_index'],episode_id=row['episode_id'],
                 fold=old['fold'],class_id=old['class_id'],frames={})
        with np.load(HERE/'predictions'/f'{index:06}.npz',allow_pickle=False) as zp,np.load(sb['old_region_path'],allow_pickle=False) as zs:
            for frame in ('cli','original'):
                shape=(1024,1024) if frame=='cli' else tuple(row['query_size_hw'])
                target=helper.resize(truth,shape,'nearest')>.5
                masks=helper.old_support(scoring,shape,frame,hashed,checks)
                s=helper.unpack(zs,('canvas' if frame=='cli' else 'original')+'/component.query_context',shape,checks)
                b,g,r=(masks[k] for k in ('B','G','R0'))
                added_domain=(~s)&(b|g|r);deleted_domain=s&~(b&g&r)
                d,conf=helper.source_margin(sources[member['source_index']]['source_score64'],shape,hashed)
                iu=dict(old['frames'][frame]['iu']);gt=int(target.sum())
                outputs={a:helper.unpack(zp,frame+'/'+a,shape,checks) for a in core.ARMS}
                metrics={};edits={};controls={}
                for arm,pred in outputs.items():
                    add,delete=pred&~s,s&~pred
                    assert not np.any(add&~added_domain) and not np.any(delete&~deleted_domain)
                    assert np.array_equal(pred[s&b&g&r],s[s&b&g&r])
                    assert np.array_equal(pred[~s&~(b|g|r)],s[~s&~(b|g|r)])
                    I=int((pred&target).sum());FP=int((pred&~target).sum());FN=gt-I;U=I+FP+FN
                    iu[arm]=[I,U];metrics[arm]=dict(I=I,U=U,TP=I,FP=FP,FN=FN,GT=gt)
                    edits[arm]=gross(pred,s,target)
                    assert rec['frames'][frame]['edit_counts'][arm]['add']==int(add.sum())
                    assert rec['frames'][frame]['edit_counts'][arm]['delete']==int(delete.sum())
                    base=old['frames'][frame]['iu']['strong.component.query_context']
                    assert I==base[0]+edits[arm]['add_TP']-edits[arm]['delete_TP']
                    base_fp=base[1]-gt
                    assert FP==base_fp+edits[arm]['add_FP']-edits[arm]['delete_FP']
                    checks['direct_IU_confusion_gross_domain_protection']=checks['direct_IU_confusion_gross_domain_protection']+1
                main=outputs[core.ARMS[0]];feature_only=outputs[core.ARMS[1]];budget=outputs[core.ARMS[-1]]
                rebuilt=s.copy()
                for domain,count,descending,value in ((added_domain,int((main&~s).sum()),True,True),
                                                       (deleted_domain,int((s&~main).sum()),False,False)):
                    ids=np.flatnonzero(domain.ravel());order=np.argsort(-d.ravel()[ids] if descending else d.ravel()[ids],kind='stable')
                    assert count<=len(ids);rebuilt.ravel()[ids[order[:count]]]=value
                assert np.array_equal(budget,rebuilt)
                checks['direct_sameframe_budget_counts_positions']+=1
                for name,baseline in {'B':b,'N':masks['N'],core.ARMS[1]:feature_only,core.ARMS[2]:outputs[core.ARMS[2]],core.ARMS[-1]:budget}.items():
                    controls[name]=gross(main,baseline,target)
                out['frames'][frame]=dict(iu=iu,metrics=metrics,gross_vs_strong_S=edits,gross_main_vs_controls=controls)
        records.append(out)
        if (index+1)%100==0:print(dict(scored=index+1,seconds=time.monotonic()-tick),flush=True)
    arms=list(records[0]['frames']['original']['iu']);observed={f:sorted({str(r['class_id']) for r in records if str(r['fold'])==f},key=int) for f in expected}
    report=dict(state='COMPLETE_FIXED600_TASK_QUERY_GRAPH_FOUR_ARMS_DIRECTLY_VERIFIED',n=600,new_masks=4800,
                arms=arms,new_arms=list(core.ARMS),frames={},checks=dict(checks),preGT_all4800_gate_seconds=pregt,
                protocol_sha256=sha(HERE/'protocol.json'),seal_sha256=sha(HERE/'sealed.json'),
                wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                raw_feature_reads=0,new_fits=0,new_predictions=0,query_GT_reads=600,
                scope='One previously exposed exact600 queue, no full6200 expansion or domain/GT/frame selection')
    for frame in ('cli','original'):
        rows=[dict(fold=r['fold'],class_id=r['class_id'],iu=r['frames'][frame]['iu']) for r in records]
        report['frames'][frame]=dict(official920={a:pool(rows,a,expected) for a in arms},
            observed600_slots={a:pool(rows,a,observed) for a in arms},
            gross_vs_strong_S={a:{k:sum(r['frames'][frame]['gross_vs_strong_S'][a][k] for r in records) for k in ACTS} for a in core.ARMS},
            gross_main_vs_controls={a:{k:sum(r['frames'][frame]['gross_main_vs_controls'][a][k] for r in records) for k in ACTS} for a in records[0]['frames'][frame]['gross_main_vs_controls']})
    with (HERE/'episodes.jsonl').open('w') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report['episodes_sha256']=sha(HERE/'episodes.jsonl')
    write(HERE/'report.json',report)
    print(json.dumps({k:report[k] for k in ('state','n','new_masks','wall_seconds','frames')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
