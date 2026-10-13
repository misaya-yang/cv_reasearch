"""Score all six complete arms only after all7200 masks are sealed."""
from collections import Counter,defaultdict
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
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


helper=module('constellation_score_helper',BASE/'evidence/local/reference_background_conflict6200_20261011/score_and_verify.py')
core=module('constellation_score_core',HERE/'core.py')
read,write,sha,lines=helper.read,helper.write,helper.sha,helper.lines


def main():
    assert not (HERE/'report.json').exists(),'Never rescore closed candidate'
    torch.set_num_threads(2)
    started,cpu=time.monotonic(),time.process_time()
    protocol,config,seal=read(HERE/'protocol.json'),read(HERE/'config.json'),read(HERE/'sealed.json')
    assert seal['state']=='ALL600_SIX_COMPLETE_ARMS_TWO_FRAMES_SEALED_BEFORE_GT'
    assert seal['n']==config['n']==600 and seal['mask_count']==7200
    assert seal['protocol_sha256']==sha(HERE/'protocol.json')==sha(HERE/'protocol.frozen.json')==config['protocol_sha256']
    assert seal['config_sha256']==sha(HERE/'config.json')
    assert tuple(protocol['arms'])==tuple(config['arms'])==core.ARMS
    for p,digest in protocol['source_sha256'].items():assert sha(p)==digest,p
    members=read(ALIGN/'membership.json');bindings=lines(OLD/'score_bindings.jsonl')
    expected_keys={'original_hw'}|{frame+'/'+arm for frame in ('cli','original') for arm in core.ARMS}
    checks=Counter()
    records=[]
    for i,(m,entry) in enumerate(zip(members,seal['records'])):
        assert entry['index']==m['subset_index']==i and entry['episode_id']==m['episode_id']
        path=HERE/'predictions'/entry['filename']
        assert sha(path)==entry['prediction_sha256']
        rec=read(HERE/'records'/f'{i:06}.json');records.append(rec)
        assert rec['prediction_sha256']==entry['prediction_sha256'] and rec['index']==i and rec['episode_id']==m['episode_id']
        assert rec['query_GT_reads']==rec['new_encoder_calls']==rec['new_CRF_calls']==0
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)==expected_keys
            shape=tuple(bindings[m['source_index']]['row']['query_size_hw'])
            assert tuple(z['original_hw'])==shape
            for frame,hw in (('cli',(1024,1024)),('original',shape)):
                for arm in core.ARMS:helper.unpack(z,frame+'/'+arm,hw,checks)
        checks['pre_GT_prediction_record_identity']+=1
    assert len(seal['records'])==len(records)==len(members)==600
    frames={frame:defaultdict(lambda:defaultdict(lambda:[0,0])) for frame in ('cli','original')}
    gross={frame:{arm:Counter() for arm in core.ARMS} for frame in frames}
    cases=[];available=Counter();fitcost=Counter();BGregimes=Counter()
    with (HERE/'episodes.jsonl').open('w') as log:
        for i,(m,entry,record) in enumerate(zip(members,seal['records'],records)):
            binding=bindings[m['source_index']];row=binding['row'];assert row['episode_id']==m['episode_id']
            gt=helper.truth(row,checks)
            assert sha(m['old_region']['old_region_path'])==m['old_region']['old_region_current_sha256']
            bb=binding['old_final_masks']['B'];assert sha(bb['path'])==bb['sha256']
            case=dict(index=i,source_index=m['source_index'],episode_id=m['episode_id'],fold=row['fold'],loader_class_id=row['loader_class_id'],frames={})
            stats=record['registration_stats']
            for name,a in stats['arms'].items():
                available[name+'_valid' if a['available'] else name+'_fallback_S']+=1
                for k in ('assignment_steps','pose_fit_attempts','pose_fits'):fitcost[k]+=a[k]
                if a['available']:
                    h=a['energy_history'];d=a['energy_after_assignment'];assert len(h)==a['pose_fits']==len(d)+1
                    for j,e in enumerate(d):
                        assert e<=h[j]+1e-9*max(1.,abs(h[j])) and h[j+1]<=e+1e-9*max(1.,abs(e))
                        checks['actual_solver_energy_monotone']+=2
                    r=np.array(a['physical_rotation']);assert np.linalg.det(r)>0 and np.allclose(r.T@r,np.eye(2),atol=1e-12) and a['physical_scale']>0
                    checks['actual_similarity_pose_orientation_scale']+=1
            if 'BG_penalty' in stats:
                a=stats['BG_penalty']
                for k in ('entirely_query_row_constant','entirely_reference_column_constant'):BGregimes[k]+=int(a[k])
                BGregimes['nonadditive_interaction_gt_1e-12']+=int(a['max_abs_nonadditive_interaction']>1e-12)
            with np.load(m['old_region']['old_region_path'],allow_pickle=False) as oldz,np.load(bb['path'],allow_pickle=False) as bz,np.load(HERE/'predictions'/entry['filename'],allow_pickle=False) as nz:
                for frame,shape in (('cli',(1024,1024)),('original',tuple(row['query_size_hw']))):
                    truth=helper.resize(gt,shape,'nearest')>.5
                    s=helper.unpack(oldz,('canvas' if frame=='cli' else 'original')+'/component.query_context',shape,checks)
                    b=helper.unpack(bz,bb[frame+'_key'],shape,checks)
                    outputs={arm:helper.unpack(nz,frame+'/'+arm,shape,checks) for arm in core.ARMS}
                    wide=outputs[core.ARMS[4]]
                    assert not np.any(s&~wide) and not np.any((wide&~s)&~b)
                    details=dict(iu={},edits={},density={})
                    for arm,out in dict(B=b,S=s,**outputs).items():
                        I,U=int((out&truth).sum()),int((out|truth).sum())
                        details['iu'][arm]=[I,U]
                        key=(str(row['fold']),str(row['loader_class_id']))
                        frames[frame][arm][key][0]+=I;frames[frame][arm][key][1]+=U
                        if arm in core.ARMS:
                            add,delete=out&~s,s&~out
                            assert not delete.any() and not np.any(add&~wide)
                            cnt=dict(add_TP=int((add&truth).sum()),add_FP=int((add&~truth).sum()),delete_TP=0,delete_FP=0)
                            assert cnt['add_TP']+cnt['add_FP']==record['frames'][frame]['arms'][arm]['add']
                            gross[frame][arm].update(cnt);details['edits'][arm]=cnt
                            checks['S_preservation_wide_domain_edit_density']+=1
                    assert int((outputs[core.ARMS[0]]&~s).sum())==int((outputs[core.ARMS[-1]]&~s).sum())
                    pool=int((wide&~s).sum());k=int((outputs[core.ARMS[0]]&~s).sum())
                    details['density']=dict(pool=pool,added=k,budget_nondegenerate=bool(0<k<pool),main_equals_wide=bool(np.array_equal(outputs[core.ARMS[0]],wide)))
                    checks['same_actual_add_budget']+=1
                    case['frames'][frame]=details
            log.write(json.dumps(case)+'\n');cases.append(case)
    expected=helper.expected_classes()['lvis']
    summary={}
    for frame,data in frames.items():
        metrics={}
        for arm,classes in data.items():
            assert len(classes)==600 and all(cl in expected[fold] for fold,cl in classes)
            assert Counter(fold for fold,cl in classes)==Counter({str(i):60 for i in range(10)})
            means=[I/max(U,1) for I,U in classes.values()]
            metrics[arm]=dict(observed600=100*sum(means)/600,official920=100*sum(means)/920)
            checks['actual_class_fold_identity']+=600
        paired={}
        for other in ('B','S',*core.ARMS[1:]):
            diffs=[]
            for c in cases:
                a=c['frames'][frame]['iu'][core.ARMS[0]];b=c['frames'][frame]['iu'][other]
                diffs.append(a[0]/max(a[1],1)-b[0]/max(b[1],1))
            paired[other]=dict(delta_pp=100*sum(diffs)/600,improved=sum(x>0 for x in diffs),declined=sum(x<0 for x in diffs),same=sum(x==0 for x in diffs))
        summary[frame]=dict(metrics=metrics,gross={k:dict(v) for k,v in gross[frame].items()},paired=paired,
                            budget_nondegenerate_cases=sum(c['frames'][frame]['density']['budget_nondegenerate'] for c in cases),
                            main_equals_wide_cases=sum(c['frames'][frame]['density']['main_equals_wide'] for c in cases))
    report=dict(state='ALL600_SIX_COMPLETE_ARMS_SCORED_DIRECTLY_VERIFIED',n=600,mask_count=7200,summary=summary,checks=dict(checks),available=dict(available),actual_optimization_counts=dict(fitcost),BG_cost_regimes=dict(BGregimes),
                wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,query_GT_reads=600,
                production_receipt=read(HERE/'production_receipt.json'),protocol_sha256=sha(HERE/'protocol.json'),seal_sha256=sha(HERE/'sealed.json'),episodes_sha256=sha(HERE/'episodes.jsonl'),
                new_encoder_calls=0,new_predictions=0,raw_feature_reads=0,scope='Same original600 complete composition, allmasks sealed first. Added points constrained by fitted legalRMask template; particle subset registration is not semantic proof or fullR explanation. No task-index winner or query-label adaptation.')
    write(HERE/'report.json',report)
    orig,cli=summary['original'],summary['cli']
    adopted=bool(orig['paired']['S']['delta_pp']>=1. and cli['paired']['S']['delta_pp']>0 and all(orig['paired'][k]['delta_pp']>0 and cli['paired'][k]['delta_pp']>0 for k in core.ARMS[1:]))
    decision=dict(state='PASS_FIXED600_SCREEN_REQUIRES_MULTIDOMAIN_VALIDATION' if adopted else 'REJECTED_TASK_CONSTELLATION600_NO_EXPANSION',n=600,accepted=adopted,goal_complete=False,
                  primary=orig['metrics'][core.ARMS[0]]['observed600'],strong_S=orig['metrics']['S']['observed600'],delta_vs_S_pp=orig['paired']['S']['delta_pp'],required_delta_pp=1.,CLI_delta_vs_S_pp=cli['paired']['S']['delta_pp'],
                  full_control_deltas_original_pp={k:orig['paired'][k]['delta_pp'] for k in core.ARMS[1:]},no6200_expansion=not adopted,no_parameter_followups=True,
                  reason='Frozen S+1pp, positive CLI vs S and positive bothframe full-control deltas; no ablationwinner substitution or same-recipe tuning.')
    write(HERE/'screening_decision.json',decision)
    print(json.dumps(dict(state=decision['state'],primary=decision['primary'],delta_vs_S_pp=decision['delta_vs_S_pp'],wall_seconds=report['wall_seconds'],available=dict(available)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
