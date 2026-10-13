"""Post-seal diagnosis of old query-role labels, no new method or inference."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/local/strong_negative_exposure600_20261011'


def tensor_sha(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(json.dumps([list(a.shape),a.dtype.str]).encode());h.update(a.tobytes());return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'report.json').exists()
    start,cpu=time.monotonic(),time.process_time();torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('negative_exposure_helper',ROOT/'evidence/local/reference_background_conflict6200_20261011/score_and_verify.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    members=h.read(ROOT/'evidence/local/strong600_conflict_alignment_20261011/membership.json')
    bindings=h.lines(ROOT/'evidence/local/task_conflict_execution6200_20261011/score_bindings.jsonl')
    graph=ROOT/'evidence/local/query_task_graph600_20261011';ridge=ROOT/'evidence/local/task_discriminant600_20261011'
    checks=Counter();hashed=set();totals=Counter();cases=[]
    def expanded(mask):return np.repeat(np.repeat(mask,16,axis=0),16,axis=1)
    for i,m in enumerate(members):
        binding=bindings[m['source_index']];row=binding['row'];assert row['episode_id']==m['episode_id']
        masks=h.old_support(binding,(1024,1024),'cli',hashed,checks)
        path=m['old_region']['old_region_path'];assert h.sha(path)==m['old_region']['old_region_current_sha256']
        with np.load(path,allow_pickle=False) as z:s=h.unpack(z,'canvas/component.query_context',(1024,1024),checks)
        b,g,r=(masks[k] for k in ('B','G','R0'))
        fg=s&b&g&r;bg=~s&~b&~g&~r
        f=fg.reshape(64,16,64,16).mean((1,3)).ravel();a=bg.reshape(64,16,64,16).mean((1,3)).ravel()
        gr=h.read(graph/'records'/f'{i:06}.json');rr=h.read(ridge/'records'/f'{i:06}.json')
        assert gr['episode_id']==rr['episode_id']==m['episode_id']
        assert tensor_sha(f)==gr['seed_sha256']['foreground']==rr['seed_sha256']['query_foreground']
        assert tensor_sha(a)==gr['seed_sha256']['background']==rr['seed_sha256']['query_background']
        checks['actual_graph_ridge_seed_SHA256']+=2
        # This is a mask-label projection, not a full-pipeline loss theorem.
        negative=b&~s
        assert not np.any(negative&fg) and not np.any(negative&bg)
        positive=s&~b
        assert not np.any(positive&fg) and not np.any(positive&bg)
        checks['opposite_S_B_decisions_unlabelled']+=2
        mixed=~((b&g&r)|(~b&~g&~r))
        assert not np.any(mixed&fg) and not np.any(mixed&bg)
        checks['mixed_weak_controls_ignore_any_S_bit_in_role_projection']+=1
        zero=expanded(a.reshape(64,64)==0)
        if a.sum()>0:
            ids=np.searchsorted(np.cumsum(a),(np.arange(128)+.5)*a.sum()/128,side='right')
            sampled=np.zeros(4096,bool);sampled[ids]=True
        else:sampled=np.zeros(4096,bool)
        sampled=expanded(sampled.reshape(64,64))
        gt=h.resize(h.truth(row,checks),(1024,1024),'nearest')>.5
        subsets=dict(S_deleted_B=negative,zero_native_BG_mass=negative&zero,positive_native_BG_mass_not_sampled=negative&~zero&~sampled,native_BG_sampled_via_other_pixels=negative&sampled,
                     S_added_B_foreground=positive,weak_mixed_S_deleted_B=negative&mixed,weak_all_FG_S_deleted_B=negative&~mixed)
        counts={}
        for name,mask in subsets.items():
            counts[name+'/TP']=int((mask&gt).sum());counts[name+'/FP']=int((mask&~gt).sum())
        for role in ('TP','FP'):
            assert sum(counts[k+'/'+role] for k in ('zero_native_BG_mass','positive_native_BG_mass_not_sampled','native_BG_sampled_via_other_pixels'))==counts['S_deleted_B/'+role]
            checks['native_coverage_partition_integer']+=1
        totals.update(counts);cases.append(dict(index=i,episode_id=m['episode_id'],counts=counts))
    old=h.read(ROOT/'evidence/local/strong600_conflict_alignment_20261011/report.json')['gross']['cli']['region_vs_B']
    assert totals['S_deleted_B/TP']==old['delete_TP'] and totals['S_deleted_B/FP']==old['delete_FP']
    assert totals['S_added_B_foreground/TP']==old['add_TP'] and totals['S_added_B_foreground/FP']==old['add_FP']
    report=dict(state='COMPLETE600_OLD_QUERY_ROLE_NEGATIVE_EXPOSURE_AUDIT',n=600,frame='CLI1024',totals=dict(totals),records=cases,checks=dict(checks),query_GT_reads=600,
                new_predictions=0,raw_feature_reads=0,new_encoder_calls=0,new_fits=0,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                interpretation='Every S0/B1 pixel is absent from both old query role labels. A native descriptor may still get background supervision from other pixels; actual128 BG sampler exposure is separated. These two pipelines retain S through host/output, so this is local training-input omission, not total Mask blindness, universal loss, proof of causation or proof new roles improve accuracy. TP omitted negatives would be harmful if blindly relabelledBG. No new candidate frozen.')
    h.write(OUT/'report.json',report)
    print(json.dumps(dict(state=report['state'],totals=dict(totals),checks=dict(checks),wall_seconds=report['wall_seconds'])))


if __name__=='__main__':main()
