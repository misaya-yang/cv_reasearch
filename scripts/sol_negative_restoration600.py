"""Join saved predictions with exact BG exposure; post-seal diagnosis only."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/local/negative_restoration600_20261011'
METHODS={'graph':('query_task_graph600_20261011','query_task_graph_complete'),
         'discriminant':('task_discriminant600_20261011','joint_task_discriminant_complete')}

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'report.json').exists()
    start,cpu=time.monotonic(),time.process_time();torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('exposure',ROOT/'scripts/sol_strong_negative_exposure600.py')
    e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    spec=importlib.util.spec_from_file_location('helper',ROOT/'evidence/local/reference_background_conflict6200_20261011/score_and_verify.py')
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    members=h.read(ROOT/'evidence/local/strong600_conflict_alignment_20261011/membership.json')
    bindings=h.lines(ROOT/'evidence/local/task_conflict_execution6200_20261011/score_bindings.jsonl')
    previous=h.read(ROOT/'evidence/local/strong_veto_reversal600_20261011/report.json')
    exposure=h.read(ROOT/'evidence/local/strong_negative_exposure600_20261011/report.json')
    seals={k:h.read(ROOT/'evidence/local'/d/'sealed.json') for k,(d,_) in METHODS.items()}
    # Authenticate all sealed outputs before reading query truth.
    for name,(directory,_) in METHODS.items():
        for i,m in enumerate(members):
            rec=seals[name]['records'][i]
            assert rec['index']==i and rec['episode_id']==m['episode_id']
            assert h.sha(ROOT/'evidence/local'/directory/'predictions'/rec['filename'])==rec['prediction_sha256']
    checks=Counter();hashed=set();totals=Counter();records=[]
    for i,m in enumerate(members):
        binding=bindings[m['source_index']];row=binding['row'];assert row['episode_id']==m['episode_id']
        support=h.old_support(binding,(1024,1024),'cli',hashed,checks)
        p=m['old_region'];assert h.sha(p['old_region_path'])==p['old_region_current_sha256']
        with np.load(p['old_region_path'],allow_pickle=False) as z:s=h.unpack(z,'canvas/component.query_context',(1024,1024),checks)
        b,g,r=(support[k] for k in ('B','G','R0'))
        fg=s&b&g&r;bg=~s&~b&~g&~r
        f=fg.reshape(64,16,64,16).mean((1,3)).ravel();a=bg.reshape(64,16,64,16).mean((1,3)).ravel()
        for name,(directory,_) in METHODS.items():
            rec=h.read(ROOT/'evidence/local'/directory/'records'/f'{i:06}.json')['seed_sha256']
            assert e.tensor_sha(f)==rec['foreground' if name=='graph' else 'query_foreground']
            assert e.tensor_sha(a)==rec['background' if name=='graph' else 'query_background'];checks['seed_identity']+=2
        selected=np.zeros(4096,bool)
        if a.sum()>0:selected[np.searchsorted(np.cumsum(a),(np.arange(128)+.5)*a.sum()/128,side='right')]=True
        expand=lambda x:np.repeat(np.repeat(x.reshape(64,64),16,0),16,1)
        zero=expand(a==0);sampled=expand(selected)
        groups={'zero_BG_mass':zero,'BG_mass_unsampled128':~zero&~sampled,'BG_sampled128':sampled}
        gt=h.resize(h.truth(row,checks),(1024,1024),'nearest')>.5;negative=b&~s;counts={}
        for group,mask in groups.items():
            for role,target in [('TP',gt),('FP',~gt)]:counts[f'denominator/{group}/{role}']=int((negative&mask&target).sum())
        for name,(directory,arm) in METHODS.items():
            rec=seals[name]['records'][i]
            with np.load(ROOT/'evidence/local'/directory/'predictions'/rec['filename'],allow_pickle=False) as z:pred=h.unpack(z,'cli/'+arm,(1024,1024),checks)
            restore=negative&pred
            for group,mask in groups.items():
                for role,target in [('TP',gt),('FP',~gt)]:counts[f'{name}/{group}/{role}']=int((restore&mask&target).sum())
            for role in ('TP','FP'):
                assert sum(counts[f'{name}/{group}/{role}'] for group in groups)==previous['records'][i]['frames']['cli']['methods'][name]['reverse_S_B_'+role];checks['prior_per_case_restore_reconciliation']+=1
        old=exposure['records'][i]['counts']
        for role in ('TP','FP'):
            assert counts[f'denominator/zero_BG_mass/{role}']==old['zero_native_BG_mass/'+role]
            assert counts[f'denominator/BG_mass_unsampled128/{role}']==old['positive_native_BG_mass_not_sampled/'+role]
            assert counts[f'denominator/BG_sampled128/{role}']==old['native_BG_sampled_via_other_pixels/'+role];checks['prior_exposure_reconciliation']+=3
        totals.update(counts);records.append(dict(index=i,episode_id=m['episode_id'],counts=counts))
    rates={}
    for name in METHODS:
        rates[name]={}
        for group in groups:
            rates[name][group]={role:totals[f'{name}/{group}/{role}']/totals[f'denominator/{group}/{role}'] if totals[f'denominator/{group}/{role}'] else None for role in ('TP','FP')}
    report=dict(n=600,frame='CLI1024',totals=dict(totals),restoration_rates=rates,records=records,checks=dict(checks),
        wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        raw_reads=0,new_predictions=0,new_fits=0,query_GT_reads=600,
        scope='Observational saved-mask join, not a intervention or causal claim. Graph uses all positive BG mass; 128 sampling applies only to discriminant. No new candidate.')
    h.write(OUT/'report.json',report)
    print(json.dumps({k:report[k] for k in ('totals','restoration_rates','checks','wall_seconds')}))

if __name__=='__main__':main()
