"""One frozen evaluation of the new 500 cache-only reference/query pairs."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from datetime import datetime, timezone
import multiprocessing
import os
from pathlib import Path
import shutil
import time
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[_name]='2'
import numpy as np
import torch
import lab
import plan_common as pc
import plan_report as pr

BASE=lab.ASSETS/'a'
VALIDATION=BASE/'x1012_R5_validation'
CANDIDATE=BASE/'x1012_R5_candidate/final_tau0475'
OUT=BASE/'x1012_R5_results'
ARMS=('FoRIS','MEAN','VP_previous','UN_midrange','candidate')


def initialize():
    pc.setup('cpu');torch.set_num_threads(2)


def score_one(job):
    b,r=job
    with np.load(r['prediction_path'],allow_pickle=False) as z:
        predictions={a:pc.unpack(z[a]) for a in ('candidate','UN_midrange','VP_previous')}
    for arm,old in [('FoRIS','B'),('MEAN','G')]:
        spec=b['old_arms'][old]
        with np.load(spec['path'],allow_pickle=False) as z:predictions[arm]=pc.unpack(z[spec['cli_key']])
    result=pr.score_record(b,predictions)
    result['pair_signature']=b['pair_signature']
    return result


def main():
    initialize();OUT.mkdir(exist_ok=True)
    if (OUT/'report.json').exists():
        raise RuntimeError('This frozen validation is already complete; do not reselect or rescore')
    frozen=OUT/'frozen';frozen.mkdir(exist_ok=True)
    for i,p in enumerate((Path(__file__),Path(pc.__file__),Path(pr.__file__),Path(lab.__file__))):
        target=frozen/f'{i}_{p.name}'
        if target.exists():assert pc.sha(target)==pc.sha(p)
        else:shutil.copyfile(p,target)
    manifest_path=VALIDATION/'manifest.baselines.json'
    rows=pc.read(manifest_path);natural=pc.read(CANDIDATE/'sealed.json')
    baseline=pc.read(VALIDATION/'baseline/sealed.json')
    assert len(rows)==natural['n']==baseline['n']==500
    assert pc.sha(VALIDATION/'manifest.json')==baseline['manifest_sha256']
    assert pc.sha(manifest_path)==baseline['manifest_baselines_sha256']
    records={r['index']:r for r in natural['records']}
    assert len(records)==500 and set(records)=={pc.key(b) for b in rows}
    assert len({b['pair_signature'] for b in rows})==500
    for b in rows:
        r=records[pc.key(b)]
        assert pc.sha(r['prediction_path'])==r['prediction_sha256']
        for key in ('B','G'):
            spec=b['old_arms'][key];assert pc.sha(spec['path'])==spec['sha256']
    selection={'state':'FINAL_METHOD_FROZEN_BEFORE_NEW_PAIR_GT','candidate':'VP_UN_GLOBAL_TAU_0.475',
               'global_tau':.475,'pixel_refinement':False,'reference_rho_gt_point9':'raw_T',
               'selection_rule':'development worst-domain FoRIS delta, then ten-domain mean, then closest to .5',
               'development_gate_passed':False,'reason':'closest actually evaluated complete method; remaining development negatives preserved',
               'development_selection_path':str(BASE/'x1012_R4_global/completion.json'),
               'development_selection_sha256':pc.sha(BASE/'x1012_R4_global/completion.json'),
               'final_config_sha256':pc.sha(CANDIDATE/'config.json'),
               'final_predictions_seal_sha256':pc.sha(CANDIDATE/'sealed.json'),
               'baseline_seal_sha256':pc.sha(VALIDATION/'baseline/sealed.json'),
               'manifest_sha256':pc.sha(VALIDATION/'manifest.json'),
               'bound_manifest_sha256':pc.sha(manifest_path),
               'new_pair_n':500,'old_photographs':True,'new_encoder_calls':0,
               'post_score_parameter_changes_allowed':False}
    selection_path=OUT/'selection.json'
    if selection_path.exists():assert pc.read(selection_path)==selection
    else:pc.write(selection_path,selection)
    marker=OUT/'score_started.json'
    if not marker.exists():
        pc.write(marker,{'started_at_utc':datetime.now(timezone.utc).isoformat(),'selection_sha256':pc.sha(selection_path),
                         'all_natural_hashes_verified_before_GT':True,'entrypoint_sha256':pc.sha(__file__)})
    else:
        assert pc.read(marker)['selection_sha256']==pc.sha(selection_path)
        assert pc.read(marker)['entrypoint_sha256']==pc.sha(__file__),'No scoring repair after GT without preserving attempt'
    started=time.monotonic();scored=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=initialize) as pool:
        for i,r in enumerate(pool.map(score_one,((b,records[pc.key(b)]) for b in rows),chunksize=1),1):
            scored.append(r)
            if i<=3 or i%100==0:print({'phase':'R5-final-score','completed':i,'n':500,'seconds':round(time.monotonic()-started,1)},flush=True)
    notes=[
        '冻结主候选VP_UN_GLOBAL_TAU_0.475：单参考、无新DINO编码、无pixel后处理、无域分支；参考rho>.9原T。',
        '500个未用有向配对、每域50，原照片均来自已有缓存。不是全新照片、不是完整论文基准。',
        '所有候选/诊断/完整FoRIS/MEAN自然输出已封存校验后，才执行本次GT评分。',
        'VP_previous与UN_midrange为事先指定的同配对诊断，不按本验证结果重新选择方法或参数。']
    result=pr.report('R5_validation_final',OUT,scored,ARMS,notes,extra={'selection':selection})
    result['validation_scored']=True;result['post_score_parameter_changes']=False
    comparisons={}
    for name in ('MEAN','VP_previous','UN_midrange'):
        projection=[dict(r,metrics={'FoRIS':r['metrics'][name],'candidate':r['metrics']['candidate']}) for r in scored]
        comparisons['candidate_minus_'+name]=pr.summary(projection,('FoRIS','candidate'))
    pc.write(OUT/'paired_comparisons.json',comparisons)
    negative=[d for d,t in result['summary'].items() if t['candidate']['delta_to_foris_pp']<0]
    positive=[d for d,t in result['summary'].items() if t['candidate']['delta_to_foris_pp']>0]
    positive_ci=[d for d,t in result['summary'].items() if t['candidate']['paired_query_photo_bootstrap95_pp'][0]>0]
    mean_negative=[d for d,t in comparisons['candidate_minus_MEAN'].items() if t['candidate']['delta_to_foris_pp']<0]
    # Separate direct integer aggregation verifies point estimates without re-reading GT.
    for d,t in result['summary'].items():
        for arm,s in t.items():
            totals=defaultdict(lambda:[0,0])
            for r in scored:
                if r['dataset']!=d:continue
                counts=totals[(str(r['fold']),str(r['class_id']))]
                counts[0]+=r['metrics'][arm]['I'];counts[1]+=r['metrics'][arm]['U']
            recomputed=float(np.mean([100*i/max(u,1) for i,u in totals.values()]))
            assert abs(recomputed-s['class_miou'])<1e-9
    result['validation_verdict']={'negative_domains':negative,'positive_domains':positive,
                                  'positive_95CI_domains':positive_ci,'all_ten_positive':len(positive)==10,
                                  'negative_vs_MEAN_domains':mean_negative}
    pc.write(OUT/'report.json',result)
    source=lab.ROOT/'docs/research/x1012_tasks/R5_validation_final.md'
    s=source.read_text().replace('全部为开发/诊断结果，未评分confirm。','本报告为冻结后的500个旧照片新配对一次验证评分，评分后不调参。')
    s+='\n## 冻结验证结论\n\n'+f'相对FoRIS，{len(positive)}域正、{len(negative)}域负；负域：'+','.join(negative)+f'。正向95%区间域：'+','.join(positive_ci)+'。\n'
    s+='\n相对MEAN、旧VP及原.5的照片配对区间见paired_comparisons.json。所有参数保持评分前冻结值。\n'
    source.write_text(s);(OUT/'REPORT.md').write_text(s)
    completion={'state':'COMPLETE_FROZEN_NEW_PAIR_VALIDATION','n':500,'per_domain_n':{d:t['candidate']['n'] for d,t in result['summary'].items()},
                **result['validation_verdict'],'new_encoder_calls':0,'post_score_parameter_changes':False,
                'score_seconds':time.monotonic()-started,'point_estimates_recomputed_from_integers':True,
                'report_sha256':pc.sha(OUT/'report.json'),'selection_sha256':pc.sha(selection_path)}
    pc.write(OUT/'completion.json',completion)
    print(completion,flush=True)


if __name__=='__main__':main()
