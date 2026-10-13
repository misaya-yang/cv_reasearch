"""A4 oracle of IND-0/IND-N, before spending compute on selector predictors."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import numpy as np
import lab
import plan_common as pc
import plan_report as pr
OUT=lab.ASSETS/'a/x1012_A4';FIELDS=lab.ASSETS/'a/x1012_P0/fields'


def main():
    pc.setup('cpu');OUT.mkdir(exist_ok=True);tasks=pc.seed_rows(150)
    sources=[]
    for b in tasks:
        r=pc.read(FIELDS/'records'/(pc.key(b)+'.json'));assert pc.sha(r['path'])==r['sha256'];sources.append(r)
    pc.write(OUT/'sources_sealed_before_GT.json',{'sources':sources,'new_encoder_calls':0,'confirm_scored':False})
    records=[];groups=defaultdict(list)
    for b,r in zip(tasks,sources):
        with np.load(r['path'],allow_pickle=False) as z:
            masks={a:pc.unpack(z['mask/'+a]) for a in ('IND0','INDN')}
        masks['FoRIS']=lab.foris_cli(b).numpy()
        rec=pr.score_record(b,masks)
        choose='INDN' if rec['metrics']['INDN']['IoU']>rec['metrics']['IND0']['IoU'] else 'IND0'
        rec['metrics']['oracle_episode']=dict(rec['metrics'][choose]);rec['oracle_episode_selected_arm']=choose
        records.append(rec);groups[(rec['dataset'],str(rec['fold']),str(rec['class_id']))].append(rec)
    for _,rows in groups.items():
        # Fractional class-pooled oracle: do not falsely call per-episode choices a class upper bound.
        ratio=sum(r['metrics']['IND0']['I'] for r in rows)/max(sum(r['metrics']['IND0']['U'] for r in rows),1)
        previous=None
        for _ in range(100):
            pick=tuple('INDN' if r['metrics']['INDN']['I']-ratio*r['metrics']['INDN']['U']>r['metrics']['IND0']['I']-ratio*r['metrics']['IND0']['U'] else 'IND0' for r in rows)
            new=sum(r['metrics'][a]['I'] for r,a in zip(rows,pick))/max(sum(r['metrics'][a]['U'] for r,a in zip(rows,pick)),1)
            if pick==previous or abs(new-ratio)<1e-12:break
            previous,ratio=pick,new
        for r,a in zip(rows,pick):r['metrics']['oracle_class']=dict(r['metrics'][a]);r['oracle_class_selected_arm']=a
    arms=('FoRIS','IND0','INDN','oracle_episode','oracle_class')
    result=pr.report('A4',OUT,records,arms,['先计算两个固定自然变体之间的GT选择上界，严格执行计划的上界止损。',
        'oracle_episode沿计划逐例取IoU较高者；另补class-pooled fractional oracle，避免把逐例最优误当按类上界。',
        'oracle只作诊断，读GT，不是可部署选择器；paired区间为已选择oracle掩码的描述性区间。',
        '数据是固定10域已曝光原开发队列，未读取confirm GT。'])
    gaps={d:result['summary'][d]['oracle_class']['delta_to_foris_pp'] for d in ('coco','suim','isic')}
    failed=[d for d,x in gaps.items() if x < -2]
    verdict={'state':'COMPLETE_UPPER_GATE_CLOSED' if failed else 'UPPER_PASSED_PREDICTORS_REQUIRED',
             'upper_delta_to_foris_pp':gaps,'failed_domains':failed,
             'predictor_and_natural_selector_test_skipped':bool(failed),
             'skip_authority':'plan section 5: upper bound below criterion skips natural rule',
             'new_encoder_calls':0,'confirm_scored':False}
    pc.write(OUT/'completion.json',verdict)
    path=lab.ROOT/'docs/research/x1012_tasks/A4.md'
    with path.open('a') as f:
        f.write('\n## 判据\n\n'+str(verdict)+'\n')
        if failed:f.write('\n至少一个关键域的两臂class上界仍低FoRIS超过2pp；按计划§5关闭两臂切换路线，不继续参考预测量、阈值选择或连续调制。\n')
    print(verdict,flush=True)


if __name__=='__main__':main()
