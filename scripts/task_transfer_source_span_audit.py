#!/usr/bin/env python3
"""Dense reference control and oracle changes outside sampled reference span.

Reference-only dense fits are sealed before opening query labels. Subsequent
null-space decompositions use query-trained oracle weights: diagnostics only.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_transfer_identifiability as d

np, torch, F, base = d.np, d.torch, d.F, d.base
OUT = base.ASSETS / 'a/task_transfer_source_span60_20261010'
ARMS = ['sparse_reference', 'dense_reference', 'joint_oracle',
        'joint_sampled_rowspace', 'joint_sampled_nullspace', 'joint_nullspace_safe']


@torch.inference_mode()
def fit_dense(x, coverage, allowed, validate=False):
    c = torch.from_numpy(coverage[allowed].astype('float64'))
    if c.sum() == 0 or (1-c).sum() == 0:
        return None
    x = x[allowed].double()
    fg, bg = .5*c/c.sum(), .5*(1-c)/(1-c).sum()
    a = fg + bg
    y = (fg-bg)/a
    mx, my = (a[:, None]*x).sum(0), (a*y).sum()
    xc, yc = x-mx, y-my
    weighted = (xc*a.sqrt()[:, None]).float().to('mps')
    gram = (weighted.T@weighted).cpu().double()
    matrix = gram + d.LAMBDA*torch.eye(x.shape[1], dtype=torch.float64)
    rhs = xc.T@(a*yc)
    w = torch.linalg.solve(matrix, rhs)
    b = my-mx@w
    check = None
    if validate:
        exact = xc.T@(a[:, None]*xc) + d.LAMBDA*torch.eye(x.shape[1], dtype=torch.float64)
        ew = torch.linalg.solve(exact, rhs)
        eb = my-mx@ew
        check = dict(max_gram_error=float((matrix-exact).abs().max()),
                     max_score_error=float(((x@w+b)-(x@ew+eb)).abs().max()))
        assert check['max_score_error'] < 2e-5, check
    return dict(w=w, b=b, diagnostics=dict(n_tokens=len(x), numerical_check=check,
        linear_system_residual=float((matrix@w-rhs).norm()/rhs.norm().clamp_min(1e-15))))


def prepare(out):
    if (out/'config.json').exists():
        return
    d.verify(d.OUT)
    pairs = d.read(d.OUT/'pairs.json')
    assert len(pairs) == 60
    out.mkdir(parents=True, exist_ok=True)
    (out/'frozen').mkdir(exist_ok=True)
    shutil.copyfile(__file__, out/'frozen'/Path(__file__).name)
    base.write(out/'pairs.json', pairs)
    base.write(out/'config.json', dict(n_pairs=60, representation='unit_raw', arms=ARMS,
        lambda_=d.LAMBDA, spatial_split='same8x8-token checkerboard as frozen parent',
        dense_control='All reference train-fold tokens, exact balanced soft-label pixel risk, same linear capacity and regularization',
        decomposition='Augment reference quadrature features with bias; FP64 SVD row/null projection of joint-reference coefficients',
        rank_tolerance='machine epsilon times max(matrix shape) times largest singular value',
        safe_scale='min(1, half earliest positive crossing of any4096 reference score sign); source-only scale, query-label oracle direction',
        warning='Dense-reference fields lawful, but joint and decomposed directions use query labels: oracle diagnosis only, not deployable scores',
        exposure='Same exposed60 development pairs; blocked holdout shares image context, no general impossibility or originality claim',
        new_encodings=0, no_FoRIS=True,
        source_sha256={str(p): base.sha(p) for p in [Path(__file__), Path(d.__file__)]},
        parent_sha256={str(d.OUT/n): base.sha(d.OUT/n) for n in ['config.json','pairs.json','source_sealed.json','report.json','verification.json']},
        query_truth_sha256=base.sha(base.OUT/'query_truth.json'), pairs_sha256=base.sha(out/'pairs.json')))


def verify(out):
    cfg = d.read(out/'config.json')
    for p, digest in (cfg['source_sha256']|cfg['parent_sha256']).items():
        assert base.sha(p) == digest
    assert base.sha(out/'pairs.json') == cfg['pairs_sha256']
    return cfg


def source_stage(out):
    prepare(out)
    verify(out)
    if (out/'source_sealed.json').exists():
        return
    split, start, records = d.split_grid(), time.monotonic(), []
    for folder in ['source_fields','source_records','oracle_fields','oracle_records']:
        (out/folder).mkdir(exist_ok=True)
    for i, pair in enumerate(d.read(out/'pairs.json')):
        rp = out/'source_records'/f'{i:03d}.json'
        if rp.exists():
            rec=d.read(rp); assert base.sha(out/'source_fields'/rec['file'])==rec['sha256']
            records.append(rec); continue
        attempts, fields, fits = [], {}, {}
        with base.deny_labels(base.OUT, attempts):
            (r,q), receipts = d.load_raw(pair)
            for v in ['A','B']:
                rb=pair['variants'][v]; c=d.coverage(rb['path'],rb['file_sha256'])
                for fold in [0,1]:
                    fitted=fit_dense(r,c,split==fold,validate=i in [0,20,40])
                    key=f'{v}/{fold}'
                    if fitted is None:
                        fits[key]=dict(valid=False);continue
                    fields[key+'/dense_reference/r']=d.predict(fitted,r)
                    fields[key+'/dense_reference/q']=d.predict(fitted,q)
                    fits[key]=dict(valid=True,**fitted['diagnostics'])
            assert not attempts
            name=f'{i:03d}.npz';np.savez_compressed(out/'source_fields'/name,**fields)
        rec=dict(id=pair['id'],dataset=pair['dataset'],index=i,file=name,sha256=base.sha(out/'source_fields'/name),
                 fits=fits,raw=receipts,query_label_access_attempts=attempts,new_encodings=0)
        base.write(rp,rec);records.append(rec);torch.mps.empty_cache()
        base.write(out/'activity.json',dict(state='DENSE_REFERENCE_ONLY',n=len(records),total=60,
            seconds=time.monotonic()-start,driver_bytes=torch.mps.driver_allocated_memory(),new_encodings=0))
        if (i+1)%10==0:print(json.dumps(dict(stage='DENSE_REFERENCE',n=i+1)),flush=True)
    base.write(out/'source_sealed.json',dict(n_pairs=60,records=records,query_GT_reads=0,new_encodings=0,
        config_sha256=base.sha(out/'config.json'),seconds=time.monotonic()-start))


def safe_scale(reference_score, change):
    opposite=(reference_score*change)<0
    limits=-reference_score[opposite]/change[opposite]
    if np.any((reference_score==0)&(change!=0)):
        return 0.
    return min(1., .5*float(limits.min())) if len(limits) else 1.


def oracle_stage(out):
    from scipy.linalg import svd
    verify(out)
    seal=d.read(out/'source_sealed.json')
    assert seal['config_sha256']==base.sha(out/'config.json')
    cfg=d.read(out/'config.json');assert base.sha(base.OUT/'query_truth.json')==cfg['query_truth_sha256']
    truth=d.read(base.OUT/'query_truth.json');split=d.split_grid();start=time.monotonic();records=[]
    parent_seal=d.read(d.OUT/'source_sealed.json');pairs=d.read(out/'pairs.json')
    for i,pair in enumerate(pairs):
        rp=out/'oracle_records'/f'{i:03d}.json'
        if rp.exists():
            rec=d.read(rp);assert base.sha(out/'oracle_fields'/rec['file'])==rec['fields_sha256']
            records.append(rec);continue
        (r,q),_=d.load_raw(pair)
        augmented={site:np.column_stack([x.numpy().astype('float64'),np.ones(len(x))]) for site,x in [('r',r),('q',q)]}
        prec=parent_seal['records'][i];assert prec['id']==pair['id']
        assert base.sha(d.OUT/'source_fields'/prec['file'])==prec['sha256']
        orec=d.read(d.OUT/'oracle_records'/f'{i:03d}.json')
        assert base.sha(d.OUT/'oracle_fields'/prec['file'])==orec['oracle_fields_sha256']
        dense=seal['records'][i];assert base.sha(out/'source_fields'/dense['file'])==dense['sha256']
        measures,fields=[],{}
        with np.load(d.OUT/'source_fields'/prec['file']) as old,np.load(d.OUT/'oracle_fields'/prec['file']) as oracle,np.load(out/'source_fields'/dense['file']) as stored_dense:
            for v in ['A','B']:
                rb=pair['variants'][v];qb=truth[pair['id']]['masks'][v]
                cr,cq=d.coverage(rb['path'],rb['file_sha256']),d.coverage(qb['path'],qb['sha256'])
                for fold in [0,1]:
                    key=f'{v}/{fold}';parent_key=f'{v}/unit_raw/{fold}'
                    if parent_key+'/joint_to_q' not in oracle:
                        measures.append(dict(variant=v,fold=fold,valid=False,reason='parent_validity'));continue
                    rs,qs=d.sample(r,cr,split==fold),d.sample(q,cq,split==fold)
                    joint=d.fit([rs,qs]);assert joint is not None
                    theta_r=np.r_[old[parent_key+'/coefficient'],old[parent_key+'/bias']]
                    theta_j=np.r_[joint['w'].numpy(),joint['b'].numpy()]
                    z=augmented['r'][rs[3]]
                    _,s,vh=svd(z,full_matrices=False,check_finite=False)
                    tolerance=np.finfo('float64').eps*max(z.shape)*s[0]
                    basis=vh[s>tolerance];delta=theta_j-theta_r
                    parallel=basis.T@(basis@delta);null=delta-parallel
                    null_training_error=float(np.max(np.abs(z@null)))
                    assert null_training_error<1e-10
                    rbase=augmented['r']@theta_r;rchange=augmented['r']@null
                    alpha=safe_scale(rbase,rchange)
                    safe=rbase+alpha*rchange
                    assert np.array_equal(safe>0,rbase>0)
                    thetas=dict(sparse_reference=theta_r,joint_oracle=theta_j,
                        joint_sampled_rowspace=theta_r+parallel,joint_sampled_nullspace=theta_r+null,
                        joint_nullspace_safe=theta_r+alpha*null)
                    stats={};parent_errors=[]
                    for arm in ARMS:
                        stats[arm]={}
                        for site,coverage in [('r',cr),('q',cq)]:
                            score=stored_dense[key+'/dense_reference/'+site].copy() if arm=='dense_reference' else augmented[site]@thetas[arm]
                            fields[key+'/'+arm+'/'+site]=score
                            value=d.metrics(score,coverage,split!=fold);assert value is not None
                            stats[arm][site]=value
                            if arm in ['sparse_reference','joint_oracle']:
                                parent_name=('r_to_' if arm=='sparse_reference' else 'joint_to_')+site
                                saved=old[parent_key+'/'+parent_name] if arm=='sparse_reference' else oracle[parent_key+'/'+parent_name]
                                parent_errors.append(float(np.max(np.abs(score-saved))))
                    assert max(parent_errors)<2e-5,(pair['id'],v,fold,max(parent_errors))
                    fields[key+'/null_coefficients']=null
                    measures.append(dict(variant=v,fold=fold,valid=True,metrics=stats,sampled_reference_rank=len(basis),
                        sampled_reference_positions=len(z),null_training_logit_error=null_training_error,
                        parent_field_max_error=max(parent_errors),safe_scale=alpha,safe_full_reference_mask_bit_exact=True,
                        null_norm=float(np.linalg.norm(null)),parallel_norm=float(np.linalg.norm(parallel)),
                        full_reference_null_score_rms=float(np.sqrt(np.mean(rchange*rchange))),
                        safe_query_changed_tokens=int(np.count_nonzero((fields[key+'/joint_nullspace_safe/q']>0)!=(fields[key+'/sparse_reference/q']>0)))))
        name=f'{i:03d}.npz';np.savez_compressed(out/'oracle_fields'/name,**fields)
        rec=dict(id=pair['id'],dataset=pair['dataset'],file=name,fields_sha256=base.sha(out/'oracle_fields'/name),
            measures=measures,query_labels_used=True,never_deployable=True)
        base.write(rp,rec);records.append(rec);torch.mps.empty_cache()
        base.write(out/'activity.json',dict(state='ORACLE_SOURCE_SPAN',n=len(records),total=60,
            seconds=time.monotonic()-start,driver_bytes=torch.mps.driver_allocated_memory(),new_encodings=0))
        if (i+1)%10==0:print(json.dumps(dict(stage='ORACLE_SOURCE_SPAN',n=i+1)),flush=True)
    summary={}
    for domain in sorted({r['dataset'] for r in records}):
        sub=[r for r in records if r['dataset']==domain];valid=[m for r in sub for m in r['measures'] if m['valid']]
        domain_stats={}
        for arm in ARMS:
            domain_stats[arm]={}
            for site in ['r','q']:
                domain_stats[arm][site]={}
                for metric in ['auc','soft_iou','balanced_squared_risk']:
                    per_photo=[np.mean([m['metrics'][arm][site][metric] for m in r['measures'] if m['valid']]) for r in sub if any(m['valid'] for m in r['measures'])]
                    change=[np.mean([m['metrics'][arm][site][metric]-m['metrics']['sparse_reference'][site][metric] for m in r['measures'] if m['valid']]) for r in sub if any(m['valid'] for m in r['measures'])]
                    domain_stats[arm][site][metric]=dict(mean=float(np.mean(per_photo)),delta_sparse=base.bootstrap(change))
        summary[domain]=dict(n_pairs=len(sub),valid_task_folds=len(valid),arms=domain_stats,
            mean_reference_sample_rank=float(np.mean([m['sampled_reference_rank'] for m in valid])),
            safe_scale_quantiles=np.quantile([m['safe_scale'] for m in valid],[0,.25,.5,.75,1]).tolist(),
            mean_safe_query_changed_tokens=float(np.mean([m['safe_query_changed_tokens'] for m in valid])),
            max_parent_field_error=max(m['parent_field_max_error'] for m in valid),
            max_null_training_logit_error=max(m['null_training_logit_error'] for m in valid))
    base.write(out/'report.json',dict(state='COMPLETE_ORACLE_SOURCE_SPAN_DIAGNOSIS',datasets=summary,records=records,
        config_sha256=base.sha(out/'config.json'),source_seal_sha256=base.sha(out/'source_sealed.json'),new_encodings=0,
        warning='Oracle changes require query labels. Nullspace is relative to sampled reference training rows, not all reference features. Dense control and safe whole-R decisions prevent overstating source information absence.'))
    lines=['# 参考监督约束与共同oracle上界','',
        '共同模型及分解方向使用query训练标签，是隔离oracle诊断，不能作算法成绩。nullspace仅相对原quadrature参考训练矩阵；另报告全部R token风险、留出及保全R二值预测的安全步长。','',
        '|域|只学R的Q AUC|密集R Q AUC|共同oracle|仅rowspace改进|仅nullspace改进|保全R预测null步长|','|---|---:|---:|---:|---:|---:|---:|']
    for domain,s in summary.items():
        lines.append('|'+domain+'|'+'|'.join(f"{s['arms'][a]['q']['auc']['mean']:.5f}" for a in ARMS)+'|')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    base.write(out/'activity.json',dict(state='DIAGNOSED',n_pairs=60,new_encodings=0))
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();source_stage(a.out);oracle_stage(a.out)


if __name__=='__main__':
    main()
