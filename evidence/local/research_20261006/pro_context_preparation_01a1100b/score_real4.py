"""One-thread, sealed M1 first4 evaluation; query GT opens only after full audit."""
import argparse, csv, hashlib, json, os, resource, sys, time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for name in ('code','root','run','out'):p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.path.insert(0,str(args.code/'src'))
    import numpy as np
    import torch
    from PIL import Image
    from ics.experiment import sha,summarize,unpack
    from ics.methods.pro_common_context import render
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    started=time.perf_counter()
    if args.out.exists():raise ValueError('Fresh score directory required')
    seal=json.loads((args.run/'seal.json').read_text())
    receipt=json.loads((args.run/'receipt.json').read_text())
    if len(seal['files_sha256'])!=14 or seal['state']!='INFERENCE_FILES_HASHED_BEFORE_QUERY_GT':
        raise ValueError('Expected exact 14-file completed prediction seal')
    if receipt['state']!='FOUR_ARM_PREDICTIONS_SEALED_UNSCORED' or receipt['query_GT_read'] is not False:
        raise ValueError('Inference completion/GT isolation required')
    audit_files={}
    for name,expected in seal['files_sha256'].items():
        f=args.run/name
        got=sha(f)
        if got!=expected:raise ValueError('Sealed file changed: '+name)
        audit_files[name]=got
    smoke_path=args.root/'bound600_v2/smoke4.json'
    if sha(smoke_path)!=receipt['rows_sha256']:raise ValueError('Original smoke input list changed')
    rows=json.loads(smoke_path.read_text())
    eval_run=args.root/'runs/nine_public600_v2'
    eval_path=eval_run/'evaluation_manifest.json'
    eval_seal=json.loads((eval_run/'sealed.json').read_text())
    if sha(eval_path)!=eval_seal['evaluation_manifest_sha256']:
        raise ValueError('Same evaluation manifest no longer matches its seal')
    evaluation=json.loads(eval_path.read_text())
    bindings=[];predictions=[];field_arrays=[]
    raw_arms=('ctx','plain','mean-unit','uniform')
    for index,row in enumerate(rows):
        complete=receipt['results'][index]
        if complete['key']!=row['key'] or seal['keys'][index]!=row['key']:
            raise ValueError('Prediction row ordering changed')
        er=next(r for r in evaluation if r['key']==row['key'])
        for key in ('c','fold','support','query','packet_export'):
            if er[key]!=row[key]:raise ValueError('Evaluation identity differs: '+key)
        if sha(row['packet_export'])!=complete['host_packet_sha256']:
            raise ValueError('Bound query evaluation packet changed')
        for name,spec in er['evaluation_controls'].items():
            if sha(spec['path'])!=spec['sha256']:raise ValueError('Complete baseline changed: '+name)
        original_hw=tuple(complete['query_original_hw'])
        if original_hw!=tuple(row['query_image_hw']):raise ValueError('Original HW metadata differs')
        with Image.open(Path('/root/demo4_cache/data/COCO2014')/row['query']) as image:
            if (image.height,image.width)!=original_hw:raise ValueError('Actual query JPEG HW differs')
        directory=args.run/f'{index:04d}_{row["key"]}'
        pmasks={};fields={}
        with np.load(directory/'fields.npz',allow_pickle=False) as data:
            if set(data.files)!=set(raw_arms):raise ValueError('Incomplete field arms')
            for arm in raw_arms:
                field=data[arm].copy()
                if field.dtype!=np.float32 or field.shape!=(64,64) or not np.isfinite(field).all():
                    raise ValueError('Field dtype/shape/finiteness wrong')
                fields[arm]=field
        with np.load(directory/'masks.npz',allow_pickle=False) as data:
            if set(data.files)!={f'{a}_{size}' for a in raw_arms for size in ('1024','original')}:
                raise ValueError('Incomplete mask arms/shapes')
            for arm in raw_arms:
                work=unpack(data[arm+'_1024'])
                raw=data[arm+'_original'];pixels=original_hw[0]*original_hw[1]
                if raw.dtype!=np.uint8 or raw.shape!=((pixels+7)//8,):raise ValueError('Original packed shape wrong')
                bits=np.unpackbits(raw)
                if bits[pixels:].any():raise ValueError('Original padding nonzero')
                original=bits[:pixels].reshape(original_hw).astype(bool)
                expected_work,expected_original=render(fields[arm],original_hw)
                if not np.array_equal(work,expected_work) or not np.array_equal(original,expected_original):
                    raise ValueError('Stored masks differ from declared frozen two-stage renderer')
                if int(work.sum())!=complete['working_foreground_pixels'][arm]:raise ValueError('Predicted area receipt differs')
                pmasks[arm]=work
        predictions.append(pmasks);field_arrays.append(fields)
        bindings.append(er)
    # Entire 4-row seal/shape/HW/control identity audit completes BEFORE any truth.
    args.out.mkdir()
    audit={'state':'ALL_14_FILES_AND_FOUR_ROWS_VERIFIED_BEFORE_QUERY_GT',
           'files_sha256':audit_files,'inference_seal_sha256':sha(args.run/'seal.json'),
           'evaluation_manifest_sha256':sha(eval_path),'query_GT_read':False,
           'original_HW_verified':True,'two_stage_renderer_bitwise_verified':True}
    (args.out/'pre_GT_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'stage':'pre_GT_audit_complete','files':14,'rows':4}),flush=True)
    metric_names={'ctx':'ctx','plain':'plain.control','mean-unit':'mean-unit.control','uniform':'uniform.control'}
    arrays={};corrections={};details=[];pair_edits=[]
    for row,er,predicted in zip(rows,bindings,predictions):
        with np.load(er['packet_export'],allow_pickle=False) as packet:
            truth=unpack(packet['truth']);native=unpack(packet['native'])
        masks={'native':native,**{metric_names[a]:m for a,m in predicted.items()}}
        for name,spec in er['evaluation_controls'].items():
            with np.load(spec['path'],allow_pickle=False) as control:masks[name]=unpack(control[spec['key']])
        for arm,mask in masks.items():
            intersection,union=int((mask&truth).sum()),int((mask|truth).sum())
            arrays.setdefault(arm,[]).append([intersection,union])
            add,delete=mask&~native,native&~mask
            corr=dict(key=row['key'],c=row['c'],fold=row['fold'],batch=str(row.get('batch','unspecified')),
                      add_TP=int((add&truth).sum()),delete_FP=int((delete&~truth).sum()),
                      delete_TP=int((delete&truth).sum()),add_FP=int((add&~truth).sum()))
            corrections.setdefault(arm,[]).append(corr)
            details.append(dict(corr,arm=arm,intersection=intersection,union=union,
                                IoU=100*intersection/max(union,1),truth_area=int(truth.sum()),
                                predicted_area=int(mask.sum())))
        for arm in metric_names.values():
            for base,bmask in masks.items():
                if base==arm:continue
                mask=masks[arm];add,delete=mask&~bmask,bmask&~mask
                pair_edits.append(dict(key=row['key'],arm=arm,base=base,
                    add_TP=int((add&truth).sum()),delete_FP=int((delete&~truth).sum()),
                    delete_TP=int((delete&truth).sum()),add_FP=int((add&~truth).sum())))
    metric_rows=[dict(r) for r in rows];identity_to_id={}
    for row in metric_rows:
        for role in ('support','query'):
            identity=row.get(role+'_photo_id',row[role]);identity_to_id.setdefault(identity,len(identity_to_id))
            row[role]=f'canonical_photo_{identity_to_id[identity]}'
    arrays={name:np.asarray(v,dtype=np.int64) for name,v in arrays.items()}
    report,draws=summarize(metric_rows,arrays,corrections)
    report.update(state='SEALED_FIRST4_SCORED_1024_ONLY',primary='ctx',new_encoder_forwards=0,threads=1,
                  source_variant='cachehost_new_FP32_native_graph',predictions_sealed_before_scoring=True,
                  exposure='same exposed smoke4; activity/quality check only, not population evidence',
                  inference_receipt_sha256=sha(args.run/'receipt.json'),prediction_seal_sha256=sha(args.run/'seal.json'),
                  evaluation_manifest_sha256=sha(eval_path),score_source_sha256=sha(__file__),
                  original_HW_verified=True,original_resolution_quality='not_scored; original query GT not bound',
                  ground_truth_source='existing per-episode packet.truth packed1024, identical valid pixels for all arms',
                  inference_wall_seconds=receipt['wall_seconds'],inference_peak_RSS_kib=receipt['ru_maxrss_linux_kib'],
                  bootstrap_limit='four classes/photo groups, absent sampled classes handled by shared summarize; CI descriptive only',
                  score_seconds=time.perf_counter()-started,interpretation='No parameter changes or new candidates. Root decides a finite24 after complete result/cost.')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'episode_metrics.json').write_text(json.dumps(details,indent=2)+'\n')
    (args.out/'pairwise_edits.json').write_text(json.dumps(pair_edits,indent=2)+'\n')
    np.savez_compressed(args.out/'counts.npz',**arrays);np.save(args.out/'bootstrap_photo_draws.npy',draws)
    with (args.out/'episode_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    lines=['# M1 首4真实四臂质量与成本检查','',
           '14文件seal、形状、原HW及两次renderer已先于GT核对。只计1024同packet GT，原尺寸未计分。',
           '4例/4类别是活动检查，已暴露且非总体效果；bootstrap区间仅描述性。','',
           '| Arm | 1024 class-summed mIoU |','|---|---:|']
    lines += [f'| {arm} | {score:.6f} |' for arm,score in report['scores'].items()]
    lines += ['','| ctx versus | pp | paired descriptive 95% CI | up/down |','|---|---:|---|---|']
    for base,item in report['contrasts']['ctx'].items():
        lines.append(f'| {base} | {item["gain"]:+.6f} | {item["ci95"]} | {item["up"]}/{item["down"]} |')
    lines += ['',f'推断4线程完整四臂共{receipt["wall_seconds"]:.3f}s；peak RSS {receipt["ru_maxrss_linux_kib"]}KiB。评分1线程、无新encoder前向。',
              '', '完整逐例I/U及四种增删见episode_metrics.json；相对各简单/完整比较行的增删见pairwise_edits.json。']
    (args.out/'report.md').write_text('\n'.join(lines)+'\n')
    for arm in metric_names.values():
        card={'method':arm,'scope':'first4 activity check, 1024 existing packet protocol',
              'score':report['scores'][arm],'contrasts':report['contrasts'][arm],
              'corrections_vs_native':report['corrections_vs_native'][arm],
              'cost_receipt':'real_predict4_v1.json','quality_claim':'no population or acceptance inference',
              'next_action_owner':'root; do not automatically start24/600 or alter parameters'}
        (args.out/(arm.replace('.','_')+'_card.json')).write_text(json.dumps(card,indent=2)+'\n')
    print(json.dumps({'state':report['state'],'scores':report['scores'],'ctx_contrasts':report['contrasts']['ctx'],
                      'score_seconds':report['score_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
