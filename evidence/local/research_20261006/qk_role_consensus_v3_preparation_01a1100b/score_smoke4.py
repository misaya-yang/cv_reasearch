"""Verify unified QK v3 seal then independently score packet1024 and bound orig GT."""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    for name in ('code','root','run','out'):parser.add_argument('--'+name,required=True,type=Path)
    a=parser.parse_args();started=time.perf_counter()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
    sys.path.insert(0,str(a.code/'src'))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    # Existing scorer is a frozen read-only dependency, not edited by this task.
    from ics.experiment import sha,summarize,unpack
    import importlib.util
    spec=importlib.util.spec_from_file_location('fixed_qk_v3_render',a.code/'src/ics/methods/qk_role_consensus_v3.py')
    frozen_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=frozen_module;spec.loader.exec_module(frozen_module)
    render=frozen_module.render
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    seal=json.loads((a.run/'sealed.json').read_text())
    if seal['state']!='ALL4_FIXED_QK_V3_PREDICTIONS_SEALED_BEFORE_QUERY_GT' or seal['n']!=4:
        raise ValueError('Complete fixed4 seal required')
    for name,digest in seal['files_sha256'].items():
        if sha(a.run/name)!=digest:raise ValueError('Sealed output changed: '+name)
    rows_path=a.root/'bound600_v2/smoke4.json'
    if sha(rows_path)!=seal['rows_sha256']:raise ValueError('Fixed rows changed')
    rows=json.loads(rows_path.read_text())
    eval_run=a.root/'runs/nine_public600_v2';eval_path=eval_run/'evaluation_manifest.json'
    eval_seal=json.loads((eval_run/'sealed.json').read_text())
    if sha(eval_path)!=seal['evaluation_manifest_sha256'] or sha(eval_path)!=eval_seal['evaluation_manifest_sha256']:
        raise ValueError('Fair evaluation manifest changed')
    evaluation=json.loads(eval_path.read_text())
    raw=('agreement','forward','reverse','symmetric_logits','symmetric_margin','direct_H20','direct_final','mean.control')
    names={x:x if x in ('agreement','mean.control') else x+'.control' for x in raw}
    prepared=[]
    for row,source in zip(rows,seal['cases']):
        if row['key']!=source['key']:raise ValueError('Case ordering changed')
        er=next(x for x in evaluation if x['key']==row['key'])
        if er['support']!=row['support'] or er['query']!=row['query'] or er['c']!=row['c']:raise ValueError('Evaluation identity mismatch')
        if sha(source['binding_path'])!=source['binding_sha256'] or sha(row['packet_export'])!=source['packet_sha256']:
            raise ValueError('Native binding or packet changed')
        if sha(source['host_field_path'])!=source['host_sha256']:raise ValueError('Bound base changed')
        for label,spec in er['evaluation_controls'].items():
            if sha(spec['path'])!=spec['sha256']:raise ValueError('Strong baseline changed '+label)
        hw=tuple(source['original_hw']);mapping=source['original_GT_mapping']
        if hw!=tuple(row['query_image_hw']) or hw!=tuple(mapping['image_HW']) or mapping['class_foreground_value']!=row['c']+1:
            raise ValueError('Original geometry/class mapping differs')
        with Image.open(Path('/root/demo4_cache/data/COCO2014')/row['query']) as im:
            if (im.height,im.width)!=hw:raise ValueError('Actual query HW mismatch')
        gt_path=Path(mapping['path'])
        if not mapping['exists'] or not gt_path.is_file():raise ValueError('Original GT not bound; do not pretend original quality')
        # Hash only, do not decode annotation before all4 prediction validation.
        original_gt_sha=sha(gt_path)
        masks={};case=a.run/f'case{source["index"]}'
        with np.load(case/'fields.npz',allow_pickle=False) as data:
            if set(data.files)!=set(raw):raise ValueError('Incomplete field controls')
            fields={k:data[k].copy() for k in raw}
            if any(x.shape!=(64,64) or x.dtype!=np.float32 or not np.isfinite(x).all() for x in fields.values()):
                raise ValueError('Field shape/dtype/finite wrong')
            if any(float(np.max(np.abs(fields[k]-fields['mean.control'])))>.100001 for k in raw if k!='mean.control'):
                raise ValueError('Frozen small-residual bound violated')
        with np.load(case/'masks.npz',allow_pickle=False) as data:
            if set(data.files)!={k+'.'+sz for k in raw for sz in ('work','original')}:raise ValueError('Incomplete masks')
            for k in raw:
                work=unpack(data[k+'.work']);packed=data[k+'.original'];count=hw[0]*hw[1]
                if packed.dtype!=np.uint8 or packed.shape!=((count+7)//8,):raise ValueError('Original packed shape wrong')
                bits=np.unpackbits(packed)
                if bits[count:].any():raise ValueError('Nonzero padding')
                original=bits[:count].reshape(hw).astype(bool)
                ew,eo=render(fields[k],hw)
                if not np.array_equal(ew,work) or not np.array_equal(eo,original):raise ValueError('Frozen renderer mismatch')
                masks[names[k]]={'work':work,'original':original}
        prepared.append((row,er,source,masks,original_gt_sha))
    if a.out.exists():raise ValueError('Fresh independent score folder required')
    a.out.mkdir()
    (a.out/'pre_GT_audit.json').write_text(json.dumps({'state':'ALL4_SEAL_SOURCE_GEOMETRY_RENDERER_AND_GT_MAPPING_VERIFIED_BEFORE_TRUTH_DECODE',
        'n':4,'sealed_files':len(seal['files_sha256']),'seal_sha256':sha(a.run/'sealed.json'),'query_GT_decoded':False,
        'original_GT_sha256':{x[0]['key']:x[4] for x in prepared}},indent=2)+'\n')
    print(json.dumps({'stage':'all4_pre_GT_verification_complete'}),flush=True)
    arrays={size:{} for size in ('work','original')};corrections={size:{} for size in arrays};details=[]
    for row,er,source,masks,gt_sha in prepared:
        with np.load(er['packet_export'],allow_pickle=False) as packet:
            work_truth=unpack(packet['truth']);native=unpack(packet['native'])
        with Image.open(source['original_GT_mapping']['path']) as im:
            original_truth=np.asarray(im)==source['original_GT_mapping']['class_foreground_value']
        hw=tuple(source['original_hw'])
        if original_truth.shape!=hw or sha(source['original_GT_mapping']['path'])!=gt_sha:raise ValueError('Original GT changed/shape mismatch')
        def original_from_work(work):
            return F.interpolate(torch.from_numpy(work.astype(np.float32))[None,None],hw,mode='bilinear',align_corners=False)[0,0].numpy()>.5
        masks['native']={'work':native,'original':original_from_work(native)}
        for label,spec in er['evaluation_controls'].items():
            with np.load(spec['path'],allow_pickle=False) as data:work=unpack(data[spec['key']])
            masks[label]={'work':work,'original':original_from_work(work)}
        for size,truth in (('work',work_truth),('original',original_truth)):
            base=masks['native'][size]
            for label,by_size in masks.items():
                mask=by_size[size];i,u=int((mask&truth).sum()),int((mask|truth).sum())
                arrays[size].setdefault(label,[]).append([i,u]);added=mask&~base;deleted=base&~mask
                corr={'key':row['key'],'c':row['c'],'fold':row['fold'],'batch':str(row.get('batch','unspecified')),
                      'add_TP':int((added&truth).sum()),'delete_FP':int((deleted&~truth).sum()),
                      'delete_TP':int((deleted&truth).sum()),'add_FP':int((added&~truth).sum())}
                corrections[size].setdefault(label,[]).append(corr)
                details.append(dict(corr,arm=label,resolution=size,intersection=i,union=u,IoU=100*i/max(u,1),
                                    truth_pixels=int(truth.sum()),predicted_pixels=int(mask.sum())))
    metric_rows=[dict(r) for r in rows];photo_ids={}
    for row in metric_rows:
        for role in ('reference','support','query'):
            if role not in row:continue
            identity=row.get(role+'_photo_id',row[role]);photo_ids.setdefault(identity,len(photo_ids));row[role]=f'canonical_photo_{photo_ids[identity]}'
    result={}
    for size in arrays:
        values={k:np.asarray(v,np.int64) for k,v in arrays[size].items()}
        report,draws=summarize(metric_rows,values,corrections[size])
        report.update(scope='fixed4 activity check, not population performance/independent confirmation',primary='agreement',
                      frozen_family='QK substantial revision v3',independent_method_increment=0,new_encoder_forwards=0,
                      seal_sha256=sha(a.run/'sealed.json'),GT='packet.truth1024' if size=='work' else 'fixed annotation class+1/HW mapping',
                      native_original_comparison='cached work1024 native/controls mapped with same final bilinear stage; not separately reproduced RGB-original native entry')
        result[size]=report;np.savez_compressed(a.out/(size+'_counts.npz'),**values)
        np.save(a.out/(size+'_bootstrap_photo_draws.npy'),draws)
    combined={'state':'FIXED_QK_V3_SMOKE4_SCORED_AFTER_ALL4_SEAL','threads_score':1,'query_GT_decoded_only_after_unified_seal':True,
              'new_encoder_forwards':0,'n':4,'work':result['work'],'original':result['original'],'cost_seal':seal,
              'score_seconds':time.perf_counter()-started,'score_source_sha256':sha(__file__),
              'not_claimed':['population gain','independent confirmation','independent new method','deployment encoder cost0']}
    (a.out/'report.json').write_text(json.dumps(combined,indent=2)+'\n')
    (a.out/'episode_metrics.json').write_text(json.dumps(details,indent=2)+'\n')
    with (a.out/'episode_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    lines=['# 固定QK-v3 smoke4完整质量结果','','仅4已暴露例；独立增量0；不扩24/600。',
           '1024用同packet.truth，原尺寸用已绑定annotation；原native/control为缓存work映射同最终renderer。','',
           '| Arm | 1024 class-summed mIoU | original class-summed mIoU |','|---|---:|---:|']
    lines += [f'| {k} | {v:.6f} | {result["original"]["scores"][k]:.6f} |' for k,v in result['work']['scores'].items()]
    for size in result:
        lines += ['',f'## {size}: agreement配对比较（CI仅小样本描述）','','| vs | Δ pp | CI95 | up/down/tie |','|---|---:|---|---|']
        for label,value in result[size]['contrasts']['agreement'].items():
            lines.append(f'| {label} | {value["gain"]:+.6f} | {value["ci95"]} | {value["up"]}/{value["down"]}/{value["tie"]} |')
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'state':combined['state'],'work':result['work']['scores'],'original':result['original']['scores'],
                      'agreement_work':result['work']['contrasts']['agreement'],'score_seconds':combined['score_seconds']},indent=2))


if __name__=='__main__':main()
