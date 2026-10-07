#!/usr/bin/env python3
"""One fixed600 grayscale inference/score, optionalH20 omitted, no GT during infer."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import resource

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
PRIMARY=('mean','gray','rgb','brightness_texture')
GENERATED=PRIMARY+('standalone_gray','standalone_rgb')
ARMS=GENERATED+('native',)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def rows(path):
    data=json.loads(Path(path).read_text());return data if isinstance(data,list) else data['episodes']


def array_mask(value,shape):
    import numpy as np
    a=np.asarray(value)
    if a.dtype==np.uint8 and a.ndim==1 and a.size==(int(np.prod(shape))+7)//8:
        return np.unpackbits(a,count=int(np.prod(shape))).reshape(shape).astype(bool)
    if a.shape==tuple(shape) and np.isin(a,(0,1)).all():return a.astype(bool)
    raise ValueError('Explicit bound binary/packed mask shape required')


def model(binding):
    import torch
    from ics.data import TimmDINOv3
    from ics.methods.grayscaled_dino import ActualPatch128Encoder
    assets=binding['encoder_binding'];mdir=Path(binding['model_dir'])
    if sha(mdir/'model.safetensors')!=assets['checkpoint_sha256'] or sha(mdir/'config.json')!=assets['config_sha256']:raise ValueError('Frozen128 checkpoint/config mismatch')
    encoder=ActualPatch128Encoder(TimmDINOv3(str(mdir)));source=Path(inspect.getsourcefile(type(encoder.model)))
    if sha(source)!=assets['timm_eva_sha256']:raise ValueError('Actual loaded Eva source mismatch')
    return encoder,source


def infer_case(row,encoder,assets):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from ics.methods import grayscaled_dino as method
    from ics.methods.huber_graph import make_mean_inputs
    from ics.methods.prepared_cpu_bundle import mean_base
    started=time.perf_counter()
    inputs={key:sha(row[key]) for key in ('reference_rgb','reference_mask','query_rgb','feature_export','packet_export','native_npz')}
    reference=np.asarray(Image.open(row['reference_rgb']).convert('RGB')).copy();query=np.asarray(Image.open(row['query_rgb']).convert('RGB')).copy()
    mr=np.asarray(Image.open(row['reference_mask'])).copy()
    if mr.ndim!=2 or not np.isin(mr,(0,1,255)).all():raise ValueError('Bound complete original binary reference mask required')
    f=torch.load(row['feature_export'],map_location='cpu',weights_only=True,mmap=True)
    q,r=f['q'].numpy().copy(),f['r'].numpy().copy();del f
    with np.load(row['packet_export'],allow_pickle=False) as p:
        if not set(p.files).issubset({'cov','score','native'}):raise ValueError('Infer only GT-free packet is allowed')
        cov,score=p['cov'].copy(),p['score'].copy()
    tick=time.perf_counter();graph,origin=make_mean_inputs(q,r,cov,score);base,base_origin=mean_base(graph,origin.get('graph_storage_dtype','float64'));base_seconds=time.perf_counter()-tick
    result=method.predict(reference,mr!=0,query,q,r,base,encoder,
        producer_binding={'producer':'actualEva_FP32_wholephysical128_RGB_and_grayscale_final8patchgrid','assets':assets,'input_hashes':inputs},q_h20=None,r_h20=None)
    if 'h20' in result['fields']:raise RuntimeError('Absent H20 must never be silently substituted')
    for view in ('gray','rgb'):
        field=base.copy() if result['info']['abstention'] else (.5+.5*np.tanh(result['margins'][view]/.07)).reshape(64,64)
        result['fields']['standalone_'+view]=field
        result['masks']['standalone_'+view]=dict(zip(('work','original'),method.render(field,query.shape[:2])))
    with np.load(row['native_npz'],allow_pickle=False) as p:
        nw=array_mask(p['mask_work'],(1024,1024));no=array_mask(p['mask_original'],query.shape[:2])
    expected=F.interpolate(torch.from_numpy(nw.astype(np.float32))[None,None],query.shape[:2],mode='bilinear',align_corners=False)[0,0].numpy()>.5
    if not np.array_equal(no,expected):raise ValueError('Explicit native cached1024 original renderer mismatch')
    result['masks']['native']={'work':nw,'original':no}
    if row.get('historical_mean'):
        old=row['historical_mean'];inputs['historical_mean']=sha(old['path'])
        with np.load(old['path'],allow_pickle=False) as p:work=array_mask(p[old['key']],(1024,1024))
        original=F.interpolate(torch.from_numpy(work.astype(np.float32))[None,None],query.shape[:2],mode='bilinear',align_corners=False)[0,0].numpy()>.5
        result['masks']['historical_mean']={'work':work,'original':original}
    for key,digest in inputs.items():
        path=row['historical_mean']['path'] if key=='historical_mean' else row[key]
        if sha(path)!=digest:raise ValueError('Frozen source asset changed during case')
    result['info'].update(input_hashes=inputs,shared_mean_seconds=base_seconds,shared_mean_origin=base_origin,
        case_complete_seconds=time.perf_counter()-started,query_GT_opened=False,
        native_original_variant='cached1024binary_secondstage_trueRGBHW_bilinear>.5; not original native CRF',
        standalone_readout='.5+.5*tanh(same margin/.07); no new encoding; no ref classes -> explicit MEAN abstention',
        standalone_is_control=True)
    return result,tuple(query.shape[:2])


def save_case(out,result,shape):
    import numpy as np
    out.mkdir()
    np.savez_compressed(out/'predictions.npz',**{name+'_'+space:np.packbits(value.ravel()) for name,masks in result['masks'].items() for space,value in masks.items()},original_hw=np.array(shape,np.int32))
    np.savez_compressed(out/'fields.npz',**result['fields']);np.savez_compressed(out/'margins.npz',**result['margins'])
    dump(out/'receipt.json',result['info'])
    names=('predictions.npz','fields.npz','margins.npz','receipt.json')
    return {'original_query_hw':list(shape),'hashes':{name:sha(out/name) for name in names},'new_encoder_forwards':result['info']['new_encoder_forwards']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('parity','worker','seal','score'));p.add_argument('--binding',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--worker-index',type=int);p.add_argument('--workers',type=int,default=4);p.add_argument('--old-first4',type=Path);p.add_argument('--evaluation',type=Path)
    args=p.parse_args();threads=2 if args.stage in ('worker','parity') else 1
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]=str(threads)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    binding=json.loads(args.binding.read_text());cases=rows(binding['inference_manifest']);assets=binding['encoder_binding']
    if args.stage in ('worker','parity'):
        encoder,eva_source=model(binding)
        print(json.dumps({'state':'FROZEN_MODEL_READY','stage':args.stage,'threads':threads}),flush=True)
        if args.stage=='parity':
            if args.out.exists() or args.old_first4 is None:raise ValueError('Fresh explicit parity output and actual original first4 required')
            args.out.mkdir();checks=[];calls=0
            for row in cases:
                result,shape=infer_case(row,encoder,assets);key=row['id'];save_case(args.out/key,result,shape);calls+=result['info']['new_encoder_forwards']
                with np.load(args.old_first4/key/'fields.npz',allow_pickle=False) as oldfields, np.load(args.old_first4/key/'predictions.npz',allow_pickle=False) as oldmasks:
                    for arm in PRIMARY:
                        maximum=float(np.abs(result['fields'][arm]-oldfields[arm]).max())
                        xor={space:int(np.count_nonzero(result['masks'][arm][space]!=oldmasks[arm+'_'+space].astype(bool))) for space in ('work','original')}
                        checks.append({'id':key,'arm':arm,'field_maximum_difference':maximum,'mask_XOR':xor})
                        if maximum!=0 or any(xor.values()):raise RuntimeError('Original fixed primary recipe/source output parity failed; do not start600')
            summary={'state':'ALL_FIRST4_PRIMARY_FIELDS_EXACT0_MASK_XOR0','checks':checks,'new_actual128_forwards':calls,'H20_control':'explicit omit, not measured','standalone':'same margins, extra encoding0'}
            dump(args.out/'parity.json',summary);print(json.dumps(summary),flush=True);return
        if args.worker_index is None or not 0<=args.worker_index<args.workers:raise ValueError('Explicit owned worker index required')
        config=json.loads((args.out/'config.json').read_text())
        for source,digest in config['source_hashes'].items():
            if sha(source)!=digest:raise ValueError('Frozen600 source changed')
        began=time.perf_counter();completed=0;newcalls=0;seals={}
        for index,row in enumerate(cases):
            if index%args.workers!=args.worker_index:continue
            result,shape=infer_case(row,encoder,assets);name=row['occurrence_id'];seals[name]=save_case(args.out/'cases'/name,result,shape)
            completed+=1;newcalls+=result['info']['new_encoder_forwards']
            print(json.dumps({'state':'CASE_WRITTEN_GT_CLOSED','worker':args.worker_index,'completed':completed,'source_index':index,'occurrence':name,'new_actual128_forwards':newcalls,'case_seconds':result['info']['case_complete_seconds']}),flush=True)
        dump(args.out/f'worker{args.worker_index}.json',{'state':'WORKER_ALL_PREDICTIONS_CLOSED_GT_UNREAD','completed':completed,'new_encoder_forwards':newcalls,'cases':seals,'wall_after_model_seconds':time.perf_counter()-began,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)})
        return
    if args.stage=='seal':
        if len(cases)!=600:raise ValueError('Exactly fixed600 required')
        config=json.loads((args.out/'config.json').read_text());records={};calls=0
        for index in range(args.workers):
            data=json.loads((args.out/f'worker{index}.json').read_text());calls+=data['new_encoder_forwards'];records.update(data['cases'])
        if set(records)!=set(row['occurrence_id'] for row in cases):raise ValueError('All600 fixed occurrences must exist before seal')
        for occurrence,record in records.items():
            for name,digest in record['hashes'].items():
                if sha(args.out/'cases'/occurrence/name)!=digest:raise ValueError('Output changed before unifiedseal')
        dump(args.out/'sealed600.json',{'state':'ALL_FIXED600_PREDICTIONS_SEALED_QUERY_GT_UNOPENED','query_GT_opened':False,'cases':records,'new_encoder_forwards':calls,'config_sha256':sha(args.out/'config.json'),'inference_manifest_sha256':sha(binding['inference_manifest'])})
        print(json.dumps({'sealed600':True,'actual_new128_forwards':calls,'H20':'omitted_not_measured'}));return
    if args.evaluation is None:raise ValueError('Evaluation mapping required ONLY for post-seal score')
    started=time.perf_counter();seal=json.loads((args.out/'sealed600.json').read_text());config=json.loads((args.out/'config.json').read_text())
    if seal['state']!='ALL_FIXED600_PREDICTIONS_SEALED_QUERY_GT_UNOPENED' or sha(args.out/'config.json')!=seal['config_sha256'] or sha(binding['inference_manifest'])!=seal['inference_manifest_sha256']:raise ValueError('All frozen600 masks must seal before anyGT')
    for source,digest in config['source_hashes'].items():
        if sha(source)!=digest:raise ValueError('Frozen source changed')
    for occurrence,record in seal['cases'].items():
        for name,digest in record['hashes'].items():
            if sha(args.out/'cases'/occurrence/name)!=digest:raise ValueError('Any fixed prediction changed')
    evaluation=rows(args.evaluation);byid={row['occurrence_id']:row for row in evaluation}
    if set(byid)!=set(seal['cases']):raise ValueError('Evaluation exactly same600 occurrences required')
    from PIL import Image
    from ics.experiment import summarize
    spaces={space:{'arrays':{},'native_edits':{},'mean_edits':{},'details':[]} for space in ('work','original')};score_rows=[]
    for row in cases:
        key=row['occurrence_id'];ev=byid[key];shape=seal['cases'][key]['original_query_hw']
        annotation=np.asarray(Image.open(ev['query_annotation']));original_gt=annotation==ev['reference_class_id']
        if list(original_gt.shape)!=shape:raise ValueError('Real originalGT shape mismatch; no GT resize permitted')
        if ev.get('source_packet_sha256') and sha(ev['source_packet'])!=ev['source_packet_sha256']:raise ValueError('Evaluation packet identity changed')
        with np.load(ev['source_packet'],allow_pickle=False) as p:work_gt=array_mask(p['truth'],(1024,1024))
        source={'c':int(ev['reference_class_id'])-1,'fold':row.get('fold',-1),'batch':str(row.get('batch','public0')),'support':row['reference_rgb'],'query':row['query_rgb'],'key':key};score_rows.append(source)
        with np.load(args.out/'cases'/key/'predictions.npz',allow_pickle=False) as p:
            names=sorted(name.removesuffix('_work') for name in p.files if name.endswith('_work'))
            for space,truth in (('work',work_gt),('original',original_gt)):
                masks={name:array_mask(p[name+'_'+space],(1024,1024) if space=='work' else shape) for name in names}
                for name,mask in masks.items():
                    arm='gray' if name=='gray' else ('native' if name=='native' else name+'.control');m=spaces[space]
                    iu=[int((mask&truth).sum()),int((mask|truth).sum())];m['arrays'].setdefault(arm,[]).append(iu)
                    edits={}
                    for baseline,label in (('native','native_edits'),('mean','mean_edits')):
                        add,delete=mask&~masks[baseline],masks[baseline]&~mask
                        edit={**source,'add_TP':int((add&truth).sum()),'add_FP':int((add&~truth).sum()),'delete_TP':int((delete&truth).sum()),'delete_FP':int((delete&~truth).sum())}
                        m[label].setdefault(arm,[]).append(edit);edits[baseline]={k:edit[k] for k in ('add_TP','add_FP','delete_TP','delete_FP')}
                    m['details'].append({'occurrence':key,'arm':arm,'I':iu[0],'U':iu[1],'edits':edits})
    reports={}
    for space,data in spaces.items():
        arrays={key:np.asarray(value,np.int64) for key,value in data['arrays'].items()}
        if any(len(value)!=600 for value in arrays.values()):raise ValueError('Historical/current controls must be complete and paired; no silent subset')
        summary,draws=summarize(score_rows,arrays,data['native_edits'])
        summary['corrections_vs_current_MEAN']={arm:{k:int(sum(row[k] for row in values)) for k in ('add_TP','add_FP','delete_TP','delete_FP')} for arm,values in data['mean_edits'].items()}
        reports[space]=summary;dump(args.out/f'episode_metrics_{space}.json',data['details']);np.save(args.out/f'bootstrap_photo_draws_{space}.npy',draws)
    report={'spaces':reports,'source_prediction_seal_SHA256':sha(args.out/'sealed600.json'),'query_GT_after_all600_seals':True,'H20_control':'NOT_MEASURED_NO_BOUND_H20_IN_FULL600','primary_recipe_unchanged_after_FIRST4':True,'standalone_controls':'same margin fixed .5+.5*tanh/.07,extraencoder0; not independentmethods','candidate_increment':1,'actual_new128_forwards':seal['new_encoder_forwards'],'score_seconds':time.perf_counter()-started,'native_original':'cached1024 Prosecondstageadaptation,not originalnativeCRF','exposure':'reusedpublic600development;notindependentconfirmation'}
    dump(args.out/'report.json',report);print(json.dumps({'score_complete':True,'work_scores':reports['work']['scores'],'original_scores':reports['original']['scores'],'seconds':report['score_seconds']}))


if __name__=='__main__':main()
