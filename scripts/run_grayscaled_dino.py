#!/usr/bin/env python3
"""Fixed first4 grayscale/RGB patch matching: all six masks seal before GT score."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
KEYS=('0_0_72','1_0_73','2_0_74','3_0_75')
ARMS=('mean','gray','rgb','brightness_texture','h20','native')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('infer','score'));p.add_argument('--binding',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]='2' if args.stage=='infer' else '1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from ics.methods import grayscaled_dino as method
    torch.set_num_threads(2 if args.stage=='infer' else 1);torch.set_num_interop_threads(1)
    binding=json.loads(args.binding.read_text());rows=binding['cases']
    if tuple(row['id'] for row in rows)!=KEYS:raise ValueError('Exact predetermined first4 required, no GT selection')
    if args.stage=='infer':
        if args.out.exists():raise FileExistsError('Fresh complete-first4 namespace required')
        started=time.perf_counter();assets=binding['encoder_binding'];mdir=Path(binding['model_dir'])
        if sha(mdir/'model.safetensors')!=assets['checkpoint_sha256'] or sha(mdir/'config.json')!=assets['config_sha256']:raise ValueError('Frozen model asset mismatch')
        from ics.data import TimmDINOv3
        encoder=method.ActualPatch128Encoder(TimmDINOv3(str(mdir)))
        eva=Path(inspect.getsourcefile(type(encoder.model)))
        if sha(eva)!=assets['timm_eva_sha256']:raise ValueError('Actual Eva source mismatch')
        sources=[Path(__file__),Path(method.__file__),ROOT/'src/ics/methods/object_crop_cls.py',ROOT/'src/ics/methods/huber_graph.py',ROOT/'src/ics/methods/prepared_cpu_bundle.py',ROOT/'src/ics/data.py',eva]
        args.out.mkdir(parents=True)
        config={'recipe':method.Config().__dict__,'source_hashes':{str(path):sha(path) for path in sources},'encoder_assets':assets,
            'new_view_maximum_forwards':16,'predefined_cases':list(KEYS),'query_GT_in_inference':False,
            'native_original_variant':'cached native1024 binary -> FP32 bilinear trueoriginalHW >.5; not original native CRF value',
            'candidate_independent_increment_before_complete_checks':0,'no_semantic_invariance_claim':True}
        dump(args.out/'config.json',config)
        case_seals=[];total_calls=0
        class LoggedEncoder:
            def __call__(self,image):
                nonlocal total_calls
                tick=time.perf_counter();value=encoder(image);total_calls+=1
                print(json.dumps({'state':'ACTUAL_PATCH128_FORWARD','number':total_calls,'seconds':time.perf_counter()-tick}),flush=True)
                return value
        for row in rows:
            case_started=time.perf_counter();case=args.out/row['id'];case.mkdir()
            inputs={name:sha(row[name]) for name in ('reference_rgb','reference_mask','query_rgb','feature_export','packet_export','h20_export','native_npz')}
            if inputs['h20_export']!=row['h20_export_sha256'] or inputs['native_npz']!=row['native_sha256']:raise ValueError('Bound cache input changed')
            reference=np.asarray(Image.open(row['reference_rgb']).convert('RGB')).copy();query=np.asarray(Image.open(row['query_rgb']).convert('RGB')).copy()
            mask=np.asarray(Image.open(row['reference_mask'])).copy()
            if mask.ndim!=2 or not np.isin(mask,(0,1,255)).all():raise ValueError('Full original binary R mask required')
            feature=torch.load(row['feature_export'],map_location='cpu',weights_only=True,mmap=True)
            q,r=feature['q'].numpy().copy(),feature['r'].numpy().copy();del feature
            with np.load(row['packet_export'],allow_pickle=False) as packet:cov,score=packet['cov'].copy(),packet['score'].copy()
            from ics.methods.huber_graph import make_mean_inputs
            from ics.methods.prepared_cpu_bundle import mean_base
            tick=time.perf_counter();graph,origin=make_mean_inputs(q,r,cov,score);base,base_origin=mean_base(graph,origin.get('graph_storage_dtype','float64'));base_seconds=time.perf_counter()-tick
            with np.load(row['h20_export'],allow_pickle=False) as cache:
                qh,rh=cache['q'].copy(),cache['r'].copy();producer=json.loads(str(cache['producer_json']))
            if qh.shape!=(4096,1024) or rh.shape!=qh.shape or qh.dtype!=np.float32 or producer['H20']!='actual_after_block20':raise ValueError('Explicit FP32 native4096 H20 without prefix required')
            if producer['input_key']!=row['id'] or producer['weights_sha256']!=assets['checkpoint_sha256'] or producer['native_source_sha256']!=assets['timm_eva_sha256']:raise ValueError('H20 producer identity mismatch')
            result=method.predict(reference,mask!=0,query,q,r,base,LoggedEncoder(),q_h20=qh,r_h20=rh,
                producer_binding={'producer':'actualEvaFP32physical128finalpatch8x8_RGB_and_gray','assets':assets,'existing_raw_H20':producer,'input_hashes':inputs})
            with np.load(row['native_npz'],allow_pickle=False) as native:
                nw,no=native['mask_work'].astype(bool),native['mask_original'].astype(bool)
            expected=F.interpolate(torch.from_numpy(nw.astype(np.float32))[None,None],query.shape[:2],mode='bilinear',align_corners=False)[0,0].numpy()>.5
            if nw.shape!=(1024,1024) or no.shape!=query.shape[:2] or not np.array_equal(expected,no):raise ValueError('Native cached1024 explicit renderer mismatch')
            result['masks']['native']={'work':nw,'original':no}
            if tuple(result['masks'])!=ARMS:raise ValueError('All exact six full rows required')
            np.savez_compressed(case/'predictions.npz',**{name+'_'+space:value.astype(np.uint8) for name,masks in result['masks'].items() for space,value in masks.items()})
            np.savez_compressed(case/'fields.npz',**result['fields'])
            np.savez_compressed(case/'margins.npz',**result['margins'])
            if 'tokens' in result:np.savez_compressed(case/'tokens.npz',**result['tokens'])
            for name,digest in inputs.items():
                if sha(row[name])!=digest:raise ValueError('Input asset changed during inference')
            result['info'].update(id=row['id'],input_hashes=inputs,shared_mean_seconds=base_seconds,shared_mean_origin=base_origin,
                case_complete_seconds=time.perf_counter()-case_started,query_GT_opened=False)
            dump(case/'receipt.json',result['info'])
            names=('predictions.npz','fields.npz','margins.npz','receipt.json')+ (('tokens.npz',) if (case/'tokens.npz').exists() else ())
            case_seals.append({'id':row['id'],'original_query_hw':list(query.shape[:2]),'hashes':{name:sha(case/name) for name in names}})
            print(json.dumps({'state':'CASE_PREDICTIONS_WRITTEN_GT_CLOSED','case':row['id'],'case_seconds':result['info']['case_complete_seconds']}),flush=True)
            if row['id']==KEYS[0] and (result['info']['abstention'] or result['info']['original_RGB_gray_exactly_equal']):
                dump(args.out/'degenerate_stop.json',{'first_case':case_seals[0],'state':'FIRST_CASE_EXACT_REPRESENTATION_OR_GEOMETRY_DEGENERACY_NO_EXPANSION','query_GT_opened':False});return
        if total_calls>16:raise RuntimeError('Predetermined physical view budget exceeded')
        summary={'state':'ALL_FOUR_SIX_ARM_PREDICTIONS_SEALED','query_GT_opened':False,'cases':case_seals,
            'config_sha256':sha(args.out/'config.json'),'new_encoder_forwards':total_calls,
            'complete_process_seconds':time.perf_counter()-started,'peak_process_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))}
        dump(args.out/'sealed4.json',summary);print(json.dumps(summary),flush=True)
    else:
        seal=json.loads((args.out/'sealed4.json').read_text())
        if seal['state']!='ALL_FOUR_SIX_ARM_PREDICTIONS_SEALED' or sha(args.out/'config.json')!=seal['config_sha256']:raise ValueError('All first4 predictions must seal before any GT score')
        config=json.loads((args.out/'config.json').read_text())
        for path,digest in config['source_hashes'].items():
            if sha(path)!=digest:raise ValueError('Immutable actual source changed')
        for case in seal['cases']:
            for name,digest in case['hashes'].items():
                if sha(args.out/case['id']/name)!=digest:raise ValueError('Sealed first4 artifact changed')
        metrics={};totals={};diagnostics={}
        from scipy.stats import spearmanr
        for row,sealed_case in zip(rows,seal['cases']):
            annotation=np.asarray(Image.open(row['query_annotation']));original_gt=annotation==row['reference_class_id']
            if list(original_gt.shape)!=sealed_case['original_query_hw']:raise ValueError('Actual original GT/HW mismatch')
            with np.load(row['packet_export'],allow_pickle=False) as packet:work_gt=np.unpackbits(packet['truth']).reshape(1024,1024).astype(bool)
            case=args.out/row['id'];case_metrics={}
            with np.load(case/'predictions.npz',allow_pickle=False) as predictions:
                for name in predictions.files:
                    space='original' if name.endswith('_original') else 'work';truth=original_gt if space=='original' else work_gt
                    pred,base=predictions[name].astype(bool),predictions['mean_'+space].astype(bool)
                    add,delete=pred&~base,base&~pred;intersection,union=int((pred&truth).sum()),int((pred|truth).sum())
                    m={'I':intersection,'U':union,'IoU':intersection/max(union,1),'add_TP':int((add&truth).sum()),'add_FP':int((add&~truth).sum()),'delete_TP':int((delete&truth).sum()),'delete_FP':int((delete&~truth).sum())}
                    case_metrics[name]=m;total=totals.setdefault(name,{'class_IU':{},'add_TP':0,'add_FP':0,'delete_TP':0,'delete_FP':0})
                    iu=total['class_IU'].setdefault(str(row['reference_class_id']),[0,0]);iu[0]+=intersection;iu[1]+=union
                    for key in ('add_TP','add_FP','delete_TP','delete_FP'):total[key]+=m[key]
            metrics[row['id']]=case_metrics
            purity=work_gt.reshape(64,16,64,16).mean((1,3)).ravel();diagnostics[row['id']]={}
            with np.load(case/'margins.npz',allow_pickle=False) as margin:
                for name in margin.files:
                    correlation=spearmanr(margin[name],purity).statistic
                    diagnostics[row['id']][name]={'token_purity_rho_after_all_seals':float(correlation) if np.isfinite(correlation) else None,'no_independent_tokens_claim':True}
        for total in totals.values():total['class_summed_mIoU_percent']=100*float(np.mean([i/max(u,1) for i,u in total['class_IU'].values()]))
        report={'kind':'predetermined_exposed_first4_complete_grayscaleDINO_observation','metrics':metrics,'totals':totals,'diagnostics':diagnostics,
            'query_GT_only_after_all4_seals':True,'fixed_recipe_no_GT_adjustment':True,'new_encoder_forwards':seal['new_encoder_forwards'],
            'generalized_quality':'not established by4 reused development cases','source_and_all_prediction_hashes_verified':True}
        dump(args.out/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
