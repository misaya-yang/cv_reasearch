"""Unify four fixed M5 seals, then 1CPU work packet.truth/original GT scoring."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,hashlib,time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path('/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b')
    run=root/'runs/pro_message_audit_v1';batch=run/'smoke4_fixed_v1'
    rows=json.loads((root/'bound600_v2/smoke4.json').read_text())
    sources=[]
    for index,row in enumerate(rows):
        if index==0:
            directory=run/'outputs_v3_cached';receipt_path=directory/'sealed.json'
            receipt=json.loads(receipt_path.read_text());binding_path=Path(receipt['binding_path'])
            inference_seconds=receipt['inference_seconds'];capture_seconds=json.loads((run/'outputs_v2/episode_00_0_0_72/capture.json').read_text())['paired_capture_seconds']
            errors=receipt['native_all_ROI_layer_maxabs'];count=receipt['native_all_ROI_count'];module=receipt['method_sha256']
            entry_seconds=receipt['complete_cached_entry_seconds'];kind='cache_resume_without_reencoding_case0'
        else:
            directory=batch/f'case{index}'/f'episode_{index:02d}_{row["key"]}'
            receipt_path=directory/'receipt.json';binding_path=directory/'binding.json'
            receipt=json.loads(receipt_path.read_text())
            if receipt['state']!='COMPLETE_PREDICTIONS_SEALED_UNSCORED':raise ValueError('Missing assigned complete prediction')
            inference_seconds=receipt['prediction_seconds'];capture_seconds=receipt['capture']['paired_capture_seconds']
            errors=receipt['all_ROI_native_layer_maxabs'];count=receipt['beta1_all_ROI_count'];module=receipt['method_source_sha256']
            entry_seconds=receipt['episode_complete_seconds'];kind='fresh_paired_capture_and_all_branch_prediction'
        diagnostics=json.loads((directory/'inference_diagnostics.json').read_text())
        native=[a['native'] for a in diagnostics['reference_audits']]
        native += [a['native'] for partition in diagnostics['query_audits'] for a in partition['regions']]
        if (len(native)!=count or len(native)-len(diagnostics['reference_audits'])!=60
                or not all(e['passed'] for a in native for e in a['native_layer_errors'])):
            raise ValueError('All fixed60 query/specifed reference ROI must pass original strict gate')
        for label in ('fields','masks'):
            if sha(directory/(label+'.npz'))!=receipt[label+'_sha256']:raise ValueError('Changed individual sealed output')
        binding=json.loads(binding_path.read_text())
        if binding['row']['key']!=row['key']:raise ValueError('Fixed smoke case identity mismatch')
        annotation=Path('/root/autodl-tmp/datasets/ics/COCO2014/annotations')/Path(row['query']).with_suffix('.png')
        sources.append({'index':index,'key':row['key'],'class':row['c'],'directory':str(directory),
                        'receipt_path':str(receipt_path),'receipt_sha256':sha(receipt_path),
                        'binding_path':str(binding_path),'binding_sha256':sha(binding_path),
                        'fields_sha256':receipt['fields_sha256'],'masks_sha256':receipt['masks_sha256'],
                        'module_sha256':module,'packet_path':row['packet_export'],'packet_sha256':sha(row['packet_export']),
                        'original_GT_mapping':{'path':str(annotation),'class_foreground_value':row['c']+1,
                                               'image_HW':row['query_image_hw'],'exists':annotation.is_file()},
                        'native_ROI_count':count,'native_layer_maxabs':errors,
                        'inference_seconds':inference_seconds,'capture_seconds':capture_seconds,
                        'entry_seconds':entry_seconds,'entry_kind':kind,'peak_RSS_bytes':receipt['peak_RSS_bytes'],
                        'host_variant':receipt.get('host_variant',receipt.get('producer',{}).get('host_variant')),
                        'gate':binding['producer']['projection_gate']})
    if len({x['module_sha256'] for x in sources})!=1:raise ValueError('Method version differed between four cases')
    seal={'state':'ALL_FOUR_FIXED_SMOKE_PREDICTIONS_SEALED','n':4,'source_rows_sha256':sha(root/'bound600_v2/smoke4.json'),
          'cases':sources,'truth_not_opened_by_new_case_inference':True,
          'case0_reused_its_existing_complete_seal':True,'no_extra_cohort_or_variant':True,
          'threads_inference':12,'group_RSS_limit_bytes':16<<30}
    seal_path=batch/'merged_sealed.json';seal_path.write_text(json.dumps(seal,indent=2)+'\n')
    # All query GT reads below this point follow the unified four-output seal.
    started=time.perf_counter();records=[];totals={};original_totals={}
    for source in sources:
        directory=Path(source['directory']);row=rows[source['index']]
        with np.load(source['packet_path'],allow_pickle=False) as packet:
            packed=packet['truth'].copy()
        if packed.shape==(131072,) and packed.dtype==np.uint8:
            work_truth=np.unpackbits(packed).reshape(1024,1024).astype(bool)
        elif packed.shape==(1024,1024) and packed.dtype==bool:work_truth=packed
        else:raise ValueError('Unexpected packet.truth schema')
        original_truth=None;mapping=source['original_GT_mapping'];original_gt_sha=None
        if mapping['exists']:
            with Image.open(mapping['path']) as image:original_truth=np.asarray(image)==mapping['class_foreground_value']
            if original_truth.shape!=tuple(mapping['image_HW']):raise ValueError('Original GT mapping geometry mismatch')
            original_gt_sha=sha(mapping['path'])
        with np.load(directory/'masks.npz',allow_pickle=False) as npz:masks={k:npz[k].copy() for k in npz.files}
        arms=sorted(k[:-5] for k in masks if k.endswith('.work'))
        per_arm={}
        for arm in arms:
            per_arm[arm]={}
            for name,truth in (('work',work_truth),('original',original_truth)):
                if truth is None:continue
                mask=masks[arm+'.'+name];base=masks['zero.'+name]
                if mask.dtype!=bool or mask.shape!=truth.shape:raise ValueError('Complete output shape/type mismatch')
                intersection=int((mask&truth).sum());union=int((mask|truth).sum())
                added=mask&~base;deleted=~mask&base
                values={'intersection':intersection,'union':union,'IoU_percent':100*intersection/max(union,1),
                        'truth_pixels':int(truth.sum()),'prediction_pixels':int(mask.sum()),
                        'add_TP':int((added&truth).sum()),'add_FP':int((added&~truth).sum()),
                        'delete_TP':int((deleted&truth).sum()),'delete_FP':int((deleted&~truth).sum())}
                per_arm[arm][name]=values
                target=totals if name=='work' else original_totals
                pair=target.setdefault(arm,{}).setdefault(source['class'],[0,0]);pair[0]+=intersection;pair[1]+=union
        for arm in arms:
            for name,value in per_arm[arm].items():value['delta_vs_MEAN_pp']=value['IoU_percent']-per_arm['zero'][name]['IoU_percent']
        records.append({'index':source['index'],'key':source['key'],'class':source['class'],
                        'metrics':per_arm,'original_gt_sha256':original_gt_sha})
    def aggregate(target):
        return {arm:{'class_summed_mIoU':float(np.mean([100*i/max(u,1) for i,u in values.values()])),
                     'classes':len(values),'class_IU':{str(c):p for c,p in values.items()}} for arm,values in target.items()}
    work=aggregate(totals)
    original=aggregate(original_totals) if all(s['original_GT_mapping']['exists'] for s in sources) else {}
    for output in (work,original):
        for arm,value in output.items():value['delta_vs_MEAN_pp']=value['class_summed_mIoU']-output['zero']['class_summed_mIoU']
    report={'state':'FOUR_FIXED_SMOKE_CASES_SCORED_AFTER_UNIFIED_SEAL','n':4,'threads_score':1,
            'source_keys':[x['key'] for x in sources],'merged_seal_sha256':sha(seal_path),
            'work_GT':'exact supplied packet.truth','original_GT':'fixed annotation mapping after unified seal where bound',
            'host_variant':sources[0]['host_variant'],'work':work,'original':original,'episodes':records,
            'cost':{'sources':[{k:x[k] for k in ('index','key','capture_seconds','inference_seconds','entry_seconds','entry_kind','peak_RSS_bytes')} for x in sources],
                    'group':json.loads((batch/'group_state.json').read_text())},
            'scoring_seconds':time.perf_counter()-started,'data_exposure':'fixed reused public/dev4; not independent confirmation',
            'not_claimed':['cohort-wide stable gain','novelty','new method count','standalone pure_FP32_host_parity']}
    (batch/'score_smoke4.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':report['state'],'work':work,'original':original,'scoring_seconds':report['scoring_seconds']}))


if __name__=='__main__':main()
