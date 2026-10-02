"""Frozen student task evaluation; cluster by image/first-frame sequence."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from takeover_train_rank import get_arrays,Ranker,SEEDS,VARIANTS
from takeover_flip_score import bootstrap_image_groups


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    data=get_arrays(a.features);torch.set_num_threads(2)
    x=torch.from_numpy(data['features']);predictions={k:data[k] for k in ('head','largest','teacher_choice')}
    predictions.update({f'fixed_token{i+1}':np.full(len(x),i) for i in range(3)})
    for variant in VARIANTS:
        for seed in SEEDS:
            state=torch.load(a.models/f'{variant}_seed{seed}/best.pt',weights_only=True,map_location='cpu')
            model=Ranker(state['dim']).eval();model.load_state_dict(state['model'])
            with torch.inference_mode():predictions[f'{variant}_seed{seed}']=model((x[...,:state['dim']]-state['mean'])/state['std']).argmax(-1).numpy()
    report={'status':'COMPLETED_FROZEN_STUDENT_TEST','features':str(a.features),'models':str(a.models),'no_test_tuning':True,'summary':{},'rows':[]}
    q=data['quality'];bq=data.get('boundary_quality')
    for regime_id,regime in enumerate(('central','near_boundary','box')):
        idx=np.flatnonzero(data['regimes']==regime_id);report['summary'][regime]={}
        if not len(idx):continue
        for name,choice in predictions.items():
            values=q[idx,choice[idx]];baseline=q[idx,data['head'][idx]];largest=q[idx,data['largest'][idx]]
            rows=[{'image_id':int(data['image_ids'][i]),'delta':float(v-bas),'vs_largest':float(v-lg)} for i,v,bas,lg in zip(idx,values,baseline,largest)]
            summary={'mean_iou':float(values.mean()),'delta_vs_head':float((values-baseline).mean()),'image_cluster_vs_head95':bootstrap_image_groups(rows,'delta'),'delta_vs_largest':float((values-largest).mean()),'image_cluster_vs_largest95':bootstrap_image_groups(rows,'vs_largest')}
            if bq is not None:summary['mean_boundary_iou']=float(bq[idx,choice[idx]].mean())
            report['summary'][regime][name]=summary
    for j in range(len(x)):
        report['rows'].append({'image_id':int(data['image_ids'][j]),'regime':int(data['regimes'][j]),'iou_per_mask':q[j].tolist(),'selections':{k:int(v[j]) for k,v in predictions.items()}})
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))
    native=a.features.with_name(a.features.stem+'_teacher.json')
    if native.exists():
        native_report=json.loads(native.read_text())
        if native_report.get('rows') and 'quality_all4' in native_report['rows'][0]:
            from takeover_compare_native import main as compare_native
            native_output=a.output.with_name(a.output.stem+'_native.json')
            compare_native(argparse.Namespace(students=a.output,native=native,output=native_output))
        elif native_report.get('status')=='COMPLETED_RESPONSE_FEATURE_EXPORT':
            # A teacher already in flight before baseline instrumentation can
            # reuse the exact same images/prompts' native baseline export.
            reference=a.features.parent.parent/'interior_v1'/(a.features.stem+'_teacher.json')
            if not reference.exists():raise FileNotFoundError(reference)
            from takeover_compare_native import main as compare_native
            compare_native(argparse.Namespace(students=a.output,native=reference,response=native,output=a.output.with_name(a.output.stem+'_native.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('features','models','output'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
