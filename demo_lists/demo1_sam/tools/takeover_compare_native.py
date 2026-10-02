"""Pair frozen student rows with the official native dynamic SAM2 selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from takeover_flip_score import bootstrap_image_groups


def main(a):
    if a.output.exists():raise FileExistsError(a.output)
    students=json.loads(a.students.read_text());native=json.loads(a.native.read_text())
    assert students['status']=='COMPLETED_FROZEN_STUDENT_TEST' and native['status'] in ('COMPLETED','COMPLETED_RESPONSE_FEATURE_EXPORT')
    rows=[]
    response_path=getattr(a,'response',None)
    if response_path is not None:
        response=json.loads(response_path.read_text())
        assert response['status']=='COMPLETED_RESPONSE_FEATURE_EXPORT'
        assert len(students['rows'])==len(response['rows'])
        lookup={(r['image_id'],r['annotation_id'],r['regime']):r for r in native['rows']}
        assert len(lookup)==len(native['rows'])
        paired=[]
        for s,r in zip(students['rows'],response['rows']):
            regime=('central','near_boundary','box')[s['regime']]
            assert s['image_id']==r['image_id'] and regime==r['regime']
            # The old in-flight teacher already stored all selected quality
            # values and IDs. Verify the candidate indexing without inventing
            # a token0 value or rerunning the original GPU experiment.
            for label in ('head','tangent','exact_views'):
                assert s['iou_per_mask'][r['choices'][label]]==r[label+'_iou']
            paired.append((s,lookup[(r['image_id'],r['annotation_id'],regime)]))
    else:
        assert len(students['rows'])==len(native['rows'])
        paired=zip(students['rows'],native['rows'])
    for s,n in paired:
        regime=('central','near_boundary','box')[s['regime']]
        assert s['image_id']==n['image_id'] and regime==n['regime']
        # This also verifies row order by every original candidate quality,
        # rather than relying solely on image IDs containing several objects.
        if response_path is None:assert np.array_equal(s['iou_per_mask'],n['quality_all4'][1:])
        row={'image_id':n['image_id'],'annotation_id':n['annotation_id'],'regime':regime}
        for name,c in s['selections'].items():
            row[name+'_iou']=s['iou_per_mask'][c]
            row[name+'_delta_native']=s['iou_per_mask'][c]-n['native_dynamic_iou']
        row['native_dynamic_iou']=n['native_dynamic_iou'];rows.append(row)
    summary={}
    for regime in ('central','near_boundary','box'):
        subset=[r for r in rows if r['regime']==regime];summary[regime]={}
        if not subset:continue
        for name in students['rows'][0]['selections']:
            summary[regime][name]={'mean_iou':float(np.mean([r[name+'_iou'] for r in subset])),'native_mean_iou':float(np.mean([r['native_dynamic_iou'] for r in subset])),'mean_delta_native':float(np.mean([r[name+'_delta_native'] for r in subset])),'image_cluster_delta_native95':bootstrap_image_groups(subset,name+'_delta_native')}
    result={'status':'COMPLETED','students':str(a.students),'native':str(a.native),'native_protocol':native.get('native_baseline_protocol',native.get('protocol')),'summary':summary,'rows':rows}
    if response_path is not None:result['candidate_protocol']='Student uses ordinary MATH candidates; native SDPA dynamic quality matched by image/annotation/regime from preceding independent export; no numeric-equivalence claim'
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('students','native','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--response',type=Path)
    main(p.parse_args())
