"""Deterministic legal RGB/full-mask check; not actual DINO quality evidence."""
import importlib.util
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from ics.methods import object_removal_response as method


class ToyEncoder:
    """Global color arrangement response; patch tokens deliberately have no signal."""
    def __call__(self, view):
        red = view[..., 0].astype(int)-view[..., 2].astype(int) > 40
        blue = view[..., 2].astype(int)-view[..., 0].astype(int) > 40
        weight = (.5-np.arange(128)/128)[:, None]
        arrangement = float(((red.astype(float)-blue)*weight).sum()/20)
        natural = np.any(view != np.array((124,116,104)), axis=-1).mean()
        tokens = np.zeros((69, 3), float)
        tokens[:, 0] = 1.
        tokens[0] = [1., arrangement, natural]
        return tokens


def main():
    path = ROOT/'evidence/local/research_20261006/object_cls_revision0_01a1100b/check_object_cls.py'
    spec = importlib.util.spec_from_file_location('legal_RGB_fixture_only', path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    rgb = np.zeros((48, 120, 3), np.uint8)
    selected = np.zeros((48, 120), bool);selected[5:37, 9:105] = True
    a,b,coverage,g = method.contextual_views(rgb, selected)
    assert g['resized_hw']==[51,128] and g['outside_region_original_pixels_identical']
    assert np.all(a[51+(128-51)//2:] == (124,116,104))
    _,_,zero,_ = method.contextual_views(rgb, np.zeros_like(selected))
    assert zero.sum()==0
    assert method.response(np.array([1.,0.]),np.array([1.,0.]))==(None,0.)
    assert method.response(np.array([1.,0.]),np.array([1.,1e-8]))[0] is None
    r,mask,q,qf,rf,base,target,distractor=fixtures.fixture()
    result=method.predict(r,mask,q,qf,rf,base,ToyEncoder(), producer_binding={'producer':'toy_rgb_not_DINO'})
    recovered=int(((result['fields']['delta_cls']>.5)&target).sum())
    deleted=int((~(result['fields']['delta_cls']>.5)&distractor).sum())
    assert recovered > 0 and deleted > 0
    assert np.array_equal(result['fields']['delta_patch_all'],base)
    assert np.array_equal(result['fields']['delta_patch_masked'],base)
    r,mask,q,qf,rf,base,target,distractor=fixtures.fixture(rotated=True)
    negative=method.predict(r,mask,q,qf,rf,base,ToyEncoder(),producer_binding={'producer':'same_toy_RGB_not_DINO'})
    kept=int(((negative['fields']['delta_cls']>.5)&target).sum())
    assert kept < 64
    assert result['info']['new_encoder_forwards'] <= 71
    assert all(p['labeled_query_pixels']==0 and p['outside_region_original_pixels_identical'] for p in result['info']['pairs'])
    assert result['info']['bound']['maximum_absolute_edit'] <= .1+1e-12
    truth=method.region_pixels(np.flatnonzero(target),(1024,1024))
    report=dict(kind='deterministic_legal_full_RGB_removal_not_DINO',revision=0,independent_new_method_count=0,
        query_GT_used=False,zero_response_abstention_checked=True,geometric_zeroROI_checked=True,
        physical_whole_image_aspect=True,four_complete_Ward_partitions=True,
        positive=dict(recovered_target_tokens=recovered,deleted_distractor_tokens=deleted,
                      complete_work_TP=int((result['masks']['delta_cls']['work']&truth).sum())),
        negative=dict(same_class180degree_pose_change=True,prior_true_tokens=64,remaining_true_tokens=kept,
                      complete_work_TP=int((negative['masks']['delta_cls']['work']&truth).sum())),
        samecrop_patch_control_deliberately_aliased=True,no_DINO_quality_claim=True,
        max_edit=result['info']['bound']['maximum_absolute_edit'],toy_actual_calls=result['info']['new_encoder_forwards'])
    Path(__file__).with_name('check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
