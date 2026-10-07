"""Legal RGB toy contract; frozen deterministic encoder, not DINO performance."""
import json
import os
from pathlib import Path
import sys
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from ics.methods import object_crop_cls as method


class ArrangementEncoder:
    """CLS reads red-above-blue RGB relation; deliberately aliased final patches."""
    def __call__(self, view, coverage):
        red = view[..., 0].astype(int)-view[..., 2].astype(int) > 40
        blue = view[..., 2].astype(int)-view[..., 0].astype(int) > 40
        proper = red.sum() > 4 and blue.sum() > 4 and np.nonzero(red)[0].mean() < np.nonzero(blue)[0].mean()
        return dict(cls=np.array([1. if proper else -1., 0.]), patch_all=np.array([0., 1.]),
                    patch_masked=np.array([0., 1.]))


def fixture(rotated=False):
    reference = np.empty((1024, 1024, 3), np.uint8)
    reference[:] = [128, 0, 128]
    mask = np.zeros((1024, 1024), bool)
    mask[320:448, 320:448] = True
    reference[320:384, 320:448] = [255, 0, 0]
    reference[384:448, 320:448] = [0, 0, 255]
    query = np.empty_like(reference)
    query[:] = [0, 255, 0]
    target, distractor = np.zeros((64, 64), bool), np.zeros((64, 64), bool)
    target[12:20, 8:16] = True
    distractor[36:44, 40:48] = True
    query[192:256, 128:256] = [0, 0, 255] if rotated else [255, 0, 0]
    query[256:320, 128:256] = [255, 0, 0] if rotated else [0, 0, 255]
    query[576:640, 640:768] = [0, 0, 255]
    query[640:704, 640:768] = [255, 0, 0]
    # One deterministic native color feature ignores arrangement and aliases
    # reference foreground with magenta background. No query GT enters it.
    def native(rgb):
        green = rgb.reshape(64, 16, 64, 16, 3).mean(axis=(1, 3))[..., 1] > 128
        return np.stack((~green, green), axis=-1).reshape(4096, 2).astype(float)
    q, r = native(query), native(reference)
    base = np.full((64, 64), .1)
    base[target] = .52 if rotated else .48
    base[distractor] = .1 if rotated else .52
    return reference, mask, query, q, r, base, target, distractor


def main():
    rgb = np.zeros((48, 120, 3), np.uint8)
    rgb[:] = [255, 0, 0]
    selected = np.zeros((48, 120), bool)
    selected[5:37, 9:105] = True
    _, _, geometry = method.isolated_view(rgb, selected)
    assert geometry['original_bbox_hw'] == [32, 96] and geometry['resized_hw'] == [43, 128]
    reference, mask, query, q, r, base, target, distractor = fixture()
    result = method.predict(reference, mask, query, q, r, base, ArrangementEncoder(),
                            producer_binding=dict(producer='deterministic_arrangement_contract_not_DINO'))
    assert np.array_equal(result['fields']['cls'] > .5, target)
    assert np.array_equal(result['fields']['patch_masked'] > .5, base > .5)
    assert np.array_equal(result['fields']['patch_all'] > .5, base > .5)
    assert np.array_equal(result['fields']['native_roi_mean'] > .5, base > .5)
    assert result['info']['new_encoder_forwards'] <= 69
    assert result['info']['bound']['maximum_absolute_edit'] <= .1+1e-12
    assert result['masks']['cls']['work'].shape == (1024, 1024)
    for value in result['masks'].values():
        assert value['work'].dtype == bool and value['original'].shape == query.shape[:2]
    reference, mask, query, q, r, base, target, distractor = fixture(rotated=True)
    negative = method.predict(reference, mask, query, q, r, base, ArrangementEncoder(),
                              producer_binding=dict(producer='same_deterministic_arrangement_contract_not_DINO'))
    assert (base[target] > .5).all() and not (negative['fields']['cls'][target] > .5).any()
    truth = method.region_pixels(np.flatnonzero(target), query.shape[:2])
    report = dict(kind='deterministic_legal_full_RGB_chain_not_DINO', independent_new_method_count=0, revision=0,
                  fixed_four_legal_partitions_complete=True, physical_aspect_and_padding=True,
                  positive=dict(recovered_target_tokens=64, deleted_distractor_tokens=64,
                                strongest_masked_patch_control_has_no_class_signal=True,
                                complete_work_true_pixels=int((result['masks']['cls']['work'] & truth).sum())),
                  negative=dict(legal_same_class180degree_pose_change=True, prior_true_tokens=64,
                                candidate_true_tokens=0, complete_work_true_pixels=int((negative['masks']['cls']['work'] & truth).sum())),
                  maximum_field_edit=result['info']['bound']['maximum_absolute_edit'],
                  logical_new_views=result['info']['descriptor_image_requests'],
                  actual_toy_encoder_calls=result['info']['new_encoder_forwards'],
                  query_GT_used_in_predict=False, no_real_DINO_quality_claim=True)
    Path(__file__).with_name('check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
