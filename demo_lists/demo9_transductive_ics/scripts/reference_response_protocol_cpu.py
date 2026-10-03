#!/usr/bin/env python3
"""CPU-only metadata leakage counterexamples; no annotations are opened."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tics.reference_response_protocol import coco_photo_id, validate_episode_manifest


def image(split, number):
    return split+'/COCO_'+split+'_'+str(number).zfill(12)+'.jpg'


def fixture():
    return dict(schema='reference_response_episodes_v1', annotation_protocol='official_COCO20i',
                held_fold=0, coco_year=2014, test_scope='frozen_confirmation',
                train=[dict(id='train0', c=1, support=image('train2014', 1), query=image('train2014', 2))],
                dev=[dict(id='dev0', c=2, support=image('val2014', 3), query=image('val2014', 4))],
                test=[dict(id='test0', c=0, support=image('val2014', 5), query=image('val2014', 6)),
                      dict(id='test1', c=4, support=image('val2014', 6), query=image('val2014', 7))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve existing receipt')
    result = validate_episode_manifest(fixture(), prior_development_photo_ids=set())
    assert result['test_classes'] == [0, 4]
    assert result['test_photo_connected_groups'] == [['test0', 'test1']]
    assert not result['independent_episode_bootstrap_allowed']
    assert coco_photo_id('val2017/000000000006.jpg') == coco_photo_id(image('val2014', 6))
    assert not result['image_files_verified'] and not result['official_mask_files_verified']
    checks = [dict(name='official_interleaved_fold_and_all_role_test_dependencies', passed=True),
              dict(name='photo_ID_stable_across_years_directory_names_and_roles', passed=True)]
    rejected = []
    def reject(name, change, registry=set()):
        data = fixture()
        change(data)
        try:
            validate_episode_manifest(data, prior_development_photo_ids=registry)
        except ValueError as exc:
            rejected.append(dict(name=name, reason=str(exc)))
        else:
            raise AssertionError('Invalid manifest accepted: '+name)
    reject('held_class_in_training', lambda d: d['train'][0].update(c=0))
    reject('nonheld_class_in_test', lambda d: d['test'][0].update(c=1))
    reject('query_is_reference_same_photo', lambda d: d['test'][0].update(query=d['test'][0]['support']))
    reject('train_photo_reappears_as_test_query_under_different_split_path',
           lambda d: d['test'][0].update(query=image('val2014', 1)))
    reject('dev_reference_reappears_as_test_query', lambda d: d['test'][0].update(query=image('val2014', 3)))
    reject('old_dev_reference_called_fresh_test', lambda d: None, registry={5})
    reject('old_dev_query_called_fresh_test', lambda d: None, registry={7})
    reject('missing_exposure_registry', lambda d: None, registry=None)
    reject('path_traversal', lambda d: d['test'][0].update(query='../val2014/000000000006.jpg'))
    reject('filename_split_disagrees', lambda d: d['test'][0].update(query='val2014/COCO_train2014_000000000006.jpg'))
    reject('empty_training', lambda d: d.update(train=[]))
    reject('extra_label_or_area_input_metadata', lambda d: d['train'][0].update(query_area=.2))
    reject('nonofficial_mask_protocol', lambda d: d.update(annotation_protocol='rebuilt_instance_masks'))
    reject('duplicate_episode_ID', lambda d: d['dev'][0].update(id='train0'))
    assert len(rejected) == 14
    reused = fixture()
    reused['test_scope'] = 'reused_development'
    audit = validate_episode_manifest(reused, prior_development_photo_ids={5, 7})
    assert audit['test_prior_development_overlap'] == 2
    assert audit['test_exposure_scope'] == 'reused_development'
    checks.append(dict(name='reused_dev_is_allowed_only_with_explicit_nonfresh_scope', passed=True))
    assert fixture() == copy.deepcopy(fixture())
    paths = [Path(__file__), Path(__file__).resolve().parents[1]/'tics/reference_response_protocol.py']
    receipt = dict(state='CPU_RESPONSE_PROTOCOL_METADATA_PASSED', checks=checks,
                   rejected=rejected, synthetic_episode_contract=result,
                   source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                   model_loaded=False, image_or_annotation_opened=False, real_task_gain_measured=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(state=receipt['state'], passed=len(checks), rejected=len(rejected),
                          image_or_annotation_opened=False, real_task_gain_measured=False)))


if __name__ == '__main__':
    main()
