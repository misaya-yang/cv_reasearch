#!/usr/bin/env python3
"""Synthetic geometry, paired-input and pre-candidate metric choice checks."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ics.official_data import array_hash, file_hash
from ics.metrics import counts
from score_sealed_baselines import choose_metric, score_runs


def main():
    torch.set_num_threads(2)
    assert choose_metric(60.9, 61.0, 60.9)['primary'] == 'original'
    assert choose_metric(62., 60.9, 60.9)['primary'] == 'cli'
    assert choose_metric(60.9, 62., 60.9)['primary'] == 'original'
    assert choose_metric(62., 62., 60.9)['final_protocol_claim_on_hold']
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = (np.indices((7, 11)).sum(axis=0)%3 == 0).astype(np.uint8)
        label = root/'truth.png';Image.fromarray(raw*255).save(label)
        x = torch.from_numpy(raw)[None, None].float()
        original = F.interpolate(x, (23, 31), mode='nearest')[0, 0].numpy() > .5
        cli = F.interpolate(x, (1024, 1024), mode='nearest')[0, 0].numpy() > .5
        wrong = F.interpolate(torch.from_numpy(original)[None, None].float(),
                              (1024, 1024), mode='nearest')[0, 0].numpy() > .5
        assert not np.array_equal(cli, wrong)
        row = dict(episode_id='fixture', dataset='coco', fold=0, loader_class_id=0,
                   query_photo_id='photo', query_size_hw=[23, 31], query_mask_path=str(label),
                   query_mask_hash=array_hash(raw), query_rgb_hash='same-frozen-rgb')
        runs = []
        for arm, mask in [('foris.crf', cli), ('insid3', wrong)]:
            run = root/arm;(run/'predictions').mkdir(parents=True)
            (run/'manifest.json').write_text(json.dumps([row]))
            (run/'config.json').write_text(json.dumps(dict(split_role='smoke', arms=[arm], prepared_protocol=None)))
            np.savez_compressed(run/'predictions/episode.npz', original_hw=np.array([23, 31]),
                                **{'original/'+arm:np.packbits(original), 'cli/'+arm:np.packbits(mask)})
            record = dict(episode_id='fixture', prediction_file='episode.npz',
                          prediction_sha256=file_hash(run/'predictions/episode.npz'))
            (run/'inference.jsonl').write_text(json.dumps(record)+'\n')
            seal = dict(state='ALL_PREDICTIONS_SEALED', n=1,
                        manifest_sha256=file_hash(run/'manifest.json'), config_sha256=file_hash(run/'config.json'),
                        inference_index_sha256=file_hash(run/'inference.jsonl'))
            (run/'sealed.json').write_text(json.dumps(seal));runs.append(run)
        report = score_runs(runs, root/'score')['datasets']['coco']
        assert report['frames']['original']['miou'] == {'foris.crf':100., 'insid3':100.}
        assert report['frames']['cli']['miou']['foris.crf'] == 100.
        assert report['frames']['cli']['miou']['insid3'] < 100.
        assert report['primary_policy']['primary'] == 'cli'
        assert report['frames']['cli']['gross_edits']['insid3']['foris.crf']['percent_of_gt_area'][1] > 0
        balance=report['frames']['cli']['edit_balance']['insid3']['foris.crf']
        for mode,mask in [('add_only',cli|wrong),('delete_only',cli&wrong),('full',wrong)]:
            intersection,union=counts(mask,cli)
            assert np.isclose(balance['miou'][mode],100*intersection/union)
        unchanged=report['frames']['original']['edit_balance']['insid3']['foris.crf']['per_class']['0:0']
        assert unchanged['addition']['profit'] is None and unchanged['deletion']['profit'] is None
        custom=score_runs(runs,root/'custom_parent',baselines=['insid3'])['datasets']['coco']['primary']
        assert set(custom['paired'])=={'insid3'}
        assert custom['paired']['insid3']['foris.crf']['delta_pp']>0
        other = dict(row, query_rgb_hash='changed')
        (runs[1]/'manifest.json').write_text(json.dumps([other]))
        seal = json.loads((runs[1]/'sealed.json').read_text());seal['manifest_sha256']=file_hash(runs[1]/'manifest.json')
        (runs[1]/'sealed.json').write_text(json.dumps(seal))
        label.unlink()
        try:
            score_runs(runs, root/'mismatched')
            raise AssertionError('Different paired inputs accepted')
        except ValueError as error:
            assert 'Paired frozen inputs differ' in str(error)
    print('PASS: raw-label frames, paired identity, metric choice, gross normalization and classwise only-add/only-delete/full accounting')


if __name__ == '__main__':
    main()
