#!/usr/bin/env python3
"""Synthetic regression for public-loader RGB/GT size mismatches and dual metrics.

All files and look records live in a disposable fixture, not research outputs.
"""
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
sys.path.insert(0, str(REPO/'scripts'))


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import frozen_dino_plan as runner
    from ics.official_data import build_dataset, file_hash, load_inputs, record_episode, reset_sampling_seed
    # Load the actual official COCO module once; its unedited build(args) then
    # reads the miniature file layout below through the same path relocation.
    build_dataset('coco', 0, REPO.parent/'cv_data')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary);assets = root/'assets';base = assets/'demo4_cache/data/COCO2014'
        (base/'splits/val').mkdir(parents=True);(base/'val2014').mkdir()
        (base/'annotations/val2014').mkdir(parents=True)
        names = ['val2014/a.jpg', 'val2014/b.jpg']
        with open(base/'splits/val/fold0.pkl', 'wb') as f:
            pickle.dump({c: names for c in range(80)}, f)
        pattern = ((np.indices((7, 11)).sum(axis=0) % 3) == 0).astype(np.uint8)
        for name in names:
            Image.new('RGB', (31, 23), (140, 60, 20)).save(base/name)
            Image.fromarray(pattern*73).save(base/'annotations'/Path(name).with_suffix('.png'))
        reset_sampling_seed();ds=build_dataset('coco',0,assets)
        row=record_episode(ds,0,0,assets,root/'masks')
        assert row['global_class_id']==72
        assert row['query_mask_size_hw']==[7,11] and row['query_size_hw']==[23,31]
        rgb,mask,query=load_inputs(row,assets)
        assert mask.shape==(7,11) and rgb.size==(31,23)
        truth=torch.from_numpy(pattern)[None,None].float()
        original=F.interpolate(truth,(23,31),mode='nearest')[0,0].numpy()>.5
        cli=F.interpolate(truth,(1024,1024),mode='nearest')[0,0].numpy()>.5
        twice=F.interpolate(torch.from_numpy(original)[None,None].float(),
                            (1024,1024),mode='nearest')[0,0].numpy()>.5
        assert not np.array_equal(cli,twice) # catches accidental double-resizing
        out=root/'run';(out/'predictions').mkdir(parents=True)
        filename=runner.prediction_file(row)
        arms=['foris.crf','rcg','rcg.fine','mean']
        payload={kind+'/'+arm:np.packbits(m) for kind,m in [('original',original),('cli',cli)] for arm in arms}
        np.savez_compressed(out/'predictions'/filename,original_hw=np.array([23,31]),**payload)
        runner.write(out/'manifest.json',[row])
        runner.write(out/'config.json',dict(candidate_id='synthetic-label-frame-fixture',
                     split_role='smoke',arms=arms,assets=str(assets)))
        index=dict(episode_id=row['episode_id'],prediction_file=filename,
                   prediction_sha256=file_hash(out/'predictions'/filename),
                   foris_seconds=0.,B0_inference_upper_bound_seconds=0.,combined_pipeline_seconds=0.)
        (out/'inference.jsonl').write_text(json.dumps(index)+'\n')
        runner.write(out/'sealed.json',dict(state='ALL_PREDICTIONS_SEALED',n=1,
                     manifest_sha256=file_hash(out/'manifest.json'),config_sha256=file_hash(out/'config.json'),
                     inference_index_sha256=file_hash(out/'inference.jsonl'),query_labels_opened=False))
        runner.REPO=root
        runner.score(SimpleNamespace(out=out))
        report=json.loads((out/'report.json').read_text())['datasets']['coco']
        assert all(x==100 for x in report['miou'].values())
        assert all(x==100 for x in report['cli_miou'].values())
        record=json.loads((out/'episode_metrics.jsonl').read_text())
        assert record['source_mask_size_hw']==[7,11] and record['evaluation_size_hw']==[23,31]
        assert record['source_truth_pixels']==int(pattern.sum())
        assert record['truth_pixels']==int(original.sum())
    print('PASS: official mismatched masks retained, raw reference passed through, raw GT mapped directly to both frames')


if __name__ == '__main__':
    main()
