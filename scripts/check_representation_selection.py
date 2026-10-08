#!/usr/bin/env python3
"""Synthetic check that the winning representation cannot switch on half 1."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ics.official_data import array_hash, file_hash
import score_representation_premise as scorer


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ('predictions', 'representations', 'fields'):
            (root/name).mkdir()
        truth = np.zeros((64, 64), dtype=np.uint8); truth[:, :32] = 1
        mask = root/'truth.png'; Image.fromarray(truth*255).save(mask)
        labels, index = [], []
        fields = ['O/24/raw', 'Q/16/raw', 'K/24/raw', 'C2concat/raw', 'C2concat/deb']
        for name in ('coco', 'pascal_part', 'paco_part'):
            for half in (0, 1):
                row = dict(dataset=name, fold=0, loader_class_id=0,
                           episode_id=name+str(half), query_mask_path=str(mask),
                           query_mask_hash=array_hash(truth))
                photo = next('fixture'+str(i) for i in range(100)
                             if scorer.photo_half(dict(row, query_photo_id='fixture'+str(i))) == half)
                row['query_photo_id'] = photo
                labels.append(row)
                file = row['episode_id']+'.npz'
                np.savez(root/'predictions'/file, **{'cli/rcg.fine': np.packbits(np.ones((1024, 1024), bool))})
                response = {'O/24/raw': np.zeros(4096), 'Q/16/raw': truth.astype(float).ravel()*(1 if half == 0 else -1),
                            'K/24/raw': truth.ravel()*half, 'C2concat/raw': truth.ravel(),
                            'C2concat/deb': np.zeros(4096)}
                np.savez(root/'representations'/file, **response)
                np.savez(root/'fields'/file, score=np.zeros((64, 64)))
                index.append(dict(episode_id=row['episode_id'], prediction_file=file,
                                  representation_file=file, representation_validity=dict.fromkeys(fields, True),
                                  prediction_sha256=file_hash(root/'predictions'/file),
                                  representation_sha256=file_hash(root/'representations'/file),
                                  fields_sha256=file_hash(root/'fields'/file)))
        manifest = root/'manifest.json'; manifest.write_text(json.dumps(labels))
        (root/'config.json').write_text(json.dumps(dict(split_role='dev', prepared_protocol=dict(seed=1))))
        (root/'inference.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index))
        (root/'sealed.json').write_text(json.dumps(dict(state='ALL_PREDICTIONS_SEALED', manifest_sha256=file_hash(manifest))))
        out = root/'result'
        with patch.object(sys, 'argv', ['score', '--run', str(root), '--out', str(out)]), \
                contextlib.redirect_stdout(io.StringIO()):
            scorer.main()
        result = json.loads((out/'decision.json').read_text())
        assert result['candidates']['C1']['representation'] == 'Q/16/raw'
        assert not result['candidates']['C1']['passed']
        assert result['candidates']['C2']['passed']
        verify = [json.loads(s) for s in (out/'half1_episode_auc.jsonl').read_text().splitlines()]
        assert all('K/24/raw' not in r['auc'] for r in verify)
        print('PASS: selection winner fails verification; unselected verification winner is never measured; no reselection')


if __name__ == '__main__':
    main()
