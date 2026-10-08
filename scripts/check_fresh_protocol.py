#!/usr/bin/env python3
"""Check fresh seed/length isolation and correlated query-photo bootstrap."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ics.metrics import summarize
from ics import official_data
import frozen_dino_plan as runner


def main():
    # Opposite outcomes from two references of the same query are perfectly
    # paired at photo level. Independent episode sampling invents variance.
    rows = [dict(fold=0, class_id=0, query_photo_id='photo',
                 iu={'a': [1, 1], 'b': [0, 1]}),
            dict(fold=0, class_id=0, query_photo_id='photo',
                 iu={'a': [0, 1], 'b': [1, 1]})]
    grouped = summarize(rows, baselines=('a',), repetitions=1000, unit='query_photo')
    episode = summarize(rows, baselines=('a',), repetitions=1000)
    assert grouped['paired']['a']['b']['ci95_pp'] == [0, 0]
    assert episode['paired']['a']['b']['ci95_pp'] == [-100, 100]
    assert grouped['n'] == 2 and grouped['query_photo_count'] == 1
    assert grouped['miou'] == episode['miou']
    # A repeated draw must retain its weight in the point estimate, not become
    # an unweighted average of photograph IoUs.
    weighted = rows+[dict(fold=0, class_id=0, query_photo_id='other',
                          iu={'a': [1, 10], 'b': [0, 10]})]
    got = summarize(weighted, baselines=('a',), repetitions=1000, unit='query_photo')
    assert np.isclose(got['miou']['a'], 100*2/12)
    assert np.allclose(got['paired']['a']['b']['ci95_pp'], [-10, 0])
    try:
        summarize([dict(fold=0, class_id=0, iu={'a': [1, 1]})],
                  baselines=('a',), repetitions=1, unit='query_photo')
        raise AssertionError('Missing photo identity silently accepted')
    except ValueError:
        pass

    class Dataset:
        class_ids = range(2)
        def __init__(self, name):
            self.benchmark = name
        def __len__(self):
            return 2500

    def build(name, fold, assets):
        if name == 'suim':
            raise FileNotFoundError('fixture missing SUIM pool')
        return Dataset(name)

    def record(ds, index, fold, *args):
        return dict(dataset=ds.benchmark, fold=fold, official_index=index,
                    episode_id=f'{ds.benchmark}/{fold}/{index}', draw=float(np.random.rand()))

    with tempfile.TemporaryDirectory() as tmp:
        assets = Path(tmp)/'assets'
        (assets/'setup').mkdir(parents=True)
        runner.write(assets/'setup/download_receipt.json', dict(assets=[]))
        loaders = assets/'third_party/foris_official/datasets'
        loaders.mkdir(parents=True)
        names = ['coco', 'suim', 'pascal_part', 'lvis']
        for name in names:
            (loaders/(name+'.py')).write_text('# fixture loader\n')
        for split, seed in [('dev', 1), ('val', 2)]:
            out = Path(tmp)/split
            args = SimpleNamespace(assets=assets, out=out, datasets=names,
                                   split=split, smoke=False, fold=None)
            with patch.object(official_data, 'build_dataset', build), \
                    patch.object(official_data, 'record_episode', record), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.prepare(args)
            prepared = json.loads((out/'prepared.json').read_text())
            frozen = runner.read_rows(out/'manifest.json')
            assert prepared['seed'] == seed and prepared['n'] == 6000
            assert set(prepared['pending_datasets']) == {'suim'}
            assert prepared['folds']['coco/0']['selected_length'] == 500
            assert prepared['folds']['lvis/9']['selected_length'] == 200
            assert {r['dataset'] for r in frozen} == {'coco', 'pascal_part', 'lvis'}
            for name, folds in official_data.FOLDS.items():
                if name not in prepared['pending_datasets'] and name in names:
                    for fold in folds:
                        first = next(r for r in frozen if r['dataset'] == name and r['fold'] == fold)
                        assert first['draw'] == np.random.RandomState(seed).rand()
                        assert first['episode_id'].startswith(f'{split}_s{seed}/')
    print('PASS: query groups retain correlated draws and point weights; fresh seeds/lengths; missing dataset isolation')


if __name__ == '__main__':
    main()
