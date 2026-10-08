#!/usr/bin/env python3
"""Check independent pixel and class/fold/paired statistics contracts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from ics.metrics import counts, gross_edits, summarize


def main():
    p = np.array([1, 1, 0, 0, 1], bool)
    b = np.array([0, 1, 1, 0, 0], bool)
    t = np.array([1, 0, 1, 0, 0], bool)
    ignore = np.array([0, 0, 0, 0, 1], bool)
    assert counts(p, t, ignore) == [1, 3]
    assert gross_edits(p, b, t, ignore) == [1, 0, 1, 0]
    rows = [dict(fold=0, class_id=0, iu={'a': [1, 1], 'b': [1, 1]}),
            dict(fold=0, class_id=0, iu={'a': [0, 9], 'b': [0, 9]}),
            dict(fold=0, class_id=1, iu={'a': [1, 1], 'b': [1, 1]}),
            dict(fold=1, class_id=0, iu={'a': [0, 1], 'b': [0, 1]})]
    report = summarize(rows, baselines=('a',), repetitions=1000)
    # Fold 0: (10 + 100)/2=55. Fold 1: 0. Official fold mean=27.5;
    # episode-IoU averaging and all-class averaging give different answers.
    assert report['miou']['a'] == 27.5
    assert np.isclose(report['all_class_miou']['a'], 110/3)
    assert report['paired']['a']['b']['ci95_pp'] == [0, 0]
    assert report == summarize(rows, baselines=('a',), repetitions=1000)
    duplicate = summarize(rows + [rows[0]], baselines=('a',), repetitions=1000)
    assert duplicate['n'] == 5
    assert np.isclose(duplicate['miou']['a'], (100*2/11 + 100)/4)
    missing = summarize(rows, baselines=('a',), repetitions=10,
                        expected_classes={0: [0, 1, 2], 1: [0]})
    assert missing['missing_classes']['0'] == ['2']
    assert np.isclose(missing['miou']['a'], 110/6)
    print('PASS: pixels/ignore, gross edits, pooled classes, fold means, paired draws, repeats, missing classes')


if __name__ == '__main__':
    main()
