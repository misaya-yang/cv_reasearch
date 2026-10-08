#!/usr/bin/env python3
"""Check class edit profits against independently edited masks."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ics.metrics import counts, gross_edits
from edit_balance import summarize_edits


def main():
    truth = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    masks = [(np.array([1, 0, 1, 0, 0, 0], bool), np.array([1, 1, 0, 1, 0, 0], bool)),
             (truth.copy(), np.zeros(6, bool))]
    rows, direct = [], []
    for cls,(base, changed) in enumerate(masks):
        rows.append(dict(fold=0,class_id=cls,iu={'parent':counts(base,truth),'candidate':counts(changed,truth)},
                         edits={'candidate':{'parent':gross_edits(changed,base,truth)}}))
        added = base | changed
        deleted = base & changed
        direct.append({name:100*counts(mask,truth)[0]/counts(mask,truth)[1]
                       for name,mask in [('parent',base),('add_only',added),('delete_only',deleted),('full',changed)]})
    report = summarize_edits(rows,'parent','candidate')
    for mode in direct[0]:
        assert np.isclose(report['miou'][mode], np.mean([v[mode] for v in direct]))
    assert report['per_class']['0:0']['addition']['profit'] > 0
    assert report['per_class']['0:0']['deletion']['profit'] > 0
    assert report['per_class']['0:1']['deletion']['profit'] < 0
    assert report['delta_pp']['delete_only'] < 0
    absent = summarize_edits(rows,'parent','candidate',expected_classes={0:[0,1,2]})
    assert absent['per_class']['0:2']['addition']['profit'] is None
    print('PASS: actual only-add/only-delete/full masks match class accounting; profitable and harmful classes kept separate; zero edits N/A')


if __name__ == '__main__':
    main()
