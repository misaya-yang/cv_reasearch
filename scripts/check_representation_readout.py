#!/usr/bin/env python3
"""Check source memberships, margins, canonical photo grouping and AUC macro."""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ics.representations import branch_margins, source_reference_roles, unit_tokens
from score_representation_premise import photo_key, photo_half, macro


def main():
    torch.set_num_threads(2)
    r = F.normalize(torch.tensor([[1., 0, 0, 0], [1., .1, 0, 0],
                                 [.9, .2, 0, 0], [0., 1, 0, 0],
                                 [0., 0, 1, 0], [0., 0, 0, 1]]), dim=-1)
    fg = torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)
    fi, bi = source_reference_roles(r.T.reshape(4, 2, 3), fg.reshape(2, 3))
    assert fi.tolist() == [0, 1] and bi.tolist() == [2]
    bank = {k: torch.stack([r, r.clone()]) for k in ['Q/16', 'K/16', 'Q/24', 'K/24']}
    bases = {k: torch.empty(4, 0) for k in bank}
    fields, valid = branch_margins(bank, bases, fi, bi, device='cpu')
    assert not bank and len(fields) == 10 and all(valid.values())
    x = unit_tokens(r)
    expected = (x@(F.normalize(x[fi].mean(0), dim=0)-F.normalize(x[bi].mean(0), dim=0))).numpy()
    for name in fields:
        assert np.max(np.abs(fields[name]-expected)) < 2e-7
    row = dict(dataset='coco', query_photo_id='COCO_val2014_000000001234.jpg')
    crop = dict(dataset='paco_part', query_photo_id='val2017/000000001234.jpg', query_crop=[1, 2, 3, 4])
    assert photo_key(row) == photo_key(crop) and photo_half(row) == photo_half(crop)
    items = [dict(fold=0, class_id=0, auc={'x': .1}),
             dict(fold=0, class_id=0, auc={'x': .9}),
             dict(fold=0, class_id=1, auc={'x': .2}),
             dict(fold=1, class_id=0, auc={'x': .8}),
             dict(fold=1, class_id=0, auc={})]
    value = macro(items, 'x')
    assert np.isclose(value['auc'], .575) and value['valid_episodes'] == 4
    assert value['valid_classes'] == 3 and value['valid_folds'] == 2
    print('PASS: exact FoRIS reference roles, unit margin and concat, no-op zero bases, photo/crop grouping, N/A-aware class/fold AUC')


if __name__ == '__main__':
    main()
