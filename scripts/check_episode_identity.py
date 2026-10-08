#!/usr/bin/env python3
"""Check identity matching, repeated draws, resize aliases and annotation conflicts."""
from pathlib import Path
import sys
import copy
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from ics.development_data import exclude_development


def row(key, label='target', rgb='query'):
    return dict(episode_id=key, dataset='suim', fold=-1, loader_class_id=0,
                global_class_id='FV', reference_rgb_hash='ref', reference_mask_hash='ref-mask',
                query_rgb_hash=rgb, query_mask_hash=label, query_photo_id=None)


def main():
    dev = [row('development')]
    official = [row('draw0'), row('draw1'), row('draw2', rgb='other')]
    expected = {'suim': {-1: [0]}}
    confirm, audit = exclude_development(official, dev, expected)
    assert [r['episode_id'] for r in confirm] == ['draw2']
    assert audit['excluded_official_draws'] == 2 and audit['eligible_for_confirmation']
    # Source repeats that are not development exposures retain full draw weight.
    confirm, _ = exclude_development(official+[row('draw3', rgb='other')], dev, expected)
    assert len(confirm) == 2
    ambiguous = [row('bad-label', label='different')]
    _, audit = exclude_development(ambiguous, dev, expected)
    assert audit['annotation_conflicts'] and not audit['eligible_for_confirmation']
    _, audit = exclude_development(official, dev, {'suim': {-1: [0, 1]}})
    assert audit['empty_confirm_classes'] == [dict(dataset='suim', fold='-1', class_id='1')]
    source = row('large-original', rgb='full-size')
    source['legacy_export_views'] = {'cap1600': {k:v for k,v in dev[0].items()
                                              if k.endswith('_hash')}}
    resized_dev = copy.deepcopy(dev);resized_dev[0]['legacy_pack_resizing_reported'] = True
    _, audit = exclude_development([source, row('kept', rgb='other')], resized_dev, expected)
    assert audit['excluded_official_draws'] == 1
    assert audit['matches'][0]['matched_views'] == ['cap1600']
    assert not audit['unresolved_resized_development_ids']
    _, audit = exclude_development([row('kept', rgb='other')], resized_dev, expected)
    assert audit['unresolved_resized_development_ids'] and not audit['eligible_for_confirmation']
    print('PASS: all duplicate exposures removed, legal repeats retained, labels/classes/resize identities audited')


if __name__ == '__main__':
    main()
