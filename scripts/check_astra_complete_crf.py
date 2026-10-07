#!/usr/bin/env python3
"""Check original CPU CRF pipelines with independently verified raw replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--row', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--prior-parity', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    sys.path.insert(0, str(a.code_root / 'src'))
    import numpy as np
    import torch
    from ics.astra300 import complete_baselines as c
    from ics.astra300.common import load_episode
    from ics.astra300.providers import bind_episode
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    row = json.loads(a.row.read_text())
    ep = c.bind_assets(bind_episode(load_episode(dict(row, artifacts={})), row), json.loads(a.assets.read_text()))
    bundle = c._episode_bundle(ep)
    prior = json.loads(a.prior_parity.read_text())
    assert prior['source_id'] == ep.source_id
    assert prior['actual_paired_forward_batches'] == 1 and prior['raw_paired_vs_single_cache_max_abs_error'] == 0
    assert prior['source_archive_sha256'] == c.SOURCE_ARCHIVE_SHA256
    earlier = prior['outputs']['insid3.release_bilinear.control']['cache_info']['actual_native_cache']
    assert earlier == bundle['raw_binding']
    report = dict(source_id=ep.source_id, query_GT_read=False, actual_new_encoder_forwards=0,
                  verified_raw_replay_prior_sha256=sha(a.prior_parity), assets_sha256=sha(a.assets),
                  before_input='raw pair independently proven bitwise equal to actual paired model in prior receipt',
                  complete_baseline_code_sha256=sha(c.__file__), outputs={})
    arrays = {}
    for family, key, after_function in [('foris','foris.complete_crf.control',c.complete_foris),
                                        ('insid3','insid3.release_crf640.control',c.complete_insid3_crf640)]:
        start = time.monotonic()
        try:
            host, encoder, _, _ = c._host(ep, family, 'crf', bundle['raw'], bundle['basis'])
            with torch.inference_mode(), torch.autocast('cpu', enabled=False):
                before = c._run_host(ep, host, encoder).cpu().numpy().copy()
                after_result = after_function(ep)
            after = after_result.mask_original
            different = int(np.count_nonzero(before != after))
            assert different == 0, 'Full original/cache-adapter CRF mask differs'
            report['outputs'][key] = dict(state='complete', mask_difference_pixels=different,
                                         original_shape=list(after.shape), wall_seconds=time.monotonic()-start,
                                         cache_info=after_result.info)
            arrays[key] = after
        except Exception as error:
            report['outputs'][key] = dict(state='failed', error=repr(error), wall_seconds=time.monotonic()-start)
    if 'foris.complete_crf.control' in arrays:
        arrays['foris_p0'] = c.foris_p0(ep)
        report['foris_producer'] = c.foris_producer(ep)
        report['foris_renderer'] = c.foris_renderer(ep)
    report['CRF_dependency_hashes'] = {}
    for name, module in list(sys.modules.items()):
        if name.startswith(('CRF', 'Permutohedral')) and getattr(module, '__file__', None):
            f = Path(module.__file__)
            report['CRF_dependency_hashes'][name] = dict(path=str(f), sha256=sha(f))
    report['all_complete'] = all(x['state'] == 'complete' for x in report['outputs'].values())
    np.savez_compressed(a.out.with_suffix('.npz'), **arrays)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(all_complete=report['all_complete'], outputs=report['outputs']), default=str)[:1000])


if __name__ == '__main__':
    main()
