"""Compare original and tiled M06 rendering on one real cached native pair."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ics.pro30.common import load_episode
from ics.pro30 import common, differential_06, render_exact_optim
from ics.cpu100.common import Result as CPUResult, render


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    raw = a.manifest.read_bytes()
    row = json.loads(raw.decode().splitlines()[0])
    ep = load_episode(row)
    assert ep.q.shape == ep.r.shape == (4096, 1024)
    assert ep.producer['FoRIS_Part1_applied'] is False
    runs, outputs = {}, {}
    for name, fn in [('original', differential_06.differential),
                     ('optimized', render_exact_optim.m06_exact_blocked)]:
        differential_06._CACHE = None
        common._BR_CACHE.clear()
        common._PCA_CACHE.clear()
        render_exact_optim._AXES.clear()
        start = time.monotonic()
        output = fn(ep)
        runs[name] = dict(whole_call_seconds=time.monotonic()-start,
                         core_seconds=output.info['postprocess_seconds'],
                         renderer_seconds=output.info['renderer_seconds'])
        outputs[name] = output
    field = outputs['original'].field
    old = render(ep, CPUResult(field, {}))
    new = render_exact_optim.render_exact_blocked(ep, CPUResult(field, {}))
    differences = {name: int(np.count_nonzero(old[name] != new[name]))
                   for name in ('margin', 'work', 'original')}
    differences['complete_method_field'] = int(np.count_nonzero(
        outputs['original'].field != outputs['optimized'].field))
    differences['complete_method_mask'] = int(np.count_nonzero(
        outputs['original'].mask_original != outputs['optimized'].mask_original))
    report = dict(source_id=ep.source_id, query_GT_read=False,
                  encoder_forwards=0, methods_added=0, cold_method_caches=True,
                  timing_scope='whole method including renderer; same loaded input; original then optimized',
                  manifest_sha256=hashlib.sha256(raw).hexdigest(), timings=runs,
                  differing_elements=differences, bitwise_equal=not any(differences.values()),
                  same_solver_and_budget=True)
    a.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
