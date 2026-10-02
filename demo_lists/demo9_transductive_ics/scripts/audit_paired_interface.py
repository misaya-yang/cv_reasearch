#!/usr/bin/env python3
"""Audit the supplied paired-cache script without editing its source or accepting a mean-only match."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--prepared-root', default='/root/autodl-tmp/demo9')
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=10)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
os.environ['DEMO9_CACHE'] = str(out / 'cache')
sys.path[:0] = [str(Path(a.prepared_root) / 'scripts'), os.environ.get('DEMO4_ROOT', '/root/autodl-tmp/demo4')]
import icx.common as common
old_iu = common.iu
calls = 0
checks = []
def checked_iu(p, g):
    global calls
    calls += 1
    # The supplied script calls iu(released, truth), iu(cached, truth), iu(released, cached).
    if calls % 3 == 0:
        checks.append(dict(e=len(checks), exact=bool(common.torch.equal(p, g)),
                           changed_pixels=int((p != g).sum()),
                           mask_iou=float((p & g).sum()) / max(float((p | g).sum()), 1.0)))
    return old_iu(p, g)
common.iu = checked_iu
src = Path(a.prepared_root) / 'scripts/cache_episodes.py'
report = dict(state='RUNNING', source=str(src), source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
              n=a.n, fold=0, contract='Unmodified paired encoding/cache script versus released INSID3; exact masks required. No method score.')
def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2))
save()
t0 = time.time()
try:
    sys.argv = [str(src), '--fold', '0', '--n', str(a.n), '--check', str(a.n)]
    runpy.run_path(str(src), run_name='__main__')
    if calls != 3 * a.n or len(checks) != a.n:
        raise RuntimeError(f'Unexpected audit call count: {calls}; checks={len(checks)}')
    report.update(state='PASSED_EXACT' if all(x['exact'] for x in checks) else 'FAILED_EXACT_INTERFACE',
                  exact_count=sum(x['exact'] for x in checks), checks=checks)
except BaseException as ex:
    report.update(state='ERROR', error=repr(ex), checks=checks)
    raise
finally:
    # This smoke cache has no further scientific use. Delete only files created under this own run's cache.
    removed = []
    cache = out / 'cache'
    if cache.exists():
        for path in cache.iterdir():
            if path.is_file():
                removed.append(dict(path=str(path), bytes=path.stat().st_size))
                path.unlink()
        cache.rmdir()
    report.update(elapsed_s=time.time()-t0, removed_smoke_cache=removed)
    save()
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}), flush=True)
