"""Two CPU workers, full-size repeated counterexample and unopened GT sentinel."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from check_local import fixture

ROOT = Path(__file__).resolve().parents[4]


def main():
    with tempfile.TemporaryDirectory(prefix='constellation_local_check_') as folder:
        folder = Path(folder)
        source, output = folder / 'input.npz', folder / 'predictions'
        q, r, cov, base, matched, unmatched = fixture(1024)
        np.savez_compressed(source, q=q, r=r, cov=cov, base=base,
                            query_gt=np.array([{'forbidden': True}], dtype=object))
        cmd = [sys.executable, str(ROOT / 'scripts/run_reference_constellation_local.py'),
               '--inputs', str(source), str(source), '--out', str(output), '--workers', '2']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        if result.returncode:
            raise RuntimeError(result.stderr)
        seal = json.loads((output / 'prediction_complete.json').read_text())
        assert seal['occurrences'] == 2 and not seal['query_gt_used']
        paths = sorted(output.glob('*.npz'))
        assert len(paths) == 2
        first = None
        for path in paths:
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert receipt['input_keys_read'] == ['q', 'r', 'cov', 'base']
            assert not receipt['query_gt_used'] and receipt['method']['retained_poses'] == 1
            with np.load(path, allow_pickle=False) as packet:
                for key in ('constellation_local', 'constellation_global_v1.control', 'mean.control',
                            'clipped_mean.control', 'constellation_bag.control', 'constellation_prior.control'):
                    assert packet[key].shape == (131072,) and packet[key].dtype == np.uint8
                field, global_field = packet['output_field'], packet['global_v1_field']
                region = packet['local_region']
                np.testing.assert_array_equal(field[~region], base[~region])
                assert int((field[unmatched] > .5).sum()) == 16
                assert int((global_field[unmatched] > .5).sum()) == 0
                if first is None:
                    first = packet['constellation_local'].copy()
                else:
                    np.testing.assert_array_equal(first, packet['constellation_local'])
        again = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert again.returncode != 0 and 'FileExistsError' in again.stderr
        report = dict(method='reference_constellation_local_v2', synthetic_occurrences=2,
                      workers=2, feature_shape=[4096, 1024], repeated_occurrences_preserved=True,
                      query_gt_sentinel_unread=True, all_six_masks_packed1024=True,
                      local_outside_values_preserved=True, global_v1_and_base_bag_controls_saved=True,
                      unmatched_component_local_kept=16, unmatched_component_global_kept=0,
                      overwrite_rejected=True, real_episodes=0, no_gpu=True, no_server=True)
        Path(__file__).with_name('runner_check.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
