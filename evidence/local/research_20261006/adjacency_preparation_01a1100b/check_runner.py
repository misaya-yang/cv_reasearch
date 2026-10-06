"""Four synthetic occurrences, two local CPU workers; includes a forbidden-GT sentinel."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def main():
    rng = np.random.RandomState(61)
    with tempfile.TemporaryDirectory(prefix='adjacency_runner_check_') as temp:
        temp = Path(temp)
        source = temp / 'input.npz'
        q, r = rng.normal(size=(16, 8)), rng.normal(size=(16, 8))
        cov = np.ones((4, 4))
        base = rng.uniform(.1, .9, size=(4, 4))
        # Loading this key under allow_pickle=False must fail. Inference must never open it.
        np.savez(source, q=q, r=r, cov=cov, base=base,
                 query_gt=np.array([{'forbidden': True}], dtype=object))
        output = temp / 'predictions'
        cmd = [sys.executable, str(ROOT / 'scripts/run_reference_adjacency.py'),
               '--inputs', *([str(source)] * 4), '--out', str(output), '--workers', '2']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        if result.returncode:
            raise RuntimeError(result.stderr)
        complete = json.loads((output / 'prediction_complete.json').read_text())
        assert complete['occurrences'] == 4 and not complete['query_gt_used']
        paths = sorted(output.glob('*.npz'))
        assert len(paths) == 4
        first = None
        for path in paths:
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert receipt['input_keys_read'] == ['q', 'r', 'cov', 'base']
            with np.load(path, allow_pickle=False) as packet:
                for name in ('adjacency_map', 'base_control', 'adjacency_bilinear_control', 'base_nearest_control'):
                    assert packet[name].shape == (131072,)
                if first is None:
                    first = packet['adjacency_map'].copy()
                else:
                    np.testing.assert_array_equal(first, packet['adjacency_map'])
        rerun = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert rerun.returncode != 0 and 'FileExistsError' in rerun.stderr
        report = dict(synthetic_occurrences=4, workers=2, duplicate_occurrences_preserved=True,
                      forbidden_query_gt_key_unread=True, output_masks_1024=True,
                      existing_output_overwrite_rejected=True, real_episodes=0,
                      no_server=True, no_gpu=True)
        Path(__file__).with_name('runner_check.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
