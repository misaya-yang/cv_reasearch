"""Full cached dimensions, 32 anchors, two workers and forbidden-GT sentinel."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def main():
    with tempfile.TemporaryDirectory(prefix='absorption_check_') as folder:
        folder = Path(folder)
        source, output = folder/'input.npz', folder/'predictions'
        rng = np.random.RandomState(20261006)
        q = rng.normal(size=(4096, 1024)).astype(np.float32)
        r = np.zeros_like(q)
        r[np.arange(4096), np.repeat(np.arange(32), 128)] = 1
        cov = np.zeros((64, 64))
        cov[:32] = 1
        base = rng.uniform(.1, .9, size=(64, 64))
        np.savez_compressed(source, q=q, r=r, cov=cov, base=base,
                            query_gt=np.array([{'forbidden': True}], dtype=object))
        cmd = [sys.executable, str(ROOT/'scripts/run_reference_absorption.py'),
               '--inputs', str(source), str(source), '--out', str(output), '--workers', '2']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        if result.returncode:
            raise RuntimeError(result.stderr)
        seal = json.loads((output/'prediction_complete.json').read_text())
        assert seal['occurrences'] == 2 and not seal['query_gt_used']
        summaries = []
        for path in sorted(output.glob('*.npz')):
            receipt = json.loads(path.with_suffix('.json').read_text())
            assert receipt['input_keys_read'] == ['q', 'r', 'cov', 'base'] and not receipt['query_gt_used']
            assert receipt['method']['reference_anchors'] == 32
            assert receipt['method']['graph_nodes'] == 96
            with np.load(path, allow_pickle=False) as data:
                packed = [key for key in data.files if not key.startswith('field_')]
                assert len(packed) == 8
                for key in packed:
                    assert data[key].shape == (131072,) and data[key].dtype == np.uint8
            summaries.append(dict(anchors=receipt['method']['reference_anchors'],
                                  graph_nodes=receipt['method']['graph_nodes'],
                                  active_tokens=receipt['method']['active_tokens'],
                                  candidate_seconds=receipt['method']['wall_seconds'],
                                  complete_seconds=receipt['wall_seconds']))
        again = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        assert again.returncode != 0 and 'FileExistsError' in again.stderr
        report = dict(kind='synthetic_runner_only', real_episodes=0, workers=2, occurrences=2,
                      shape=[4096, 1024], dense_32_anchor_path_checked=True,
                      query_gt_sentinel_unread=True, eight_packed1024_outputs=True,
                      overwrite_rejected=True, summaries=summaries, no_gpu=True, no_server=True)
        Path(__file__).with_name('runner_check.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
