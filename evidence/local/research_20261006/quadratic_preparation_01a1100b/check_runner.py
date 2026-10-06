"""Active quadratic classification through two CPU workers, query GT unread."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from check_quadratic import fixture

ROOT = Path(__file__).resolve().parents[4]


def main():
    with tempfile.TemporaryDirectory(prefix='quadratic_runner_check_') as directory:
        root = Path(directory)
        q, r, cov, _, _ = fixture()
        score = np.linspace(0, 1, 4096, dtype=np.float32).reshape(64, 64)
        poison = np.array([{'query_GT_must_remain_unread': True}], dtype=object)
        np.savez_compressed(root / 'features.npz', q=q, r=r, query_gt=poison)
        np.savez_compressed(root / 'packet.npz', cov=cov, score=score, truth=poison, native=poison)
        row = dict(c=1, fold=0, support='support', query='query',
                   feature_export='features.npz', packet_export='packet.npz')
        manifest = root / 'rows.json'
        manifest.write_text(json.dumps([row, row]))
        output = root / 'run'
        command = [sys.executable, str(ROOT / 'scripts/run_cpu_feature_candidates.py'), 'infer',
                   '--backend', 'prepared', '--prepared-methods', 'reference_quadratic',
                   '--primary-method', 'reference_quadratic', '--manifest', str(manifest),
                   '--root', str(root), '--out', str(output), '--workers', '2', '--threads', '1',
                   '--cpu-budget', '2', '--memory-gb', '2']
        result = subprocess.run(command, capture_output=True, text=True, timeout=40)
        if result.returncode:
            raise RuntimeError(result.stderr)
        seal = json.loads((output / 'sealed.json').read_text())
        assert seal['n'] == 2 and not seal['query_gt_opened']
        pids = set()
        for path in (output / 'receipts').glob('*.json'):
            receipt = json.loads(path.read_text())
            info = receipt['methods']['reference_quadratic']
            pids.add(receipt['pid'])
            assert not info['abstention'] and not receipt['query_gt_opened']
            assert info['quadratic_solver']['relative_linear_residual'] < 1e-10
            assert info['foreground_modes'] == 4 and info['background_modes'] == 2
        with np.load(output / 'predictions/000000.npz', allow_pickle=False) as packet:
            for key in ('reference_quadratic', 'quadratic_linear.control', 'quadratic_homogeneous.control',
                        'quadratic_kernel_mean.control', 'quadratic_nearest.control',
                        'quadratic_subspace.control', 'mean.control'):
                assert packet[key].shape == (131072,)
        report = dict(synthetic_occurrences=2, workers=2, observed_worker_count=len(pids),
                      active_quadratic_fit=True, query_GT_sentinels_unread=True,
                      masks_1024=True, predictions_sealed=True, no_extra_geometry_or_RGB=True,
                      real_episodes=0, no_gpu=True, no_server=True)
        Path(__file__).with_name('runner_check.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
