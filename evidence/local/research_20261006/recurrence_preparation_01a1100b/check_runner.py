"""Two-worker recurrence inference on synthetic caches; poisoned GT stays unread."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from check_recurrence import fixture

ROOT = Path(__file__).resolve().parents[4]


def main():
    with tempfile.TemporaryDirectory(prefix='recurrence_runner_check_') as directory:
        root = Path(directory)
        q, r, cov, score, target, distractor = fixture()
        # This score is input to the actual shared MEAN producer, not a claim
        # that the constructed core-test base is an existing cached MEAN field.
        score[target | distractor] = .95
        poison = np.array([{'must_not_read_query_GT': True}], dtype=object)
        np.savez_compressed(root / 'features.npz', q=q, r=r, query_gt=poison)
        np.savez_compressed(root / 'packet.npz', cov=cov, score=score, truth=poison, native=poison)
        row = dict(c=1, fold=0, support='support', query='query',
                   feature_export='features.npz', packet_export='packet.npz')
        manifest = root / 'rows.json'
        manifest.write_text(json.dumps([row, row]))
        output = root / 'run'
        command = [sys.executable, str(ROOT / 'scripts/run_cpu_feature_candidates.py'), 'infer',
                   '--backend', 'prepared', '--prepared-methods', 'query_recurrence',
                   '--primary-method', 'query_recurrence', '--manifest', str(manifest),
                   '--root', str(root), '--out', str(output), '--workers', '2', '--threads', '1',
                   '--cpu-budget', '2', '--memory-gb', '2']
        result = subprocess.run(command, capture_output=True, text=True, timeout=40)
        if result.returncode:
            raise RuntimeError(result.stderr)
        seal = json.loads((output / 'sealed.json').read_text())
        assert seal['n'] == 2 and not seal['query_gt_opened']
        worker_pids = set()
        for path in (output / 'receipts').glob('*.json'):
            receipt = json.loads(path.read_text())
            worker_pids.add(receipt['pid'])
            info = receipt['methods']['query_recurrence']
            assert not info['abstention'], info
            assert len(info['selected_seed_labels']) >= 2
            assert not receipt['query_gt_opened']
        with np.load(output / 'predictions/000000.npz', allow_pickle=False) as packet:
            for key in ('query_recurrence', 'recurrence_all_seed.control', 'recurrence_single_seed.control', 'mean.control'):
                assert packet[key].shape == (131072,)
        report = dict(synthetic_occurrences=2, workers=2, observed_worker_count=len(worker_pids), active_recurrence=True,
                      query_GT_sentinels_unread=True, masks_1024=True, predictions_sealed=True,
                      base_recomputed_from_supplied_source_score=True, no_extra_geometry_or_RGB=True,
                      real_episodes=0, no_gpu=True, no_server=True)
        Path(__file__).with_name('runner_check.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
