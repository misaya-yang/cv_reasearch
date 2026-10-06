"""Small synthetic batch/fallback/scoring checks; no encoder or real cohort."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def run(args):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_pro_role_prediction.py'), *args],
                            cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + '\n' + result.stderr)
    return result.stdout


def main():
    with tempfile.TemporaryDirectory(prefix='pro_m3_contract_') as directory:
        tmp = Path(directory)
        feature = np.zeros((4096, 1024), dtype=np.float32)
        feature[:, 0] = 1
        np.savez_compressed(tmp / 'feature.npz', q=feature, r=feature)
        cov = np.zeros((64, 64), dtype=np.float32)
        cov[0, :4] = 1
        native = np.zeros((1024, 1024), dtype=bool)
        native[200:400, 300:500] = True
        packed = np.packbits(native)
        score = np.linspace(0, 1, 4096, dtype=np.float32).reshape(64, 64)
        np.savez_compressed(tmp / 'packet.npz', cov=cov, score=score, native=packed, truth=packed)
        np.savez_compressed(tmp / 'sentinel.npz', cov=cov, score=score, native=packed,
                            truth=np.array({'must_not_be_opened_during_infer': True}, dtype=object))
        rows = [dict(c=0, fold=0, support='ref', query='query', feature_export=str(tmp / 'feature.npz'),
                     packet_export=str(tmp / 'packet.npz')) for _ in range(4)]
        (tmp / 'rows.json').write_text(json.dumps(rows))
        run(['infer', '--manifest', str(tmp / 'rows.json'), '--out', str(tmp / 'run'),
             '--workers', '2', '--threads', '1'])
        run(['score', '--out', str(tmp / 'run')])
        report = json.loads((tmp / 'run/score/report.json').read_text())
        assert report['n'] == 4 and all(v == 100 for v in report['scores'].values())
        assert report['predictions_sealed_before_scoring']
        for i in range(4):
            receipt = json.loads((tmp / 'run/receipts' / f'{i:06d}.json').read_text())
            assert receipt['input_keys_read'] == ['q', 'r', 'cov', 'score', 'native']
            assert not receipt['query_gt_opened']
        rows = [dict(rows[0], packet_export=str(tmp / 'sentinel.npz'))]
        (tmp / 'sentinel_rows.json').write_text(json.dumps(rows))
        run(['infer', '--manifest', str(tmp / 'sentinel_rows.json'), '--out', str(tmp / 'sentinel_run'),
             '--workers', '1', '--threads', '1'])
        assert (tmp / 'sentinel_run/sealed.json').is_file()
        checks = dict(status='passed', synthetic_occurrences=4, workers=2,
                      all_four_complete_native_fallback_masks_scored=100,
                      inference_did_not_open_unreadable_query_GT=True,
                      repeated_source_occurrences_preserved=True, scoring_after_seal=True,
                      active_full4096_tree_exercised=False, actual_cohort_cost='unmeasured', real_episodes=0)
        (Path(__file__).parent / 'runner_check.json').write_text(json.dumps(checks, indent=2) + '\n')
        print(json.dumps(checks))


if __name__ == '__main__':
    main()
