"""Freeze complete predeclared case1/2 outputs, then stop only our original tail."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(sys.argv[1])
    ownership = json.loads((root/'remaining_ownership.json').read_text())
    pid = ownership['pid']
    started = time.perf_counter()
    while True:
        if not Path(f'/proc/{pid}/cmdline').exists():
            raise RuntimeError('Owned original process exited before case2 marker')
        for line in (root/'remaining.log').read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('occurrence') == '2_0_74' and event.get('sealed') == 2:
                break
        else:
            if time.perf_counter()-started > 720:
                raise RuntimeError('No completed case2 marker within guard budget')
            time.sleep(.05)
            continue
        break
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if str(root/'code_ce/scripts/run_pro_paired_environment.py') not in command or str(root/'manifest_remaining.json') not in command:
        raise RuntimeError('Owned PID command identity changed; refuse signals')
    os.kill(pid, signal.SIGSTOP)
    try:
        run = root/'sealed_remaining'
        manifest = json.loads((run/'inference_manifest.json').read_text())
        config = json.loads((run/'config.json').read_text())
        expected = {'1_0_73', '2_0_74'}
        seal = dict(state='SELECTED_COMPLETE_PREDICTIONS_SEALED', n=2,
                    retained_ids=['1_0_73', '2_0_74'], selection='predeclared original rows; no GT/quality selection',
                    manifest_sha256=sha(run/'inference_manifest.json'), config_sha256=sha(run/'config.json'),
                    prediction_sha256={}, receipt_sha256={}, statistics_sha256={}, query_gt_used=False,
                    original_owned_pid=pid, original_plan_had_unexecuted_case3=True)
        for path, digest in config['source_sha256'].items():
            if sha(path) != digest:
                raise RuntimeError('Immutable inference snapshot changed')
        for row in manifest:
            key = row['id']
            if key not in expected:
                continue
            prediction, receipt = run/'predictions'/(key+'.npz'), run/'receipts'/(key+'.json')
            statistic = run/'statistics'/(key+'.npz')
            record = json.loads(receipt.read_text())
            if record['info']['query_gt_used'] or any(sha(row[field]) != digest for field, digest in record['input_sha256'].items()):
                raise RuntimeError('GT flag/input binding mismatch')
            seal['prediction_sha256'][key], seal['receipt_sha256'][key] = sha(prediction), sha(receipt)
            seal['statistics_sha256'][key] = sha(statistic)
        if set(seal['prediction_sha256']) != expected:
            raise RuntimeError('Complete source rows missing')
        (run/'partial_seal.json').write_text(json.dumps(seal, indent=2)+'\n')
    except Exception:
        os.kill(pid, signal.SIGCONT)
        raise
    os.killpg(pid, signal.SIGTERM)
    os.kill(pid, signal.SIGCONT)  # Deliver TERM to a stopped owned process.
    record = dict(state='CONTROLLED_STOP_AFTER_COMPLETE_CASE2', retained_ids=seal['retained_ids'],
                  partial_seal_sha256=sha(run/'partial_seal.json'), owned_pid=pid,
                  last_progress_event=event, reason='avoid duplicate case3; preserve active case2 and closed case1/2 outputs')
    (root/'tail_stop_receipt.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
