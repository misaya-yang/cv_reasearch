"""Verify phase fallback quality, screen decoder cost, then queue formal timing."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/takeover_20261001_v1/local_student_head_v1'
OUT = BASE/'phase_followup_v1'


def main():
    OUT.mkdir(exist_ok=False)
    state = {'status': 'STARTING', 'pid': os.getpid(), 'stages': [],
             'protocol': 'Fixed trained student/margins/budgets; change only execution of original local head to phase matrices. Verify task quality on the same old disjoint128 (implementation recheck, not new test). Shared GPU six-arm screening includes strongest implicit phase; no exclusive latency claim. Formal six-arm stage waits CPU-only until existing formal four-arm stage terminates, avoiding mutual CUDA exclusivity waits. Minimum free6GB follows observed full-decoder own peak1.95GB/reserved3.17GB; preserve errors, old jobs and all assets.'}

    def save():
        tmp = OUT/'queue_status.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(OUT/'queue_status.json')

    def run(name, command):
        state.update(status='RUNNING', current_stage=name)
        entry = {'name': name, 'command': command, 'started_unix': time.time()}
        state['stages'].append(entry)
        with (OUT/(name+'.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            entry['pid'] = child.pid
            save()
            code = child.wait()
        entry.update(returncode=code, finished_unix=time.time())
        save()
        if code:
            raise RuntimeError(name+' failed; preserve report and traceback')

    save()
    try:
        run('phase_quality128', [sys.executable, 'tools/takeover_eval_local_head.py', '--phase-fallback', '--source-run', str(BASE.parent/'quality_followup_v1/cached_128'), '--subset', str(ROOT/'assets/coco_flip_validation_seed2028_v1'), '--output', str(BASE/'phase_disjoint128.json')])
        old = json.loads((BASE/'disjoint128.json').read_text())
        new = json.loads((BASE/'phase_disjoint128.json').read_text())
        def keyed(report):
            rows = report['rows']
            result = {(r['image_id'], r['annotation_id'], r['regime'], r['budget']): r for r in rows}
            assert len(result) == len(rows)
            return result
        a, b = keyed(old), keyed(new)
        assert a.keys() == b.keys()
        original_error = max(abs(a[k]['original_iou']-b[k]['original_iou']) for k in a)
        candidate_error = max(abs(a[k]['predicted_iou']-b[k]['predicted_iou']) for k in a)
        compatibility = {'matched_rows': len(a), 'max_reference_iou_difference': original_error,
                         'max_candidate_iou_difference': candidate_error,
                         'rows_with_different_candidate_iou': sum(a[k]['predicted_iou'] != b[k]['predicted_iou'] for k in a),
                         'phase_summary': new['summary'],
                         'scope': 'Same128 images, implementation compatibility only; old test is not a new independent cohort, no refitting.'}
        (BASE/'phase_quality_compatibility.json').write_text(json.dumps(compatibility, indent=2)+'\n')
        if original_error > 1e-10:
            raise RuntimeError('Reference task outputs changed; inspect matched evidence before attributing phase effects')
        run('phase_shared_decoder128', [sys.executable, 'tools/takeover_compact_perf.py', '--phase-fallback', '--concurrent-diagnostic', '--minimum-free-mib', '6000', '--output-dir', str(BASE/'phase_perf_p128_shared')])
        state.update(status='WAITING_FORMAL_PREDECESSOR_CPU_ONLY', current_stage='phase_formal_decoder128')
        save()
        while True:
            path = BASE/'perf_p128_fourarm/report.json'
            prior = json.loads(path.read_text()).get('status') if path.exists() else 'NOT_CREATED'
            if prior in ('COMPLETED_EXCLUSIVE_COMPILED_DECODER_TIMING', 'ERROR'):
                break
            if state.get('formal_predecessor_status') != prior:
                state['formal_predecessor_status'] = prior
                save()
            time.sleep(10)
        run('phase_formal_decoder128', [sys.executable, 'tools/takeover_compact_perf.py', '--phase-fallback', '--minimum-free-mib', '6000', '--output-dir', str(BASE/'phase_perf_p128_formal')])
        state.update(status='COMPLETED', current_stage=None)
    except BaseException as exc:
        state.update(status='ERROR', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
