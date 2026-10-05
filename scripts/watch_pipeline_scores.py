#!/usr/bin/env python3
"""Score ready sealed outputs beside a live GPU queue, without retries or shutdown.

Only one CPU scorer is launched at a time. A missing/replaced upstream process is
terminal after already-ready outputs are handled; a state file alone is not live.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from experiment_pipeline import process_identity
from experiment_resource_guard import check_artifacts


def provisioned():
    cpu = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    memory = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    cores = os.cpu_count() if cpu[0] == 'max' else int(cpu[0]) / int(cpu[1])
    return cores >= 8 and (memory == 'max' or int(memory) >= 16 * 1024**3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--upstream-receipt', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--poll-seconds', type=float, default=10)
    a = p.parse_args()
    if a.poll_seconds <= 0:
        p.error('poll-seconds must be positive')
    stages = json.loads(a.plan.read_text())['stages']
    if any(s['kind'] != 'cpu' for s in stages):
        raise ValueError('Only CPU scoring stages are accepted')
    upstream = json.loads(a.upstream_receipt.read_text())
    a.state.parent.mkdir(parents=True, exist_ok=True)
    with a.state.open('x') as f:
        json.dump({}, f)
    state = dict(pid=os.getpid(), start_ticks=process_identity(os.getpid())['start_ticks'],
                 upstream=upstream, completed=[], failed=[], active=None)

    def save(status):
        state.update(status=status, observed_at=time.time())
        tmp = a.state.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(a.state)

    child, stream, selected = None, None, None
    save('WAITING_SEALED_OUTPUTS')
    while True:
        if child is not None:
            if child.poll() is None:
                time.sleep(a.poll_seconds)
                continue
            stream.close()
            good = child.returncode == 0 and check_artifacts(selected['produces'], Path(selected['cwd']))
            state['completed' if good else 'failed'].append(selected['name'])
            state['active'] = None
            save('SCORER_COMPLETED' if good else 'SCORER_FAILED')
            child = None
        remaining = [s for s in stages if s['name'] not in state['completed'] + state['failed']]
        if not remaining:
            save('COMPLETED' if not state['failed'] else 'COMPLETED_WITH_FAILURES')
            return int(bool(state['failed']))
        ready = [s for s in remaining if check_artifacts(s['requires'], Path(s['cwd']))]
        if ready and provisioned():
            selected = ready[0]
            if any(Path(x['path']).exists() for x in selected['produces']):
                state['failed'].append(selected['name'])
                state['existing_output_conflict'] = selected['name']
                save('EXISTING_OUTPUT_REFUSED')
                continue
            env = dict(os.environ, **selected.get('env', {}))
            env['CUDA_VISIBLE_DEVICES'] = ''
            log = a.state.parent / (selected['name'] + '.log')
            stream = log.open('xb')
            child = subprocess.Popen(selected['argv'], cwd=selected['cwd'], env=env,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            identity = process_identity(child.pid)
            state['active'] = dict(name=selected['name'], pid=child.pid,
                                   start_ticks=identity['start_ticks'] if identity else None, log=str(log))
            save('SCORING')
            continue
        identity = process_identity(upstream['supervisor_pid'])
        same_boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == upstream['boot_id']
        if not same_boot or identity is None or identity['start_ticks'] != upstream['supervisor_start_ticks']:
            state['unscored'] = [s['name'] for s in remaining]
            save('UPSTREAM_TERMINAL_UNSCORED_OUTPUTS')
            return 2
        save('WAITING_SEALED_OUTPUTS' if provisioned() else 'WAITING_RESOURCES')
        time.sleep(a.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
