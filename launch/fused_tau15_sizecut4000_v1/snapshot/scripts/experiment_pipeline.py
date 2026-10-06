#!/usr/bin/env python3
"""Sequential prepared stages without runtime/idle limits, retries, or shutdown.

Adoption verifies Linux proc identity and the declared Python script before waiting.
An adopted process has no recoverable exit status: only declared outputs are checked.
CPU stages run their declared commands; this supervisor never imports model runtimes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment_resource_guard import check_artifacts, declared_producers, gpu_inventory


def process_identity(pid, proc_root=Path('/proc')):
    """Return start ticks and argv; zombies count as finished, never signal PIDs."""
    try:
        root = proc_root / str(pid)
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] in ('Z', 'X'):
            return None
        argv = (root / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return dict(start_ticks=int(stat[19]), argv=argv,
                    cwd=str((root / 'cwd').resolve(strict=True)))
    except FileNotFoundError:
        return None


def matches_stage(identity, stage):
    cwd = Path(stage.get('cwd', '.')).resolve()
    scripts = [str((Path(arg) if Path(arg).is_absolute() else cwd / arg).resolve())
               for arg in stage['argv'][1:] if arg.endswith('.py')]
    actual_cwd = Path(identity['cwd'])
    actual = {str((Path(arg) if Path(arg).is_absolute() else actual_cwd / arg).resolve())
              for arg in identity['argv'][1:] if arg.endswith('.py')}
    return bool(scripts) and all(script in actual for script in scripts)


def validate(plan):
    stages = plan.get('stages', [])
    names = set()
    for stage in stages:
        name = stage.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Stage names must be nonempty and unique')
        names.add(name)
        if stage.get('kind') not in ('cpu', 'gpu'):
            raise ValueError('Stage kind must be cpu or gpu')
        argv = stage.get('argv')
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a for a in argv):
            raise ValueError('Stage argv must be a nonempty string list')
        if not Path(stage.get('cwd', '.')).is_dir():
            raise ValueError('Stage cwd missing')
        for key in ('requires', 'produces', 'success_checks'):
            for item in stage.get(key, []):
                if not isinstance(item, dict) or not isinstance(item.get('path'), str) or not item['path']:
                    raise ValueError('Artifact checks require nonempty paths')
    declared_producers(stages)
    return stages


class Pipeline:
    def __init__(self, plan, state_file, log_file, poll_seconds=2, inventory=gpu_inventory,
                 share_gpu_states=()):
        self.stages = validate(plan)
        self.path = Path(state_file).resolve()
        self.log_path = Path(log_file).resolve()
        if self.path == self.log_path:
            raise ValueError('State and event log must differ')
        self.poll = poll_seconds
        self.inventory = inventory
        self.share_gpu_states = tuple(map(Path, share_gpu_states))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Refuse reuse, including an active guard's files. Never append an old run.
        with self.path.open('x') as stream:
            stream.write('{}')
        self.log = self.log_path.open('x')
        self.status = dict(state='INITIAL', supervisor_pid=os.getpid(), owned_processes=[],
                           plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                           time_limit=None, idle_limit=None, automatic_shutdown=False)
        self.status['shared_gpu_state_files'] = [str(p) for p in self.share_gpu_states]

    def shared_gpu_pids(self):
        """Only explicitly coordinated workers with matching live process identities."""
        allowed = set()
        for path in self.share_gpu_states:
            try:
                state = json.loads(path.read_text())
                for entry in state.get('shared_peer_processes', state.get('owned_processes', [])):
                    identity = process_identity(entry['pid'])
                    if identity and identity['start_ticks'] == entry.get('start_ticks'):
                        allowed.add(entry['pid'])
            except (OSError, ValueError, KeyError):
                continue
        return allowed

    def emit(self, state, **fields):
        event = dict(state=state, timestamp=time.time(), **fields)
        self.status.update(state=state, last_event=event)
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        temporary.write_text(json.dumps(self.status, indent=2))
        temporary.replace(self.path)
        self.log.write(json.dumps(event) + '\n')
        self.log.flush()
        print(json.dumps(event), flush=True)

    def outputs_good(self, stage):
        declared_producers(self.stages)
        cwd = Path(stage.get('cwd', '.')).resolve()
        return (check_artifacts(stage.get('produces', []), cwd)
                and check_artifacts(stage.get('success_checks', []), cwd))

    def run(self, adopt_pid=None, adopt_stage=None):
        start = 0
        if adopt_pid is not None:
            start = next(i for i, s in enumerate(self.stages) if s['name'] == adopt_stage)
            stage = self.stages[start]
            identity = process_identity(adopt_pid)
            if identity is None or not matches_stage(identity, stage):
                self.emit('ADOPTION_REJECTED', stage=adopt_stage, pid=adopt_pid)
                return 2
            if not check_artifacts(stage.get('requires', []), Path(stage.get('cwd', '.')).resolve()):
                self.emit('ADOPTION_REJECTED_REQUIRES_FAILED', stage=adopt_stage)
                return 3
            if not stage.get('produces'):
                self.emit('ADOPTION_REJECTED_NO_DECLARED_OUTPUTS', stage=adopt_stage)
                return 2
            self.status['owned_processes'].append(dict(pid=adopt_pid, start_ticks=identity['start_ticks'],
                                                       stage=adopt_stage, ownership='adopted_monitor_only'))
            self.emit('ADOPTED_RUNNING', stage=adopt_stage, pid=adopt_pid, start_ticks=identity['start_ticks'])
            while True:
                current = process_identity(adopt_pid)
                if current is None or current['start_ticks'] != identity['start_ticks']:
                    break
                time.sleep(self.poll)
            good = self.outputs_good(stage)
            self.emit('ADOPTED_OUTPUTS_VERIFIED' if good else 'ADOPTED_OUTPUTS_FAILED',
                      stage=adopt_stage, exit_status=None, exit_status_known=False)
            if not good:
                return 3
            start += 1
        for index in range(start, len(self.stages)):
            stage = self.stages[index]
            cwd = Path(stage.get('cwd', '.')).resolve()
            if not check_artifacts(stage.get('requires', []), cwd):
                self.emit('REQUIRES_FAILED', stage=stage['name'])
                return 3
            if stage['kind'] == 'gpu':
                last = None
                while True:
                    current = self.inventory()
                    foreign = set(current['pids']) - self.shared_gpu_pids()
                    if current['known'] and current['visible'] and not foreign:
                        break
                    waiting = (current['known'], current['visible'], tuple(current['pids']))
                    if waiting != last:
                        self.emit('WAITING_GPU_AVAILABLE', stage=stage['name'], inventory=current)
                        last = waiting
                    time.sleep(self.poll)
            environment = dict(os.environ)
            environment.update(stage.get('env', {}))
            if stage['kind'] == 'cpu':
                environment['CUDA_VISIBLE_DEVICES'] = ''
            output = self.log_path.parent / (self.log_path.name + '.stage-' + str(index) + '.log')
            with output.open('xb') as stream:
                process = subprocess.Popen(stage['argv'], cwd=cwd, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                identity = process_identity(process.pid) if Path('/proc').is_dir() else None
                self.status['owned_processes'].append(dict(pid=process.pid, stage=stage['name'],
                    start_ticks=identity['start_ticks'] if identity else None, ownership='launched'))
                self.emit('STAGE_RUNNING', stage=stage['name'], pid=process.pid, output_log=str(output))
                # No deadline or utilization policy; interruption leaves the worker intact.
                while process.poll() is None:
                    time.sleep(self.poll)
                code = process.returncode
            good = code == 0 and self.outputs_good(stage)
            self.emit('STAGE_COMPLETED' if good else 'STAGE_FAILED', stage=stage['name'], returncode=code)
            if not good:
                return 4
        self.emit('QUEUE_COMPLETED')
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--state-file', type=Path, required=True)
    parser.add_argument('--log-file', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=2)
    parser.add_argument('--adopt-pid', type=int)
    parser.add_argument('--adopt-stage')
    parser.add_argument('--share-gpu-with-state', type=Path, action='append', default=[],
                        help='Explicitly coordinate with live workers recorded by another owned pipeline')
    parser.add_argument('--run', action='store_true', help='Execute; default only validates the plan')
    args = parser.parse_args()
    pipeline = None
    try:
        if args.poll_seconds <= 0:
            raise ValueError('poll-seconds must be positive')
        plan = json.loads(args.plan.read_text())
        stages = validate(plan)
        if (args.adopt_pid is None) != (args.adopt_stage is None):
            raise ValueError('Both adoption options are required together')
        if args.adopt_pid is not None and (args.adopt_pid <= 0 or args.adopt_stage not in {s['name'] for s in stages}):
            raise ValueError('Invalid adoption PID or stage')
        if not args.run:
            print(json.dumps(dict(state='PLAN_VALIDATED_NO_EXECUTION', stages=[s['name'] for s in stages])))
            return 0
        pipeline = Pipeline(plan, args.state_file, args.log_file, args.poll_seconds,
                            share_gpu_states=args.share_gpu_with_state)
        return pipeline.run(args.adopt_pid, args.adopt_stage)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        if pipeline:
            pipeline.emit('SUPERVISOR_ERROR', error_type=type(error).__name__, detail=str(error))
        else:
            print(str(error), file=sys.stderr)
        return 5
    except KeyboardInterrupt:
        if pipeline:
            pipeline.emit('SUPERVISOR_INTERRUPTED_WORKERS_PRESERVED')
        return 130
    finally:
        if pipeline:
            pipeline.log.close()


if __name__ == '__main__':
    sys.exit(main())
