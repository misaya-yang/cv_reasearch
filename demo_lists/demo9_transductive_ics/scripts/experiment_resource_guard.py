#!/usr/bin/env python3
"""Finite prepared-stage driver with shared-GPU-safe, provider-aware poweroff.

Default is DRY RUN: no GPU probe, CUDA tensor, stage or shutdown is executed.
Future --run verifies Linux/root/AutoDL layout, then checks a real visible GPU
and a tiny CUDA tensor in a separate process. CPU preparation must finish before GPU rental; only bounded CPU handoffs are allowed;
one zero-utilization sample never triggers poweroff. A running owned stage
with continuous known-zero GPU activity for 60s is interrupted; queue
exhaustion/unready next stage or failure triggers immediate terminal handling.
Any foreign/uncertain GPU PID blocks shutdown, even at zero utilization.
Only the driver-created subprocess group may be interrupted; no other jobs are killed.

AutoDL documents /usr/bin/shutdown, with NO '-h now', as its instance shutdown:
https://www.autodl.com/docs/save_money/#_3
Provider lifecycle documentation was verified by the primary agent; runtime
still requires explicit AutoDL configuration, root permission and an executable
helper. The guard records SHUTDOWN_REQUESTED, never claims billing has stopped
without platform confirmation. There is no generic guest-OS shutdown fallback.

Plan schema:
 {"platform":"autodl", "cuda_python":"/root/miniconda3/bin/python",
  "stages":[{"name":"probe","kind":"gpu","argv":["python","probe.py"],
             "cwd":"/own/project","timeout_seconds":1500,
             "requires":[{"path":"cache.pt"}],
             "produces":[{"path":"probe_result.json","json_equals":{"state":"COMPLETED"}}],
             "success_checks":[{"path":"result.json","json_equals":{"state":"COMPLETED"}}]}]}
A missing requires artifact may defer ONLY to an earlier named stage that declares
that exact path in produces. Static code/assets still pass CPU checks; generated
inputs are explicitly pending, checked after production and before consumption.
Before renting GPU, run --preflight-only once on the prepared CPU filesystem.
Then run only once explicitly armed on a future GPU instance:
 python experiment_resource_guard.py --plan plan.json --state-file guard.json \
   --run --allow-shutdown
Secrets/command arguments are not included in the state log. No loops manufacture
GPU work, no failed-stage retry, no model/download/setup orchestration.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time

AUTODL_DOC = 'https://www.autodl.com/docs/save_money/#_3'
SHUTDOWN_PATH = '/usr/bin/shutdown'


def decision(active_kind=None, next_ready=False, stage_failed=False, gpu_known=True,
             gpu_pids=(), owned_pids=(), idle_seconds=0, idle_grace=60):
    """Pure CPU-testable lifecycle rule; no utilization-based ownership inference."""
    foreign=set(gpu_pids)-set(owned_pids)
    if foreign:return 'DEFER_FOREIGN_GPU'
    if active_kind:
        return 'GPU_RUNNING' if active_kind == 'gpu' else 'CPU_HANDOFF_GAP'
    if not gpu_known:
        return 'BLOCKED_UNKNOWN_GPU_INVENTORY'
    if stage_failed:
        return 'SHUTDOWN_ELIGIBLE_STAGE_FAILURE'
    if next_ready:
        return 'READY_NEXT_STAGE'
    return 'IDLE_GRACE' if idle_seconds < idle_grace else 'SHUTDOWN_ELIGIBLE_NO_RUNNABLE_STAGE'


def gpu_inventory():
    """Ownership and activity are separate. Failed util queries are UNKNOWN."""
    try:
        visible=subprocess.run(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
        processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
        indices=[x.strip() for x in visible.stdout.splitlines() if x.strip()]
        pids=[x.strip() for x in processes.stdout.splitlines() if x.strip()]
        if visible.returncode or processes.returncode or any(not re.fullmatch(r'\d+',x) for x in indices+pids):
            return dict(known=False,visible=False,pids=[],utilization=None)
        util=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        raw=[x.strip() for x in util.stdout.splitlines() if x.strip()]
        utilization=max(map(float,raw)) if util.returncode==0 and raw and all(re.fullmatch(r'\d+(?:\.\d+)?',x) for x in raw) else None
        return dict(known=True,visible=bool(indices),pids=sorted(set(map(int,pids))),utilization=utilization)
    except (OSError,subprocess.SubprocessError):
        return dict(known=False,visible=False,pids=[],utilization=None)


def idle_activity(zero_since,utilization,now,limit):
    if utilization is None or utilization>0:return None,False
    if zero_since is None:zero_since=now
    return zero_since,now-zero_since>=limit


def owned_tree_pids(process):
    """Only the new process group created by this driver is owned."""
    pids=set()
    try:
        text=subprocess.check_output(['ps','-eo','pid=,pgid='],text=True,timeout=5)
        pids.update(int(pid) for pid,pgid in (line.split() for line in text.splitlines()) if int(pgid)==process.pid)
    except (OSError,subprocess.SubprocessError,ValueError):
        try:
            if os.getpgid(process.pid)==process.pid:pids.add(process.pid)
        except ProcessLookupError:pass
    return pids


def stop_owned_tree(process):
    owned=owned_tree_pids(process)
    def verified_group():
        for pid in owned:
            try:
                if os.getpgid(pid)==process.pid and process.pid!=os.getpgrp():return True
            except ProcessLookupError:pass
        return False
    try:
        if verified_group():os.killpg(process.pid,signal.SIGTERM)
        elif process.poll() is None:process.terminate()
    except ProcessLookupError:pass
    if process.poll() is None:
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if verified_group():os.killpg(process.pid,signal.SIGKILL)
            else:process.kill()
            process.wait(timeout=10)
    # A parent may already have failed while its owned workers remain alive.
    # Group identity is rechecked; no unrelated PID/group is signalled.
    if verified_group():
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    return owned


def sha256_file(path, chunk_bytes=1024*1024):
    """Bounded-memory checksum, including large frozen weights on no-card hosts."""
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(chunk_bytes),b''):
            digest.update(chunk)
    return digest.hexdigest()


def provider_shutdown(path=SHUTDOWN_PATH):
    """The actual AutoDL helper may be executable ASCII without a shebang.

    Shell execution is used only for ENOEXEC of the verified provider helper,
    not as a generic guest-OS poweroff fallback. Never print its contents.
    """
    options=dict(stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
    try:return subprocess.run([str(path)],**options).returncode
    except OSError as error:
        if error.errno!=errno.ENOEXEC:raise
        return subprocess.run(['/bin/bash',str(path)],**options).returncode


def check_artifacts(checks, cwd, verify_hash=True):
    for item in checks:
        path=Path(item['path']);path=path if path.is_absolute() else cwd/path
        if not path.exists():return False
        if verify_hash and 'sha256' in item and sha256_file(path)!=item['sha256']:
            return False
        if 'json_equals' in item:
            try:
                value=json.loads(path.read_text())
                if any(value.get(k)!=expected for k,expected in item['json_equals'].items()):return False
            except (OSError,ValueError,TypeError):return False
    return True


def artifact_path(item,cwd):
    path=Path(item['path'])
    return (path if path.is_absolute() else cwd/path).resolve()


def declared_producers(stages):
    """Only outputs within each producer's own cwd; no forward/self inputs."""
    producers={}
    for index,stage in enumerate(stages):
        cwd=Path(stage.get('cwd','.')).resolve()
        for item in stage.get('produces',[]):
            if not isinstance(item,dict) or not isinstance(item.get('path'),str) or not item['path']:
                raise ValueError('produces needs artifact dicts with nonempty paths')
            path=artifact_path(item,cwd)
            try:path.relative_to(cwd)
            except ValueError:raise ValueError('produced artifact escapes producer cwd: '+str(path))
            if path==cwd or (path.exists() and path.is_dir()):raise ValueError('produces must name files, not stage directories')
            if str(path) in producers:raise ValueError('Multiple producers for artifact: '+str(path))
            producers[str(path)]=dict(index=index,stage=stage['name'],checks=item)
    return producers


def validate_plan(plan):
    stages=plan.get('stages',[])
    names=[]
    for stage in stages:
        if stage.get('kind') not in ('gpu','cpu') or not isinstance(stage.get('argv'),list) or not stage['argv']:
            raise ValueError('Every stage needs cpu/gpu kind and nonempty argv list')
        if not all(isinstance(v,str) for v in stage['argv']):raise ValueError('argv entries must be strings')
        if not 0<float(stage.get('timeout_seconds',0))<=43200:
            raise ValueError('Every prepared stage needs a finite timeout_seconds <=43200')
        if stage['kind']=='cpu' and (stage.get('role') not in ('handoff','finalize') or float(stage['timeout_seconds'])>60):
            raise ValueError('CPU preparation must finish before renting GPU; run-time CPU handoff/finalize is capped at60s')
        names.append(stage['name'])
    if len(names)!=len(set(names)):raise ValueError('Stage names must be unique')
    declared_producers(stages)
    return stages


class ResourceGuard:
    def __init__(self,plan,state_file,allow_shutdown=False,idle_grace=60,poll_seconds=2,
                 inventory=gpu_inventory,clock=time.monotonic,sleeper=time.sleep,poweroff=None):
        self.plan=plan;self.stages=validate_plan(plan);self.path=state_file
        self.allow_shutdown=allow_shutdown;self.idle_grace=idle_grace;self.poll_seconds=poll_seconds
        self.inventory=inventory;self.clock=clock;self.sleep=sleeper;self.poweroff=poweroff
        self.started=clock();self.events=[]
        self.preflight_path=state_file.with_suffix('.preflight.json')
        self.status=dict(state='INITIAL',billing_stop_confirmed=False,provider_documentation=AUTODL_DOC,
            stage_names=[r['name'] for r in self.stages],events=self.events)
    def emit(self,state,**fields):
        event=dict(state=state,elapsed_seconds=self.clock()-self.started,**fields)
        self.events.append(event);self.status.update(state=state,last_event=event)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.path.with_suffix(self.path.suffix+'.tmp')
        tmp.write_text(json.dumps(self.status));tmp.replace(self.path)
        print(json.dumps(event),flush=True)
    def plan_digest(self):
        return hashlib.sha256(json.dumps(self.plan,sort_keys=True).encode()).hexdigest()
    def build_preflight(self):
        """CPU-only. Run before GPU rental; run() never compiles/hashes code."""
        if not self.stages:
            self.emit('PREFLIGHT_EMPTY_QUEUE_NOT_RUNNABLE');return False
        files={};pending=[]
        try:
            producers=declared_producers(self.stages)
            for index,stage in enumerate(self.stages):
                cwd=Path(stage.get('cwd','.')).resolve()
                if not cwd.is_dir():raise ValueError('stage cwd missing')
                first=stage['argv'][0];exe=Path(first) if '/' in first else Path(shutil.which(first) or '/missing')
                if not exe.is_absolute():exe=cwd/exe
                if not exe.is_file() or not os.access(exe,os.X_OK):raise ValueError('stage executable missing')
                if '-c' in stage['argv'] and 'python' in Path(first).name.lower():
                    compile(stage['argv'][stage['argv'].index('-c')+1],'<prepared_inline_python>','exec')
                paths=[Path(v) if Path(v).is_absolute() else cwd/v for v in stage['argv'][1:] if v.endswith('.py')]
                paths += [Path(v) if Path(v).is_absolute() else cwd/v for v in stage.get('code_files',[])]
                for path in paths:
                    text=path.read_text();compile(text,str(path),'exec')
                    files[str(path.resolve())]=dict(sha256=sha256_file(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
                for artifact in stage.get('requires',[]):
                    path=artifact_path(artifact,cwd);producer=producers.get(str(path))
                    if producer is not None and producer['index']>=index:
                        raise ValueError('forward/self dependency forbidden: '+str(path))
                    if not path.exists():
                        if producer is None:raise ValueError('undeclared missing input: '+str(path))
                        pending.append(dict(path=str(path),consumer=stage['name'],producer=producer['stage'],
                            checks=artifact,state='PENDING_DECLARED_PRODUCTION_NOT_EXISTS_VALIDATED'))
                        continue
                    if not check_artifacts([artifact],cwd):raise ValueError('existing input checksum/readiness mismatch')
                    # Generated paths will change legitimately. Do not include them
                    # in immutable initial-file metadata, but audit their checks now.
                    if producer is None:
                        files[str(path)]=dict(sha256=sha256_file(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
                for artifact in stage.get('cpu_artifacts',[]):
                    path=artifact_path(artifact,cwd)
                    if not check_artifacts([artifact],cwd):raise ValueError('CPU artifact checksum/readiness mismatch')
                    files[str(path)]=dict(sha256=sha256_file(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
            receipt=dict(state='PREFLIGHT_COMPLETED',plan_sha256=self.plan_digest(),files=files,pending_produced_inputs=pending)
            self.preflight_path.parent.mkdir(parents=True,exist_ok=True)
            self.preflight_path.write_text(json.dumps(receipt));self.emit('PREFLIGHT_COMPLETED',verified_files=len(files),pending_produced_inputs=pending);return True
        except (OSError,ValueError,SyntaxError):
            self.emit('PREFLIGHT_FAILED_NO_GPU_CALLS');return False
    def preflight_ready(self):
        try:
            receipt=json.loads(self.preflight_path.read_text())
            if receipt.get('state')!='PREFLIGHT_COMPLETED' or receipt.get('plan_sha256')!=self.plan_digest():return False
            return all(Path(path).stat().st_size==info['size'] and Path(path).stat().st_mtime_ns==info['mtime_ns'] for path,info in receipt['files'].items())
        except (OSError,ValueError,KeyError):return False
    def startup_failure(self):
        current=self.inventory()
        verified=self.poweroff is not None or (sys.platform.startswith('linux') and os.geteuid()==0 and Path('/root/autodl-tmp').is_dir() and self.plan.get('platform')=='autodl')
        if verified and current['known'] and current['visible'] and not current['pids']:
            self.terminal(True)
        elif current['pids']:self.emit('DEFER_FOREIGN_GPU',gpu_pids=current['pids'])
        else:self.emit('NO_CONFIRMED_GPU_NO_SHUTDOWN')
    def tensor_check(self):
        python=self.plan.get('cuda_python',sys.executable)
        program='import torch; assert torch.cuda.is_available(); x=torch.ones(1,device="cuda"); assert x.item()==1'
        try:return subprocess.run([python,'-c',program],capture_output=True,timeout=30).returncode==0
        except (OSError,subprocess.SubprocessError):return False
    def cuda_ready(self):
        current=self.inventory()
        if self.poweroff is None and (not sys.platform.startswith('linux') or os.geteuid()!=0 or not Path('/root/autodl-tmp').is_dir() or self.plan.get('platform')!='autodl'):
            self.emit('NOT_VERIFIED_LINUX_AUTODL_LAYOUT_NO_STAGE_EXECUTED');return False
        if not current['known'] or not current['visible']:
            self.emit('NO_CONFIRMED_GPU_NO_CUDA_OR_STAGE_EXECUTED');return False
        if current['pids']:
            self.emit('DEFER_FOREIGN_GPU',gpu_pids=current['pids']);return False
        good=self.tensor_check()
        self.emit('GPU_CUDA_READINESS_CONFIRMED' if good else 'CUDA_READINESS_FAILED_NO_STAGE_EXECUTED')
        return good
    def terminal(self,failed,stopped_owned=()):
        current=self.inventory()
        state=decision(stage_failed=failed,gpu_known=current['known'],gpu_pids=current['pids'],
            owned_pids=stopped_owned,idle_seconds=self.idle_grace,idle_grace=self.idle_grace)
        if not state.startswith('SHUTDOWN_ELIGIBLE'):
            self.emit(state,gpu_pids=current['pids']);return False
        if not self.allow_shutdown:
            self.emit('SHUTDOWN_NOT_ARMED',terminal_reason=state);return False
        if self.plan.get('platform')!='autodl':
            self.emit('SHUTDOWN_BLOCKED_UNVERIFIED_PROVIDER',terminal_reason=state);return False
        # Runtime permission/helper verification; this is NOT a provider billing acknowledgement.
        if self.poweroff is None and (not sys.platform.startswith('linux') or os.geteuid()!=0 or not Path('/root/autodl-tmp').is_dir() or not os.access(SHUTDOWN_PATH,os.X_OK)):
            self.emit('SHUTDOWN_BLOCKED_HELPER_OR_PERMISSION',terminal_reason=state);return False
        # Recheck after all permission/readiness work to minimize the startup race.
        second=self.inventory()
        if not second['known'] or set(second['pids'])-set(stopped_owned):
            self.emit('SHUTDOWN_BLOCKED_GPU_RECHECK',gpu_pids=second['pids']);return False
        self.emit('SHUTDOWN_REQUESTED',terminal_reason=state,helper=SHUTDOWN_PATH,
            billing_stop_confirmed=False,provider_documented_action=True)
        try:
            if self.poweroff is not None:code=self.poweroff()
            else:
                code=provider_shutdown()
            if code:self.emit('SHUTDOWN_REQUEST_FAILED',returncode=code)
            # Keep SHUTDOWN_REQUESTED on successful command exit: platform not confirmed.
            return code==0
        except (OSError,subprocess.SubprocessError):
            self.emit('SHUTDOWN_REQUEST_RESULT_UNKNOWN',billing_stop_confirmed=False);return False
    def run(self):
        if not self.stages or not self.preflight_ready():
            self.emit('PREPARED_PREFLIGHT_MISSING_OR_EMPTY_QUEUE');self.startup_failure();return 3
        if not self.cuda_ready():
            self.startup_failure();return 2
        try:producers=declared_producers(self.stages)
        except (OSError,ValueError):
            self.emit('PRODUCED_PATH_CHANGED_OR_INVALID');self.terminal(True);return 7
        for stage in self.stages:
            cwd=Path(stage.get('cwd','.')).resolve()
            try:
                inputs_good=all(check_artifacts([item],cwd,verify_hash=str(artifact_path(item,cwd)) in producers)
                    for item in stage.get('requires',[]))
            except (OSError,ValueError):inputs_good=False
            if not inputs_good:
                self.emit('NO_RUNNABLE_NEXT_STAGE',stage=stage['name']);self.terminal(False);return 3
            if stage['kind']=='gpu':
                check=self.inventory()
                if not check['known'] or check['pids']:
                    self.emit('DEFER_FOREIGN_GPU',gpu_pids=check['pids']);return 4
            environment=dict(os.environ,DEMO4_GPU_FRAC='.3',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            environment.update(stage.get('env',{}))
            logfile=self.path.parent/(stage['name']+'.log')
            process=None;stopped=()
            try:
                with logfile.open('ab') as log:
                    process=subprocess.Popen(stage['argv'],cwd=cwd,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    began=self.clock();zero_since=None;self.emit(decision(active_kind=stage['kind']),stage=stage['name'],own_child_pid=process.pid)
                    while process.poll() is None:
                        current=self.inventory();owned=owned_tree_pids(process)
                        foreign=set(current['pids'])-owned
                        if foreign and self.status['state']!='DEFER_FOREIGN_GPU':
                            self.emit('DEFER_FOREIGN_GPU',gpu_pids=sorted(foreign),own_child_pid=process.pid)
                        zero_since,idle_timeout=idle_activity(zero_since,current.get('utilization'),self.clock(),self.idle_grace)
                        timed_out=self.clock()-began>float(stage['timeout_seconds'])
                        if idle_timeout or timed_out:
                            stopped=stop_owned_tree(process)
                            self.emit('OWNED_GPU_SUSTAINED_IDLE_STOP' if idle_timeout else 'OWN_STAGE_TIMEOUT',stage=stage['name'],stopped_owned_pids=sorted(stopped))
                            self.terminal(True,stopped_owned=stopped);return 5
                        self.sleep(self.poll_seconds)
                    code=process.returncode
                declared_producers(self.stages) # reject runtime symlink/directory escapes
                healthy=code==0 and check_artifacts(stage.get('produces',[]),cwd,verify_hash=True) and check_artifacts(stage.get('success_checks',[]),cwd,verify_hash=False)
                self.emit('STAGE_COMPLETED' if healthy else 'STAGE_FAILED',stage=stage['name'],returncode=code)
                if not healthy:
                    stopped=stop_owned_tree(process)
                    self.terminal(True,stopped_owned=stopped);return 6
            except (OSError,ValueError,subprocess.SubprocessError):
                if process is not None and process.poll() is None:
                    stopped=stop_owned_tree(process)
                self.emit('STAGE_DRIVER_ERROR',stage=stage['name']);self.terminal(True,stopped_owned=stopped);return 7
        self.emit('QUEUE_COMPLETED_NO_RUNNABLE_NEXT_STAGE')
        self.terminal(False);return 0


def self_check():
    assert decision(active_kind='gpu',gpu_pids=[11],owned_pids=[11])=='GPU_RUNNING'
    assert decision(active_kind='cpu')=='CPU_HANDOFF_GAP'
    assert decision(next_ready=True)=='READY_NEXT_STAGE'
    assert decision(stage_failed=True)=='SHUTDOWN_ELIGIBLE_STAGE_FAILURE'
    assert decision(stage_failed=True,gpu_pids=[77])=='DEFER_FOREIGN_GPU'
    assert decision(gpu_known=False,stage_failed=True)=='BLOCKED_UNKNOWN_GPU_INVENTORY'
    zero,stop=idle_activity(None,0,0,60);assert not stop
    zero,stop=idle_activity(zero,0,59,60);assert not stop
    zero,stop=idle_activity(zero,5,59.5,60);assert zero is None and not stop
    zero,stop=idle_activity(zero,0,80,60);assert not stop
    zero,stop=idle_activity(zero,0,140,60);assert stop
    zero,stop=idle_activity(zero,None,141,60);assert zero is None and not stop
    with tempfile.TemporaryDirectory(prefix='resource_guard_cpu_') as folder:
        root=Path(folder);calls=[];idle=lambda:dict(known=True,visible=True,pids=[],utilization=0)
        guard=ResourceGuard(dict(platform='autodl',stages=[]),root/'terminal.json',allow_shutdown=True,
            inventory=idle,poweroff=lambda:calls.append('mock') or 0)
        assert guard.terminal(False) and len(calls)==1 and guard.status['state']=='SHUTDOWN_REQUESTED'
        calls.clear();guard.inventory=lambda:dict(known=True,visible=True,pids=[999],utilization=0)
        assert not guard.terminal(True) and not calls and guard.status['state']=='DEFER_FOREIGN_GPU'
        for kind in ('gpu','cpu'):
            calls=[]
            # This is CPU-only fixture code labelled with the future lifecycle kind.
            stages=[dict(name=kind+'_stall_CPU_stub',kind=kind,role='handoff',argv=[sys.executable,'-c','import time; time.sleep(5)'],timeout_seconds=120 if kind=='gpu' else 60)]
            clock=[0.]
            guard=ResourceGuard(dict(platform='autodl',stages=stages),root/(kind+'.json'),allow_shutdown=True,
                inventory=idle,clock=lambda:clock[0],sleeper=lambda delay:clock.__setitem__(0,clock[0]+delay),
                idle_grace=60,poll_seconds=5,poweroff=lambda:calls.append('mock') or 0)
            def owned_inventory():
                pid=guard.status.get('last_event',{}).get('own_child_pid')
                return dict(known=True,visible=True,pids=[pid] if pid else [],utilization=0)
            guard.inventory=owned_inventory
            assert guard.build_preflight()
            guard.cuda_ready=lambda:True
            assert guard.run()==5 and len(calls)==1
            assert any(r['state']=='OWNED_GPU_SUSTAINED_IDLE_STOP' for r in guard.events)
            assert clock[0]>=60
        stages=[dict(name='prepared_CPU_stub',kind='gpu',argv=[sys.executable,'-c','pass'],timeout_seconds=60)]
        for visible in (True,False):
            calls=[]
            readiness=ResourceGuard(dict(platform='autodl',stages=stages),root/('readiness_'+str(visible)+'.json'),allow_shutdown=True,
                inventory=lambda:dict(known=True,visible=visible,pids=[],utilization=0),poweroff=lambda:calls.append('mock') or 0)
            assert readiness.build_preflight()
            readiness.tensor_check=lambda:False
            assert readiness.run()==2
            assert len(calls)==(1 if visible else 0)
            assert readiness.status['state']==('SHUTDOWN_REQUESTED' if visible else 'NO_CONFIRMED_GPU_NO_SHUTDOWN')
        # Independent prepared cards: a measured negative is COMPLETED and may
        # hand off immediately; an ERROR report stops even when child exits0.
        for first_state in ('COMPLETED','ERROR'):
            marker=root/('next_'+first_state+'.json');first=root/('first_'+first_state+'.json')
            def report_stage(name,path,state):
                program='import json,pathlib;pathlib.Path('+repr(str(path))+').write_text(json.dumps('+repr(dict(state=state,scientific_outcome='NEGATIVE'))+'))'
                return dict(name=name,kind='gpu',argv=[sys.executable,'-c',program],timeout_seconds=60,
                    success_checks=[dict(path=str(path),json_equals=dict(state='COMPLETED'))])
            stages=[report_stage('first_'+first_state,first,first_state),report_stage('next_'+first_state,marker,'COMPLETED')]
            calls=[]
            sequence=ResourceGuard(dict(platform='autodl',stages=stages),root/('sequence_'+first_state+'.json'),
                allow_shutdown=True,inventory=lambda:dict(known=True,visible=True,pids=[],utilization=50),
                poll_seconds=.01,poweroff=lambda:calls.append('mock') or 0)
            assert sequence.build_preflight();sequence.cuda_ready=lambda:True
            assert sequence.run()==(0 if first_state=='COMPLETED' else 6)
            assert marker.exists()==(first_state=='COMPLETED') and len(calls)==1
            assert any(x['state']==('STAGE_COMPLETED' if first_state=='COMPLETED' else 'STAGE_FAILED') for x in sequence.events)
        # Runtime-only assets are declared explicitly; no scientific dependency
        # or permission inference. Both real subprocesses here are CPU fixtures.
        for valid_output in (True,False):
            data=root/('generated_'+str(valid_output)+'.json');done=root/('consumed_'+str(valid_output)+'.json')
            output=dict(path=str(data),json_equals=dict(state='READY'))
            producer=dict(name='producer_'+str(valid_output),kind='gpu',cwd=str(root),timeout_seconds=60,
                argv=[sys.executable,'-c','import pathlib,json;pathlib.Path('+repr(str(data))+').write_text(json.dumps('+repr(dict(state='READY' if valid_output else 'ERROR'))+'))'],
                produces=[output])
            consumer=dict(name='consumer_'+str(valid_output),kind='gpu',cwd=str(root),timeout_seconds=60,
                argv=[sys.executable,'-c','import pathlib;pathlib.Path('+repr(str(done))+').write_text("done")'],
                requires=[output],produces=[dict(path=str(done))])
            calls=[];generated=ResourceGuard(dict(platform='autodl',stages=[producer,consumer]),root/('generated_guard_'+str(valid_output)+'.json'),
                allow_shutdown=True,inventory=lambda:dict(known=True,visible=True,pids=[],utilization=50),
                poll_seconds=.01,poweroff=lambda:calls.append('mock') or 0)
            assert generated.build_preflight()
            receipt=json.loads(generated.preflight_path.read_text())
            assert receipt['pending_produced_inputs'][0]['producer']==producer['name'] and not data.exists()
            generated.cuda_ready=lambda:True
            assert generated.run()==(0 if valid_output else 6)
            assert done.exists()==valid_output and len(calls)==1
        # Undeclared/forward missing inputs, path escapes and stale known hashes
        # never become runnable merely by adding a produces declaration.
        def forbidden_inventory():raise AssertionError('Pending-asset CPU preflight invoked GPU')
        missing=root/'undeclared.bin'
        stub=dict(name='missing',kind='gpu',cwd=str(root),timeout_seconds=60,argv=[sys.executable,'-c','pass'],requires=[dict(path=str(missing))])
        rejected=ResourceGuard(dict(platform='autodl',stages=[stub]),root/'undeclared.json',inventory=forbidden_inventory)
        assert not rejected.build_preflight()
        future=dict(name='future',kind='gpu',cwd=str(root),timeout_seconds=60,argv=[sys.executable,'-c','pass'],produces=[dict(path=str(missing))])
        rejected=ResourceGuard(dict(platform='autodl',stages=[stub,future]),root/'forward.json',inventory=forbidden_inventory)
        assert not rejected.build_preflight()
        stale=root/'stale.bin';stale.write_bytes(b'wrong')
        producer=dict(future,name='stale_producer',produces=[dict(path=str(stale))])
        consumer=dict(stub,name='stale_consumer',requires=[dict(path=str(stale),sha256=hashlib.sha256(b'expected').hexdigest())])
        rejected=ResourceGuard(dict(platform='autodl',stages=[producer,consumer]),root/'stale.json',inventory=forbidden_inventory)
        assert not rejected.build_preflight()
        try:
            ResourceGuard(dict(stages=[dict(future,produces=[dict(path='../escaped.pt')])]),root/'escape.json')
        except ValueError:pass
        else:raise AssertionError('Produced path escape accepted')
        # No-card preflight must stream large files and never call CUDA/GPU hooks.
        from unittest.mock import patch
        weights=root/'frozen_weight_fixture.bin';expected=hashlib.sha256()
        block=b'fixed-weight-bytes'*4096
        with weights.open('wb') as stream:
            for _ in range(33):stream.write(block);expected.update(block)
        def forbidden_gpu(*args,**kwargs):raise AssertionError('CPU preflight touched GPU/CUDA/shutdown')
        artifacts=[dict(path=str(weights),sha256=expected.hexdigest())]
        stage=dict(name='preflight_only_fixture',kind='gpu',argv=[sys.executable,'-c','pass'],timeout_seconds=60,requires=artifacts)
        preflight=ResourceGuard(dict(platform='autodl',stages=[stage]),root/'no_card.json',inventory=forbidden_gpu,poweroff=forbidden_gpu)
        preflight.cuda_ready=forbidden_gpu;preflight.tensor_check=forbidden_gpu
        with patch.object(Path,'read_bytes',side_effect=AssertionError('Whole-file read is forbidden')):
            assert sha256_file(weights)==expected.hexdigest()
            assert check_artifacts(artifacts,root)
            assert preflight.build_preflight() and preflight.preflight_ready()
        assert preflight.status['state']=='PREFLIGHT_COMPLETED'
        assert not guard.status['billing_stop_confirmed']
    print(json.dumps(dict(cpu_state_machine='PASSED',cases=['ownedCUDAcontext_0util_60s','CPUchild_stalled','dutycycle_zero_under60s','util_unknown_not_idle','healthyqueued','immediate_terminal','foreignGPU_even0util','LinuxAutoDLlayout_required','GPUvisible_CUDAunavailable_shutdown','NOCARD_no_shutdown','streamed_weight_SHA','preflight_no_CUDA_or_GPU_calls','scientific_negative_healthy_next','ERROR_report_stops_even_exit0','declared_producer_consumer','invalid_produced_output_stops','undeclared_missing_rejected','forward_dependency_rejected','produced_path_escape_rejected','stale_input_hash_rejected'],real_GPU_calls=0,real_shutdown_calls=0,billing_stop_never_falsely_confirmed=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--state-file',type=Path)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--allow-shutdown',action='store_true')
    parser.add_argument('--idle-seconds',type=float,default=60)
    parser.add_argument('--self-check',action='store_true')
    a=parser.parse_args()
    if a.self_check:self_check();return
    if not a.plan or not a.state_file or a.idle_seconds<0:parser.error('--plan and --state-file required, idle-seconds>=0')
    plan=json.loads(a.plan.read_text());stages=validate_plan(plan)
    if a.preflight_only:
        guard=ResourceGuard(plan,a.state_file,allow_shutdown=False,idle_grace=a.idle_seconds)
        raise SystemExit(0 if guard.build_preflight() else 3)
    if not a.run:
        print(json.dumps(dict(state='DRY_RUN',stage_names=[s['name'] for s in stages],
            GPU_and_shutdown_actions_disabled=True,provider_documentation=AUTODL_DOC)));return
    guard=ResourceGuard(plan,a.state_file,allow_shutdown=a.allow_shutdown,idle_grace=a.idle_seconds)
    raise SystemExit(guard.run())


if __name__=='__main__':main()
