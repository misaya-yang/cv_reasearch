"""Local-only execution primitives for the fixed W71 weekend queue."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '2'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', MAX_JOBS='2')
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import time
from datetime import datetime, timezone
from collections import OrderedDict

ROOT = Path(__file__).resolve().parents[2]
ASSETS = (ROOT.parent/'cv_data').resolve()
OUT = ASSETS/'a/x1013_W71_offline'
PYTHON = Path.home()/'miniconda3/envs/aidemo/bin/python'
MEMORY_LIMIT_GIB = 25
MPS_OWNER = False
sys.path[:0] = [str(ROOT/'scripts/x1012'), str(ROOT/'scripts/x1013'), str(ROOT/'scripts/x1012s'), str(ROOT/'src')]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=1)
        stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def event(kind, **fields):
    row = dict(utc=utc(), pid=os.getpid(), kind=kind, **fields)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


_OFFLINE = False
def offline():
    """Deny internet in every Python process; UNIX multiprocessing sockets work."""
    global _OFFLINE
    if _OFFLINE: return
    def audit(name, args):
        if name in ('socket.connect', 'socket.connect_ex') and args[0].family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('W71_OFFLINE_NETWORK_DENIED')
        if name == 'socket.getaddrinfo':
            raise RuntimeError('W71_OFFLINE_DNS_DENIED')
    sys.addaudithook(audit); _OFFLINE = True


def initialize():
    offline()
    import torch
    import common as c
    from threadpoolctl import threadpool_limits
    c.initialize(); torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    global LIMITER
    LIMITER = threadpool_limits(2)
    os.nice(max(0, 15-os.nice(0)))
    return c


class StopRun(RuntimeError): pass


def deadline():
    p = OUT/'session.json'
    return read(p)['deadline_epoch'] if p.exists() else float('inf')


def checkpoint():
    flag=OUT/'STOP_REQUESTED'
    if (flag.exists() and read(flag)['active']) or time.time() >= deadline():
        raise StopRun('Stop requested or persisted 48-hour deadline reached')


def heartbeat(stage, done=0, total=0, **extra):
    import psutil
    process = psutil.Process()
    write(OUT/'progress.json', dict(utc=utc(), epoch=time.time(), pid=os.getpid(), stage=stage,
          done=done, total=total, rss_gib=process.memory_info().rss/2**30,
          available_gib=psutil.virtual_memory().available/2**30, memory_budget_gib=MEMORY_LIMIT_GIB,
          conservative_job_memory_gib=job_memory_gib(), **extra))


def job_memory_gib():
    """RSS tree + this encoder's Metal allocation; overlap is conservative."""
    import psutil
    process=psutil.Process(); rss=0
    for p in [process]+process.children(recursive=True):
        try: rss+=p.memory_info().rss
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    torch=sys.modules.get('torch')
    metal=torch.mps.driver_allocated_memory() if MPS_OWNER and torch is not None else 0
    return (rss+metal)/2**30


def capacity(minimum_gib=6, maximum_wait=120, reserve_gib=3):
    """Backpressure only; never change scientific settings or kill other jobs."""
    import psutil
    began=time.monotonic(); reclaimed=False
    while (psutil.virtual_memory().available < minimum_gib*2**30 or
           job_memory_gib()>MEMORY_LIMIT_GIB-reserve_gib or psutil.cpu_percent(interval=.2)>80):
        if not reclaimed and (psutil.virtual_memory().available < minimum_gib*2**30 or job_memory_gib()>MEMORY_LIMIT_GIB-reserve_gib):
            import gc
            gc.collect()
            torch=sys.modules.get('torch')
            if torch is not None and torch.backends.mps.is_available(): torch.mps.empty_cache()
            reclaimed=True
            if (psutil.virtual_memory().available >= minimum_gib*2**30 and
                job_memory_gib()<=MEMORY_LIMIT_GIB-reserve_gib and psutil.cpu_percent(interval=.2)<=80):
                break
        checkpoint(); heartbeat('waiting_for_resources'); time.sleep(2)
        if time.monotonic()-began>maximum_wait:
            raise RuntimeError('Resource budget unavailable for 120s; restart with fewer workers')
    checkpoint()


def worker_budget(requested, available_gib, current_job_gib=0):
    return max(1,min(requested,int(max(0,available_gib-6)//2),int(max(0,MEMORY_LIMIT_GIB-current_job_gib-3)//2)))


def valid_record(path):
    if not Path(path).exists(): return None
    r = read(path)
    p = Path(r['prediction_path'])
    if not p.is_file() or sha(p) != r['prediction_sha256']:
        raise RuntimeError(f'Committed prediction is missing/corrupt: {path}')
    return r


def check_mask(binding,role):
    """Check original binary-mask identity before resizing; GT only in scorers."""
    import numpy as np
    from PIL import Image
    from ics.official_data import array_hash
    row=binding['row']; expected=row.get(role+'_mask_hash')
    if expected:
        with Image.open(row[role+'_mask_path']) as image:
            value=(np.asarray(image.convert('L'))>0).astype(np.uint8)
        assert array_hash(value)==expected, f'Frozen {role} mask changed'


class Encoder:
    """One MPS owner, bounded RAM cache, measured 80% submission duty cycle."""
    def __init__(self, layers=(23,), cache_limit=24, duty=.8):
        global MPS_OWNER
        import torch
        import common as c
        self.torch, self.c = torch, c
        # timm emits intermediates in traversal order, regardless of list order.
        self.layers, self.limit, self.duty = tuple(sorted(set(layers))), cache_limit, duty
        self.cache = OrderedDict(); self.calls = 0; self.encode_seconds = 0.
        source = ASSETS/'a/x1013_official/frozen/encoder.py'
        spec = importlib.util.spec_from_file_location('w71_encoder_impl', source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.model = module.TimmDINOv3(str(ASSETS/'demo4_cache/models/dinov3-vitl16-timm')).to('mps').eval().requires_grad_(False)
        MPS_OWNER = True

    def features(self, binding):
        from PIL import Image
        torch = self.torch; result = {}; traces = []
        with self.c.rc.pc.inference_only(binding), torch.inference_mode():
            for role in ('reference', 'query'):
                row = binding['row']; path = ASSETS/row[role+'_path']; crop = row.get(role+'_crop')
                key = (str(path), json.dumps(crop))
                if key not in self.cache:
                    checkpoint(); capacity(); tick = time.monotonic()
                    with Image.open(path) as image:
                        image = image.convert('RGB')
                        if crop is not None: image = image.crop(tuple(crop))
                        if row.get(role+'_rgb_hash'):
                            import numpy as np
                            from ics.official_data import array_hash
                            assert array_hash(np.asarray(image))==row[role+'_rgb_hash'], 'Frozen RGB input changed'
                        x = self.c.rc.host.transform(image)[None].to('mps')
                    maps = self.model.get_intermediate_layers(x, n=list(self.layers), reshape=True)
                    torch.mps.synchronize()
                    self.cache[key] = {layer: m[0].cpu().permute(1,2,0).reshape(4096,1024).contiguous()
                                       for layer,m in zip(self.layers,maps)}
                    elapsed = time.monotonic()-tick; self.encode_seconds += elapsed; self.calls += 1
                    del maps, x
                    time.sleep(elapsed*(1/self.duty-1))
                    if self.calls % 32 == 0: torch.mps.empty_cache()
                    fresh = True
                else: fresh = False
                self.cache.move_to_end(key)
                # Hold role tensors before evicting; R may equal an earlier Q.
                result[role] = self.cache[key]
                traces.append(dict(role=role, fresh=fresh))
                while len(self.cache) > self.limit: self.cache.popitem(last=False)
        return result, traces
