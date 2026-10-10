"""Three real COCO encodes: separate R/Q versus one horizontal joint canvas.

Each arm uses frozen DINOv3 O24 at the same side/device and identical evidence
rules, without positional debias. Native unnormalized FP32 outputs and exact
inputs are saved losslessly before deriving unit tokens. No masked arm is part
of this probe. Side768 capacity is unverified until the real protected run.
The documented MPS budget is reinstated at0.4 of recommended memory; an OOM
stops the probe with a receipt, without a retry or a larger memory allowance.
"""
import builtins
from contextlib import contextmanager
import importlib.metadata
import io
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inmask_evidence as base


def rss_max():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def memory():
    return dict(allocated_bytes=int(torch.mps.current_allocated_memory()),
                driver_bytes=int(torch.mps.driver_allocated_memory()), pid=os.getpid(), rss_peak_bytes=rss_max())


class ObservedMemory:
    """Actual sampled allocator maxima; torch MPS has no exact peak/reset API."""
    def __init__(self):
        self.before = memory()
        self.peak, self.samples, self.stopped = dict(self.before), 0, threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.stopped.is_set():
            value = memory()
            for key in ('allocated_bytes', 'driver_bytes', 'rss_peak_bytes'):
                self.peak[key] = max(self.peak[key], value[key])
            self.samples += 1
            self.stopped.wait(.001)

    def finish(self):
        self.stopped.set(); self.thread.join()
        after = memory()
        for key in ('allocated_bytes', 'driver_bytes', 'rss_peak_bytes'):
            self.peak[key] = max(self.peak[key], after[key])
        return dict(before=self.before, after=after, observed_peak=self.peak, samples=self.samples,
                    sampling_interval_seconds=.001, exact_allocator_peak=False, rss_peak_source='OS RUSAGE_SELF ru_maxrss')


@contextmanager
def deny_query_labels(rows, assets, attempts):
    from PIL import Image
    forbidden = set()
    for row in rows:
        for key in ('query_mask_path', 'query_ignore_mask_path', 'query_annotation_path'):
            if row.get(key):
                path = Path(row[key])
                forbidden.add(str((path if path.is_absolute() else assets / path).resolve()))
        # COCO manifests also retain the original source annotation identity.
        for name in row.get('source_files', {}):
            path = Path(name)
            if path.stem == Path(row['query_path']).stem and 'annotations' in path.parts:
                forbidden.add(str((path if path.is_absolute() else assets / path).resolve()))
    saved = builtins.open, io.open, Image.open
    def wrap(original):
        def opened(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)) and str(Path(os.fsdecode(path)).resolve()) in forbidden:
                attempts.append(str(path))
                raise RuntimeError('Query label denied during joint fields: ' + str(path))
            return original(path, *args, **kwargs)
        return opened
    builtins.open, io.open, Image.open = map(wrap, saved)
    try:
        yield
    finally:
        builtins.open, io.open, Image.open = saved


def joint_fields(args):
    source, out = base.opened(args)
    assets, grid = Path(args.assets).resolve(), args.side // 16
    assert args.cohort and Path(source.where).resolve() == assets / 'a/coco_role_competition200_20261010'
    assert args.name == 'joint_coco_try' and args.limit == 3 and args.shard == '0/1'
    assert args.side == 768 and args.device == 'mps' and args.encoder == 'dinov3' and not args.masked
    assert torch.backends.mps.is_available()
    assert not any(os.environ.get(k) == '0.0' or os.environ.get(k) == '0' for k in ('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_LOW_WATERMARK_RATIO'))
    assert not (out / 'OOM.json').exists() and not (out / 'FAILED.json').exists(), 'A stopped probe is not retried'
    rows = source.rows[:3]
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(base.REPO / 'src'), str(base.REPO / 'scripts')]
    from raw_feature_cache import RawFeatureCache, tensor_hash
    from ics.data import TimmDINOv3
    sources = [Path(__file__).resolve(), Path(base.__file__).resolve(), base.REPO / 'src/ics/data.py',
               base.REPO / 'src/ics/official_data.py', base.REPO / 'scripts/raw_feature_cache.py',
               assets / 'third_party/foris_official/utils/data.py']
    source_hashes = {str(p): base.sha(p) for p in sources}
    mdir = assets / 'demo4_cache/models/dinov3-vitl16-timm'
    recommended = int(torch.mps.recommended_max_memory())
    profile = dict(model='DINOv3-L/16', architecture=base.read(mdir / 'config.json')['architecture'],
        weights_sha256=base.sha(mdir / 'model.safetensors'), model_config_sha256=base.sha(mdir / 'config.json'),
        source_sha256=source_hashes, torch_version=str(torch.__version__), timm_version=importlib.metadata.version('timm'),
        torchvision_version=importlib.metadata.version('torchvision'), producer_device=args.device,
        encoder_dtype='float32', storage_dtype='float32', branch='O/24', output='model.norm final block output; no token L2 or positional projection',
        preprocessing=dict(source_sha256=base.sha(assets / 'third_party/foris_official/utils/data.py'),
            rgb=True, side=args.side, mean=[.485,.456,.406], std=[.229,.224,.225],
            input='FP32 CHW; separate768x768 or unmasked horizontal R/Q768x1536'),
        mps_memory_fraction=.4, mps_recommended_max_bytes=recommended, mps_requested_cap_bytes=int(.4 * recommended))
    cache = RawFeatureCache(out / 'raw_cache', profile)
    plan = dict(state='FROZEN_BEFORE_ENCODING_OR_QUERY_GT', arguments=vars(args),
        git_head=subprocess.run(['git','rev-parse','HEAD'], cwd=base.REPO, capture_output=True, text=True, check=True).stdout.strip(),
        ids=[source.describe(r)['id'] for r in rows], manifest_sha256=base.sha(Path(source.where) / 'manifest.json'),
        prefix_rows=rows, profile_id=cache.profile_id, profile_path=str(cache.folder / 'profile.json'),
        profile_sha256=base.sha(cache.folder / 'profile.json'), source_sha256=source_hashes,
        arms=['separate', 'joint'], expected_encoder_requests=9, normal_mps_memory_protection=True,
        mps_memory_fraction=.4, recommended_max_memory_bytes=recommended, requested_cap_bytes=int(.4 * recommended),
        mps_environment={k:v for k,v in os.environ.items() if k.startswith('PYTORCH_MPS_')},
        query_GT_reads=0, joint_features='actual full-canvas encoder output; never synthesized from separate features')
    if (out / 'joint_plan.json').exists():
        assert base.read(out / 'joint_plan.json') == plan, 'Frozen arguments/source/input profile changed'
    else:
        base.write(out / 'joint_plan.json', plan)
    torch.mps.set_per_process_memory_fraction(.4)              # reinstate the documented11.23GiB budget before model allocation
    torch.manual_seed(0)
    initializing, initialized_at = ObservedMemory(), time.monotonic()
    try:
        encoder = TimmDINOv3(str(mdir)).eval().requires_grad_(False).to('mps')
        torch.mps.synchronize()
    except RuntimeError as error:
        failure = dict(state='OOM' if 'out of memory' in str(error).lower() else 'FAILED',
            operation='model_initialization', error=str(error), seconds=time.monotonic()-initialized_at,
            **initializing.finish(), completed_raw_requests=0, query_GT_reads=0,
            joint_plan_sha256=base.sha(out / 'joint_plan.json'), no_retry=True, no_memory_limit_relaxation=True)
        base.write(out / (failure['state'] + '.json'), failure)
        raise
    base.write(out / 'encoder_initialization.json', dict(state='REAL_MODEL_INITIALIZED',
        seconds=time.monotonic()-initialized_at, **initializing.finish()))
    transform = base.official_transform(assets, args.side)
    (out / 'raw_inputs').mkdir(exist_ok=True)
    requests, attempts = [], []

    @torch.inference_mode()
    def tokens(x, operation, row):
        array = x[0].numpy()
        key = cache.key(array)
        input_path = out / 'raw_inputs' / (key + '.npy')
        if input_path.exists():
            assert tensor_hash(np.load(input_path, allow_pickle=False)) == tensor_hash(array)
        else:
            with input_path.open('xb') as handle:
                np.save(handle, array, allow_pickle=False)
        gh, gw = array.shape[1] // 16, array.shape[2] // 16
        info_path = cache.folder / key / 'entry.json'
        metrics, live = dict(state='CACHE_HIT', synchronized_forward_seconds=0., encoder_calls=0), None
        if not info_path.exists():
            sampled = ObservedMemory()
            started = time.monotonic()
            try:
                device_input = x.to('mps'); torch.mps.synchronize()
                tick = time.monotonic()
                maps = encoder.get_intermediate_layers(device_input, n=1, reshape=True)[0]
                torch.mps.synchronize()
                forward_seconds = time.monotonic() - tick
                assert maps.device.type == 'mps' and maps.dtype == torch.float32
                assert tuple(maps.shape) == (1,1024,gh,gw)
                live = maps[0].cpu().permute(1,2,0).contiguous().numpy().reshape(-1,1024)
                assert live.shape == (gh * gw, 1024) and live.dtype == np.float32
                del maps, device_input
            except RuntimeError as error:
                metrics = dict(state='OOM' if 'out of memory' in str(error).lower() else 'FAILED',
                    operation=operation, episode_id=row['episode_id'], input_shape=list(x.shape),
                    seconds=time.monotonic()-started, error=str(error), **sampled.finish())
                base.write(out / (metrics['state'] + '.json'), dict(metrics, joint_plan_sha256=base.sha(out / 'joint_plan.json'),
                    completed_raw_requests=len(requests), query_GT_reads=0, no_retry=True, no_memory_limit_relaxation=True))
                raise
            metrics = dict(state='REAL_ENCODER_SUCCESS', synchronized_forward_seconds=forward_seconds,
                encode_and_cpu_copy_seconds=time.monotonic()-started, encoder_calls=1, **sampled.finish())
            with cache._locked(array, True):
                assert not info_path.exists(), 'One producer owns this fresh raw cache'
                cache._write(array, {'O/24':live}, dict(episode_id=row['episode_id'], operation=operation,
                    encoder_device='mps', encoder_dtype='float32', batch=1, actual_full_canvas=operation == 'joint.canvas'))
        raw = cache.read(array, ('O/24',))['O/24']
        assert raw.shape == (gh * gw, 1024)
        unit = F.normalize(torch.from_numpy(raw.reshape(gh,gw,1024)), dim=2)
        if live is not None:
            assert np.array_equal(raw, live)
            assert torch.equal(unit, F.normalize(torch.from_numpy(live.reshape(gh,gw,1024)), dim=2))
        info = base.read(info_path)
        request = dict(episode_id=row['episode_id'], operation=operation, key=key, input_path=str(input_path),
            input_file_sha256=base.sha(input_path), input_tensor_hash=info['input_tensor_hash'], input_shape=list(array.shape),
            entry_path=str(info_path), entry_sha256=base.sha(info_path), payload_path=str(info_path.parent / info['file']),
            payload_sha256=info['file_sha256'], raw_shape=list(raw.shape), raw_tensor_sha256=info['features']['O/24']['tensor_sha256'],
            unit_tensor_sha256=tensor_hash(unit.numpy()), native_raw_serialized_parity=True,
            serialized_unit_feature_parity=True, new_cache_entry=live is not None, telemetry=metrics)
        requests.append(request)
        base.write(out / 'raw_bindings.json', requests)
        print(dict(operation=operation, episode_id=row['episode_id'], state=metrics['state'],
            forward_seconds=metrics['synchronized_forward_seconds'], raw_shape=list(raw.shape)), flush=True)
        return unit, request

    def compute(source, rows, described, i):
        assert all(base.sha(Path(p)) == h for p,h in source_hashes.items()), 'Frozen source changed'
        row = rows[i]
        reference, mask, query = source.images(row)
        xr, xq = transform(reference)[None].float(), transform(query)[None].float()
        assert tuple(xr.shape) == tuple(xq.shape) == (1,3,args.side,args.side)
        coverage = base.reference_coverage(mask, grid=grid, side=args.side)
        candidate = base.candidate_tokens(source.prediction(row),grid)
        Rs, sr = tokens(xr, 'separate.reference', row)
        Qs, sq = tokens(xq, 'separate.query', row)
        both, sj = tokens(torch.cat((xr,xq),3), 'joint.canvas', row)
        assert tuple(both.shape) == (grid,2*grid,1024)
        Rj, Qj = both[:,:grid].reshape(-1,1024), both[:,grid:].reshape(-1,1024)
        Rs, Qs = Rs.reshape(-1,1024), Qs.reshape(-1,1024)
        fields, note = {}, {}
        for name,R,Q in [('separate',Rs,Qs),('joint',Rj,Qj)]:
            found, note = base.evidence(R,Q,coverage,candidate)
            fields.update({f'{name}.{k}':v for k,v in found.items() if not k.startswith('aux.')})
        target = base.target_tokens(coverage)
        background = (coverage < .05) & ~target
        if background.any():
            toward = F.normalize(Rj[target].mean(0),dim=0) - F.normalize(Rj[background].mean(0),dim=0)
            fields['shift.joint'] = ((Qj-Qs) @ toward).double().numpy()
        return fields, dict(note, side=args.side, separate_seconds=sr['telemetry']['synchronized_forward_seconds']+sq['telemetry']['synchronized_forward_seconds'],
            joint_seconds=sj['telemetry']['synchronized_forward_seconds'], raw_requests=[sr,sq,sj], query_GT_reads=0,
            encoder_calls=sum(r['telemetry']['encoder_calls'] for r in [sr,sq,sj]),
            new_cache_count=sum(r['new_cache_entry'] for r in [sr,sq,sj]),
            query_label_attempts=len(attempts), joint_plan_sha256=base.sha(out / 'joint_plan.json'),
            reference_tokens_kept=float((Rj*Rs).sum(1).mean()), query_tokens_kept=float((Qj*Qs).sum(1).mean()))

    with deny_query_labels(source.rows, assets, attempts):
        base.make_fields(args, compute=compute, grid=grid)
    assert not attempts
    records = [base.read(out / 'records' / f'{i:06d}.json') for i in range(3)]
    assert [r['id'] for r in records] == base.read(out / 'config.json')['ids'] == plan['ids']
    for i, record in enumerate(records):
        assert base.sha(out / 'fields' / f'{i:06d}.npz') == record['sha256']
    all_requests = [r for rec in records for r in rec['raw_requests']]
    for request in all_requests:
        array = np.load(request['input_path'],allow_pickle=False)
        raw = cache.read(array, ('O/24',))['O/24']
        assert tensor_hash(raw) == request['raw_tensor_sha256']
        gh,gw = array.shape[1]//16, array.shape[2]//16
        assert tensor_hash(F.normalize(torch.from_numpy(raw.reshape(gh,gw,1024)),dim=2).numpy()) == request['unit_tensor_sha256']
    base.write(out / 'raw_replay.json', dict(state='PASS_SERIALIZED_RAW_AND_UNIT_FEATURE_PARITY', n=3, requests=len(all_requests),
        raw_bindings_sha256=base.sha(out / 'raw_bindings.json'), profile_sha256=base.sha(cache.folder / 'profile.json'),
        encoder_calls=0, query_GT_reads=0))
    base.write(out / 'sealed.json', dict(n=3, fields={r['id']:r['sha256'] for r in records}, query_GT_read=False,
        config_sha256=base.sha(out / 'config.json'),
        encoder_calls=sum(r['telemetry']['encoder_calls'] for r in all_requests), script_sha256=sorted({r['script_sha256'] for r in records}),
        joint_plan_sha256=base.sha(out / 'joint_plan.json'), raw_replay_sha256=base.sha(out / 'raw_replay.json')))
    peaks = [r['telemetry']['observed_peak'] for r in all_requests if r['telemetry']['encoder_calls']]
    base.write(out / 'SUCCESS.json', dict(state='REAL_THREE_EPISODE_FIELDS_SEALED', n=3, actual_encoder_calls=len(peaks),
        unique_raw_inputs=len({r['key'] for r in all_requests}), profile_id=cache.profile_id, query_GT_reads=0,
        query_label_attempts=0, requested_cap_bytes=int(.4*recommended), recommended_max_memory_bytes=recommended,
        observed_peak={k:max(p[k] for p in peaks) for k in ('allocated_bytes','driver_bytes','rss_peak_bytes')},
        exact_allocator_peak=False, mps_memory_fraction=.4, no_memory_limit_override=True,
        sealed_sha256=base.sha(out / 'sealed.json')))
    print(dict(state='REAL_THREE_EPISODE_FIELDS_SEALED', n=3, encoder_calls=len(peaks)), flush=True)


if __name__ == '__main__':
    parser = base.arguments(['fields'])
    parser.add_argument('--device', default='mps', choices=['mps','cuda','cpu'])
    parser.add_argument('--side', type=int, default=768)
    parser.add_argument('--masked', action='store_true')
    parser.add_argument('--encoder', default='dinov3', choices=['dinov3'])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    joint_fields(args)
