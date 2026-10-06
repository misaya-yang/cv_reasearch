"""Spawn-isolated source FoRIS readout of actual GPU encoder feature maps.

The caller owns the only real encoder. Each worker constructs only the source
FoRIS decoder and its private CRF using the existing frozen positional basis.
CUDA tensors travel through torch multiprocessing IPC without changing strides;
the producer keeps their storage alive until the result acknowledges the job.
"""
from contextlib import contextmanager
import queue
import sys
import time
import traceback

import torch


class ForbiddenEncoder(torch.nn.Module):
    """Constructor-compatible placeholder that cannot execute an encoder pass."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("Readout workers must never execute an encoder")

    def get_intermediate_layers(self, *args, **kwargs):
        raise RuntimeError("Readout workers must never execute an encoder")


def configure_threads(threads=2):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_decoder(manifest, device, foris_root=None):
    """Use the real FoRIS constructor, but require a basis and prohibit DINO."""
    from .native_basis import reuse_native_basis

    if not manifest.get("projection_basis"):
        raise ValueError("A frozen native positional basis is required; no dummy forward")
    sys.path.insert(0, str(foris_root or manifest["foris_root"]))
    from models.foris import FoRIS

    with reuse_native_basis(FoRIS, manifest["projection_basis"]) as basis_receipt:
        host = FoRIS(encoder=ForbiddenEncoder(), image_size=1024, svd_components=500,
                     tau=0.6, mask_refiner="crf", resize_to_orig_size=False,
                     device=device).eval().requires_grad_(False)
    if not isinstance(host.encoder, ForbiddenEncoder):
        raise RuntimeError("Decoder unexpectedly contains a real encoder")
    return host, basis_receipt


def clear_episode(host):
    host._ref_images = host._ref_masks = host._tgt_image = host._orig_tgt_size = None


def validate_job(job):
    fmap, ref, mask, target = (job[k] for k in ("fmaps", "ref_images", "ref_masks", "target"))
    if tuple(ref.shape) != (1, 3, 1024, 1024) or tuple(target.shape) != (3, 1024, 1024):
        raise ValueError("Expected the actual transformed reference/query RGB at 1024")
    if tuple(mask.shape) != (1, 1024, 1024) or mask.dtype != torch.bool:
        raise ValueError("Expected the source-transformed one-reference boolean mask")
    if tuple(fmap.shape) != (1, 2, 1024, 64, 64) or fmap.dtype != torch.float32:
        raise ValueError("Expected actual FP32 [1, reference+query, 1024, 64, 64] maps")
    if tuple(fmap.stride()) != tuple(job["fmap_stride"]):
        raise ValueError("IPC changed feature strides and therefore the normalization layout")
    if len({x.device for x in (fmap, ref, mask, target)}) != 1:
        raise ValueError("RGB, mask and all feature matrices must stay on the same device")


@contextmanager
def cached_features(host, job):
    """Return the exact raw fmap from source predict's sole extraction call."""
    validate_job(job)
    name = "_extract_features"
    had_value, old_value = name in host.__dict__, host.__dict__.get(name)
    receipt = {"calls": 0, "encoder_forwards": 0,
               "shape": list(job["fmaps"].shape), "stride": list(job["fmaps"].stride())}

    def extract(images):
        if tuple(images.shape) != (1, 2, 3, 1024, 1024):
            raise ValueError("Source predict changed its one-reference paired batch layout")
        # Check both actual normalized RGB items, not just a nominal shape.
        if not (torch.equal(images[0, :1], job["ref_images"])
                and torch.equal(images[0, 1], job["target"])):
            raise ValueError("Cached features were paired with different transformed RGB")
        receipt["calls"] += 1
        if receipt["calls"] != 1:
            raise RuntimeError("Source readout requested more than one feature extraction")
        return job["fmaps"]

    setattr(host, name, extract)
    try:
        yield receipt
        if receipt["calls"] != 1:
            raise RuntimeError("Source readout did not consume the cached pair exactly once")
    finally:
        if had_value:
            setattr(host, name, old_value)
        else:
            delattr(host, name)


@torch.inference_mode()
def readout(host, job):
    """Original segment/predict, original target state, original binarizer/CRF."""
    import numpy as np
    from .foris import observe

    started = time.monotonic()
    validate_job(job)
    if host.positional_basis.device != job["fmaps"].device:
        raise ValueError("Readout host and cached features must share their CUDA device")
    if host._ref_images is not None or host._tgt_image is not None:
        raise RuntimeError("Worker still has active state from a previous job")
    previous_decision = host.should_debiass
    try:
        host._ref_images, host._ref_masks = job["ref_images"], job["ref_masks"]
        host._tgt_image, host._orig_tgt_size = job["target"], tuple(job["orig_target_size"])
        with cached_features(host, job) as cache_receipt, observe(host) as observed:
            prediction = host.segment()
        # These CPU copies also synchronize this process's GPU readout/CRF.
        packed = np.packbits(prediction.reshape(1024, 1024).bool().cpu().numpy())
        score = observed["score"].float().cpu().numpy().copy()
        pre = observed["pre"].bool().cpu().numpy()
        if not np.isfinite(score).all() or score.shape != (64, 64):
            raise ValueError("Source readout did not produce a finite native 64 x 64 score")
        return dict(id=job["id"], key=job["key"], arm=job["arm"], prediction=packed,
                    score=score, pre=np.packbits(pre), pre_shape=list(pre.shape),
                    receipt=dict(cache_receipt, private_target_state=True, private_crf=True,
                                 complete_source_segment=True, debias_applied=bool(host.should_debiass)),
                    timings=dict(seconds_readout_and_cpu_copy=time.monotonic() - started,
                                 seconds_queue_wait=started - job["submitted_at"]))
    finally:
        clear_episode(host)
        host.should_debiass = previous_decision


def _worker(manifest, device, foris_root, threads, jobs, results, worker_index):
    try:
        configure_threads(threads)
        # BLAS limits are per process and remain active for sklearn clustering.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=threads):
            host, basis = build_decoder(manifest, device, foris_root)
            import os
            results.put(dict(state="READY", worker=worker_index, pid=os.getpid(),
                             torch_threads=torch.get_num_threads(), basis=basis,
                             encoder_instances=0, private_crf=True))
            while True:
                job = jobs.get()
                if job is None:
                    break
                answer = readout(host, job)
                answer["worker"] = worker_index
                # Drop all received CUDA tensors before acknowledging ownership.
                job = None
                results.put(dict(state="RESULT", result=answer))
    except BaseException:
        results.put(dict(state="ERROR", worker=worker_index, traceback=traceback.format_exc()))


class ReadoutPool:
    """Four (or three) private decoder processes; bounded parent-owned IPC."""

    def __init__(self, manifest, device="cuda", foris_root=None, workers=4, threads=2):
        if workers not in (3, 4) or threads != 2:
            raise ValueError("This runner fixes 3/4 readout workers, each with 2 CPU threads")
        self.context = torch.multiprocessing.get_context("spawn")
        self.jobs, self.results = self.context.Queue(), self.context.Queue()
        self.pending, self.ready = {}, []
        self.processes = [self.context.Process(target=_worker,
            args=(manifest, device, foris_root, threads, self.jobs, self.results, index))
            for index in range(workers)]
        self.closed = False

    def start(self):
        for process in self.processes:
            process.start()
        try:
            while len(self.ready) < len(self.processes):
                event = self._event()
                if event["state"] != "READY":
                    raise RuntimeError(f"Unexpected startup event: {event['state']}")
                self.ready.append(event)
        except BaseException:
            self.close()
            raise
        return self.ready

    def submit(self, job):
        if self.closed or job["id"] in self.pending:
            raise ValueError("Closed pool or duplicate job ID")
        validate_job(job)
        job["submitted_at"] = time.monotonic()
        self.pending[job["id"]] = job  # Pin producer CUDA storage until acknowledged.
        self.jobs.put(job)

    def _event(self, block=True):
        while True:
            try:
                event = self.results.get(timeout=1) if block else self.results.get_nowait()
            except queue.Empty:
                failed = [p.pid for p in self.processes if p.exitcode is not None]
                if failed:
                    raise RuntimeError(f"Readout processes exited unexpectedly: {failed}")
                if not block:
                    return None
                continue
            if event["state"] == "ERROR":
                raise RuntimeError(f"Readout worker {event['worker']} failed:\n{event['traceback']}")
            return event

    def collect(self, block=True):
        event = self._event(block)
        if event is None:
            return None
        if event["state"] != "RESULT":
            raise RuntimeError(f"Unexpected readout event: {event['state']}")
        result = event["result"]
        if result["id"] not in self.pending:
            raise RuntimeError("Unknown or duplicate worker result")
        self.pending.pop(result["id"])
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Drain on success before calling close. Failed jobs must stay pinned
        # until their consumers exit; do not free live CUDA IPC storage early.
        if not self.pending:
            for _ in self.processes:
                self.jobs.put(None)
        for process in self.processes:
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        self.pending.clear()
        for channel in (self.jobs, self.results):
            channel.cancel_join_thread()
            channel.close()
