"""Bound CPU-only native DINOv3 final-LN views, with a bounded per-worker cache.

Import is NumPy-only. Constructing the real adapter explicitly loads the
existing local timm model. It never downloads, accepts a mask, or reads labels.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np

ARCHITECTURE = "vit_large_patch16_dinov3"
INPUT_SHAPE = (1024, 1024, 3)
OUTPUT_SHAPE = (64, 64, 1024)
MAX_CACHED_VIEWS = 16
_NATIVE_KINDS = {"frozen_DINOv3_FP32_native_final_LN_patches",
                 "frozen_DINOv3_FP32_native_final_LN_patches_then_unit"}
_WORKER_ENCODERS = {}
_FACTORY_LOCK = threading.RLock()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Known SHA256 required: " + name)
    return value


def _assets(producer):
    assets = producer.get("model_assets", producer)
    if not isinstance(assets, dict):
        raise ValueError("Explicit model asset dictionary required")
    return {"checkpoint_sha256": _hash(assets.get("checkpoint_sha256"), "checkpoint"),
            "config_sha256": _hash(assets.get("config_sha256", assets.get("model_config_sha256")), "config"),
            "timm_eva_sha256": assets.get("timm_eva_sha256"),
            "layernorm_source_sha256": assets.get("layernorm_source_sha256", assets.get("norm_source_sha256"))}


def assert_matches_producer(producer, binding):
    """Compare every available frozen source identity; unknown fields stay unknown."""
    if (not isinstance(producer, dict) or producer.get("FoRIS_Part1_applied") is not False
            or producer.get("kind") not in _NATIVE_KINDS):
        raise ValueError("Native FP32 final-LN DINO producer required; processed cache cannot be substituted")
    expected, actual = _assets(producer), _assets(binding)
    for key, value in expected.items():
        if value is not None and _hash(value, key) != _hash(actual.get(key), "actual " + key):
            raise ValueError("Frozen encoder producer identity differs: " + key)
    architecture = producer.get("architecture", producer.get("model_assets", {}).get("architecture"))
    if architecture is not None and architecture != ARCHITECTURE:
        raise ValueError("Episode architecture differs from the pinned DINOv3-L/16")
    if binding.get("architecture") != ARCHITECTURE:
        raise ValueError("Callback is not the pinned architecture")
    if tuple(binding.get("output_shape", ())) != OUTPUT_SHAPE or binding.get("native_output_dtype") != "float32":
        raise ValueError("Callback must return native FP32 patch grid, before unit normalization")
    return expected


def _stamp(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns, s.st_ino


class CachedCPUEncoder:
    """LRU wrapper usable with a declared fake callback for interface tests.

    The wrapper checks shape/precision and identity. It does not certify a
    custom callback as a real model; only the real factory sets that binding.
    Returned cached arrays are read-only to prevent accidental modifications.
    """
    def __init__(self, callback: Callable, binding: dict, *, max_cached_views=16):
        if not callable(callback) or not isinstance(max_cached_views, int) or not 0 <= max_cached_views <= MAX_CACHED_VIEWS:
            raise ValueError("Callable and bounded0..16 cache required")
        self._callback = callback
        self._real_execution = isinstance(callback, _TimmCPUForward)
        self._binding = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
        if tuple(self._binding.get("output_shape", ())) != OUTPUT_SHAPE or self._binding.get("native_output_dtype") != "float32":
            raise ValueError("Explicit native FP32 output interface binding required")
        self._binding_key = hashlib.sha256(json.dumps(self._binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self._maximum = max_cached_views
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._hits = self._misses = self._attempts = self._successes = 0
        self._forward_wall = self._forward_cpu = 0.

    @property
    def binding(self):
        return json.loads(json.dumps(self._binding, sort_keys=True))

    def validate_producer(self, producer):
        return assert_matches_producer(producer, self._binding)

    def __call__(self, rgb):
        image = np.asarray(rgb)
        if image.shape != INPUT_SHAPE or image.dtype != np.uint8:
            raise ValueError("Encoder input must be uint8 RGB[1024,1024,3], already transformed by the method")
        image = np.ascontiguousarray(image)
        key = self._binding_key + ":" + hashlib.sha256(memoryview(image).cast("B")).hexdigest()
        with self._lock:
            if self._real_execution:
                self._callback.check_frozen()
            if key in self._cache:
                self._hits += 1
                value = self._cache.pop(key)
                self._cache[key] = value
                return value.view()
            self._misses += 1
            self._attempts += 1
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                value = np.asarray(self._callback(image))
            finally:
                self._forward_wall += time.perf_counter()-wall
                self._forward_cpu += time.process_time()-cpu
            if value.shape != OUTPUT_SHAPE or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Callback output must be finite native FP32[64,64,1024]; no FP16, unit export or missing prefixes")
            self._successes += 1
            value = np.array(value, dtype=np.float32, copy=True, order="C")
            value.setflags(write=False)
            if self._maximum:
                self._cache[key] = value
                while len(self._cache) > self._maximum:
                    self._cache.popitem(last=False)
            return value.view()

    def stats(self):
        with self._lock:
            real = self._real_execution
            return {"cache_hits": self._hits, "cache_misses": self._misses,
                    "callback_forward_attempts": self._attempts, "callback_successful_forwards": self._successes,
                    "new_encoder_forwards": self._attempts if real else 0,
                    "real_encoder_execution": real, "precision": "float32",
                    "model_training": self._callback._model.training if real else None,
                    "gradient_disabled": all(not t.requires_grad for t in self._callback._model.parameters()) if real else None,
                    "forward_wall_seconds": self._forward_wall, "forward_cpu_seconds": self._forward_cpu,
                    "cached_views": len(self._cache), "maximum_cached_views": self._maximum,
                    "cached_feature_bytes": sum(x.nbytes for x in self._cache.values()),
                    "maximum_feature_bytes": self._maximum*int(np.prod(OUTPUT_SHAPE))*4,
                    "view_keys": list(self._cache), "binding_sha256": self._binding_key,
                    "query_gt_read": False}

    def clear_cache(self):
        with self._lock:
            self._cache.clear()


class _TimmCPUForward:
    def __init__(self, model_dir, producer, threads):
        if not isinstance(threads, int) or not 1 <= threads <= 32:
            raise ValueError("Explicit1..32 CPU threads required; root controls total worker budget")
        self._directory = Path(model_dir).resolve()
        self._config = self._directory/"config.json"
        self._checkpoint = self._directory/"model.safetensors"
        cfg = json.loads(self._config.read_text())
        if cfg.get("architecture") != ARCHITECTURE:
            raise ValueError("Exact existing DINOv3-L/16 architecture required")
        config_hash, checkpoint_hash = _sha(self._config), _sha(self._checkpoint)
        expected = _assets(producer)
        if expected["config_sha256"] != config_hash or expected["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Existing model files differ from Episode.producer; no substitute or download")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # Imports are deliberately lazy: NumPy-only cache checks need no Torch.
        import torch
        import timm
        from safetensors.torch import load_file
        torch.set_num_threads(threads)
        # CPU FP32 never uses TF32; keep flags disabled to make the contract explicit.
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        eva = importlib.import_module("timm.models.eva")
        eva_path = Path(inspect.getsourcefile(eva))
        eva_hash = _sha(eva_path)
        if expected["timm_eva_sha256"] is not None and expected["timm_eva_sha256"] != eva_hash:
            raise ValueError("timm Eva source identity differs from Episode.producer")
        with torch.device("cpu"):
            model = timm.create_model(ARCHITECTURE, pretrained=False, num_classes=0)
        model.load_state_dict(load_file(str(self._checkpoint), device="cpu"), strict=True)
        model = model.to(device="cpu", dtype=torch.float32).eval().requires_grad_(False)
        if getattr(model, "num_prefix_tokens", None) != 5:
            raise ValueError("Pinned five-prefix Eva DINO interface required")
        norm = getattr(model, "norm", None)
        if (not isinstance(norm, torch.nn.LayerNorm) or tuple(norm.normalized_shape) != (1024,)
                or abs(float(norm.eps)-1e-6) > 1e-12):
            raise ValueError("Pinned native final LayerNorm(1024,eps1e-6) required")
        norm_path = Path(inspect.getsourcefile(type(norm)))
        norm_hash = _sha(norm_path)
        if expected["layernorm_source_sha256"] is not None and expected["layernorm_source_sha256"] != norm_hash:
            raise ValueError("LayerNorm source differs from Episode.producer")
        if config_hash != _sha(self._config) or checkpoint_hash != _sha(self._checkpoint):
            raise ValueError("Frozen assets changed during strict model loading")
        self._torch, self._model = torch, model
        self._stamps = {p: _stamp(p) for p in (self._config, self._checkpoint, eva_path, norm_path, Path(__file__))}
        self.binding = {"kind": "frozen_DINOv3_FP32_native_final_LN_patches",
                        "FoRIS_Part1_applied": False, "architecture": ARCHITECTURE,
                        "model_input_side": 1024, "input_shape": list(INPUT_SHAPE),
                        "output_shape": list(OUTPUT_SHAPE), "native_output_dtype": "float32",
                        "prefix_tokens": 5, "execution_kind": "real_frozen_cpu_model", "device": "cpu",
                        "frozen": True, "eval": True, "autocast": False, "TF32": False,
                        "cpu_threads": threads, "torch_version": torch.__version__, "timm_version": timm.__version__,
                        "input_normalization": {"mean": [.485,.456,.406], "std": [.229,.224,.225]},
                        "model_assets": {"checkpoint_sha256": checkpoint_hash, "config_sha256": config_hash,
                                         "timm_eva_sha256": eva_hash, "layernorm_source_sha256": norm_hash},
                        "layernorm": {"class": type(norm).__module__+"."+type(norm).__qualname__,
                                      "normalized_shape": [1024], "eps": float(norm.eps),
                                      "source": str(norm_path), "source_sha256": norm_hash},
                        "encoder_source_sha256": _sha(__file__)}
        assert_matches_producer(producer, self.binding)
        self.check_frozen()

    def check_frozen(self):
        if any(_stamp(p) != stamp for p, stamp in self._stamps.items()):
            raise ValueError("Bound model or implementation files changed after loading")
        torch = self._torch
        if any(m.training for m in self._model.modules()):
            raise ValueError("Frozen encoder unexpectedly entered training mode")
        for tensor in itertools_chain(self._model.parameters(), self._model.buffers()):
            if tensor.device.type != "cpu" or (tensor.is_floating_point() and tensor.dtype != torch.float32):
                raise ValueError("CPU FP32 model tensors required; no GPU or mixed precision")
            if tensor.requires_grad:
                raise ValueError("Frozen model tensor unexpectedly requires gradients")

    def __call__(self, image):
        self.check_frozen()
        torch = self._torch
        x = torch.from_numpy(image.copy()).permute(2,0,1).float()/255
        mean = x.new_tensor((.485,.456,.406))[:,None,None]
        std = x.new_tensor((.229,.224,.225))[:,None,None]
        with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
            tokens = self._model.forward_features(((x-mean)/std)[None])
        if tokens.shape != (1,4101,1024) or tokens.dtype != torch.float32 or tokens.device.type != "cpu":
            raise ValueError("Native final-LN five-prefix FP32 output interface differs")
        value = tokens[0,5:].reshape(OUTPUT_SHAPE).numpy().copy()
        self.check_frozen()
        return value


def itertools_chain(*groups):
    for group in groups:
        yield from group


def get_cpu_encoder(model_dir, producer, *, threads=1, max_cached_views=16):
    """Once-per-worker model factory; root may inject callback and its binding.

    Root's runner calls `module.configure_encoder(enc, enc.binding)`. A caller
    must preserve that identity on each Episode using `enc.validate_producer`.
    An unchanged loaded model is reused without repeatedly hashing its weights.
    """
    if (not isinstance(producer, dict) or producer.get("FoRIS_Part1_applied") is not False
            or producer.get("kind") not in _NATIVE_KINDS):
        raise ValueError("Native frozen producer required before model construction")
    if not isinstance(threads, int) or not 1 <= threads <= 32:
        raise ValueError("Explicit1..32 CPU threads required")
    if not isinstance(max_cached_views, int) or not 0 <= max_cached_views <= MAX_CACHED_VIEWS:
        raise ValueError("Bounded0..16 cached views required")
    expected = _assets(producer)
    key = (os.getpid(), str(Path(model_dir).resolve()), expected["checkpoint_sha256"],
           expected["config_sha256"])
    with _FACTORY_LOCK:
        if key not in _WORKER_ENCODERS:
            forward = _TimmCPUForward(model_dir, producer, threads)
            encoder = CachedCPUEncoder(forward, forward.binding, max_cached_views=max_cached_views)
            encoder._real_forward = forward
            _WORKER_ENCODERS[key] = encoder
        encoder = _WORKER_ENCODERS[key]
        if encoder.binding["cpu_threads"] != threads or encoder._maximum != max_cached_views:
            raise ValueError("Existing worker encoder configuration differs; do not load a second copy")
        encoder._real_forward.check_frozen()
        encoder.validate_producer(producer)
        return encoder
