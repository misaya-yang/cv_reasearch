"""Shared complete-decoder arms for the next controlled comparison."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sam_shared_decoder/execution_baselines"))
sys.path.insert(0, str(ROOT / "research/quality_mechanisms"))

import torch
from baselines import PositionCache, official_predict, make_fixture, tensor_bytes
import experimental_methods as experimental
from benchmark import compare, measure, synchronize, reset_memory, memory_result

# Both dense contraction routes and both practical merged attention routes
# receive the same phase head as the candidates. The unchanged merged control
# and factor phase arm identify which improvement produced any gain.
ARMS = (
    ("dense_phase_associated", "dense_phase", "explicit", "associated", "sparse"),
    ("dense_phase_projected", "dense_phase", "explicit", "projected", "dense"),
    ("cached_merged_phase_explicit", "cached_merged_phase", "explicit", "auto", "auto"),
    ("cached_merged_phase_sdpa", "cached_merged_phase", "sdpa", "auto", "auto"),
    ("factor_merged_phase", "factor_merged_phase", "explicit", "auto", "auto"),
    ("factor_implicit_phase", "factor_implicit_phase", "explicit", "auto", "auto"),
    ("factor_phase", "factor_phase", "explicit", "auto", "auto"),
    ("cached_merged_control", "cached_merged", "explicit", "auto", "auto"),
)


def configure(threads=2):
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def arm_functions(arm, model, image, pe, dense, tokens):
    label, method, attention, order, write = arm
    plans = experimental.plans_for(method, model, image.shape[-1] * image.shape[-2],
                                   tokens, order, write)
    factory = experimental.CacheFactory(method, plans)

    def build():
        return factory.build(model, image, dense, PositionCache.build(model, pe, plans))

    def forward(sparse, cache):
        return experimental.forward(method, model, cache, sparse, attention, plans)

    return build, forward, plans


def atomic_json(path, value):
    import json
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)
