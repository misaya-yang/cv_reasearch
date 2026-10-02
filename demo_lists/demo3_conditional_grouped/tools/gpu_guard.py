"""Human-controlled stop marker, checked before GPU worker imports."""
import pathlib

def require_gpu_unpaused(script_file):
    root=pathlib.Path(script_file).resolve().parents[1]
    if (root/'GPU_PAUSED').exists():
        raise RuntimeError('GPU_PAUSED: user requested no GPU work. Explicit renewed user authorization is required before removing this marker.')
