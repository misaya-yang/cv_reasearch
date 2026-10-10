"""Inference-only CPU lattice backend for the original CRF on Apple Silicon.

Keep the downloaded CRF source unchanged. Its CPU binding predates the NHWC
wrapper and interprets float tensors as doubles; build a derived binding with
correct metadata and float32 pointers. The CRF solver/settings stay unchanged.
CUDA numerical parity is not assumed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import types

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(crf_source, runtime):
    from torch.utils.cpp_extension import load

    root = Path(crf_source).resolve() / 'src'
    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    original = root / 'PermutohedralFiltering/source/cpu/LatticeFilterKernel.cpp'
    source = original.read_text()
    replacements = {
        'input_tensor.size(1)': 'input_tensor.size(rank - 1)',
        'input_tensor.size(i + 2)': 'input_tensor.size(i + 1)',
        'image_tensor.size(1)': 'image_tensor.size(rank - 1)',
        'assert(reference_image_tensor.dims() == rank);':
            'TORCH_CHECK(image_tensor.dim() == rank, "Image/input rank mismatch");',
        'input_tensor.type()': 'input_tensor.options()',
        'double': 'float',
    }
    for before, after in replacements.items():
        if before not in source:
            raise ValueError('CPU source changed; missing expected construct: ' + before)
        source = source.replace(before, after)
    anchor = 'int rank = input_tensor.ndimension();'
    guards = '''
    TORCH_CHECK(input_tensor.device().is_cpu() && image_tensor.device().is_cpu(), "CPU tensors required");
    TORCH_CHECK(input_tensor.scalar_type() == at::kFloat && image_tensor.scalar_type() == at::kFloat, "float32 required");
    TORCH_CHECK(input_tensor.is_contiguous() && image_tensor.is_contiguous(), "Contiguous NHWC tensors required");
    TORCH_CHECK(rank == 4 && image_tensor.dim() == 4, "NHWC rank 4 required");
    TORCH_CHECK(input_tensor.size(0) == image_tensor.size(0) && input_tensor.size(1) == image_tensor.size(1) && input_tensor.size(2) == image_tensor.size(2), "Image/input spatial shape mismatch");
'''
    source = source.replace(anchor, anchor + guards)
    derived = runtime / 'LatticeFilterKernel_m4.cpp'
    if not derived.exists() or derived.read_text() != source:
        derived.write_text(source)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('MAX_JOBS', '2')
    backend = load(name='ics_permutohedral_m4', sources=[str(derived)],
                   extra_include_paths=[str(original.parent)],
                   extra_cflags=['-O3', '-DNDEBUG'], build_directory=str(runtime), verbose=False)

    class PermutohedralLayer(torch.nn.Module):
        def __init__(self, bilateral, theta_alpha, theta_beta, theta_gamma, nhwc=True):
            super().__init__()
            self.bilateral = bilateral
            self.parameters_ = (theta_alpha, theta_beta, theta_gamma)

        def forward(self, x, image):
            if torch.is_grad_enabled() and (x.requires_grad or image.requires_grad):
                raise RuntimeError('M4 lattice backend supports inference only')
            if x.ndim != 4 or image.ndim != 4 or x.shape[0] != image.shape[0] or x.shape[2:] != image.shape[2:]:
                raise ValueError('Expected aligned NCHW logits/image')
            if not torch.isfinite(x).all() or not torch.isfinite(image).all():
                raise ValueError('Nonfinite CRF inputs')
            values = x.detach().to('cpu', torch.float32).permute(0, 2, 3, 1).contiguous()
            colours = image.detach().to('cpu', torch.float32).permute(0, 2, 3, 1).contiguous()
            out = backend.forward(values, colours, self.bilateral, *self.parameters_)
            return out.permute(0, 3, 1, 2).contiguous().to(device=x.device, dtype=x.dtype)

    # The original CRF imports this one symbol; avoid importing its CUDA-only
    # module. The original CRF optimizer, normalization and refiners are used.
    module = types.ModuleType('PermutohedralFiltering')
    module.PermutohedralLayer = PermutohedralLayer
    sys.modules['PermutohedralFiltering'] = module
    sys.path.insert(0, str(root))
    receipt = dict(backend='original CPU lattice, float32 NHWC binding',
                   source=str(original), source_sha256=sha(original),
                   derived=str(derived), derived_sha256=sha(derived),
                   replacements=replacements, torch=torch.__version__,
                   cuda_parity='unverified', inference_only=True)
    (runtime / 'build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return PermutohedralLayer, receipt
