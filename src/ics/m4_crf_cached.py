"""Optional exact CPU CRF geometry reuse; values are recomputed each call."""
from pathlib import Path
import os
import sys
import types
import torch


def build_backend(crf_source, runtime):
    from torch.utils.cpp_extension import load
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    os.environ.setdefault('MAX_JOBS', '2')
    root = Path(crf_source).resolve()/'src/PermutohedralFiltering/source/cpu'
    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    header = (root/'PermutohedralLatticeCPU.h').read_text()
    header = header.replace('PermutohedralLatticeCPU', 'PreparedPermutohedralLatticeCPU')
    header = header.replace('#include "../Devices.hpp"', '#include "'+str(root.parent/'Devices.hpp')+'"')
    header = header.replace('    int idx;', '    int idx;\n    std::unique_ptr<int[]> neighbours;')
    anchor = '    void filter(T * output, const T* input, const T* positions, bool reverse) {'
    assert anchor in header
    methods = r'''
    void prepare_neighbours() {
        const int count = hashTable.size();
        neighbours.reset(new int[(pd + 1) * count * 2]);
        std::unique_ptr<short[]> n1(new short[pd + 1]);
        std::unique_ptr<short[]> n2(new short[pd + 1]);
        for (int axis = 0; axis <= pd; ++axis) {
            for (int i = 0; i < count; ++i) {
                const short* key = hashTable.getKeys() + i * pd;
                for (int k = 0; k < pd; ++k) {
                    n1[k] = key[k] + 1;
                    n2[k] = key[k] - 1;
                }
                // Last axis is implicit; lookup hashes only the first pd keys.
                if (axis < pd) {
                    n1[axis] = key[axis] - pd;
                    n2[axis] = key[axis] + pd;
                }
                T* a = hashTable.lookup(n1.get(), false);
                neighbours[(axis * count + i) * 2] = a ? int(a - hashTable.values) : -1;
                T* b = hashTable.lookup(n2.get(), false);
                neighbours[(axis * count + i) * 2 + 1] = b ? int(b - hashTable.values) : -1;
            }
        }
    }

    void filter_reuse(T* output, const T* input, bool reverse) {
        const int count = hashTable.size();
        std::memset(hashTable.values, 0, sizeof(T) * count * vd);
        // Same point/remainder/channel accumulation order as splat_point().
        for (int n = 0; n < N; ++n) {
            for (int remainder = 0; remainder <= pd; ++remainder) {
                const MatrixEntry r = matrix[n * (pd + 1) + remainder];
                T* vertex = hashTable.values + r.offset;
                for (int k = 0; k < vd - 1; ++k)
                    vertex[k] += r.weight * input[n * (vd - 1) + k];
                vertex[vd - 1] += r.weight;
            }
        }
        auto new_values = new T[vd * hashTable.capacity];
        auto zero = new T[vd]{0};
        for (int axis = reverse ? pd : 0; axis >= 0 && axis <= pd; reverse ? --axis : ++axis) {
            for (int i = 0; i < count; ++i) {
                const int a = neighbours[(axis * count + i) * 2];
                const int b = neighbours[(axis * count + i) * 2 + 1];
                const T* n1_value = a < 0 ? zero : hashTable.values + a;
                const T* n2_value = b < 0 ? zero : hashTable.values + b;
                const T* oldVal = hashTable.values + i * vd;
                T* newVal = new_values + i * vd;
                for (int k = 0; k < vd; ++k)
                    newVal[k] = (0.25 * n1_value[k] + 0.5 * oldVal[k] + 0.25 * n2_value[k]);
            }
            std::swap(hashTable.values, new_values);
        }
        delete[] new_values;
        delete[] zero;
        slice(output);
    }

'''
    header = header.replace(anchor, methods+anchor)
    header_path = runtime/'PermutohedralLatticeCPU_cached.h'
    if not header_path.exists() or header_path.read_text() != header:
        header_path.write_text(header)
    source = runtime/'prepared.cpp'
    source_text = r'''
#include <torch/extension.h>
#include <mutex>
#include "PermutohedralLatticeCPU_cached.h"

class Geometry {
    at::Tensor positions;
    int batch, height, width, channels, pd, pixels;
    std::vector<std::unique_ptr<PreparedPermutohedralLatticeCPU<float>>> lattices;
    bool initialized = false;
    std::mutex mutex;
public:
    Geometry(const at::Tensor& image, int channels_, bool bilateral,
             float theta_alpha, float theta_beta, float theta_gamma) : channels(channels_) {
        TORCH_CHECK(image.device().is_cpu() && image.scalar_type() == at::kFloat && image.dim() == 4 && image.is_contiguous(), "Contiguous CPU FP32 NHWC image required");
        batch = image.size(0); height = image.size(1); width = image.size(2);
        TORCH_CHECK(batch > 0 && height > 0 && width > 0 && channels > 0, "Nonempty dimensions required");
        const int colours = bilateral ? image.size(3) : 0;
        pd = 2 + colours; pixels = height * width;
        positions = at::zeros({batch * pixels * pd}, image.options());
        int shape[] = {height, width};
        for (int b = 0; b < batch; ++b) {
            ComputeKernel<CPUDevice, float>()(
                image.data_ptr<float>() + b * pixels * colours,
                positions.data_ptr<float>() + b * pixels * pd,
                pixels, 2, shape, colours, bilateral ? theta_alpha : theta_gamma,
                bilateral ? theta_beta : -1);
            lattices.emplace_back(new PreparedPermutohedralLatticeCPU<float>(pd, channels + 1, pixels));
        }
    }

    at::Tensor forward(const at::Tensor& values) {
        std::lock_guard<std::mutex> guard(mutex);
        TORCH_CHECK(values.device().is_cpu() && values.scalar_type() == at::kFloat && values.is_contiguous(), "Contiguous CPU FP32 values required");
        TORCH_CHECK(values.dim() == 4 && values.size(0) == batch && values.size(1) == height && values.size(2) == width && values.size(3) == channels, "Prepared geometry shape changed");
        auto output = at::zeros_like(values);
        for (int b = 0; b < batch; ++b) {
            auto in = values.data_ptr<float>() + b * pixels * channels;
            auto out = output.data_ptr<float>() + b * pixels * channels;
            if (!initialized) {
                lattices[b]->filter(out, in, positions.data_ptr<float>() + b * pixels * pd, false);
                lattices[b]->prepare_neighbours();
            } else {
                lattices[b]->filter_reuse(out, in, false);
            }
        }
        initialized = true;
        return output;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<Geometry>(m, "Geometry")
        .def(pybind11::init<const at::Tensor&, int, bool, float, float, float>())
        .def("forward", &Geometry::forward);
}
'''
    if not source.exists() or source.read_text() != source_text:
        source.write_text(source_text)
    return load(name='ics_permutohedral_m4_prepared_v1', sources=[str(source)],
        extra_include_paths=[str(root)], extra_cflags=['-O3', '-DNDEBUG'],
        build_directory=str(runtime), verbose=False)


def enable_on_crf(crf, backend):
    """Bound cache: one spatial geometry and one bilateral image per CRF host."""
    for layer in (crf.spatial_filter, crf.bilateral_filter):
        layer._prepared_geometry = None
        layer._prepared_key = None
        layer._prepared_colours = None
        layer._original_forward = layer.forward

        def forward(self, x, image):
            if torch.is_grad_enabled() and (x.requires_grad or image.requires_grad):
                raise RuntimeError('Prepared CPU lattice supports inference only')
            if x.ndim != 4 or image.ndim != 4 or x.shape[0] != image.shape[0] or x.shape[2:] != image.shape[2:]:
                raise ValueError('Expected aligned NCHW logits/image')
            if not torch.isfinite(x).all() or not torch.isfinite(image).all():
                raise ValueError('Nonfinite CRF inputs')
            values = x.detach().to('cpu', torch.float32).permute(0, 2, 3, 1).contiguous()
            colours = image.detach().to('cpu', torch.float32).permute(0, 2, 3, 1).contiguous()
            key = (tuple(values.shape), self.bilateral, self.parameters_)
            changed = self._prepared_key != key
            if self.bilateral and not changed:
                changed = not torch.equal(colours, self._prepared_colours)
            if changed:
                self._prepared_geometry = backend.Geometry(colours, values.shape[-1], self.bilateral, *self.parameters_)
                self._prepared_key = key
                self._prepared_colours = colours.clone() if self.bilateral else None
            result = self._prepared_geometry.forward(values)
            return result.permute(0, 3, 1, 2).contiguous().to(device=x.device, dtype=x.dtype)
        layer.forward = types.MethodType(forward, layer)
    return crf
